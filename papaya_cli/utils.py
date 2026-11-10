"""Shared configuration, logging, device detection, retry, and robots.txt utilities."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

T = TypeVar("T")

CLASSES: tuple[str, ...] = (
    "unripe",
    "early_ripening",
    "ripe",
    "overripe",
    "rotten",
)

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
)

DEFAULT_USER_AGENT = "papaya-cli/0.1 (+research-dataset-builder)"


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure root logger using Rich when installed, else standard StreamHandler."""
    level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger("papaya_cli")
    if logger.handlers:
        logger.setLevel(level)
        return logger

    try:
        from rich.logging import RichHandler  # type: ignore

        handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
        fmt = "%(message)s"
    except ImportError:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Load YAML configuration from disk. Returns empty dict if path is None."""
    if path is None:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load config files. Install with: pip install pyyaml"
        ) from exc

    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}")
    return data


def detect_device(preferred: str = "auto") -> str:
    """Return 'cuda' when available and preferred=='auto', else 'cpu' or explicit value."""
    if preferred != "auto":
        return preferred
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def retry_with_backoff(
    fn: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    logger: logging.Logger | None = None,
) -> T:
    """Call `fn()` retrying with exponential backoff on transient failures."""
    attempt = 0
    while True:
        try:
            return fn()
        except exceptions as exc:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_for = base_delay * (2 ** (attempt - 1))
            if logger is not None:
                logger.debug(
                    "Retry %d/%d after error (%s); sleeping %.2fs",
                    attempt,
                    max_retries,
                    exc,
                    sleep_for,
                )
            time.sleep(sleep_for)


class RobotsChecker:
    """Thread-safe per-origin robots.txt cache."""

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT, timeout: float = 5.0) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self._lock = threading.Lock()
        self._parsers: dict[str, RobotFileParser | None] = {}

    def can_fetch(self, url: str) -> bool:
        """Return True if `url` is allowed by the host's robots.txt (fail-open if absent)."""
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False
        origin = f"{parsed.scheme}://{parsed.netloc}"

        with self._lock:
            if origin not in self._parsers:
                self._parsers[origin] = self._load_parser(f"{origin}/robots.txt")
            parser = self._parsers[origin]

        if parser is None:
            return True
        return parser.can_fetch(self.user_agent, url)

    def _load_parser(self, robots_url: str) -> RobotFileParser | None:
        # ponytail: single global lock covers cache miss; per-origin locks if crawl concurrency scales.
        try:
            import urllib.request

            req = urllib.request.Request(
                robots_url, headers={"User-Agent": self.user_agent}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            parser = RobotFileParser()
            parser.parse(raw.splitlines())
            return parser
        except Exception:
            return None


def progress_iter(
    iterable: Iterable[T], desc: str = "", total: int | None = None
) -> Iterator[T]:
    """Wrap iterable with tqdm progress bar when available."""
    try:
        from tqdm import tqdm  # type: ignore

        yield from tqdm(iterable, desc=desc, total=total)
    except ImportError:
        yield from iterable


def iter_class_images(
    root: Path, classes: Iterable[str] = CLASSES
) -> Iterator[tuple[str, Path]]:
    """Yield `(class_name, image_path)` pairs in deterministic sorted order."""
    for cls_name in classes:
        cls_dir = root / cls_name
        if not cls_dir.is_dir():
            continue
        for path in sorted(cls_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield cls_name, path

