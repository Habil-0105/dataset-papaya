"""Web image scraper for California papaya classes with resume, retry, and robots.txt."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

from papaya_cli.utils import (
    CLASSES,
    DEFAULT_USER_AGENT,
    IMAGE_EXTENSIONS,
    RobotsChecker,
    load_dotenv,
    retry_with_backoff,
)

logger = logging.getLogger("papaya_cli.scraper")


@dataclass
class ScraperConfig:
    """Configuration options for image scraping."""

    output_dir: Path = Path("data/raw")
    per_class: int = 200
    min_size: tuple[int, int] = (320, 320)
    delay: float = 1.0
    threads: int = 4
    sources: tuple[str, ...] = ("bing", "google")
    max_retries: int = 3
    respect_robots: bool = True
    unsplash_access_key: str = ""
    pexels_api_key: str = ""
    queries: dict[str, list[str]] = field(default_factory=dict)


class SourceManifest:
    """Thread-safe JSONL manifest tracking downloaded image URLs and queries."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.records: dict[str, dict[str, str]] = {}
        self.seen_urls: set[str] = set()
        if path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rel = rec.get("relpath", "")
                url = rec.get("source_url", "")
                if rel:
                    self.records[rel] = rec
                if url:
                    self.seen_urls.add(url)

    def add(self, relpath: str, source_url: str, query: str, cls_name: str) -> None:
        record = {
            "relpath": relpath,
            "source_url": source_url,
            "query": query,
            "class": cls_name,
        }
        with self._lock:
            self.records[relpath] = record
            if source_url:
                self.seen_urls.add(source_url)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def has_url(self, url: str) -> bool:
        with self._lock:
            return url in self.seen_urls


def count_existing_images(class_dir: Path) -> int:
    """Count existing image files in `class_dir`."""
    if not class_dir.is_dir():
        return 0
    return sum(
        1
        for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_and_save_image(
    data: bytes, dest_path: Path, min_size: tuple[int, int]
) -> bool:
    """Verify image bytes meet `min_size` and save to `dest_path`."""
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for image validation. Install with: pip install pillow"
        ) from exc

    try:
        with Image.open(BytesIO(data)) as img:
            img.verify()
        with Image.open(BytesIO(data)) as img:
            width, height = img.size
            if width < min_size[0] or height < min_size[1]:
                return False
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            rgb = img.convert("RGB")
            rgb.save(dest_path, format="JPEG", quality=92)
        return True
    except Exception:
        return False


def download_image_to_file(
    url: str,
    dest_path: Path,
    min_size: tuple[int, int],
    robots: RobotsChecker | None = None,
    max_retries: int = 3,
    delay: float = 0.0,
    timeout: float = 15.0,
) -> bool:
    """Download an image URL with robots.txt check, retry/backoff, and size validation."""
    if dest_path.exists():
        return False
    if robots is not None and not robots.can_fetch(url):
        logger.debug("Blocked by robots.txt: %s", url)
        return False
    if delay > 0:
        time.sleep(delay)

    try:
        import requests  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "requests is required for downloading images. Install with: pip install requests"
        ) from exc

    def _fetch() -> bytes:
        resp = requests.get(
            url,
            headers={"User-Agent": DEFAULT_USER_AGENT},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.content

    try:
        content = retry_with_backoff(
            _fetch,
            max_retries=max_retries,
            base_delay=max(0.5, delay),
            exceptions=(requests.RequestException,),
            logger=logger,
        )
    except Exception as exc:
        logger.debug("Failed downloading %s: %s", url, exc)
        return False

    return validate_and_save_image(content, dest_path, min_size)


def _fetch_unsplash_urls(
    query: str, limit: int, access_key: str, max_retries: int = 3
) -> list[str]:
    if not access_key:
        return []
    import requests  # type: ignore

    urls: list[str] = []
    per_page = min(30, limit)
    pages = max(1, (limit + per_page - 1) // per_page)
    for page in range(1, pages + 1):
        def _req() -> dict[str, Any]:
            resp = requests.get(
                "https://api.unsplash.com/search/photos",
                params={"query": query, "per_page": per_page, "page": page},
                headers={
                    "Authorization": f"Client-ID {access_key}",
                    "User-Agent": DEFAULT_USER_AGENT,
                },
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()

        try:
            payload = retry_with_backoff(_req, max_retries=max_retries, logger=logger)
        except Exception as exc:
            logger.warning("Unsplash query '%s' failed: %s", query, exc)
            break
        results = payload.get("results", [])
        if not results:
            break
        for item in results:
            url = item.get("urls", {}).get("regular") or item.get("urls", {}).get("full")
            if url:
                urls.append(url)
            if len(urls) >= limit:
                return urls
    return urls


def _fetch_pexels_urls(
    query: str, limit: int, api_key: str, max_retries: int = 3
) -> list[str]:
    if not api_key:
        return []
    import requests  # type: ignore

    urls: list[str] = []
    per_page = min(80, limit)
    pages = max(1, (limit + per_page - 1) // per_page)
    for page in range(1, pages + 1):
        def _req() -> dict[str, Any]:
            resp = requests.get(
                "https://api.pexels.com/v1/search",
                params={"query": query, "per_page": per_page, "page": page},
                headers={
                    "Authorization": api_key,
                    "User-Agent": DEFAULT_USER_AGENT,
                },
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()

        try:
            payload = retry_with_backoff(_req, max_retries=max_retries, logger=logger)
        except Exception as exc:
            logger.warning("Pexels query '%s' failed: %s", query, exc)
            break
        photos = payload.get("photos", [])
        if not photos:
            break
        for item in photos:
            url = item.get("src", {}).get("large") or item.get("src", {}).get("original")
            if url:
                urls.append(url)
            if len(urls) >= limit:
                return urls
    return urls


def _scrape_api_urls(
    urls: list[str],
    class_dir: Path,
    cls_name: str,
    query: str,
    needed: int,
    cfg: ScraperConfig,
    manifest: SourceManifest,
    robots: RobotsChecker | None,
) -> int:
    """Download API-provided image URLs concurrently, skipping existing/seen URLs."""
    candidates = [u for u in urls if not manifest.has_url(u)]
    if not candidates or needed <= 0:
        return 0

    saved = 0
    lock = threading.Lock()

    def _worker(url: str) -> bool:
        nonlocal saved
        with lock:
            if saved >= needed:
                return False
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
        dest = class_dir / f"{digest}.jpg"
        if dest.exists():
            return False
        ok = download_image_to_file(
            url,
            dest,
            min_size=cfg.min_size,
            robots=robots,
            max_retries=cfg.max_retries,
            delay=cfg.delay,
        )
        if not ok:
            return False
        with lock:
            if saved >= needed:
                dest.unlink(missing_ok=True)
                return False
            saved += 1
        manifest.add(f"{cls_name}/{dest.name}", url, query, cls_name)
        return True

    with ThreadPoolExecutor(max_workers=max(1, cfg.threads)) as pool:
        futures = [pool.submit(_worker, u) for u in candidates[: needed * 2]]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:
                logger.debug("Worker error: %s", exc)
    return saved


def _crawl_with_icrawler(
    source: str,
    query: str,
    class_dir: Path,
    cls_name: str,
    needed: int,
    cfg: ScraperConfig,
    manifest: SourceManifest,
    robots: RobotsChecker | None,
) -> int:
    """Crawl Bing or Google images via icrawler with robots/retry/manifest hooks."""
    if needed <= 0:
        return 0
    try:
        from icrawler import ImageDownloader  # type: ignore
        from icrawler.builtin import BingImageCrawler, GoogleImageCrawler  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "icrawler is required for Bing/Google scraping. Install with: pip install icrawler"
        ) from exc

    crawler_cls = BingImageCrawler if source == "bing" else GoogleImageCrawler
    tl = threading.local()
    saved_counter = {"count": 0}
    count_lock = threading.Lock()

    class GuardedDownloader(ImageDownloader):  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            orig_write = self.storage.write

            def _tracked_write(filename: str, data: bytes) -> None:
                orig_write(filename, data)
                url = getattr(tl, "current_url", "")
                manifest.add(f"{cls_name}/{filename}", url, query, cls_name)
                with count_lock:
                    saved_counter["count"] += 1

            self.storage.write = _tracked_write

        def download(
            self,
            task: dict[str, Any],
            default_ext: str,
            timeout: int = 5,
            max_retry: int = 3,
            overwrite: bool = False,
            **kwargs: Any,
        ) -> None:
            url = task.get("file_url", "")
            if not url or manifest.has_url(url):
                return
            if robots is not None and not robots.can_fetch(url):
                logger.debug("Blocked by robots.txt: %s", url)
                return
            if cfg.delay > 0:
                time.sleep(cfg.delay)
            tl.current_url = url
            retry_with_backoff(
                lambda: super(GuardedDownloader, self).download(
                    task,
                    default_ext,
                    timeout=timeout,
                    max_retry=1,
                    overwrite=False,
                    **kwargs,
                ),
                max_retries=cfg.max_retries,
                base_delay=max(0.5, cfg.delay),
                logger=logger,
            )

    crawler = crawler_cls(
        downloader_cls=GuardedDownloader,
        downloader_threads=max(1, cfg.threads),
        storage={"root_dir": str(class_dir)},
    )
    crawler.crawl(
        keyword=query,
        max_num=needed,
        min_size=cfg.min_size,
        file_idx_offset="auto",
    )
    return saved_counter["count"]


def build_scraper_config(
    raw_cfg: dict[str, Any],
    output_dir: Path | None = None,
    per_class: int | None = None,
    threads: int | None = None,
    delay: float | None = None,
) -> ScraperConfig:
    """Build `ScraperConfig` from parsed YAML dict and optional CLI overrides."""
    load_dotenv()
    s_cfg = raw_cfg.get("scraper", {})
    min_sz = s_cfg.get("min_size", [320, 320])
    return ScraperConfig(
        output_dir=output_dir or Path(s_cfg.get("output_dir", "data/raw")),
        per_class=per_class if per_class is not None else int(s_cfg.get("per_class", 200)),
        min_size=(int(min_sz[0]), int(min_sz[1])),
        delay=delay if delay is not None else float(s_cfg.get("delay", 1.0)),
        threads=threads if threads is not None else int(s_cfg.get("threads", 4)),
        sources=tuple(s_cfg.get("sources", ["bing", "google"])),
        max_retries=int(s_cfg.get("max_retries", 3)),
        respect_robots=bool(s_cfg.get("respect_robots", True)),
        unsplash_access_key=(
            s_cfg.get("unsplash_access_key") or os.environ.get("UNSPLASH_ACCESS_KEY", "")
        ),
        pexels_api_key=(
            s_cfg.get("pexels_api_key") or os.environ.get("PEXELS_API_KEY", "")
        ),
        queries=dict(raw_cfg.get("queries", {})),
    )


def scrape_dataset(cfg: ScraperConfig) -> dict[str, int]:
    """Scrape images for all classes, resuming from existing files in `cfg.output_dir`."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = SourceManifest(cfg.output_dir / "sources.jsonl")
    robots = RobotsChecker() if cfg.respect_robots else None
    summary: dict[str, int] = {}

    for cls_name in CLASSES:
        class_dir = cfg.output_dir / cls_name
        class_dir.mkdir(parents=True, exist_ok=True)
        existing = count_existing_images(class_dir)
        if existing >= cfg.per_class:
            logger.info(
                "[%s] Resume: already has %d/%d images, skipping.",
                cls_name,
                existing,
                cfg.per_class,
            )
            summary[cls_name] = existing
            continue

        queries = cfg.queries.get(cls_name) or [f"california papaya {cls_name.replace('_', ' ')}"]
        logger.info(
            "[%s] Starting scrape: %d existing, target %d (%d queries).",
            cls_name,
            existing,
            cfg.per_class,
            len(queries),
        )

        for query in queries:
            current = count_existing_images(class_dir)
            remaining = cfg.per_class - current
            if remaining <= 0:
                break
            per_query_target = max(1, remaining)

            for source in cfg.sources:
                current = count_existing_images(class_dir)
                remaining = cfg.per_class - current
                if remaining <= 0:
                    break
                take = min(per_query_target, remaining)
                try:
                    if source in ("bing", "google"):
                        _crawl_with_icrawler(
                            source, query, class_dir, cls_name, take, cfg, manifest, robots
                        )
                    elif source == "unsplash":
                        urls = _fetch_unsplash_urls(
                            query, take * 2, cfg.unsplash_access_key, cfg.max_retries
                        )
                        _scrape_api_urls(
                            urls, class_dir, cls_name, query, take, cfg, manifest, robots
                        )
                    elif source == "pexels":
                        urls = _fetch_pexels_urls(
                            query, take * 2, cfg.pexels_api_key, cfg.max_retries
                        )
                        _scrape_api_urls(
                            urls, class_dir, cls_name, query, take, cfg, manifest, robots
                        )
                    else:
                        logger.warning("Unknown source '%s', skipping.", source)
                except Exception as exc:
                    logger.warning(
                        "[%s] Source '%s' error for query '%s': %s",
                        cls_name,
                        source,
                        query,
                        exc,
                    )

        final_count = count_existing_images(class_dir)
        summary[cls_name] = final_count
        logger.info("[%s] Finished with %d images.", cls_name, final_count)

    return summary

