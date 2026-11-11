"""Web image scraper for California papaya classes with resume, retry, and robots.txt."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
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

API_KEY_SOURCES: dict[str, tuple[str, str]] = {
    "unsplash": ("unsplash_access_key", "UNSPLASH_ACCESS_KEY"),
    "pexels": ("pexels_api_key", "PEXELS_API_KEY"),
    "pixabay": ("pixabay_api_key", "PIXABAY_API_KEY"),
    "flickr": ("flickr_api_key", "FLICKR_API_KEY"),
    "serper": ("serper_api_key", "SERPER_API_KEY"),
}


@dataclass
class ScraperConfig:
    """Configuration options for image scraping."""

    output_dir: Path = Path("data/raw")
    per_class: int = 200  # <= 0 means unlimited across all queries and sources
    per_query_limit: int = 100
    min_size: tuple[int, int] = (320, 320)
    delay: float = 1.0
    threads: int = 4
    sources: tuple[str, ...] = ("bing", "google")
    max_retries: int = 3
    respect_robots: bool = True
    unsplash_access_key: str = ""
    pexels_api_key: str = ""
    pixabay_api_key: str = ""
    flickr_api_key: str = ""
    serper_api_key: str = ""
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


def _fetch_json(
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    json_body: dict[str, Any] | None = None,
    max_retries: int = 3,
) -> dict[str, Any]:
    import requests  # type: ignore

    merged_headers = {"User-Agent": DEFAULT_USER_AGENT, **(headers or {})}

    def _req() -> dict[str, Any]:
        resp = requests.request(
            method,
            url,
            params=params,
            headers=merged_headers,
            json=json_body,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    return retry_with_backoff(
        _req,
        max_retries=max_retries,
        exceptions=(requests.RequestException, ValueError),
        logger=logger,
    )


def _fetch_unsplash_urls(
    query: str, limit: int, access_key: str, max_retries: int = 3
) -> list[str]:
    if not access_key:
        return []
    urls: list[str] = []
    per_page = min(30, limit)
    pages = max(1, (limit + per_page - 1) // per_page)
    for page in range(1, pages + 1):
        payload = _fetch_json(
            "GET",
            "https://api.unsplash.com/search/photos",
            params={"query": query, "per_page": per_page, "page": page},
            headers={"Authorization": f"Client-ID {access_key}"},
            max_retries=max_retries,
        )
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
    urls: list[str] = []
    per_page = min(80, limit)
    pages = max(1, (limit + per_page - 1) // per_page)
    for page in range(1, pages + 1):
        payload = _fetch_json(
            "GET",
            "https://api.pexels.com/v1/search",
            params={"query": query, "per_page": per_page, "page": page},
            headers={"Authorization": api_key},
            max_retries=max_retries,
        )
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


def _fetch_pixabay_urls(
    query: str, limit: int, api_key: str, max_retries: int = 3
) -> list[str]:
    if not api_key:
        return []
    urls: list[str] = []
    per_page = max(3, min(200, limit))
    pages = max(1, (limit + per_page - 1) // per_page)
    for page in range(1, pages + 1):
        payload = _fetch_json(
            "GET",
            "https://pixabay.com/api/",
            params={
                "key": api_key,
                "q": query,
                "image_type": "photo",
                "per_page": per_page,
                "page": page,
            },
            max_retries=max_retries,
        )
        hits = payload.get("hits", [])
        if not hits:
            break
        for item in hits:
            url = item.get("largeImageURL") or item.get("webformatURL")
            if url:
                urls.append(url)
            if len(urls) >= limit:
                return urls
    return urls


def _fetch_flickr_urls(
    query: str, limit: int, api_key: str, max_retries: int = 3
) -> list[str]:
    if not api_key:
        return []
    urls: list[str] = []
    per_page = min(100, limit)
    pages = max(1, (limit + per_page - 1) // per_page)
    for page in range(1, pages + 1):
        payload = _fetch_json(
            "GET",
            "https://www.flickr.com/services/rest/",
            params={
                "method": "flickr.photos.search",
                "api_key": api_key,
                "text": query,
                "format": "json",
                "nojsoncallback": 1,
                "content_type": 1,
                "media": "photos",
                "extras": "url_l,url_c,url_z",
                "per_page": per_page,
                "page": page,
            },
            max_retries=max_retries,
        )
        photos = payload.get("photos", {}).get("photo", [])
        if not photos:
            break
        for item in photos:
            url = item.get("url_l") or item.get("url_c") or item.get("url_z")
            if not url and item.get("server") and item.get("id") and item.get("secret"):
                url = f"https://live.staticflickr.com/{item['server']}/{item['id']}_{item['secret']}_b.jpg"
            if url:
                urls.append(url)
            if len(urls) >= limit:
                return urls
    return urls


def _fetch_serper_urls(
    query: str, limit: int, api_key: str, max_retries: int = 3
) -> list[str]:
    if not api_key:
        return []
    urls: list[str] = []
    num = min(100, limit)
    pages = max(1, (limit + num - 1) // num)
    for page in range(1, pages + 1):
        payload = _fetch_json(
            "POST",
            "https://google.serper.dev/images",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json_body={"q": query, "num": num, "page": page},
            max_retries=max_retries,
        )
        images = payload.get("images", [])
        if not images:
            break
        for item in images:
            url = item.get("imageUrl")
            if url:
                urls.append(url)
            if len(urls) >= limit:
                return urls
    return urls


def _fetch_wikimedia_urls(
    query: str, limit: int, max_retries: int = 3
) -> list[str]:
    payload = _fetch_json(
        "GET",
        "https://commons.wikimedia.org/w/api.php",
        params={
            "action": "query",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {query}",
            "gsrnamespace": 6,
            "gsrlimit": min(50, limit),
            "prop": "imageinfo",
            "iiprop": "url|mime",
            "iiurlwidth": 1024,
            "format": "json",
        },
        max_retries=max_retries,
    )
    pages = payload.get("query", {}).get("pages", {})
    urls: list[str] = []
    for page in pages.values():
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        mime = str(info.get("mime", ""))
        if not mime.startswith("image/") or "svg" in mime:
            continue
        url = info.get("thumburl") or info.get("url")
        if url:
            urls.append(url)
        if len(urls) >= limit:
            break
    return urls


def _fetch_inaturalist_urls(
    cls_name: str, limit: int, max_retries: int = 3
) -> list[str]:
    ripeness_hint = cls_name.replace("_", " ")
    payload = _fetch_json(
        "GET",
        "https://api.inaturalist.org/v1/observations",
        params={
            "taxon_name": "Carica papaya",
            "q": ripeness_hint,
            "photos": "true",
            "quality_grade": "research",
            "per_page": min(100, limit),
            "page": 1,
        },
        max_retries=max_retries,
    )
    urls: list[str] = []
    for obs in payload.get("results", []):
        for photo in obs.get("photos", []):
            url = photo.get("url")
            if url:
                urls.append(url.replace("square.", "large."))
            if len(urls) >= limit:
                return urls
    return urls


def _fetch_duckduckgo_urls(
    query: str, limit: int, max_retries: int = 3
) -> list[str]:
    try:
        try:
            from ddgs import DDGS  # type: ignore
        except ImportError:
            from duckduckgo_search import DDGS  # type: ignore

        results = DDGS().images(query, max_results=limit)
        return [item["image"] for item in results if item.get("image")][:limit]
    except ImportError:
        pass

    import requests  # type: ignore

    headers = {"User-Agent": DEFAULT_USER_AGENT}

    def _get_vqd() -> str:
        resp = requests.get(
            "https://duckduckgo.com/",
            params={"q": query},
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        match = re.search(r"vqd=([\d-]+)", resp.text) or re.search(
            r'vqd="([^"]+)"', resp.text
        )
        if not match:
            raise ValueError("DuckDuckGo vqd token not found")
        return match.group(1)

    vqd = retry_with_backoff(_get_vqd, max_retries=max_retries, logger=logger)
    payload = _fetch_json(
        "GET",
        "https://duckduckgo.com/i.js",
        params={"l": "id-id", "o": "json", "q": query, "vqd": vqd, "f": ",,,"},
        headers={"Referer": "https://duckduckgo.com/"},
        max_retries=max_retries,
    )
    return [
        item["image"]
        for item in payload.get("results", [])
        if item.get("image")
    ][:limit]


def _scrape_api_urls(
    urls: list[str],
    class_dir: Path,
    cls_name: str,
    query: str,
    needed: int | None,
    cfg: ScraperConfig,
    manifest: SourceManifest,
    robots: RobotsChecker | None,
) -> int:
    """Download API-provided image URLs concurrently, skipping existing/seen URLs."""
    candidates = [u for u in urls if not manifest.has_url(u)]
    if not candidates or (needed is not None and needed <= 0):
        return 0

    saved = 0
    lock = threading.Lock()

    def _worker(url: str) -> bool:
        nonlocal saved
        if needed is not None:
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
            if needed is not None and saved >= needed:
                dest.unlink(missing_ok=True)
                return False
            saved += 1
        manifest.add(f"{cls_name}/{dest.name}", url, query, cls_name)
        return True

    batch = candidates if needed is None else candidates[: needed * 2]
    with ThreadPoolExecutor(max_workers=max(1, cfg.threads)) as pool:
        futures = [pool.submit(_worker, u) for u in batch]
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


def _resolve_key(s_cfg: dict[str, Any], field_name: str, env_name: str) -> str:
    return str(s_cfg.get(field_name) or os.environ.get(env_name, ""))


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
    key_kwargs = {
        field_name: _resolve_key(s_cfg, field_name, env_name)
        for field_name, env_name in API_KEY_SOURCES.values()
    }
    return ScraperConfig(
        output_dir=output_dir or Path(s_cfg.get("output_dir", "data/raw")),
        per_class=per_class if per_class is not None else int(s_cfg.get("per_class", 200)),
        per_query_limit=int(s_cfg.get("per_query_limit", 100)),
        min_size=(int(min_sz[0]), int(min_sz[1])),
        delay=delay if delay is not None else float(s_cfg.get("delay", 1.0)),
        threads=threads if threads is not None else int(s_cfg.get("threads", 4)),
        sources=tuple(s_cfg.get("sources", ["bing", "google"])),
        max_retries=int(s_cfg.get("max_retries", 3)),
        respect_robots=bool(s_cfg.get("respect_robots", True)),
        queries=dict(raw_cfg.get("queries", {})),
        **key_kwargs,
    )


def _fetch_urls_for_source(
    source: str,
    query: str,
    cls_name: str,
    limit: int,
    cfg: ScraperConfig,
    warned_missing_keys: set[str],
) -> list[str]:
    if source in API_KEY_SOURCES:
        field_name, env_name = API_KEY_SOURCES[source]
        api_key = getattr(cfg, field_name, "")
        if not api_key:
            if source not in warned_missing_keys:
                warned_missing_keys.add(source)
                logger.info(
                    "Skipping source '%s': set %s in config.yaml or %s in .env to enable.",
                    source,
                    field_name,
                    env_name,
                )
            return []
        keyed_fetchers = {
            "unsplash": _fetch_unsplash_urls,
            "pexels": _fetch_pexels_urls,
            "pixabay": _fetch_pixabay_urls,
            "flickr": _fetch_flickr_urls,
            "serper": _fetch_serper_urls,
        }
        return keyed_fetchers[source](query, limit, api_key, cfg.max_retries)

    if source == "wikimedia":
        return _fetch_wikimedia_urls(query, limit, cfg.max_retries)
    if source == "inaturalist":
        return _fetch_inaturalist_urls(cls_name, limit, cfg.max_retries)
    if source == "duckduckgo":
        return _fetch_duckduckgo_urls(query, limit, cfg.max_retries)
    raise ValueError(f"Unknown source '{source}'")


def scrape_dataset(cfg: ScraperConfig) -> dict[str, int]:
    """Scrape images for all classes, resuming from existing files in `cfg.output_dir`."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = SourceManifest(cfg.output_dir / "sources.jsonl")
    robots = RobotsChecker() if cfg.respect_robots else None
    warned_missing_keys: set[str] = set()
    unlimited = cfg.per_class <= 0
    summary: dict[str, int] = {}

    for cls_name in CLASSES:
        class_dir = cfg.output_dir / cls_name
        class_dir.mkdir(parents=True, exist_ok=True)
        existing = count_existing_images(class_dir)
        if not unlimited and existing >= cfg.per_class:
            logger.info(
                "[%s] Resume: already has %d/%d images, skipping.",
                cls_name,
                existing,
                cfg.per_class,
            )
            summary[cls_name] = existing
            continue

        queries = cfg.queries.get(cls_name) or [f"california papaya {cls_name.replace('_', ' ')}"]
        target_label = "unlimited" if unlimited else str(cfg.per_class)
        logger.info(
            "[%s] Starting scrape: %d existing, target %s (%d queries).",
            cls_name,
            existing,
            target_label,
            len(queries),
        )

        for query_idx, query in enumerate(queries):
            if not unlimited and count_existing_images(class_dir) >= cfg.per_class:
                break

            for source in cfg.sources:
                if not unlimited:
                    remaining = cfg.per_class - count_existing_images(class_dir)
                    if remaining <= 0:
                        break
                    crawl_limit = remaining
                    fetch_limit = remaining * 2
                    dl_limit: int | None = remaining
                else:
                    crawl_limit = cfg.per_query_limit
                    fetch_limit = cfg.per_query_limit
                    dl_limit = None

                if source == "inaturalist" and query_idx > 0:
                    continue
                try:
                    if source in ("bing", "google"):
                        _crawl_with_icrawler(
                            source, query, class_dir, cls_name, crawl_limit, cfg, manifest, robots
                        )
                    elif source in API_KEY_SOURCES or source in (
                        "wikimedia",
                        "inaturalist",
                        "duckduckgo",
                    ):
                        urls = _fetch_urls_for_source(
                            source, query, cls_name, fetch_limit, cfg, warned_missing_keys
                        )
                        _scrape_api_urls(
                            urls, class_dir, cls_name, query, dl_limit, cfg, manifest, robots
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

