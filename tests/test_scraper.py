"""Unit tests for scraper and shared utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from papaya_cli.scraper import (
    ScraperConfig,
    SourceManifest,
    _fetch_flickr_urls,
    _fetch_inaturalist_urls,
    _fetch_pixabay_urls,
    _fetch_serper_urls,
    _fetch_urls_for_source,
    _fetch_wikimedia_urls,
    build_scraper_config,
    count_existing_images,
    scrape_dataset,
)
from papaya_cli.utils import CLASSES, RobotsChecker, load_dotenv, retry_with_backoff


class TestUtilsAndScraper(unittest.TestCase):
    def test_retry_with_backoff_succeeds_after_transient_errors(self) -> None:
        calls = {"n": 0}

        def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise ValueError("transient")
            return "ok"

        with patch("time.sleep"):
            result = retry_with_backoff(flaky, max_retries=3, base_delay=0.01)
        self.assertEqual(result, "ok")
        self.assertEqual(calls["n"], 3)

    def test_robots_checker_respects_disallow(self) -> None:
        checker = RobotsChecker()
        from urllib.robotparser import RobotFileParser

        rp = RobotFileParser()
        rp.parse(["User-agent: *", "Disallow: /private/"])
        checker._parsers["https://example.com"] = rp

        self.assertTrue(checker.can_fetch("https://example.com/public/a.jpg"))
        self.assertFalse(checker.can_fetch("https://example.com/private/b.jpg"))

    def test_source_manifest_persistence_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = SourceManifest(root / "sources.jsonl")
            manifest.add("ripe/001.jpg", "https://example.com/1.jpg", "ripe papaya", "ripe")

            reloaded = SourceManifest(root / "sources.jsonl")
            self.assertTrue(reloaded.has_url("https://example.com/1.jpg"))
            self.assertIn("ripe/001.jpg", reloaded.records)

    def test_scrape_dataset_skips_completed_classes_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cls_name in CLASSES:
                cls_dir = root / cls_name
                cls_dir.mkdir(parents=True)
                for i in range(2):
                    (cls_dir / f"img_{i}.jpg").write_bytes(b"fake")

            self.assertEqual(count_existing_images(root / "ripe"), 2)
            cfg = ScraperConfig(output_dir=root, per_class=2, sources=("bing",))
            with patch("papaya_cli.scraper._crawl_with_icrawler") as mock_crawl:
                summary = scrape_dataset(cfg)
                mock_crawl.assert_not_called()
            self.assertEqual(summary["ripe"], 2)

    def test_load_dotenv_sets_unsplash_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text('UNSPLASH_ACCESS_KEY="test-key-123"\n', encoding="utf-8")
            with patch.dict("os.environ", {}, clear=True):
                load_dotenv(env_file)
                cfg = build_scraper_config({"scraper": {"sources": ["unsplash"]}})
            self.assertEqual(cfg.unsplash_access_key, "test-key-123")
            self.assertIn("unsplash", cfg.sources)

    def test_new_source_fetchers_parse_urls_and_skip_missing_keys(self) -> None:
        cfg = ScraperConfig()
        warned: set[str] = set()
        self.assertEqual(
            _fetch_urls_for_source("pixabay", "pepaya", "ripe", 5, cfg, warned), []
        )
        self.assertIn("pixabay", warned)

        with patch(
            "papaya_cli.scraper._fetch_json",
            return_value={"hits": [{"largeImageURL": "https://pix.example/1.jpg"}]},
        ):
            self.assertEqual(
                _fetch_pixabay_urls("pepaya", 2, "key"), ["https://pix.example/1.jpg"]
            )

        with patch(
            "papaya_cli.scraper._fetch_json",
            return_value={
                "photos": {
                    "photo": [{"server": "12", "id": "34", "secret": "ab"}]
                }
            },
        ):
            self.assertEqual(
                _fetch_flickr_urls("pepaya", 2, "key"),
                ["https://live.staticflickr.com/12/34_ab_b.jpg"],
            )

        with patch(
            "papaya_cli.scraper._fetch_json",
            return_value={"images": [{"imageUrl": "https://serp.example/1.jpg"}]},
        ):
            self.assertEqual(
                _fetch_serper_urls("pepaya", 2, "key"), ["https://serp.example/1.jpg"]
            )

        with patch(
            "papaya_cli.scraper._fetch_json",
            return_value={
                "query": {
                    "pages": {
                        "1": {"imageinfo": [{"mime": "image/jpeg", "thumburl": "https://wiki.example/1.jpg"}]},
                        "2": {"imageinfo": [{"mime": "image/svg+xml", "url": "https://wiki.example/bad.svg"}]},
                    }
                }
            },
        ):
            self.assertEqual(
                _fetch_wikimedia_urls("pepaya", 5), ["https://wiki.example/1.jpg"]
            )

        with patch(
            "papaya_cli.scraper._fetch_json",
            return_value={
                "results": [{"photos": [{"url": "https://inat.example/1/square.jpg"}]}]
            },
        ):
            self.assertEqual(
                _fetch_inaturalist_urls("ripe", 5), ["https://inat.example/1/large.jpg"]
            )

    def test_unlimited_mode_runs_all_queries_and_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = ScraperConfig(
                output_dir=root,
                per_class=0,
                per_query_limit=50,
                sources=("bing", "wikimedia"),
                queries={cls_name: ["q1", "q2"] for cls_name in CLASSES},
            )
            with (
                patch("papaya_cli.scraper._crawl_with_icrawler") as m_crawl,
                patch("papaya_cli.scraper._fetch_wikimedia_urls", return_value=[]) as m_wiki,
            ):
                scrape_dataset(cfg)

            self.assertEqual(m_crawl.call_count, len(CLASSES) * 2)
            self.assertEqual(m_wiki.call_count, len(CLASSES) * 2)


if __name__ == "__main__":
    unittest.main()

