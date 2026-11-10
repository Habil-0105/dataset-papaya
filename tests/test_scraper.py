"""Unit tests for scraper and shared utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from papaya_cli.scraper import (
    ScraperConfig,
    SourceManifest,
    count_existing_images,
    scrape_dataset,
)
from papaya_cli.utils import CLASSES, RobotsChecker, retry_with_backoff


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


if __name__ == "__main__":
    unittest.main()

