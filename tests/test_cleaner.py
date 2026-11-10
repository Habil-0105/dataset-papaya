"""Unit tests for dataset cleaner."""

from __future__ import annotations

import csv
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from papaya_cli.cleaner import CleanerConfig, clean_dataset, hamming_distance
from papaya_cli.scraper import SourceManifest


def _make_fake_pil(sizes: dict[str, tuple[int, int]], hashes: dict[str, int]):
    """Build a minimal PIL.Image stub so cleaner tests run without Pillow installed."""

    class FakeImageObj:
        def __init__(self, path: Path) -> None:
            self.path = path
            if path.read_bytes() == b"CORRUPT":
                raise OSError("corrupt image")
            self.size = sizes.get(path.name, (400, 400))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def verify(self) -> None:
            if self.path.read_bytes() == b"CORRUPT":
                raise OSError("corrupt")

        def convert(self, mode: str):
            return self

        def resize(self, sz):
            return self

        def getdata(self):
            # Produce deterministic 9x8 pixel grid encoded from hash value.
            val = hashes.get(self.path.name, 0)
            pixels = []
            for row in range(8):
                for col in range(8):
                    bit = (val >> (63 - (row * 8 + col))) & 1
                    pixels.append(255 if bit else 0)
                pixels.append(128)
            return pixels

    pil_mod = types.ModuleType("PIL")
    img_mod = types.ModuleType("PIL.Image")
    img_mod.open = lambda p: FakeImageObj(Path(p))
    pil_mod.Image = img_mod
    return {"PIL": pil_mod, "PIL.Image": img_mod}


class TestCleaner(unittest.TestCase):
    def test_hamming_distance(self) -> None:
        self.assertEqual(hamming_distance("0000000000000000", "000000000000000f"), 4)

    def test_clean_dataset_filters_corrupt_small_duplicates_and_non_papaya(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            clean_dir = root / "clean"
            (raw_dir / "ripe").mkdir(parents=True)

            (raw_dir / "ripe" / "good.jpg").write_bytes(b"good-img")
            (raw_dir / "ripe" / "good_dup.jpg").write_bytes(b"good-img")
            (raw_dir / "ripe" / "small.jpg").write_bytes(b"small-img")
            (raw_dir / "ripe" / "corrupt.jpg").write_bytes(b"CORRUPT")
            (raw_dir / "ripe" / "not_papaya.jpg").write_bytes(b"banana-img")

            manifest = SourceManifest(raw_dir / "sources.jsonl")
            manifest.add("ripe/good.jpg", "https://example.com/good.jpg", "ripe papaya", "ripe")

            fake_pil = _make_fake_pil(
                sizes={"small.jpg": (100, 100)},
                hashes={
                    "good.jpg": 0xAAAAAAAAAAAAAAAA,
                    "good_dup.jpg": 0xAAAAAAAAAAAAAAAA,
                    "not_papaya.jpg": 0x0000000000000000,
                },
            )
            cfg = CleanerConfig(
                input_dir=raw_dir,
                output_dir=clean_dir,
                min_size=(320, 320),
                clip_threshold=0.5,
            )

            def fake_clip(p: Path) -> float:
                return 0.1 if "not_papaya" in p.name else 0.9

            with patch.dict(sys.modules, fake_pil):
                stats = clean_dataset(cfg, papaya_classifier=fake_clip)

            self.assertEqual(stats["kept"], 1)
            self.assertEqual(stats["corrupt_or_small"], 2)
            self.assertEqual(stats["duplicate"], 1)
            self.assertEqual(stats["non_papaya"], 1)
            self.assertTrue((clean_dir / "ripe" / "good.jpg").exists())

            with (clean_dir / "metadata.csv").open("r", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source_url"], "https://example.com/good.jpg")
            self.assertEqual(rows[0]["query"], "ripe papaya")


if __name__ == "__main__":
    unittest.main()
