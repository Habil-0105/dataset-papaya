"""Unit tests for auto-annotator."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from papaya_cli.annotator import AnnotatorConfig, annotate_dataset


def _make_fake_pil():
    class FakeDraw:
        def rectangle(self, *args, **kwargs):
            pass

        def text(self, *args, **kwargs):
            pass

    class FakeImageObj:
        size = (640, 480)

        def __init__(self, path: Path) -> None:
            self.name = path.name

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def convert(self, mode: str):
            return self

        def copy(self):
            return self

        def crop(self, box):
            return self

        def save(self, dest: Path, **kwargs):
            Path(dest).write_bytes(b"preview-jpg")

    pil_mod = types.ModuleType("PIL")
    img_mod = types.ModuleType("PIL.Image")
    draw_mod = types.ModuleType("PIL.ImageDraw")
    img_mod.open = lambda p: FakeImageObj(Path(p))
    draw_mod.Draw = lambda img: FakeDraw()
    pil_mod.Image = img_mod
    pil_mod.ImageDraw = draw_mod
    return {"PIL": pil_mod, "PIL.Image": img_mod, "PIL.ImageDraw": draw_mod}


class TestAnnotator(unittest.TestCase):
    def test_annotate_multi_object_needs_review_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_dir = root / "clean"
            ann_dir = root / "annotated"
            (clean_dir / "ripe").mkdir(parents=True)
            (clean_dir / "unripe").mkdir(parents=True)

            (clean_dir / "ripe" / "a.jpg").write_bytes(b"a")
            (clean_dir / "unripe" / "b.jpg").write_bytes(b"b")

            def fake_detector(img, prompt: str, conf: float):
                return [
                    {"bbox": [10.0, 20.0, 110.0, 120.0], "score": 0.9},
                    {"bbox": [200.0, 100.0, 350.0, 300.0], "score": 0.5},
                    {"bbox": [0.0, 0.0, 10.0, 10.0], "score": 0.2},  # below conf
                ]

            def fake_ripeness(img):
                # a.jpg is in ripe/ (matches), b.jpg is in unripe/ (mismatches -> needs_review)
                if getattr(img, "name", "") == "a.jpg":
                    return ("ripe", 0.88)
                return ("rotten", 0.77)

            cfg = AnnotatorConfig(
                input_dir=clean_dir,
                output_dir=ann_dir,
                conf_threshold=0.35,
                preview=True,
            )

            with patch.dict(sys.modules, _make_fake_pil()):
                stats = annotate_dataset(cfg, detector_fn=fake_detector, ripeness_fn=fake_ripeness)
                # Second run should resume without calling detector/ripeness again.
                stats_resume = annotate_dataset(
                    cfg,
                    detector_fn=lambda *a: self.fail("should not re-detect"),
                    ripeness_fn=lambda *a: self.fail("should not re-classify"),
                )

            self.assertEqual(stats["annotated"], 2)
            self.assertEqual(stats["verified"], 1)
            self.assertEqual(stats["needs_review"], 1)
            self.assertEqual(stats["boxes"], 4)
            self.assertEqual(stats_resume["resumed"], 2)

            manifest = json.loads((ann_dir / "annotations.json").read_text(encoding="utf-8"))
            by_rel = {item["relpath"]: item for item in manifest["images"]}
            self.assertFalse(by_rel["ripe/a.jpg"]["needs_review"])
            self.assertTrue(by_rel["unripe/b.jpg"]["needs_review"])
            self.assertTrue((ann_dir / "preview" / "ripe" / "a.jpg").exists())
            self.assertTrue((ann_dir / "preview" / "unripe" / "b.jpg").exists())


if __name__ == "__main__":
    unittest.main()
