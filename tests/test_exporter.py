"""Unit tests for dataset exporter (YOLO, COCO, Pascal VOC, stratified split)."""

from __future__ import annotations

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from papaya_cli.exporter import (
    ExporterConfig,
    bbox_to_yolo,
    export_dataset,
    stratified_split,
)
from papaya_cli.utils import CLASSES


class TestExporter(unittest.TestCase):
    def test_bbox_to_yolo_normalization(self) -> None:
        xc, yc, bw, bh = bbox_to_yolo([100.0, 50.0, 300.0, 250.0], width=400, height=500)
        self.assertAlmostEqual(xc, 0.5)
        self.assertAlmostEqual(yc, 0.3)
        self.assertAlmostEqual(bw, 0.5)
        self.assertAlmostEqual(bh, 0.4)

    def test_stratified_split_is_deterministic(self) -> None:
        records = [
            {"relpath": f"{c}/img_{i}.jpg", "class": c}
            for c in ("unripe", "ripe")
            for i in range(10)
        ]
        s1 = stratified_split(records, (70, 20, 10), seed=42)
        s2 = stratified_split(records, (70, 20, 10), seed=42)
        self.assertEqual(s1, s2)
        self.assertEqual(len(s1["train"]), 14)
        self.assertEqual(len(s1["val"]), 4)
        self.assertEqual(len(s1["test"]), 2)

    def test_export_all_formats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_dir = root / "clean"
            ann_dir = root / "annotated"
            exp_dir = root / "export"
            ann_dir.mkdir(parents=True)

            images = []
            for idx, cls_name in enumerate(CLASSES):
                (clean_dir / cls_name).mkdir(parents=True)
                img_path = clean_dir / cls_name / "sample.jpg"
                img_path.write_bytes(b"jpg")
                images.append(
                    {
                        "relpath": f"{cls_name}/sample.jpg",
                        "filepath": str(img_path),
                        "filename": "sample.jpg",
                        "class": cls_name,
                        "class_id": idx,
                        "width": 400,
                        "height": 400,
                        "needs_review": idx == 4,
                        "detections": [
                            {
                                "bbox": [40.0, 80.0, 200.0, 240.0],
                                "score": 0.9,
                                "class": cls_name,
                                "class_id": idx,
                            }
                        ],
                    }
                )

            (ann_dir / "annotations.json").write_text(
                json.dumps({"classes": list(CLASSES), "images": images}),
                encoding="utf-8",
            )

            cfg = ExporterConfig(
                input_dir=ann_dir,
                output_dir=exp_dir,
                format="all",
                split=(70, 20, 10),
                seed=42,
            )
            summary = export_dataset(cfg)
            self.assertEqual(sum(summary.values()), 5)

            # YOLO artifacts
            self.assertTrue((exp_dir / "classes.txt").exists())
            self.assertTrue((exp_dir / "data.yaml").exists())
            train_labels = list((exp_dir / "labels" / "train").glob("*.txt"))
            self.assertEqual(len(train_labels), 5)
            parts = train_labels[0].read_text(encoding="utf-8").strip().split()
            self.assertEqual(len(parts), 5)

            # COCO artifacts
            coco_json = json.loads(
                (exp_dir / "coco" / "train" / "_annotations.coco.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(coco_json["categories"]), 5)
            self.assertEqual(len(coco_json["images"]), 5)
            self.assertEqual(len(coco_json["annotations"]), 5)

            # VOC artifacts
            voc_xmls = list((exp_dir / "voc" / "train" / "Annotations").glob("*.xml"))
            self.assertEqual(len(voc_xmls), 5)
            tree = ET.parse(voc_xmls[0])
            self.assertEqual(tree.getroot().findtext("object/bndbox/xmin"), "40")


if __name__ == "__main__":
    unittest.main()

