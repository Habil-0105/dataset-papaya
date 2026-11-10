"""Dataset exporter supporting deterministic train/val/test splits and YOLO, COCO, VOC formats."""

from __future__ import annotations

import json
import logging
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from papaya_cli.utils import CLASSES, progress_iter

logger = logging.getLogger("papaya_cli.exporter")


@dataclass
class ExporterConfig:
    """Configuration for dataset splitting and exporting."""

    input_dir: Path = Path("data/annotated")
    output_dir: Path = Path("data/export")
    format: str = "yolo"
    split: tuple[float, float, float] = (70.0, 20.0, 10.0)
    seed: int = 42
    verified_only: bool = False


def stratified_split(
    records: list[dict[str, Any]],
    split_ratios: tuple[float, float, float] = (70.0, 20.0, 10.0),
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Deterministically split `records` into train/val/test stratified by class."""
    total_ratio = sum(split_ratios)
    if total_ratio <= 0 or any(r < 0 for r in split_ratios):
        raise ValueError(f"Invalid split ratios: {split_ratios}")
    r_train, r_val, _ = (r / total_ratio for r in split_ratios)

    rng = random.Random(seed)
    by_class: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_class.setdefault(rec["class"], []).append(rec)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for cls_name in sorted(by_class):
        group = list(by_class[cls_name])
        group.sort(key=lambda r: r["relpath"])
        rng.shuffle(group)
        n = len(group)
        n_train = int(round(n * r_train))
        n_val = int(round(n * r_val))
        if n > 0 and n_train == 0 and r_train > 0:
            n_train = 1
        if n_train + n_val > n:
            n_val = max(0, n - n_train)

        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train : n_train + n_val])
        splits["test"].extend(group[n_train + n_val :])

    return splits


def bbox_to_yolo(
    bbox: list[float], width: int, height: int
) -> tuple[float, float, float, float]:
    """Convert `[x1, y1, x2, y2]` pixel bbox to normalized `(xc, yc, w, h)`."""
    x1, y1, x2, y2 = bbox
    xc = max(0.0, min(1.0, ((x1 + x2) / 2.0) / width))
    yc = max(0.0, min(1.0, ((y1 + y2) / 2.0) / height))
    bw = max(0.0, min(1.0, (x2 - x1) / width))
    bh = max(0.0, min(1.0, (y2 - y1) / height))
    return xc, yc, bw, bh


def _export_filename(rec: dict[str, Any]) -> str:
    return f"{rec['class']}__{rec['filename']}"


def _resolve_source_image(rec: dict[str, Any], input_dir: Path) -> Path:
    fp = Path(rec["filepath"])
    if fp.exists():
        return fp
    fallback = input_dir.parent / "clean" / rec["relpath"]
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Source image not found for {rec['relpath']}: {fp}")


def _export_yolo(
    splits: dict[str, list[dict[str, Any]]],
    input_dir: Path,
    out_dir: Path,
    classes: list[str],
) -> None:
    for split_name, items in splits.items():
        img_dir = out_dir / "images" / split_name
        lbl_dir = out_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for rec in progress_iter(items, desc=f"YOLO {split_name}"):
            src_img = _resolve_source_image(rec, input_dir)
            exp_name = _export_filename(rec)
            shutil.copy2(src_img, img_dir / exp_name)

            w, h = int(rec["width"]), int(rec["height"])
            lines: list[str] = []
            for det in rec.get("detections", []):
                xc, yc, bw, bh = bbox_to_yolo(det["bbox"], w, h)
                cid = int(det.get("class_id", classes.index(rec["class"])))
                lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            lbl_path = lbl_dir / f"{Path(exp_name).stem}.txt"
            lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    (out_dir / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    names_yaml = "\n".join(f"  {i}: {name}" for i, name in enumerate(classes))
    data_yaml = (
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(classes)}\n"
        f"names:\n{names_yaml}\n"
    )
    (out_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")


def _export_coco(
    splits: dict[str, list[dict[str, Any]]],
    input_dir: Path,
    out_dir: Path,
    classes: list[str],
) -> None:
    categories = [
        {"id": i + 1, "name": name, "supercategory": "papaya"}
        for i, name in enumerate(classes)
    ]

    for split_name, items in splits.items():
        split_dir = out_dir / "coco" / split_name
        img_dir = split_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        images_json: list[dict[str, Any]] = []
        ann_json: list[dict[str, Any]] = []
        ann_id = 1

        for img_id, rec in enumerate(progress_iter(items, desc=f"COCO {split_name}"), start=1):
            src_img = _resolve_source_image(rec, input_dir)
            exp_name = _export_filename(rec)
            shutil.copy2(src_img, img_dir / exp_name)

            images_json.append(
                {
                    "id": img_id,
                    "file_name": exp_name,
                    "width": int(rec["width"]),
                    "height": int(rec["height"]),
                }
            )
            for det in rec.get("detections", []):
                x1, y1, x2, y2 = det["bbox"]
                bw = max(0.0, float(x2) - float(x1))
                bh = max(0.0, float(y2) - float(y1))
                cid = int(det.get("class_id", classes.index(rec["class"]))) + 1
                ann_json.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cid,
                        "bbox": [round(float(x1), 2), round(float(y1), 2), round(bw, 2), round(bh, 2)],
                        "area": round(bw * bh, 2),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

        coco_payload = {
            "images": images_json,
            "annotations": ann_json,
            "categories": categories,
        }
        (split_dir / "_annotations.coco.json").write_text(
            json.dumps(coco_payload, indent=2), encoding="utf-8"
        )


def _export_voc(
    splits: dict[str, list[dict[str, Any]]],
    input_dir: Path,
    out_dir: Path,
) -> None:
    for split_name, items in splits.items():
        img_dir = out_dir / "voc" / split_name / "JPEGImages"
        ann_dir = out_dir / "voc" / split_name / "Annotations"
        img_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)

        for rec in progress_iter(items, desc=f"VOC {split_name}"):
            src_img = _resolve_source_image(rec, input_dir)
            exp_name = _export_filename(rec)
            shutil.copy2(src_img, img_dir / exp_name)

            root = ET.Element("annotation")
            ET.SubElement(root, "folder").text = "JPEGImages"
            ET.SubElement(root, "filename").text = exp_name
            size = ET.SubElement(root, "size")
            ET.SubElement(size, "width").text = str(int(rec["width"]))
            ET.SubElement(size, "height").text = str(int(rec["height"]))
            ET.SubElement(size, "depth").text = "3"

            for det in rec.get("detections", []):
                obj = ET.SubElement(root, "object")
                ET.SubElement(obj, "name").text = str(det.get("class", rec["class"]))
                ET.SubElement(obj, "difficult").text = "0"
                bnd = ET.SubElement(obj, "bndbox")
                x1, y1, x2, y2 = det["bbox"]
                ET.SubElement(bnd, "xmin").text = str(int(round(x1)))
                ET.SubElement(bnd, "ymin").text = str(int(round(y1)))
                ET.SubElement(bnd, "xmax").text = str(int(round(x2)))
                ET.SubElement(bnd, "ymax").text = str(int(round(y2)))

            tree = ET.ElementTree(root)
            ET.indent(tree, space="  ")
            tree.write(ann_dir / f"{Path(exp_name).stem}.xml", encoding="utf-8", xml_declaration=True)


def build_exporter_config(
    raw_cfg: dict[str, Any],
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    fmt: str | None = None,
    split: tuple[float, float, float] | None = None,
    seed: int | None = None,
    verified_only: bool | None = None,
) -> ExporterConfig:
    """Build `ExporterConfig` from YAML config dict and CLI overrides."""
    e_cfg = raw_cfg.get("exporter", {})
    raw_split = e_cfg.get("split", [70, 20, 10])
    return ExporterConfig(
        input_dir=input_dir or Path(e_cfg.get("input_dir", "data/annotated")),
        output_dir=output_dir or Path(e_cfg.get("output_dir", "data/export")),
        format=(fmt or str(e_cfg.get("format", "yolo"))).lower(),
        split=split or (float(raw_split[0]), float(raw_split[1]), float(raw_split[2])),
        seed=seed if seed is not None else int(e_cfg.get("seed", 42)),
        verified_only=(
            verified_only
            if verified_only is not None
            else bool(e_cfg.get("verified_only", False))
        ),
    )


def export_dataset(cfg: ExporterConfig) -> dict[str, int]:
    """Split and export annotated dataset in YOLO, COCO, Pascal VOC, or all formats."""
    manifest_path = cfg.input_dir / "annotations.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Annotations manifest not found: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    classes = list(payload.get("classes") or CLASSES)
    records = list(payload.get("images", []))

    if cfg.verified_only:
        records = [r for r in records if not r.get("needs_review")]

    splits = stratified_split(records, cfg.split, cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    fmt = cfg.format.lower()
    if fmt not in ("yolo", "coco", "voc", "all"):
        raise ValueError(f"Unsupported export format: {cfg.format}")

    if fmt in ("yolo", "all"):
        _export_yolo(splits, cfg.input_dir, cfg.output_dir, classes)
    if fmt in ("coco", "all"):
        _export_coco(splits, cfg.input_dir, cfg.output_dir, classes)
    if fmt in ("voc", "all"):
        _export_voc(splits, cfg.input_dir, cfg.output_dir)

    summary = {k: len(v) for k, v in splits.items()}
    logger.info(
        "Export (%s) complete -> train=%d, val=%d, test=%d at %s",
        fmt,
        summary["train"],
        summary["val"],
        summary["test"],
        cfg.output_dir,
    )
    return summary

