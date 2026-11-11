"""Dataset cleaning: corrupt/size checks, perceptual deduplication, CLIP papaya filter, metadata.csv."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from papaya_cli.utils import (
    CLASSES,
    detect_device,
    iter_class_images,
    progress_iter,
)

logger = logging.getLogger("papaya_cli.cleaner")

METADATA_FIELDS = (
    "filename",
    "class",
    "source_url",
    "query",
    "width",
    "height",
    "phash",
    "sha256",
)


@dataclass
class CleanerConfig:
    """Configuration for dataset cleaning."""

    input_dir: Path = Path("data/raw")
    output_dir: Path = Path("data/clean")
    min_size: tuple[int, int] = (320, 320)
    phash_threshold: int = 8
    clip_model: str = "openai/clip-vit-base-patch32"
    clip_threshold: float = 0.6
    device: str = "auto"
    skip_clip: bool = False


class ClipPapayaFilter:
    """Zero-shot CLIP classifier checking for a whole California papaya fruit."""

    PROMPTS = (
        "a photo of a whole oblong California papaya fruit",
        "a photo of sliced papaya, papaya salad, food dish, or juice",
        "a photo of papaya leaves, seeds, tree bark, or skincare product",
        "a photo of another fruit such as mango, banana, melon, or avocado",
        "an illustration, logo, text banner, or photo without papaya fruit",
    )

    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.device = detect_device(device)
        self.model_name = model_name
        self._model: Any = None
        self._processor: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # type: ignore
            from transformers import CLIPModel, CLIPProcessor  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "torch and transformers are required for CLIP filtering. "
                "Install with: pip install torch transformers"
            ) from exc

        logger.info("Loading CLIP filter model '%s' on %s", self.model_name, self.device)
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()
        self._torch = torch

    def score_papaya(self, image_path: Path) -> float:
        """Return probability in [0, 1] that `image_path` depicts a papaya."""
        self._ensure_loaded()
        from PIL import Image  # type: ignore

        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
        inputs = self._processor(
            text=list(self.PROMPTS),
            images=rgb,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            outputs = self._model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]
        return float(probs[0].item())


def compute_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_phash(image: Any) -> str:
    """Compute 64-bit perceptual hash hex string (imagehash.phash or Pillow dHash fallback)."""
    try:
        import imagehash  # type: ignore

        return str(imagehash.phash(image))
    except ImportError:
        # 8x8 difference hash fallback using Pillow only.
        gray = image.convert("L").resize((9, 8))
        pixels = list(gray.getdata())
        bits = 0
        for row in range(8):
            offset = row * 9
            for col in range(8):
                bits = (bits << 1) | int(pixels[offset + col] > pixels[offset + col + 1])
        return f"{bits:016x}"


def hamming_distance(hex_a: str, hex_b: str) -> int:
    """Compute Hamming distance between two hex perceptual hashes."""
    return (int(hex_a, 16) ^ int(hex_b, 16)).bit_count()


def _load_sources_map(input_dir: Path) -> dict[str, dict[str, str]]:
    manifest = input_dir / "sources.jsonl"
    sources: dict[str, dict[str, str]] = {}
    if not manifest.exists():
        return sources
    with manifest.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rel = rec.get("relpath", "")
            if rel:
                sources[rel] = rec
    return sources


def _load_existing_metadata(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_metadata(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(METADATA_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in METADATA_FIELDS})


def build_cleaner_config(
    raw_cfg: dict[str, Any],
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    device: str | None = None,
) -> CleanerConfig:
    """Build `CleanerConfig` from YAML config dict and optional overrides."""
    c_cfg = raw_cfg.get("cleaner", {})
    min_sz = c_cfg.get("min_size", [320, 320])
    return CleanerConfig(
        input_dir=input_dir or Path(c_cfg.get("input_dir", "data/raw")),
        output_dir=output_dir or Path(c_cfg.get("output_dir", "data/clean")),
        min_size=(int(min_sz[0]), int(min_sz[1])),
        phash_threshold=int(c_cfg.get("phash_threshold", 8)),
        clip_model=str(c_cfg.get("clip_model", "openai/clip-vit-base-patch32")),
        clip_threshold=float(c_cfg.get("clip_threshold", 0.6)),
        device=device or str(c_cfg.get("device", "auto")),
    )


def clean_dataset(
    cfg: CleanerConfig,
    papaya_classifier: Callable[[Path], float] | None = None,
) -> dict[str, int]:
    """Clean raw images, deduplicate, filter non-papaya via CLIP, and write metadata.csv."""
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for cleaning images. Install with: pip install pillow"
        ) from exc

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cfg.output_dir / "metadata.csv"
    rows: list[dict[str, Any]] = _load_existing_metadata(metadata_path)
    seen_relpaths = {f"{r['class']}/{r['filename']}" for r in rows}
    seen_sha256 = {r["sha256"] for r in rows if r.get("sha256")}
    seen_phashes = [r["phash"] for r in rows if r.get("phash")]

    sources_map = _load_sources_map(cfg.input_dir)
    if papaya_classifier is None and not cfg.skip_clip:
        clip_filter = ClipPapayaFilter(cfg.clip_model, cfg.device)
        papaya_classifier = clip_filter.score_papaya

    stats = {
        "total": 0,
        "resumed": 0,
        "kept": len(rows),
        "corrupt_or_small": 0,
        "duplicate": 0,
        "non_papaya": 0,
    }

    items = list(iter_class_images(cfg.input_dir, CLASSES))
    for cls_name, src_path in progress_iter(items, desc="Cleaning"):
        stats["total"] += 1
        relpath = f"{cls_name}/{src_path.name}"
        dest_dir = cfg.output_dir / cls_name
        dest_path = dest_dir / src_path.name

        if relpath in seen_relpaths and dest_path.exists():
            stats["resumed"] += 1
            continue

        # 1. Corrupt & minimum size verification.
        try:
            with Image.open(src_path) as img:
                img.verify()
            with Image.open(src_path) as img:
                width, height = img.size
                if (
                    width < cfg.min_size[0]
                    or height < cfg.min_size[1]
                    or max(width / height, height / width) > 3.5
                ):
                    stats["corrupt_or_small"] += 1
                    continue
                phash = compute_phash(img)
        except Exception:
            stats["corrupt_or_small"] += 1
            continue

        # 2. Exact & perceptual duplicate check.
        sha256 = compute_sha256(src_path)
        if sha256 in seen_sha256 or any(
            hamming_distance(phash, prev) <= cfg.phash_threshold for prev in seen_phashes
        ):
            stats["duplicate"] += 1
            continue

        # 3. Zero-shot CLIP papaya presence check.
        if papaya_classifier is not None:
            try:
                score = papaya_classifier(src_path)
                if score < cfg.clip_threshold:
                    stats["non_papaya"] += 1
                    continue
            except Exception as exc:
                logger.warning("CLIP check failed on %s: %s", src_path, exc)
                continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)

        seen_relpaths.add(relpath)
        seen_sha256.add(sha256)
        seen_phashes.append(phash)

        src_meta = sources_map.get(relpath, {})
        rows.append(
            {
                "filename": src_path.name,
                "class": cls_name,
                "source_url": src_meta.get("source_url", ""),
                "query": src_meta.get("query", ""),
                "width": width,
                "height": height,
                "phash": phash,
                "sha256": sha256,
            }
        )
        stats["kept"] += 1

    _write_metadata(metadata_path, rows)
    logger.info(
        "Clean finished: %d kept (%d resumed), %d corrupt/small, %d duplicates, %d non-papaya.",
        stats["kept"],
        stats["resumed"],
        stats["corrupt_or_small"],
        stats["duplicate"],
        stats["non_papaya"],
    )
    return stats

