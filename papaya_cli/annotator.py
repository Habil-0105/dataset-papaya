"""Auto-annotation with Grounding DINO / OWL-ViT, CLIP ripeness check, and preview rendering."""

from __future__ import annotations

import json
import logging
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

logger = logging.getLogger("papaya_cli.annotator")

RIPENESS_PROMPTS: dict[str, str] = {
    "unripe": "a photo of an unripe green papaya fruit",
    "early_ripening": "a photo of a half-ripe papaya fruit turning yellow",
    "ripe": "a photo of a ripe orange papaya fruit",
    "overripe": "a photo of an overripe soft wrinkled papaya fruit",
    "rotten": "a photo of a rotten moldy spoiled papaya fruit",
}


@dataclass
class AnnotatorConfig:
    """Configuration for open-vocabulary detection and ripeness verification."""

    input_dir: Path = Path("data/clean")
    output_dir: Path = Path("data/annotated")
    detector: str = "grounding-dino"
    detector_model: str = "IDEA-Research/grounding-dino-tiny"
    clip_model: str = "openai/clip-vit-base-patch32"
    prompt: str = "papaya"
    conf_threshold: float = 0.35
    preview: bool = True
    device: str = "auto"


class OpenVocabDetector:
    """Open-vocabulary object detector supporting Grounding DINO and OWL-ViT."""

    def __init__(
        self,
        backend: str = "grounding-dino",
        model_name: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "auto",
    ) -> None:
        self.backend = backend.lower()
        self.model_name = model_name
        self.device = detect_device(device)
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # type: ignore
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "torch and transformers are required for auto-annotation. "
                "Install with: pip install torch transformers"
            ) from exc

        logger.info("Loading detector '%s' (%s) on %s", self.model_name, self.backend, self.device)
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_name
        ).to(self.device)
        self._model.eval()
        self._torch = torch

    def detect(
        self, image: Any, prompt: str, conf_threshold: float
    ) -> list[dict[str, Any]]:
        """Return list of `{bbox: [x1, y1, x2, y2], score: float}` above `conf_threshold`."""
        self._ensure_loaded()
        rgb = image.convert("RGB")
        width, height = rgb.size

        if "owl" in self.backend or "owl" in self.model_name.lower():
            inputs = self._processor(
                text=[[f"a photo of a {prompt}"]], images=rgb, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self._torch.no_grad():
                outputs = self._model(**inputs)
            target_sizes = self._torch.tensor([[height, width]], device=self.device)
            result = self._processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=conf_threshold
            )[0]
        else:
            gd_prompt = f"{prompt.strip().rstrip('.').lower()}."
            inputs = self._processor(images=rgb, text=gd_prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self._torch.no_grad():
                outputs = self._model(**inputs)
            try:
                result = self._processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    box_threshold=conf_threshold,
                    text_threshold=conf_threshold,
                    target_sizes=[(height, width)],
                )[0]
            except TypeError:
                result = self._processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    threshold=conf_threshold,
                    text_threshold=conf_threshold,
                    target_sizes=[(height, width)],
                )[0]

        boxes: list[dict[str, Any]] = []
        for box, score in zip(result["boxes"].tolist(), result["scores"].tolist()):
            if float(score) < conf_threshold:
                continue
            x1, y1, x2, y2 = box
            x1 = max(0.0, min(float(width), float(x1)))
            y1 = max(0.0, min(float(height), float(y1)))
            x2 = max(0.0, min(float(width), float(x2)))
            y2 = max(0.0, min(float(height), float(y2)))
            if x2 > x1 and y2 > y1:
                boxes.append({"bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)], "score": round(float(score), 4)})
        return boxes


class ClipRipenessClassifier:
    """Zero-shot CLIP classifier verifying papaya ripeness class."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = model_name
        self.device = detect_device(device)
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # type: ignore
            from transformers import CLIPModel, CLIPProcessor  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "torch and transformers are required for CLIP ripeness verification."
            ) from exc

        logger.info("Loading CLIP ripeness model '%s' on %s", self.model_name, self.device)
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()
        self._torch = torch

    def classify(self, image: Any) -> tuple[str, float]:
        """Return `(predicted_class, confidence)` across the 5 ripeness classes."""
        self._ensure_loaded()
        prompts = [RIPENESS_PROMPTS[c] for c in CLASSES]
        inputs = self._processor(
            text=prompts,
            images=image.convert("RGB"),
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            outputs = self._model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]
        idx = int(probs.argmax().item())
        return CLASSES[idx], float(probs[idx].item())


def render_preview(
    image: Any,
    detections: list[dict[str, Any]],
    cls_name: str,
    needs_review: bool,
    dest_path: Path,
) -> None:
    """Draw bounding boxes and ripeness/review labels and save to `dest_path`."""
    from PIL import ImageDraw  # type: ignore

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    color = (220, 50, 47) if needs_review else (46, 160, 67)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        score = det["score"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        tag = f"{cls_name} {score:.2f}"
        if needs_review:
            tag += " [NEEDS REVIEW]"
        text_y = max(0.0, y1 - 14.0)
        draw.text((x1 + 2, text_y), tag, fill=color)

    if not detections:
        draw.text((8, 8), f"{cls_name} [NO DETECTION - NEEDS REVIEW]", fill=color)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest_path, format="JPEG", quality=90)


def build_annotator_config(
    raw_cfg: dict[str, Any],
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    conf: float | None = None,
    preview: bool | None = None,
    detector: str | None = None,
    device: str | None = None,
) -> AnnotatorConfig:
    """Build `AnnotatorConfig` from YAML dict and CLI overrides."""
    a_cfg = raw_cfg.get("annotator", {})
    return AnnotatorConfig(
        input_dir=input_dir or Path(a_cfg.get("input_dir", "data/clean")),
        output_dir=output_dir or Path(a_cfg.get("output_dir", "data/annotated")),
        detector=detector or str(a_cfg.get("detector", "grounding-dino")),
        detector_model=str(a_cfg.get("detector_model", "IDEA-Research/grounding-dino-tiny")),
        clip_model=str(a_cfg.get("clip_model", "openai/clip-vit-base-patch32")),
        prompt=str(a_cfg.get("prompt", "papaya")),
        conf_threshold=conf if conf is not None else float(a_cfg.get("conf_threshold", 0.35)),
        preview=preview if preview is not None else bool(a_cfg.get("preview", True)),
        device=device or str(a_cfg.get("device", "auto")),
    )


def annotate_dataset(
    cfg: AnnotatorConfig,
    detector_fn: Callable[[Any, str, float], list[dict[str, Any]]] | None = None,
    ripeness_fn: Callable[[Any], tuple[str, float]] | None = None,
) -> dict[str, int]:
    """Auto-annotate clean images, verify ripeness with CLIP, and optionally render previews."""
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for annotation. Install with: pip install pillow"
        ) from exc

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cfg.output_dir / "annotations.json"

    existing_records: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            for item in loaded.get("images", []):
                existing_records[item["relpath"]] = item
        except Exception:
            existing_records = {}

    if detector_fn is None:
        det_model = OpenVocabDetector(cfg.detector, cfg.detector_model, cfg.device)
        detector_fn = det_model.detect
    if ripeness_fn is None:
        rip_model = ClipRipenessClassifier(cfg.clip_model, cfg.device)
        ripeness_fn = rip_model.classify

    stats = {
        "total": 0,
        "resumed": 0,
        "annotated": 0,
        "verified": 0,
        "needs_review": 0,
        "boxes": 0,
    }

    items = list(iter_class_images(cfg.input_dir, CLASSES))
    for cls_name, img_path in progress_iter(items, desc="Annotating"):
        stats["total"] += 1
        relpath = f"{cls_name}/{img_path.name}"

        if relpath in existing_records:
            rec = existing_records[relpath]
            stats["resumed"] += 1
            stats["annotated"] += 1
            stats["boxes"] += len(rec.get("detections", []))
            if rec.get("needs_review"):
                stats["needs_review"] += 1
            else:
                stats["verified"] += 1
            continue

        with Image.open(img_path) as img:
            rgb = img.convert("RGB")
            width, height = rgb.size
            raw_boxes = detector_fn(rgb, cfg.prompt, cfg.conf_threshold)

            detections: list[dict[str, Any]] = []
            for box in raw_boxes:
                if float(box["score"]) >= cfg.conf_threshold:
                    detections.append(
                        {
                            "bbox": box["bbox"],
                            "score": float(box["score"]),
                            "class": cls_name,
                            "class_id": CLASSES.index(cls_name),
                        }
                    )

            # Verify ripeness on top-scoring crop when available, else full image.
            crop_img = rgb
            if detections:
                top_box = max(detections, key=lambda d: d["score"])["bbox"]
                crop_img = rgb.crop(tuple(int(round(v)) for v in top_box))
            clip_class, clip_conf = ripeness_fn(crop_img)

            needs_review = (clip_class != cls_name) or (len(detections) == 0)
            status = "needs_review" if needs_review else "verified"

            if cfg.preview:
                preview_path = cfg.output_dir / "preview" / cls_name / f"{img_path.stem}.jpg"
                render_preview(rgb, detections, cls_name, needs_review, preview_path)

        record = {
            "relpath": relpath,
            "filepath": str(img_path.resolve()),
            "filename": img_path.name,
            "class": cls_name,
            "class_id": CLASSES.index(cls_name),
            "width": width,
            "height": height,
            "clip_class": clip_class,
            "clip_confidence": round(float(clip_conf), 4),
            "needs_review": needs_review,
            "status": status,
            "detections": detections,
        }
        existing_records[relpath] = record

        stats["annotated"] += 1
        stats["boxes"] += len(detections)
        if needs_review:
            stats["needs_review"] += 1
        else:
            stats["verified"] += 1

    payload = {
        "classes": list(CLASSES),
        "conf_threshold": cfg.conf_threshold,
        "images": [existing_records[k] for k in sorted(existing_records)],
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "Annotate finished: %d images (%d resumed), %d verified, %d needs_review, %d boxes.",
        stats["annotated"],
        stats["resumed"],
        stats["verified"],
        stats["needs_review"],
        stats["boxes"],
    )
    return stats

