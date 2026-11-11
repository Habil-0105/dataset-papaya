"""Command-line interface for papaya-cli."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from papaya_cli.annotator import annotate_dataset, build_annotator_config
from papaya_cli.cleaner import build_cleaner_config, clean_dataset
from papaya_cli.exporter import build_exporter_config, export_dataset
from papaya_cli.scraper import build_scraper_config, scrape_dataset
from papaya_cli.utils import load_config, setup_logging


def _load_optional_config(config_path: str | None) -> dict[str, Any]:
    if config_path is None:
        default = Path("config.yaml")
        return load_config(default) if default.exists() else {}
    return load_config(config_path)


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level argparse parser with all dataset pipeline subcommands."""
    parser = argparse.ArgumentParser(
        prog="papaya-cli",
        description=(
            "Build an object detection dataset for California papaya (Carica papaya) "
            "across 5 ripeness classes: unripe, early_ripening, ripe, overripe, rotten."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging."
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # scrape
    p_scrape = sub.add_parser(
        "scrape",
        help="Scrape papaya images per ripeness class from web sources.",
    )
    p_scrape.add_argument("--config", type=str, default=None, help="Path to config.yaml.")
    p_scrape.add_argument("--output", type=Path, default=None, help="Output raw directory.")
    p_scrape.add_argument(
        "--per-class",
        type=int,
        default=None,
        help="Target images per class (0 = unlimited; default from config).",
    )
    p_scrape.add_argument("--threads", type=int, default=None, help="Download threads.")
    p_scrape.add_argument("--delay", type=float, default=None, help="Delay between requests (s).")

    # clean
    p_clean = sub.add_parser(
        "clean",
        help="Deduplicate (pHash), remove corrupt/small files, and filter non-papaya via CLIP.",
    )
    p_clean.add_argument("--config", type=str, default=None, help="Path to config.yaml.")
    p_clean.add_argument("--input", type=Path, default=None, help="Input raw directory.")
    p_clean.add_argument("--output", type=Path, default=None, help="Output clean directory.")
    p_clean.add_argument(
        "--device", type=str, default=None, help="Device: auto, cuda, or cpu."
    )

    # annotate
    p_ann = sub.add_parser(
        "annotate",
        help="Auto-annotate papaya bounding boxes and verify ripeness with CLIP.",
    )
    p_ann.add_argument("--config", type=str, default=None, help="Path to config.yaml.")
    p_ann.add_argument("--input", type=Path, default=None, help="Input clean directory.")
    p_ann.add_argument("--output", type=Path, default=None, help="Output annotated directory.")
    p_ann.add_argument(
        "--conf", type=float, default=None, help="Confidence threshold (default: 0.35)."
    )
    p_ann.add_argument(
        "--detector",
        type=str,
        default=None,
        choices=["grounding-dino", "owlvit"],
        help="Open-vocabulary detector backend.",
    )
    p_ann.add_argument(
        "--preview",
        dest="preview",
        action="store_true",
        default=None,
        help="Save bounding box visualizations in preview/.",
    )
    p_ann.add_argument(
        "--no-preview",
        dest="preview",
        action="store_false",
        help="Disable saving preview visualizations.",
    )
    p_ann.add_argument(
        "--device", type=str, default=None, help="Device: auto, cuda, or cpu."
    )

    # export
    p_exp = sub.add_parser(
        "export",
        help="Split dataset into train/val/test and export YOLO, COCO, or Pascal VOC.",
    )
    p_exp.add_argument("--config", type=str, default=None, help="Path to config.yaml.")
    p_exp.add_argument("--input", type=Path, default=None, help="Input annotated directory.")
    p_exp.add_argument("--output", type=Path, default=None, help="Output export directory.")
    p_exp.add_argument(
        "--format",
        type=str,
        default=None,
        choices=["yolo", "coco", "voc", "all"],
        help="Export annotation format (default: yolo).",
    )
    p_exp.add_argument(
        "--split",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VAL", "TEST"),
        default=None,
        help="Train/val/test split ratios (default: 70 20 10).",
    )
    p_exp.add_argument("--seed", type=int, default=None, help="Random seed for split.")
    p_exp.add_argument(
        "--verified-only",
        action="store_true",
        default=None,
        help="Exclude images flagged as needs_review.",
    )

    # run-all
    p_all = sub.add_parser(
        "run-all",
        help="Run the complete pipeline: scrape -> clean -> annotate -> export.",
    )
    p_all.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config.yaml."
    )
    p_all.add_argument("--per-class", type=int, default=None, help="Override images per class.")
    p_all.add_argument("--conf", type=float, default=None, help="Override detection confidence.")
    p_all.add_argument(
        "--format",
        type=str,
        default=None,
        choices=["yolo", "coco", "voc", "all"],
        help="Override export format.",
    )
    p_all.add_argument(
        "--device", type=str, default=None, help="Device: auto, cuda, or cpu."
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose)

    raw_cfg = _load_optional_config(args.config)

    if args.command == "scrape":
        cfg = build_scraper_config(
            raw_cfg,
            output_dir=args.output,
            per_class=args.per_class,
            threads=args.threads,
            delay=args.delay,
        )
        scrape_dataset(cfg)
    elif args.command == "clean":
        cfg = build_cleaner_config(
            raw_cfg,
            input_dir=args.input,
            output_dir=args.output,
            device=args.device,
        )
        clean_dataset(cfg)
    elif args.command == "annotate":
        cfg = build_annotator_config(
            raw_cfg,
            input_dir=args.input,
            output_dir=args.output,
            conf=args.conf,
            preview=args.preview,
            detector=args.detector,
            device=args.device,
        )
        annotate_dataset(cfg)
    elif args.command == "export":
        split_tuple = tuple(args.split) if args.split is not None else None
        cfg = build_exporter_config(
            raw_cfg,
            input_dir=args.input,
            output_dir=args.output,
            fmt=args.format,
            split=split_tuple,  # type: ignore[arg-type]
            seed=args.seed,
            verified_only=args.verified_only,
        )
        export_dataset(cfg)
    elif args.command == "run-all":
        s_cfg = build_scraper_config(raw_cfg, per_class=args.per_class)
        scrape_dataset(s_cfg)

        c_cfg = build_cleaner_config(raw_cfg, input_dir=s_cfg.output_dir, device=args.device)
        clean_dataset(c_cfg)

        a_cfg = build_annotator_config(
            raw_cfg,
            input_dir=c_cfg.output_dir,
            conf=args.conf,
            device=args.device,
        )
        annotate_dataset(a_cfg)

        e_cfg = build_exporter_config(
            raw_cfg,
            input_dir=a_cfg.output_dir,
            fmt=args.format,
        )
        export_dataset(e_cfg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

