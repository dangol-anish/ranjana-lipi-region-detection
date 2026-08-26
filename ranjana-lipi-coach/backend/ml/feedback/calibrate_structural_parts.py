#!/usr/bin/env python3
"""Calibrate structural part coverage thresholds from real correct samples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml.feedback.structural_part_feedback import (  # noqa: E402
    VALIDATED_STRUCTURAL_CLASSES,
    load_structural_part_template,
)
from ml.feedback.template_feedback import dilate_mask  # noqa: E402


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate structural part thresholds")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=BACKEND_ROOT / "ml" / "processed",
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=BACKEND_ROOT / "ml" / "saved_models" / "structural_part_masks",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=BACKEND_ROOT / "ml" / "saved_models" / "structural_part_masks" / "calibration_report.csv",
    )
    parser.add_argument("--samples-per-class", type=int, default=60)
    parser.add_argument("--percentile", type=float, default=10.0)
    parser.add_argument("--safety-margin", type=float, default=0.05)
    parser.add_argument("--min-threshold", type=float, default=0.20)
    parser.add_argument("--max-threshold", type=float, default=0.70)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_normalized(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return (image.astype(np.float32) / 255.0).clip(0.0, 1.0)


def part_coverage(image: np.ndarray, mask: np.ndarray, tolerance_pixels: int) -> float:
    user_ink = image > 0.05
    tolerated_user = dilate_mask(user_ink, tolerance_pixels)
    mask_pixels = int(np.count_nonzero(mask))
    if mask_pixels == 0:
        return 1.0
    covered_pixels = int(np.count_nonzero(mask & tolerated_user))
    return covered_pixels / mask_pixels


def calibrated_threshold(
    coverages: list[float],
    percentile: float,
    safety_margin: float,
    min_threshold: float,
    max_threshold: float,
) -> float:
    if not coverages:
        return min_threshold
    raw = float(np.percentile(np.asarray(coverages, dtype=np.float32), percentile))
    return float(np.clip(raw - safety_margin, min_threshold, max_threshold))


def load_parts_config(mask_root: Path, class_name: str) -> dict[str, Any]:
    path = mask_root / class_name / "parts.json"
    return json.loads(path.read_text(encoding="utf-8"))


def save_parts_config(mask_root: Path, class_name: str, config: dict[str, Any]) -> None:
    path = mask_root / class_name / "parts.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def calibrate_class(
    class_name: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    template = load_structural_part_template(class_name, args.mask_root)
    samples = image_files(args.processed_root / class_name)[: args.samples_per_class]
    images = [read_normalized(path) for path in samples]
    config = load_parts_config(args.mask_root, class_name)
    rows: list[dict[str, Any]] = []

    for part_config, part in zip(config["parts"], template.parts):
        coverages = [
            part_coverage(image, part.mask, template.tolerance_pixels)
            for image in images
        ]
        old_threshold = float(part_config.get("min_coverage", part.min_coverage))
        new_threshold = calibrated_threshold(
            coverages=coverages,
            percentile=args.percentile,
            safety_margin=args.safety_margin,
            min_threshold=args.min_threshold,
            max_threshold=args.max_threshold,
        )
        if not args.dry_run:
            part_config["min_coverage"] = round(new_threshold, 4)
            part_config["calibration"] = {
                "samples": len(coverages),
                "percentile": args.percentile,
                "safety_margin": args.safety_margin,
                "p10": float(np.percentile(coverages, 10)) if coverages else None,
                "p25": float(np.percentile(coverages, 25)) if coverages else None,
                "median": float(np.percentile(coverages, 50)) if coverages else None,
                "mean": float(np.mean(coverages)) if coverages else None,
                "min": float(np.min(coverages)) if coverages else None,
                "max": float(np.max(coverages)) if coverages else None,
            }
        rows.append(
            {
                "class": class_name,
                "part": part.name,
                "label": part.label,
                "samples": len(coverages),
                "old_min_coverage": old_threshold,
                "new_min_coverage": new_threshold,
                "min": float(np.min(coverages)) if coverages else 0.0,
                "p10": float(np.percentile(coverages, 10)) if coverages else 0.0,
                "p25": float(np.percentile(coverages, 25)) if coverages else 0.0,
                "median": float(np.percentile(coverages, 50)) if coverages else 0.0,
                "mean": float(np.mean(coverages)) if coverages else 0.0,
                "max": float(np.max(coverages)) if coverages else 0.0,
            }
        )

    if not args.dry_run:
        save_parts_config(args.mask_root, class_name, config)
    return rows


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    all_rows: list[dict[str, Any]] = []
    for class_name in VALIDATED_STRUCTURAL_CLASSES:
        rows = calibrate_class(class_name, args)
        all_rows.extend(rows)
        print(f"{class_name}: calibrated {len(rows)} parts")

    write_report(args.report, all_rows)
    print(f"Saved calibration report: {args.report}")
    if args.dry_run:
        print("Dry run only: parts.json files were not updated.")


if __name__ == "__main__":
    main()
