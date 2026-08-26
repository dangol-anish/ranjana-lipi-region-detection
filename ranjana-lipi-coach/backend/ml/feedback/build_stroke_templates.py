#!/usr/bin/env python3
"""Build statistical stroke templates from normalized correct samples."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml.feedback.template_feedback import (  # noqa: E402
    DEFAULT_ALLOWED_THRESHOLD,
    DEFAULT_ALLOWED_TOLERANCE_PIXELS,
    DEFAULT_INK_THRESHOLD,
    DEFAULT_REQUIRED_THRESHOLD,
    DEFAULT_USER_TOLERANCE_PIXELS,
    build_template_from_images,
    build_template_feedback,
    save_template,
    with_baselines,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build statistical handwriting templates for every class")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=BACKEND_ROOT / "ml" / "processed_general",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=BACKEND_ROOT / "ml" / "saved_models" / "stroke_templates",
    )
    parser.add_argument("--required-threshold", type=float, default=DEFAULT_REQUIRED_THRESHOLD)
    parser.add_argument("--allowed-threshold", type=float, default=DEFAULT_ALLOWED_THRESHOLD)
    parser.add_argument("--user-tolerance-pixels", type=int, default=DEFAULT_USER_TOLERANCE_PIXELS)
    parser.add_argument("--allowed-tolerance-pixels", type=int, default=DEFAULT_ALLOWED_TOLERANCE_PIXELS)
    parser.add_argument("--ink-threshold", type=float, default=DEFAULT_INK_THRESHOLD)
    parser.add_argument("--baseline-percentile", type=float, default=80.0)
    parser.add_argument("--preview", action="store_true", help="Also save preview PNGs for inspection.")
    return parser.parse_args()


def class_dirs(processed_root: Path) -> list[Path]:
    return sorted(
        path
        for path in processed_root.iterdir()
        if path.is_dir() and path.name != "references"
    )


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


def save_preview(template, output_path: Path) -> None:
    mean = (template.mean_ink_map.clip(0.0, 1.0) * 255).astype(np.uint8)
    required = (template.required_mask.astype(np.uint8) * 255)
    allowed = (template.allowed_mask.astype(np.uint8) * 255)
    preview = np.concatenate([mean, required, allowed], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview, mode="L").save(output_path)


def percentile_baselines(template, images: list[np.ndarray], percentile: float = 95.0) -> tuple[dict[str, float], dict[str, float]]:
    fine_scores: dict[str, list[float]] = {}
    broad_scores: dict[str, list[float]] = {}
    for image in images:
        feedback = build_template_feedback(
            class_name=template.class_name,
            input_image=image,
            template=template,
            fine_problem_threshold=1.0,
            broad_problem_threshold=1.0,
        )
        for region in feedback["fine_grid"]["all_regions"]:
            fine_scores.setdefault(str(region["region"]), []).append(float(region["score"]))
        for region in feedback["broad_bands"]["all_regions"]:
            broad_scores.setdefault(str(region["region"]), []).append(float(region["score"]))

    fine_baseline = {
        region: float(np.percentile(values, percentile))
        for region, values in fine_scores.items()
    }
    broad_baseline = {
        region: float(np.percentile(values, percentile))
        for region, values in broad_scores.items()
    }
    return fine_baseline, broad_baseline


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for class_dir in class_dirs(args.processed_root):
        class_name = class_dir.name
        files = image_files(class_dir)
        if not files:
            rows.append({"class": class_name, "source_count": 0, "status": "skipped-no-images"})
            continue

        images = [read_normalized(path) for path in files]
        template = build_template_from_images(
            class_name=class_name,
            images=images,
            required_threshold=args.required_threshold,
            allowed_threshold=args.allowed_threshold,
            allowed_tolerance_pixels=args.allowed_tolerance_pixels,
            user_tolerance_pixels=args.user_tolerance_pixels,
            ink_threshold=args.ink_threshold,
        )
        fine_baseline, broad_baseline = percentile_baselines(template, images, percentile=args.baseline_percentile)
        template = with_baselines(template, fine_baseline, broad_baseline)
        save_template(template, args.output_root / f"{class_name}.npz")
        if args.preview:
            save_preview(template, args.output_root / "previews" / f"{class_name}_template_preview.png")
        rows.append(
            {
                "class": class_name,
                "source_count": template.source_count,
                "required_pixels": int(np.count_nonzero(template.required_mask)),
                "allowed_pixels": int(np.count_nonzero(template.allowed_mask)),
                "status": "built",
            }
        )
        print(
            f"{class_name}: source={template.source_count} "
            f"required={rows[-1]['required_pixels']} allowed={rows[-1]['allowed_pixels']}"
        )

    manifest_path = args.output_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["class", "source_count", "required_pixels", "allowed_pixels", "status"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved templates to: {args.output_root}")
    print(f"Saved manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
