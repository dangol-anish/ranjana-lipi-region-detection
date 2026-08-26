#!/usr/bin/env python3
"""Build structural masks from the controlled good/flawed validation set."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml.feedback.structural_part_feedback import VALIDATED_STRUCTURAL_CLASSES  # noqa: E402
from ml.preprocessing.normalize import apply_fixed_transform, compute_reference_transform  # noqa: E402


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
BANDS = {
    "top": ("Top structural part", "top", "top-center"),
    "middle": ("Middle structural part", "middle", "middle-center"),
    "bottom": ("Bottom structural part", "bottom", "bottom-center"),
}


def structural_data_root() -> Path:
    preferred = PROJECT_ROOT / "data" / "StructuralValidation"
    actual = PROJECT_ROOT / "data" / "StructureValidation"
    return preferred if preferred.is_dir() else actual


def output_root() -> Path:
    return BACKEND_ROOT / "ml" / "saved_models" / "structural_part_masks"


def controlled_transform_path() -> Path:
    return output_root() / "controlled_alignment_transforms.json"


def first_image(directory: Path) -> Path:
    images = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No image files found in {directory}")
    return images[0]


def load_normalized(path: Path, transform: dict[str, Any]) -> np.ndarray:
    return apply_fixed_transform(str(path), transform, canvas_size=128)


def region_mask(region: str, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    y0 = {"top": 0, "middle": height // 3, "bottom": (2 * height) // 3}[region]
    y1 = {"top": height // 3, "middle": (2 * height) // 3, "bottom": height}[region]
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, :] = True
    return mask


def clean_mask(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= 8:
            cleaned[labels == label] = 255
    if cv2.countNonZero(cleaned) == 0:
        cleaned = binary
    return cleaned > 0


def build_missing_part_mask(
    good: np.ndarray,
    flawed: np.ndarray,
    region: str,
) -> np.ndarray:
    good_ink = good > 0.05
    flawed_ink = flawed > 0.05
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tolerated_flawed = cv2.dilate(flawed_ink.astype(np.uint8), kernel, iterations=1) > 0
    missing = good_ink & ~tolerated_flawed & region_mask(region, good.shape)
    missing = clean_mask(missing)

    if int(np.count_nonzero(missing)) < 12:
        # Fallback: if the edited flaw is too subtle, use the good ink inside
        # the broad intended region so the sample still defines a checkable part.
        missing = clean_mask(good_ink & region_mask(region, good.shape))

    return missing.astype(np.uint8) * 255


def save_overlay(class_name: str, good: np.ndarray, masks: list[tuple[str, np.ndarray]]) -> None:
    base = (good * 255).astype(np.uint8)
    overlay = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    colors = {
        "top": np.asarray([255, 70, 70], dtype=np.uint8),
        "middle": np.asarray([255, 190, 40], dtype=np.uint8),
        "bottom": np.asarray([70, 140, 255], dtype=np.uint8),
    }
    for name, mask in masks:
        color = colors[name]
        active = mask > 0
        overlay[active] = (0.35 * overlay[active] + 0.65 * color).astype(np.uint8)
    Image.fromarray(overlay).resize((384, 384), Image.Resampling.NEAREST).save(
        output_root() / class_name / "parts_overlay.png"
    )


def main() -> None:
    data_root = structural_data_root()
    if not data_root.is_dir():
        raise FileNotFoundError("Missing data/StructuralValidation or data/StructureValidation")

    root = output_root()
    root.mkdir(parents=True, exist_ok=True)
    transforms: dict[str, Any] = {}

    for class_name in VALIDATED_STRUCTURAL_CLASSES:
        class_root = data_root / class_name
        good_path = first_image(class_root / "good")
        transform = compute_reference_transform(str(good_path), canvas_size=128)
        transforms[class_name] = transform
        good = load_normalized(good_path, transform)

        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "class_name": class_name,
            "source": str(data_root),
            "good_source": str(good_path),
            "mask_builder": "controlled_good_minus_flawed",
            "tolerance_pixels": 2,
            "parts": [],
        }
        masks_for_overlay: list[tuple[str, np.ndarray]] = []

        for region, (label, broad_region, fine_region) in BANDS.items():
            flawed_path = first_image(class_root / region)
            flawed = load_normalized(flawed_path, transform)
            mask = build_missing_part_mask(good, flawed, region)
            mask_name = f"{region}_required.png"
            Image.fromarray(mask).save(class_dir / mask_name)
            masks_for_overlay.append((region, mask))
            config["parts"].append(
                {
                    "name": f"{region}_required",
                    "label": label,
                    "mask": mask_name,
                    "broad_region": broad_region,
                    "fine_region": fine_region,
                    "min_coverage": 0.55,
                    "required": True,
                    "source_flaw": str(flawed_path),
                }
            )

        (class_dir / "parts.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        save_overlay(class_name, good, masks_for_overlay)
        print(f"{class_name}: saved controlled structural masks from {class_root}")

    controlled_transform_path().write_text(json.dumps(transforms, indent=2), encoding="utf-8")
    print(f"Saved controlled transforms: {controlled_transform_path()}")


if __name__ == "__main__":
    main()
