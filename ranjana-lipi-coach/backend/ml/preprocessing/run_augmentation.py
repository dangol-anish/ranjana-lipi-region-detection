#!/usr/bin/env python3
"""Generate augmented normalized samples for the selected Phase 2 classes."""

from __future__ import annotations

import shutil
import sys
from itertools import cycle
from pathlib import Path

import cv2

try:
    from .augment import augment_image
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from augment import augment_image


TARGET_COUNT = 800
SELECTED_CLASSES = ("aa", "a", "ka", "da", "dda")


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def class_images(class_dir: Path) -> list[Path]:
    return sorted(path for path in class_dir.glob("*.png") if path.is_file())


def reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def augment_class(processed_root: Path, augmented_root: Path, class_name: str) -> int:
    source_dir = processed_root / class_name
    output_dir = augmented_root / class_name
    source_images = class_images(source_dir)

    if not source_images:
        raise FileNotFoundError(f"No normalized PNG images found in {source_dir}")

    reset_output_dir(output_dir)

    for index, source_path in enumerate(cycle(source_images), start=1):
        if index > TARGET_COUNT:
            break

        image = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[skip] Could not read normalized image: {source_path}")
            continue

        augmented = augment_image(image.astype("float32") / 255.0)
        output_path = output_dir / f"{class_name}_{index:04d}.png"
        cv2.imwrite(str(output_path), (augmented * 255).astype("uint8"))

    return len(class_images(output_dir))


def main() -> None:
    root = project_root()
    processed_root = root / "ranjana-lipi-coach" / "backend" / "ml" / "processed"
    augmented_root = root / "ranjana-lipi-coach" / "backend" / "ml" / "augmented"
    transforms_path = processed_root / "alignment_transforms.json"

    if not transforms_path.is_file():
        raise FileNotFoundError(
            f"Missing corrected alignment transforms: {transforms_path}. "
            "Run build_dataset.py before regenerating augmented images."
        )

    print(f"Processed source: {processed_root}")
    print(f"Augmented output: {augmented_root}")
    print(f"Alignment transforms: {transforms_path}")
    print(f"Target per class: {TARGET_COUNT}")
    print()

    final_counts: dict[str, int] = {}
    for class_name in SELECTED_CLASSES:
        final_counts[class_name] = augment_class(processed_root, augmented_root, class_name)

    print("Final Augmented Counts")
    for class_name in SELECTED_CLASSES:
        print(f"{class_name}: {final_counts[class_name]}")


if __name__ == "__main__":
    main()
