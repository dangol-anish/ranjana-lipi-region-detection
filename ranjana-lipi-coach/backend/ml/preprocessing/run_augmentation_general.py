#!/usr/bin/env python3
"""Generate augmented normalized samples for every processed_general class."""

from __future__ import annotations

import argparse
import json
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
LOW_SOURCE_WARNING_THRESHOLD = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate augmented_general images for all processed_general classes"
    )
    parser.add_argument("--target-count", type=int, default=TARGET_COUNT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate processed_general class counts without writing augmented_general.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def class_images(class_dir: Path) -> list[Path]:
    return sorted(path for path in class_dir.glob("*.png") if path.is_file())


def reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def class_names(processed_root: Path) -> list[str]:
    return sorted(
        path.name
        for path in processed_root.iterdir()
        if path.is_dir() and path.name != "references"
    )


def augment_class(
    processed_root: Path,
    augmented_root: Path,
    class_name: str,
    target_count: int,
) -> dict[str, int | bool]:
    source_dir = processed_root / class_name
    output_dir = augmented_root / class_name
    source_images = class_images(source_dir)

    if not source_images:
        raise FileNotFoundError(f"No normalized PNG images found in {source_dir}")

    reset_output_dir(output_dir)

    for index, source_path in enumerate(cycle(source_images), start=1):
        if index > target_count:
            break

        image = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[skip] Could not read normalized image: {source_path}")
            continue

        augmented = augment_image(image.astype("float32") / 255.0)
        output_path = output_dir / f"{class_name}_{index:04d}.png"
        cv2.imwrite(str(output_path), (augmented * 255).astype("uint8"))

    return {
        "source_count": len(source_images),
        "augmented_count": len(class_images(output_dir)),
        "heavy_reuse": len(source_images) < LOW_SOURCE_WARNING_THRESHOLD,
    }


def main() -> None:
    args = parse_args()
    root = project_root()
    processed_root = root / "ranjana-lipi-coach" / "backend" / "ml" / "processed_general"
    augmented_root = root / "ranjana-lipi-coach" / "backend" / "ml" / "augmented_general"
    transforms_path = processed_root / "alignment_transforms.json"

    if not transforms_path.is_file():
        raise FileNotFoundError(
            f"Missing general alignment transforms: {transforms_path}. "
            "Run build_dataset_general.py before augmentation."
        )

    classes = class_names(processed_root)
    if len(classes) != 62:
        raise RuntimeError(f"Expected 62 processed_general classes, found {len(classes)}")

    print(f"Processed general source: {processed_root}")
    print(f"Augmented general output: {augmented_root}")
    print(f"Alignment transforms: {transforms_path}")
    print(f"Target per class: {args.target_count}")
    print(f"Class count: {len(classes)}")
    print()

    if args.dry_run:
        for class_name in classes:
            source_count = len(class_images(processed_root / class_name))
            reuse_note = " (heavy reuse expected)" if source_count < LOW_SOURCE_WARNING_THRESHOLD else ""
            print(f"{class_name}: source={source_count}{reuse_note}")
        print("Dry run complete: processed_general is ready for augmentation.")
        return

    summary: dict[str, dict[str, int | bool]] = {}
    for class_name in classes:
        result = augment_class(processed_root, augmented_root, class_name, args.target_count)
        summary[class_name] = result
        reuse_note = " (heavy reuse expected)" if result["heavy_reuse"] else ""
        print(
            f"{class_name}: source={result['source_count']} "
            f"augmented={result['augmented_count']}{reuse_note}"
        )

    summary_path = augmented_root / "augmentation_summary.json"
    augmented_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
