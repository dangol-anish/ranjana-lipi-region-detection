#!/usr/bin/env python3
"""Build reference-anchored normalized assets for every class in data/Dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

try:
    from .normalize import NormalizationError, apply_fixed_transform, compute_reference_transform
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from normalize import NormalizationError, apply_fixed_transform, compute_reference_transform


IMAGE_EXTENSIONS = {
    ".bmp",
    ".dib",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build processed_general normalized images for all 62 classes"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate class/reference discovery without writing processed_general.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def find_child_case_insensitive(parent: Path, child_name: str) -> Path:
    target = child_name.lower()
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == target:
            return child
    raise FileNotFoundError(f"Could not find {child_name!r} folder under {parent}")


def class_names(dataset_root: Path) -> list[str]:
    return sorted(path.name for path in dataset_root.iterdir() if path.is_dir())


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_reference_image(data_dir: Path, class_name: str) -> Path:
    reference_root = find_child_case_insensitive(data_dir, "Reference")
    class_dir = find_child_case_insensitive(reference_root, class_name)
    candidates = sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one reference image for {class_name}, found {len(candidates)}"
        )
    return candidates[0]


def save_normalized_png(image_path: Path, output_path: Path, transform: dict) -> None:
    normalized = apply_fixed_transform(image_path, transform)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), (normalized * 255).astype("uint8"))


def compute_class_transforms(data_dir: Path, classes: list[str]) -> dict[str, dict]:
    transforms: dict[str, dict] = {}
    for class_name in classes:
        reference_path = find_reference_image(data_dir, class_name)
        transform = compute_reference_transform(reference_path)
        transform["reference_image"] = str(reference_path)
        transforms[class_name] = transform
        bbox = transform["bbox"]
        print(
            f"[transform] {class_name}: bbox=({bbox['x']}, {bbox['y']}, "
            f"{bbox['width']}x{bbox['height']}), scale={transform['scale']:.4f}, "
            f"offset=({transform['x_offset']}, {transform['y_offset']})"
        )
    return transforms


def process_class(
    dataset_root: Path,
    data_dir: Path,
    output_root: Path,
    class_name: str,
    transform: dict,
) -> tuple[int, int]:
    class_dir = dataset_root / class_name
    output_dir = output_root / class_name
    processed = 0
    skipped = 0

    for source_path in image_files(class_dir):
        output_path = output_dir / f"{source_path.stem}.png"
        try:
            save_normalized_png(source_path, output_path, transform)
            processed += 1
        except (NormalizationError, OSError, ValueError) as exc:
            skipped += 1
            print(f"[skip] {class_name}: {source_path} -> {exc}")

    reference_path = find_reference_image(data_dir, class_name)
    reference_output = output_root / "references" / f"{class_name}.png"
    try:
        save_normalized_png(reference_path, reference_output, transform)
        print(f"[reference] {class_name}: {reference_path} -> {reference_output}")
    except (NormalizationError, OSError, ValueError) as exc:
        print(f"[skip-reference] {class_name}: {reference_path} -> {exc}")

    return processed, skipped


def main() -> None:
    args = parse_args()
    root = project_root()
    data_dir = root / "data"
    dataset_root = find_child_case_insensitive(data_dir, "Dataset")
    output_root = root / "ranjana-lipi-coach" / "backend" / "ml" / "processed_general"
    classes = class_names(dataset_root)

    if len(classes) != 62:
        raise RuntimeError(f"Expected 62 classes under {dataset_root}, found {len(classes)}")

    print(f"Source data: {data_dir}")
    print(f"Processed general output: {output_root}")
    print(f"Class count: {len(classes)}")
    print("Classes:", ", ".join(classes))
    print()

    if args.dry_run:
        for class_name in classes:
            reference_path = find_reference_image(data_dir, class_name)
            source_count = len(image_files(dataset_root / class_name))
            print(f"{class_name}: source={source_count} reference={reference_path}")
        print("Dry run complete: all 62 classes and references were found.")
        return

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    transforms = compute_class_transforms(data_dir, classes)
    transforms_path = output_root / "alignment_transforms.json"
    transforms_path.write_text(json.dumps(transforms, indent=2), encoding="utf-8")
    print(f"[transforms] saved: {transforms_path}")
    print()

    summary: dict[str, dict[str, int]] = {}
    for class_name in classes:
        processed, skipped = process_class(dataset_root, data_dir, output_root, class_name, transforms[class_name])
        summary[class_name] = {"processed": processed, "skipped": skipped}
        print(f"{class_name}: processed={processed}, skipped={skipped}")

    summary_path = output_root / "processing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
