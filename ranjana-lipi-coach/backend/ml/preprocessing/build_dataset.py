#!/usr/bin/env python3
"""Build normalized image assets for the selected Phase 2 characters."""

from __future__ import annotations

import sys
import json
import shutil
from pathlib import Path

import cv2

try:
    from .normalize import NormalizationError, apply_fixed_transform, compute_reference_transform
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from normalize import NormalizationError, apply_fixed_transform, compute_reference_transform


SELECTED_CLASSES = ("aa", "a", "ka", "da", "dda")
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


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def find_child_case_insensitive(parent: Path, child_name: str) -> Path:
    target = child_name.lower()
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == target:
            return child
    raise FileNotFoundError(f"Could not find {child_name!r} folder under {parent}")


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def save_normalized_png(image_path: Path, output_path: Path, transform: dict) -> None:
    normalized = apply_fixed_transform(image_path, transform)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), (normalized * 255).astype("uint8"))


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


def process_class(
    data_dir: Path,
    output_root: Path,
    class_name: str,
    transform: dict,
) -> tuple[int, int]:
    dataset_dir = find_child_case_insensitive(data_dir, "Dataset")
    class_dir = find_child_case_insensitive(dataset_dir, class_name)
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


def compute_class_transforms(data_dir: Path) -> dict[str, dict]:
    transforms: dict[str, dict] = {}
    for class_name in SELECTED_CLASSES:
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


def main() -> None:
    root = project_root()
    data_dir = root / "data"
    output_root = root / "ranjana-lipi-coach" / "backend" / "ml" / "processed"

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    print(f"Source data: {data_dir}")
    print(f"Processed output: {output_root}")
    print("Selected classes:", ", ".join(SELECTED_CLASSES))
    print()

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    transforms = compute_class_transforms(data_dir)
    transforms_path = output_root / "alignment_transforms.json"
    transforms_path.write_text(json.dumps(transforms, indent=2), encoding="utf-8")
    print(f"[transforms] saved: {transforms_path}")
    print()

    for class_name in SELECTED_CLASSES:
        processed, skipped = process_class(data_dir, output_root, class_name, transforms[class_name])
        print(f"{class_name}: processed={processed}, skipped={skipped}")


if __name__ == "__main__":
    main()
