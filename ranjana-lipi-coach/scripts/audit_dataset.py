#!/usr/bin/env python3
"""Audit the existing Ranjana Lipi dataset without modifying source data."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {
    ".bmp",
    ".dib",
    ".gif",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass
class ClassAudit:
    class_name: str
    image_count: int
    min_width: int | None
    max_width: int | None
    mean_width: float | None
    min_height: int | None
    max_height: int | None
    mean_height: float | None
    formats: str
    corrupt_count: int
    corrupt_files: str


def find_data_dir(start: Path) -> Path:
    candidates = [
        start / "data",
        start.parent / "data",
        start.parent.parent / "data",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find data/ from the current directory or nearby parents."
    )


def find_child_case_insensitive(parent: Path, child_name: str) -> Path:
    target = child_name.lower()
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == target:
            return child
    raise FileNotFoundError(f"Could not find {child_name!r} folder under {parent}")


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def read_image_size(path: Path) -> tuple[int, int, str]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.width, image.height, image.format or path.suffix.lstrip(".").upper()


def audit_class(class_dir: Path) -> ClassAudit:
    widths: list[int] = []
    heights: list[int] = []
    formats: Counter[str] = Counter()
    corrupt_files: list[str] = []
    image_files = sorted(path for path in class_dir.rglob("*") if is_image_file(path))

    for image_path in image_files:
        try:
            width, height, image_format = read_image_size(image_path)
            widths.append(width)
            heights.append(height)
            formats[image_format.upper()] += 1
        except (OSError, UnidentifiedImageError) as exc:
            corrupt_files.append(f"{image_path}: {exc}")

    return ClassAudit(
        class_name=class_dir.name,
        image_count=len(image_files),
        min_width=min(widths) if widths else None,
        max_width=max(widths) if widths else None,
        mean_width=mean(widths) if widths else None,
        min_height=min(heights) if heights else None,
        max_height=max(heights) if heights else None,
        mean_height=mean(heights) if heights else None,
        formats=";".join(f"{fmt}:{count}" for fmt, count in sorted(formats.items())),
        corrupt_count=len(corrupt_files),
        corrupt_files=" | ".join(corrupt_files),
    )


def find_reference_images(data_dir: Path) -> dict[str, list[Path]]:
    dataset_dir = find_child_case_insensitive(data_dir, "Dataset")
    class_names = {path.name for path in dataset_dir.iterdir() if path.is_dir()}
    references: dict[str, list[Path]] = {class_name: [] for class_name in sorted(class_names)}

    for image_path in sorted(path for path in data_dir.rglob("*") if is_image_file(path)):
        try:
            image_path.relative_to(dataset_dir)
            continue
        except ValueError:
            pass

        parts_lower = [part.lower() for part in image_path.parts]
        stem_lower = image_path.stem.lower()
        for class_name in class_names:
            class_lower = class_name.lower()
            if class_lower in parts_lower or stem_lower == class_lower:
                references[class_name].append(image_path)
                break

    return references


def write_dataset_csv(rows: list[ClassAudit], output_path: Path) -> None:
    fieldnames = [
        "class_name",
        "image_count",
        "min_width",
        "max_width",
        "mean_width",
        "min_height",
        "max_height",
        "mean_height",
        "formats",
        "corrupt_count",
        "corrupt_files",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "class_name": row.class_name,
                    "image_count": row.image_count,
                    "min_width": row.min_width or "",
                    "max_width": row.max_width or "",
                    "mean_width": f"{row.mean_width:.2f}" if row.mean_width else "",
                    "min_height": row.min_height or "",
                    "max_height": row.max_height or "",
                    "mean_height": f"{row.mean_height:.2f}" if row.mean_height else "",
                    "formats": row.formats,
                    "corrupt_count": row.corrupt_count,
                    "corrupt_files": row.corrupt_files,
                }
            )


def write_reference_csv(references: dict[str, list[Path]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "class_name",
                "has_reference",
                "reference_count",
                "reference_paths",
                "status",
            ],
        )
        writer.writeheader()
        for class_name, paths in sorted(references.items()):
            if len(paths) == 1:
                status = "ok"
            elif paths:
                status = "multiple_references_found"
            else:
                status = "missing_reference"
            writer.writerow(
                {
                    "class_name": class_name,
                    "has_reference": bool(paths),
                    "reference_count": len(paths),
                    "reference_paths": " | ".join(str(path) for path in paths),
                    "status": status,
                }
            )


def print_dataset_table(rows: list[ClassAudit]) -> None:
    print("\nDataset Audit Summary")
    print(
        f"{'Class':<12} {'Count':>7} {'Avg WxH':>16} {'Min WxH':>16} "
        f"{'Max WxH':>16} {'Formats':<16} {'Corrupt':>8}"
    )
    print("-" * 96)
    for row in rows:
        avg_dims = (
            f"{row.mean_width:.1f}x{row.mean_height:.1f}"
            if row.mean_width and row.mean_height
            else "n/a"
        )
        min_dims = (
            f"{row.min_width}x{row.min_height}"
            if row.min_width and row.min_height
            else "n/a"
        )
        max_dims = (
            f"{row.max_width}x{row.max_height}"
            if row.max_width and row.max_height
            else "n/a"
        )
        print(
            f"{row.class_name:<12} {row.image_count:>7} {avg_dims:>16} "
            f"{min_dims:>16} {max_dims:>16} {row.formats:<16} {row.corrupt_count:>8}"
        )


def print_top_classes(rows: list[ClassAudit]) -> None:
    print("\nTop 5 Classes by Image Count")
    for rank, row in enumerate(
        sorted(rows, key=lambda item: item.image_count, reverse=True)[:5], start=1
    ):
        print(f"{rank}. {row.class_name}: {row.image_count} images")


def print_reference_summary(references: dict[str, list[Path]]) -> None:
    found = [name for name, paths in references.items() if paths]
    missing = [name for name, paths in references.items() if not paths]
    multiple = [name for name, paths in references.items() if len(paths) > 1]

    print("\nReference Image Summary")
    print(f"Classes with a reference image: {len(found)}")
    print(f"Classes missing a reference image: {len(missing)}")
    if missing:
        print("Missing:", ", ".join(missing))
    if multiple:
        print("Classes with multiple candidate references:", ", ".join(multiple))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    data_dir = find_data_dir(script_dir)
    dataset_dir = find_child_case_insensitive(data_dir, "Dataset")

    class_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    rows = [audit_class(class_dir) for class_dir in class_dirs]

    dataset_report_path = script_dir / "dataset_audit_report.csv"
    reference_report_path = script_dir / "reference_audit_report.csv"

    write_dataset_csv(rows, dataset_report_path)
    references = find_reference_images(data_dir)
    write_reference_csv(references, reference_report_path)

    print(f"Data directory: {data_dir}")
    print(f"Dataset directory: {dataset_dir}")
    print_dataset_table(rows)
    print_top_classes(rows)
    print_reference_summary(references)
    print(f"\nSaved dataset report: {dataset_report_path}")
    print(f"Saved reference report: {reference_report_path}")


if __name__ == "__main__":
    main()
