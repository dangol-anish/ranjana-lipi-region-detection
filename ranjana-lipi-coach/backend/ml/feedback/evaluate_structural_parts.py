#!/usr/bin/env python3
"""Evaluate structural-part feedback on good and flawed validated samples."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from ml.feedback.structural_part_feedback import VALIDATED_STRUCTURAL_CLASSES  # noqa: E402
from ml.inference.pipeline import analyze_attempt, load_validated_structural_part_template  # noqa: E402


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate structural-part feedback")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "Final_demo_images"
        / "16_structural_part_strict_calibrated_validation"
        / "structural_part_validation_5_strict_calibrated.csv",
    )
    parser.add_argument("--good-samples-per-class", type=int, default=5)
    return parser.parse_args()


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def intended_region_from_filename(path: Path) -> str:
    stem = path.stem.lower()
    if any(token in stem for token in ("center", "middle")):
        return "middle"
    if "top" in stem:
        return "top"
    if any(token in stem for token in ("bottom", "lower")):
        return "bottom"
    return ""


def summarize_feedback(
    sample_type: str,
    class_name: str,
    path: Path,
    intended_region: str = "",
) -> dict[str, Any]:
    try:
        result = analyze_attempt(path.read_bytes(), class_name)
        feedback = result["feedback"]
        error = ""
    except Exception as exc:
        feedback = {}
        error = str(exc)

    recognizer = feedback.get("recognizer", {})
    broad_regions = feedback.get("broad_bands", {}).get("all_regions", [])
    fine_regions = feedback.get("fine_grid", {}).get("all_regions", [])
    broad_problems = feedback.get("broad_bands", {}).get("problem_regions", [])
    fine_problems = feedback.get("fine_grid", {}).get("problem_regions", [])
    structural_parts = feedback.get("structural_parts", [])

    return {
        "type": sample_type,
        "class": class_name,
        "filename": path.name,
        "path": str(path),
        "intended_region": intended_region,
        "feedback_method": feedback.get("feedback_method", ""),
        "overall_score": feedback.get("overall_score", ""),
        "wrong_character": bool(feedback.get("wrong_character", False)),
        "predicted_class": recognizer.get("predicted_class", ""),
        "recognizer_confidence": recognizer.get("confidence", ""),
        "matches_target": recognizer.get("matches_target", ""),
        "broad_top_1": broad_regions[0]["region"] if broad_regions else "",
        "broad_top_1_score": broad_regions[0].get("score", "") if broad_regions else "",
        "broad_problem_regions": ";".join(region["region"] for region in broad_problems),
        "fine_top_1": fine_regions[0]["region"] if fine_regions else "",
        "fine_top_1_score": fine_regions[0].get("score", "") if fine_regions else "",
        "fine_problem_regions": ";".join(region["region"] for region in fine_problems),
        "parts": ";".join(
            f"{part['part']}:{part['coverage']:.3f}:{part['min_coverage']:.3f}:{part['is_problem']}"
            for part in structural_parts
        ),
        "error": error,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    load_validated_structural_part_template.cache_clear()
    rows: list[dict[str, Any]] = []

    for class_name in VALIDATED_STRUCTURAL_CLASSES:
        good_dir = args.data_root / "Dataset" / class_name
        for path in image_files(good_dir)[: args.good_samples_per_class]:
            rows.append(summarize_feedback("good_sample", class_name, path))

        flawed_dir = args.data_root / "FlawedValidation" / class_name
        for path in image_files(flawed_dir):
            rows.append(
                summarize_feedback(
                    "flawed_sample",
                    class_name,
                    path,
                    intended_region=intended_region_from_filename(path),
                )
            )

    if not rows:
        raise RuntimeError("No samples found for structural-part evaluation")

    write_csv(args.output, rows)

    good_rows = [row for row in rows if row["type"] == "good_sample"]
    flawed_rows = [row for row in rows if row["type"] == "flawed_sample"]
    good_flagged = sum(
        bool(row["broad_problem_regions"]) and not row["wrong_character"]
        for row in good_rows
    )
    flawed_matched_or_blocked = sum(
        (
            bool(row["intended_region"])
            and row["intended_region"] == row["broad_top_1"]
            and bool(row["broad_problem_regions"])
        )
        or row["wrong_character"]
        for row in flawed_rows
    )

    print(f"Saved: {args.output}")
    print(f"Good samples flagged: {good_flagged}/{len(good_rows)}")
    print(
        "Flawed samples intended broad #1 or blocked: "
        f"{flawed_matched_or_blocked}/{len(flawed_rows)}"
    )


if __name__ == "__main__":
    main()
