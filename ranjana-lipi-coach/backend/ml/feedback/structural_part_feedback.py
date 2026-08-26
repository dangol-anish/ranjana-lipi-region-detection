"""Rule-assisted structural part-mask feedback for validated characters.

This layer is intentionally stricter than the statistical handwriting envelope.
It checks whether the submitted writing covers explicitly defined required
parts of the taught form. That makes it useful for coaching missing strokes,
where autoencoders and average templates can hide the absence of a part.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ml.feedback.grid_feedback import DEFAULT_COLS, DEFAULT_INK_THRESHOLD, DEFAULT_ROWS, broad_bands, grid_cells
from ml.feedback.template_feedback import dilate_mask


VALIDATED_STRUCTURAL_CLASSES = (
    "a",
    "aa",
    "ka",
    "cha",
    "ga",
    "da",
    "dda",
    "gha",
    "ta",
    "nna",
    "ma",
    "jha",
    "ja",
    "ddha",
    "ya",
)
DEFAULT_COVERAGE_TOLERANCE_PIXELS = 5
DEFAULT_MISSING_THRESHOLD = 0.45
DEFAULT_EXTRA_THRESHOLD = 0.35
DEFAULT_MAX_REGIONS = 3


@dataclass(frozen=True)
class StructuralPart:
    name: str
    label: str
    broad_region: str
    fine_region: str
    mask: np.ndarray
    min_coverage: float
    required: bool


@dataclass(frozen=True)
class StructuralPartTemplate:
    class_name: str
    parts: list[StructuralPart]
    source: str
    tolerance_pixels: int


def structural_masks_root() -> Path:
    return Path(__file__).resolve().parents[1] / "saved_models" / "structural_part_masks"


def _load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read structural part mask: {path}")
    return mask > 0


def load_structural_part_template(
    class_name: str,
    root: Path | None = None,
) -> StructuralPartTemplate:
    base = root or structural_masks_root()
    config_path = base / class_name / "parts.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing structural part config: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    parts: list[StructuralPart] = []
    for item in config["parts"]:
        parts.append(
            StructuralPart(
                name=str(item["name"]),
                label=str(item["label"]),
                broad_region=str(item["broad_region"]),
                fine_region=str(item["fine_region"]),
                mask=_load_mask(base / class_name / str(item["mask"])),
                min_coverage=float(item.get("min_coverage", 0.55)),
                required=bool(item.get("required", True)),
            )
        )

    return StructuralPartTemplate(
        class_name=class_name,
        parts=parts,
        source=str(config.get("source", "reference")),
        tolerance_pixels=int(config.get("tolerance_pixels", DEFAULT_COVERAGE_TOLERANCE_PIXELS)),
    )


def _region_to_row_col(region: str) -> tuple[int, int]:
    row_names = {"top": 0, "middle": 1, "bottom": 2}
    col_names = {"left": 0, "center": 1, "right": 2}
    row_name, col_name = region.split("-", maxsplit=1)
    return row_names[row_name], col_names[col_name]


def _part_result(part: StructuralPart, user_ink: np.ndarray, tolerance_pixels: int) -> dict[str, Any]:
    if part.mask.shape != user_ink.shape:
        raise ValueError(f"Part mask shape {part.mask.shape} does not match input shape {user_ink.shape}")

    tolerated_user = dilate_mask(user_ink, tolerance_pixels)
    part_pixels = int(np.count_nonzero(part.mask))
    covered_pixels = int(np.count_nonzero(part.mask & tolerated_user))
    coverage = covered_pixels / max(part_pixels, 1)
    missing_score = max(0.0, 1.0 - coverage)
    is_missing = coverage < part.min_coverage
    is_problem = part.required and is_missing
    row, col = _region_to_row_col(part.fine_region)
    return {
        "part": part.name,
        "label": part.label,
        "region": part.fine_region,
        "broad_region": part.broad_region,
        "row": row,
        "col": col,
        "coverage": float(coverage),
        "min_coverage": float(part.min_coverage),
        "required": bool(part.required),
        "missing_score": float(missing_score),
        "score": float(missing_score),
        "normalized_score": float(missing_score),
        "z_score": float(missing_score),
        "part_pixels": part_pixels,
        "covered_pixels": covered_pixels,
        "is_missing": bool(is_missing),
        "is_problem": bool(is_problem),
        "dominant_issue": "missing" if is_problem else "none",
        "message": (
            f"{part.label} appears incomplete"
            if is_problem
            else f"{part.label} looks present"
        ),
    }


def _empty_grid_regions(height: int, width: int, rows: int, cols: int) -> list[dict[str, Any]]:
    regions = []
    for cell in grid_cells(height, width, rows, cols):
        regions.append(
            {
                "row": cell.row,
                "col": cell.col,
                "region": cell.name,
                "label": cell.name,
                "bounds": {
                    "x_start": cell.x_start,
                    "x_end": cell.x_end,
                    "y_start": cell.y_start,
                    "y_end": cell.y_end,
                },
                "score": 0.0,
                "normalized_score": 0.0,
                "z_score": 0.0,
                "missing_score": 0.0,
                "coverage": 1.0,
                "parts": [],
                "is_problem": False,
                "dominant_issue": "none",
                "message": f"{cell.name} looks structurally acceptable",
            }
        )
    return regions


def _empty_broad_regions(height: int, width: int) -> list[dict[str, Any]]:
    regions = []
    for band in broad_bands(height, width):
        regions.append(
            {
                "band": band.index,
                "region": band.name,
                "label": band.name,
                "bounds": {
                    "x_start": band.x_start,
                    "x_end": band.x_end,
                    "y_start": band.y_start,
                    "y_end": band.y_end,
                },
                "score": 0.0,
                "normalized_score": 0.0,
                "z_score": 0.0,
                "missing_score": 0.0,
                "coverage": 1.0,
                "parts": [],
                "is_problem": False,
                "dominant_issue": "none",
                "message": f"{band.name} region looks structurally acceptable",
            }
        )
    return regions


def _merge_part_results_into_regions(
    part_results: list[dict[str, Any]],
    height: int,
    width: int,
    rows: int,
    cols: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fine_regions = _empty_grid_regions(height, width, rows, cols)
    broad_regions = _empty_broad_regions(height, width)
    fine_by_name = {region["region"]: region for region in fine_regions}
    broad_by_name = {region["region"]: region for region in broad_regions}

    for part in part_results:
        fine = fine_by_name[part["region"]]
        broad = broad_by_name[part["broad_region"]]
        for region in (fine, broad):
            region["parts"].append(part)
            if not part["required"]:
                continue
            region["score"] = max(float(region["score"]), float(part["missing_score"]))
            region["missing_score"] = float(region["score"])
            region["normalized_score"] = float(region["score"])
            region["z_score"] = float(region["score"])
            region["coverage"] = min(float(region["coverage"]), float(part["coverage"]))
            region["is_problem"] = bool(region["is_problem"] or part["is_problem"])
            if region["is_problem"]:
                region["dominant_issue"] = "missing"
                region["message"] = f"{region['region']} has an incomplete required part"

    fine_ranked = sorted(fine_regions, key=lambda item: item["score"], reverse=True)
    broad_ranked = sorted(broad_regions, key=lambda item: item["score"], reverse=True)
    return fine_ranked, broad_ranked


def _problem_regions(regions: list[dict[str, Any]], max_regions: int) -> list[dict[str, Any]]:
    return [region for region in regions if region["is_problem"] and region["score"] > 0.0][:max_regions]


def build_structural_part_feedback(
    class_name: str,
    input_image: np.ndarray,
    template: StructuralPartTemplate,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
    max_regions: int = DEFAULT_MAX_REGIONS,
    ink_threshold: float = DEFAULT_INK_THRESHOLD,
) -> dict[str, Any]:
    image = np.asarray(input_image, dtype=np.float32)
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 2:
        raise ValueError("Expected 2D normalized grayscale input")

    user_ink = image > ink_threshold
    part_results = [
        _part_result(part, user_ink, template.tolerance_pixels)
        for part in template.parts
    ]
    fine_regions, broad_regions = _merge_part_results_into_regions(part_results, image.shape[0], image.shape[1], rows, cols)
    fine_problem_regions = _problem_regions(fine_regions, max_regions)
    broad_problem_regions = _problem_regions(broad_regions, max_regions)
    required_part_results = [part for part in part_results if part["required"]]
    score_source = required_part_results if required_part_results else part_results
    missing_scores = np.asarray([part["missing_score"] for part in score_source], dtype=np.float32)
    mean_missing_score = float(missing_scores.mean()) if missing_scores.size else 0.0
    max_missing_score = float(missing_scores.max()) if missing_scores.size else 0.0
    overall_score = max(0.0, 100.0 * (1.0 - min(1.0, mean_missing_score)))

    return {
        "class_name": class_name,
        "feedback_method": "structural_part_mask",
        "grid": {"rows": rows, "cols": cols},
        "overall_score": float(overall_score),
        "mean_error": mean_missing_score,
        "std_error": float(missing_scores.std()) if missing_scores.size else 0.0,
        "max_region_error": max_missing_score,
        "required_parts_count": len(required_part_results),
        "threshold": DEFAULT_MISSING_THRESHOLD,
        "mean_z_score": mean_missing_score,
        "std_z_score": float(missing_scores.std()) if missing_scores.size else 0.0,
        "max_z_score": max_missing_score,
        "problem_regions": fine_problem_regions,
        "all_regions": fine_regions,
        "fine_grid": {
            "rows": rows,
            "cols": cols,
            "problem_regions": fine_problem_regions,
            "all_regions": fine_regions,
            "threshold": DEFAULT_MISSING_THRESHOLD,
            "scoring": "required_part_coverage",
        },
        "broad_bands": {
            "bands": ["top", "middle", "bottom"],
            "problem_regions": broad_problem_regions,
            "all_regions": broad_regions,
            "threshold": DEFAULT_MISSING_THRESHOLD,
            "scoring": "required_part_coverage",
        },
        "structural_parts": part_results,
        "threshold_settings": {
            "feedback_method": "structural_part_mask",
            "ink_threshold": ink_threshold,
            "coverage_tolerance_pixels": template.tolerance_pixels,
            "missing_threshold": DEFAULT_MISSING_THRESHOLD,
            "advisory_parts_excluded_from_score": True,
        },
    }
