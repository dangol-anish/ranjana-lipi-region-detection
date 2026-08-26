"""Statistical stroke-template feedback for handwriting structure.

The template is learned from many normalized correct samples, so the app does
not demand pixel-perfect matching to one reference image. It separates:

- required stroke zones: places where correct writers often place ink
- allowed variation zones: places where some correct writers place ink
- unlikely zones: places outside the learned handwriting envelope

At inference time, missing strokes are required-zone pixels not covered by
nearby user ink, and extra strokes are user-ink pixels outside the allowed
variation zone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import json

from ml.feedback.grid_feedback import (
    DEFAULT_COLS,
    DEFAULT_INK_THRESHOLD,
    DEFAULT_ROWS,
    broad_bands,
    grid_cells,
)


DEFAULT_REQUIRED_THRESHOLD = 0.42
DEFAULT_ALLOWED_THRESHOLD = 0.04
DEFAULT_USER_TOLERANCE_PIXELS = 5
DEFAULT_ALLOWED_TOLERANCE_PIXELS = 7
DEFAULT_MISSING_WEIGHT = 0.75
DEFAULT_EXTRA_WEIGHT = 0.25
DEFAULT_FINE_PROBLEM_THRESHOLD = 0.10
DEFAULT_BROAD_PROBLEM_THRESHOLD = 0.10
DEFAULT_MIN_REQUIRED_PIXELS = 8
DEFAULT_MIN_TEMPLATE_REQUIRED_PIXELS = 120


@dataclass(frozen=True)
class StrokeTemplate:
    class_name: str
    mean_ink_map: np.ndarray
    required_mask: np.ndarray
    allowed_mask: np.ndarray
    source_count: int
    required_threshold: float
    allowed_threshold: float
    user_tolerance_pixels: int
    allowed_tolerance_pixels: int
    fine_baseline: dict[str, float] | None = None
    broad_baseline: dict[str, float] | None = None


def _odd_kernel_size(radius: int) -> int:
    return max(1, int(radius) * 2 + 1)


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    bool_mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return bool_mask
    kernel_size = _odd_kernel_size(radius)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(bool_mask.astype(np.uint8), kernel, iterations=1)
    return dilated.astype(bool)


def build_template_from_images(
    class_name: str,
    images: list[np.ndarray],
    required_threshold: float = DEFAULT_REQUIRED_THRESHOLD,
    allowed_threshold: float = DEFAULT_ALLOWED_THRESHOLD,
    allowed_tolerance_pixels: int = DEFAULT_ALLOWED_TOLERANCE_PIXELS,
    user_tolerance_pixels: int = DEFAULT_USER_TOLERANCE_PIXELS,
    ink_threshold: float = DEFAULT_INK_THRESHOLD,
) -> StrokeTemplate:
    if not images:
        raise ValueError(f"No images supplied for template: {class_name}")

    masks = []
    for image in images:
        image_array = np.asarray(image, dtype=np.float32)
        if image_array.ndim != 2:
            raise ValueError("Template images must be 2D normalized grayscale arrays")
        masks.append(image_array > ink_threshold)

    stack = np.stack(masks, axis=0).astype(np.float32)
    mean_ink_map = stack.mean(axis=0)
    required_mask = mean_ink_map >= required_threshold
    if int(np.count_nonzero(required_mask)) < DEFAULT_MIN_TEMPLATE_REQUIRED_PIXELS:
        positive_values = mean_ink_map[mean_ink_map > allowed_threshold]
        if positive_values.size:
            adaptive_threshold = max(
                allowed_threshold,
                min(required_threshold, float(np.percentile(positive_values, 85))),
            )
            required_mask = mean_ink_map >= adaptive_threshold
    allowed_base = mean_ink_map >= allowed_threshold
    allowed_mask = dilate_mask(allowed_base, allowed_tolerance_pixels)

    return StrokeTemplate(
        class_name=class_name,
        mean_ink_map=mean_ink_map.astype(np.float32),
        required_mask=required_mask.astype(bool),
        allowed_mask=allowed_mask.astype(bool),
        source_count=len(images),
        required_threshold=required_threshold,
        allowed_threshold=allowed_threshold,
        user_tolerance_pixels=user_tolerance_pixels,
        allowed_tolerance_pixels=allowed_tolerance_pixels,
    )


def save_template(template: StrokeTemplate, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        class_name=np.asarray(template.class_name),
        mean_ink_map=template.mean_ink_map.astype(np.float32),
        required_mask=template.required_mask.astype(np.uint8),
        allowed_mask=template.allowed_mask.astype(np.uint8),
        source_count=np.asarray(template.source_count, dtype=np.int32),
        required_threshold=np.asarray(template.required_threshold, dtype=np.float32),
        allowed_threshold=np.asarray(template.allowed_threshold, dtype=np.float32),
        user_tolerance_pixels=np.asarray(template.user_tolerance_pixels, dtype=np.int32),
        allowed_tolerance_pixels=np.asarray(template.allowed_tolerance_pixels, dtype=np.int32),
        fine_baseline=np.asarray(json.dumps(template.fine_baseline or {})),
        broad_baseline=np.asarray(json.dumps(template.broad_baseline or {})),
    )


def load_template(template_path: Path) -> StrokeTemplate:
    if not template_path.is_file():
        raise FileNotFoundError(f"Missing stroke template: {template_path}")

    with np.load(template_path, allow_pickle=False) as data:
        return StrokeTemplate(
            class_name=str(data["class_name"].item()),
            mean_ink_map=np.asarray(data["mean_ink_map"], dtype=np.float32),
            required_mask=np.asarray(data["required_mask"], dtype=np.uint8).astype(bool),
            allowed_mask=np.asarray(data["allowed_mask"], dtype=np.uint8).astype(bool),
            source_count=int(data["source_count"].item()),
            required_threshold=float(data["required_threshold"].item()),
            allowed_threshold=float(data["allowed_threshold"].item()),
            user_tolerance_pixels=int(data["user_tolerance_pixels"].item()),
            allowed_tolerance_pixels=int(data["allowed_tolerance_pixels"].item()),
            fine_baseline=json.loads(str(data["fine_baseline"].item())) if "fine_baseline" in data else {},
            broad_baseline=json.loads(str(data["broad_baseline"].item())) if "broad_baseline" in data else {},
        )


def with_baselines(
    template: StrokeTemplate,
    fine_baseline: dict[str, float],
    broad_baseline: dict[str, float],
) -> StrokeTemplate:
    return StrokeTemplate(
        class_name=template.class_name,
        mean_ink_map=template.mean_ink_map,
        required_mask=template.required_mask,
        allowed_mask=template.allowed_mask,
        source_count=template.source_count,
        required_threshold=template.required_threshold,
        allowed_threshold=template.allowed_threshold,
        user_tolerance_pixels=template.user_tolerance_pixels,
        allowed_tolerance_pixels=template.allowed_tolerance_pixels,
        fine_baseline=fine_baseline,
        broad_baseline=broad_baseline,
    )


def structural_error_maps(
    input_image: np.ndarray,
    template: StrokeTemplate,
    ink_threshold: float = DEFAULT_INK_THRESHOLD,
) -> dict[str, np.ndarray]:
    user = np.asarray(input_image, dtype=np.float32)
    if user.ndim == 3 and user.shape[0] == 1:
        user = user[0]
    if user.shape != template.mean_ink_map.shape:
        raise ValueError(f"Input shape {user.shape} does not match template {template.mean_ink_map.shape}")
    if user.ndim != 2:
        raise ValueError("Expected a 2D normalized grayscale input image")

    user_ink = user > ink_threshold
    user_with_tolerance = dilate_mask(user_ink, template.user_tolerance_pixels)
    missing_mask = template.required_mask & ~user_with_tolerance
    missing_weight_map = template.mean_ink_map * (~user_with_tolerance).astype(np.float32)
    extra_mask = user_ink & ~template.allowed_mask
    structural_error = np.zeros_like(user, dtype=np.float32)
    structural_error += DEFAULT_MISSING_WEIGHT * missing_weight_map
    structural_error[extra_mask] += DEFAULT_EXTRA_WEIGHT
    return {
        "user_ink": user_ink,
        "missing_mask": missing_mask,
        "missing_weight_map": missing_weight_map,
        "extra_mask": extra_mask,
        "structural_error": structural_error,
    }


def _region_stats(
    region_name: str,
    bounds: dict[str, int],
    missing_region: np.ndarray,
    missing_weight_region: np.ndarray,
    extra_region: np.ndarray,
    required_region: np.ndarray,
    expected_region: np.ndarray,
    user_region: np.ndarray,
    missing_weight: float,
    extra_weight: float,
    min_required_pixels: int,
) -> dict[str, Any]:
    required_pixels = int(np.count_nonzero(required_region))
    user_pixels = int(np.count_nonzero(user_region))
    missing_pixels = int(np.count_nonzero(missing_region))
    extra_pixels = int(np.count_nonzero(extra_region))
    expected_weight = float(np.asarray(expected_region, dtype=np.float32).sum())
    missing_weighted_pixels = float(np.asarray(missing_weight_region, dtype=np.float32).sum())
    missing_ratio = missing_weighted_pixels / max(expected_weight, 1e-6)
    extra_ratio = extra_pixels / max(user_pixels, 1)
    combined_score = missing_weight * missing_ratio + extra_weight * extra_ratio
    insufficient_data = required_pixels < min_required_pixels and user_pixels < min_required_pixels
    dominant_issue = "missing" if missing_ratio >= extra_ratio else "extra"
    if missing_pixels == 0 and extra_pixels == 0:
        dominant_issue = "none"

    return {
        "region": region_name,
        "label": region_name,
        "bounds": bounds,
        "required_pixels": required_pixels,
        "user_ink_pixels": user_pixels,
        "missing_pixels": missing_pixels,
        "missing_weighted_pixels": missing_weighted_pixels,
        "expected_weight": expected_weight,
        "extra_pixels": extra_pixels,
        "missing_ratio": float(missing_ratio),
        "extra_ratio": float(extra_ratio),
        "score": float(combined_score),
        "normalized_score": float(combined_score),
        "mean_error": float(combined_score),
        "z_score": float(combined_score),
        "dominant_issue": dominant_issue,
        "insufficient_data": bool(insufficient_data),
    }


def _message_for_region(region: dict[str, Any]) -> str:
    name = str(region["region"])
    dominant_issue = region["dominant_issue"]
    missing_ratio = float(region["missing_ratio"])
    extra_ratio = float(region["extra_ratio"])
    if dominant_issue == "missing":
        return f"{name} appears to be missing required stroke structure"
    if dominant_issue == "extra":
        return f"{name} has ink outside the usual writing envelope"
    if missing_ratio > 0 or extra_ratio > 0:
        return f"{name} has minor structural variation"
    return f"{name} looks structurally acceptable"


def _rank_regions(
    regions: list[dict[str, Any]],
    problem_threshold: float,
    max_regions: int,
    baseline: dict[str, float] | None = None,
) -> dict[str, Any]:
    for region in regions:
        baseline_score = float((baseline or {}).get(str(region["region"]), 0.0))
        adjusted_score = max(0.0, float(region["score"]) - baseline_score)
        region["baseline_score"] = baseline_score
        region["adjusted_score"] = adjusted_score
        region["normalized_score"] = adjusted_score
        region["z_score"] = adjusted_score

    ranked = sorted(regions, key=lambda item: item["adjusted_score"], reverse=True)
    max_score = float(ranked[0]["adjusted_score"]) if ranked else 0.0
    problem_regions: list[dict[str, Any]] = []
    for region in ranked:
        is_problem = (
            not region.get("insufficient_data")
            and float(region["adjusted_score"]) >= problem_threshold
            and float(region["adjusted_score"]) > 0.0
        )
        region["is_problem"] = bool(is_problem)
        region["message"] = _message_for_region(region)
        if is_problem and len(problem_regions) < max_regions:
            problem_regions.append(region)

    scores = np.asarray([float(region["adjusted_score"]) for region in ranked], dtype=np.float32)
    return {
        "all_regions": ranked,
        "problem_regions": problem_regions,
        "mean_error": float(scores.mean()) if scores.size else 0.0,
        "std_error": float(scores.std()) if scores.size else 0.0,
        "max_region_error": max_score,
        "threshold": float(problem_threshold),
    }


def aggregate_template_regions(
    maps: dict[str, np.ndarray],
    template: StrokeTemplate,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
    missing_weight: float = DEFAULT_MISSING_WEIGHT,
    extra_weight: float = DEFAULT_EXTRA_WEIGHT,
    min_required_pixels: int = DEFAULT_MIN_REQUIRED_PIXELS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    missing = maps["missing_mask"]
    missing_weight_map = maps["missing_weight_map"]
    extra = maps["extra_mask"]
    user_ink = maps["user_ink"]
    height, width = missing.shape

    fine: list[dict[str, Any]] = []
    for cell in grid_cells(height, width, rows, cols):
        bounds = {
            "x_start": cell.x_start,
            "x_end": cell.x_end,
            "y_start": cell.y_start,
            "y_end": cell.y_end,
        }
        region_slice = np.s_[cell.y_start : cell.y_end, cell.x_start : cell.x_end]
        stats = _region_stats(
            region_name=cell.name,
            bounds=bounds,
            missing_region=missing[region_slice],
            missing_weight_region=missing_weight_map[region_slice],
            extra_region=extra[region_slice],
            required_region=template.required_mask[region_slice],
            expected_region=template.mean_ink_map[region_slice],
            user_region=user_ink[region_slice],
            missing_weight=missing_weight,
            extra_weight=extra_weight,
            min_required_pixels=min_required_pixels,
        )
        stats["row"] = cell.row
        stats["col"] = cell.col
        fine.append(stats)

    broad: list[dict[str, Any]] = []
    for band in broad_bands(height, width):
        bounds = {
            "x_start": band.x_start,
            "x_end": band.x_end,
            "y_start": band.y_start,
            "y_end": band.y_end,
        }
        region_slice = np.s_[band.y_start : band.y_end, band.x_start : band.x_end]
        stats = _region_stats(
            region_name=band.name,
            bounds=bounds,
            missing_region=missing[region_slice],
            missing_weight_region=missing_weight_map[region_slice],
            extra_region=extra[region_slice],
            required_region=template.required_mask[region_slice],
            expected_region=template.mean_ink_map[region_slice],
            user_region=user_ink[region_slice],
            missing_weight=missing_weight,
            extra_weight=extra_weight,
            min_required_pixels=min_required_pixels,
        )
        stats["band"] = band.index
        broad.append(stats)

    return fine, broad


def build_template_feedback(
    class_name: str,
    input_image: np.ndarray,
    template: StrokeTemplate,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
    max_regions: int = 3,
    fine_problem_threshold: float = DEFAULT_FINE_PROBLEM_THRESHOLD,
    broad_problem_threshold: float = DEFAULT_BROAD_PROBLEM_THRESHOLD,
    missing_weight: float = DEFAULT_MISSING_WEIGHT,
    extra_weight: float = DEFAULT_EXTRA_WEIGHT,
    min_required_pixels: int = DEFAULT_MIN_REQUIRED_PIXELS,
    ink_threshold: float = DEFAULT_INK_THRESHOLD,
) -> dict[str, Any]:
    maps = structural_error_maps(input_image, template, ink_threshold=ink_threshold)
    fine_regions, broad_regions = aggregate_template_regions(
        maps=maps,
        template=template,
        rows=rows,
        cols=cols,
        missing_weight=missing_weight,
        extra_weight=extra_weight,
        min_required_pixels=min_required_pixels,
    )
    fine_feedback = _rank_regions(fine_regions, fine_problem_threshold, max_regions, template.fine_baseline)
    broad_feedback = _rank_regions(broad_regions, broad_problem_threshold, max_regions, template.broad_baseline)

    required_total = int(np.count_nonzero(template.required_mask))
    user_total = int(np.count_nonzero(maps["user_ink"]))
    missing_total = int(np.count_nonzero(maps["missing_mask"]))
    extra_total = int(np.count_nonzero(maps["extra_mask"]))
    expected_total = float(template.mean_ink_map.sum())
    missing_weighted_total = float(maps["missing_weight_map"].sum())
    missing_ratio = missing_weighted_total / max(expected_total, 1e-6)
    extra_ratio = extra_total / max(user_total, 1)
    global_structural_error = missing_weight * missing_ratio + extra_weight * extra_ratio
    overall_score = max(0.0, 100.0 * (1.0 - min(1.0, global_structural_error)))

    return {
        "class_name": class_name,
        "feedback_method": "statistical_template",
        "grid": {"rows": rows, "cols": cols},
        "overall_score": float(overall_score),
        "mean_error": fine_feedback["mean_error"],
        "std_error": fine_feedback["std_error"],
        "max_region_error": fine_feedback["max_region_error"],
        "threshold": fine_feedback["threshold"],
        "mean_z_score": fine_feedback["mean_error"],
        "std_z_score": fine_feedback["std_error"],
        "max_z_score": fine_feedback["max_region_error"],
        "problem_regions": fine_feedback["problem_regions"],
        "all_regions": fine_feedback["all_regions"],
        "fine_grid": {
            "rows": rows,
            "cols": cols,
            "problem_regions": fine_feedback["problem_regions"],
            "all_regions": fine_feedback["all_regions"],
            "threshold": fine_feedback["threshold"],
            "scoring": "structural_template_error",
        },
        "broad_bands": {
            "bands": ["top", "middle", "bottom"],
            "problem_regions": broad_feedback["problem_regions"],
            "all_regions": broad_feedback["all_regions"],
            "threshold": broad_feedback["threshold"],
            "scoring": "structural_template_error",
        },
        "template_stats": {
            "source_count": template.source_count,
            "required_threshold": template.required_threshold,
            "allowed_threshold": template.allowed_threshold,
            "user_tolerance_pixels": template.user_tolerance_pixels,
            "allowed_tolerance_pixels": template.allowed_tolerance_pixels,
            "fine_baseline_regions": len(template.fine_baseline or {}),
            "broad_baseline_regions": len(template.broad_baseline or {}),
            "required_pixels": required_total,
            "user_ink_pixels": user_total,
            "missing_pixels": missing_total,
            "missing_weighted_pixels": missing_weighted_total,
            "expected_weight": expected_total,
            "extra_pixels": extra_total,
            "missing_ratio": float(missing_ratio),
            "extra_ratio": float(extra_ratio),
            "structural_error": float(global_structural_error),
        },
        "threshold_settings": {
            "feedback_method": "statistical_template",
            "fine_problem_threshold": fine_problem_threshold,
            "broad_problem_threshold": broad_problem_threshold,
            "missing_weight": missing_weight,
            "extra_weight": extra_weight,
            "ink_threshold": ink_threshold,
            "min_required_pixels": min_required_pixels,
        },
    }
