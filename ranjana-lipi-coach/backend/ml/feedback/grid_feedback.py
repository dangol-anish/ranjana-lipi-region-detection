"""Fixed-grid aggregation for autoencoder reconstruction error feedback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROWS = 3
DEFAULT_COLS = 3
DEFAULT_STD_MULTIPLIER = 1.25
DEFAULT_MIN_RELATIVE_TO_MAX = 0.55
# Tuned so held-out correct validation samples do not trigger false regional
# warnings; deliberate flawed-sample validation should refine this threshold.
DEFAULT_MIN_PROBLEM_REGION_ERROR = 0.02
Z_SCORE_EPSILON = 1e-6
DEFAULT_INK_THRESHOLD = 0.05
DEFAULT_MIN_INK_PIXELS = 8


@dataclass(frozen=True)
class GridCell:
    row: int
    col: int
    name: str
    y_start: int
    y_end: int
    x_start: int
    x_end: int


@dataclass(frozen=True)
class BroadBand:
    index: int
    name: str
    y_start: int
    y_end: int
    x_start: int
    x_end: int


def position_name(row: int, col: int, rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS) -> str:
    if rows != 3 or cols != 3:
        return f"row-{row + 1}-col-{col + 1}"

    row_names = ("top", "middle", "bottom")
    col_names = ("left", "center", "right")
    return f"{row_names[row]}-{col_names[col]}"


def grid_cells(height: int, width: int, rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS) -> list[GridCell]:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    cells: list[GridCell] = []
    for row in range(rows):
        y_start = round(row * height / rows)
        y_end = round((row + 1) * height / rows)
        for col in range(cols):
            x_start = round(col * width / cols)
            x_end = round((col + 1) * width / cols)
            cells.append(
                GridCell(
                    row=row,
                    col=col,
                    name=position_name(row, col, rows, cols),
                    y_start=y_start,
                    y_end=y_end,
                    x_start=x_start,
                    x_end=x_end,
                )
            )
    return cells


def broad_bands(height: int, width: int) -> list[BroadBand]:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")

    names = ("top", "middle", "bottom")
    bands: list[BroadBand] = []
    for index, name in enumerate(names):
        y_start = round(index * height / len(names))
        y_end = round((index + 1) * height / len(names))
        bands.append(
            BroadBand(
                index=index,
                name=name,
                y_start=y_start,
                y_end=y_end,
                x_start=0,
                x_end=width,
            )
        )
    return bands


def reconstruction_error_map(input_image: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
    input_array = np.asarray(input_image, dtype=np.float32)
    reconstruction_array = np.asarray(reconstruction, dtype=np.float32)

    if input_array.shape != reconstruction_array.shape:
        raise ValueError(
            f"input and reconstruction must have same shape, got "
            f"{input_array.shape} and {reconstruction_array.shape}"
        )
    if input_array.ndim == 3 and input_array.shape[0] == 1:
        input_array = input_array[0]
        reconstruction_array = reconstruction_array[0]
    if input_array.ndim != 2:
        raise ValueError("expected single-channel 2D images or 1xHxW arrays")

    return np.square(input_array - reconstruction_array)


def ink_relevance_mask(
    input_image: np.ndarray,
    reconstruction: np.ndarray,
    ink_threshold: float = DEFAULT_INK_THRESHOLD,
) -> np.ndarray:
    input_array = np.asarray(input_image, dtype=np.float32)
    reconstruction_array = np.asarray(reconstruction, dtype=np.float32)

    if input_array.ndim == 3 and input_array.shape[0] == 1:
        input_array = input_array[0]
        reconstruction_array = reconstruction_array[0]
    if input_array.shape != reconstruction_array.shape:
        raise ValueError(
            f"input and reconstruction must have same shape, got "
            f"{input_array.shape} and {reconstruction_array.shape}"
        )
    if input_array.ndim != 2:
        raise ValueError("expected single-channel 2D images or 1xHxW arrays")

    return (input_array > ink_threshold) | (reconstruction_array > ink_threshold)


def _masked_region_stats(
    error_region: np.ndarray,
    mask_region: np.ndarray | None,
    min_ink_pixels: int = DEFAULT_MIN_INK_PIXELS,
) -> dict[str, Any]:
    if error_region.size == 0:
        return {
            "mean_error": 0.0,
            "max_error": 0.0,
            "ink_pixel_count": 0,
            "total_pixel_count": 0,
            "insufficient_data": True,
        }

    if mask_region is None:
        values = error_region.reshape(-1)
        return {
            "mean_error": float(values.mean()),
            "max_error": float(values.max()),
            "ink_pixel_count": int(values.size),
            "total_pixel_count": int(values.size),
            "insufficient_data": False,
        }

    relevant_values = error_region[mask_region]
    ink_pixel_count = int(relevant_values.size)
    insufficient_data = ink_pixel_count < min_ink_pixels
    return {
        "mean_error": float(relevant_values.mean()) if ink_pixel_count else 0.0,
        "max_error": float(relevant_values.max()) if ink_pixel_count else 0.0,
        "ink_pixel_count": ink_pixel_count,
        "total_pixel_count": int(error_region.size),
        "insufficient_data": bool(insufficient_data),
    }


def aggregate_grid_errors(
    error_map: np.ndarray,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
    ink_mask: np.ndarray | None = None,
    min_ink_pixels: int = DEFAULT_MIN_INK_PIXELS,
) -> list[dict[str, Any]]:
    error_array = np.asarray(error_map, dtype=np.float32)
    if error_array.ndim != 2:
        raise ValueError("error_map must be a 2D array")
    if ink_mask is not None and ink_mask.shape != error_array.shape:
        raise ValueError("ink_mask must have the same shape as error_map")

    height, width = error_array.shape
    results: list[dict[str, Any]] = []
    for cell in grid_cells(height, width, rows, cols):
        region = error_array[cell.y_start : cell.y_end, cell.x_start : cell.x_end]
        mask_region = (
            ink_mask[cell.y_start : cell.y_end, cell.x_start : cell.x_end]
            if ink_mask is not None
            else None
        )
        stats = _masked_region_stats(region, mask_region, min_ink_pixels)
        results.append(
            {
                "row": cell.row,
                "col": cell.col,
                "region": cell.name,
                "bounds": {
                    "x_start": cell.x_start,
                    "x_end": cell.x_end,
                    "y_start": cell.y_start,
                    "y_end": cell.y_end,
                },
                **stats,
            }
        )
    return results


def aggregate_broad_band_errors(
    error_map: np.ndarray,
    ink_mask: np.ndarray | None = None,
    min_ink_pixels: int = DEFAULT_MIN_INK_PIXELS,
) -> list[dict[str, Any]]:
    """Aggregate reconstruction error into equal top/middle/bottom bands."""

    error_array = np.asarray(error_map, dtype=np.float32)
    if error_array.ndim != 2:
        raise ValueError("error_map must be a 2D array")
    if ink_mask is not None and ink_mask.shape != error_array.shape:
        raise ValueError("ink_mask must have the same shape as error_map")

    height, width = error_array.shape
    results: list[dict[str, Any]] = []
    for band in broad_bands(height, width):
        region = error_array[band.y_start : band.y_end, band.x_start : band.x_end]
        mask_region = (
            ink_mask[band.y_start : band.y_end, band.x_start : band.x_end]
            if ink_mask is not None
            else None
        )
        stats = _masked_region_stats(region, mask_region, min_ink_pixels)
        results.append(
            {
                "band": band.index,
                "region": band.name,
                "bounds": {
                    "x_start": band.x_start,
                    "x_end": band.x_end,
                    "y_start": band.y_start,
                    "y_end": band.y_end,
                },
                **stats,
            }
        )
    return results


def load_feedback_messages(config_path: Path | None = None) -> dict[str, dict[str, str]]:
    path = config_path or Path(__file__).with_name("feedback_messages.json")
    with path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def message_for_region(
    class_name: str,
    region_name: str,
    messages: dict[str, dict[str, str]],
) -> str:
    return (
        messages.get(class_name, {}).get(region_name)
        or messages.get("default", {}).get(region_name)
        or f"{region_name} needs improvement"
    )


def default_saved_models_root() -> Path:
    return Path(__file__).resolve().parents[1] / "saved_models"


def load_region_baseline(
    class_name: str,
    saved_models_root: Path | None = None,
) -> dict[str, Any] | None:
    baseline_path = (saved_models_root or default_saved_models_root()) / f"region_baseline_{class_name}.json"
    if not baseline_path.is_file():
        return None

    with baseline_path.open("r", encoding="utf-8") as baseline_file:
        return json.load(baseline_file)


def _baseline_for_region(
    baseline_regions: dict[str, dict[str, Any]] | None,
    region_name: str,
) -> tuple[float | None, float | None]:
    if not baseline_regions or region_name not in baseline_regions:
        return None, None

    region_baseline = baseline_regions[region_name]
    return float(region_baseline["mean"]), float(region_baseline["std"])


def _rank_regions(
    regions: list[dict[str, Any]],
    class_name: str,
    loaded_messages: dict[str, dict[str, str]],
    std_multiplier: float,
    min_relative_to_max: float,
    min_problem_region_error: float,
    max_regions: int,
    baseline_regions: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    enriched_regions: list[dict[str, Any]] = []
    for region in regions:
        baseline_mean, baseline_std = _baseline_for_region(baseline_regions, region["region"])
        if region.get("insufficient_data"):
            z_score = -1_000_000.0
            baseline_std_used = None
        elif baseline_mean is None or baseline_std is None:
            z_score = float(region["mean_error"])
            baseline_std_used = None
        else:
            baseline_std_used = max(float(baseline_std), Z_SCORE_EPSILON)
            z_score = (float(region["mean_error"]) - float(baseline_mean)) / baseline_std_used

        feedback_item = {
            **region,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "baseline_std_used": baseline_std_used,
            "z_score": float(z_score),
            "message": message_for_region(class_name, region["region"], loaded_messages),
        }
        enriched_regions.append(feedback_item)

    ranked = sorted(enriched_regions, key=lambda item: item["z_score"], reverse=True)
    sufficient_regions = [region for region in ranked if not region.get("insufficient_data")]
    z_scores = np.asarray([region["z_score"] for region in sufficient_regions], dtype=np.float32)
    mean_errors = np.asarray([region["mean_error"] for region in sufficient_regions], dtype=np.float32)
    mean_error = float(mean_errors.mean()) if mean_errors.size else 0.0
    std_error = float(mean_errors.std()) if mean_errors.size else 0.0
    max_error = float(mean_errors.max()) if mean_errors.size else 0.0
    mean_z_score = float(z_scores.mean()) if z_scores.size else 0.0
    std_z_score = float(z_scores.std()) if z_scores.size else 0.0
    max_z_score = float(z_scores.max()) if z_scores.size else 0.0
    z_threshold = max(
        mean_z_score + std_multiplier * std_z_score,
        max_z_score * min_relative_to_max,
        1.0,
    )

    problem_regions: list[dict[str, Any]] = []
    for region in ranked:
        normalized_score = region["z_score"] / max(max_z_score, Z_SCORE_EPSILON)
        is_problem = (
            not region.get("insufficient_data")
            and region["z_score"] >= z_threshold
            and region["mean_error"] > min_problem_region_error
        )
        region["normalized_score"] = float(normalized_score)
        region["is_problem"] = bool(is_problem)
        if is_problem and len(problem_regions) < max_regions:
            problem_regions.append(region)

    return {
        "mean_error": mean_error,
        "std_error": std_error,
        "max_region_error": max_error,
        "mean_z_score": mean_z_score,
        "std_z_score": std_z_score,
        "max_z_score": max_z_score,
        "threshold": float(z_threshold),
        "scoring": "z_score" if baseline_regions else "raw_error",
        "problem_regions": problem_regions,
        "all_regions": ranked,
    }


def build_region_feedback(
    class_name: str,
    input_image: np.ndarray,
    reconstruction: np.ndarray,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
    std_multiplier: float = DEFAULT_STD_MULTIPLIER,
    min_relative_to_max: float = DEFAULT_MIN_RELATIVE_TO_MAX,
    min_problem_region_error: float = DEFAULT_MIN_PROBLEM_REGION_ERROR,
    max_regions: int = 3,
    messages: dict[str, dict[str, str]] | None = None,
    baseline: dict[str, Any] | None = None,
    ink_threshold: float = DEFAULT_INK_THRESHOLD,
    min_ink_pixels: int = DEFAULT_MIN_INK_PIXELS,
) -> dict[str, Any]:
    """Return ranked fixed-grid feedback from an input/reconstruction pair.

    The threshold is intentionally conservative for now: it surfaces cells whose
    mean error is both high relative to the grid's own distribution and reasonably
    close to the highest-error cell. Phase 6 deliberate-mistake validation should
    tune these constants using known flawed samples.
    """

    if max_regions <= 0:
        raise ValueError("max_regions must be positive")

    error_map = reconstruction_error_map(input_image, reconstruction)
    ink_mask = ink_relevance_mask(input_image, reconstruction, ink_threshold=ink_threshold)
    loaded_messages = messages or load_feedback_messages()
    loaded_baseline = baseline if baseline is not None else load_region_baseline(class_name)
    fine_regions = aggregate_grid_errors(error_map, rows, cols, ink_mask=ink_mask, min_ink_pixels=min_ink_pixels)
    fine_feedback = _rank_regions(
        regions=fine_regions,
        class_name=class_name,
        loaded_messages=loaded_messages,
        std_multiplier=std_multiplier,
        min_relative_to_max=min_relative_to_max,
        min_problem_region_error=min_problem_region_error,
        max_regions=max_regions,
        baseline_regions=loaded_baseline.get("fine_grid") if loaded_baseline else None,
    )
    broad_regions = aggregate_broad_band_errors(error_map, ink_mask=ink_mask, min_ink_pixels=min_ink_pixels)
    broad_feedback = _rank_regions(
        regions=broad_regions,
        class_name=class_name,
        loaded_messages=loaded_messages,
        std_multiplier=std_multiplier,
        min_relative_to_max=min_relative_to_max,
        min_problem_region_error=min_problem_region_error,
        max_regions=max_regions,
        baseline_regions=loaded_baseline.get("broad_bands") if loaded_baseline else None,
    )

    raw_global_mean_error = float(error_map.mean())
    overall_score = max(0.0, 100.0 * (1.0 - min(1.0, raw_global_mean_error * 12.0)))
    threshold_settings = {
        "std_multiplier": std_multiplier,
        "min_relative_to_max": min_relative_to_max,
        "min_problem_region_error": min_problem_region_error,
        "scoring": fine_feedback["scoring"],
        "ink_threshold": ink_threshold,
        "min_ink_pixels": min_ink_pixels,
    }
    fine_grid = {
        "rows": rows,
        "cols": cols,
        "mean_error": fine_feedback["mean_error"],
        "raw_global_mean_error": raw_global_mean_error,
        "std_error": fine_feedback["std_error"],
        "max_region_error": fine_feedback["max_region_error"],
        "threshold": fine_feedback["threshold"],
        "mean_z_score": fine_feedback["mean_z_score"],
        "std_z_score": fine_feedback["std_z_score"],
        "max_z_score": fine_feedback["max_z_score"],
        "scoring": fine_feedback["scoring"],
        "problem_regions": fine_feedback["problem_regions"],
        "all_regions": fine_feedback["all_regions"],
    }
    broad_bands_feedback = {
        "bands": ["top", "middle", "bottom"],
        "mean_error": broad_feedback["mean_error"],
        "std_error": broad_feedback["std_error"],
        "max_region_error": broad_feedback["max_region_error"],
        "threshold": broad_feedback["threshold"],
        "mean_z_score": broad_feedback["mean_z_score"],
        "std_z_score": broad_feedback["std_z_score"],
        "max_z_score": broad_feedback["max_z_score"],
        "scoring": broad_feedback["scoring"],
        "problem_regions": broad_feedback["problem_regions"],
        "all_regions": broad_feedback["all_regions"],
    }
    return {
        "class_name": class_name,
        "grid": {"rows": rows, "cols": cols},
        "overall_score": float(overall_score),
        "mean_error": fine_feedback["mean_error"],
        "std_error": fine_feedback["std_error"],
        "max_region_error": fine_feedback["max_region_error"],
        "threshold": fine_feedback["threshold"],
        "mean_z_score": fine_feedback["mean_z_score"],
        "std_z_score": fine_feedback["std_z_score"],
        "max_z_score": fine_feedback["max_z_score"],
        "threshold_settings": threshold_settings,
        "problem_regions": fine_feedback["problem_regions"],
        "all_regions": fine_feedback["all_regions"],
        "fine_grid": fine_grid,
        "broad_bands": broad_bands_feedback,
    }
