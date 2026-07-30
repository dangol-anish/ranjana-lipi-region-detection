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


@dataclass(frozen=True)
class GridCell:
    row: int
    col: int
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


def aggregate_grid_errors(
    error_map: np.ndarray,
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
) -> list[dict[str, Any]]:
    error_array = np.asarray(error_map, dtype=np.float32)
    if error_array.ndim != 2:
        raise ValueError("error_map must be a 2D array")

    height, width = error_array.shape
    results: list[dict[str, Any]] = []
    for cell in grid_cells(height, width, rows, cols):
        region = error_array[cell.y_start : cell.y_end, cell.x_start : cell.x_end]
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
                "mean_error": float(region.mean()) if region.size else 0.0,
                "max_error": float(region.max()) if region.size else 0.0,
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
    cells = aggregate_grid_errors(error_map, rows, cols)
    cell_means = np.asarray([cell["mean_error"] for cell in cells], dtype=np.float32)
    mean_error = float(cell_means.mean()) if cell_means.size else 0.0
    std_error = float(cell_means.std()) if cell_means.size else 0.0
    max_error = float(cell_means.max()) if cell_means.size else 0.0
    threshold = max(
        mean_error + std_multiplier * std_error,
        max_error * min_relative_to_max,
        min_problem_region_error,
    )
    loaded_messages = messages or load_feedback_messages()

    ranked = sorted(cells, key=lambda item: item["mean_error"], reverse=True)
    problem_regions: list[dict[str, Any]] = []
    for cell in ranked:
        normalized_score = cell["mean_error"] / max(max_error, 1e-8)
        is_problem = cell["mean_error"] >= threshold and cell["mean_error"] > 0
        feedback_item = {
            **cell,
            "normalized_score": float(normalized_score),
            "is_problem": bool(is_problem),
            "message": message_for_region(class_name, cell["region"], loaded_messages),
        }
        if is_problem and len(problem_regions) < max_regions:
            problem_regions.append(feedback_item)

    overall_score = max(0.0, 100.0 * (1.0 - min(1.0, mean_error * 12.0)))
    return {
        "class_name": class_name,
        "grid": {"rows": rows, "cols": cols},
        "overall_score": float(overall_score),
        "mean_error": mean_error,
        "std_error": std_error,
        "max_region_error": max_error,
        "threshold": float(threshold),
        "threshold_settings": {
            "std_multiplier": std_multiplier,
            "min_relative_to_max": min_relative_to_max,
            "min_problem_region_error": min_problem_region_error,
        },
        "problem_regions": problem_regions,
        "all_regions": [
            {
                **cell,
                "normalized_score": float(cell["mean_error"] / max(max_error, 1e-8)),
                "message": message_for_region(class_name, cell["region"], loaded_messages),
            }
            for cell in ranked
        ],
    }
