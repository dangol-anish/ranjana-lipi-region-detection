#!/usr/bin/env python3
"""Build initial structural part masks for the 5 validated characters."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


PART_CONFIG: dict[str, list[dict[str, object]]] = {
    "a": [
        {"name": "top_head", "label": "Top head stroke", "roi": [8, 10, 120, 36], "broad": "top", "fine": "top-center", "min_coverage": 0.45, "required": True},
        {"name": "middle_body", "label": "Middle body curve", "roi": [18, 36, 88, 88], "broad": "middle", "fine": "middle-center", "min_coverage": 0.60, "required": True},
        {"name": "right_stem", "label": "Right vertical stem", "roi": [82, 28, 120, 104], "broad": "middle", "fine": "middle-right", "min_coverage": 0.35, "required": False},
        {"name": "lower_tail", "label": "Lower tail", "roi": [40, 86, 118, 124], "broad": "bottom", "fine": "bottom-center", "min_coverage": 0.35, "required": False},
    ],
    "aa": [
        {"name": "top_head", "label": "Top head stroke", "roi": [8, 10, 120, 36], "broad": "top", "fine": "top-center", "min_coverage": 0.35, "required": False},
        {"name": "left_loop", "label": "Left loop", "roi": [12, 38, 70, 92], "broad": "middle", "fine": "middle-left", "min_coverage": 0.50, "required": False},
        {"name": "right_stem", "label": "Right vertical stem", "roi": [78, 30, 122, 112], "broad": "middle", "fine": "middle-right", "min_coverage": 0.50, "required": False},
        {"name": "lower_tail", "label": "Lower tail", "roi": [46, 88, 122, 126], "broad": "bottom", "fine": "bottom-center", "min_coverage": 0.35, "required": True},
    ],
    "ka": [
        {"name": "top_head", "label": "Top head stroke", "roi": [8, 8, 112, 36], "broad": "top", "fine": "top-center", "min_coverage": 0.35, "required": False},
        {"name": "center_loop", "label": "Center loop", "roi": [25, 45, 83, 92], "broad": "middle", "fine": "middle-center", "min_coverage": 0.43, "required": True},
        {"name": "right_stem", "label": "Right stem", "roi": [78, 28, 120, 108], "broad": "middle", "fine": "middle-right", "min_coverage": 0.35, "required": True},
        {"name": "lower_tail", "label": "Lower tail", "roi": [45, 88, 122, 126], "broad": "bottom", "fine": "bottom-center", "min_coverage": 0.35, "required": False},
    ],
    "da": [
        {"name": "top_head", "label": "Top head stroke", "roi": [8, 8, 120, 36], "broad": "top", "fine": "top-center", "min_coverage": 0.35, "required": False},
        {"name": "middle_curve", "label": "Middle curve", "roi": [16, 38, 100, 88], "broad": "middle", "fine": "middle-center", "min_coverage": 0.38, "required": True},
        {"name": "bottom_curve", "label": "Bottom curve", "roi": [20, 78, 106, 122], "broad": "bottom", "fine": "bottom-center", "min_coverage": 0.35, "required": False},
    ],
    "dda": [
        {"name": "top_head", "label": "Top head stroke", "roi": [8, 8, 120, 38], "broad": "top", "fine": "top-center", "min_coverage": 0.35, "required": False},
        {"name": "left_curve", "label": "Left middle curve", "roi": [15, 42, 78, 94], "broad": "middle", "fine": "middle-left", "min_coverage": 0.40, "required": True},
        {"name": "right_stem", "label": "Right stem", "roi": [78, 28, 122, 116], "broad": "middle", "fine": "middle-right", "min_coverage": 0.35, "required": False},
        {"name": "lower_tail", "label": "Lower tail", "roi": [56, 86, 122, 126], "broad": "bottom", "fine": "bottom-center", "min_coverage": 0.35, "required": False},
    ],
}


def reference_path(class_name: str) -> Path:
    return BACKEND_ROOT / "ml" / "processed" / "references" / f"{class_name}.png"


def output_root() -> Path:
    return BACKEND_ROOT / "ml" / "saved_models" / "structural_part_masks"


def remove_small_components(mask: np.ndarray, min_area: int = 12) -> np.ndarray:
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def build_part_mask(reference: np.ndarray, roi: list[int]) -> np.ndarray:
    x0, y0, x1, y1 = [int(value) for value in roi]
    mask = np.zeros_like(reference, dtype=np.uint8)
    ink = reference > 12
    mask[y0:y1, x0:x1] = ink[y0:y1, x0:x1].astype(np.uint8) * 255
    mask = remove_small_components(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def save_overlay(class_name: str, reference: np.ndarray, part_masks: list[tuple[str, np.ndarray]]) -> None:
    overlay = cv2.cvtColor(reference, cv2.COLOR_GRAY2RGB)
    colors = [(255, 70, 70), (70, 180, 70), (70, 120, 255), (255, 180, 40), (180, 80, 255)]
    for index, (_name, mask) in enumerate(part_masks):
        color = np.asarray(colors[index % len(colors)], dtype=np.uint8)
        active = mask > 0
        overlay[active] = (0.45 * overlay[active] + 0.55 * color).astype(np.uint8)
    Image.fromarray(overlay).resize((384, 384), Image.Resampling.NEAREST).save(
        output_root() / class_name / "parts_overlay.png"
    )


def main() -> None:
    root = output_root()
    root.mkdir(parents=True, exist_ok=True)
    for class_name, parts in PART_CONFIG.items():
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        reference = cv2.imread(str(reference_path(class_name)), cv2.IMREAD_GRAYSCALE)
        if reference is None:
            raise FileNotFoundError(f"Missing processed reference: {reference_path(class_name)}")

        config = {
            "class_name": class_name,
            "source": str(reference_path(class_name)),
            "tolerance_pixels": 3,
            "parts": [],
        }
        saved_masks: list[tuple[str, np.ndarray]] = []
        for part in parts:
            mask = build_part_mask(reference, part["roi"])
            mask_name = f"{part['name']}.png"
            Image.fromarray(mask).save(class_dir / mask_name)
            saved_masks.append((str(part["name"]), mask))
            config["parts"].append(
                {
                    "name": part["name"],
                    "label": part["label"],
                    "mask": mask_name,
                    "broad_region": part["broad"],
                    "fine_region": part["fine"],
                    "min_coverage": part["min_coverage"],
                    "required": part["required"],
                    "roi": part["roi"],
                }
            )
        (class_dir / "parts.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        save_overlay(class_name, reference, saved_masks)
        print(f"{class_name}: saved {len(parts)} structural part masks")


if __name__ == "__main__":
    main()
