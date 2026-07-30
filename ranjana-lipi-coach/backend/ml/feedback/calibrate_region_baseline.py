#!/usr/bin/env python3
"""Calibrate per-region reconstruction-error baselines on correct validation images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

try:
    from .grid_feedback import (
        DEFAULT_INK_THRESHOLD,
        DEFAULT_MIN_INK_PIXELS,
        aggregate_broad_band_errors,
        aggregate_grid_errors,
        ink_relevance_mask,
        reconstruction_error_map,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from grid_feedback import (
        DEFAULT_INK_THRESHOLD,
        DEFAULT_MIN_INK_PIXELS,
        aggregate_broad_band_errors,
        aggregate_grid_errors,
        ink_relevance_mask,
        reconstruction_error_map,
    )

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
sys.path.append(str(TRAINING_DIR))
from autoencoder import RanjanaAutoencoder  # noqa: E402
from dataset import CLASSES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate fine-grid and broad-band error baselines per class"
    )
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--ink-threshold", type=float, default=DEFAULT_INK_THRESHOLD)
    parser.add_argument("--min-ink-pixels", type=int, default=DEFAULT_MIN_INK_PIXELS)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        print("MPS requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        return np.asarray(image, dtype=np.float32) / 255.0


def load_val_split(saved_models_root: Path) -> dict[str, list[Path]]:
    split_path = saved_models_root / "val_split.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Validation split not found: {split_path}")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    return {
        class_name: [Path(path) for path in split["validation_files"][class_name]]
        for class_name in CLASSES
    }


def load_autoencoder(saved_models_root: Path, class_name: str, device: torch.device) -> RanjanaAutoencoder:
    checkpoint_path = saved_models_root / f"autoencoder_{class_name}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing autoencoder checkpoint: {checkpoint_path}")

    model = RanjanaAutoencoder().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def reconstruction_for_image(model: RanjanaAutoencoder, image: np.ndarray, device: torch.device) -> np.ndarray:
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        reconstruction = model(image_tensor)
    return reconstruction.cpu().squeeze(0).squeeze(0).numpy()


def summarize_regions(region_errors: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for region_name, errors in sorted(region_errors.items()):
        values = np.asarray(errors, dtype=np.float32)
        summary[region_name] = {
            "mean": float(values.mean()) if values.size else 0.0,
            "std": float(values.std(ddof=0)) if values.size else 0.0,
            "sample_count": int(values.size),
        }
    return summary


def calibrate_class(
    class_name: str,
    image_paths: list[Path],
    saved_models_root: Path,
    device: torch.device,
    rows: int,
    cols: int,
    ink_threshold: float,
    min_ink_pixels: int,
) -> dict[str, Any]:
    model = load_autoencoder(saved_models_root, class_name, device)
    fine_errors: dict[str, list[float]] = {}
    broad_errors: dict[str, list[float]] = {}
    image_error_means: list[float] = []

    for image_path in image_paths:
        image = load_image(image_path)
        reconstruction = reconstruction_for_image(model, image, device)
        error_map = reconstruction_error_map(image, reconstruction)
        ink_mask = ink_relevance_mask(image, reconstruction, ink_threshold=ink_threshold)
        image_error_means.append(float(error_map[ink_mask].mean()) if np.any(ink_mask) else 0.0)

        for region in aggregate_grid_errors(error_map, rows, cols, ink_mask=ink_mask, min_ink_pixels=min_ink_pixels):
            if not region["insufficient_data"]:
                fine_errors.setdefault(region["region"], []).append(float(region["mean_error"]))
        for region in aggregate_broad_band_errors(error_map, ink_mask=ink_mask, min_ink_pixels=min_ink_pixels):
            if not region["insufficient_data"]:
                broad_errors.setdefault(region["region"], []).append(float(region["mean_error"]))

    return {
        "class_name": class_name,
        "rows": rows,
        "cols": cols,
        "validation_sample_count": len(image_paths),
        "scoring": "ink_masked_z_score",
        "ink_threshold": ink_threshold,
        "min_ink_pixels": min_ink_pixels,
        "mean_image_error": float(np.asarray(image_error_means, dtype=np.float32).mean()),
        "fine_grid": summarize_regions(fine_errors),
        "broad_bands": summarize_regions(broad_errors),
        "validation_files": [str(path.resolve()) for path in image_paths],
    }


def main() -> None:
    args = parse_args()
    root = project_root()
    saved_models_root = root / "ranjana-lipi-coach" / "backend" / "ml" / "saved_models"
    device = resolve_device(args.device)
    val_split = load_val_split(saved_models_root)

    for class_name in CLASSES:
        baseline = calibrate_class(
            class_name=class_name,
            image_paths=val_split[class_name],
            saved_models_root=saved_models_root,
            device=device,
            rows=args.rows,
            cols=args.cols,
            ink_threshold=args.ink_threshold,
            min_ink_pixels=args.min_ink_pixels,
        )
        output_path = saved_models_root / f"region_baseline_{class_name}.json"
        output_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
        print(
            f"{class_name}: calibrated {baseline['validation_sample_count']} validation images "
            f"-> {output_path}"
        )


if __name__ == "__main__":
    main()
