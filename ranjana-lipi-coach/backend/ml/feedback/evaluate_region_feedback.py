#!/usr/bin/env python3
"""Evaluate fixed-grid feedback on normalized images with trained autoencoders."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    from .grid_feedback import build_region_feedback
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from grid_feedback import build_region_feedback

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
sys.path.append(str(TRAINING_DIR))
from autoencoder import RanjanaAutoencoder  # noqa: E402
from dataset import CLASSES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-grid region feedback evaluation")
    parser.add_argument("--class-name", choices=CLASSES)
    parser.add_argument("--image", type=Path, help="Path to one normalized 128x128 PNG")
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--max-regions", type=int, default=3)
    parser.add_argument("--min-problem-region-error", type=float, default=0.02)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument(
        "--correct-val-smoke-test",
        action="store_true",
        help="Evaluate one held-out correct validation image per class from val_split.json.",
    )
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


def load_image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("L")
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


def load_autoencoder(saved_models_root: Path, class_name: str, device: torch.device) -> RanjanaAutoencoder:
    checkpoint_path = saved_models_root / f"autoencoder_{class_name}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing autoencoder checkpoint: {checkpoint_path}")
    model = RanjanaAutoencoder().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_one(
    class_name: str,
    image_path: Path,
    saved_models_root: Path,
    device: torch.device,
    rows: int,
    cols: int,
    max_regions: int,
    min_problem_region_error: float,
) -> dict[str, object]:
    model = load_autoencoder(saved_models_root, class_name, device)
    image_tensor = load_image_tensor(image_path).to(device)
    with torch.no_grad():
        reconstruction = model(image_tensor)

    input_array = image_tensor.cpu().squeeze(0).squeeze(0).numpy()
    reconstruction_array = reconstruction.cpu().squeeze(0).squeeze(0).numpy()
    feedback = build_region_feedback(
        class_name=class_name,
        input_image=input_array,
        reconstruction=reconstruction_array,
        rows=rows,
        cols=cols,
        max_regions=max_regions,
        min_problem_region_error=min_problem_region_error,
    )
    feedback["image_path"] = str(image_path.resolve())
    return feedback


def first_val_image_per_class(saved_models_root: Path) -> dict[str, Path]:
    split_path = saved_models_root / "val_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    return {
        class_name: Path(split["validation_files"][class_name][0])
        for class_name in CLASSES
    }


def main() -> None:
    args = parse_args()
    root = project_root()
    saved_models_root = root / "ranjana-lipi-coach" / "backend" / "ml" / "saved_models"
    device = resolve_device(args.device)

    if args.correct_val_smoke_test:
        results = []
        for class_name, image_path in first_val_image_per_class(saved_models_root).items():
            result = evaluate_one(
                class_name,
                image_path,
                saved_models_root,
                device,
                args.rows,
                args.cols,
                args.max_regions,
                args.min_problem_region_error,
            )
            results.append(result)
            print(
                f"{class_name}: overall={result['overall_score']:.2f} "
                f"mean_error={result['mean_error']:.6f} "
                f"problem_regions={len(result['problem_regions'])}"
            )
        print(json.dumps(results, indent=2))
        return

    if args.class_name is None or args.image is None:
        raise SystemExit("Provide --class-name and --image, or use --correct-val-smoke-test")

    result = evaluate_one(
        args.class_name,
        args.image,
        saved_models_root,
        device,
        args.rows,
        args.cols,
        args.max_regions,
        args.min_problem_region_error,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
