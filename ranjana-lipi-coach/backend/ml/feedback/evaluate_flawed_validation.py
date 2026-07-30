#!/usr/bin/env python3
"""Evaluate broad-region flawed validation samples for demo readiness."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

try:
    from .grid_feedback import build_region_feedback
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from grid_feedback import build_region_feedback

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
PREPROCESSING_DIR = Path(__file__).resolve().parents[1] / "preprocessing"
sys.path.append(str(TRAINING_DIR))
sys.path.append(str(PREPROCESSING_DIR))
from autoencoder import RanjanaAutoencoder  # noqa: E402
from dataset import CLASSES  # noqa: E402
from normalize import apply_fixed_transform  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
BROAD_REGION_MAP = {
    "top": {"top-left", "top-center", "top-right", "middle-left", "middle-center", "middle-right"},
    "upper": {"top-left", "top-center", "top-right", "middle-left", "middle-center", "middle-right"},
    "middle": {"middle-left", "middle-center", "middle-right", "top-center", "bottom-center"},
    "center": {"middle-left", "middle-center", "middle-right", "top-center", "bottom-center"},
    "central": {"middle-left", "middle-center", "middle-right", "top-center", "bottom-center"},
    "bottom": {
        "bottom-left",
        "bottom-center",
        "bottom-right",
        "middle-left",
        "middle-center",
        "middle-right",
    },
    "lower": {
        "bottom-left",
        "bottom-center",
        "bottom-right",
        "middle-left",
        "middle-center",
        "middle-right",
    },
    "left": {"top-left", "middle-left", "bottom-left", "top-center", "middle-center", "bottom-center"},
    "right": {"top-right", "middle-right", "bottom-right", "top-center", "middle-center", "bottom-center"},
}
BROAD_BAND_TOKENS = {
    "top": "top",
    "upper": "top",
    "middle": "middle",
    "center": "middle",
    "central": "middle",
    "bottom": "bottom",
    "lower": "bottom",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate data/FlawedValidation images with broad-region matching"
    )
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-regions", type=int, default=3)
    parser.add_argument("--min-problem-region-error", type=float, default=0.012)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV output path. Defaults to saved_models/flawed_validation_results_v2.csv.",
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


def flawed_images(root: Path) -> list[tuple[str, Path]]:
    samples: list[tuple[str, Path]] = []
    for class_name in CLASSES:
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((class_name, path))
    return samples


def expected_regions_from_name(path: Path) -> set[str]:
    tokens = set(re.split(r"[^a-z0-9]+", path.stem.lower()))
    expected: set[str] = set()
    for token in tokens:
        expected.update(BROAD_REGION_MAP.get(token, set()))
    return expected


def intended_broad_band_from_name(path: Path) -> str:
    tokens = re.split(r"[^a-z0-9]+", path.stem.lower())
    for token in tokens:
        if token in BROAD_BAND_TOKENS:
            return BROAD_BAND_TOKENS[token]
    return "unknown"


def load_autoencoder(saved_models_root: Path, class_name: str, device: torch.device) -> RanjanaAutoencoder:
    checkpoint_path = saved_models_root / f"autoencoder_{class_name}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing autoencoder checkpoint: {checkpoint_path}")
    model = RanjanaAutoencoder().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_alignment_transforms(processed_root: Path) -> dict[str, dict[str, object]]:
    transforms_path = processed_root / "alignment_transforms.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(f"Missing alignment transforms: {transforms_path}")
    return json.loads(transforms_path.read_text(encoding="utf-8"))


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def save_demo_overlay(normalized: np.ndarray, result: dict[str, object], output_path: Path) -> None:
    image = Image.fromarray((normalized * 255).astype(np.uint8), mode="L").convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = _font(11)
    problem_names = {region["region"] for region in result["problem_regions"]}
    top_names = {region["region"] for region in result["predicted_top_regions"]}

    for region in result["all_regions"]:
        bounds = region["bounds"]
        x0, y0 = bounds["x_start"], bounds["y_start"]
        x1, y1 = bounds["x_end"], bounds["y_end"]
        name = region["region"]
        if name in problem_names:
            fill = (255, 50, 50, 88)
            outline = (255, 0, 0, 255)
        elif name in top_names:
            fill = (255, 190, 0, 55)
            outline = (230, 140, 0, 220)
        else:
            fill = (255, 255, 255, 0)
            outline = (40, 120, 255, 120)
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=fill, outline=outline, width=1)
        draw.text((x0 + 2, y0 + 2), name.replace("-", "\n"), fill=(0, 0, 0, 210), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.resize((384, 384), resample=Image.Resampling.NEAREST).save(output_path)


def evaluate_sample(
    class_name: str,
    image_path: Path,
    saved_models_root: Path,
    output_root: Path,
    transform: dict[str, object],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, object]:
    normalized = apply_fixed_transform(image_path, transform)
    image_tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(device)
    model = load_autoencoder(saved_models_root, class_name, device)
    with torch.no_grad():
        reconstruction = model(image_tensor).cpu().squeeze(0).squeeze(0).numpy()

    result = build_region_feedback(
        class_name=class_name,
        input_image=normalized,
        reconstruction=reconstruction,
        rows=args.rows,
        cols=args.cols,
        max_regions=args.max_regions,
        min_problem_region_error=args.min_problem_region_error,
    )
    predicted_top_regions = result["fine_grid"]["all_regions"][: args.top_k]
    predicted_broad_bands = result["broad_bands"]["all_regions"][: args.top_k]
    expected = expected_regions_from_name(image_path)
    predicted_names = {region["region"] for region in predicted_top_regions}
    strict_names = {region["region"] for region in result["problem_regions"]}
    matched = bool(expected & predicted_names) if expected else None

    result.update(
        {
            "image_path": str(image_path.resolve()),
            "filename": image_path.name,
            "intended_broad_band": intended_broad_band_from_name(image_path),
            "expected_regions": sorted(expected),
            "predicted_top_regions": predicted_top_regions,
            "predicted_broad_bands": predicted_broad_bands,
            "matched_expected_in_top_k": matched,
            "matched_expected_in_problem_regions": bool(expected & strict_names) if expected else None,
        }
    )
    overlay_path = output_root / f"flawed_{class_name}_{image_path.stem}_overlay.png"
    save_demo_overlay(normalized, result, overlay_path)
    result["overlay_path"] = str(overlay_path.resolve())
    return result


def region_names(regions: list[dict[str, object]]) -> str:
    return ";".join(str(region["region"]) for region in regions)


def write_summary_csv(results: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "filename",
        "class_name",
        "intended_flaw_region",
        "fine_grid_top_3_problem_regions_ranked",
        "broad_bands_top_3_problem_regions_ranked",
        "overall_reconstruction_error_score",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "filename": result["filename"],
                    "class_name": result["class_name"],
                    "intended_flaw_region": result["intended_broad_band"],
                    "fine_grid_top_3_problem_regions_ranked": region_names(result["predicted_top_regions"]),
                    "broad_bands_top_3_problem_regions_ranked": region_names(result["predicted_broad_bands"]),
                    "overall_reconstruction_error_score": f"{float(result['overall_score']):.4f}",
                }
            )


def main() -> None:
    args = parse_args()
    root = project_root()
    data_root = root / "data" / "FlawedValidation"
    saved_models_root = root / "ranjana-lipi-coach" / "backend" / "ml" / "saved_models"
    processed_root = root / "ranjana-lipi-coach" / "backend" / "ml" / "processed"
    output_root = saved_models_root / "flawed_validation"
    device = resolve_device(args.device)
    transforms = load_alignment_transforms(processed_root)

    samples = flawed_images(data_root)
    if not samples:
        raise FileNotFoundError(f"No flawed validation images found under {data_root}")

    results = [
        evaluate_sample(class_name, path, saved_models_root, output_root, transforms[class_name], device, args)
        for class_name, path in samples
    ]
    matched = [item for item in results if item["matched_expected_in_top_k"] is True]
    evaluated = [item for item in results if item["matched_expected_in_top_k"] is not None]

    summary = {
        "sample_count": len(results),
        "evaluated_with_expected_region_count": len(evaluated),
        "top_k": args.top_k,
        "top_k_match_count": len(matched),
        "top_k_match_rate": len(matched) / len(evaluated) if evaluated else None,
        "note": (
            "Broad-region demo validation: filenames like top/middle/bottom are "
            "matched against that band plus adjacent center cells, because "
            "deliberate flaws may span multiple grid cells after normalization."
        ),
        "results": results,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = saved_models_root / "flawed_validation_results.json"
    csv_output_path = args.csv_output or saved_models_root / "flawed_validation_results_v2.csv"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_csv(results, csv_output_path)

    broad_rank_1 = 0
    broad_top_2 = 0
    broad_evaluated = 0
    for result in results:
        intended = result["intended_broad_band"]
        if intended == "unknown":
            continue
        broad_evaluated += 1
        broad_regions = [region["region"] for region in result["predicted_broad_bands"]]
        if broad_regions and broad_regions[0] == intended:
            broad_rank_1 += 1
        if intended in broad_regions[:2]:
            broad_top_2 += 1

    print(
        f"Flawed validation: {len(matched)}/{len(evaluated)} matched expected broad "
        f"region in top {args.top_k} regions"
    )
    print(
        f"Broad-band #1 match: {broad_rank_1}/{broad_evaluated}; "
        f"broad-band top-2 match: {broad_top_2}/{broad_evaluated}"
    )
    print()
    print(
        "filename,intended_flaw_region,fine_grid_top_3_problem_regions_ranked,"
        "broad_bands_top_3_problem_regions_ranked,overall_reconstruction_error_score"
    )
    for result in results:
        top_regions = ", ".join(region["region"] for region in result["predicted_top_regions"])
        broad_regions = ", ".join(region["region"] for region in result["predicted_broad_bands"])
        expected = ", ".join(result["expected_regions"]) or "unknown"
        print(
            f"{Path(result['image_path']).name},{result['intended_broad_band']},"
            f"\"{top_regions}\",\"{broad_regions}\",{result['overall_score']:.2f}"
            f"  expected=[{expected}] problem_regions={len(result['problem_regions'])}"
        )
    print(f"Saved results: {output_path}")
    print(f"Saved CSV: {csv_output_path}")
    print(f"Saved overlays: {output_root}")


if __name__ == "__main__":
    main()
