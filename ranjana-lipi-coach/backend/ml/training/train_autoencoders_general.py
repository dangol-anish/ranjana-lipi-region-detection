#!/usr/bin/env python3
"""Train one general autoencoder per class for all 62 Ranjana dataset classes."""

from __future__ import annotations

import argparse
import csv
import json
import random
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader, Dataset

from autoencoder import RanjanaAutoencoder

FEEDBACK_DIR = Path(__file__).resolve().parents[1] / "feedback"
import sys

sys.path.append(str(FEEDBACK_DIR))
from grid_feedback import (  # noqa: E402
    DEFAULT_INK_THRESHOLD,
    DEFAULT_MIN_INK_PIXELS,
    aggregate_broad_band_errors,
    aggregate_grid_errors,
    ink_relevance_mask,
    reconstruction_error_map,
)


class ReconstructionImageDataset(Dataset):
    def __init__(self, image_paths: list[Path]) -> None:
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image = image.convert("L")
            array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0), str(image_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 62 general per-class autoencoders")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--visualization-samples", type=int, default=3)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--ink-threshold", type=float, default=DEFAULT_INK_THRESHOLD)
    parser.add_argument("--min-ink-pixels", type=int, default=DEFAULT_MIN_INK_PIXELS)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def log_line(log_path: Path, message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(line + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        print("MPS requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def png_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.glob("*.png") if path.is_file())


def class_names(processed_root: Path) -> list[str]:
    return sorted(
        path.name
        for path in processed_root.iterdir()
        if path.is_dir() and path.name != "references"
    )


def load_val_split(saved_models_root: Path, classes: list[str]) -> dict[str, list[Path]]:
    split_path = saved_models_root / "val_split_general.json"
    if not split_path.is_file():
        raise FileNotFoundError(
            f"General validation split not found: {split_path}. "
            "Run train_recognizer_general.py first, at least as a dry run."
        )

    split = json.loads(split_path.read_text(encoding="utf-8"))
    return {
        class_name: [Path(path) for path in split["validation_files"][class_name]]
        for class_name in classes
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_seen = 0

    for images, _paths in loader:
        images = images.to(device)
        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            reconstructions = model(images)
            loss = criterion(reconstructions, images)
            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_seen += batch_size

    return total_loss / max(1, total_seen)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def heatmap_from_error(error: np.ndarray) -> Image.Image:
    max_value = float(error.max())
    normalized = error / max(max_value, 1e-8)
    heatmap = np.zeros((*normalized.shape, 3), dtype=np.uint8)
    heatmap[..., 0] = (normalized * 255).astype(np.uint8)
    heatmap[..., 1] = ((1.0 - np.abs(normalized - 0.5) * 2.0) * 180).astype(np.uint8)
    heatmap[..., 2] = ((1.0 - normalized) * 255).astype(np.uint8)
    return Image.fromarray(heatmap, mode="RGB")


def save_reconstruction_visualizations(
    model: nn.Module,
    image_paths: list[Path],
    class_name: str,
    output_dir: Path,
    device: torch.device,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    title_font = _font(16)
    model.eval()

    for index, image_path in enumerate(image_paths, start=1):
        dataset = ReconstructionImageDataset([image_path])
        image_tensor, _path = dataset[0]
        with torch.no_grad():
            reconstruction = model(image_tensor.unsqueeze(0).to(device)).cpu().squeeze(0)

        input_array = image_tensor.squeeze(0).numpy()
        reconstruction_array = reconstruction.squeeze(0).numpy()
        error_array = np.abs(input_array - reconstruction_array)
        input_image = Image.fromarray((input_array * 255).astype(np.uint8), mode="L").convert("RGB")
        reconstruction_image = Image.fromarray(
            (reconstruction_array * 255).astype(np.uint8),
            mode="L",
        ).convert("RGB")
        heatmap_image = heatmap_from_error(error_array)
        panel_w, panel_h = 128, 128
        padding = 18
        header_h = 34
        canvas = Image.new(
            "RGB",
            (panel_w * 3 + padding * 4, panel_h + header_h + padding),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        labels = ("Input", "Reconstruction", "Error heatmap")
        panels = (input_image, reconstruction_image, heatmap_image)
        for panel_index, (label, panel) in enumerate(zip(labels, panels)):
            x = padding + panel_index * (panel_w + padding)
            draw.text((x, 8), label, fill="black", font=title_font)
            canvas.paste(panel.resize((panel_w, panel_h)), (x, header_h))

        output_path = output_dir / f"autoencoder_{class_name}_reconstruction_{index}.png"
        canvas.save(output_path)
        saved_paths.append(str(output_path.resolve()))

    return saved_paths


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


def calibrate_baseline(
    model: nn.Module,
    class_name: str,
    val_paths: list[Path],
    output_root: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> str:
    fine_errors: dict[str, list[float]] = {}
    broad_errors: dict[str, list[float]] = {}
    image_error_means: list[float] = []
    model.eval()

    for image_path in val_paths:
        with Image.open(image_path) as image:
            image = image.convert("L")
            input_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(input_array).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            reconstruction = model(image_tensor).cpu().squeeze(0).squeeze(0).numpy()
        error_map = reconstruction_error_map(input_array, reconstruction)
        ink_mask = ink_relevance_mask(input_array, reconstruction, ink_threshold=args.ink_threshold)
        image_error_means.append(float(error_map[ink_mask].mean()) if np.any(ink_mask) else 0.0)
        for region in aggregate_grid_errors(
            error_map,
            args.rows,
            args.cols,
            ink_mask=ink_mask,
            min_ink_pixels=args.min_ink_pixels,
        ):
            if not region["insufficient_data"]:
                fine_errors.setdefault(region["region"], []).append(float(region["mean_error"]))
        for region in aggregate_broad_band_errors(
            error_map,
            ink_mask=ink_mask,
            min_ink_pixels=args.min_ink_pixels,
        ):
            if not region["insufficient_data"]:
                broad_errors.setdefault(region["region"], []).append(float(region["mean_error"]))

    baseline = {
        "class_name": class_name,
        "rows": args.rows,
        "cols": args.cols,
        "validation_sample_count": len(val_paths),
        "scoring": "ink_masked_z_score",
        "ink_threshold": args.ink_threshold,
        "min_ink_pixels": args.min_ink_pixels,
        "mean_image_error": float(np.asarray(image_error_means, dtype=np.float32).mean()),
        "fine_grid": summarize_regions(fine_errors),
        "broad_bands": summarize_regions(broad_errors),
        "validation_files": [str(path.resolve()) for path in val_paths],
    }
    output_path = output_root / f"region_baseline_{class_name}.json"
    output_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    return str(output_path.resolve())


def train_one(
    class_name: str,
    train_paths: list[Path],
    val_paths: list[Path],
    output_root: Path,
    device: torch.device,
    args: argparse.Namespace,
    log_path: Path,
) -> dict[str, Any]:
    train_loader = DataLoader(
        ReconstructionImageDataset(train_paths),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        ReconstructionImageDataset(val_paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = RanjanaAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    checkpoint_path = output_root / f"autoencoder_{class_name}.pt"
    eval_dir = output_root / f"eval_{class_name}"
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        log_line(
            log_path,
            f"{class_name} epoch {epoch:03d}/{args.epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}",
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_name": class_name,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                log_line(log_path, f"{class_name}: early stopping after {epoch} epochs")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    visualizations = save_reconstruction_visualizations(
        model,
        val_paths[: args.visualization_samples],
        class_name,
        eval_dir,
        device,
    )
    baseline_path = calibrate_baseline(model, class_name, val_paths, output_root, device, args)
    return {
        "class_name": class_name,
        "success": True,
        "best_epoch": best_epoch,
        "final_reconstruction_loss": best_val_loss,
        "checkpoint": str(checkpoint_path.resolve()),
        "eval_dir": str(eval_dir.resolve()),
        "visualizations": visualizations,
        "baseline": baseline_path,
        "error": "",
    }


def save_summary(results: dict[str, dict[str, Any]], output_root: Path) -> None:
    (output_root / "autoencoder_general_metrics.json").write_text(
        json.dumps({"per_class": results}, indent=2),
        encoding="utf-8",
    )
    with (output_root / "autoencoder_general_summary.csv").open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["character", "trained_successfully", "final_reconstruction_loss", "errors"],
        )
        writer.writeheader()
        for class_name, result in results.items():
            loss = result.get("final_reconstruction_loss")
            writer.writerow(
                {
                    "character": class_name,
                    "trained_successfully": "YES" if result["success"] else "NO",
                    "final_reconstruction_loss": f"{loss:.6f}" if isinstance(loss, float) else "",
                    "errors": result.get("error", ""),
                }
            )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    root = project_root()
    ml_root = root / "ranjana-lipi-coach" / "backend" / "ml"
    processed_root = ml_root / "processed_general"
    augmented_root = ml_root / "augmented_general"
    saved_models_root = ml_root / "saved_models"
    output_root = saved_models_root / "autoencoders_general"
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "training_log.txt"
    classes = class_names(processed_root)

    if len(classes) != 62:
        raise RuntimeError(f"Expected 62 processed_general classes, found {len(classes)}")

    val_split = load_val_split(saved_models_root, classes)
    log_path.write_text("", encoding="utf-8")
    log_line(log_path, f"Device: {device}")
    log_line(log_path, f"Output root: {output_root}")
    log_line(log_path, f"Classes: {len(classes)}")

    class_train_paths: dict[str, list[Path]] = {}
    for class_name in classes:
        train_paths = png_files(augmented_root / class_name)
        val_paths = val_split[class_name]
        if not train_paths:
            raise FileNotFoundError(f"No augmented_general images found for {class_name}")
        missing_val = [path for path in val_paths if not path.is_file()]
        if missing_val:
            raise FileNotFoundError(f"Validation files missing for {class_name}: {missing_val[:3]}")
        class_train_paths[class_name] = train_paths
        log_line(log_path, f"{class_name}: train={len(train_paths)} val={len(val_paths)}")

    if args.dry_run:
        log_line(log_path, "Dry run complete: general autoencoder inputs are ready.")
        return

    results: dict[str, dict[str, Any]] = {}
    for class_name in classes:
        log_line(log_path, f"START {class_name}")
        try:
            results[class_name] = train_one(
                class_name,
                class_train_paths[class_name],
                val_split[class_name],
                output_root,
                device,
                args,
                log_path,
            )
            log_line(
                log_path,
                f"SUCCESS {class_name} final_reconstruction_loss={results[class_name]['final_reconstruction_loss']:.6f}",
            )
        except Exception as exc:  # noqa: BLE001 - long run should continue class by class.
            error_text = f"{type(exc).__name__}: {exc}"
            log_line(log_path, f"FAILED {class_name}: {error_text}")
            with log_path.open("a", encoding="utf-8") as log_file:
                traceback.print_exc(file=log_file)
            results[class_name] = {
                "class_name": class_name,
                "success": False,
                "final_reconstruction_loss": None,
                "error": error_text,
            }

    save_summary(results, output_root)
    log_line(log_path, "FINAL SUMMARY")
    for class_name, result in results.items():
        loss = result.get("final_reconstruction_loss")
        loss_text = f"{loss:.6f}" if isinstance(loss, float) else ""
        log_line(
            log_path,
            f"{class_name} trained={'YES' if result['success'] else 'NO'} "
            f"final_reconstruction_loss={loss_text} error={result.get('error', '')}",
        )


if __name__ == "__main__":
    main()
