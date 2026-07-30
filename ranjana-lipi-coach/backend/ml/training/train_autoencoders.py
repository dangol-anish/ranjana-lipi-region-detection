#!/usr/bin/env python3
"""Train Phase 5 per-character autoencoders for region-deviation detection."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from .autoencoder import RanjanaAutoencoder
    from .dataset import CLASSES
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from autoencoder import RanjanaAutoencoder
    from dataset import CLASSES


class ReconstructionImageDataset(Dataset):
    """Load normalized grayscale images as reconstruction targets."""

    def __init__(self, image_paths: list[str | Path]) -> None:
        self.image_paths = [Path(path) for path in image_paths]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image = image.convert("L")
            array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0), str(image_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one reconstruction autoencoder per selected Ranjana class"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--visualization-samples", type=int, default=3)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda", "mps"),
        help="Default is CPU so the script works on machines without a GPU.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths/splits and print counts without training.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


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


def load_val_split(saved_models_root: Path) -> dict[str, list[Path]]:
    split_path = saved_models_root / "val_split.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Validation split not found: {split_path}")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    validation_files = split.get("validation_files", {})
    return {
        class_name: [Path(path) for path in validation_files[class_name]]
        for class_name in CLASSES
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


def reconstruction_errors(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[float], list[str]]:
    model.eval()
    errors: list[float] = []
    paths: list[str] = []
    with torch.no_grad():
        for images, batch_paths in loader:
            images = images.to(device)
            reconstructions = model(images)
            batch_errors = torch.mean((images - reconstructions) ** 2, dim=(1, 2, 3))
            errors.extend(batch_errors.cpu().tolist())
            paths.extend(batch_paths)
    return errors, paths


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def save_loss_curves(
    histories: dict[str, dict[str, list[float]]],
    output_path: Path,
) -> None:
    width, height = 1100, 680
    margin_left, margin_right = 80, 40
    margin_top, margin_bottom = 70, 80
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _font(16)
    title_font = _font(22)
    draw.text((margin_left, 24), "Autoencoder Reconstruction Loss", fill="black", font=title_font)
    draw.rectangle(
        [margin_left, margin_top, margin_left + plot_w, margin_top + plot_h],
        outline=(40, 40, 40),
    )

    all_losses: list[float] = []
    max_epochs = 0
    for history in histories.values():
        all_losses.extend(history["train_loss"])
        all_losses.extend(history["val_loss"])
        max_epochs = max(max_epochs, len(history["train_loss"]))

    if not all_losses:
        image.save(output_path)
        return

    y_min = 0.0
    y_max = max(all_losses) * 1.05
    colors = {
        "aa": (35, 155, 86),
        "a": (39, 124, 179),
        "ka": (142, 68, 173),
        "da": (215, 48, 39),
        "dda": (230, 126, 34),
    }

    def point(epoch: int, value: float) -> tuple[float, float]:
        x = margin_left + ((epoch - 1) / max(1, max_epochs - 1)) * plot_w
        y = margin_top + plot_h - ((value - y_min) / max(1e-8, y_max - y_min)) * plot_h
        return x, y

    for class_name, history in histories.items():
        color = colors[class_name]
        points = [
            point(epoch, value)
            for epoch, value in enumerate(history["val_loss"], start=1)
        ]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        elif points:
            x, y = points[0]
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)

    legend_x = margin_left + 16
    legend_y = margin_top + 16
    for offset, class_name in enumerate(CLASSES):
        y = legend_y + offset * 24
        draw.rectangle([legend_x, y + 4, legend_x + 20, y + 14], fill=colors[class_name])
        draw.text((legend_x + 28, y), f"{class_name} val_loss", fill="black", font=font)

    draw.text((margin_left, height - 48), "Epoch", fill="black", font=font)
    draw.text((16, margin_top + plot_h // 2), "MSE", fill="black", font=font)
    image.save(output_path)


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
    model.eval()
    title_font = _font(16)

    for index, image_path in enumerate(image_paths, start=1):
        dataset = ReconstructionImageDataset([image_path])
        image_tensor, _ = dataset[0]
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


def save_checkpoint(
    model: nn.Module,
    output_path: Path,
    class_name: str,
    epoch: int,
    val_loss: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_name": class_name,
            "epoch": epoch,
            "val_loss": val_loss,
            "args": vars(args),
        },
        output_path,
    )


def train_one_autoencoder(
    class_name: str,
    train_paths: list[Path],
    val_paths: list[Path],
    saved_models_root: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    checkpoint_path = saved_models_root / f"autoencoder_{class_name}.pt"
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"{class_name} epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(model, checkpoint_path, class_name, epoch, val_loss, args)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"{class_name}: early stopping after {epoch} epochs.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    errors, paths = reconstruction_errors(model, val_loader, device)
    visualizations = save_reconstruction_visualizations(
        model,
        val_paths[: args.visualization_samples],
        class_name,
        saved_models_root,
        device,
    )

    return {
        "class_name": class_name,
        "train_count": len(train_paths),
        "validation_count": len(val_paths),
        "best_val_loss": best_val_loss,
        "val_error_mean": float(np.mean(errors)) if errors else 0.0,
        "val_error_std": float(np.std(errors)) if errors else 0.0,
        "val_error_min": float(np.min(errors)) if errors else 0.0,
        "val_error_max": float(np.max(errors)) if errors else 0.0,
        "checkpoint": str(checkpoint_path.resolve()),
        "history": history,
        "validation_errors": [
            {"path": path, "mse": error}
            for path, error in zip(paths, errors)
        ],
        "visualizations": visualizations,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    root = project_root()
    ml_root = root / "ranjana-lipi-coach" / "backend" / "ml"
    augmented_root = ml_root / "augmented"
    saved_models_root = ml_root / "saved_models"
    saved_models_root.mkdir(parents=True, exist_ok=True)
    val_split = load_val_split(saved_models_root)

    print(f"Device: {device}")
    print(f"Saved models: {saved_models_root}")
    print("Classes:", ", ".join(CLASSES))

    class_train_paths: dict[str, list[Path]] = {}
    for class_name in CLASSES:
        train_paths = png_files(augmented_root / class_name)
        val_paths = val_split[class_name]
        if not train_paths:
            raise FileNotFoundError(f"No augmented training images found for {class_name}")
        missing_val = [path for path in val_paths if not path.is_file()]
        if missing_val:
            raise FileNotFoundError(
                f"Validation files missing for {class_name}: {missing_val[:3]}"
            )
        class_train_paths[class_name] = train_paths
        print(f"{class_name}: train={len(train_paths)} val={len(val_paths)}")

    if args.dry_run:
        print("Dry run complete: datasets and validation split are ready.")
        return

    results: dict[str, object] = {
        "classes": list(CLASSES),
        "training_source": "backend/ml/augmented/<class_name>/*.png",
        "validation_source": "backend/ml/saved_models/val_split.json",
        "per_class": {},
    }
    histories: dict[str, dict[str, list[float]]] = {}

    for class_name in CLASSES:
        print(f"\nTraining autoencoder for {class_name}")
        class_result = train_one_autoencoder(
            class_name,
            class_train_paths[class_name],
            val_split[class_name],
            saved_models_root,
            args,
            device,
        )
        histories[class_name] = class_result["history"]
        results["per_class"][class_name] = class_result

    curves_path = saved_models_root / "autoencoder_training_curves.png"
    metrics_path = saved_models_root / "autoencoder_metrics.json"
    save_loss_curves(histories, curves_path)
    results["training_curves"] = str(curves_path.resolve())

    with metrics_path.open("w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=2)

    print("\nFinal Autoencoder Validation Reconstruction Error")
    for class_name in CLASSES:
        class_result = results["per_class"][class_name]
        print(
            f"{class_name}: "
            f"mean_mse={class_result['val_error_mean']:.6f} "
            f"std={class_result['val_error_std']:.6f} "
            f"max={class_result['val_error_max']:.6f}"
        )
    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved curves: {curves_path}")


if __name__ == "__main__":
    main()
