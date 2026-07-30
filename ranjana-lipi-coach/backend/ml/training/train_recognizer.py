#!/usr/bin/env python3
"""Train Model 1: a 5-class Ranjana Lipi recognizer."""

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
from torch.utils.data import DataLoader

try:
    from .dataset import CLASSES, ImagePathDataset, build_recognizer_splits
    from .model import RanjanaRecognizerCNN
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from dataset import CLASSES, ImagePathDataset, build_recognizer_splits
    from model import RanjanaRecognizerCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the 5-class Ranjana recognizer")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda", "mps"),
        help="Default is CPU so the script works on machines without a GPU.",
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


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += batch_size

    return total_loss / total_seen, total_correct / total_seen


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(predictions)
            y_true.extend(labels.tolist())
    return y_true, y_pred


def confusion_matrix(y_true: list[int], y_pred: list[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred):
        matrix[actual, predicted] += 1
    return matrix


def compute_metrics(matrix: np.ndarray) -> dict[str, object]:
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    per_class: dict[str, dict[str, float | int]] = {}

    for index, class_name in enumerate(CLASSES):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index, :].sum() - true_positive)
        support = int(matrix[index, :].sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return {"accuracy": accuracy, "per_class": per_class}


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def save_curves(history: dict[str, list[float]], output_path: Path) -> None:
    width, height = 1000, 620
    margin_left, margin_right = 80, 40
    margin_top, margin_bottom = 50, 80
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _font(16)
    title_font = _font(22)

    draw.text((margin_left, 16), "Recognizer Training Curves", fill="black", font=title_font)
    draw.rectangle(
        [margin_left, margin_top, margin_left + plot_w, margin_top + plot_h],
        outline=(40, 40, 40),
    )

    epochs = list(range(1, len(history["train_loss"]) + 1))
    if not epochs:
        image.save(output_path)
        return

    all_values = (
        history["train_loss"]
        + history["val_loss"]
        + history["train_acc"]
        + history["val_acc"]
    )
    y_min = 0.0
    y_max = max(1.0, max(all_values))

    def point(epoch: int, value: float) -> tuple[float, float]:
        x = margin_left + ((epoch - 1) / max(1, len(epochs) - 1)) * plot_w
        y = margin_top + plot_h - ((value - y_min) / (y_max - y_min)) * plot_h
        return x, y

    series = [
        ("train_loss", (215, 48, 39)),
        ("val_loss", (142, 68, 173)),
        ("train_acc", (39, 124, 179)),
        ("val_acc", (35, 155, 86)),
    ]
    for name, color in series:
        points = [point(epoch, value) for epoch, value in zip(epochs, history[name])]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)
        else:
            draw.line(points, fill=color, width=3)

    legend_x = margin_left + 16
    legend_y = margin_top + 16
    for offset, (name, color) in enumerate(series):
        y = legend_y + offset * 24
        draw.rectangle([legend_x, y + 4, legend_x + 20, y + 14], fill=color)
        draw.text((legend_x + 28, y), name, fill="black", font=font)

    draw.text((margin_left, height - 48), "Epoch", fill="black", font=font)
    draw.text((16, margin_top + plot_h // 2), "Value", fill="black", font=font)
    image.save(output_path)


def save_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    cell = 90
    label_w = 120
    title_h = 70
    width = label_w + cell * len(CLASSES) + 30
    height = title_h + label_w + cell * len(CLASSES)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _font(16)
    title_font = _font(22)
    max_value = int(matrix.max()) if matrix.size else 1

    draw.text((30, 20), "Recognizer Confusion Matrix", fill="black", font=title_font)
    draw.text((label_w + 80, 52), "Predicted", fill="black", font=font)
    draw.text((16, title_h + label_w + 120), "Actual", fill="black", font=font)

    for col, class_name in enumerate(CLASSES):
        x = label_w + col * cell
        draw.text((x + 28, title_h + 55), class_name, fill="black", font=font)
    for row, class_name in enumerate(CLASSES):
        y = title_h + label_w + row * cell
        draw.text((34, y + 34), class_name, fill="black", font=font)

    for row in range(len(CLASSES)):
        for col in range(len(CLASSES)):
            value = int(matrix[row, col])
            intensity = int(255 - (value / max(1, max_value)) * 180)
            color = (intensity, intensity, 255)
            x0 = label_w + col * cell
            y0 = title_h + label_w + row * cell
            draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=color, outline=(80, 80, 80))
            draw.text((x0 + 34, y0 + 34), str(value), fill="black", font=font)

    image.save(output_path)


def save_checkpoint(
    model: nn.Module,
    output_path: Path,
    epoch: int,
    val_accuracy: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": list(CLASSES),
            "epoch": epoch,
            "val_accuracy": val_accuracy,
            "args": vars(args),
        },
        output_path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    root = project_root()
    ml_root = root / "ranjana-lipi-coach" / "backend" / "ml"
    saved_models_root = ml_root / "saved_models"
    saved_models_root.mkdir(parents=True, exist_ok=True)

    train_samples, val_samples, _ = build_recognizer_splits(
        ml_root=ml_root,
        seed=args.seed,
        save_val_split=True,
    )
    train_dataset = ImagePathDataset(train_samples)
    val_dataset = ImagePathDataset(val_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = RanjanaRecognizerCNN(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_accuracy = -1.0
    epochs_without_improvement = 0
    checkpoint_path = saved_models_root / "recognizer_best.pt"
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes: {', '.join(CLASSES)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            epochs_without_improvement = 0
            save_checkpoint(model, checkpoint_path, epoch, val_acc, args)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    curves_path = saved_models_root / "recognizer_training_curves.png"
    save_curves(history, curves_path)

    y_true, y_pred = collect_predictions(model, val_loader, device)
    matrix = confusion_matrix(y_true, y_pred, len(CLASSES))
    metrics = compute_metrics(matrix)
    metrics["classes"] = list(CLASSES)
    metrics["confusion_matrix"] = matrix.tolist()
    metrics["best_checkpoint"] = str(checkpoint_path.resolve())

    confusion_path = saved_models_root / "recognizer_confusion_matrix.png"
    metrics_path = saved_models_root / "recognizer_metrics.json"
    save_confusion_matrix(matrix, confusion_path)
    with metrics_path.open("w", encoding="utf-8") as json_file:
        json.dump(metrics, json_file, indent=2)

    print("\nFinal Validation Metrics")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    for class_name, class_metrics in metrics["per_class"].items():
        print(
            f"{class_name}: "
            f"precision={class_metrics['precision']:.4f} "
            f"recall={class_metrics['recall']:.4f} "
            f"f1={class_metrics['f1']:.4f} "
            f"support={class_metrics['support']}"
        )
    print(f"\nSaved checkpoint: {checkpoint_path}")
    print(f"Saved validation split: {saved_models_root / 'val_split.json'}")
    print(f"Saved curves: {curves_path}")
    print(f"Saved confusion matrix: {confusion_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
