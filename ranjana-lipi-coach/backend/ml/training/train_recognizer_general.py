#!/usr/bin/env python3
"""Train a 62-class recognizer on processed_general and augmented_general data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import RanjanaRecognizerCNN


SPLIT_SEED = 42
ORIGINAL_TRAIN_FRACTION = 0.85
LOW_DATA_CLASSES = {"lu", "luu", "rii"}


class ImagePathDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("L")
            array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0), torch.tensor(label, dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the 62-class general recognizer")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--dry-run", action="store_true")
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


def class_names(processed_root: Path) -> list[str]:
    return sorted(
        path.name
        for path in processed_root.iterdir()
        if path.is_dir() and path.name != "references"
    )


def split_originals(paths: list[Path], seed: int) -> tuple[list[Path], list[Path]]:
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    train_count = int(len(shuffled) * ORIGINAL_TRAIN_FRACTION)
    return sorted(shuffled[:train_count]), sorted(shuffled[train_count:])


def build_splits(
    ml_root: Path,
    classes: list[str],
    seed: int,
    save_path: Path,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], dict[str, object]]:
    processed_root = ml_root / "processed_general"
    augmented_root = ml_root / "augmented_general"
    class_to_idx = {class_name: index for index, class_name in enumerate(classes)}
    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []
    split: dict[str, object] = {
        "classes": classes,
        "class_to_idx": class_to_idx,
        "seed": seed,
        "original_train_fraction": ORIGINAL_TRAIN_FRACTION,
        "validation_files": {},
    }

    for class_name in classes:
        label = class_to_idx[class_name]
        original_paths = png_files(processed_root / class_name)
        augmented_paths = png_files(augmented_root / class_name)
        if not original_paths:
            raise FileNotFoundError(f"No processed_general images found for {class_name}")
        if not augmented_paths:
            raise FileNotFoundError(f"No augmented_general images found for {class_name}")

        original_train, original_val = split_originals(original_paths, seed + label)
        train_samples.extend((path, label) for path in original_train)
        train_samples.extend((path, label) for path in augmented_paths)
        val_samples.extend((path, label) for path in original_val)
        split["validation_files"][class_name] = [str(path.resolve()) for path in original_val]

    save_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return train_samples, val_samples, split


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


def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(labels.tolist())
    return y_true, y_pred


def confusion_matrix(y_true: list[int], y_pred: list[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred):
        matrix[actual, predicted] += 1
    return matrix


def compute_metrics(matrix: np.ndarray, classes: list[str]) -> dict[str, object]:
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    per_class: dict[str, dict[str, float | int]] = {}

    for index, class_name in enumerate(classes):
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


def save_confusion_matrix(matrix: np.ndarray, classes: list[str], output_path: Path) -> None:
    cell = 26
    label_w = 120
    title_h = 60
    width = label_w + cell * len(classes) + 20
    height = title_h + label_w + cell * len(classes)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _font(9)
    title_font = _font(18)
    max_value = int(matrix.max()) if matrix.size else 1

    draw.text((30, 18), "General Recognizer Confusion Matrix", fill="black", font=title_font)
    for col, class_name in enumerate(classes):
        x = label_w + col * cell
        draw.text((x + 2, title_h + 34), class_name[:4], fill="black", font=font)
    for row, class_name in enumerate(classes):
        y = title_h + label_w + row * cell
        draw.text((28, y + 8), class_name[:12], fill="black", font=font)

    for row in range(len(classes)):
        for col in range(len(classes)):
            value = int(matrix[row, col])
            intensity = int(255 - (value / max(1, max_value)) * 180)
            color = (intensity, intensity, 255)
            x0 = label_w + col * cell
            y0 = title_h + label_w + row * cell
            draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=color, outline=(120, 120, 120))
            if value:
                draw.text((x0 + 8, y0 + 8), str(value), fill="black", font=font)

    image.save(output_path)


def save_checkpoint(
    model: nn.Module,
    output_path: Path,
    classes: list[str],
    epoch: int,
    val_accuracy: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": classes,
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
    processed_root = ml_root / "processed_general"
    saved_models_root = ml_root / "saved_models"
    saved_models_root.mkdir(parents=True, exist_ok=True)
    classes = class_names(processed_root)

    if len(classes) != 62:
        raise RuntimeError(f"Expected 62 processed_general classes, found {len(classes)}")

    split_path = saved_models_root / "val_split_general.json"
    train_samples, val_samples, _split = build_splits(ml_root, classes, args.seed, split_path)

    print(f"Device: {device}")
    print(f"Classes: {len(classes)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Validation split: {split_path}")
    print("Low-data classes included: lu, luu, rii")

    if args.dry_run:
        print("Dry run complete: general recognizer data is ready.")
        return

    train_loader = DataLoader(
        ImagePathDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        ImagePathDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = RanjanaRecognizerCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    checkpoint_path = saved_models_root / "recognizer_general_best.pt"
    best_accuracy = -1.0
    epochs_without_improvement = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
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
            save_checkpoint(model, checkpoint_path, classes, epoch, val_acc, args)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    y_true, y_pred = collect_predictions(model, val_loader, device)
    matrix = confusion_matrix(y_true, y_pred, len(classes))
    metrics = compute_metrics(matrix, classes)
    metrics["classes"] = classes
    metrics["confusion_matrix"] = matrix.tolist()
    metrics["history"] = history
    metrics["best_checkpoint"] = str(checkpoint_path.resolve())

    confusion_path = saved_models_root / "recognizer_general_confusion_matrix.png"
    metrics_path = saved_models_root / "recognizer_general_metrics.json"
    save_confusion_matrix(matrix, classes, confusion_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nFinal General Recognizer Validation Metrics")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nLow-data class metrics")
    for class_name in sorted(LOW_DATA_CLASSES):
        class_metrics = metrics["per_class"][class_name]
        print(
            f"{class_name}: precision={class_metrics['precision']:.4f} "
            f"recall={class_metrics['recall']:.4f} "
            f"f1={class_metrics['f1']:.4f} "
            f"support={class_metrics['support']}"
        )
    print(f"\nSaved checkpoint: {checkpoint_path}")
    print(f"Saved confusion matrix: {confusion_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
