"""Dataset and split helpers for the 5-class Ranjana Lipi recognizer."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CLASSES = ("aa", "a", "ka", "da", "dda")
CLASS_TO_IDX = {class_name: index for index, class_name in enumerate(CLASSES)}
SPLIT_SEED = 42
ORIGINAL_TRAIN_FRACTION = 0.85


class ImagePathDataset(Dataset):
    """Load grayscale image tensors from file paths and integer labels."""

    def __init__(self, samples: list[tuple[str | Path, int]]) -> None:
        self.samples = [(Path(path), int(label)) for path, label in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("L")
            array = np.asarray(image, dtype=np.float32) / 255.0

        tensor = torch.from_numpy(array).unsqueeze(0)
        return tensor, torch.tensor(label, dtype=torch.long)


def _png_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.glob("*.png") if path.is_file())


def _split_originals(paths: list[Path], seed: int) -> tuple[list[Path], list[Path]]:
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    train_count = int(len(shuffled) * ORIGINAL_TRAIN_FRACTION)
    return sorted(shuffled[:train_count]), sorted(shuffled[train_count:])


def build_recognizer_splits(
    ml_root: Path,
    seed: int = SPLIT_SEED,
    save_val_split: bool = True,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], dict[str, object]]:
    """Create reproducible train/validation samples for Model 1.

    Training uses 85% of original processed images plus all generated augmented
    images. Validation uses only the held-out 15% original processed images.
    """

    processed_root = ml_root / "processed"
    augmented_root = ml_root / "augmented"
    saved_models_root = ml_root / "saved_models"

    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []
    val_split: dict[str, object] = {
        "classes": list(CLASSES),
        "class_to_idx": CLASS_TO_IDX,
        "seed": seed,
        "original_train_fraction": ORIGINAL_TRAIN_FRACTION,
        "validation_files": {},
    }

    for class_name in CLASSES:
        label = CLASS_TO_IDX[class_name]
        original_paths = _png_files(processed_root / class_name)
        augmented_paths = _png_files(augmented_root / class_name)
        if not original_paths:
            raise FileNotFoundError(f"No processed images found for {class_name}")
        if not augmented_paths:
            raise FileNotFoundError(f"No augmented images found for {class_name}")

        original_train, original_val = _split_originals(
            original_paths,
            seed + label,
        )
        train_samples.extend((path, label) for path in original_train)
        train_samples.extend((path, label) for path in augmented_paths)
        val_samples.extend((path, label) for path in original_val)
        val_split["validation_files"][class_name] = [str(path.resolve()) for path in original_val]

    if save_val_split:
        saved_models_root.mkdir(parents=True, exist_ok=True)
        split_path = saved_models_root / "val_split.json"
        with split_path.open("w", encoding="utf-8") as json_file:
            json.dump(val_split, json_file, indent=2)

    return train_samples, val_samples, val_split
