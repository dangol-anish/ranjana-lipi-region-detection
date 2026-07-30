"""Unified ML inference pipeline for practice attempts."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ml.feedback.grid_feedback import build_region_feedback
from ml.preprocessing.normalize import normalize_image
from ml.training.autoencoder import RanjanaAutoencoder
from ml.training.dataset import CLASSES
from ml.training.model import RanjanaRecognizerCNN


CANVAS_SIZE = 128
RECOGNIZER_MISMATCH_SCORE_CAP = 65.0


@dataclass(frozen=True)
class RecognizerResult:
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]


def backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ml_root() -> Path:
    return backend_root() / "ml"


def decode_upload_image(image_bytes: bytes) -> np.ndarray:
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Could not decode uploaded image")
    return decoded


def tensor_from_normalized(normalized: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0).unsqueeze(0)


@lru_cache(maxsize=1)
def load_recognizer(device_name: str = "cpu") -> tuple[RanjanaRecognizerCNN, list[str]]:
    device = torch.device(device_name)
    checkpoint_path = ml_root() / "saved_models" / "recognizer_best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing recognizer checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint.get("classes", list(CLASSES))
    model = RanjanaRecognizerCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes


@lru_cache(maxsize=len(CLASSES))
def load_autoencoder(class_name: str, device_name: str = "cpu") -> RanjanaAutoencoder:
    if class_name not in CLASSES:
        raise ValueError(f"Unsupported class: {class_name}")

    device = torch.device(device_name)
    checkpoint_path = ml_root() / "saved_models" / f"autoencoder_{class_name}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing autoencoder checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = RanjanaAutoencoder().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def recognize(normalized: np.ndarray, device_name: str = "cpu") -> RecognizerResult:
    device = torch.device(device_name)
    model, classes = load_recognizer(device_name)
    image_tensor = tensor_from_normalized(normalized).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities_tensor = torch.softmax(logits, dim=1).cpu().squeeze(0)

    probabilities = {
        class_name: float(probabilities_tensor[index])
        for index, class_name in enumerate(classes)
    }
    predicted_index = int(torch.argmax(probabilities_tensor).item())
    predicted_class = classes[predicted_index]
    return RecognizerResult(
        predicted_class=predicted_class,
        confidence=probabilities[predicted_class],
        probabilities=probabilities,
    )


def reconstruct_and_feedback(
    normalized: np.ndarray,
    target_class: str,
    rows: int = 3,
    cols: int = 3,
    device_name: str = "cpu",
) -> dict[str, Any]:
    device = torch.device(device_name)
    model = load_autoencoder(target_class, device_name)
    image_tensor = tensor_from_normalized(normalized).to(device)

    with torch.no_grad():
        reconstruction = model(image_tensor).cpu().squeeze(0).squeeze(0).numpy()

    return build_region_feedback(
        class_name=target_class,
        input_image=normalized,
        reconstruction=reconstruction,
        rows=rows,
        cols=cols,
        max_regions=3,
        min_problem_region_error=0.012,
    )


def analyze_attempt(
    image_bytes: bytes,
    target_class: str,
    rows: int = 3,
    cols: int = 3,
    device_name: str = "cpu",
) -> dict[str, Any]:
    decoded = decode_upload_image(image_bytes)
    normalized = normalize_image(decoded, canvas_size=CANVAS_SIZE)
    recognizer_result = recognize(normalized, device_name)
    feedback = reconstruct_and_feedback(normalized, target_class, rows, cols, device_name)

    feedback["recognizer"] = {
        "predicted_class": recognizer_result.predicted_class,
        "confidence": recognizer_result.confidence,
        "probabilities": recognizer_result.probabilities,
        "matches_target": recognizer_result.predicted_class == target_class,
    }
    if not feedback["recognizer"]["matches_target"]:
        feedback["recognizer"]["warning"] = (
            f"Attempt resembles {recognizer_result.predicted_class}, not {target_class}."
        )
        feedback["raw_autoencoder_score"] = feedback["overall_score"]
        feedback["overall_score"] = min(
            float(feedback["overall_score"]),
            RECOGNIZER_MISMATCH_SCORE_CAP,
        )
    return {"normalized": normalized, "feedback": feedback}
