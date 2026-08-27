"""Unified ML inference pipeline for practice attempts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ml.feedback.grid_feedback import DEFAULT_INK_THRESHOLD, build_region_feedback
from ml.feedback.structural_part_feedback import (
    StructuralPartTemplate,
    VALIDATED_STRUCTURAL_CLASSES,
    build_structural_part_feedback,
    load_structural_part_template,
)
from ml.feedback.template_feedback import StrokeTemplate, build_template_feedback, load_template
from ml.preprocessing.normalize import apply_fixed_transform
from ml.training.autoencoder import RanjanaAutoencoder
from ml.training.dataset import CLASSES as VALIDATED_CLASSES
from ml.training.model import RanjanaRecognizerCNN


CANVAS_SIZE = 128
VALIDATED_CLASS_SET = frozenset(VALIDATED_CLASSES)
STRUCTURAL_CLASS_SET = frozenset(VALIDATED_STRUCTURAL_CLASSES)
MIN_NORMALIZED_INK_PIXELS = 250
STRUCTURAL_WRONG_CLASS_MIN_BEST_SCORE = 60.0
STRUCTURAL_WRONG_CLASS_MIN_SCORE_GAP = 15.0
INSUFFICIENT_INPUT_MESSAGE = "Insufficient input — please draw the full character."
WRONG_CHARACTER_MESSAGE_TEMPLATE = (
    "This attempt looks like {predicted_class}, not {target_class}. "
    "Please choose the matching character or redraw the selected one."
)


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


def normalized_ink_pixel_count(
    normalized: np.ndarray,
    ink_threshold: float = DEFAULT_INK_THRESHOLD,
) -> int:
    return int(np.count_nonzero(np.asarray(normalized, dtype=np.float32) > ink_threshold))


def has_enough_ink(normalized: np.ndarray) -> bool:
    return normalized_ink_pixel_count(normalized) >= MIN_NORMALIZED_INK_PIXELS


def insufficient_input_feedback(
    class_name: str,
    normalized: np.ndarray,
    rows: int = 3,
    cols: int = 3,
) -> dict[str, Any]:
    ink_pixel_count = normalized_ink_pixel_count(normalized)
    return {
        "class_name": class_name,
        "grid": {"rows": rows, "cols": cols},
        "overall_score": 0.0,
        "mean_error": 0.0,
        "std_error": 0.0,
        "max_region_error": 0.0,
        "threshold": 0.0,
        "mean_z_score": 0.0,
        "std_z_score": 0.0,
        "max_z_score": 0.0,
        "insufficient_input": True,
        "message": INSUFFICIENT_INPUT_MESSAGE,
        "warning": INSUFFICIENT_INPUT_MESSAGE,
        "ink_pixel_count": ink_pixel_count,
        "min_required_ink_pixels": MIN_NORMALIZED_INK_PIXELS,
        "threshold_settings": {
            "ink_threshold": DEFAULT_INK_THRESHOLD,
            "min_required_ink_pixels": MIN_NORMALIZED_INK_PIXELS,
        },
        "problem_regions": [],
        "all_regions": [],
        "fine_grid": {
            "rows": rows,
            "cols": cols,
            "problem_regions": [],
            "all_regions": [],
            "insufficient_input": True,
        },
        "broad_bands": {
            "bands": ["top", "middle", "bottom"],
            "problem_regions": [],
            "all_regions": [],
            "insufficient_input": True,
        },
    }


def wrong_character_feedback(
    class_name: str,
    recognizer_result: RecognizerResult,
    rows: int = 3,
    cols: int = 3,
    model_route: str = "validated_5_class",
) -> dict[str, Any]:
    message = WRONG_CHARACTER_MESSAGE_TEMPLATE.format(
        predicted_class=recognizer_result.predicted_class,
        target_class=class_name,
    )
    return {
        "class_name": class_name,
        "target_class": class_name,
        "predicted_class": recognizer_result.predicted_class,
        "recognizer_confidence": recognizer_result.confidence,
        "grid": {"rows": rows, "cols": cols},
        "overall_score": 0.0,
        "mean_error": 0.0,
        "std_error": 0.0,
        "max_region_error": 0.0,
        "threshold": 0.0,
        "mean_z_score": 0.0,
        "std_z_score": 0.0,
        "max_z_score": 0.0,
        "wrong_character": True,
        "message": message,
        "warning": message,
        "threshold_settings": {
            "wrong_character_blocking": True,
        },
        "problem_regions": [],
        "all_regions": [],
        "fine_grid": {
            "rows": rows,
            "cols": cols,
            "problem_regions": [],
            "all_regions": [],
            "wrong_character": True,
        },
        "broad_bands": {
            "bands": ["top", "middle", "bottom"],
            "problem_regions": [],
            "all_regions": [],
            "wrong_character": True,
        },
        "recognizer": {
            "predicted_class": recognizer_result.predicted_class,
            "confidence": recognizer_result.confidence,
            "probabilities": recognizer_result.probabilities,
            "matches_target": False,
            "model_route": model_route,
            "blocked_scoring": True,
            "warning": message,
        },
    }


@lru_cache(maxsize=1)
def load_alignment_transforms() -> dict[str, dict[str, Any]]:
    transforms_path = ml_root() / "processed" / "alignment_transforms.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(f"Missing alignment transforms: {transforms_path}")

    with transforms_path.open("r", encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=1)
def load_general_alignment_transforms() -> dict[str, dict[str, Any]]:
    transforms_path = ml_root() / "processed_general" / "alignment_transforms.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(f"Missing general alignment transforms: {transforms_path}")

    with transforms_path.open("r", encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=1)
def load_controlled_alignment_transforms() -> dict[str, dict[str, Any]]:
    transforms_path = ml_root() / "saved_models" / "structural_part_masks" / "controlled_alignment_transforms.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(f"Missing controlled structural alignment transforms: {transforms_path}")

    with transforms_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_attempt_for_class(
    image: np.ndarray,
    target_class: str,
    canvas_size: int = CANVAS_SIZE,
) -> np.ndarray:
    transforms = load_alignment_transforms()
    if target_class not in transforms:
        raise ValueError(f"Missing alignment transform for class: {target_class}")
    return apply_fixed_transform(image, transforms[target_class], canvas_size=canvas_size)


def normalize_controlled_structural_attempt_for_class(
    image: np.ndarray,
    target_class: str,
    canvas_size: int = CANVAS_SIZE,
) -> np.ndarray:
    transforms = load_controlled_alignment_transforms()
    if target_class not in transforms:
        raise ValueError(f"Missing controlled structural alignment transform for class: {target_class}")
    return apply_fixed_transform(image, transforms[target_class], canvas_size=canvas_size)


def normalize_general_attempt_for_class(
    image: np.ndarray,
    target_class: str,
    canvas_size: int = CANVAS_SIZE,
) -> np.ndarray:
    transforms = load_general_alignment_transforms()
    if target_class not in transforms:
        raise ValueError(f"Missing general alignment transform for class: {target_class}")
    return apply_fixed_transform(image, transforms[target_class], canvas_size=canvas_size)


@lru_cache(maxsize=1)
def load_recognizer(device_name: str = "cpu") -> tuple[RanjanaRecognizerCNN, list[str]]:
    device = torch.device(device_name)
    checkpoint_path = ml_root() / "saved_models" / "recognizer_best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing recognizer checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint.get("classes", list(VALIDATED_CLASSES))
    model = RanjanaRecognizerCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes


@lru_cache(maxsize=1)
def load_general_recognizer(device_name: str = "cpu") -> tuple[RanjanaRecognizerCNN, list[str]]:
    device = torch.device(device_name)
    checkpoint_path = ml_root() / "saved_models" / "recognizer_general_best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing general recognizer checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint["classes"]
    model = RanjanaRecognizerCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, classes


@lru_cache(maxsize=len(VALIDATED_CLASSES))
def load_autoencoder(class_name: str, device_name: str = "cpu") -> RanjanaAutoencoder:
    if class_name not in VALIDATED_CLASSES:
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


@lru_cache(maxsize=128)
def load_general_autoencoder(class_name: str, device_name: str = "cpu") -> RanjanaAutoencoder:
    device = torch.device(device_name)
    checkpoint_path = ml_root() / "saved_models" / "autoencoders_general" / f"autoencoder_{class_name}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing general autoencoder checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = RanjanaAutoencoder().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@lru_cache(maxsize=128)
def load_general_region_baseline(class_name: str) -> dict[str, Any]:
    baseline_path = ml_root() / "saved_models" / "autoencoders_general" / f"region_baseline_{class_name}.json"
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Missing general region baseline: {baseline_path}")

    with baseline_path.open("r", encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=128)
def load_stroke_template(class_name: str) -> StrokeTemplate:
    template_path = ml_root() / "saved_models" / "stroke_templates" / f"{class_name}.npz"
    return load_template(template_path)


@lru_cache(maxsize=128)
def load_validated_structural_part_template(class_name: str) -> StructuralPartTemplate:
    return load_structural_part_template(class_name)


def merge_template_and_autoencoder_feedback(
    template_feedback: dict[str, Any],
    autoencoder_feedback: dict[str, Any],
) -> dict[str, Any]:
    """Return template feedback as primary, preserving reconstruction output."""

    merged = dict(template_feedback)
    merged["autoencoder_feedback"] = autoencoder_feedback
    merged["autoencoder_overall_score"] = autoencoder_feedback.get("overall_score")
    merged["autoencoder_problem_regions"] = autoencoder_feedback.get("problem_regions", [])
    return merged


def merge_structural_template_and_autoencoder_feedback(
    structural_feedback: dict[str, Any],
    template_feedback: dict[str, Any],
    autoencoder_feedback: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(structural_feedback)
    merged["template_feedback"] = template_feedback
    merged["autoencoder_feedback"] = autoencoder_feedback
    merged["template_overall_score"] = template_feedback.get("overall_score")
    merged["autoencoder_overall_score"] = autoencoder_feedback.get("overall_score")
    return merged


def attach_recognizer_metadata(
    feedback: dict[str, Any],
    recognizer_result: RecognizerResult | None,
    target_class: str,
    model_route: str,
    blocked_scoring: bool = False,
) -> dict[str, Any]:
    if recognizer_result is None:
        feedback["recognizer"] = {
            "predicted_class": None,
            "confidence": 0.0,
            "probabilities": {},
            "matches_target": False,
            "model_route": model_route,
            "skipped": True,
            "blocked_scoring": blocked_scoring,
        }
        feedback["predicted_class"] = None
        feedback["recognizer_confidence"] = 0.0
        return feedback

    matches_target = recognizer_result.predicted_class == target_class
    feedback["recognizer"] = {
        "predicted_class": recognizer_result.predicted_class,
        "confidence": recognizer_result.confidence,
        "probabilities": recognizer_result.probabilities,
        "matches_target": matches_target,
        "model_route": model_route,
        "blocked_scoring": blocked_scoring,
    }
    feedback["predicted_class"] = recognizer_result.predicted_class
    feedback["recognizer_confidence"] = recognizer_result.confidence
    return feedback


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


def recognize_general(normalized: np.ndarray, device_name: str = "cpu") -> RecognizerResult:
    device = torch.device(device_name)
    model, classes = load_general_recognizer(device_name)
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

    autoencoder_feedback = build_region_feedback(
        class_name=target_class,
        input_image=normalized,
        reconstruction=reconstruction,
        rows=rows,
        cols=cols,
        max_regions=3,
        min_problem_region_error=0.012,
    )
    template = load_stroke_template(target_class)
    template_feedback = build_template_feedback(
        class_name=target_class,
        input_image=normalized,
        template=template,
        rows=rows,
        cols=cols,
        max_regions=3,
    )
    structural_template = load_validated_structural_part_template(target_class)
    structural_feedback = build_structural_part_feedback(
        class_name=target_class,
        input_image=normalized,
        template=structural_template,
        rows=rows,
        cols=cols,
        max_regions=3,
    )
    return merge_structural_template_and_autoencoder_feedback(
        structural_feedback,
        template_feedback,
        autoencoder_feedback,
    )


def reconstruct_and_feedback_general(
    normalized: np.ndarray,
    target_class: str,
    rows: int = 3,
    cols: int = 3,
    device_name: str = "cpu",
) -> dict[str, Any]:
    device = torch.device(device_name)
    model = load_general_autoencoder(target_class, device_name)
    baseline = load_general_region_baseline(target_class)
    image_tensor = tensor_from_normalized(normalized).to(device)

    with torch.no_grad():
        reconstruction = model(image_tensor).cpu().squeeze(0).squeeze(0).numpy()

    autoencoder_feedback = build_region_feedback(
        class_name=target_class,
        input_image=normalized,
        reconstruction=reconstruction,
        rows=rows,
        cols=cols,
        max_regions=3,
        min_problem_region_error=0.012,
        baseline=baseline,
    )
    template = load_stroke_template(target_class)
    template_feedback = build_template_feedback(
        class_name=target_class,
        input_image=normalized,
        template=template,
        rows=rows,
        cols=cols,
        max_regions=3,
    )
    return merge_template_and_autoencoder_feedback(template_feedback, autoencoder_feedback)


def structural_feedback_for_selected_class(
    normalized: np.ndarray,
    target_class: str,
    rows: int = 3,
    cols: int = 3,
) -> dict[str, Any]:
    structural_template = load_validated_structural_part_template(target_class)
    return build_structural_part_feedback(
        class_name=target_class,
        input_image=normalized,
        template=structural_template,
        rows=rows,
        cols=cols,
        max_regions=3,
    )


def structural_match_scores(
    decoded_image: np.ndarray,
    selected_class: str,
    selected_normalized: np.ndarray,
    selected_feedback: dict[str, Any],
    rows: int = 3,
    cols: int = 3,
) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    for class_name in VALIDATED_STRUCTURAL_CLASSES:
        if class_name == selected_class:
            feedback = selected_feedback
        else:
            normalized = normalize_controlled_structural_attempt_for_class(
                decoded_image,
                class_name,
                canvas_size=CANVAS_SIZE,
            )
            if not has_enough_ink(normalized):
                score = 0.0
                feedback_method = "insufficient_input"
            else:
                feedback = structural_feedback_for_selected_class(normalized, class_name, rows, cols)
                score = float(feedback.get("overall_score", 0.0))
                feedback_method = str(feedback.get("feedback_method", "structural_part_mask"))
            scores.append(
                {
                    "class_name": class_name,
                    "score": score,
                    "feedback_method": feedback_method,
                }
            )
            continue

        scores.append(
            {
                "class_name": class_name,
                "score": float(feedback.get("overall_score", 0.0)),
                "feedback_method": str(feedback.get("feedback_method", "structural_part_mask")),
            }
        )

    return sorted(scores, key=lambda item: float(item["score"]), reverse=True)


def should_block_structural_wrong_character(
    match_scores: list[dict[str, Any]],
    target_class: str,
) -> tuple[bool, dict[str, Any] | None, float]:
    if not match_scores:
        return False, None, 0.0

    selected = next((item for item in match_scores if item["class_name"] == target_class), None)
    best = match_scores[0]
    if selected is None or best["class_name"] == target_class:
        return False, best, 0.0

    selected_score = float(selected["score"])
    best_score = float(best["score"])
    score_gap = best_score - selected_score
    should_block = (
        best_score >= STRUCTURAL_WRONG_CLASS_MIN_BEST_SCORE
        and score_gap >= STRUCTURAL_WRONG_CLASS_MIN_SCORE_GAP
    )
    return should_block, best, score_gap


def analyze_attempt(
    image_bytes: bytes,
    target_class: str,
    rows: int = 3,
    cols: int = 3,
    device_name: str = "cpu",
) -> dict[str, Any]:
    decoded = decode_upload_image(image_bytes)
    if target_class in STRUCTURAL_CLASS_SET:
        normalized = normalize_controlled_structural_attempt_for_class(decoded, target_class, canvas_size=CANVAS_SIZE)
        model_route = "controlled_structural_15_class"
        if not has_enough_ink(normalized):
            feedback = insufficient_input_feedback(target_class, normalized, rows, cols)
            attach_recognizer_metadata(feedback, None, target_class, model_route)
            return {"normalized": normalized, "feedback": feedback}
        feedback = structural_feedback_for_selected_class(normalized, target_class, rows, cols)
        match_scores = structural_match_scores(decoded, target_class, normalized, feedback, rows, cols)
        should_block, best_match, score_gap = should_block_structural_wrong_character(match_scores, target_class)
        if should_block and best_match is not None:
            structural_result = RecognizerResult(
                predicted_class=str(best_match["class_name"]),
                confidence=float(best_match["score"]) / 100.0,
                probabilities={
                    str(item["class_name"]): float(item["score"]) / 100.0
                    for item in match_scores
                },
            )
            blocked_feedback = wrong_character_feedback(target_class, structural_result, rows, cols, model_route)
            blocked_feedback["structural_match"] = {
                "selected_class": target_class,
                "selected_score": next(
                    float(item["score"]) for item in match_scores if item["class_name"] == target_class
                ),
                "best_class": best_match["class_name"],
                "best_score": float(best_match["score"]),
                "score_gap": float(score_gap),
                "all_scores": match_scores,
                "min_best_score": STRUCTURAL_WRONG_CLASS_MIN_BEST_SCORE,
                "min_score_gap": STRUCTURAL_WRONG_CLASS_MIN_SCORE_GAP,
            }
            return {"normalized": normalized, "feedback": blocked_feedback}
        try:
            recognizer_result = (
                recognize(normalized, device_name)
                if target_class in VALIDATED_CLASS_SET
                else recognize_general(normalized, device_name)
            )
        except Exception:
            recognizer_result = None
        feedback["target_class_structural_scoring"] = True
        feedback["structural_match"] = {
            "selected_class": target_class,
            "selected_score": next(
                float(item["score"]) for item in match_scores if item["class_name"] == target_class
            ),
            "best_class": match_scores[0]["class_name"] if match_scores else target_class,
            "best_score": float(match_scores[0]["score"]) if match_scores else float(feedback.get("overall_score", 0.0)),
            "score_gap": float(score_gap),
            "all_scores": match_scores,
            "min_best_score": STRUCTURAL_WRONG_CLASS_MIN_BEST_SCORE,
            "min_score_gap": STRUCTURAL_WRONG_CLASS_MIN_SCORE_GAP,
        }
        feedback["recognizer_used_as_gate"] = False
        attach_recognizer_metadata(feedback, recognizer_result, target_class, model_route, blocked_scoring=False)
    elif target_class in VALIDATED_CLASS_SET:
        normalized = normalize_attempt_for_class(decoded, target_class, canvas_size=CANVAS_SIZE)
        model_route = "validated_5_class"
        if not has_enough_ink(normalized):
            feedback = insufficient_input_feedback(target_class, normalized, rows, cols)
            attach_recognizer_metadata(feedback, None, target_class, model_route)
            return {"normalized": normalized, "feedback": feedback}
        recognizer_result = recognize(normalized, device_name)
        if recognizer_result.predicted_class != target_class:
            feedback = wrong_character_feedback(target_class, recognizer_result, rows, cols, model_route)
            return {"normalized": normalized, "feedback": feedback}
        feedback = reconstruct_and_feedback(normalized, target_class, rows, cols, device_name)
    else:
        normalized = normalize_general_attempt_for_class(decoded, target_class, canvas_size=CANVAS_SIZE)
        model_route = "general_62_class"
        if not has_enough_ink(normalized):
            feedback = insufficient_input_feedback(target_class, normalized, rows, cols)
            attach_recognizer_metadata(feedback, None, target_class, model_route)
            return {"normalized": normalized, "feedback": feedback}
        recognizer_result = recognize_general(normalized, device_name)
        if recognizer_result.predicted_class != target_class:
            feedback = wrong_character_feedback(target_class, recognizer_result, rows, cols, model_route)
            return {"normalized": normalized, "feedback": feedback}
        feedback = reconstruct_and_feedback_general(normalized, target_class, rows, cols, device_name)

    attach_recognizer_metadata(feedback, recognizer_result, target_class, model_route)
    if (
        recognizer_result is not None
        and model_route != "controlled_structural_15_class"
        and not feedback["recognizer"]["matches_target"]
    ):
        warning = f"Attempt resembles {recognizer_result.predicted_class}, not {target_class}."
        feedback["recognizer"]["warning"] = warning
        feedback["warning"] = warning
    return {"normalized": normalized, "feedback": feedback}
