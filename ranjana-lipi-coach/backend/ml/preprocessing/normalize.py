"""Normalize Ranjana Lipi handwriting images onto a fixed square canvas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class NormalizationError(ValueError):
    """Raised when an image cannot be normalized into a usable character crop."""


def _load_grayscale(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, (str, Path)):
        grayscale = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        if grayscale is None:
            raise NormalizationError(f"Could not read image: {image}")
        return grayscale

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a path-like value or a numpy array")

    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _binarize_ink(grayscale: np.ndarray) -> np.ndarray:
    if grayscale.dtype != np.uint8:
        grayscale = cv2.normalize(grayscale, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    background = cv2.GaussianBlur(grayscale, (0, 0), sigmaX=25, sigmaY=25)
    illumination_corrected = cv2.divide(grayscale, background, scale=255)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_boosted = clahe.apply(illumination_corrected)
    denoised = cv2.GaussianBlur(contrast_boosted, (3, 3), 0)

    adaptive_binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )
    _otsu_threshold, otsu_binary = cv2.threshold(
        denoised,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )

    adaptive_ink_ratio = cv2.countNonZero(adaptive_binary) / float(adaptive_binary.shape[0] * adaptive_binary.shape[1])
    if adaptive_ink_ratio < 0.001 or adaptive_ink_ratio > 0.90:
        binary = otsu_binary
    else:
        binary = adaptive_binary

    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = _remove_small_components(binary)
    return binary


def _remove_small_components(binary: np.ndarray) -> np.ndarray:
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = max(20, int(binary.shape[0] * binary.shape[1] * 0.00003))
    cleaned = np.zeros_like(binary)

    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label] = 255

    return cleaned


def _find_ink_bbox(binary: np.ndarray) -> tuple[int, int, int, int]:
    nonzero = cv2.findNonZero(binary)
    if nonzero is None:
        raise NormalizationError("No ink pixels found after thresholding")

    x, y, width, height = cv2.boundingRect(nonzero)
    ink_ratio = cv2.countNonZero(binary) / float(binary.shape[0] * binary.shape[1])
    if ink_ratio < 0.001:
        raise NormalizationError("Image is near-empty after thresholding")
    if ink_ratio > 0.95:
        raise NormalizationError("Image is almost fully foreground after thresholding")
    if width < 3 or height < 3:
        raise NormalizationError("Detected ink bounding box is too small")

    return x, y, width, height


def _resize_to_reference_shape(binary: np.ndarray, reference_shape: list[int]) -> np.ndarray:
    reference_height, reference_width = reference_shape
    if binary.shape[:2] == (reference_height, reference_width):
        return binary

    return cv2.resize(
        binary,
        (reference_width, reference_height),
        interpolation=cv2.INTER_AREA,
    )


def _crop_with_padding(binary: np.ndarray, bbox: dict[str, int]) -> np.ndarray:
    x = bbox["x"]
    y = bbox["y"]
    width = bbox["width"]
    height = bbox["height"]

    crop = np.zeros((height, width), dtype=np.uint8)
    source_height, source_width = binary.shape[:2]

    src_x_start = max(0, x)
    src_y_start = max(0, y)
    src_x_end = min(source_width, x + width)
    src_y_end = min(source_height, y + height)

    if src_x_start >= src_x_end or src_y_start >= src_y_end:
        raise NormalizationError("Reference crop does not overlap the input image")

    dst_x_start = src_x_start - x
    dst_y_start = src_y_start - y
    dst_x_end = dst_x_start + (src_x_end - src_x_start)
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    crop[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = binary[
        src_y_start:src_y_end,
        src_x_start:src_x_end,
    ]
    return crop


def compute_reference_transform(
    reference_image: str | Path | np.ndarray,
    canvas_size: int = 128,
) -> dict[str, Any]:
    """Compute fixed normalization parameters from a class reference image.

    The returned transform is intentionally anchored to the reference character's
    ink box. User attempts for the same class must reuse these parameters so that
    fixed grid cells keep the same anatomical meaning across correct and flawed
    samples.
    """

    if canvas_size <= 0:
        raise ValueError("canvas_size must be positive")

    grayscale = _load_grayscale(reference_image)
    binary = _binarize_ink(grayscale)
    x, y, width, height = _find_ink_bbox(binary)

    target_extent = max(1, int(canvas_size * 0.86))
    scale = target_extent / float(max(width, height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    x_offset = (canvas_size - resized_width) // 2
    y_offset = (canvas_size - resized_height) // 2

    return {
        "canvas_size": canvas_size,
        "source_shape": [int(grayscale.shape[0]), int(grayscale.shape[1])],
        "bbox": {
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
        },
        "scale": float(scale),
        "resized_width": int(resized_width),
        "resized_height": int(resized_height),
        "x_offset": int(x_offset),
        "y_offset": int(y_offset),
    }


def apply_fixed_transform(
    image: str | Path | np.ndarray,
    transform_params: dict[str, Any],
    canvas_size: int = 128,
) -> np.ndarray:
    """Normalize an image using a precomputed reference transform.

    Unlike normalize_image(), this does not compute an ink bounding box from the
    input attempt. The reference crop, scale, and offsets are reused exactly. Ink
    outside the reference crop is clipped, and crop areas outside the input frame
    are padded with background.
    """

    if canvas_size <= 0:
        raise ValueError("canvas_size must be positive")

    transform_canvas_size = int(transform_params.get("canvas_size", canvas_size))
    if transform_canvas_size != canvas_size:
        raise ValueError(
            f"Transform was computed for canvas_size={transform_canvas_size}, "
            f"but canvas_size={canvas_size} was requested"
        )

    required_keys = {"source_shape", "bbox", "resized_width", "resized_height", "x_offset", "y_offset"}
    missing = sorted(required_keys - set(transform_params))
    if missing:
        raise ValueError(f"Transform is missing required keys: {', '.join(missing)}")

    grayscale = _load_grayscale(image)
    binary = _binarize_ink(grayscale)
    binary = _resize_to_reference_shape(binary, transform_params["source_shape"])
    cropped = _crop_with_padding(binary, transform_params["bbox"])

    resized_width = int(transform_params["resized_width"])
    resized_height = int(transform_params["resized_height"])
    x_offset = int(transform_params["x_offset"])
    y_offset = int(transform_params["y_offset"])

    resized = cv2.resize(
        cropped,
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    )
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    canvas[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized

    ink_ratio = cv2.countNonZero(canvas) / float(canvas_size * canvas_size)
    if ink_ratio < 0.001:
        raise NormalizationError("Image is near-empty after applying the fixed reference transform")
    if ink_ratio > 0.95:
        raise NormalizationError("Image is almost fully foreground after applying the fixed reference transform")

    return canvas.astype(np.float32) / 255.0


def normalize_image(image: str | Path | np.ndarray, canvas_size: int = 128) -> np.ndarray:
    """Normalize a handwritten character image to a centered square float array.

    The returned image uses white foreground ink on a black background with values
    in the range 0.0 to 1.0.
    """

    if canvas_size <= 0:
        raise ValueError("canvas_size must be positive")

    grayscale = _load_grayscale(image)
    binary = _binarize_ink(grayscale)
    x, y, width, height = _find_ink_bbox(binary)

    cropped = binary[y : y + height, x : x + width]
    target_extent = max(1, int(canvas_size * 0.86))
    scale = target_extent / float(max(width, height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))

    resized = cv2.resize(
        cropped,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    x_offset = (canvas_size - resized_width) // 2
    y_offset = (canvas_size - resized_height) // 2
    canvas[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized

    return canvas.astype(np.float32) / 255.0
