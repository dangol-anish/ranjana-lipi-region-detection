"""Normalize Ranjana Lipi handwriting images onto a fixed square canvas."""

from __future__ import annotations

from pathlib import Path

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

    denoised = cv2.GaussianBlur(grayscale, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )

    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    return binary


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
