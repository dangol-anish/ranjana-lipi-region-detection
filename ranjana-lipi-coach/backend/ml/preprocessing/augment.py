"""Mild handwriting augmentations for normalized Ranjana Lipi characters."""

from __future__ import annotations

import random

import cv2
import numpy as np


def _as_float_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")
    if image.ndim != 2:
        raise ValueError("augment_image expects a single-channel normalized image")

    output = image.astype(np.float32, copy=False)
    if output.max(initial=0) > 1.0:
        output = output / 255.0
    return np.clip(output, 0.0, 1.0)


def _elastic_grid_distortion(image: np.ndarray) -> np.ndarray:
    height, width = image.shape
    displacement = random.uniform(0.8, 1.8)
    grid_size = 4

    coarse_dx = np.random.uniform(-displacement, displacement, (grid_size, grid_size)).astype(
        np.float32
    )
    coarse_dy = np.random.uniform(-displacement, displacement, (grid_size, grid_size)).astype(
        np.float32
    )
    dx = cv2.resize(coarse_dx, (width, height), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(coarse_dy, (width, height), interpolation=cv2.INTER_CUBIC)

    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    map_x = (x_coords + dx).astype(np.float32)
    map_y = (y_coords + dy).astype(np.float32)
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def augment_image(image: np.ndarray) -> np.ndarray:
    """Apply mild, realistic handwriting variation to a normalized character."""

    # Distortions must stay mild enough that they represent natural handwriting
    # variation of a CORRECT character, not enough to make it resemble another class
    # or move strokes into different fixed-grid feedback regions.
    normalized = _as_float_image(image)
    height, width = normalized.shape
    center = (width / 2.0, height / 2.0)

    rotation = random.uniform(-8.0, 8.0)
    scale = random.uniform(0.95, 1.05)
    translation_x = random.uniform(-4.0, 4.0)
    translation_y = random.uniform(-4.0, 4.0)

    transform = cv2.getRotationMatrix2D(center, rotation, scale)
    transform[0, 2] += translation_x
    transform[1, 2] += translation_y

    augmented = cv2.warpAffine(
        normalized,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    augmented = _elastic_grid_distortion(augmented)

    stroke_variation = random.choice(("none", "dilate", "erode"))
    if stroke_variation != "none":
        kernel = np.ones((2, 2), np.uint8)
        stroke_image = (augmented * 255).astype(np.uint8)
        if stroke_variation == "dilate":
            stroke_image = cv2.dilate(stroke_image, kernel, iterations=1)
        else:
            stroke_image = cv2.erode(stroke_image, kernel, iterations=1)
        augmented = stroke_image.astype(np.float32) / 255.0

    noise_sigma = random.uniform(0.0, 0.025)
    if noise_sigma > 0:
        noise = np.random.normal(0.0, noise_sigma, augmented.shape).astype(np.float32)
        augmented = augmented + noise

    return np.clip(augmented, 0.0, 1.0).astype(np.float32)
