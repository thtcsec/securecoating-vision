"""Numpy patch descriptors for the LIBAD adapter.

The official DA-Core paper uses frozen DINOv3 ViT-S/16 features. This module is
an interface-compatible local descriptor so the evidence gate, 10-split harness,
and 90-second demo can run without downloading that backbone. Results produced
with this descriptor are not comparable to the published LIBAD table.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Image is required")
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.float32)
    if array.ndim == 3 and array.shape[2] >= 3:
        return cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    if array.ndim == 3 and array.shape[2] == 1:
        return array[:, :, 0].astype(np.float32)
    raise ValueError(f"Unsupported image shape: {array.shape}")


def resize_gray(image: np.ndarray, size: Sequence[int]) -> np.ndarray:
    height, width = int(size[0]), int(size[1])
    gray = to_gray(image)
    if gray.shape[0] == height and gray.shape[1] == width:
        return gray
    return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)


def extract_patch_features(
    image: np.ndarray,
    patch_size: int = 16,
    stride: int = 8,
    image_size: Optional[Sequence[int]] = (64, 64),
) -> np.ndarray:
    """Return an (N, D) patch-feature matrix for one grayscale or BGR image."""
    gray = resize_gray(image, image_size) if image_size is not None else to_gray(image)
    height, width = gray.shape
    if height < patch_size or width < patch_size:
        raise ValueError("Image is smaller than the configured patch size")

    features = []
    for row in range(0, height - patch_size + 1, stride):
        for col in range(0, width - patch_size + 1, stride):
            patch = gray[row:row + patch_size, col:col + patch_size]
            grad_x, grad_y = np.gradient(patch)
            magnitude = np.hypot(grad_x, grad_y)
            hist, _ = np.histogram(magnitude, bins=8, range=(0.0, 64.0), density=True)
            feature = np.concatenate(
                [
                    np.array(
                        [float(patch.mean()), float(patch.std()), float(magnitude.mean()), float(magnitude.std())],
                        dtype=np.float32,
                    ),
                    hist.astype(np.float32),
                ]
            )
            features.append(feature)
    stacked = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    return stacked / np.clip(norms, 1e-6, None)


def extract_image_features(
    images: Iterable[np.ndarray],
    patch_size: int = 16,
    stride: int = 8,
    image_size: Optional[Sequence[int]] = (64, 64),
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack patch features from many images and return (features, image_index)."""
    blocks = []
    index = []
    for image_id, image in enumerate(images):
        feats = extract_patch_features(
            image, patch_size=patch_size, stride=stride, image_size=image_size
        )
        blocks.append(feats)
        index.append(np.full((feats.shape[0],), image_id, dtype=np.int32))
    if not blocks:
        return np.zeros((0, 12), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return np.concatenate(blocks, axis=0), np.concatenate(index, axis=0)
