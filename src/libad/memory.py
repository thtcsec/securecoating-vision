"""PatchCore FPS and DA-Core density-aware FPS coreset selection.

PatchCore farthest-point sampling follows Roth et al. DA-Core density-aware FPS
follows Sui et al., LIBAD (2026), equations (4)-(9). This file reimplements the
published selection rule so the industrial gate can consume modality scores. It
is not an original anomaly-detection algorithm.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

CoresetMethod = Literal["fps", "density_fps"]


def pairwise_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    left_norm = np.sum(left * left, axis=1, keepdims=True)
    right_norm = np.sum(right * right, axis=1, keepdims=True).T
    return np.sqrt(np.clip(left_norm + right_norm - 2.0 * left @ right.T, 0.0, None))


def _quantile_normalize(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float32)
    return ranks / float(len(values) - 1)


def knn_density(features: np.ndarray, knn: int = 8) -> np.ndarray:
    """Paper eq. (6)-(7): Gaussian kNN density, then log1p."""
    if len(features) == 0:
        return np.zeros((0,), dtype=np.float32)
    distances = pairwise_distances(features, features)
    np.fill_diagonal(distances, np.inf)
    k = max(1, min(int(knn), len(features) - 1))
    knn_distances = np.partition(distances, kth=k - 1, axis=1)[:, :k]
    tau = float(np.median(knn_distances)) or 1e-6
    density = np.sum(np.exp(-np.square(knn_distances / tau)), axis=1)
    return np.log1p(density).astype(np.float32)


def farthest_point_sample(features: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Standard PatchCore FPS (coverage only)."""
    n_samples = len(features)
    if n_samples == 0 or count <= 0:
        return np.zeros((0,), dtype=np.int32)
    count = min(count, n_samples)
    selected = [int(rng.integers(0, n_samples))]
    min_dist = pairwise_distances(features, features[selected[0] : selected[0] + 1])[:, 0]
    for _ in range(1, count):
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        new_dist = pairwise_distances(features, features[nxt : nxt + 1])[:, 0]
        min_dist = np.minimum(min_dist, new_dist)
        min_dist[selected] = -1.0
    return np.asarray(selected, dtype=np.int32)


def density_aware_fps(
    features: np.ndarray,
    count: int,
    rng: np.random.Generator,
    density_weight: float = 0.7,
    knn: int = 8,
) -> np.ndarray:
    """DA-Core density-aware FPS (Sui et al. 2026)."""
    n_samples = len(features)
    if n_samples == 0 or count <= 0:
        return np.zeros((0,), dtype=np.int32)
    count = min(count, n_samples)
    density = _quantile_normalize(knn_density(features, knn=knn))
    selected = [int(rng.integers(0, n_samples))]
    min_dist = pairwise_distances(features, features[selected[0] : selected[0] + 1])[:, 0]
    remaining = np.ones(n_samples, dtype=bool)
    remaining[selected[0]] = False
    for _ in range(1, count):
        if not np.any(remaining):
            break
        coverage = min_dist.copy()
        coverage[~remaining] = 0.0
        finite = coverage[remaining]
        span = float(finite.max() - finite.min()) if len(finite) else 0.0
        if span <= 1e-12:
            normalized = np.zeros_like(coverage)
        else:
            normalized = (coverage - finite.min()) / span
            normalized[~remaining] = 0.0
        score = normalized * (1.0 + float(density_weight) * density)
        score[~remaining] = -1.0
        nxt = int(np.argmax(score))
        selected.append(nxt)
        remaining[nxt] = False
        new_dist = pairwise_distances(features, features[nxt : nxt + 1])[:, 0]
        min_dist = np.minimum(min_dist, new_dist)
    return np.asarray(selected, dtype=np.int32)


def select_coreset(
    features: np.ndarray,
    ratio: float = 0.25,
    method: CoresetMethod = "density_fps",
    density_weight: float = 0.7,
    knn: int = 8,
    seed: int = 0,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if len(features) == 0:
        return features
    count = max(1, int(round(len(features) * float(ratio))))
    count = min(count, len(features))
    rng = np.random.default_rng(int(seed))
    if method == "fps":
        index = farthest_point_sample(features, count, rng)
    elif method == "density_fps":
        index = density_aware_fps(
            features, count, rng, density_weight=density_weight, knn=knn
        )
    else:
        raise ValueError(f"Unknown coreset method: {method}")
    return features[index]


def nearest_memory_distance(query: np.ndarray, memory: np.ndarray) -> np.ndarray:
    """Per-patch min distance to the memory bank."""
    if len(query) == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(memory) == 0:
        return np.full((len(query),), np.inf, dtype=np.float32)
    distances = pairwise_distances(query, memory)
    return distances.min(axis=1).astype(np.float32)


def image_anomaly_score(query: np.ndarray, memory: np.ndarray) -> float:
    """PatchCore-style image score: max of per-patch nearest-neighbor distances."""
    patch_scores = nearest_memory_distance(query, memory)
    if len(patch_scores) == 0:
        return 0.0
    return float(np.max(patch_scores))
