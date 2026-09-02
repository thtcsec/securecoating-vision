"""Modality-level PatchCore / DA-Core scoring for the LIBAD adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence

import numpy as np

from libad.features import FEATURE_DIM, extract_patch_features
from libad.memory import CoresetMethod, image_anomaly_score, select_coreset

Modality = Literal["vis", "xray_l"]


@dataclass
class ModalityMemory:
    name: Modality
    memory: np.ndarray
    method: str
    reference_score: float
    coreset_ratio: float
    n_train_patches: int


@dataclass
class SampleScore:
    sample_id: str
    vis_score: Optional[float]
    xray_score: Optional[float]
    fused_score: Optional[float]
    vis_available: bool
    xray_available: bool
    label: Optional[int] = None
    defect_group: Optional[str] = None
    extras: Dict[str, float] = field(default_factory=dict)


class MemoryAnomalyScorer:
    """Train per-modality memory banks and score VIS / X-rayL samples.

    `method="fps"` is PatchCore-style. `method="density_fps"` is DA-Core coreset
    selection from Sui et al. Late fusion is a conservative max, not the paper's
    OCSVM head, unless an external official implementation is wired in.
    """

    def __init__(
        self,
        method: CoresetMethod = "density_fps",
        coreset_ratio: float = 0.05,
        density_weight: float = 0.7,
        knn: int = 8,
        patch_size: int = 16,
        stride: int = 8,
        image_size: Sequence[int] = (64, 64),
        vis_scale: float = 1.0,
        xray_scale: float = 1.0,
        seed: int = 0,
        min_coreset_count: int = 0,
    ):
        self.method = method
        self.coreset_ratio = coreset_ratio
        self.density_weight = density_weight
        self.knn = knn
        self.patch_size = patch_size
        self.stride = stride
        self.image_size = tuple(image_size)
        self.vis_scale = vis_scale
        self.xray_scale = xray_scale
        self.seed = seed
        self.min_coreset_count = int(min_coreset_count)
        self.memories: Dict[str, ModalityMemory] = {}

    def _extract(self, image: np.ndarray) -> np.ndarray:
        return extract_patch_features(
            image,
            patch_size=self.patch_size,
            stride=self.stride,
            image_size=self.image_size,
        )

    def _fit_modality(
        self,
        name: Modality,
        images: Sequence[np.ndarray],
    ) -> Optional[ModalityMemory]:
        if not images:
            return None
        patches = [self._extract(image) for image in images]
        stacked = np.concatenate(patches, axis=0)
        if len(stacked) > 4000:
            rng = np.random.default_rng(int(self.seed))
            keep = rng.choice(len(stacked), 4000, replace=False)
            stacked = stacked[keep]
        memory = select_coreset(
            stacked,
            ratio=self.coreset_ratio,
            method=self.method,
            density_weight=self.density_weight,
            knn=self.knn,
            seed=self.seed,
            min_count=self.min_coreset_count,
        )
        train_scores = [image_anomaly_score(item, memory) for item in patches]
        reference = float(np.median(train_scores)) if train_scores else 1.0
        bank = ModalityMemory(
            name=name,
            memory=memory,
            method=self.method,
            reference_score=max(reference, 1e-6),
            coreset_ratio=self.coreset_ratio,
            n_train_patches=int(stacked.shape[0]),
        )
        self.memories[name] = bank
        return bank

    def fit(
        self,
        vis_images: Optional[Sequence[np.ndarray]] = None,
        xray_images: Optional[Sequence[np.ndarray]] = None,
    ) -> None:
        self.memories = {}
        if vis_images:
            self._fit_modality("vis", vis_images)
        if xray_images:
            self._fit_modality("xray_l", xray_images)

    def _normalized_score(self, name: Modality, image: Optional[np.ndarray]) -> Optional[float]:
        if image is None or name not in self.memories:
            return None
        bank = self.memories[name]
        raw = image_anomaly_score(self._extract(image), bank.memory)
        return raw / bank.reference_score

    def score_sample(
        self,
        sample_id: str,
        vis_a: Optional[np.ndarray] = None,
        vis_b: Optional[np.ndarray] = None,
        xray_l: Optional[np.ndarray] = None,
        label: Optional[int] = None,
        defect_group: Optional[str] = None,
    ) -> SampleScore:
        vis_scores: List[float] = []
        for frame in (vis_a, vis_b):
            value = self._normalized_score("vis", frame)
            if value is not None:
                vis_scores.append(value)
        vis_score = max(vis_scores) * self.vis_scale if vis_scores else None
        xray_score = self._normalized_score("xray_l", xray_l)
        if xray_score is not None:
            xray_score *= self.xray_scale
        fused = None
        extras: Dict[str, float] = {}
        if vis_score is not None and xray_score is not None:
            fused = max(vis_score, xray_score)
            extras["mean_fusion"] = float(0.5 * (vis_score + xray_score))
        elif vis_score is not None:
            fused = vis_score
        elif xray_score is not None:
            fused = xray_score
        return SampleScore(
            sample_id=sample_id,
            vis_score=vis_score,
            xray_score=xray_score,
            fused_score=fused,
            vis_available=vis_score is not None,
            xray_available=xray_score is not None,
            label=label,
            defect_group=defect_group,
            extras=extras,
        )

    def provenance(self) -> Dict[str, object]:
        return {
            "algorithm_family": "memory-bank nearest-neighbor",
            "coreset_method": self.method,
            "coreset_attribution": (
                "fps follows PatchCore (Roth et al.); density_fps follows DA-Core "
                "(Sui et al., LIBAD 2026). Neither is claimed as a SecureCoating-Vision algorithm."
            ),
            "feature_backbone": "numpy_patch_descriptor",
            "feature_dim": FEATURE_DIM,
            "late_fusion": "max of modality-normalized image scores",
            "memories": {
                name: {
                    "n_memory": int(bank.memory.shape[0]),
                    "n_train_patches": bank.n_train_patches,
                    "reference_score": bank.reference_score,
                    "method": bank.method,
                }
                for name, bank in self.memories.items()
            },
        }
