"""LIBAD dataset loader with official-split preference and a protocol fixture.

Official comparable numbers require the 4.84 GB Hugging Face release plus the
10 official split files. When those artifacts are absent, this module builds a
tiny spatially aligned VIS/X-rayL fixture so the adapter, gate, and 10-seed
protocol can be tested without claiming paper-table performance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from libad.protocol import (
    OFFICIAL_SPLIT_SEEDS,
    PROJECT_ROOT,
    load_libad_config,
    official_libad_present,
    splits_root_from_config,
)

DEFECT_GROUPS = (
    "wrinkling",
    "particle",
    "pit",
    "unevenness",
    "barefoil",
    "scratch",
    "polarity",
    "debonding",
    "crack",
    "streak",
    "pinhole",
)


@dataclass
class LibadSample:
    sample_id: str
    split: str
    label: int
    defect_group: str
    vis_a: np.ndarray
    vis_b: np.ndarray
    xray_l: np.ndarray
    vis_a_path: Optional[str] = None
    vis_b_path: Optional[str] = None
    xray_l_path: Optional[str] = None


@dataclass
class LibadSplit:
    seed: int
    source: str
    train: List[LibadSample]
    val: List[LibadSample]
    test: List[LibadSample]
    comparable_to_paper: bool = False

    def all_samples(self) -> List[LibadSample]:
        return [*self.train, *self.val, *self.test]


def _electrode_canvas(size: int, rng: np.random.Generator, mean: float = 142.0) -> np.ndarray:
    noise = rng.normal(mean, 7.0, (size, size))
    yy, xx = np.mgrid[0:size, 0:size]
    grain = 4.0 * np.sin(xx / 9.0) + 3.0 * np.cos(yy / 11.0)
    return np.clip(noise + grain, 0, 255).astype(np.uint8)


def _draw_scratch(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = image.copy()
    y = int(rng.integers(8, out.shape[0] - 8))
    cv2.line(out, (2, y), (out.shape[1] - 3, y + int(rng.integers(-3, 4))), 0, 3)
    cv2.line(out, (4, y + 4), (out.shape[1] - 5, y + 6), 12, 2)
    return out


def _draw_density_blob(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = image.copy()
    center = (
        int(rng.integers(14, out.shape[1] - 14)),
        int(rng.integers(14, out.shape[0] - 14)),
    )
    cv2.circle(out, center, int(rng.integers(8, 12)), 0, -1)
    cv2.circle(out, center, int(rng.integers(4, 7)), 255, -1)
    return out


def build_protocol_fixture(
    seed: int,
    image_size: int = 64,
    n_train: int = 12,
    n_val: int = 4,
    n_test_normal: int = 6,
    n_test_anomaly: int = 6,
) -> LibadSplit:
    """Deterministic aligned VIS/X-rayL fixture for one official seed."""
    rng = np.random.default_rng(int(seed))
    train: List[LibadSample] = []
    val: List[LibadSample] = []
    test: List[LibadSample] = []

    def make_normal(sample_id: str, split: str, group: str) -> LibadSample:
        vis = _electrode_canvas(image_size, rng)
        xray = _electrode_canvas(image_size, rng, mean=128.0)
        return LibadSample(sample_id, split, 0, group, vis, vis.copy(), xray)

    def make_anomaly(sample_id: str, group: str, kind: str) -> LibadSample:
        vis = _electrode_canvas(image_size, rng)
        xray = _electrode_canvas(image_size, rng, mean=128.0)
        if kind == "surface":
            vis = _draw_scratch(vis, rng)
        elif kind == "internal":
            xray = _draw_density_blob(xray, rng)
        else:
            vis = _draw_scratch(vis, rng)
            xray = _draw_density_blob(xray, rng)
        return LibadSample(sample_id, "test", 1, group, vis, vis.copy(), xray)

    for index in range(n_train):
        group = DEFECT_GROUPS[index % len(DEFECT_GROUPS)]
        train.append(make_normal(f"fix-{seed}-train-{index:02d}", "train", group))
    for index in range(n_val):
        group = DEFECT_GROUPS[index % len(DEFECT_GROUPS)]
        val.append(make_normal(f"fix-{seed}-val-{index:02d}", "val", group))
    for index in range(n_test_normal):
        group = DEFECT_GROUPS[index % len(DEFECT_GROUPS)]
        test.append(make_normal(f"fix-{seed}-testn-{index:02d}", "test", group))
    kinds = ["surface", "internal", "both", "surface", "internal", "both"]
    for index in range(n_test_anomaly):
        group = DEFECT_GROUPS[(index + 3) % len(DEFECT_GROUPS)]
        test.append(make_anomaly(f"fix-{seed}-testa-{index:02d}", group, kinds[index % len(kinds)]))
    return LibadSplit(
        seed=int(seed),
        source="protocol_fixture",
        train=train,
        val=val,
        test=test,
        comparable_to_paper=False,
    )


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read LIBAD image: {path}")
    return image


def _load_official_split_ids(splits_root: Path, seed: int) -> Optional[Dict[str, List[str]]]:
    candidates = [
        splits_root / f"{seed}.json",
        splits_root / f"split_{seed}.json",
        splits_root / f"{seed}.jsonl",
    ]
    for path in candidates:
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            splits = payload.get("splits", payload)
            return {
                "train": list(splits.get("train", [])),
                "val": list(splits.get("val", [])),
                "test": list(splits.get("test", [])),
            }
    train_txt = splits_root / str(seed) / "train.txt"
    if train_txt.is_file():
        out: Dict[str, List[str]] = {}
        for name in ("train", "val", "test"):
            file_path = splits_root / str(seed) / f"{name}.txt"
            if file_path.is_file():
                out[name] = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if out:
            return out
    return None


def _iter_official_samples(dataset_root: Path) -> List[Tuple[str, str, int, Path]]:
    records: List[Tuple[str, str, int, Path]] = []
    for group_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        group = group_dir.name.split("_", 1)[-1] if "_" in group_dir.name else group_dir.name
        for label_name, label in (("normal", 0), ("anomaly", 1)):
            folder = group_dir / label_name
            if not folder.is_dir():
                continue
            stems = {path.name[:-6] for path in folder.glob("*A.tiff")}
            stems.update({path.name[:-5] for path in folder.glob("*A.tif")})
            for stem in sorted(stems):
                records.append((stem, group, label, folder))
    return records


def load_official_split(seed: int, config: Optional[dict] = None) -> Optional[LibadSplit]:
    cfg = config or load_libad_config()
    dataset_root = PROJECT_ROOT / cfg["paths"]["dataset_root"]
    if not dataset_root.is_dir():
        return None
    split_ids = _load_official_split_ids(splits_root_from_config(cfg), seed)
    records = _iter_official_samples(dataset_root)
    if not records:
        return None
    by_id = {stem: (group, label, folder) for stem, group, label, folder in records}

    def resolve(sample_id: str, split: str) -> Optional[LibadSample]:
        if sample_id not in by_id:
            return None
        group, label, folder = by_id[sample_id]
        vis_a = folder / f"{sample_id}A.tiff"
        vis_b = folder / f"{sample_id}B.tiff"
        xray_l = folder / f"{sample_id}L.tiff"
        if not vis_a.is_file():
            vis_a = folder / f"{sample_id}A.tif"
        if not vis_b.is_file():
            vis_b = folder / f"{sample_id}B.tif"
        if not xray_l.is_file():
            xray_l = folder / f"{sample_id}L.tif"
        if not vis_a.is_file() or not xray_l.is_file():
            return None
        vis_b_img = _read_gray(vis_b) if vis_b.is_file() else _read_gray(vis_a)
        return LibadSample(
            sample_id=sample_id,
            split=split,
            label=label,
            defect_group=group,
            vis_a=_read_gray(vis_a),
            vis_b=vis_b_img,
            xray_l=_read_gray(xray_l),
            vis_a_path=str(vis_a),
            vis_b_path=str(vis_b) if vis_b.is_file() else None,
            xray_l_path=str(xray_l),
        )

    if split_ids is None:
        return None
    loaded = {
        name: [sample for sample_id in ids if (sample := resolve(sample_id, name))]
        for name, ids in split_ids.items()
    }
    if not loaded.get("train") or not loaded.get("test"):
        return None
    return LibadSplit(
        seed=int(seed),
        source="official_libad_splits",
        train=loaded.get("train", []),
        val=loaded.get("val", []),
        test=loaded.get("test", []),
        comparable_to_paper=True,
    )


def load_split(seed: int, allow_fixture: bool = True, config: Optional[dict] = None) -> LibadSplit:
    if seed not in OFFICIAL_SPLIT_SEEDS:
        raise ValueError(f"Split seed {seed} is not one of the 10 official LIBAD seeds")
    official = load_official_split(seed, config=config)
    if official is not None:
        return official
    if not allow_fixture:
        raise FileNotFoundError(
            "Official LIBAD dataset/splits were not found. Download the CC BY 4.0 "
            "release from https://huggingface.co/datasets/Evenrose/LIBAD"
        )
    cfg = config or load_libad_config()
    image_size = int(cfg["features"]["image_size"][0])
    return build_protocol_fixture(seed, image_size=image_size)


def load_official_splits(
    seeds: Sequence[int] = OFFICIAL_SPLIT_SEEDS,
    allow_fixture: bool = True,
    config: Optional[dict] = None,
) -> List[LibadSplit]:
    return [load_split(int(seed), allow_fixture=allow_fixture, config=config) for seed in seeds]


def dataset_status(config: Optional[dict] = None) -> Dict[str, object]:
    cfg = config or load_libad_config()
    present = official_libad_present(cfg)
    splits = splits_root_from_config(cfg)
    return {
        "official_dataset_present": present,
        "official_splits_present": splits.is_dir() and any(splits.iterdir()) if splits.exists() else False,
        "comparable_to_paper": present and splits.is_dir(),
        "dataset_root": str(PROJECT_ROOT / cfg["paths"]["dataset_root"]),
        "splits_root": str(splits),
        "official_split_seeds": list(OFFICIAL_SPLIT_SEEDS),
        "license": "CC BY 4.0",
    }
