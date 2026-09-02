"""LIBAD dataset loader with official-split preference and a protocol fixture.

Official comparable numbers require the 4.84 GB Hugging Face release plus the
10 official split files. When those artifacts are absent, this module builds a
tiny spatially aligned VIS/X-rayL fixture so the adapter, gate, and 10-seed
protocol can be tested without claiming paper-table performance.
"""

from __future__ import annotations

import csv
import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from libad.protocol import (
    OFFICIAL_SPLIT_SEEDS,
    PROJECT_ROOT,
    load_libad_config,
    official_libad_present,
    sha256_tree,
    splits_root_from_config,
    tree_stat_fingerprint,
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
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot read LIBAD image: {path}")
    if image.ndim == 3:
        if image.shape[2] == 1:
            image = image[:, :, 0]
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype in (np.float32, np.float64):
        finite = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = float(np.min(finite)), float(np.max(finite))
        if hi <= 1.5 and lo >= 0.0 and (hi - lo) > 1e-6:
            # Official X-rayL is a narrow float band; stretch for the local uint8 descriptor.
            image = np.clip((finite - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        elif hi <= 1.5:
            image = np.clip(finite * 255.0, 0, 255).astype(np.uint8)
        else:
            image = cv2.normalize(finite, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image


def _adapter_frame(
    path: Path,
    image_size: Optional[Sequence[int]] = None,
    cache_root: Optional[Path] = None,
    dataset_root: Optional[Path] = None,
) -> np.ndarray:
    """Load one view, convert to uint8, optionally cache a 64x64 preview."""
    cache_path: Optional[Path] = None
    if cache_root is not None and dataset_root is not None:
        try:
            relative = Path(path).resolve().relative_to(Path(dataset_root).resolve())
            cache_path = Path(cache_root) / relative.with_suffix(".png")
        except ValueError:
            cache_path = None
        if cache_path is not None and cache_path.is_file():
            cached = cv2.imread(str(cache_path), cv2.IMREAD_GRAYSCALE)
            if cached is not None:
                return cached
    image = _read_gray(path)
    if image_size is not None:
        height, width = int(image_size[0]), int(image_size[1])
        if image.shape[0] != height or image.shape[1] != width:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cache_path), image)
    return image


def _official_csv_split_path(splits_root: Path, seed: int) -> Optional[Path]:
    preferred = splits_root / f"LIBAD_normal70base_val15normal_seed{int(seed)}.csv"
    if preferred.is_file():
        return preferred
    fallback = splits_root / f"LIBAD_normal70train_noval_seed{int(seed)}.csv"
    return fallback if fallback.is_file() else None


def _load_official_csv_split_ids(path: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split = str(row.get("split", "")).strip()
            key = str(row.get("sample_key", "")).strip()
            if split in out and key:
                out[split].append(key)
    return out


def _record_lookup(
    by_id: Dict[str, Tuple[str, int, Path, str]],
    sample_id: str,
) -> Optional[Tuple[str, int, Path, str]]:
    if sample_id in by_id:
        return by_id[sample_id]
    suffix = "/" + sample_id
    matches = [item for key, item in by_id.items() if key.endswith(suffix)]
    if len(matches) == 1:
        return matches[0]
    return None


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
    csv_path = _official_csv_split_path(splits_root, seed)
    if csv_path is not None:
        return _load_official_csv_split_ids(csv_path)
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


def _index_official_samples(
    records: List[Tuple[str, str, int, Path]],
) -> Dict[str, Tuple[str, int, Path, str]]:
    """Map CSV sample_key and unique stems to (group, label, folder, stem)."""
    by_id: Dict[str, Tuple[str, int, Path, str]] = {}
    for stem, group, label, folder in records:
        class_dir = folder.parent.name
        label_name = folder.name
        payload = (group, label, folder, stem)
        by_id[f"{class_dir}/{label_name}/{stem}"] = payload
        by_id.setdefault(stem, payload)
    return by_id


def _official_sample_paths(folder: Path, sample_id: str) -> Tuple[Path, Path, Path]:
    def choose(suffix: str) -> Path:
        tiff = folder / f"{sample_id}{suffix}.tiff"
        return tiff if tiff.is_file() else folder / f"{sample_id}{suffix}.tif"

    return choose("A"), choose("B"), choose("L")


def _split_manifest_is_complete(
    split_ids: Optional[Dict[str, List[str]]],
    by_id: Dict[str, Tuple[str, int, Path, str]],
) -> bool:
    if split_ids is None:
        return False
    required_names = ("train", "val", "test")
    if any(not split_ids.get(name) for name in required_names):
        return False
    split_sets = {name: set(split_ids[name]) for name in required_names}
    if any(len(split_sets[name]) != len(split_ids[name]) for name in required_names):
        return False
    if (
        split_sets["train"] & split_sets["val"]
        or split_sets["train"] & split_sets["test"]
        or split_sets["val"] & split_sets["test"]
    ):
        return False
    for sample_id in set().union(*split_sets.values()):
        record = _record_lookup(by_id, sample_id)
        if record is None:
            return False
        _, _, folder, stem = record
        if not all(path.is_file() for path in _official_sample_paths(folder, stem)):
            return False
    return True


def load_official_split(seed: int, config: Optional[dict] = None) -> Optional[LibadSplit]:
    cfg = config or load_libad_config()
    dataset_root = PROJECT_ROOT / cfg["paths"]["dataset_root"]
    if not dataset_root.is_dir():
        return None
    split_ids = _load_official_split_ids(splits_root_from_config(cfg), seed)
    records = _iter_official_samples(dataset_root)
    if not records or split_ids is None:
        return None
    by_id = _index_official_samples(records)

    if not _split_manifest_is_complete(split_ids, by_id):
        return None
    required_names = ("train", "val", "test")
    image_size = tuple(cfg.get("features", {}).get("image_size") or (64, 64))
    dataset_root = dataset_root.resolve()
    cache_root = dataset_root.parent / "_preview64"
    progress = {"n": 0}

    def resolve(sample_id: str, split: str) -> Optional[LibadSample]:
        record = _record_lookup(by_id, sample_id)
        if record is None:
            return None
        group, label, folder, stem = record
        vis_a, vis_b, xray_l = _official_sample_paths(folder, stem)
        if not vis_a.is_file() or not vis_b.is_file() or not xray_l.is_file():
            return None
        progress["n"] += 1
        if progress["n"] % 80 == 0:
            print(f"  seed {seed}: loaded {progress['n']} samples (64x64 cache)", flush=True)
        return LibadSample(
            sample_id=sample_id,
            split=split,
            label=label,
            defect_group=group,
            vis_a=_adapter_frame(vis_a, image_size, cache_root, dataset_root),
            vis_b=_adapter_frame(vis_b, image_size, cache_root, dataset_root),
            xray_l=_adapter_frame(xray_l, image_size, cache_root, dataset_root),
            vis_a_path=str(vis_a),
            vis_b_path=str(vis_b),
            xray_l_path=str(xray_l),
        )

    loaded = {
        name: [sample for sample_id in ids if (sample := resolve(sample_id, name))]
        for name, ids in split_ids.items()
    }
    if any(len(loaded.get(name, [])) != len(split_ids[name]) for name in required_names):
        return None
    return LibadSplit(
        seed=int(seed),
        source="libad_structured_inputs",
        train=loaded.get("train", []),
        val=loaded.get("val", []),
        test=loaded.get("test", []),
        # These are official inputs, but this repository's local runner uses a
        # numpy patch descriptor rather than the authors' DINOv3 implementation.
        comparable_to_paper=False,
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


def _want_tree_verify(verify_trees: Optional[bool]) -> bool:
    if verify_trees is not None:
        return bool(verify_trees)
    flag = os.environ.get("SECURECOATING_LIBAD_VERIFY_TREES", "").strip().lower()
    return flag in {"1", "true", "yes"}


def _is_sha256_hex(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _manifest_fingerprint_matches(
    manifest: dict,
    dataset_root: Path,
    splits_root: Path,
) -> bool:
    dataset_fp = tree_stat_fingerprint(dataset_root)
    splits_fp = tree_stat_fingerprint(splits_root)
    if dataset_fp is None or splits_fp is None:
        return False
    try:
        return (
            int(manifest.get("dataset_n_files") or 0) == dataset_fp["n_files"]
            and int(manifest.get("dataset_total_bytes") or 0) == dataset_fp["total_bytes"]
            and int(manifest.get("splits_n_files") or 0) == splits_fp["n_files"]
            and int(manifest.get("splits_total_bytes") or 0) == splits_fp["total_bytes"]
        )
    except (TypeError, ValueError):
        return False


def dataset_status(
    config: Optional[dict] = None,
    verify_trees: Optional[bool] = None,
) -> Dict[str, object]:
    cfg = config or load_libad_config()
    present = official_libad_present(cfg)
    splits = splits_root_from_config(cfg)
    valid_seeds: List[int] = []
    invalid_seeds: List[int] = []
    if present and splits.is_dir():
        by_id = _index_official_samples(
            _iter_official_samples(PROJECT_ROOT / cfg["paths"]["dataset_root"])
        )
        for seed in OFFICIAL_SPLIT_SEEDS:
            split_ids = _load_official_split_ids(splits, seed)
            if not _split_manifest_is_complete(split_ids, by_id):
                invalid_seeds.append(seed)
            else:
                valid_seeds.append(seed)
    else:
        invalid_seeds = list(OFFICIAL_SPLIT_SEEDS)
    protocol_complete = len(valid_seeds) == len(OFFICIAL_SPLIT_SEEDS)
    manifest_path = PROJECT_ROOT / cfg["paths"].get(
        "official_artifact_manifest", "data/libad/official_artifact_manifest.json"
    )
    dataset_tree_sha256: Optional[str] = None
    splits_tree_sha256: Optional[str] = None
    manifest_verified = False
    manifest_error: Optional[str] = None
    tree_verify = _want_tree_verify(verify_trees)
    if protocol_complete and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            recorded_dataset = manifest.get("dataset_tree_sha256")
            recorded_splits = manifest.get("splits_tree_sha256")
            seeds_ok = manifest.get("official_split_seeds") == list(OFFICIAL_SPLIT_SEEDS)
            hashes_ok = _is_sha256_hex(recorded_dataset) and _is_sha256_hex(recorded_splits)
            fingerprint_ok = _manifest_fingerprint_matches(
                manifest,
                PROJECT_ROOT / cfg["paths"]["dataset_root"],
                splits,
            )
            if tree_verify:
                dataset_tree_sha256 = sha256_tree(PROJECT_ROOT / cfg["paths"]["dataset_root"])
                splits_tree_sha256 = sha256_tree(splits)
                manifest_verified = (
                    hashes_ok
                    and seeds_ok
                    and recorded_dataset == dataset_tree_sha256
                    and recorded_splits == splits_tree_sha256
                )
            else:
                dataset_tree_sha256 = recorded_dataset if hashes_ok else None
                splits_tree_sha256 = recorded_splits if hashes_ok else None
                manifest_verified = hashes_ok and seeds_ok and fingerprint_ok
            if not manifest_verified:
                manifest_error = "Artifact manifest hashes, fingerprint, or official seed list do not match"
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            manifest_error = f"Artifact manifest is unreadable: {exc}"
    elif protocol_complete:
        manifest_error = "Hash-verified official artifact manifest is missing"
    official_protocol_complete = protocol_complete and manifest_verified
    blockers: List[str] = []
    if not present:
        blockers.append("Official LIBAD dataset directory is absent or empty")
    if not protocol_complete:
        blockers.append("All 10 official train/val/test split seeds are not file/identity complete")
    if not manifest_verified:
        blockers.append(manifest_error or "Official artifact manifest is not verified")
    blockers.append(
        "Local runner uses numpy_patch_descriptor, not the authors' official DINOv3/DA-Core implementation"
    )
    return {
        "official_dataset_present": present,
        "official_splits_present": protocol_complete,
        "official_protocol_complete": official_protocol_complete,
        "input_structure_complete": protocol_complete,
        "official_artifact_manifest": str(manifest_path),
        "official_artifact_manifest_verified": manifest_verified,
        "official_artifact_manifest_error": manifest_error,
        "tree_hash_verified_live": bool(tree_verify and manifest_verified),
        "dataset_tree_sha256": dataset_tree_sha256,
        "splits_tree_sha256": splits_tree_sha256,
        "valid_official_split_seeds": valid_seeds,
        "invalid_official_split_seeds": invalid_seeds,
        "comparable_to_paper": False,
        "comparability_blockers": blockers,
        "dataset_root": str(PROJECT_ROOT / cfg["paths"]["dataset_root"]),
        "splits_root": str(splits),
        "official_split_seeds": list(OFFICIAL_SPLIT_SEEDS),
        "license": "CC BY 4.0",
        "mounted_sample_count": len(_official_sample_index(cfg)),
    }


_OFFICIAL_INDEX_LOCK = threading.Lock()
_OFFICIAL_INDEX_CACHE: Optional[Tuple[str, Tuple[Dict[str, Any], ...]]] = None
_SAMPLE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$")


def reset_official_sample_index_cache() -> None:
    global _OFFICIAL_INDEX_CACHE
    with _OFFICIAL_INDEX_LOCK:
        _OFFICIAL_INDEX_CACHE = None


def _official_sample_index(config: Optional[dict] = None) -> Tuple[Dict[str, Any], ...]:
    """Metadata-only index of complete VIS-A / VIS-B / X-rayL triples. No fixture rows."""
    global _OFFICIAL_INDEX_CACHE
    cfg = config or load_libad_config()
    root = PROJECT_ROOT / cfg["paths"]["dataset_root"]
    cache_key = str(root)
    with _OFFICIAL_INDEX_LOCK:
        if _OFFICIAL_INDEX_CACHE and _OFFICIAL_INDEX_CACHE[0] == cache_key:
            return _OFFICIAL_INDEX_CACHE[1]
        items: List[Dict[str, Any]] = []
        if root.is_dir():
            for stem, group, label, folder in _iter_official_samples(root):
                vis_a, vis_b, xray_l = _official_sample_paths(folder, stem)
                if not (vis_a.is_file() and vis_b.is_file() and xray_l.is_file()):
                    continue
                items.append(
                    {
                        "sample_id": stem,
                        "defect_group": group,
                        "label": "anomaly" if label else "normal",
                    }
                )
        cached = tuple(items)
        _OFFICIAL_INDEX_CACHE = (cache_key, cached)
        return cached


def list_official_samples(
    offset: int = 0,
    limit: int = 12,
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    """Page official mounted samples. Empty when the release is absent; never a fixture."""
    cfg = config or load_libad_config()
    present = official_libad_present(cfg)
    records = _official_sample_index(cfg) if present else tuple()
    safe_offset = max(0, int(offset))
    safe_limit = max(0, min(int(limit), 100))
    page = list(records[safe_offset:safe_offset + safe_limit]) if safe_limit else []
    groups = sorted({item["defect_group"] for item in records})
    return {
        "official_dataset_present": present,
        "total": len(records),
        "offset": safe_offset,
        "limit": safe_limit,
        "returned": len(page),
        "items": page,
        "defect_groups": groups,
        "image_payloads_included": False,
        "source": "official_release" if present else "not_mounted",
        "comparable_to_paper": False,
    }


def load_official_sample_view(
    sample_id: str,
    view: str,
    config: Optional[dict] = None,
) -> np.ndarray:
    """Load one official grayscale view. Refuses path-like ids and missing triples."""
    if view not in {"vis_a", "vis_b", "xray_l"}:
        raise ValueError("Unsupported multimodal view")
    if os.path.basename(sample_id) != sample_id or not _SAMPLE_ID_PATTERN.fullmatch(sample_id):
        raise FileNotFoundError("sample_id must be a bounded basename")
    cfg = config or load_libad_config()
    if not official_libad_present(cfg):
        raise FileNotFoundError("Official multimodal release is not mounted")
    root = PROJECT_ROOT / cfg["paths"]["dataset_root"]
    match = next((item for item in _official_sample_index(cfg) if item["sample_id"] == sample_id), None)
    if match is None:
        raise FileNotFoundError("Official sample was not found")
    records = {
        stem: folder
        for stem, _group, _label, folder in _iter_official_samples(root)
        if stem == sample_id
    }
    folder = records.get(sample_id)
    if folder is None:
        raise FileNotFoundError("Official sample folder was not found")
    vis_a, vis_b, xray_l = _official_sample_paths(folder, sample_id)
    path = {"vis_a": vis_a, "vis_b": vis_b, "xray_l": xray_l}[view]
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise FileNotFoundError("Official sample escaped the dataset root") from exc
    image = _read_gray(path)
    return image
