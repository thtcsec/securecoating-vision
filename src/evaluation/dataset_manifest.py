"""Integrity validation for immutable, roll-disjoint dataset manifests."""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict


REQUIRED_SPLITS = ("train", "val", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_roll_disjoint_manifest(
    manifest_path: str,
    dataset_root: str,
    evaluation_dataset_dir: str = None,
) -> Dict[str, object]:
    """Validate paths, hashes, and group separation in a dataset manifest."""
    manifest_file = Path(manifest_path).resolve()
    root = Path(dataset_root).resolve()
    with manifest_file.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if not isinstance(manifest, dict) or not isinstance(manifest.get("splits"), dict):
        raise ValueError("Manifest must contain a splits object")
    missing = [split for split in REQUIRED_SPLITS if not manifest["splits"].get(split)]
    if missing:
        raise ValueError(f"Manifest must contain non-empty splits: {missing}")

    seen_paths = set()
    roll_to_split: Dict[str, str] = {}
    file_count = 0
    test_paths = set()
    for split in REQUIRED_SPLITS:
        entries = manifest["splits"][split]
        if not isinstance(entries, list):
            raise ValueError(f"Manifest split {split} must be a list")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"Manifest entry in {split} must be an object")
            relative_path = entry.get("path")
            roll_id = entry.get("roll_id")
            expected_hash = entry.get("sha256", "")
            if not relative_path or not roll_id or len(expected_hash) != 64:
                raise ValueError(f"Manifest entry in {split} requires path, roll_id, and sha256")
            path = (root / relative_path).resolve()
            if root not in path.parents or not path.is_file():
                raise FileNotFoundError(f"Manifest artifact is outside or missing: {relative_path}")
            normalized = os.path.normcase(str(path))
            if normalized in seen_paths:
                raise ValueError(f"Manifest path appears more than once: {relative_path}")
            seen_paths.add(normalized)
            if split == "test":
                test_paths.add(normalized)
            actual_hash = _sha256(path)
            if actual_hash.lower() != expected_hash.lower():
                raise ValueError(f"Manifest hash mismatch: {relative_path}")
            previous_split = roll_to_split.setdefault(str(roll_id), split)
            if previous_split != split:
                raise ValueError(
                    f"Roll {roll_id} appears in both {previous_split} and {split}"
                )
            file_count += 1

    if evaluation_dataset_dir is not None:
        evaluation_root = Path(evaluation_dataset_dir).resolve()
        expected_paths = {
            os.path.normcase(str(path.resolve()))
            for path in evaluation_root.joinpath("images").iterdir()
            if path.is_file()
        }
        if test_paths != expected_paths:
            raise ValueError("Manifest test split does not exactly match evaluation images")

    return {
        "manifest_sha256": _sha256(manifest_file),
        "dataset_root": str(root),
        "splits": {split: len(manifest["splits"][split]) for split in REQUIRED_SPLITS},
        "roll_count": len(roll_to_split),
        "file_count": file_count,
    }
