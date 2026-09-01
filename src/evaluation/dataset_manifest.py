"""Integrity validation for immutable, roll-disjoint dataset manifests."""

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict


REQUIRED_SPLITS = ("train", "val", "test")
DETECTION_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


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
    test_label_paths = set()
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
                if evaluation_dataset_dir is not None:
                    label_relative_path = entry.get("label_path")
                    expected_label_hash = entry.get("label_sha256", "")
                    if not label_relative_path or len(expected_label_hash) != 64:
                        raise ValueError(
                            "Evaluation test entries require label_path and label_sha256"
                        )
                    label_path = (root / label_relative_path).resolve()
                    if root not in label_path.parents or not label_path.is_file():
                        raise FileNotFoundError(
                            f"Manifest label artifact is outside or missing: {label_relative_path}"
                        )
                    normalized_label = os.path.normcase(str(label_path))
                    if normalized_label in test_label_paths:
                        raise ValueError(f"Manifest label appears more than once: {label_relative_path}")
                    test_label_paths.add(normalized_label)
                    if _sha256(label_path).lower() != expected_label_hash.lower():
                        raise ValueError(f"Manifest label hash mismatch: {label_relative_path}")
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
        labels_root = evaluation_root / "labels"
        expected_labels = {
            os.path.normcase(str((labels_root / f"{path.stem}.txt").resolve()))
            for path in evaluation_root.joinpath("images").iterdir()
            if path.is_file()
        }
        if test_label_paths != expected_labels:
            raise ValueError("Manifest test labels do not exactly match evaluation images")

    return {
        "manifest_sha256": _sha256(manifest_file),
        "dataset_root": str(root),
        "splits": {split: len(manifest["splits"][split]) for split in REQUIRED_SPLITS},
        "roll_count": len(roll_to_split),
        "file_count": file_count,
    }


def validate_detection_dataset_manifest(
    manifest_path: str,
    dataset_root: str,
) -> Dict[str, object]:
    """Validate a YOLO detection dataset and return a deterministic tree digest."""
    manifest_file = Path(manifest_path).resolve()
    root = Path(dataset_root).resolve()
    with manifest_file.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    splits = manifest.get("splits") if isinstance(manifest, dict) else None
    classes = manifest.get("classes") if isinstance(manifest, dict) else None
    if not isinstance(splits, dict) or not isinstance(classes, dict) or not classes:
        raise ValueError("Detection manifest requires non-empty splits and classes objects")
    class_ids = {int(class_id) for class_id in classes}
    if class_ids != set(range(len(class_ids))):
        raise ValueError("Detection class IDs must be contiguous and zero-based")

    seen_names: Dict[str, str] = {}
    split_counts: Dict[str, int] = {}
    label_count = 0
    tree_digest = hashlib.sha256()
    for split in REQUIRED_SPLITS:
        names = splits.get(split)
        if not isinstance(names, list) or not names:
            raise ValueError(f"Detection split {split} must be a non-empty list")
        split_counts[split] = len(names)
        split_names = set()
        for name in names:
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError(f"Unsafe detection filename in {split}: {name!r}")
            if Path(name).suffix.lower() not in DETECTION_IMAGE_SUFFIXES:
                raise ValueError(f"Unsupported detection image suffix: {name}")
            normalized = os.path.normcase(name)
            if normalized in split_names:
                raise ValueError(f"Detection image appears more than once in {split}: {name}")
            split_names.add(normalized)
            previous = seen_names.setdefault(normalized, split)
            if previous != split:
                raise ValueError(f"Detection image {name} appears in both {previous} and {split}")

            image_path = (root / "images" / split / name).resolve()
            label_path = (root / "labels" / split / f"{Path(name).stem}.txt").resolve()
            for artifact in (image_path, label_path):
                if root not in artifact.parents or not artifact.is_file():
                    raise FileNotFoundError(f"Missing detection artifact: {artifact}")
                relative = artifact.relative_to(root).as_posix()
                tree_digest.update(relative.encode("utf-8"))
                tree_digest.update(b"\0")
                tree_digest.update(_sha256(artifact).encode("ascii"))
                tree_digest.update(b"\n")

            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"Invalid YOLO label width: {label_path}:{line_number}")
                try:
                    class_id = int(fields[0])
                    coords = [float(value) for value in fields[1:]]
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid YOLO label value: {label_path}:{line_number}"
                    ) from exc
                if class_id not in class_ids:
                    raise ValueError(f"Unknown class ID: {label_path}:{line_number}")
                if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in coords):
                    raise ValueError(f"Out-of-range YOLO coordinate: {label_path}:{line_number}")
                if coords[2] <= 0.0 or coords[3] <= 0.0:
                    raise ValueError(f"Non-positive YOLO box: {label_path}:{line_number}")
                label_count += 1

    actual_images = {
        os.path.normcase(str(path.resolve()))
        for split in REQUIRED_SPLITS
        for path in (root / "images" / split).iterdir()
        if path.is_file() and path.suffix.lower() in DETECTION_IMAGE_SUFFIXES
    }
    expected_images = {
        os.path.normcase(str((root / "images" / split / name).resolve()))
        for split in REQUIRED_SPLITS
        for name in splits[split]
    }
    if actual_images != expected_images:
        raise ValueError("Detection manifest does not exactly match the image tree")

    return {
        "manifest_sha256": _sha256(manifest_file),
        "dataset_tree_sha256": tree_digest.hexdigest(),
        "source": manifest.get("source"),
        "seed": manifest.get("seed"),
        "splits": split_counts,
        "image_count": len(seen_names),
        "label_object_count": label_count,
    }
