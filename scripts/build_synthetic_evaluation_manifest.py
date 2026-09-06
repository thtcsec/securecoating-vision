"""Build a ZIP-safe synthetic evaluator manifest (no coating_defects dependency).

Train/val entries are unique stub images under data/evaluation/reference/.
Test entries hash every image+label in data/evaluation/{images,labels}.

This fixture intentionally reuses development imagery in the test split; the
manifest records that fact and must never be treated as an independent
benchmark or as the CoatingVision RGB defense metrics source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "data" / "evaluation"
REFERENCE_ROOT = EVAL_ROOT / "reference"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entry(path: Path, roll_id: str, label_path: Path | None = None) -> dict:
    item = {
        "path": path.relative_to(ROOT).as_posix(),
        "roll_id": roll_id,
        "sha256": sha256(path),
    }
    if label_path is not None:
        item["label_path"] = label_path.relative_to(ROOT).as_posix()
        item["label_sha256"] = sha256(label_path)
    return item


def _write_unique_stub(path: Path, rgb: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    frame[:, :] = rgb
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Failed to write reference stub: {path}")


def ensure_reference_stubs() -> tuple[Path, Path]:
    """Create deterministic unique train/val stubs that do not overlap test images."""
    train_path = REFERENCE_ROOT / "images" / "train" / "synthetic_ref_train.jpg"
    val_path = REFERENCE_ROOT / "images" / "val" / "synthetic_ref_val.jpg"
    _write_unique_stub(train_path, (11, 22, 33))
    _write_unique_stub(val_path, (44, 55, 66))
    if sha256(train_path) == sha256(val_path):
        raise RuntimeError("Reference stubs must have distinct content hashes")
    return train_path, val_path


def build_manifest() -> dict:
    train_path, val_path = ensure_reference_stubs()
    evaluation_images = sorted(
        path
        for path in (EVAL_ROOT / "images").iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not evaluation_images:
        raise FileNotFoundError("No synthetic evaluation images found under data/evaluation/images")

    # Stubs must not collide with fixture test imagery.
    stub_hashes = {sha256(train_path), sha256(val_path)}
    for image in evaluation_images:
        if sha256(image) in stub_hashes:
            raise RuntimeError(f"Reference stub collides with evaluation image: {image.name}")

    evaluation_entries = []
    for image in evaluation_images:
        label = EVAL_ROOT / "labels" / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Missing evaluation label: {label}")
        evaluation_entries.append(entry(image, f"synthetic-test-{image.stem}", label))

    return {
        "schema_version": "synthetic-evaluator-manifest/v1",
        "evidence_class": "SYNTHETIC_EVALUATOR_FIXTURE",
        "used_for_defense_rgb_metrics": False,
        "development_image_reuse": True,
        "cross_taxonomy_run": True,
        "fixture_taxonomy": {
            "0": "scratch",
            "1": "void",
            "2": "blister",
            "3": "delamination",
        },
        "model_taxonomy_note": (
            "Runtime CoatingVision ONNX detector is 2-class "
            "(surface_crack / delamination_crack). Numbers from this fixture are "
            "cross-taxonomy pipeline smoke only."
        ),
        "note": (
            "Self-contained ZIP-safe fixture. Train/val are unique stubs under "
            "data/evaluation/reference. Test images intentionally reuse development "
            "imagery; not an independent benchmark and not RGB defense evidence."
        ),
        "splits": {
            "train": [entry(train_path, "synthetic-reference-train")],
            "val": [entry(val_path, "synthetic-reference-val")],
            "test": evaluation_entries,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ZIP-safe synthetic evaluator manifest under data/evaluation."
    )
    parser.add_argument(
        "--output",
        default="reports/synthetic_evaluation_manifest.json",
        help="Manifest path relative to the project root",
    )
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if ROOT not in output.parents and output != ROOT:
        raise ValueError("Manifest output must remain inside the project root")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_manifest()
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(output.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
