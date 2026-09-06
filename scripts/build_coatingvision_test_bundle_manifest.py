"""Regenerate data/coatingvision_real_test/manifest.json bit-compatible with the shipped schema.

Reads the on-disk ZIP-safe bundle (88 test pairs + Ultralytics stubs
``_zip_stub_train.jpg`` / ``_zip_stub_val.jpg``) and writes the same
``coatingvision-real-test/v1`` fields used by validators and tests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data" / "coatingvision_real_test"
STUB_TRAIN = "_zip_stub_train.jpg"
STUB_VAL = "_zip_stub_val.jpg"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _tree_hash(root: Path, *relative_dirs: str, suffixes: set[str]) -> str:
    digest = hashlib.sha256()
    files: list[Path] = []
    for rel in relative_dirs:
        base = root / rel
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix.lower() in suffixes:
                files.append(path)
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_manifest(bundle: Path = BUNDLE) -> dict:
    test_images = sorted((bundle / "images" / "test").glob("*.jpg"))
    test_labels = sorted((bundle / "labels" / "test").glob("*.txt"))
    if len(test_images) != 88 or len(test_labels) != 88:
        raise SystemExit(
            f"Expected 88 image/label pairs, found {len(test_images)}/{len(test_labels)}"
        )
    stub_train = bundle / "images" / "train" / STUB_TRAIN
    stub_val = bundle / "images" / "val" / STUB_VAL
    for stub in (stub_train, stub_val):
        if not stub.is_file():
            raise SystemExit(f"Missing Ultralytics stub image: {stub}")
        label = bundle / "labels" / stub.parent.name / f"{stub.stem}.txt"
        if not label.is_file():
            raise SystemExit(f"Missing Ultralytics stub label: {label}")

    test_names: list[str] = []
    label_sha256: dict[str, str] = {}
    for image in test_images:
        label = bundle / "labels" / "test" / f"{image.stem}.txt"
        if not label.is_file():
            raise SystemExit(f"Missing label for {image.name}")
        test_names.append(image.name)
        label_sha256[label.name] = sha256_file(label)

    return {
        "schema_version": "coatingvision-real-test/v1",
        "evidence_class": "public_real_optical_image_split",
        "used_for_defense_rgb_metrics": True,
        "factory_roll_disjoint": False,
        "seed": 71,
        "source": "CoatingVision, Figshare DOI 10.6084/m9.figshare.29260121.v1, CC BY 4.0",
        "source_doi": "10.6084/m9.figshare.29260121.v1",
        "license": "CC BY 4.0",
        "purpose": (
            "ZIP-safe held-out image-disjoint test split for reproducing "
            "reports/coatingvision_real_test_metrics.json; not factory roll-disjoint validation"
        ),
        "classes": {"0": "surface_crack", "1": "delamination_crack"},
        "splits": {
            "train": [STUB_TRAIN],
            "val": [STUB_VAL],
            "test": list(test_names),
        },
        "split_roles": {
            "train": "ultralytics_stub_not_training_data",
            "val": "ultralytics_stub_not_metrics_evidence",
            "test": "held_out_image_disjoint_metrics_split",
        },
        "used_for_defense_rgb_metrics_by_split": {
            "train": False,
            "val": False,
            "test": True,
        },
        "test_image_names": list(test_names),
        "label_sha256": label_sha256,
        "tree_hashes": {
            "images": _tree_hash(bundle, "images", suffixes=IMAGE_SUFFIXES),
            "labels": _tree_hash(bundle, "labels", suffixes={".txt"}),
            "images_and_labels": _tree_hash(
                bundle, "images", "labels", suffixes=IMAGE_SUFFIXES | {".txt"}
            ),
        },
        "parent_full_detect_tree": (
            "data/coatingvision_real_detect (gitignored; train/val excluded from ZIP)"
        ),
        "image_source": "data/demo_real/images (same 88 test JPEGs)",
        "label_source": "data/coatingvision_real_detect/labels/test",
        "stub_note": (
            "train/val contain one stub image+label each so Ultralytics accepts "
            "the YAML; stubs are not training data and are not metrics evidence"
        ),
    }


def write_manifest(bundle: Path = BUNDLE) -> Path:
    payload = build_manifest(bundle)
    target = bundle / "manifest.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def main() -> None:
    target = write_manifest()
    payload = json.loads(target.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "wrote": target.as_posix(),
                "schema_version": payload["schema_version"],
                "test_count": len(payload["test_image_names"]),
                "tree_hashes": payload["tree_hashes"],
                "manifest_sha256": sha256_file(target),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
