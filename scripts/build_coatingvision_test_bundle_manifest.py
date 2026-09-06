"""Write data/coatingvision_real_test/manifest.json with integrity hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data" / "coatingvision_real_test"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    test_images = sorted((BUNDLE / "images" / "test").glob("*.jpg"))
    test_labels = sorted((BUNDLE / "labels" / "test").glob("*.txt"))
    if len(test_images) != 88 or len(test_labels) != 88:
        raise SystemExit(
            f"Expected 88 image/label pairs, found {len(test_images)}/{len(test_labels)}"
        )
    entries = []
    for image in test_images:
        label = BUNDLE / "labels" / "test" / f"{image.stem}.txt"
        if not label.is_file():
            raise SystemExit(f"Missing label for {image.name}")
        entries.append(
            {
                "image": image.name,
                "image_sha256": sha256(image),
                "label": label.name,
                "label_sha256": sha256(label),
            }
        )
    blob = "\n".join(
        f"{e['image']}:{e['image_sha256']}:{e['label_sha256']}" for e in entries
    ).encode()
    manifest = {
        "source": "CoatingVision, Figshare DOI 10.6084/m9.figshare.29260121.v1, CC BY 4.0",
        "evidence_class": "public_real_optical_image_split",
        "used_for_defense_rgb_metrics": True,
        "factory_roll_disjoint": False,
        "split": "fixed image-disjoint test split (seed 71); not factory roll-disjoint",
        "seed": 71,
        "classes": {"0": "surface_crack", "1": "delamination_crack"},
        "zip_safe": True,
        "ultralytics_train_val_stubs": True,
        "stub_note": (
            "images/train and images/val contain a single copied test pair so "
            "Ultralytics can resolve paths; they are not training evidence."
        ),
        "authoritative_report": "reports/coatingvision_real_test_metrics.json",
        "splits": {
            "train": ["stub_train.jpg"],
            "val": ["stub_val.jpg"],
            "test": [e["image"] for e in entries],
        },
        "test_count": 88,
        "test_entries": entries,
        "test_split_fingerprint_sha256": hashlib.sha256(blob).hexdigest(),
    }
    target = BUNDLE / "manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target} fingerprint={manifest['test_split_fingerprint_sha256']}")


if __name__ == "__main__":
    main()
