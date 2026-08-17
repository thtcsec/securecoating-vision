"""Create a deterministic YOLO detection split from the public CoatingVision data.

The raw archive is deliberately kept outside version control.  This script only
uses images that have released bounding-box labels, and writes a manifest with
the source DOI and exact split membership for reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "external" / "coatingvision" / "detection"
DEST = ROOT / "data" / "coatingvision_real_detect"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()
    image_dir, label_dir = RAW / "images", RAW / "labels"
    pairs = sorted(
        (label.with_suffix(".jpg"), label)
        for label in label_dir.glob("*.txt")
        if (image_dir / label.with_suffix(".jpg").name).is_file()
    )
    pairs = [(image_dir / image.name, label) for image, label in pairs]
    if not pairs:
        raise FileNotFoundError("No CoatingVision detection image/label pairs found")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n = len(pairs)
    cuts = {"train": int(n * 0.70), "val": int(n * 0.85), "test": n}
    splits = {
        "train": pairs[: cuts["train"]],
        "val": pairs[cuts["train"] : cuts["val"]],
        "test": pairs[cuts["val"] :],
    }
    manifest = {
        "source": "CoatingVision, Figshare DOI 10.6084/m9.figshare.29260121.v1, CC BY 4.0",
        "seed": args.seed,
        "classes": {"0": "surface_crack", "1": "delamination_crack"},
        "splits": {},
    }
    for split, items in splits.items():
        out_images, out_labels = DEST / "images" / split, DEST / "labels" / split
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)
        manifest["splits"][split] = [image.name for image, _ in items]
        for image, label in items:
            shutil.copy2(image, out_images / image.name)
            shutil.copy2(label, out_labels / label.name)
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: len(value) for key, value in splits.items()}, indent=2))


if __name__ == "__main__":
    main()
