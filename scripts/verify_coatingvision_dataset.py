"""Verify the local CoatingVision detection dataset without changing it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.dataset_manifest import validate_detection_dataset_manifest  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default="data/coatingvision_real_detect",
        help="Dataset root containing manifest.json, images/, and labels/.",
    )
    args = parser.parse_args(argv)
    root = (PROJECT_ROOT / args.dataset_root).resolve()
    result = validate_detection_dataset_manifest(root / "manifest.json", root)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
