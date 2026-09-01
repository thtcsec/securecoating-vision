"""Evaluate a real-data CoatingVision checkpoint on the untouched test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation.dataset_manifest import validate_detection_dataset_manifest  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    weights = Path(args.weights).resolve()
    model = YOLO(str(weights))
    dataset_root = ROOT / "data" / "coatingvision_real_detect"
    provenance = validate_detection_dataset_manifest(
        dataset_root / "manifest.json", dataset_root
    )
    metrics = model.val(
        data=str(ROOT / "configs" / "coatingvision_real_detect.yaml"),
        split="test", imgsz=args.imgsz, device=args.device, plots=True,
    )
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": "CoatingVision detection subset, Figshare DOI 10.6084/m9.figshare.29260121.v1, CC BY 4.0",
        "evidence_class": "public_real_optical_image_split",
        "factory_roll_disjoint": False,
        "split": "fixed image-disjoint test split (seed 71); not factory roll-disjoint",
        "weights": str(weights.relative_to(ROOT)),
        "weights_sha256": sha256(weights),
        "model_task": model.task,
        "class_names": {str(key): value for key, value in model.names.items()},
        "imgsz": args.imgsz,
        "dataset_provenance": provenance,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "command": (
            f"python scripts/evaluate_coatingvision_real.py --weights "
            f"{weights.relative_to(ROOT).as_posix()} --imgsz {args.imgsz} --device {args.device}"
        ),
        "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
        "speed_ms_per_image": {
            key: float(value) for key, value in metrics.speed.items()
        },
    }
    target = ROOT / "reports" / "coatingvision_real_test_metrics.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
