"""Evaluate a real-data CoatingVision checkpoint on the untouched test split."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    model = YOLO(args.weights)
    metrics = model.val(
        data=str(ROOT / "configs" / "coatingvision_real_detect.yaml"),
        split="test", imgsz=args.imgsz, device=args.device, plots=True,
    )
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": "CoatingVision detection subset, Figshare DOI 10.6084/m9.figshare.29260121.v1, CC BY 4.0",
        "split": "fixed test split (seed 71)",
        "weights": args.weights,
        "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
    }
    target = ROOT / "reports" / "coatingvision_real_test_metrics.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
