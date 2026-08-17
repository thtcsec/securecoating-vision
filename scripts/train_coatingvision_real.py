"""Train and validate a YOLO detector on the real CoatingVision split."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", action="store_true", help="Resume the latest interrupted run.")
    args = parser.parse_args()
    data = ROOT / "configs" / "coatingvision_real_detect.yaml"
    run_dir = ROOT / "outputs" / "coatingvision_real" / f"yolo26n_{args.epochs}ep"
    model = YOLO(str(run_dir / "weights" / "last.pt" if args.resume else ROOT / "yolo26n.pt"))
    if args.resume:
        model.train(resume=True)
        return
    model.train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project=str(ROOT / "outputs" / "coatingvision_real"),
        name=f"yolo26n_{args.epochs}ep", seed=71, deterministic=True,
        workers=4, patience=10, save=True, plots=True,
    )


if __name__ == "__main__":
    main()
