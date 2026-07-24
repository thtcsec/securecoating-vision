"""
SecureCoating-Vision: Official Ultralytics mAP Validation Script
==================================================================
Runs official Ultralytics segmentation validator on outputs/best.pt using
configs/evaluation.yaml to compute true Box & Mask mAP metrics:
- Box mAP50, Box mAP50-95
- Mask mAP50, Mask mAP50-95

Outputs automatically to:
- reports/ultralytics_validation_results.json

Usage:
    python scripts/run_ultralytics_validation.py
"""

import os
import sys
import json
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

logging.basicConfig(level=logging.INFO)


def run_ultralytics_validation():
    from ultralytics import YOLO

    model_pt = "outputs/best.pt"
    config_yaml = "configs/evaluation.yaml"

    if not os.path.exists(model_pt):
        print(f"[ERROR] PyTorch weights not found at {model_pt}")
        sys.exit(1)

    if not os.path.exists(config_yaml):
        print(f"[ERROR] Evaluation config not found at {config_yaml}")
        sys.exit(1)

    print()
    print("=" * 70)
    print("  SecureCoating-Vision: Ultralytics Segmentation Validation")
    print("  Computing true Box & Mask mAP metrics on demonstration evaluation subset")
    print("=" * 70)

    model = YOLO(model_pt)
    results = model.val(
        data=config_yaml,
        imgsz=640,
        split="val",
        plots=False,
        save_json=False,
        verbose=True,
    )

    box_map50 = float(results.box.map50)
    box_map = float(results.box.map)
    mask_map50 = float(results.seg.map50)
    mask_map = float(results.seg.map)

    print("\n" + "=" * 70)
    print("  ULTRALYTICS mAP VALIDATION RESULTS")
    print("=" * 70)
    print(f"  Box mAP@0.50:       {box_map50*100:.2f}%")
    print(f"  Box mAP@0.50:0.95:  {box_map*100:.2f}%")
    print(f"  Mask mAP@0.50:      {mask_map50*100:.2f}%")
    print(f"  Mask mAP@0.50:0.95: {mask_map*100:.2f}%")
    print("=" * 70)

    out_json = os.path.join(PROJECT_ROOT, "reports", "ultralytics_validation_results.json")
    res_dict = {
        "model": model_pt,
        "config": config_yaml,
        "box_map50": round(box_map50, 4),
        "box_map50_95": round(box_map, 4),
        "mask_map50": round(mask_map50, 4),
        "mask_map50_95": round(mask_map, 4),
    }

    with open(out_json, "w") as f:
        json.dump(res_dict, f, indent=2)

    print(f"  Saved: {out_json}\n")
    return res_dict


if __name__ == "__main__":
    run_ultralytics_validation()
