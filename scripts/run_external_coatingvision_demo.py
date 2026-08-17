"""Run the project's trained YOLO segmentation model on a real CoatingVision image.

This utility is intentionally evidence-first: the input is a CC BY 4.0 image
from the CoatingVision dataset, and the output includes the exact raw model
detections plus a rendered overlay. It does not fabricate a defect when no
detection is returned.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inference.predictor import CoatingPredictor  # noqa: E402
from inference.multi_stage_pipeline import MultiStageIndustrialPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        default="data/external/coatingvision/detection/images/image_1001.jpg",
        help="Path to a real CoatingVision optical image.",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/external_coatingvision_demo",
        help="Directory for the model-output overlay and JSON record.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional .pt model path; defaults to configs/model.yaml.",
    )
    args = parser.parse_args()

    image_path = (PROJECT_ROOT / args.image).resolve()
    out_dir = (PROJECT_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read {image_path}")

    with open(PROJECT_ROOT / "configs" / "model.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.weights:
        config["model"]["weights_path"] = args.weights
    predictor = CoatingPredictor(config)
    result = predictor.predict(image, thermal=None, height=None)
    pipeline = MultiStageIndustrialPipeline(predictor=predictor)
    pipeline_result = pipeline.execute_inspection(
        sample_id=image_path.stem,
        optical_rgb=image,
        thermal_raw=None,
        height_map=None,
        enable_plc_signal=False,
    )

    overlay = image.copy()
    for det in result["detections"]:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        label = f'{det["class_name"]} {det["confidence"]:.2f}'
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 210, 0), 2)
        cv2.putText(
            overlay, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 210, 0), 2, cv2.LINE_AA,
        )

    output_image = out_dir / "coatingvision_model_output.png"
    cv2.imwrite(str(output_image), overlay)
    evidence = {
        "input_image": str(image_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "input_source": "CoatingVision dataset, CC BY 4.0, Figshare DOI 10.6084/m9.figshare.29260121.v1",
        "input_is_real_optical_data": True,
        "secondary_sensor_data": "not supplied; model ran in RGB-only degraded mode",
        "engine": result["engine"],
        "latency_ms": round(float(result["latency_ms"]), 2),
        "detections": result["detections"],
        "seven_stage_pipeline": {
            **pipeline_result.to_summary_dict(),
            "raw_detections": pipeline_result.raw_detections,
            "measurement_policy": "No dimensional metrology or release decision in RGB-only mode.",
        },
        "output_overlay": str(output_image.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }
    with open(out_dir / "coatingvision_model_output.json", "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)

    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
