"""Run the project's trained YOLO segmentation model on a real CoatingVision image.

This utility is intentionally evidence-first: the input is a CC BY 4.0 image
from the CoatingVision dataset, and the output includes the exact raw model
detections plus a rendered overlay. It does not fabricate a defect when no
detection is returned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from inference.predictor import CoatingPredictor  # noqa: E402
from inference.multi_stage_pipeline import (  # noqa: E402
    MultiStageIndustrialPipeline,
    serialize_detection_metadata,
)
from libad.protocol import git_source_provenance  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_evidence_path(path: Path) -> str:
    """Prefer a repository-relative path, but support explicit external output dirs."""
    resolved = path.resolve()
    try:
        evidence_path = resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        evidence_path = resolved
    return str(evidence_path).replace("\\", "/")


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
    pipeline_summary = pipeline_result.to_summary_dict()
    detector_metadata = [serialize_detection_metadata(item) for item in result["detections"]]
    if pipeline_summary["raw_detections"] != detector_metadata:
        raise RuntimeError("Repeated inference produced inconsistent raw detections; artifact not written")
    if pipeline_summary["raw_detections_count"] != len(result["detections"]):
        raise RuntimeError("raw_detections_count does not match the serialized detector output")
    if pipeline_summary["overall_verdict"] != pipeline_summary["plc_gate_action"]:
        raise RuntimeError("Verdict and planned PLC gate action disagree; artifact not written")
    if pipeline_summary["overall_verdict"] == "HOLD":
        if pipeline_summary["standards_compliant"]:
            raise RuntimeError("HOLD cannot be serialized as standards_compliant=true")
        if pipeline_summary["root_cause_report"].get("severity_level") == "NOMINAL":
            raise RuntimeError("HOLD cannot be serialized as a nominal/process-in-control state")

    model_path = None
    if predictor.yolo_available:
        model_path = Path(predictor.yolo_engine.model_path)
    elif predictor.onnx_available:
        model_path = Path(predictor.onnx_engine.model_path)
    if model_path is None or not model_path.is_file():
        raise RuntimeError("A loaded model artifact is required to generate external evidence")
    source_provenance = git_source_provenance()

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
        "artifact_schema_version": "2.0",
        "evidence_class": "external_dataset_model_output",
        "comparable_to_paper": False,
        "input_image": str(image_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "input_sha256": _sha256(image_path),
        "input_source": "CoatingVision dataset, CC BY 4.0, Figshare DOI 10.6084/m9.figshare.29260121.v1",
        "input_is_real_optical_data": True,
        "secondary_sensor_data": "not supplied; model ran in RGB-only degraded mode",
        "engine": result["engine"],
        "model_artifact": str(model_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "model_sha256": _sha256(model_path),
        "commit_sha": source_provenance["commit"],
        "source_tree_dirty": source_provenance["working_tree_dirty"],
        "source_diff_sha256": source_provenance["source_diff_sha256"],
        "latency_ms": round(float(result["latency_ms"]), 2),
        "detections": detector_metadata,
        "seven_stage_pipeline": pipeline_summary,
        "output_overlay": _portable_evidence_path(output_image),
    }
    with open(out_dir / "coatingvision_model_output.json", "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, allow_nan=False)

    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
