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
ZIP_SAFE_ROOT = ROOT / "data" / "coatingvision_real_test"
FULL_DETECT_ROOT = ROOT / "data" / "coatingvision_real_detect"
ZIP_SAFE_YAML = ROOT / "configs" / "coatingvision_real_test.yaml"
FULL_DETECT_YAML = ROOT / "configs" / "coatingvision_real_detect.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_dataset(dataset_root: Path | None, data_yaml: Path | None) -> tuple[Path, Path]:
    """Prefer the ZIP-safe test bundle when present; else full local detect tree."""
    if dataset_root is not None:
        root = dataset_root if dataset_root.is_absolute() else (ROOT / dataset_root)
        root = root.resolve()
        if data_yaml is not None:
            yaml_path = data_yaml if data_yaml.is_absolute() else (ROOT / data_yaml)
            return root, yaml_path.resolve()
        if root == ZIP_SAFE_ROOT.resolve() or root.name == "coatingvision_real_test":
            return root, ZIP_SAFE_YAML.resolve()
        return root, FULL_DETECT_YAML.resolve()

    if data_yaml is not None:
        yaml_path = data_yaml if data_yaml.is_absolute() else (ROOT / data_yaml)
        yaml_path = yaml_path.resolve()
        if yaml_path.name == "coatingvision_real_test.yaml":
            return ZIP_SAFE_ROOT.resolve(), yaml_path
        if yaml_path.name == "coatingvision_real_detect.yaml":
            return FULL_DETECT_ROOT.resolve(), yaml_path
        preferred = ZIP_SAFE_ROOT if ZIP_SAFE_ROOT.is_dir() else FULL_DETECT_ROOT
        return preferred.resolve(), yaml_path

    if ZIP_SAFE_ROOT.is_dir() and (ZIP_SAFE_ROOT / "manifest.json").is_file():
        return ZIP_SAFE_ROOT.resolve(), ZIP_SAFE_YAML.resolve()
    return FULL_DETECT_ROOT.resolve(), FULL_DETECT_YAML.resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Ultralytics dataset YAML (default: coatingvision_real_test.yaml when present)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset root with manifest.json (default: data/coatingvision_real_test if present)",
    )
    args = parser.parse_args()
    weights = Path(args.weights)
    weights = weights.resolve() if weights.is_absolute() else (ROOT / weights).resolve()

    dataset_root, data_yaml = resolve_dataset(args.dataset_root, args.data)
    model = YOLO(str(weights))
    provenance = validate_detection_dataset_manifest(
        dataset_root / "manifest.json", dataset_root
    )
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        device=args.device,
        plots=True,
    )
    weights_rel = _rel(weights)
    dataset_rel = _rel(dataset_root)
    data_rel = _rel(data_yaml)
    zip_safe = (
        dataset_root == ZIP_SAFE_ROOT.resolve()
        or dataset_root.name == "coatingvision_real_test"
    )
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": "CoatingVision detection subset, Figshare DOI 10.6084/m9.figshare.29260121.v1, CC BY 4.0",
        "evidence_class": "public_real_optical_image_split",
        "factory_roll_disjoint": False,
        "split": "fixed image-disjoint test split (seed 71); not factory roll-disjoint",
        "weights": weights_rel,
        "weights_sha256": sha256(weights),
        "model_task": model.task,
        "class_names": {str(key): value for key, value in model.names.items()},
        "imgsz": args.imgsz,
        "dataset_root": dataset_rel,
        "data_yaml": data_rel,
        "dataset_provenance": provenance,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "final_artifact_provenance": "reports/submission_manifest.json",
        "zip_safe_dataset": "data/coatingvision_real_test",
        "zip_safe_dataset_used": zip_safe,
        "command": (
            f"python scripts/evaluate_coatingvision_real.py --weights {weights_rel} "
            f"--data {data_rel} --dataset-root {dataset_rel} "
            f"--imgsz {args.imgsz} --device {args.device}"
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
