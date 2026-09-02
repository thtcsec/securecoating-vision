"""Inspect CoatingVision public metadata for factory roll/coil grouping keys.

The Figshare release is image-attributed. This module records which keys exist
and refuses to invent factory roll IDs for a roll-disjoint split.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FACTORY_GROUP_KEYS = (
    "roll_id",
    "roll",
    "coil_id",
    "coil",
    "web_id",
    "campaign_id",
    "lot_id",
    "factory_lot",
    "electrode_roll",
)


def _normalized_keys(values: Iterable[str]) -> List[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()})


def _factory_keys_in(names: Sequence[str]) -> List[str]:
    lookup = {name.lower() for name in names}
    return [key for key in FACTORY_GROUP_KEYS if key in lookup]


def inspect_coatingvision_grouping(project_root: Path | None = None) -> Dict[str, Any]:
    """Return an inventory of grouping fields. Never synthesizes roll IDs."""
    root = Path(project_root or PROJECT_ROOT)
    demo_path = root / "data" / "demo_real" / "manifest.json"
    detection_path = root / "data" / "coatingvision_real_detect" / "manifest.json"
    labels_path = (
        root / "data" / "external" / "coatingvision" / "classification" / "labels.csv"
    )
    metrics_path = root / "reports" / "coatingvision_real_test_metrics.json"

    demo = json.loads(demo_path.read_text(encoding="utf-8"))
    sample_keys = _normalized_keys(
        key for sample in demo.get("samples") or [] if isinstance(sample, dict) for key in sample
    )

    detection_keys: List[str] = []
    detection_present = detection_path.is_file()
    if detection_present:
        detection = json.loads(detection_path.read_text(encoding="utf-8"))
        detection_keys = _normalized_keys(detection.keys())
        if "roll_id" in json.dumps(detection).lower():
            detection_keys = _normalized_keys([*detection_keys, "roll_id"])

    labels_columns: List[str] = []
    labels_present = labels_path.is_file()
    if labels_present:
        with labels_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            labels_columns = list(reader.fieldnames or [])

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    factory_keys = _factory_keys_in([*sample_keys, *detection_keys, *labels_columns])
    return {
        "evidence_class": "public_real_optical_image_split",
        "factory_roll_disjoint": False,
        "factory_group_keys_found": factory_keys,
        "can_build_roll_disjoint_split": False,
        "blocker": (
            "CoatingVision public metadata has image filenames and class flags only; "
            "no factory roll, coil, or web identifiers."
        ),
        "demo_manifest": str(demo_path),
        "demo_sample_keys": sample_keys,
        "detection_manifest_present": detection_present,
        "detection_manifest_keys": detection_keys,
        "classification_csv_present": labels_present,
        "classification_columns": labels_columns,
        "metrics_factory_roll_disjoint": bool(metrics.get("factory_roll_disjoint")),
        "metrics_split": metrics.get("split"),
    }
