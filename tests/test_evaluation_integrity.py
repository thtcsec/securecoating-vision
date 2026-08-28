"""Evaluation provenance and leakage safeguards."""

import sys
import tempfile
import unittest
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.run_evaluation import assert_no_dataset_overlap, evaluate_sample_predictions
from src.evaluation.dataset_manifest import (
    validate_detection_dataset_manifest,
    validate_roll_disjoint_manifest,
)


class TestEvaluationIntegrity(unittest.TestCase):
    @staticmethod
    def _write_detection_fixture(root: Path, *, invalid_coordinate=False):
        splits = {"train": ["train.jpg"], "val": ["val.jpg"], "test": ["test.jpg"]}
        for split, names in splits.items():
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)
            for name in names:
                (root / "images" / split / name).write_bytes(f"image-{split}".encode())
                coordinate = "1.2" if invalid_coordinate and split == "test" else "0.5"
                (root / "labels" / split / f"{Path(name).stem}.txt").write_text(
                    f"0 {coordinate} 0.5 0.2 0.2\n", encoding="utf-8"
                )
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "source": "test fixture",
                    "seed": 71,
                    "classes": {"0": "defect"},
                    "splits": splits,
                }
            ),
            encoding="utf-8",
        )
        return manifest

    def test_detection_manifest_validates_complete_disjoint_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_detection_fixture(root)
            result = validate_detection_dataset_manifest(str(manifest), str(root))
            self.assertEqual(result["splits"], {"train": 1, "val": 1, "test": 1})
            self.assertEqual(result["image_count"], 3)
            self.assertEqual(result["label_object_count"], 3)
            self.assertEqual(len(result["dataset_tree_sha256"]), 64)

    def test_detection_manifest_rejects_split_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_detection_fixture(root)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["splits"]["test"] = ["train.jpg"]
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_detection_dataset_manifest(str(manifest), str(root))

    def test_detection_manifest_rejects_invalid_yolo_coordinate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_detection_fixture(root, invalid_coordinate=True)
            with self.assertRaises(ValueError):
                validate_detection_dataset_manifest(str(manifest), str(root))

    def test_hash_overlap_is_rejected_even_when_filename_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "evaluation"
            reference = root / "reference" / "images" / "val"
            evaluation.mkdir(parents=True)
            reference.mkdir(parents=True)
            (evaluation / "renamed.png").write_bytes(b"same image bytes")
            (reference / "original.png").write_bytes(b"same image bytes")
            with self.assertRaises(RuntimeError):
                assert_no_dataset_overlap(str(evaluation), str(root / "reference"))

    def test_missing_reference_dataset_fails_leakage_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluation = Path(directory) / "evaluation"
            evaluation.mkdir()
            (evaluation / "image.png").write_bytes(b"image")
            with self.assertRaises(FileNotFoundError):
                assert_no_dataset_overlap(str(evaluation), str(Path(directory) / "missing"))

    def test_instance_metric_rejects_class_wide_mask_fallback(self):
        gt = [{
            "class_id": 0,
            "bbox_xyxy": [0, 0, 5, 5],
            "mask": np.ones((8, 8), dtype=np.uint8),
        }]
        detection = [{
            "class_id": 0,
            "box": [0, 0, 5, 5],
            "box_format": "xyxy",
            "confidence": 0.9,
        }]
        with self.assertRaises(ValueError):
            evaluate_sample_predictions(gt, detection, np.ones((8, 8), dtype=np.uint8))

    def test_box_and_mask_metrics_use_the_same_instance_pair(self):
        gt = [
            {"class_id": 0, "bbox_xyxy": [0, 0, 4, 4], "mask": np.zeros((8, 8), dtype=np.uint8)},
            {"class_id": 0, "bbox_xyxy": [4, 4, 8, 8], "mask": np.zeros((8, 8), dtype=np.uint8)},
        ]
        gt[0]["mask"][0:4, 0:4] = 1
        gt[1]["mask"][4:8, 4:8] = 1
        detection_mask = np.zeros((8, 8), dtype=np.uint8)
        detection_mask[4:8, 4:8] = 1
        result = evaluate_sample_predictions(
            gt,
            [{
                "class_id": 0,
                "box": [0, 0, 4, 4],
                "box_format": "xyxy",
                "confidence": 0.9,
                "mask": detection_mask,
            }],
            np.ones((8, 8), dtype=np.uint8),
        )[0]
        self.assertEqual(result["box_tp"], 1)
        self.assertEqual(result["mask_tp"], 0)
        self.assertEqual(result["mask_fp"], 1)
        self.assertEqual(result["mask_fn"], 1)

    def test_manifest_rejects_roll_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            artifacts.mkdir()
            paths = {}
            for name in ("train.jpg", "val.jpg", "test.jpg"):
                path = artifacts / name
                path.write_bytes(name.encode("ascii"))
                paths[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"splits": {
                "train": [{"path": "artifacts/train.jpg", "roll_id": "R1", "sha256": paths["train.jpg"]}],
                "val": [{"path": "artifacts/val.jpg", "roll_id": "R1", "sha256": paths["val.jpg"]}],
                "test": [{"path": "artifacts/test.jpg", "roll_id": "R2", "sha256": paths["test.jpg"]}],
            }}), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_roll_disjoint_manifest(str(manifest), str(root))

    def test_manifest_test_split_must_match_evaluation_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "evaluation" / "images").mkdir(parents=True)
            image = root / "evaluation" / "images" / "test.jpg"
            image.write_bytes(b"test")
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"splits": {
                "train": [{"path": "evaluation/images/test.jpg", "roll_id": "R1", "sha256": digest}],
                "val": [{"path": "evaluation/images/test.jpg", "roll_id": "R2", "sha256": digest}],
                "test": [{"path": "evaluation/images/test.jpg", "roll_id": "R3", "sha256": digest}],
            }}), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_roll_disjoint_manifest(
                    str(manifest), str(root), str(root / "evaluation")
                )


if __name__ == "__main__":
    unittest.main()
