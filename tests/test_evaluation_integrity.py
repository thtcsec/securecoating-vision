"""Evaluation provenance and leakage safeguards."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.run_evaluation import assert_no_dataset_overlap, evaluate_sample_predictions


class TestEvaluationIntegrity(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
