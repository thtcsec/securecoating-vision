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

from scripts.run_evaluation import (
    assert_no_dataset_overlap,
    evaluate_sample_predictions,
    load_gt_labels,
)
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
        self.assertEqual(result["mask_fn"], 2)

    def test_mask_iou_failures_count_as_false_negatives(self):
        gt = []
        detections = []
        for y in (0, 4):
            gt_mask = np.zeros((8, 8), dtype=np.uint8)
            gt_mask[y:y + 4, 0:4] = 1
            gt.append({"class_id": 0, "bbox_xyxy": [0, y, 4, y + 4], "mask": gt_mask})
            detections.append({
                "class_id": 0,
                "box": [0, y, 4, y + 4],
                "box_format": "xyxy",
                "confidence": 0.9,
                "mask": np.zeros((8, 8), dtype=np.uint8),
            })
        result = evaluate_sample_predictions(
            gt, detections, np.zeros((8, 8), dtype=np.uint8)
        )[0]
        self.assertEqual(result["mask_tp"], 0)
        self.assertEqual(result["mask_fp"], 2)
        self.assertEqual(result["mask_fn"], 2)

    def test_evaluation_label_parser_rejects_invalid_class_and_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "bad.txt"
            for payload in (
                "9 0 0 1 0 1 1\n",
                "0 -0.1 0 1 0 1 1\n",
                "0 nan 0 1 0 1 1\n",
                "0 0.5 0.5 0.5 0.5 0.5 0.5\n",
            ):
                label.write_text(payload, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_gt_labels(str(label), img_h=8, img_w=8)

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
            (root / "evaluation" / "labels").mkdir(parents=True)
            image = root / "evaluation" / "images" / "test.jpg"
            image.write_bytes(b"test")
            label = root / "evaluation" / "labels" / "test.txt"
            label.write_text("", encoding="utf-8")
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            label_digest = hashlib.sha256(label.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"splits": {
                "train": [{"path": "evaluation/images/test.jpg", "roll_id": "R1", "sha256": digest}],
                "val": [{"path": "evaluation/images/test.jpg", "roll_id": "R2", "sha256": digest}],
                "test": [{"path": "evaluation/images/test.jpg", "roll_id": "R3", "sha256": digest,
                          "label_path": "evaluation/labels/test.txt", "label_sha256": label_digest}],
            }}), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_roll_disjoint_manifest(
                    str(manifest), str(root), str(root / "evaluation")
                )

    def test_evaluation_manifest_rejects_changed_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "evaluation" / "images").mkdir(parents=True)
            (root / "evaluation" / "labels").mkdir(parents=True)
            image = root / "evaluation" / "images" / "test.jpg"
            label = root / "evaluation" / "labels" / "test.txt"
            image.write_bytes(b"test")
            label.write_text("0 0 0 1 0 1 1\n", encoding="utf-8")
            artifacts = root / "artifacts"
            artifacts.mkdir()
            train_image = artifacts / "train.jpg"
            val_image = artifacts / "val.jpg"
            train_image.write_bytes(b"train")
            val_image.write_bytes(b"val")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"splits": {
                "train": [{"path": "artifacts/train.jpg", "roll_id": "R1",
                           "sha256": hashlib.sha256(train_image.read_bytes()).hexdigest()}],
                "val": [{"path": "artifacts/val.jpg", "roll_id": "R2",
                         "sha256": hashlib.sha256(val_image.read_bytes()).hexdigest()}],
                "test": [{"path": "evaluation/images/test.jpg", "roll_id": "R3",
                          "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                          "label_path": "evaluation/labels/test.txt",
                          "label_sha256": hashlib.sha256(label.read_bytes()).hexdigest()}],
            }}), encoding="utf-8")
            label.write_text("tampered", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_roll_disjoint_manifest(
                    str(manifest), str(root), str(root / "evaluation")
                )

    def test_checked_in_synthetic_evaluation_report_is_honest(self):
        report_path = ROOT / "reports" / "evaluation_results.json"
        self.assertTrue(report_path.is_file())
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["evidence_class"], "SYNTHETIC_EVALUATOR_FIXTURE")
        self.assertFalse(payload["used_for_defense_rgb_metrics"])
        self.assertTrue(payload["development_image_reuse"])
        self.assertTrue(payload["cross_taxonomy_run"])
        self.assertIn("bbox-derived", payload["segmentation_metrics"]["metric_semantics"])
        self.assertNotIn("coating_defects", payload["provenance"].get("reference_dataset_dir", ""))
        self.assertTrue(
            str(payload["provenance"].get("reference_dataset_dir", "")).replace("\\", "/").startswith(
                "data/evaluation/reference"
            )
        )
        command = (ROOT / "reports" / "evaluation_command.txt").read_text(encoding="utf-8")
        self.assertIn("build_synthetic_evaluation_manifest.py", command)
        self.assertIn("data/evaluation/reference", command)
        self.assertNotIn("coating_defects", command)
        self.assertNotIn("coatingvision_real_detect", command)
        manifest = json.loads(
            (ROOT / "reports" / "synthetic_evaluation_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["evidence_class"], "SYNTHETIC_EVALUATOR_FIXTURE")
        for split in ("train", "val"):
            for entry in manifest["splits"][split]:
                self.assertTrue(entry["path"].startswith("data/evaluation/reference/"))
                self.assertFalse(entry["path"].startswith("data/coating_defects/"))
        csv_text = (ROOT / "reports" / "evaluation_results.csv").read_text(encoding="utf-8")
        self.assertIn("evidence_class=SYNTHETIC_EVALUATOR_FIXTURE", csv_text)
        self.assertIn("used_for_defense_rgb_metrics=False", csv_text)

    def test_synthetic_manifest_builder_is_self_contained(self):
        from scripts.build_synthetic_evaluation_manifest import build_manifest

        payload = build_manifest()
        self.assertEqual(payload["evidence_class"], "SYNTHETIC_EVALUATOR_FIXTURE")
        self.assertTrue(payload["development_image_reuse"])
        train_path = ROOT / payload["splits"]["train"][0]["path"]
        val_path = ROOT / payload["splits"]["val"][0]["path"]
        self.assertTrue(train_path.is_file())
        self.assertTrue(val_path.is_file())
        self.assertNotEqual(
            hashlib.sha256(train_path.read_bytes()).hexdigest(),
            hashlib.sha256(val_path.read_bytes()).hexdigest(),
        )
        # ZIP-safe reference stubs must not overlap fixture test imagery.
        assert_no_dataset_overlap(
            str(ROOT / "data/evaluation/images"),
            str(ROOT / "data/evaluation/reference"),
        )


if __name__ == "__main__":
    unittest.main()
