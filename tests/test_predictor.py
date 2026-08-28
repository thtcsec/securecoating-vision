"""Unit tests for CoatingPredictor (ONNX-primary path)."""

import os
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

from inference.predictor import CoatingPredictor  # noqa: E402

ONNX_PATH = os.path.join(ROOT, "outputs", "model.onnx")
TEST_IMG = os.path.join(ROOT, "data", "test_set", "images", "defect_val_00001.jpg")


def _load_config():
    with open(os.path.join(ROOT, "configs", "model.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestPredictor(unittest.TestCase):
    def test_container_device_override_forces_cpu(self):
        cfg = _load_config()
        cfg["inference"]["device"] = "cuda"
        with patch.dict(os.environ, {"SECURECOATING_INFERENCE_DEVICE": "cpu"}), patch.object(
            CoatingPredictor, "_init_yolo_engine", return_value=None
        ), patch.object(CoatingPredictor, "_init_onnx_engine", return_value=None):
            predictor = CoatingPredictor(cfg)
        self.assertEqual(predictor.device.type, "cpu")

    def test_invalid_container_device_override_is_rejected(self):
        with patch.dict(os.environ, {"SECURECOATING_INFERENCE_DEVICE": "gpu-magic"}):
            with self.assertRaises(ValueError):
                CoatingPredictor(_load_config())

    def test_sensor_fallback_flag_in_result(self):
        cfg = _load_config()
        cfg["inference"]["device"] = "cpu"
        predictor = CoatingPredictor(cfg)
        optical = np.random.randint(40, 200, (320, 320, 3), dtype=np.uint8)
        result = predictor.predict(optical, thermal=None, height=None)
        self.assertTrue(result["fallback_active"])
        self.assertEqual(result["status"], "Degraded Mode")
        self.assertIn("engine", result)
        self.assertEqual(result["model_version"], str(cfg["model"]["version"]))

    @unittest.skipUnless(
        os.path.isfile(ONNX_PATH) and os.path.isfile(TEST_IMG),
        "ONNX model or test image missing",
    )
    def test_onnx_primary_path_on_test_image(self):
        cfg = _load_config()
        cfg["inference"]["device"] = "cpu"
        predictor = CoatingPredictor(cfg)
        self.assertTrue(predictor.onnx_available)
        image = cv2.imread(TEST_IMG)
        thermal = np.full(image.shape[:2], 40.0, dtype=np.float32)
        height = np.zeros(image.shape[:2], dtype=np.float32)
        result = predictor.predict(image, thermal, height)
        self.assertFalse(result["fallback_active"])
        self.assertTrue(
            str(result["engine"]).startswith("ONNX Runtime")
            or str(result["engine"]).startswith("YOLO")
        )
        self.assertGreaterEqual(len(result["detections"]), 1)
        self.assertEqual(len(result["class_probabilities"]), cfg["model"]["num_classes"])
        self.assertGreater(sum(result["class_probabilities"][1:]), 0.0)

    def test_probs_from_detections_mapping(self):
        dets = [
            {"class_id": 0, "confidence": 0.9},
            {"class_id": 2, "confidence": 0.4},
        ]
        probs, predicted = CoatingPredictor._probs_from_detections(dets, num_classes=5)
        self.assertEqual(len(probs), 5)
        self.assertAlmostEqual(probs[1], 0.9)
        self.assertAlmostEqual(probs[3], 0.4)
        self.assertEqual(predicted, 1)


if __name__ == "__main__":
    unittest.main()
