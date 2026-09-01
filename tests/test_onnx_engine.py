"""Unit tests for ONNX Runtime inference engine."""

import os
import sys
import unittest

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from inference.onnx_engine import InferenceEngine  # noqa: E402

ONNX_PATH = os.path.join(ROOT, "outputs", "model.onnx")
TEST_IMG = os.path.join(ROOT, "data", "demo_real", "images", "image_1548.jpg")
REAL_CLASS_NAMES = {0: "surface_crack", 1: "delamination_crack"}


import importlib.util

ORT_AVAILABLE = importlib.util.find_spec("onnxruntime") is not None


class TestOnnxEngine(unittest.TestCase):
    @unittest.skipUnless(ORT_AVAILABLE and os.path.isfile(ONNX_PATH), "onnxruntime package or outputs/model.onnx not present")
    def test_model_loads(self):
        engine = InferenceEngine(
            ONNX_PATH,
            imgsz=512,
            conf_thresh=0.35,
            num_classes=2,
            class_names=REAL_CLASS_NAMES,
        )
        self.assertTrue(engine.is_loaded)
        self.assertIn("ExecutionProvider", engine.active_provider)

    @unittest.skipUnless(
        ORT_AVAILABLE and os.path.isfile(ONNX_PATH) and os.path.isfile(TEST_IMG),
        "onnxruntime package, ONNX model, or test image missing",
    )
    def test_infer_detects_real_coating_crack_sample(self):
        engine = InferenceEngine(
            ONNX_PATH,
            imgsz=512,
            conf_thresh=0.35,
            num_classes=2,
            class_names=REAL_CLASS_NAMES,
        )
        image = cv2.imread(TEST_IMG)
        self.assertIsNotNone(image)
        result = engine.infer(image)
        self.assertIn("segmentation_mask", result)
        self.assertEqual(result["segmentation_mask"].shape[:2], image.shape[:2])
        self.assertGreaterEqual(result["num_defects"], 1)
        self.assertGreater(result["latency_ms"], 0.0)
        class_names = {d["class_name"] for d in result["detections"]}
        self.assertIn("surface_crack", class_names)

    def test_missing_model_is_not_loaded(self):
        engine = InferenceEngine(
            os.path.join(ROOT, "outputs", "definitely_missing.onnx"),
            imgsz=640,
        )
        self.assertFalse(engine.is_loaded)

    def test_nms_and_sigmoid_helpers(self):
        boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [1.0, 1.0, 11.0, 11.0],
                [50.0, 50.0, 60.0, 60.0],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        keep = InferenceEngine._nms(boxes, scores, iou_threshold=0.5)
        self.assertEqual(keep[0], 0)
        self.assertIn(2, keep)
        sig = InferenceEngine._sigmoid(np.array([0.0, 100.0, -100.0]))
        self.assertAlmostEqual(float(sig[0]), 0.5, places=5)


if __name__ == "__main__":
    unittest.main()
