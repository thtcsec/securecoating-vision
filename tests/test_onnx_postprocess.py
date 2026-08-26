"""Artifact-independent ONNX postprocessing contract tests."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference.onnx_engine import InferenceEngine


class TestONNXPostprocess(unittest.TestCase):
    def test_instance_mask_is_cropped_to_detection_box(self):
        engine = InferenceEngine.__new__(InferenceEngine)
        engine.imgsz = 8
        engine.conf_thresh = 0.25
        engine.iou_thresh = 0.45
        prediction = np.array(
            [4.0, 4.0, 2.0, 2.0, 0.9, 0.0, 0.0, 0.0, 10.0],
            dtype=np.float32,
        ).reshape(1, 9, 1)
        prototypes = np.full((1, 1, 2, 2), 10.0, dtype=np.float32)
        result = engine._process_yolo_seg(
            prediction,
            prototypes,
            {"orig_shape": (8, 8), "pad_top": 0, "pad_left": 0, "scale": 1.0, "new_shape": (8, 8)},
            num_classes=4,
        )
        mask = result["detections"][0]["mask"]
        allowed = np.zeros((8, 8), dtype=np.uint8)
        allowed[3:5, 3:5] = 1
        self.assertEqual(int((mask * (1 - allowed)).sum()), 0)
        self.assertGreater(int(mask.sum()), 0)


if __name__ == "__main__":
    unittest.main()
