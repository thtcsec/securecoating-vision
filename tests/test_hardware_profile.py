"""Hardware-profile routing tests with deterministic synthetic capabilities."""

import os
import sys
import unittest
from unittest.mock import patch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from inference.hardware_profile import (  # noqa: E402
    HardwareCapabilities,
    resolve_inference_profile,
)


CONFIG = {
    "inference": {"hardware_profile": "auto", "device": "auto"},
    "hardware_profiles": {
        "edge": {"imgsz": 512, "precision": "fp32", "engine_priority": ["onnx", "yolo"]},
        "balanced": {"imgsz": 640, "precision": "fp32", "engine_priority": ["onnx", "yolo"]},
        "performance": {"imgsz": 640, "precision": "fp16", "engine_priority": ["yolo", "onnx"]},
    },
}


def _caps(cuda=False, memory=None, providers=("CPUExecutionProvider",)):
    return HardwareCapabilities(cuda, "Test GPU" if cuda else None, memory, providers)


class TestHardwareProfile(unittest.TestCase):
    def test_auto_cpu_selects_edge(self):
        with patch.dict(os.environ, {}, clear=True):
            result = resolve_inference_profile(CONFIG, _caps())
        self.assertEqual(result.active_profile, "edge")
        self.assertEqual(result.device, "cpu")
        self.assertEqual(result.engine_priority[0], "onnx")

    def test_auto_large_gpu_selects_performance(self):
        with patch.dict(os.environ, {}, clear=True):
            result = resolve_inference_profile(
                CONFIG, _caps(True, 8.0, ("CUDAExecutionProvider", "CPUExecutionProvider"))
            )
        self.assertEqual(result.active_profile, "performance")
        self.assertEqual(result.requested_precision, "fp16")

    def test_auto_small_gpu_selects_balanced(self):
        with patch.dict(os.environ, {}, clear=True):
            result = resolve_inference_profile(CONFIG, _caps(True, 4.0))
        self.assertEqual(result.active_profile, "balanced")

    def test_tensor_rt_is_first_when_provider_is_really_available(self):
        with patch.dict(os.environ, {}, clear=True):
            result = resolve_inference_profile(
                CONFIG,
                _caps(
                    True,
                    8.0,
                    ("TensorrtExecutionProvider", "CUDAExecutionProvider"),
                ),
            )
        self.assertEqual(result.active_profile, "performance")
        self.assertEqual(result.engine_priority[0], "onnx")

    def test_explicit_gpu_profile_falls_back_truthfully(self):
        with patch.dict(
            os.environ, {"SECURECOATING_HARDWARE_PROFILE": "performance"}, clear=True
        ):
            result = resolve_inference_profile(CONFIG, _caps())
        self.assertEqual(result.active_profile, "edge")
        self.assertIn("CUDA is unavailable", result.fallback_reason)

    def test_legacy_cpu_override_wins(self):
        with patch.dict(
            os.environ, {"SECURECOATING_INFERENCE_DEVICE": "cpu"}, clear=True
        ):
            result = resolve_inference_profile(CONFIG, _caps(True, 8.0))
        self.assertEqual(result.active_profile, "edge")

    def test_invalid_profile_is_rejected(self):
        with patch.dict(
            os.environ, {"SECURECOATING_HARDWARE_PROFILE": "magic"}, clear=True
        ):
            with self.assertRaises(ValueError):
                resolve_inference_profile(CONFIG, _caps())


if __name__ == "__main__":
    unittest.main()
