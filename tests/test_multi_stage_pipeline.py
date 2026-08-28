"""
Unit Tests for MultiStageIndustrialPipeline (7-Stage Workflow & Latency Audit)
"""

import unittest
import yaml
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from inference.predictor import CoatingPredictor
from inference.sensor_fusion import SensorFusionManager
from inference.failsafe import FailSafeManager
from inference.multi_stage_pipeline import MultiStageIndustrialPipeline
from industrial.protocol_manager import IndustrialProtocolManager
from industrial.web_synchronizer import WebSynchronizer


class TestMultiStagePipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open("configs/model.yaml", "r") as f:
            model_config = yaml.safe_load(f)
        cls.predictor = CoatingPredictor(model_config)
        cls.fusion = SensorFusionManager(target_size=(1024, 1024), enable_mock=True)
        cls.failsafe = FailSafeManager()
        cls.industrial = IndustrialProtocolManager({"enabled": True, "mock_mode": True})
        cls.web_sync = WebSynchronizer()
        cls.pipeline = MultiStageIndustrialPipeline(
            predictor=cls.predictor,
            fusion_manager=cls.fusion,
            failsafe_manager=cls.failsafe,
            industrial_manager=cls.industrial,
            web_synchronizer=cls.web_sync,
            pixel_to_mm_ratio=0.1
        )

    def test_full_7_stage_pipeline_execution(self):
        result = self.pipeline.execute_inspection(
            sample_id="UNIT_TEST_SAMPLE_001",
            cross_web_pos_mm=325.0
        )
        self.assertEqual(result.sample_id, "UNIT_TEST_SAMPLE_001")
        self.assertIn("stage_1_web_sync_ms", result.stage_latencies_ms)
        self.assertIn("stage_2_acquisition_ms", result.stage_latencies_ms)
        self.assertIn("stage_3_fusion_ms", result.stage_latencies_ms)
        self.assertIn("stage_4_ai_inference_ms", result.stage_latencies_ms)
        self.assertIn("stage_5_metrology_ms", result.stage_latencies_ms)
        self.assertIn("stage_6_decision_plc_ms", result.stage_latencies_ms)
        self.assertIn("stage_7_root_cause_ms", result.stage_latencies_ms)
        self.assertGreater(result.total_pipeline_latency_ms, 0.0)

        # Check frames generated
        self.assertEqual(result.optical_brightfield.shape, (1024, 1024, 3))
        self.assertEqual(result.optical_darkfield.shape, (1024, 1024, 3))
        self.assertEqual(result.thermal_diffusivity_phase.shape, (1024, 1024))
        self.assertEqual(result.height_topography_map.shape, (1024, 1024))

    def test_rgb_only_hold_does_not_claim_zero_anomaly(self):
        optical = np.zeros((64, 64, 3), dtype=np.uint8)
        optical[::2] = 120
        detections = [{
            "box": [1, 1, 10, 10],
            "confidence": 0.8,
            "class_id": 0,
            "class_name": "scratch",
        }]

        class DetectionPredictor:
            def predict(self, optical_image, thermal=None, height=None):
                return {
                    "segmentation_mask": np.zeros(optical_image.shape[:2], dtype=np.uint8),
                    "detections": detections,
                    "untrained_fallback": False,
                    "latency_ms": 1.0,
                    "engine": "TEST",
                    "model_version": "test",
                }

        pipeline = MultiStageIndustrialPipeline(
            predictor=DetectionPredictor(),
            fusion_manager=self.fusion,
            failsafe_manager=FailSafeManager(),
            industrial_manager=IndustrialProtocolManager({"enabled": True, "mock_mode": True}),
            web_synchronizer=WebSynchronizer(),
            pixel_to_mm_ratio=0.1,
            calibration_verified=True,
        )
        result = pipeline.execute_inspection(
            sample_id="RGB_ONLY_HOLD",
            optical_rgb=optical,
            thermal_raw=None,
            height_map=None,
            enable_plc_signal=False,
        )
        summary = result.to_summary_dict()
        self.assertEqual(summary["overall_verdict"], "HOLD")
        self.assertEqual(summary["plc_gate_action"], "HOLD")
        self.assertFalse(summary["standards_compliant"])
        self.assertEqual(summary["raw_detections_count"], 1)
        self.assertEqual(summary["defects_count"], 0)
        self.assertNotEqual(summary["root_cause_report"]["severity_level"], "NOMINAL")
        self.assertIn("HOLD", summary["root_cause_report"]["primary_root_cause"])
        self.assertIn("Raw detections present: 1", summary["root_cause_report"]["defect_signature"])



if __name__ == "__main__":
    unittest.main()
