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


if __name__ == "__main__":
    unittest.main()
