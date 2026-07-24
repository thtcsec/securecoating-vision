import os
import sys
import numpy as np
import unittest

# Add src to python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from inference.sensor_fusion import SensorFusionManager, SensorStatus

class TestSensorFusion(unittest.TestCase):
    def test_sensor_status_degradation(self):
        status = SensorStatus()
        self.assertEqual(status.degradation_level, "NONE")
        
        status.thermal_online = False
        status.update_degradation()
        self.assertEqual(status.degradation_level, "PARTIAL")
        
        status.profiler_online = False
        status.update_degradation()
        self.assertEqual(status.degradation_level, "CRITICAL")
        
        status.rgb_online = False
        status.update_degradation()
        self.assertEqual(status.degradation_level, "OFFLINE")

    def test_sensor_fusion_alignment(self):
        manager = SensorFusionManager(target_size=(100, 100))
        H_mock = np.eye(3, dtype=np.float32)
        H_mock[0, 2] = 5.0  # Translation X
        
        manager.update_calibration(h_thermal=H_mock, h_profiler=H_mock)
        self.assertTrue(np.allclose(manager.H_thermal_to_rgb, H_mock))
        self.assertTrue(np.allclose(manager.H_profiler_to_rgb, H_mock))

    def test_sensor_fusion_fuse(self):
        manager = SensorFusionManager(target_size=(100, 100), enable_mock=True)
        # Generate mock RGB image (100x100x3)
        rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # 1. Optimal mode (all online)
        res = manager.fuse(rgb, thermal_online=True, profiler_online=True)
        self.assertEqual(res.fused_tensor.shape, (100, 100, 5))
        self.assertEqual(res.sensor_status.degradation_level, "NONE")
        self.assertIsNotNone(res.thermal_frame)
        self.assertIsNotNone(res.height_frame)
        
        # 2. Degraded mode (thermal offline)
        res_degraded = manager.fuse(rgb, thermal_online=False, profiler_online=True)
        self.assertEqual(res_degraded.fused_tensor.shape, (100, 100, 5))
        self.assertEqual(res_degraded.sensor_status.degradation_level, "PARTIAL")
        self.assertIsNone(res_degraded.thermal_frame)
        self.assertIsNotNone(res_degraded.height_frame)
        # Ensure thermal channel is zero-filled in normalized fused tensor
        self.assertTrue(np.all(res_degraded.fused_tensor[:, :, 3] == 0.0))

    def test_fuse_matches_arbitrary_rgb_size(self):
        manager = SensorFusionManager(target_size=(1024, 1024), enable_mock=True)
        rgb = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        res = manager.fuse(rgb, thermal_online=True, profiler_online=True)
        self.assertEqual(res.fused_tensor.shape, (640, 480, 5))
        self.assertEqual(res.thermal_frame.shape, (640, 480))
        self.assertEqual(res.height_frame.shape, (640, 480))


if __name__ == "__main__":
    unittest.main()
