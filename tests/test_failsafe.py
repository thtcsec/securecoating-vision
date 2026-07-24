import os
import sys
import numpy as np
import unittest

# Add src to python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from inference.failsafe import FailSafeManager, SystemState

class TestFailSafe(unittest.TestCase):
    def test_failsafe_initialization(self):
        manager = FailSafeManager(max_inference_timeout_ms=1000.0)
        self.assertEqual(manager.system_state, SystemState.OPTIMAL)
        self.assertTrue(manager.health.rgb_sensor_ok)
        self.assertTrue(manager.health.thermal_sensor_ok)
        self.assertTrue(manager.health.profiler_sensor_ok)

    def test_failsafe_degradation(self):
        manager = FailSafeManager(max_inference_timeout_ms=1000.0)
        # Use simulation helper to disconnect thermal sensor
        manager.simulate_sensor_disconnect("thermal")
        report = manager.get_health_report()
        self.assertEqual(report["system_state"], "DEGRADED")
        self.assertEqual(report["sensors"]["thermal_camera"], "OFFLINE")
        
        # Disconnect RGB camera
        manager.simulate_sensor_disconnect("rgb")
        report2 = manager.get_health_report()
        self.assertEqual(report2["system_state"], "OFFLINE")

    def test_dead_frame_detection(self):
        manager = FailSafeManager(max_inference_timeout_ms=1000.0, enable_frame_validation=True)
        # Generate dead frame (zero variation)
        dead_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Detect anomaly
        is_valid = manager.validate_frame(dead_frame)
        self.assertFalse(is_valid["valid"])
        self.assertIn("dead frame", is_valid["reason"].lower())
        
        # Generate normal frame (high variation)
        normal_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        is_valid_normal = manager.validate_frame(normal_frame)
        self.assertTrue(is_valid_normal["valid"])

if __name__ == "__main__":
    unittest.main()
