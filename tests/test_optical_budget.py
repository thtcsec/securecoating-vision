"""
Unit Tests for OpticalThroughputBudgetEngine
"""

import unittest
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from industrial.optical_budget import OpticalThroughputBudgetEngine, OpticalSpecifications


class TestOpticalBudget(unittest.TestCase):

    def setUp(self):
        self.engine = OpticalThroughputBudgetEngine()

    def test_optical_budget_nominal(self):
        budget = self.engine.compute_budget(line_speed_m_s=1.8)
        self.assertEqual(budget["line_speed_m_s"], 1.8)
        self.assertGreater(budget["required_line_rate_khz"], 0.0)
        self.assertLessEqual(budget["motion_blur_pixels"], 0.50)
        self.assertTrue(budget["is_theoretically_feasible"])
        self.assertGreater(budget["total_optical_coverage_mm"], 650.0)
        self.assertLess(budget["d95_detectability_threshold_um"], 150.0)

    def test_optical_budget_high_speed_limit(self):
        budget = self.engine.compute_budget(line_speed_m_s=2.5)
        self.assertEqual(budget["line_speed_m_s"], 2.5)
        self.assertLess(budget["recommended_exposure_time_us"], 20.0)
        self.assertLessEqual(budget["motion_blur_pixels"], 0.50)


if __name__ == "__main__":
    unittest.main()
