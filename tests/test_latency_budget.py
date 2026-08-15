"""
Unit Tests for LatencyBudgetEngine (Statistical Percentiles & Guard-Band Distance)
"""

import unittest
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from industrial.latency_budget import LatencyBudgetEngine, LatencyStageDistribution


class TestLatencyBudget(unittest.TestCase):

    def setUp(self):
        self.engine = LatencyBudgetEngine(num_benchmark_samples=2000)

    def test_latency_budget_percentiles_and_guardband(self):
        budget = self.engine.compute_budget(line_speed_m_s=1.8, latency_guard_band_ms=3.0)
        self.assertIn("p50_total_ms", budget)
        self.assertIn("p99_9_total_ms", budget)
        self.assertGreater(budget["p99_9_total_ms"], budget["p50_total_ms"])
        
        # Check that 500mm installed distance provides safety margin > 1.0x
        self.assertTrue(budget["is_theoretical_budget_pass"])
        self.assertTrue(budget["is_monte_carlo_modeled"])
        self.assertGreater(budget["safety_margin_ratio"], 1.0)
        self.assertLess(budget["min_required_ejector_distance_mm"], 500.0)


if __name__ == "__main__":
    unittest.main()
