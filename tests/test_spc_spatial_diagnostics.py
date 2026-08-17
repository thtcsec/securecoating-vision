"""
Unit Tests for Tier-1 Gigafactory SPC, Spatial FFT Diagnostics & Slitting Optimizer
"""

import unittest
import numpy as np
import math
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from industrial.spc_spatial_diagnostics import (
    SpatialDiagnosticsEngine, GigafactorySPCEngine, DigitalBatteryPassportGenerator,
    EquipmentRegistry
)


class TestSPCSpatialDiagnostics(unittest.TestCase):

    def setUp(self):
        self.spatial_engine = SpatialDiagnosticsEngine()
        self.spc_engine = GigafactorySPCEngine(target_defect_density_limit_per_100m=2.0, num_lanes=4)

    def test_spatial_fft_periodic_roller_detection(self):
        # Generate defects repeating at Guide Roller R-01 circumference: ~377.0 mm (D = 120mm)
        wavelength_m = 0.377
        total_length_m = 300.0
        defect_positions = []
        
        # Periodic defect injection + some noise
        for pos in np.arange(1.0, total_length_m, wavelength_m):
            defect_positions.append(float(pos + np.random.uniform(-0.01, 0.01)))

        anomalies = self.spatial_engine.analyze_spatial_periodicity(
            defect_md_positions_m=defect_positions,
            total_scanned_length_m=total_length_m,
            spatial_bin_size_mm=10.0
        )

        self.assertGreater(len(anomalies), 0)
        matched = anomalies[0]
        self.assertAlmostEqual(matched.dominant_wavelength_mm, 377.0, delta=25.0)
        self.assertIn("Guide Roller", matched.matched_equipment)

    def test_lane_spc_cpk_grading(self):
        # Lane 1: 1 defect / 1000m -> Cpk high (EV Grade)
        # Lane 4: 25 defects / 1000m -> Cpk low (Reject)
        defects_by_lane = {1: 1, 2: 3, 3: 5, 4: 25}
        spc_res = self.spc_engine.compute_lane_spc(defects_by_lane, inspected_length_m=1000.0)

        self.assertEqual(len(spc_res), 4)
        self.assertEqual(spc_res[1].six_sigma_tier, "EV_GRADE_TIER_1")
        self.assertGreater(spc_res[1].cpk, 1.67)
        self.assertEqual(spc_res[4].six_sigma_tier, "REJECT_QUARANTINE")

    def test_slitting_yield_optimization(self):
        defects = [
            {"defect_id": "D1", "lane_id": 1, "severity": "WARNING", "linear_pos_m": 50.0},
            {"defect_id": "D2", "lane_id": 2, "severity": "WARNING", "linear_pos_m": 120.0},
            {"defect_id": "D3", "lane_id": 3, "severity": "CRITICAL", "linear_pos_m": 450.0},
            {"defect_id": "D4", "lane_id": 4, "severity": "CRITICAL", "linear_pos_m": 10.0},
            {"defect_id": "D5", "lane_id": 4, "severity": "CRITICAL", "linear_pos_m": 20.0},
            {"defect_id": "D6", "lane_id": 4, "severity": "CRITICAL", "linear_pos_m": 30.0},
        ]

        yield_plan = self.spc_engine.optimize_slitting_yield(
            roll_length_m=1200.0,
            web_width_mm=650.0,
            defects_list=defects
        )

        self.assertGreater(yield_plan.overall_recovery_yield_pct, 50.0)
        self.assertEqual(yield_plan.lane_grades[1], "EV_GRADE_TIER_1")
        self.assertEqual(yield_plan.lane_grades[2], "EV_GRADE_TIER_1")
        self.assertIn(450.0, yield_plan.recommended_splices_md_m)

    def test_digital_battery_passport_generation(self):
        spc_res = self.spc_engine.compute_lane_spc({1: 1, 2: 2, 3: 2, 4: 1}, inspected_length_m=1200.0)
        yield_plan = self.spc_engine.optimize_slitting_yield(1200.0, 650.0, [])
        passport = DigitalBatteryPassportGenerator.generate_passport(
            roll_id="ROLL_CATL_999",
            batch_id="BATCH_LFP_01",
            spc_results=spc_res,
            yield_plan=yield_plan,
            anomalies=[]
        )
        self.assertEqual(passport["roll_id"], "ROLL_CATL_999")
        self.assertIn("traceability_genealogy", passport)
        self.assertIn("six_sigma_quality_summary", passport)


if __name__ == "__main__":
    unittest.main()
