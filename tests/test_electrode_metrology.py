"""
Unit Tests for 5-Layer ElectrodeMetrologyEngine
"""

import unittest
import numpy as np
import cv2
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from inference.electrode_metrology import ElectrodeMetrologyEngine, DefectMetrologyResult
from industrial.optical_budget import OpticalCalibrationModel


class TestElectrodeMetrology(unittest.TestCase):

    def setUp(self):
        calib = OpticalCalibrationModel(x_um_per_px=50.0, y_um_per_px=50.0)
        self.engine = ElectrodeMetrologyEngine(calibration=calib)

    def test_rotated_geometry_rectangle(self):
        # Create a rotated rectangle contour
        canvas = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(canvas, (20, 20), (60, 40), 255, -1)
        contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        l, w, a, p, aspect, web_angle, eq_d = self.engine.compute_rotated_geometry(contours[0])
        self.assertGreater(a, 0.0)
        self.assertGreater(l, 0.0)
        self.assertGreater(w, 0.0)
        self.assertGreater(p, 0.0)
        self.assertGreaterEqual(web_angle, 0.0)

    def test_delamination_zero_tolerance(self):
        risk, post_cal, load_dev, hazard, guard_pass, tier, clauses = self.engine.evaluate_risk_and_guardband(
            class_name="delamination",
            length_mm=2.0,
            width_mm=1.0,
            area_mm2=2.0,
            peak_h_um=5.0,
            valley_d_um=0.0,
            u_length_mm=0.05,
            u_area_mm2=0.02
        )
        self.assertFalse(guard_pass)
        self.assertEqual(tier, "GRADE_C_REJECT")
        self.assertTrue(any("delamination" in c.lower() for c in clauses))

    def test_protrusion_risk_scoring(self):
        risk, post_cal, load_dev, hazard, guard_pass, tier, clauses = self.engine.evaluate_risk_and_guardband(
            class_name="blister",
            length_mm=1.0,
            width_mm=1.0,
            area_mm2=1.0,
            peak_h_um=20.0,
            valley_d_um=0.0,
            u_length_mm=0.05,
            u_area_mm2=0.02
        )
        self.assertFalse(guard_pass)
        self.assertEqual(tier, "GRADE_C_REJECT")
        self.assertGreater(risk, 0.6)

    def test_full_segmentation_with_baseline_leveling(self):
        seg_mask = np.zeros((200, 200), dtype=np.int32)
        cv2.rectangle(seg_mask, (50, 50), (80, 80), 2, -1)  # class 2 = void
        
        height_map = np.zeros((200, 200), dtype=np.float32)
        height_map[50:80, 50:80] = -15.0  # -15um depression
        
        results = self.engine.analyze_segmentation(seg_mask, height_map)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].class_name, "void")
        self.assertAlmostEqual(results[0].valley_depression_um, 15.0, places=1)
        self.assertGreater(results[0].area_expanded_uncertainty_mm2, 0.0)


if __name__ == "__main__":
    unittest.main()
