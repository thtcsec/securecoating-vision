"""
Unit Tests for RootCauseDiagnosticEngine (AI Closed-Loop Equipment Feedback)
"""

import unittest
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from traceability.root_cause_engine import RootCauseDiagnosticEngine, RootCauseReport


class TestRootCauseEngine(unittest.TestCase):

    def setUp(self):
        self.engine = RootCauseDiagnosticEngine()

    def test_nominal_process_diagnostics(self):
        report = self.engine.diagnose_batch(defects_metrology=[])
        self.assertEqual(report.severity_level, "NOMINAL")
        self.assertEqual(len(report.action_items), 0)

    def test_scratch_equipment_attribution(self):
        defects = [
            {"class_name": "scratch", "length_mm": 6.0, "area_mm2": 2.0},
            {"class_name": "scratch", "length_mm": 4.5, "area_mm2": 1.2}
        ]
        report = self.engine.diagnose_batch(defects)
        self.assertIn("Slot-Die", report.affected_equipment)
        self.assertGreater(len(report.action_items), 0)
        self.assertEqual(report.action_items[0].equipment_unit, "Slot-Die Lip")

    def test_delamination_drying_oven_attribution(self):
        defects = [
            {"class_name": "delamination", "area_mm2": 5.0, "peak_height_um": 2.0}
        ]
        report = self.engine.diagnose_batch(defects)
        self.assertEqual(report.severity_level, "CRITICAL")
        self.assertIn("Drying Oven", report.affected_equipment)
        self.assertTrue(any("Zone 1" in a.equipment_unit for a in report.action_items))

    def test_void_slurry_mixer_attribution(self):
        defects = [
            {"class_name": "void", "area_mm2": 2.5, "valley_depth_um": 20.0}
        ]
        report = self.engine.diagnose_batch(defects)
        self.assertIn("Mixer", report.affected_equipment)
        self.assertTrue(any("Vacuum" in a.parameter_name for a in report.action_items))


if __name__ == "__main__":
    unittest.main()
