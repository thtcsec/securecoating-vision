"""Academic and industrial metric formulas for the LIBAD adapter."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from libad.metrics import academic_metrics, auroc, industrial_gate_metrics


class TestLibadMetrics(unittest.TestCase):
    def test_perfect_ranking_has_unit_auroc(self):
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        self.assertAlmostEqual(auroc(labels, scores), 1.0)
        metrics = academic_metrics(labels, scores)
        self.assertAlmostEqual(metrics["auroc"], 1.0)
        self.assertGreater(metrics["aupr"], 0.9)
        self.assertGreater(metrics["f1_max"], 0.9)
        self.assertLess(metrics["fpr95"], 0.01)

    def test_industrial_formulas_match_the_operational_definitions(self):
        labels = [0, 0, 0, 1, 1, 1, 1]
        decisions = ["PASS", "HOLD", "REJECT", "PASS", "REJECT", "HOLD", "REJECT"]
        metrics = industrial_gate_metrics(labels, decisions)
        self.assertEqual(metrics["n"], 7)
        self.assertEqual(metrics["n_pass"], 2)
        self.assertEqual(metrics["n_reject"], 3)
        self.assertEqual(metrics["n_hold"], 2)
        self.assertAlmostEqual(metrics["automatic_decision_coverage"], 5 / 7)
        self.assertAlmostEqual(metrics["hold_rate"], 2 / 7)
        self.assertAlmostEqual(metrics["escape_rate"], 1 / 4)
        self.assertAlmostEqual(metrics["selective_risk"], 2 / 5)

    def test_hold_is_excluded_from_selective_risk(self):
        labels = [0, 1]
        decisions = ["HOLD", "HOLD"]
        metrics = industrial_gate_metrics(labels, decisions)
        self.assertEqual(metrics["automatic_decision_coverage"], 0.0)
        self.assertEqual(metrics["hold_rate"], 1.0)
        self.assertEqual(metrics["escape_rate"], 0.0)
        self.assertIsNone(metrics["selective_risk"])


if __name__ == "__main__":
    unittest.main()
