"""Official 10-seed protocol: no random split, four required experiments."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from libad.evaluate import evaluate_official_splits
from libad.protocol import OFFICIAL_SPLIT_SEEDS


class TestLibadEvaluate(unittest.TestCase):
    def test_refuses_a_non_official_seed(self):
        from libad.dataset import load_split

        with self.assertRaises(ValueError):
            load_split(123)

    def test_ten_official_seeds_and_four_experiments(self):
        report = evaluate_official_splits(
            seeds=OFFICIAL_SPLIT_SEEDS[:2],
            experiments=("unimodal_vis", "unimodal_xray_l", "multimodal", "securecoating_gate"),
            allow_fixture=True,
        )
        self.assertEqual(report["official_split_seeds"], list(OFFICIAL_SPLIT_SEEDS[:2]))
        self.assertFalse(report["comparable_to_paper"])
        self.assertEqual(report["evidence_class"], "protocol_fixture")
        for name in ("unimodal_vis", "unimodal_xray_l", "multimodal", "securecoating_gate"):
            self.assertIn(name, report["experiments"])
            self.assertIn("academic", report["experiments"][name])
        industrial = report["experiments"]["securecoating_gate"]["industrial"]
        for metric in (
            "automatic_decision_coverage",
            "hold_rate",
            "escape_rate",
            "selective_risk",
        ):
            self.assertIn(metric, industrial)
        self.assertIn("does not claim DA-Core", report["local_contribution"])
        self.assertTrue(report["predictions"])
