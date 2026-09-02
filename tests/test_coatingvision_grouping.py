"""CoatingVision public metadata cannot support a factory roll-disjoint split."""

import json
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.coatingvision_grouping import inspect_coatingvision_grouping


class TestCoatingVisionGrouping(unittest.TestCase):
    def test_public_metadata_has_no_factory_roll_keys(self):
        inventory = inspect_coatingvision_grouping(ROOT)
        self.assertEqual(inventory["evidence_class"], "public_real_optical_image_split")
        self.assertFalse(inventory["factory_roll_disjoint"])
        self.assertFalse(inventory["can_build_roll_disjoint_split"])
        self.assertEqual(inventory["factory_group_keys_found"], [])
        self.assertIn("filename", inventory["demo_sample_keys"])
        self.assertNotIn("roll_id", inventory["demo_sample_keys"])
        self.assertFalse(inventory["metrics_factory_roll_disjoint"])
        self.assertIn("not factory roll-disjoint", inventory["metrics_split"])
        if inventory["classification_csv_present"]:
            self.assertEqual(
                inventory["classification_columns"],
                [
                    "original_file_name",
                    "file_name",
                    "Surface_Crack",
                    "Delamination",
                    "Pinhole",
                    "unclassified",
                ],
            )
        if inventory["detection_manifest_present"]:
            self.assertNotIn("roll_id", inventory["detection_manifest_keys"])

    def test_published_metrics_refuse_factory_roll_disjoint(self):
        metrics = json.loads(
            (ROOT / "reports/coatingvision_real_test_metrics.json").read_text(encoding="utf-8")
        )
        self.assertFalse(metrics["factory_roll_disjoint"])
        self.assertEqual(metrics["evidence_class"], "public_real_optical_image_split")

    def test_calibration_contract_stays_unverified(self):
        calibration = yaml.safe_load(
            (ROOT / "configs/calibration.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(calibration["calibration_id"], "SIMULATION_ONLY_UNCALIBRATED")
        self.assertFalse(calibration["verified"])


if __name__ == "__main__":
    unittest.main()
