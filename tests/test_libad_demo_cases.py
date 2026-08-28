"""90-second LIBAD demo cases must land on PASS / REJECT / REJECT / HOLD."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from libad.demo_cases import DEMO_CASES, run_all_demo_cases, run_libad_demo_case
from libad.protocol import load_project_identity


class TestLibadDemoCases(unittest.TestCase):
    def test_four_cases_match_the_stage_script(self):
        secret = b"unit-test-libad-demo"
        results = run_all_demo_cases(secret=secret)
        self.assertEqual([item["decision"]["action"] for item in results], ["PASS", "REJECT", "REJECT", "HOLD"])
        self.assertIn("surface evidence", results[1]["decision"]["reason"])
        self.assertIn("complementary X-ray", results[2]["decision"]["reason"])
        self.assertEqual(results[3]["decision"]["action"], "HOLD")
        for item in results:
            cert = item["certificate"]
            self.assertEqual(cert["roll_id"], item["identity"]["roll_id"])
            self.assertEqual(cert["batch_id"], item["identity"]["batch_id"])
            self.assertEqual(cert["part_id"], item["identity"]["part_id"])
            self.assertTrue(cert["hmac_digital_signature"])
            self.assertEqual(item["expected_action"], DEMO_CASES[item["case_id"]]["expected_action"])

    def test_public_title_is_the_registered_finalist_title_plus_tagline(self):
        identity = load_project_identity()
        result = run_libad_demo_case(1, secret=b"unit-test-libad-demo")
        self.assertEqual(result["title"], identity["registered_title"])
        self.assertEqual(result["tagline"], "Evidence-Gated Multimodal Inspection for Battery Electrode Manufacturing")
        self.assertIn("High-Throughput and Zero-Trust Edge-Cloud Pipeline", result["title"])
        self.assertEqual(result["brand"], "SecureCoating-Vision")


if __name__ == "__main__":
    unittest.main()
