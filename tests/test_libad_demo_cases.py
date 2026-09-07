"""90-second LIBAD demo cases must land on PASS / REJECT / REJECT / HOLD."""

import os
import sys
import unittest
from dataclasses import fields

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from libad.demo_cases import (
    DEMO_CASES,
    _ensure_case_four_hold,
    run_all_demo_cases,
    run_libad_demo_case,
)
from libad.certificate import EvidenceCertificate
from libad.certificate import build_evidence_certificate
from libad.evidence_gate import EvidenceContracts, decide_evidence_gate
from libad.protocol import load_project_identity
from libad.scorer import SampleScore


class TestLibadDemoCases(unittest.TestCase):
    def test_four_cases_match_the_stage_script(self):
        secret = b"unit-test-libad-demo"
        results = run_all_demo_cases(secret=secret)
        self.assertEqual([item["decision"]["action"] for item in results], ["PASS", "REJECT", "REJECT", "HOLD"])
        self.assertIn("surface evidence", results[1]["decision"]["reason"])
        self.assertIn("complementary X-ray", results[2]["decision"]["reason"])
        self.assertEqual(results[3]["decision"]["action"], "HOLD")
        self.assertEqual(results[3]["decision"]["vis_state"], "UNCERTAIN")
        self.assertEqual(results[3]["decision"]["xray_state"], "NORMAL")
        self.assertEqual(results[3]["decision"]["contract_holds"], [])
        self.assertEqual(results[3]["calibration_state"], "VERIFIED")
        self.assertIn("uncertainty", results[3]["decision"]["reason"].lower())
        for item in results:
            cert = item["certificate"]
            self.assertEqual(cert["roll_id"], item["identity"]["roll_id"])
            self.assertEqual(cert["batch_id"], item["identity"]["batch_id"])
            self.assertEqual(cert["part_id"], item["identity"]["part_id"])
            self.assertTrue(cert["hmac_digital_signature"])
            self.assertEqual(item["expected_action"], DEMO_CASES[item["case_id"]]["expected_action"])
            self.assertEqual(item["evidence_class"], "protocol_fixture")
            self.assertFalse(item["comparable_to_paper"])
            self.assertEqual(
                item["release_provenance_ref"], "reports/submission_manifest.json"
            )
            self.assertEqual(
                item["fixture_generation_provenance"]["scope"],
                "protocol_fixture_generation_only",
            )
            self.assertEqual(
                cert["provenance_scope"], "protocol_fixture_generation"
            )
            self.assertEqual(
                cert["source_tree_dirty"],
                item["fixture_generation_provenance"]["working_tree_dirty_at_generation"],
            )
            self.assertEqual(
                cert["source_diff_sha256"],
                item["fixture_generation_provenance"]["source_diff_sha256"],
            )
            self.assertEqual(
                cert["source_tree_dirty"], item["source_provenance"]["working_tree_dirty"]
            )
            self.assertEqual(
                cert["source_diff_sha256"], item["source_provenance"]["source_diff_sha256"]
            )
            self.assertEqual(
                item["source_provenance"]["scope"], "protocol_fixture_generation_only"
            )

    def test_case_four_holds_when_the_descriptor_skips_the_uncertainty_band(self):
        score = SampleScore(
            sample_id="DEMO_CASE_4",
            vis_score=0.87,
            xray_score=0.95,
            fused_score=0.95,
            vis_available=True,
            xray_available=True,
        )
        contracts = EvidenceContracts()
        live = decide_evidence_gate(score.vis_score, score.xray_score)
        self.assertEqual(live.action, "PASS")
        injected, decision = _ensure_case_four_hold(score, contracts, live)
        self.assertEqual(decision.action, "HOLD")
        self.assertEqual(decision.vis_state, "UNCERTAIN")
        self.assertEqual(decision.xray_state, "NORMAL")
        self.assertEqual(decision.contract_holds, [])
        self.assertIn("uncertainty", decision.reason.lower())
        self.assertEqual(
            decision.details["protocol_fixture_score_source"],
            "uncertainty_band_injection",
        )
        self.assertGreater(injected.vis_score, 1.0)
        self.assertLess(injected.vis_score, 1.12)

    def test_public_title_is_the_registered_finalist_title_plus_tagline(self):
        identity = load_project_identity()
        result = run_libad_demo_case(1, secret=b"unit-test-libad-demo")
        self.assertEqual(result["title"], identity["registered_title"])
        self.assertEqual(result["tagline"], "Evidence-Gated Multimodal Inspection for Battery Electrode Manufacturing")
        self.assertIn("Evidence-Gated Multimodal AI", result["title"])
        self.assertNotIn("High-Throughput and Zero-Trust", result["title"])
        self.assertEqual(result["brand"], "SecureCoating-Vision")

    def test_certificate_signature_covers_source_diff_provenance(self):
        secret = b"unit-test-libad-demo"
        result = run_libad_demo_case(4, secret=secret)
        payload = result["certificate"]
        constructor = {
            field.name: payload[field.name]
            for field in fields(EvidenceCertificate)
            if field.name in payload
        }
        certificate = EvidenceCertificate(**constructor)
        self.assertTrue(certificate.verify(secret))
        certificate.source_diff_sha256 = "0" * 64
        self.assertFalse(certificate.verify(secret))

    def test_default_development_certificate_key_is_stable_within_process(self):
        certificate = build_evidence_certificate(
            roll_id="ROLL", batch_id="BATCH", part_id="PART",
            decision="HOLD", decision_reason="test", vis_score=None,
            xray_score=None, fused_score=None, vis_state="MISSING",
            xray_state="MISSING", calibration_state="UNVERIFIED",
            model_hash=None, dataset_manifest_hash=None, commit_hash="UNKNOWN",
            plc_state="SIMULATED", detector_attribution="test",
        )
        self.assertTrue(certificate.verify())


if __name__ == "__main__":
    unittest.main()
