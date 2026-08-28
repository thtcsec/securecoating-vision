"""Evidence-gate contracts for the LIBAD industrial decision layer."""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from libad.evidence_gate import EvidenceContracts, classify_score, decide_evidence_gate


class TestLibadEvidenceGate(unittest.TestCase):
    def test_normal_agreement_passes(self):
        decision = decide_evidence_gate(0.7, 0.8)
        self.assertEqual(decision.action, "PASS")
        self.assertIn("Normal agreement", decision.reason)

    def test_surface_defect_rejects_even_if_xray_is_quiet(self):
        decision = decide_evidence_gate(1.4, 0.6, strong_margin=0.18)
        self.assertEqual(decision.action, "REJECT")
        self.assertIn("surface evidence", decision.reason)

    def test_internal_xray_anomaly_rejects_when_vis_is_normal(self):
        decision = decide_evidence_gate(0.6, 1.5, strong_margin=0.18)
        self.assertEqual(decision.action, "REJECT")
        self.assertIn("complementary X-ray", decision.reason)

    def test_near_threshold_disagreement_holds(self):
        decision = decide_evidence_gate(1.05, 0.7, threshold=1.0, uncertainty_band=0.12)
        self.assertEqual(decision.action, "HOLD")
        self.assertIn("uncertainty", decision.reason.lower())

    def test_missing_modality_holds(self):
        decision = decide_evidence_gate(
            0.4,
            None,
            contracts=EvidenceContracts(xray_available=False, require_both_modalities=True),
        )
        self.assertEqual(decision.action, "HOLD")
        self.assertIn("X-rayL", decision.reason)

    def test_stale_sensor_holds(self):
        decision = decide_evidence_gate(
            0.3,
            0.3,
            contracts=EvidenceContracts(vis_stale=True),
        )
        self.assertEqual(decision.action, "HOLD")
        self.assertIn("stale", decision.reason)

    def test_unverified_calibration_holds(self):
        decision = decide_evidence_gate(
            0.2,
            0.2,
            contracts=EvidenceContracts(calibration_verified=False),
        )
        self.assertEqual(decision.action, "HOLD")
        self.assertIn("Calibration", decision.reason)

    def test_traceability_failure_holds(self):
        decision = decide_evidence_gate(
            0.2,
            0.2,
            contracts=EvidenceContracts(traceability_ok=False),
        )
        self.assertEqual(decision.action, "HOLD")

    def test_plc_ack_failure_holds(self):
        decision = decide_evidence_gate(
            0.2,
            0.2,
            contracts=EvidenceContracts(plc_ack_ok=False),
        )
        self.assertEqual(decision.action, "HOLD")
        self.assertIn("PLC ACK", decision.reason)

    def test_classify_is_one_sided_around_the_normal_reference(self):
        self.assertEqual(classify_score(0.99, threshold=1.0, uncertainty_band=0.12), "NORMAL")
        self.assertEqual(classify_score(1.05, threshold=1.0, uncertainty_band=0.12), "UNCERTAIN")
        self.assertEqual(classify_score(1.20, threshold=1.0, uncertainty_band=0.12), "ANOMALY")


if __name__ == "__main__":
    unittest.main()
