"""
Unit Tests for RollCertificateGenerator (Tamper-Evident Cryptographic Digital Certificate)
"""

import unittest
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from traceability.roll_certificate import RollCertificateGenerator, DigitalRollCertificate


class TestRollCertificate(unittest.TestCase):

    def test_certificate_generation_and_cryptographic_verification(self):
        defects = [
            {"defect_id": "d1", "class_name": "scratch", "lane_id": 1, "peak_height_um": 5.0},
            {"defect_id": "d2", "class_name": "void", "lane_id": 2, "peak_height_um": 2.0}
        ]
        cert = RollCertificateGenerator.build_certificate(
            roll_id="TEST_ROLL_100",
            batch_id="BATCH_100",
            inspected_length_m=500.0,
            total_length_m=1200.0,
            defect_records=defects,
            spc_status="IN CONTROL"
        )
        self.assertEqual(cert.roll_id, "TEST_ROLL_100")
        self.assertEqual(cert.total_defects_count, 2)
        self.assertTrue(len(cert.payload_hash_sha256) == 64)
        self.assertTrue(len(cert.hmac_digital_signature) == 64)
        # Verify cryptographic authenticity
        self.assertTrue(cert.verify_signature())

    def test_markdown_and_dict_export(self):
        cert = RollCertificateGenerator.build_certificate(
            roll_id="TEST_ROLL_200",
            batch_id="BATCH_200",
            inspected_length_m=200.0,
            total_length_m=1000.0,
            defect_records=[],
            spc_status="IN CONTROL",
            total_inspections=2000,
            failed_inspections=0,
            metric_provenance="unit-test fixture",
        )
        md = cert.to_markdown()
        self.assertIn("Provisional Battery Electrode Quality Manifest", md)
        self.assertIn("HMAC", md)

        data = cert.to_dict()
        self.assertEqual(data["overall_quality_grade"], "GRADE_A_PRIME")
        self.assertFalse(data["quality_metrics_provisional"])

    def test_missing_inspection_counts_never_claims_a_quality_grade(self):
        cert = RollCertificateGenerator.build_certificate(
            roll_id="TEST_ROLL_UNKNOWN",
            batch_id="BATCH_UNKNOWN",
            inspected_length_m=200.0,
            total_length_m=1000.0,
            defect_records=[],
        )
        self.assertEqual(cert.overall_quality_grade, "UNVERIFIED")
        self.assertFalse(cert.standards_compliant)


if __name__ == "__main__":
    unittest.main()
