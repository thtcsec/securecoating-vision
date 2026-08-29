"""Fault-injection tests for fail-closed traceability behavior."""

import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traceability.quality_memory import QualityMemory


class TestQualityMemoryFaults(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.memory = QualityMemory(os.path.join(self.tempdir.name, "quality.db"))

    def tearDown(self):
        self.tempdir.cleanup()

    @staticmethod
    def fail_connection():
        raise sqlite3.OperationalError("injected disk failure")

    def test_write_failure_is_reported(self):
        with patch.object(self.memory, "_get_connection", self.fail_connection):
            written = self.memory.add_entry(
                batch_id="B", part_id="P", has_defect=False, defect_class="none"
            )
        self.assertFalse(written)
        self.assertFalse(self.memory.healthy)

    def test_read_failures_never_return_healthy_defaults(self):
        with patch.object(self.memory, "_get_connection", self.fail_connection):
            stats = self.memory.get_batch_stats("B")
            spc = self.memory.check_spc_alarms("B")
            latency = self.memory.get_latency_stats("B")
            recent = self.memory.get_recent_inspections("B")
            audits = self.memory.get_recent_control_audits()
        self.assertEqual(stats["status"], "ERROR")
        self.assertIsNone(stats["pass_rate"])
        self.assertEqual(spc["status"], "UNKNOWN")
        self.assertEqual(latency["status"], "ERROR")
        self.assertIsNone(latency["p50_ms"])
        self.assertEqual(recent["status"], "ERROR")
        self.assertEqual(recent["records"], [])
        self.assertEqual(audits["status"], "ERROR")
        self.assertEqual(audits["records"], [])

    def test_hold_is_excluded_from_pass_rate(self):
        self.assertTrue(self.memory.add_entry(
            batch_id="B", part_id="PASS", has_defect=False,
            defect_class="none", gate_action="PASS", inspection_valid=True,
        ))
        self.assertTrue(self.memory.add_entry(
            batch_id="B", part_id="HOLD", has_defect=False,
            defect_class="none", gate_action="HOLD", inspection_valid=False,
        ))
        stats = self.memory.get_batch_stats("B")
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["passed"], 1)
        self.assertEqual(stats["held"], 1)
        self.assertEqual(stats["pass_rate"], 100.0)

    def test_duplicate_part_identity_is_rejected_transactionally(self):
        kwargs = {
            "batch_id": "B",
            "part_id": "DUPLICATE",
            "has_defect": False,
            "defect_class": "none",
        }
        self.assertTrue(self.memory.add_entry(**kwargs))
        self.assertFalse(self.memory.add_entry(**kwargs))
        self.assertTrue(self.memory.healthy)
        self.assertIn("Duplicate inspection identity rejected", self.memory.last_error)
        stats = self.memory.get_batch_stats("B")
        self.assertEqual(stats["total"], 1)

    def test_online_backup_is_readable_and_consistent(self):
        self.assertTrue(self.memory.add_entry(
            batch_id="BACKUP", part_id="P1", has_defect=False, defect_class="none"
        ))
        backup_path = os.path.join(self.tempdir.name, "backup", "quality.db")
        self.assertTrue(self.memory.backup_to(backup_path))
        backup = QualityMemory(backup_path)
        self.assertEqual(backup.get_batch_stats("BACKUP")["total"], 1)

    def test_control_audit_is_durable_and_queryable(self):
        self.assertTrue(self.memory.record_control_audit(
            audit_id="AUD_TEST1",
            operator_id="OP_A",
            action="EMERGENCY_STOP",
            reason="unit test",
            confirmation="CONFIRM EMERGENCY_STOP",
            snapshot_id="OPS_TEST",
            idempotency_key="CMD_TEST1",
            result_status="DISPATCHING",
            details={"step": "start"},
        ))
        self.assertTrue(self.memory.finalize_control_audit(
            "AUD_TEST1",
            "SIMULATED",
            signal_id="SIG_1",
            details={"step": "done"},
        ))
        payload = self.memory.get_recent_control_audits()
        self.assertEqual(payload["status"], "OK")
        self.assertEqual(payload["count"], 1)
        row = payload["records"][0]
        self.assertEqual(row["audit_id"], "AUD_TEST1")
        self.assertEqual(row["result_status"], "SIMULATED")
        self.assertEqual(row["signal_id"], "SIG_1")
        self.assertEqual(row["idempotency_key"], "CMD_TEST1")
        self.assertEqual(row["details"]["step"], "done")
        replay = self.memory.get_control_audit_by_idempotency("CMD_TEST1")
        self.assertEqual(replay["audit_id"], "AUD_TEST1")
        self.assertFalse(self.memory.record_control_audit(
            audit_id="AUD_TEST2",
            operator_id="OP_A",
            action="EMERGENCY_STOP",
            reason="duplicate idempotency key",
            confirmation="CONFIRM EMERGENCY_STOP",
            idempotency_key="CMD_TEST1",
        ))

    def test_control_audit_write_failure_is_reported(self):
        with patch.object(self.memory, "_get_connection", self.fail_connection):
            written = self.memory.record_control_audit(
                audit_id="AUD_FAIL",
                operator_id="OP_A",
                action="RESET",
                reason="unit test",
                confirmation="CONFIRM RESET",
            )
        self.assertFalse(written)
        self.assertFalse(self.memory.healthy)


if __name__ == "__main__":
    unittest.main()
