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
        self.assertEqual(stats["status"], "ERROR")
        self.assertIsNone(stats["pass_rate"])
        self.assertEqual(spc["status"], "UNKNOWN")
        self.assertEqual(latency["status"], "ERROR")
        self.assertIsNone(latency["p50_ms"])

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


if __name__ == "__main__":
    unittest.main()
