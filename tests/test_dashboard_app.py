"""Dashboard render tests for production snapshot console and isolated sandbox."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]

PRODUCTION_SNAPSHOT = {
    "schema_version": "1.1",
    "snapshot_id": "OPS_TESTSNAPSHOT",
    "generated_at_utc": "2026-08-28T16:00:00+00:00",
    "consistency": "single_api_response_non_transactional_components",
    "surface": "operations",
    "readiness": {
        "status": "DEGRADED",
        "device": "cpu",
        "system_state": "DEGRADED",
        "sensors": {"rgb": "OFFLINE", "thermal": "OFFLINE", "profiler": "OFFLINE"},
        "simulation_mode": False,
        "industrial_transport_ready": False,
        "traceability_ok": True,
        "industrial_interlock_latched": True,
        "onnx_available": False,
        "yolo_available": False,
        "primary_engine": "NONE",
    },
    "ready": False,
    "line_disposition": "HOLD_REQUIRED",
    "readiness_reasons": ["offline_sensors=rgb,thermal,profiler", "industrial_interlock_latched"],
    "roll": {
        "summary": {
            "roll_id": "ROLL_2026_CATL_001",
            "batch_id": "BATCH_2026_MSE_01",
            "line_speed_m_s": 1.8,
            "inspected_length_m": 12.5,
            "total_roll_length_m": 1200.0,
        },
        "defect_records": [],
    },
    "quality": {
        "stats": {
            "status": "OK",
            "total": 0,
            "failed": 0,
            "passed": 0,
            "held": 0,
            "pass_rate": None,
            "avg_latency_ms": None,
        },
        "spc": {"status": "NO DATA", "message": "No inspection data is available for this batch."},
        "recent_inspections": {"status": "OK", "count": 0, "records": []},
    },
    "industrial": {
        "modbus_registers": {"HR_1001": 0, "HR_1005_estop": 0},
        "opc_ua_nodes": {},
        "connection": {"plc_ip": "192.168.1.100", "modbus_port": 502, "status": "NOT READY"},
        "interlock_latched": True,
        "interlock_reason": "Startup PLC state is unverified; confirmed reset required",
    },
    "signals": [],
    "traceability": {
        "certificate_status": "OK",
        "certificate": {
            "certificate_id": "CERT_TEST",
            "payload_hash_sha256": "abc",
            "hmac_digital_signature": "def",
            "overall_quality_grade": "UNVERIFIED",
            "pass_rate_pct": None,
            "metric_provenance": "test fixture",
        },
        "certificate_error": None,
    },
    "control_audit": {"status": "OK", "count": 0, "records": []},
    "control_policy": {
        "endpoint": "/api/operations/control",
        "actions": {
            "EMERGENCY_STOP": {"confirmation": "CONFIRM EMERGENCY_STOP", "effect": "stop"},
            "RESET": {"confirmation": "CONFIRM RESET", "effect": "reset"},
            "INFERENCE_RESET": {"confirmation": "CONFIRM INFERENCE_RESET", "effect": "probe"},
        },
        "forbidden": ["PLC_PARAMETER_WRITE", "RECIPE_SLIDER", "DEFECT_INJECTION", "LIBAD_DEMO"],
    },
}


class FakeOperationsClient:
    def __init__(self, *args, **kwargs):
        pass

    def operations_snapshot(self, signal_limit=25):
        return PRODUCTION_SNAPSHOT

    def operations_control(self, **kwargs):
        return {
            "audit_id": "AUD_TEST",
            "status": "SIMULATED",
            "mock_mode": True,
            "acknowledged": False,
        }


class TestDashboardApp(unittest.TestCase):
    def test_production_dashboard_stops_without_local_stateful_fallback(self):
        with patch.dict(
            os.environ,
            {
                "SECURECOATING_ENV": "production",
                "SECURECOATING_DASHBOARD_SANDBOX": "false",
                "SECURECOATING_API_URL": "http://127.0.0.1:1",
            },
            clear=False,
        ):
            app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=30)
            app.run()
            self.assertEqual(list(app.exception), [])
            self.assertTrue(any("refuses local fallback" in item.value for item in app.error))

    def test_production_dashboard_renders_operate_diagnose_traceability(self):
        with patch.dict(
            os.environ,
            {
                "SECURECOATING_ENV": "production",
                "SECURECOATING_DASHBOARD_SANDBOX": "false",
                "SECURECOATING_API_URL": "http://127.0.0.1:1",
            },
            clear=False,
        ), patch("dashboard.api_client.InspectionApiClient", FakeOperationsClient):
            app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=30)
            app.run()
            self.assertEqual(list(app.exception), [])
            rendered = " ".join(item.value for item in app.markdown)
            self.assertIn("Line Operations", rendered)
            self.assertIn("HOLD_REQUIRED", rendered)
            self.assertIn("NO DATA", rendered)
            labels = [item.label for item in app.button]
            self.assertTrue(any("Submit control command" in label for label in labels))
            submit = next(item for item in app.button if "Submit control command" in item.label)
            self.assertTrue(submit.disabled)
            self.assertTrue(any("Snapshot is stale" in item.value for item in app.error))
            self.assertFalse(any("7-STAGE" in label for label in labels))
            self.assertFalse(any("90s" in label for label in labels))
            self.assertFalse(any("Send Parameter Offset" in label for label in labels))

    def test_dashboard_renders_without_exceptions_in_explicit_sandbox(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "SECURECOATING_ENV": "test",
                "SECURECOATING_DASHBOARD_SANDBOX": "true",
                "SECURECOATING_DB_PATH": str(Path(directory) / "quality.db"),
                "SECURECOATING_API_URL": "http://127.0.0.1:1",
            },
            clear=False,
        ):
            app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120)
            app.run()
            self.assertEqual(list(app.exception), [])
            labels = [item.label for item in app.sidebar.button]
            self.assertTrue(any("SANDBOX 7-STAGE" in label for label in labels))
            self.assertTrue(any("90s EVIDENCE" in label for label in labels))
            self.assertFalse(any("Send Parameter Offset" in label for label in labels))


if __name__ == "__main__":
    unittest.main()
