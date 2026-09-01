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
        "recent_inspections": {
            "status": "OK",
            "count": 1,
            "records": [
                {
                    "timestamp": "2026-08-28T16:00:00+00:00",
                    "part_id": "PART_HIST",
                    "run_id": "RUN_TESTHIST0001",
                    "gate_action": "HOLD",
                    "system_state": "DEGRADED",
                    "defect_class": "surface_crack",
                    "latency_ms": 12.3,
                    "inspection_valid": False,
                    "error_reason": "test fixture",
                    "has_defect": True,
                    "max_length_mm": 0.0,
                    "max_area_mm2": 0.0,
                    "peak_height_um": 0.0,
                    "fallback_active": False,
                    "model_version": "2.0.0",
                    "roll_id": "ROLL_2026_CATL_001",
                    "peak_confidence": 0.81,
                    "sample_name": "image_1.jpg",
                    "published_classes": ["Surface_Crack"],
                    "image_available": False,
                }
            ],
        },
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
    "dataset_catalog": {
        "total": 88,
        "offset": 0,
        "limit": 0,
        "returned": 0,
        "items": [],
        "provenance_verified": True,
        "library_scope": "checked_in_test_split",
        "checked_in_count": 88,
        "archive_count": 0,
        "parent_archive_images": 2227,
        "doi_url": "https://doi.org/10.6084/m9.figshare.29260121.v1",
        "dataset_name": "CoatingVision held-out test-split optical frames",
        "dataset_license": "CC BY 4.0",
        "dataset_source": "CoatingVision, Figshare DOI 10.6084/m9.figshare.29260121.v1",
        "image_payloads_included": False,
        "classification_flag_counts": {
            "Surface_Crack": 70,
            "Delamination": 8,
            "Pinhole": 12,
            "unclassified": 2,
        },
        "classification_empty_positive_count": 1,
        "classification_multi_label_count": 10,
        "classification_labeled_frame_count": 88,
    },
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
        self.timeout_seconds = 5.0
        self.snapshot_timeout_seconds = 45.0

    def operations_snapshot(self, signal_limit=25, timeout_seconds=None, scope="full"):
        payload = dict(PRODUCTION_SNAPSHOT)
        payload["snapshot_scope"] = scope
        if scope == "live":
            payload = dict(payload)
            payload.pop("dataset_catalog", None)
            payload["traceability"] = {
                "certificate_status": "DEFERRED_TO_FULL_SNAPSHOT",
                "certificate": None,
                "certificate_error": None,
            }
        return payload

    def operations_control(self, **kwargs):
        return {
            "audit_id": "AUD_TEST",
            "status": "SIMULATED",
            "mock_mode": True,
            "acknowledged": False,
        }

    def dataset_catalog(self, offset=0, limit=24, class_flag=None):
        return {
            "total": 88,
            "offset": offset,
            "limit": limit,
            "returned": min(limit, max(0, 88 - offset)),
            "items": [],
            "provenance_verified": True,
            "library_scope": "checked_in_test_split",
            "checked_in_count": 88,
            "archive_count": 0,
            "parent_archive_images": 2227,
            "doi_url": "https://doi.org/10.6084/m9.figshare.29260121.v1",
            "dataset_name": "CoatingVision held-out test-split optical frames",
            "dataset_license": "CC BY 4.0",
            "dataset_source": "CoatingVision, Figshare DOI 10.6084/m9.figshare.29260121.v1",
            "image_payloads_included": False,
            "classification_label_count": 0,
            "class_flag": class_flag,
        }

    def libad_samples(self, offset=0, limit=12):
        return {
            "official_dataset_present": False,
            "total": 0,
            "items": [],
            "source": "not_mounted",
            "comparable_to_paper": False,
        }

    def libad_protocol(self):
        return {
            "citation": {
                "title": "LIBAD: A Multimodal Anomaly Detection Benchmark for Li-Ion Battery Electrode Manufacturing",
                "url": "https://arxiv.org/abs/2608.07958",
                "dataset_url": "https://huggingface.co/datasets/Evenrose/LIBAD",
            },
            "paper_result_note": "Paper-table metrics are not claimed from this fixture.",
            "dataset": {
                "official_dataset_present": False,
                "official_protocol_complete": False,
                "comparable_to_paper": False,
                "comparability_blockers": [
                    "Official LIBAD dataset directory is absent or empty"
                ],
            },
            "local_contribution": "Evidence-gated PASS/REJECT/HOLD.",
        }


class TestDashboardApp(unittest.TestCase):
    def test_production_console_refreshes_authoritative_snapshot(self):
        source = (ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
        self.assertIn('@st.fragment(run_every="5s")', source)
        self.assertIn("render_live_strip", source)
        self.assertIn("refresh_live_telemetry", source)
        self.assertIn("render_operator_views", source)
        self.assertIn("ops_force_reload", source)
        self.assertIn("animation-duration: 0s", source)
        self.assertIn("stSidebarCollapsedControl", source)
        self.assertIn('scope="live"', source)
        self.assertIn("_retain_full_snapshot_fields", source)
        self.assertNotIn("render_production_console(snapshot, api_client)", source)

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
            markdown = " ".join(item.value for item in app.markdown)
            self.assertIn("Visual inspection replay", markdown)
            self.assertIn("REAL CAPTURE", markdown)
            self.assertIn("PUBLISHED LABEL", markdown)
            self.assertIn("LABEL HEATMAP", markdown)
            self.assertIn("replay-grid", markdown)
            self.assertIn("Line Operations", markdown)
            self.assertIn("HOLD_REQUIRED", markdown)
            self.assertIn("NO DATA", markdown)
            self.assertIn("Inference Engine", markdown)
            self.assertIn("NO MODEL", markdown)
            self.assertIn("PUBLISHED COATINGVISION CLASS", markdown)
            self.assertIn("AI DETECTOR CLASS", markdown)
            self.assertIn("Surface Crack", markdown)
            self.assertNotIn("Bắt đầu", markdown)
            self.assertFalse(any("Submit control command" in item.label for item in app.button))

            app.radio(key="ops_view").set_value("Guide")
            app.run()
            self.assertEqual(list(app.exception), [])
            self.assertIn(
                "Get started in 2 minutes",
                " ".join(item.value for item in app.markdown),
            )

            app.radio(key="ops_view").set_value("Operate")
            app.run()
            self.assertEqual(list(app.exception), [])
            labels = [item.label for item in app.button]
            self.assertTrue(any("Submit control command" in label for label in labels))
            submit = next(item for item in app.button if "Submit control command" in item.label)
            self.assertTrue(submit.disabled)
            self.assertTrue(any("Snapshot is stale" in item.value for item in app.error))
            self.assertFalse(any("7-STAGE" in label for label in labels))
            self.assertFalse(any("90s" in label for label in labels))
            self.assertFalse(any("Send Parameter Offset" in label for label in labels))
            self.assertNotIn(
                "Get started in 2 minutes",
                " ".join(item.value for item in app.markdown),
            )

            app.radio(key="ops_view").set_value("Dataset")
            app.run()
            self.assertEqual(list(app.exception), [])
            dataset_markdown = " ".join(item.value for item in app.markdown)
            self.assertIn("Dataset Library", dataset_markdown)
            self.assertIn("Published class mix", dataset_markdown)
            self.assertIn("mix-row", dataset_markdown)
            self.assertTrue(
                any("original CoatingVision" in item.value for item in app.info)
                or "Original frames" in dataset_markdown
            )
            self.assertNotIn("Get started in 2 minutes", dataset_markdown)
            self.assertTrue(any(item.label == "Previous" for item in app.button))
            self.assertTrue(any(item.label == "Next" for item in app.button))
            next_page = next(item for item in app.button if item.label == "Next")
            next_page.click()
            app.run()
            self.assertEqual(list(app.exception), [])
            self.assertEqual(app.radio(key="ops_view").value, "Dataset")
            self.assertNotIn(
                "Get started in 2 minutes",
                " ".join(item.value for item in app.markdown),
            )

            app.radio(key="ops_view").set_value("Multimodal")
            app.run()
            self.assertEqual(list(app.exception), [])
            multimodal_markdown = " ".join(item.value for item in app.markdown)
            self.assertIn("External multimodal lane", multimodal_markdown)
            self.assertIn("NOT MOUNTED", multimodal_markdown)
            self.assertIn("arxiv.org/abs/2608.07958", multimodal_markdown)
            self.assertNotIn("Get started in 2 minutes", multimodal_markdown)

            app.radio(key="ops_view").set_value("Diagnose")
            app.run()
            self.assertEqual(list(app.exception), [])
            diagnose_markdown = " ".join(item.value for item in app.markdown)
            diagnose_captions = " ".join(item.value for item in app.caption)
            self.assertIn("Readiness", diagnose_markdown)
            self.assertIn("Sensors and model", diagnose_markdown)
            self.assertIn("Throughput evidence", diagnose_markdown)
            self.assertIn("Views stay on this rail", diagnose_captions)

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
