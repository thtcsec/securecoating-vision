"""API smoke / unit tests for inspection endpoints."""

import os
import sys
import unittest
import asyncio
import struct
import zlib
import base64
import hashlib
import tempfile
import atexit
import shutil
from unittest.mock import patch

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("SECURECOATING_ENV", "test")
os.environ.setdefault("SECURECOATING_ALLOW_UNAUTHENTICATED_DEMO", "true")
os.environ.setdefault("SECURECOATING_ENABLE_SENSOR_SIMULATION", "true")
API_TEST_TEMP_DIR = tempfile.mkdtemp(prefix="securecoating_api_test_")
atexit.register(shutil.rmtree, API_TEST_TEMP_DIR, ignore_errors=True)
os.environ.setdefault("SECURECOATING_DB_PATH", os.path.join(API_TEST_TEMP_DIR, "quality.db"))

# Import after cwd so config paths resolve
from api.main import app  # noqa: E402

ONNX_PATH = os.path.join(ROOT, "outputs", "model.onnx")
TEST_IMG = os.path.join(ROOT, "data", "test_set", "images", "defect_val_00000.jpg")


class ASGITestClient:
    """Small synchronous facade over httpx's non-deprecated ASGI transport."""

    def request(self, method, url, **kwargs):
        async def run_request():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(run_request())

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = ASGITestClient()

    def test_health(self):
        resp = self.client.get("/health")
        self.assertIn(resp.status_code, {200, 503})
        body = resp.json()
        self.assertIn(body["status"], {"HEALTHY", "DEGRADED"})
        self.assertIn("onnx_available", body)
        self.assertIn("industrial_interlock_latched", body)
        model_ready = bool(body.get("onnx_available") or body.get("yolo_available"))
        if resp.status_code == 200:
            self.assertEqual(body["status"], "HEALTHY")
            self.assertTrue(model_ready)
        else:
            self.assertEqual(body["status"], "DEGRADED")
            if not model_ready:
                self.assertEqual(resp.status_code, 503)

    def test_list_samples(self):
        resp = self.client.get("/api/samples")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("samples", body)
        if os.path.isdir(os.path.join(ROOT, "data", "test_set", "images")):
            self.assertGreaterEqual(body["count"], 1)

    def test_active_roll_discovery_is_authoritative(self):
        resp = self.client.get("/api/roll/active")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["summary"]["batch_id"], "BATCH_2026_MSE_01")

    def test_demo_samples_are_hidden_when_simulation_is_disabled(self):
        with patch("api.main.SIMULATION_MODE", False):
            resp = self.client.get("/api/samples")
        self.assertEqual(resp.status_code, 404)

    def test_api_key_guard_fails_closed(self):
        with patch("api.main.REQUIRE_API_KEY", True), patch("api.main.API_KEY", "unit-secret"):
            denied = self.client.get("/api/samples")
            allowed = self.client.get("/api/samples", headers={"x-api-key": "unit-secret"})
        self.assertEqual(denied.status_code, 401)
        self.assertEqual(allowed.status_code, 200)

    def test_inspect_camera_sim(self):
        resp = self.client.post(
            "/api/inspect",
            data={
                "batch_id": "BATCH_2026_MSE_01",
                "part_id": "PART_API_001",
                "thermal_online": "true",
                "profiler_online": "true",
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["part_id"], "PART_API_001")
        self.assertIn("engine", body)
        self.assertIn("latency_ms", body)
        self.assertIn("run_id", body)
        self.assertTrue(str(body["run_id"]).startswith("RUN_"))

    def test_inspect_rejects_non_active_batch(self):
        resp = self.client.post(
            "/api/inspect",
            data={"batch_id": "BATCH_NOT_ACTIVE", "part_id": "PART_BAD_BATCH"},
        )
        self.assertEqual(resp.status_code, 409)

    def test_identifier_validation_rejects_log_and_path_metacharacters(self):
        resp = self.client.post(
            "/api/inspect",
            data={"batch_id": "BATCH_2026_MSE_01", "part_id": "bad/id\nvalue"},
        )
        self.assertEqual(resp.status_code, 422)

    def test_inspect_defaults_to_active_batch(self):
        resp = self.client.post(
            "/api/inspect",
            data={"part_id": "PART_ACTIVE_BATCH_DEFAULT"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["batch_id"], "BATCH_2026_MSE_01")

    @unittest.skipUnless(
        os.path.isfile(ONNX_PATH) and os.path.isfile(TEST_IMG),
        "ONNX model or test image missing",
    )
    def test_inspect_sample_name_detects_defect(self):
        resp = self.client.post(
            "/api/inspect",
            data={
                "batch_id": "BATCH_2026_MSE_01",
                "part_id": "PART_API_SAMPLE",
                "sample_name": "defect_val_00000.jpg",
                "thermal_online": "true",
                "profiler_online": "true",
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(
            str(body["engine"]).startswith("ONNX")
            or str(body["engine"]).startswith("YOLO")
        )
        self.assertGreaterEqual(len(body["defects_found"]), 1)
        self.assertFalse(body["passed"])

    @unittest.skipUnless(os.path.isfile(TEST_IMG), "test image missing")
    def test_inspect_upload(self):
        with open(TEST_IMG, "rb") as f:
            resp = self.client.post(
                "/api/inspect",
                data={
                    "batch_id": "BATCH_2026_MSE_01",
                    "part_id": "PART_API_UPLOAD",
                },
                files={"image": ("defect_val_00000.jpg", f, "image/jpeg")},
            )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["part_id"], "PART_API_UPLOAD")

    def test_upload_rejects_wrong_media_type(self):
        resp = self.client.post(
            "/api/inspect",
            files={"image": ("payload.txt", b"not an image", "text/plain")},
        )
        self.assertEqual(resp.status_code, 415)

    def test_image_header_bomb_is_rejected_before_decode(self):
        from api.main import _decode_upload
        from fastapi import HTTPException

        def chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

        png = b"\x89PNG\r\n\x1a\n" + chunk(
            b"IHDR", struct.pack(">IIBBBBB", 100_000, 100_000, 8, 2, 0, 0, 0)
        ) + chunk(b"IEND", b"")
        with self.assertRaises(HTTPException) as raised:
            _decode_upload(png)
        self.assertIn(raised.exception.status_code, {400, 413})

    def test_latency_metrics(self):
        resp = self.client.get("/api/metrics/latency")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("p50_ms", body)
        self.assertIn("target_ms", body)

    def test_industrial_state(self):
        resp = self.client.get("/api/industrial/state")
        self.assertEqual(resp.status_code, 200)

    def test_unknown_roll_never_returns_active_roll_data(self):
        resp = self.client.get("/api/roll/ROLL_DOES_NOT_EXIST/map")
        self.assertEqual(resp.status_code, 404)

    def test_confidence_requires_bbox_overlap(self):
        from api.main import _attach_confidences

        defects = [{"class_name": "scratch", "bbox": [0, 0, 10, 10]}]
        detections = [{
            "class_name": "scratch",
            "box": [1000, 1000, 1010, 1010],
            "confidence": 0.99,
        }]
        matched = _attach_confidences(defects, detections)
        self.assertIsNone(matched[0]["confidence"])

    def test_spc_passport_and_slitting_yield(self):
        resp_pass = self.client.get("/api/spc/passport")
        self.assertEqual(resp_pass.status_code, 200)
        body_pass = resp_pass.json()
        self.assertIn("passport_standard", body_pass)
        self.assertIn("six_sigma_quality_summary", body_pass)

        resp_slit = self.client.get("/api/spc/slitting-yield")
        self.assertEqual(resp_slit.status_code, 200)
        body_slit = resp_slit.json()
        self.assertIn("slitting_optimization", body_slit)

    def test_libad_protocol_discloses_validation_extension(self):
        resp = self.client.get("/api/libad/protocol")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("High-Throughput and Zero-Trust Edge-Cloud Pipeline", body["registered_title"])
        self.assertEqual(
            body["tagline"],
            "Evidence-Gated Multimodal Inspection for Battery Electrode Manufacturing",
        )
        self.assertIn("does not claim DA-Core", body["citation"]["da_core_attribution"])

    def test_libad_demo_case_four_is_hold(self):
        resp = self.client.get("/api/libad/demo/4")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["decision"]["action"], "HOLD")
        self.assertIn("hmac_digital_signature", body["certificate"])
        self.assertEqual(body["evidence_class"], "protocol_fixture")
        self.assertFalse(body["comparable_to_paper"])
        self.assertNotIn("frames", body)
        for encoded in body["frames_png_base64"].values():
            self.assertTrue(base64.b64decode(encoded, validate=True).startswith(b"\x89PNG"))

    def test_libad_demo_hidden_when_simulation_is_disabled(self):
        with patch("api.main.SIMULATION_MODE", False):
            resp = self.client.get("/api/libad/demo/4")
        self.assertEqual(resp.status_code, 404)

    def test_operations_snapshot_is_self_contained(self):
        resp = self.client.get("/api/operations/snapshot")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(str(body["snapshot_id"]).startswith("OPS_"))
        self.assertEqual(body["surface"], "operations")
        for key in (
            "readiness", "roll", "quality", "industrial", "signals",
            "traceability", "control_audit", "control_policy", "line_disposition",
        ):
            self.assertIn(key, body)
        self.assertIn("stats", body["quality"])
        self.assertIn("spc", body["quality"])
        self.assertIn("recent_inspections", body["quality"])
        self.assertIn("EMERGENCY_STOP", body["control_policy"]["actions"])
        self.assertNotIn("PLC_PARAMETER_WRITE", body["control_policy"]["actions"])

    def test_operations_control_rejects_wrong_confirmation_without_dispatch(self):
        before = self.client.get("/api/industrial/state").json()
        resp = self.client.post(
            "/api/operations/control",
            data={
                "action": "EMERGENCY_STOP",
                "operator_id": "OP_UNIT",
                "confirmation": "yes",
                "reason": "unit test wrong phrase",
                "idempotency_key": "CMD_WRONG_CONFIRMATION",
            },
        )
        self.assertEqual(resp.status_code, 409)
        after = self.client.get("/api/industrial/state").json()
        self.assertEqual(after["interlock_latched"], before["interlock_latched"])

    def test_operations_control_estop_is_audited_and_simulated(self):
        snap = self.client.get("/api/operations/snapshot").json()
        resp = self.client.post(
            "/api/operations/control",
            data={
                "action": "EMERGENCY_STOP",
                "operator_id": "OP_UNIT",
                "confirmation": "CONFIRM EMERGENCY_STOP",
                "reason": "unit test emergency stop",
                "snapshot_id": snap["snapshot_id"],
                "idempotency_key": "CMD_UNIT_ESTOP",
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(str(body["audit_id"]).startswith("AUD_"))
        self.assertEqual(body["status"], "SIMULATED")
        self.assertTrue(body["mock_mode"])
        self.assertFalse(body["acknowledged"])
        later = self.client.get("/api/operations/snapshot").json()
        audit_ids = [row["audit_id"] for row in later["control_audit"]["records"]]
        self.assertIn(body["audit_id"], audit_ids)
        reset = self.client.post(
            "/api/operations/control",
            data={
                "action": "RESET",
                "operator_id": "OP_UNIT",
                "confirmation": "CONFIRM RESET",
                "reason": "unit test restore after estop",
                "snapshot_id": later["snapshot_id"],
                "idempotency_key": "CMD_UNIT_RESET",
            },
        )
        self.assertEqual(reset.status_code, 200)

    def test_reset_rejects_fabricated_snapshot(self):
        resp = self.client.post(
            "/api/operations/control",
            data={
                "action": "RESET",
                "operator_id": "OP_UNIT",
                "confirmation": "CONFIRM RESET",
                "reason": "fabricated snapshot test",
                "snapshot_id": "OPS_FAKE1234",
                "idempotency_key": "CMD_FAKE_SNAPSHOT",
            },
        )
        self.assertEqual(resp.status_code, 409)
        self.assertIn("unknown", resp.json()["detail"].lower())

    def test_control_retry_is_idempotent(self):
        data = {
            "action": "EMERGENCY_STOP",
            "operator_id": "OP_RETRY",
            "confirmation": "CONFIRM EMERGENCY_STOP",
            "reason": "idempotency replay verification",
            "idempotency_key": "CMD_REPLAY_ESTOP",
        }
        first = self.client.post("/api/operations/control", data=data)
        second = self.client.post("/api/operations/control", data=data)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json()["audit_id"], second.json()["audit_id"])
        self.assertTrue(second.json()["idempotent_replay"])

    def test_legacy_control_routes_are_disabled(self):
        self.assertEqual(self.client.post("/api/industrial/estop").status_code, 410)
        self.assertEqual(self.client.post("/api/industrial/reset").status_code, 410)
        self.assertEqual(self.client.post("/api/system/inference-reset").status_code, 410)

    def test_production_control_binds_operator_identity_to_token(self):
        token = "unit-operator-secret"
        credential = {
            "operator_id": "OP_VERIFIED",
            "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
            "roles": {"operator"},
        }
        data = {
            "action": "EMERGENCY_STOP",
            "operator_id": "OP_IMPOSTOR",
            "confirmation": "CONFIRM EMERGENCY_STOP",
            "reason": "operator identity binding test",
            "idempotency_key": "CMD_OPERATOR_BINDING",
        }
        with patch("api.main.ENVIRONMENT", "production"), patch(
            "api.main.OPERATOR_CREDENTIALS", [credential]
        ):
            mismatch = self.client.post(
                "/api/operations/control",
                data=data,
                headers={"x-operator-token": token},
            )
            missing = self.client.post("/api/operations/control", data=data)
        self.assertEqual(mismatch.status_code, 403)
        self.assertEqual(missing.status_code, 401)

    def test_expired_snapshot_blocks_reset(self):
        snap = self.client.get("/api/operations/snapshot").json()
        with patch("api.main.time.monotonic", return_value=10**9):
            resp = self.client.post(
                "/api/operations/control",
                data={
                    "action": "RESET",
                    "operator_id": "OP_UNIT",
                    "confirmation": "CONFIRM RESET",
                    "reason": "expired snapshot verification",
                    "snapshot_id": snap["snapshot_id"],
                    "idempotency_key": "CMD_EXPIRED_SNAPSHOT",
                },
            )
        self.assertEqual(resp.status_code, 409)
        self.assertIn("expired", resp.json()["detail"].lower())

    def test_dispatch_exception_is_audited_and_latches_failure(self):
        with patch("api.main._dispatch_confirmed_control", side_effect=RuntimeError("boom")), patch(
            "api.main._latch_control_failure"
        ) as latch:
            resp = self.client.post(
                "/api/operations/control",
                data={
                    "action": "EMERGENCY_STOP",
                    "operator_id": "OP_UNIT",
                    "confirmation": "CONFIRM EMERGENCY_STOP",
                    "reason": "dispatch exception verification",
                    "idempotency_key": "CMD_DISPATCH_EXCEPTION",
                },
            )
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp.json()["status"], "DISPATCH_EXCEPTION")
        latch.assert_called_once()

    def test_audit_finalization_failure_requests_fail_closed_latch(self):
        with patch("api.main.quality_mem.finalize_control_audit", return_value=False), patch(
            "api.main._latch_control_failure"
        ) as latch:
            resp = self.client.post(
                "/api/operations/control",
                data={
                    "action": "EMERGENCY_STOP",
                    "operator_id": "OP_UNIT",
                    "confirmation": "CONFIRM EMERGENCY_STOP",
                    "reason": "audit finalization fault injection",
                    "idempotency_key": "CMD_FINALIZE_FAILURE",
                },
            )
        self.assertEqual(resp.status_code, 503)
        latch.assert_called_once()

    def test_inference_reset_does_not_report_a_plc_ack(self):
        from api import main as api_main

        with patch.object(api_main.failsafe, "request_inference_reset_probe", return_value=True):
            outcome = api_main._dispatch_confirmed_control(
                "INFERENCE_RESET", "unit inference reset"
            )
        self.assertIsNone(outcome["acknowledged"])
        self.assertTrue(outcome["reset_probe_accepted"])
        self.assertIsNone(outcome["signal_id"])


if __name__ == "__main__":
    unittest.main()
