"""API smoke / unit tests for inspection endpoints."""

import os
import sys
import unittest
import asyncio
import struct
import zlib
from unittest.mock import patch

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("SECURECOATING_ENV", "test")
os.environ.setdefault("SECURECOATING_ALLOW_UNAUTHENTICATED_DEMO", "true")
os.environ.setdefault("SECURECOATING_ENABLE_SENSOR_SIMULATION", "true")

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
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn(body["status"], {"HEALTHY", "DEGRADED"})
        self.assertIn("onnx_available", body)
        self.assertIn("industrial_interlock_latched", body)

    def test_list_samples(self):
        resp = self.client.get("/api/samples")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("samples", body)
        if os.path.isdir(os.path.join(ROOT, "data", "test_set", "images")):
            self.assertGreaterEqual(body["count"], 1)

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
                "batch_id": "TEST_BATCH",
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
        if body["system_state"] != "OPTIMAL":
            self.assertFalse(body["passed"])
            self.assertEqual(body["gate_action"], "HOLD")

    @unittest.skipUnless(
        os.path.isfile(ONNX_PATH) and os.path.isfile(TEST_IMG),
        "ONNX model or test image missing",
    )
    def test_inspect_sample_name_detects_defect(self):
        resp = self.client.post(
            "/api/inspect",
            data={
                "batch_id": "TEST_BATCH",
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
                    "batch_id": "TEST_BATCH",
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


if __name__ == "__main__":
    unittest.main()
