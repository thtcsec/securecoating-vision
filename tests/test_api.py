"""API smoke / unit tests for inspection endpoints."""

import os
import sys
import unittest

from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

# Import after cwd so config paths resolve
from api.main import app  # noqa: E402

ONNX_PATH = os.path.join(ROOT, "outputs", "model.onnx")
TEST_IMG = os.path.join(ROOT, "data", "test_set", "images", "defect_val_00000.jpg")


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn(body["status"], {"HEALTHY", "DEGRADED"})
        self.assertIn("onnx_available", body)

    def test_list_samples(self):
        resp = self.client.get("/api/samples")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("samples", body)
        if os.path.isdir(os.path.join(ROOT, "data", "test_set", "images")):
            self.assertGreaterEqual(body["count"], 1)

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

    def test_latency_metrics(self):
        resp = self.client.get("/api/metrics/latency")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("p50_ms", body)
        self.assertIn("target_ms", body)

    def test_industrial_state(self):
        resp = self.client.get("/api/industrial/state")
        self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
