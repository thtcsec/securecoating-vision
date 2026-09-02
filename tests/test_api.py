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
import time
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi import HTTPException

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("SECURECOATING_ENV", "test")
os.environ.setdefault("SECURECOATING_ALLOW_UNAUTHENTICATED_DEMO", "true")
os.environ.setdefault("SECURECOATING_ENABLE_SENSOR_SIMULATION", "true")
API_TEST_TEMP_DIR = tempfile.mkdtemp(prefix="securecoating_api_test_")
atexit.register(shutil.rmtree, API_TEST_TEMP_DIR, ignore_errors=True)
os.environ.setdefault("SECURECOATING_DB_PATH", os.path.join(API_TEST_TEMP_DIR, "quality.db"))
os.environ.setdefault(
    "SECURECOATING_DATASET_CATALOG_CACHE",
    os.path.join(API_TEST_TEMP_DIR, "catalog_cache.json"),
)
os.environ.setdefault(
    "SECURECOATING_INSPECTION_ARTIFACT_DIR",
    os.path.join(API_TEST_TEMP_DIR, "inspection_artifacts"),
)

# Import after cwd so config paths resolve
from api.main import app  # noqa: E402
import api.main as api_main  # noqa: E402

ONNX_PATH = os.path.join(ROOT, "outputs", "model.onnx")
TEST_IMG = os.path.join(ROOT, "data", "test_set", "images", "defect_val_00000.jpg")
REAL_DEMO_IMG = os.path.join(ROOT, "data", "demo_real", "images", "image_1548.jpg")
COATINGVISION_MASK_1548 = os.path.join(
    ROOT, "data", "external", "coatingvision", "segmentation", "masks", "image_1548.png"
)


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
        self.assertIn("calibration_verified", body)
        self.assertIn("industrial_mock_mode", body)
        model_ready = bool(body.get("onnx_available") or body.get("yolo_available"))
        if resp.status_code == 200:
            self.assertEqual(body["status"], "HEALTHY")
            self.assertTrue(model_ready)
        else:
            self.assertEqual(body["status"], "DEGRADED")
            if body.get("simulation_mode"):
                self.assertFalse(body["calibration_verified"])
            if not model_ready:
                self.assertEqual(resp.status_code, 503)

    def test_list_samples(self):
        resp = self.client.get("/api/samples")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("samples", body)
        if os.path.isdir(os.path.join(ROOT, "data", "test_set", "images")):
            self.assertGreaterEqual(body["count"], 1)

    def test_dataset_catalog_is_bounded_and_metadata_only(self):
        resp = self.client.get("/api/dataset/catalog", params={"offset": 0, "limit": 3})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertLessEqual(body["returned"], 3)
        self.assertGreaterEqual(body["total"], 80)
        self.assertFalse(body["image_payloads_included"])
        self.assertTrue(body["provenance_verified"])
        self.assertEqual(body["dataset_license"], "CC BY 4.0")
        self.assertIn(body["library_scope"], {"checked_in_test_split", "local_figshare_archive"})
        self.assertGreaterEqual(body["checked_in_count"], 80)
        if body["library_scope"] == "local_figshare_archive":
            self.assertGreaterEqual(body["total"], 2000)
        for item in body["items"]:
            self.assertEqual(item["source_type"], "REAL_OPTICAL")
            self.assertEqual(os.path.basename(item["filename"]), item["filename"])
            self.assertIn("in_checked_in_split", item)
            self.assertIn("has_published_mask", item)
            self.assertIn("published_classes", item)
            self.assertIsInstance(item["published_classes"], list)
            self.assertIn("has_published_classes", item)
        self.assertIn("classification_label_count", body)
        self.assertIn("classification_flag_counts", body)
        self.assertIsInstance(body["classification_flag_counts"], dict)
        bad_flag = self.client.get(
            "/api/dataset/catalog", params={"offset": 0, "limit": 1, "class_flag": "../secret"}
        )
        self.assertEqual(bad_flag.status_code, 422)
        if body["classification_flag_counts"].get("Delamination"):
            filtered = self.client.get(
                "/api/dataset/catalog",
                params={"offset": 0, "limit": 3, "class_flag": "Delamination"},
            )
            self.assertEqual(filtered.status_code, 200)
            filtered_body = filtered.json()
            self.assertLessEqual(filtered_body["returned"], 3)
            self.assertEqual(filtered_body["class_flag"], "Delamination")
            for item in filtered_body["items"]:
                self.assertIn("Delamination", item["published_classes"])

    def test_dataset_preview_is_hash_verified_jpeg(self):
        resp = self.client.get("/api/dataset/images/image_1548.jpg")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.headers["content-type"].startswith("image/jpeg"))
        self.assertTrue(resp.content.startswith(b"\xff\xd8\xff"))
        self.assertEqual(len(resp.headers["x-dataset-sha256"]), 64)
        for view in ("mask", "blend", "heatmap"):
            evidence = self.client.get(
                "/api/dataset/images/image_1548.jpg", params={"view": view}
            )
            if os.path.isfile(COATINGVISION_MASK_1548):
                self.assertEqual(evidence.status_code, 200)
                self.assertTrue(evidence.content.startswith(b"\xff\xd8\xff"))
                self.assertEqual(evidence.headers["x-dataset-view"], view)
                self.assertEqual(
                    evidence.headers["x-dataset-label-source"],
                    "published-segmentation-mask",
                )
            else:
                self.assertEqual(evidence.status_code, 404)
        thumb = self.client.get(
            "/api/dataset/images/image_1548.jpg", params={"view": "thumb"}
        )
        self.assertEqual(thumb.status_code, 200)
        self.assertTrue(thumb.content.startswith(b"\xff\xd8\xff"))
        self.assertEqual(thumb.headers["x-dataset-view"], "thumb")
        again = self.client.get(
            "/api/dataset/images/image_1548.jpg", params={"view": "thumb"}
        )
        self.assertEqual(again.status_code, 200)
        self.assertEqual(again.content, thumb.content)
        resp = self.client.get("/api/dataset/images/not-in-manifest.jpg")
        self.assertEqual(resp.status_code, 404)

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

    def test_simulated_inspections_advance_roll_motion(self):
        before = api_main.web_synchronizer.current_pos_m
        resp = self.client.post(
            "/api/inspect",
            data={"part_id": "PART_MOTION_ADVANCE"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(api_main.web_synchronizer.current_pos_m, before)

    def test_corrupt_named_sample_returns_http_error_not_name_error(self):
        with patch.object(api_main, "_demo_sample_path", return_value=api_main.Path("broken.jpg")), patch.object(
            api_main.cv2, "imread", return_value=None
        ):
            with self.assertRaises(HTTPException) as raised:
                api_main._load_sample_image("broken.jpg")
        self.assertEqual(raised.exception.status_code, 400)

    def test_chunked_request_body_is_limited_before_form_parsing(self):
        async def run_request():
            async def chunks():
                yield b"x" * (api_main.MAX_REQUEST_SIZE + 1)

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                return await client.post(
                    "/api/inspect",
                    content=chunks(),
                    headers={"content-type": "application/octet-stream"},
                )

        response = asyncio.run(run_request())
        self.assertEqual(response.status_code, 413)

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
        os.path.isfile(ONNX_PATH) and os.path.isfile(REAL_DEMO_IMG),
        "ONNX model or real demo image missing",
    )
    def test_inspect_attributed_real_sample_stays_fail_closed(self):
        resp = self.client.post(
            "/api/inspect",
            data={
                "batch_id": "BATCH_2026_MSE_01",
                "part_id": "PART_API_SAMPLE",
                "sample_name": "image_1548.jpg",
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
        self.assertFalse(body["passed"])
        self.assertIn(body["gate_action"], {"HOLD", "REJECT"})
        roll_snapshot = self.client.get("/api/roll/active").json()
        ledger_ids = {
            record["defect_id"]
            for record in roll_snapshot["defect_records"]
        }
        response_ids = {record["defect_id"] for record in body["defects_found"]}
        self.assertTrue(response_ids.issubset(ledger_ids))
        for view in ("mask", "blend", "heatmap"):
            evidence = self.client.get(
                f"/api/inspections/{body['run_id']}/image",
                params={"view": view},
            )
            if os.path.isfile(COATINGVISION_MASK_1548):
                self.assertEqual(evidence.status_code, 200)
                self.assertTrue(evidence.content.startswith(b"\xff\xd8\xff"))
            else:
                self.assertEqual(evidence.status_code, 404)
        self.assertEqual(body.get("sample_name"), "image_1548.jpg")
        self.assertIsInstance(body.get("published_classes"), list)

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
        image_response = self.client.get(
            f"/api/inspections/{body['run_id']}/image"
        )
        self.assertEqual(image_response.status_code, 200)
        self.assertTrue(image_response.headers["content-type"].startswith("image/jpeg"))
        self.assertGreater(len(image_response.content), 1000)
        for view in ("raw", "input", "overlay"):
            view_response = self.client.get(
                f"/api/inspections/{body['run_id']}/image",
                params={"view": view},
            )
            self.assertEqual(view_response.status_code, 200)
            self.assertTrue(view_response.content.startswith(b"\xff\xd8\xff"))
        for view in ("mask", "blend", "heatmap"):
            view_response = self.client.get(
                f"/api/inspections/{body['run_id']}/image",
                params={"view": view},
            )
            self.assertEqual(view_response.status_code, 404)
        invalid_view = self.client.get(
            f"/api/inspections/{body['run_id']}/image",
            params={"view": "ground_truth"},
        )
        self.assertEqual(invalid_view.status_code, 422)

    def test_required_artifact_failure_is_persisted_fail_closed(self):
        with patch("api.main.REQUIRE_INSPECTION_ARTIFACTS", True), patch(
            "api.main._write_inspection_artifact", return_value=False
        ):
            resp = self.client.post(
                "/api/inspect",
                data={
                    "batch_id": "BATCH_2026_MSE_01",
                    "part_id": "PART_ARTIFACT_FAILURE",
                },
            )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["gate_action"], "HOLD")
        self.assertTrue(
            any("artifact write failed" in reason for reason in body["reject_reasons"])
        )
        snapshot = self.client.get("/api/operations/snapshot").json()
        persisted = next(
            row
            for row in snapshot["quality"]["recent_inspections"]["records"]
            if row["run_id"] == body["run_id"]
        )
        self.assertFalse(persisted["inspection_valid"])
        self.assertIn("artifact write failed", persisted["error_reason"])

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
        samples = self.client.get("/api/libad/samples")
        self.assertEqual(samples.status_code, 200)
        sample_body = samples.json()
        self.assertIn("official_dataset_present", sample_body)
        self.assertFalse(sample_body["comparable_to_paper"])
        if not sample_body.get("official_dataset_present"):
            self.assertEqual(sample_body["total"], 0)
            self.assertEqual(sample_body["items"], [])
        missing = self.client.get(
            "/api/libad/samples/not-a-sample/image", params={"view": "vis_a"}
        )
        self.assertEqual(missing.status_code, 404)

    def test_dataset_catalog_disk_cache_reloads_without_full_rebuild(self):
        first = self.client.get("/api/dataset/catalog", params={"offset": 0, "limit": 1})
        self.assertEqual(first.status_code, 200)
        total = first.json()["total"]
        api_main._dataset_catalog_cache = None
        api_main._dataset_catalog_by_name = {}
        api_main._dataset_catalog_identity = {}
        started = time.perf_counter()
        second = self.client.get("/api/dataset/catalog", params={"offset": 0, "limit": 1})
        elapsed = time.perf_counter() - started
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["total"], total)
        self.assertLess(elapsed, 8.0)

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
            "throughput",
        ):
            self.assertIn(key, body)
        self.assertIn("stats", body["quality"])
        self.assertIn("spc", body["quality"])
        self.assertIn("recent_inspections", body["quality"])
        self.assertIn("EMERGENCY_STOP", body["control_policy"]["actions"])
        self.assertNotIn("PLC_PARAMETER_WRITE", body["control_policy"]["actions"])
        self.assertEqual(body["throughput"]["evidence_class"], "MEASURED_LOCAL_INFERENCE_ONLY")
        self.assertFalse(body["throughput"]["factory_line_qualified"])
        self.assertEqual(body.get("snapshot_scope", "full"), "full")
        self.assertIn("dataset_catalog", body)
        self.assertIn("certificate_status", body["traceability"])

    def test_operations_snapshot_live_scope_omits_catalog_and_certificate(self):
        resp = self.client.get("/api/operations/snapshot", params={"scope": "live"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["snapshot_scope"], "live")
        self.assertNotIn("dataset_catalog", body)
        self.assertEqual(body["traceability"]["certificate_status"], "DEFERRED_TO_FULL_SNAPSHOT")
        self.assertIsNone(body["traceability"]["certificate"])
        self.assertIn("readiness", body)
        self.assertIn("line_disposition", body)
        self.assertIn("quality", body)
        self.assertIn("recent_inspections", body["quality"])
        self.assertIn("stats", body["quality"])

    def test_operations_snapshot_rejects_unknown_scope(self):
        resp = self.client.get("/api/operations/snapshot", params={"scope": "partial"})
        self.assertEqual(resp.status_code, 422)

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

    def test_published_mask_renders_heatmap_without_claiming_model_confidence(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            mask_dir = Path(tmp, "masks").resolve()
            mask_dir.mkdir()
            binary = np.zeros((48, 64), dtype=np.uint8)
            binary[10:30, 16:48] = 255
            self.assertTrue(cv2.imwrite(str(mask_dir / "frame.png"), binary))
            optical = np.full((48, 64, 3), 90, dtype=np.uint8)
            with patch.object(api_main, "COATINGVISION_SEGMENTATION_MASK_DIR", mask_dir):
                evidence = api_main._segmentation_evidence(optical, "frame.jpg")
            self.assertIsNotNone(evidence)
            self.assertEqual(set(evidence), {"mask", "blend", "heatmap"})
            self.assertTrue(np.any(evidence["mask"]))
            self.assertTrue(np.all(evidence["mask"][15, 30] == (255, 0, 255)))
            self.assertTrue(np.any(evidence["heatmap"][15, 30] != optical[15, 30]))
            unmatched = api_main._segmentation_evidence(optical, "missing.jpg")
            self.assertIsNone(unmatched)

    def test_ai_overlay_fills_predicted_regions(self):
        import numpy as np

        canvas = np.zeros((40, 50, 3), dtype=np.uint8)
        seg_mask = np.zeros((40, 50), dtype=np.uint8)
        seg_mask[5:20, 8:28] = 1
        overlay = api_main._draw_ai_overlay(
            canvas,
            [{"bbox": [8, 5, 20, 15], "class_name": "scratch", "confidence": 0.91}],
            seg_mask,
            50,
        )
        self.assertGreater(int(overlay.sum()), 0)
        self.assertTrue(np.any(overlay[10, 15] > 0))

    def test_classification_csv_parses_published_flags_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "labels.csv"
            csv_path.write_text(
                "original_file_name,file_name,Surface_Crack,Delamination,Pinhole,unclassified\n"
                "a.png,image_1.jpg,1,0,1,0\n"
                "b.png,image_2.jpg,0,0,0,1\n"
                "bad,../secret.jpg,1,0,0,0\n",
                encoding="utf-8",
            )
            labels = api_main._parse_classification_csv(csv_path)
            self.assertEqual(labels["image_1.jpg"], ["Surface_Crack", "Pinhole"])
            self.assertEqual(labels["image_2.jpg"], ["unclassified"])
            self.assertNotIn("../secret.jpg", labels)
            self.assertEqual(
                api_main._parse_classification_csv(csv_path.with_name("missing.csv")),
                {},
            )

    def test_inspect_persists_published_classes_for_demo_sample(self):
        if not os.path.isfile(REAL_DEMO_IMG):
            self.skipTest("demo sample missing")
        resp = self.client.post(
            "/api/inspect",
            data={
                "batch_id": "BATCH_2026_MSE_01",
                "part_id": "PART_PUBLISHED_CLASS",
                "sample_name": "image_1548.jpg",
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["sample_name"], "image_1548.jpg")
        self.assertIsInstance(body["published_classes"], list)
        snapshot = self.client.get("/api/operations/snapshot").json()
        persisted = next(
            row
            for row in snapshot["quality"]["recent_inspections"]["records"]
            if row["run_id"] == body["run_id"]
        )
        self.assertEqual(persisted["sample_name"], "image_1548.jpg")
        if persisted["published_classes"] is not None:
            self.assertIsInstance(persisted["published_classes"], list)


if __name__ == "__main__":
    unittest.main()
