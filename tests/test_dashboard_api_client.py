"""Tests for the dashboard's read-only API client."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))

from api_client import InspectionApiClient


class TestInspectionApiClient(unittest.TestCase):
    def test_get_sends_api_key_and_decodes_object(self):
        client = InspectionApiClient("http://api.local/", api_key="secret", timeout_seconds=2)
        response = Mock(status_code=200)
        response.json.return_value = {"status": "HEALTHY"}
        with patch.object(client.session, "get", return_value=response) as get:
            self.assertEqual(client.health(), {"status": "HEALTHY"})
        get.assert_called_once_with(
            "http://api.local/health",
            params=None,
            headers={"x-api-key": "secret"},
            timeout=2,
        )
        response.raise_for_status.assert_not_called()

    def test_degraded_health_is_authoritative_not_offline_fallback(self):
        client = InspectionApiClient("http://api.local")
        response = Mock(status_code=503)
        response.json.return_value = {"status": "DEGRADED", "system_state": "EMERGENCY"}
        with patch.object(client.session, "get", return_value=response):
            payload = client.health()
        self.assertEqual(payload["status"], "DEGRADED")
        response.raise_for_status.assert_not_called()

    def test_http_errors_are_not_hidden(self):
        client = InspectionApiClient("http://api.local")
        response = Mock(status_code=503)
        response.raise_for_status.side_effect = RuntimeError("503")
        with patch.object(client.session, "get", return_value=response):
            with self.assertRaises(RuntimeError):
                client.batch_stats("BATCH")

    def test_non_object_json_is_rejected(self):
        client = InspectionApiClient("http://api.local")
        response = Mock(status_code=200)
        response.json.return_value = []
        with patch.object(client.session, "get", return_value=response):
            with self.assertRaises(ValueError):
                client.get("/health")

    def test_libad_demo_uses_authoritative_api(self):
        client = InspectionApiClient("http://api.local")
        response = Mock(status_code=200)
        response.json.return_value = {"case_id": 4, "decision": {"action": "HOLD"}}
        with patch.object(client.session, "get", return_value=response) as get:
            payload = client.libad_demo(4)
        self.assertEqual(payload["decision"]["action"], "HOLD")
        get.assert_called_once_with(
            "http://api.local/api/libad/demo/4",
            params=None,
            headers={},
            timeout=5.0,
        )

    def test_libad_demo_rejects_unknown_case_without_request(self):
        client = InspectionApiClient("http://api.local")
        with patch.object(client.session, "get") as get, self.assertRaises(ValueError):
            client.libad_demo(5)
        get.assert_not_called()

    def test_operations_snapshot_sends_bounded_limit(self):
        client = InspectionApiClient("http://api.local")
        response = Mock(status_code=200)
        response.json.return_value = {"snapshot_id": "OPS_1", "schema_version": "1.1"}
        with patch.object(client.session, "get", return_value=response) as get:
            payload = client.operations_snapshot(signal_limit=10)
        self.assertEqual(payload["snapshot_id"], "OPS_1")
        get.assert_called_once_with(
            "http://api.local/api/operations/snapshot",
            params={"signal_limit": 10},
            headers={},
            timeout=45.0,
        )

    def test_operations_snapshot_live_refresh_can_use_short_timeout(self):
        client = InspectionApiClient("http://api.local")
        response = Mock(status_code=200)
        response.json.return_value = {"snapshot_id": "OPS_1", "schema_version": "1.1"}
        with patch.object(client.session, "get", return_value=response) as get:
            client.operations_snapshot(signal_limit=10, timeout_seconds=5.0, scope="live")
        get.assert_called_once_with(
            "http://api.local/api/operations/snapshot",
            params={"signal_limit": 10, "scope": "live"},
            headers={},
            timeout=5.0,
        )

    def test_operations_snapshot_rejects_unknown_scope(self):
        client = InspectionApiClient("http://api.local")
        with patch.object(client.session, "get") as get, self.assertRaises(ValueError):
            client.operations_snapshot(scope="partial")
        get.assert_not_called()

    def test_operations_control_posts_confirmation_fields(self):
        client = InspectionApiClient("http://api.local", api_key="secret")
        response = Mock(status_code=200)
        response.json.return_value = {"audit_id": "AUD_1", "status": "SIMULATED"}
        with patch.object(client.session, "post", return_value=response) as post:
            payload = client.operations_control(
                action="EMERGENCY_STOP",
                operator_id="OP_1",
                confirmation="CONFIRM EMERGENCY_STOP",
                reason="unit test stop",
                snapshot_id="OPS_1",
                idempotency_key="CMD_1",
                operator_token="operator-secret",
            )
        self.assertEqual(payload["audit_id"], "AUD_1")
        self.assertEqual(payload["_http_status"], 200)
        post.assert_called_once_with(
            "http://api.local/api/operations/control",
            data={
                "action": "EMERGENCY_STOP",
                "operator_id": "OP_1",
                "confirmation": "CONFIRM EMERGENCY_STOP",
                "reason": "unit test stop",
                "snapshot_id": "OPS_1",
                "idempotency_key": "CMD_1",
            },
            headers={"x-api-key": "secret", "x-operator-token": "operator-secret"},
            timeout=5.0,
        )

    def test_inspection_image_is_bounded_and_type_checked(self):
        client = InspectionApiClient("http://api.local", api_key="secret")
        response = Mock(status_code=200)
        response.headers = {"content-type": "image/jpeg"}
        response.iter_content.return_value = [b"jpeg-bytes"]
        with patch.object(client.session, "get", return_value=response) as get:
            payload = client.inspection_image("RUN_0123456789AB", "raw")
        self.assertEqual(payload, b"jpeg-bytes")
        get.assert_called_once_with(
            "http://api.local/api/inspections/RUN_0123456789AB/image?view=raw",
            headers={"x-api-key": "secret"},
            timeout=5.0,
            stream=True,
        )
        with patch.object(client.session, "get", return_value=response):
            self.assertEqual(
                client.inspection_image("RUN_0123456789AB", "heatmap"),
                b"jpeg-bytes",
            )

    def test_inspection_image_rejects_unknown_view(self):
        client = InspectionApiClient("http://api.local")
        with self.assertRaises(ValueError):
            client.inspection_image("RUN_0123456789AB", "ground_truth")

    def test_inspection_image_rejects_oversized_response(self):
        client = InspectionApiClient("http://api.local")
        response = Mock(status_code=200)
        response.headers = {"content-type": "image/jpeg"}
        response.iter_content.return_value = [b"x" * 11]
        with patch.object(client.session, "get", return_value=response):
            with self.assertRaises(ValueError):
                client.get_bytes("/image", max_bytes=10)

    def test_dataset_catalog_sends_bounded_page(self):
        client = InspectionApiClient("http://api.local")
        response = Mock(status_code=200)
        response.json.return_value = {"total": 50, "offset": 24, "limit": 24, "items": []}
        with patch.object(client.session, "get", return_value=response) as get:
            payload = client.dataset_catalog(offset=24, limit=24)
        self.assertEqual(payload["total"], 50)
        get.assert_called_once_with(
            "http://api.local/api/dataset/catalog",
            params={"offset": 24, "limit": 24},
            headers={},
            timeout=5.0,
        )

    def test_dataset_catalog_sends_class_flag_filter(self):
        client = InspectionApiClient("http://api.local")
        response = Mock(status_code=200)
        response.json.return_value = {"total": 21, "items": [], "class_flag": "Delamination"}
        with patch.object(client.session, "get", return_value=response) as get:
            payload = client.dataset_catalog(offset=0, limit=12, class_flag="Delamination")
        self.assertEqual(payload["class_flag"], "Delamination")
        get.assert_called_once_with(
            "http://api.local/api/dataset/catalog",
            params={"offset": 0, "limit": 12, "class_flag": "Delamination"},
            headers={},
            timeout=5.0,
        )

    def test_dataset_image_is_bounded_and_basename_checked(self):
        client = InspectionApiClient("http://api.local", api_key="secret")
        response = Mock(status_code=200)
        response.headers = {"content-type": "image/jpeg"}
        response.iter_content.return_value = [b"jpeg-bytes"]
        with patch.object(client.session, "get", return_value=response) as get:
            payload = client.dataset_image("image_1548.jpg")
        self.assertEqual(payload, b"jpeg-bytes")
        get.assert_called_once_with(
            "http://api.local/api/dataset/images/image_1548.jpg",
            headers={"x-api-key": "secret"},
            timeout=5.0,
            stream=True,
        )
        with patch.object(client.session, "get", return_value=response) as get:
            self.assertEqual(client.dataset_image("image_1548.jpg", "heatmap"), b"jpeg-bytes")
        get.assert_called_once_with(
            "http://api.local/api/dataset/images/image_1548.jpg?view=heatmap",
            headers={"x-api-key": "secret"},
            timeout=5.0,
            stream=True,
        )
        with self.assertRaises(ValueError):
            client.dataset_image("../secret.jpg")
        with self.assertRaises(ValueError):
            client.dataset_image("image_1548.jpg", "prediction")

    def test_libad_sample_image_rejects_path_and_unknown_view(self):
        client = InspectionApiClient("http://api.local")
        with self.assertRaises(ValueError):
            client.libad_sample_image("../secret", "vis_a")
        with self.assertRaises(ValueError):
            client.libad_sample_image("unit1", "fixture")

    def test_operations_control_requires_idempotency_key(self):
        client = InspectionApiClient("http://api.local")
        with self.assertRaises(ValueError):
            client.operations_control(
                action="EMERGENCY_STOP",
                operator_id="OP_1",
                confirmation="CONFIRM EMERGENCY_STOP",
                reason="missing idempotency test",
                idempotency_key="",
            )


if __name__ == "__main__":
    unittest.main()
