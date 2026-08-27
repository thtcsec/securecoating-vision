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
        response.raise_for_status.assert_called_once_with()

    def test_http_errors_are_not_hidden(self):
        client = InspectionApiClient("http://api.local")
        response = Mock()
        response.raise_for_status.side_effect = RuntimeError("503")
        with patch.object(client.session, "get", return_value=response):
            with self.assertRaises(RuntimeError):
                client.batch_stats("BATCH")

    def test_non_object_json_is_rejected(self):
        client = InspectionApiClient("http://api.local")
        response = Mock()
        response.json.return_value = []
        with patch.object(client.session, "get", return_value=response):
            with self.assertRaises(ValueError):
                client.get("/health")


if __name__ == "__main__":
    unittest.main()
