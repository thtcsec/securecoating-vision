"""Dashboard render smoke test in an explicitly isolated sandbox."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


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


if __name__ == "__main__":
    unittest.main()
