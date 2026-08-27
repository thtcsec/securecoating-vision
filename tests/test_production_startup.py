"""Production startup must fail before loading engines when secrets are absent."""

import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestProductionStartup(unittest.TestCase):
    def test_missing_api_key_is_fatal(self):
        environment = os.environ.copy()
        environment.update({
            "SECURECOATING_ENV": "production",
            "SECURECOATING_FACTORY_SECRET": "test-only-factory-secret",
        })
        environment.pop("SECURECOATING_API_KEY", None)
        result = subprocess.run(
            [sys.executable, "-c", "import src.api.main"],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SECURECOATING_API_KEY must be configured", result.stderr)


if __name__ == "__main__":
    unittest.main()
