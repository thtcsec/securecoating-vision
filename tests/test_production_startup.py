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
            "SECURECOATING_FACTORY_SECRET": "f" * 32,
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

    def test_short_production_api_key_is_fatal(self):
        environment = os.environ.copy()
        environment.update({
            "SECURECOATING_ENV": "production",
            "SECURECOATING_FACTORY_SECRET": "f" * 32,
            "SECURECOATING_API_KEY": "short",
        })
        result = subprocess.run(
            [sys.executable, "-c", "import src.api.main"],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SECURECOATING_API_KEY must contain at least 32 bytes", result.stderr)

    def test_short_production_factory_secret_is_fatal(self):
        environment = os.environ.copy()
        environment.update({
            "SECURECOATING_ENV": "production",
            "SECURECOATING_FACTORY_SECRET": "short",
            "SECURECOATING_API_KEY": "a" * 32,
        })
        result = subprocess.run(
            [sys.executable, "-c", "import src.api.main"],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "SECURECOATING_FACTORY_SECRET must contain at least 32 bytes",
            result.stderr,
        )

    def test_wildcard_production_cors_is_fatal(self):
        environment = os.environ.copy()
        environment.update({
            "SECURECOATING_ENV": "production",
            "SECURECOATING_FACTORY_SECRET": "f" * 32,
            "SECURECOATING_API_KEY": "a" * 32,
            "SECURECOATING_ALLOWED_ORIGINS": "*",
        })
        result = subprocess.run(
            [sys.executable, "-c", "import src.api.main"],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Wildcard CORS origins are forbidden", result.stderr)

    def test_remote_http_production_cors_is_fatal(self):
        environment = os.environ.copy()
        environment.update({
            "SECURECOATING_ENV": "production",
            "SECURECOATING_FACTORY_SECRET": "f" * 32,
            "SECURECOATING_API_KEY": "a" * 32,
            "SECURECOATING_ALLOWED_ORIGINS": "http://operator.example.com",
        })
        result = subprocess.run(
            [sys.executable, "-c", "import src.api.main"],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must use HTTPS unless loopback", result.stderr)


if __name__ == "__main__":
    unittest.main()
