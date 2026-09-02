"""Optional evenrose/LIBAD clone is status-only and never paper-comparable."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from libad.official_code import official_code_status
import scripts.fetch_libad_official_code as fetch_libad_official_code


class TestLibadOfficialCode(unittest.TestCase):
    def test_missing_checkout_is_not_present(self):
        with tempfile.TemporaryDirectory() as directory:
            status = official_code_status(repo_root=Path(directory))
        self.assertFalse(status["present"])
        self.assertFalse(status["runnable_as_paper_baseline"])
        self.assertIn("evenrose/LIBAD", status["note"])

    def test_fetch_status_does_not_clone(self):
        with patch.object(fetch_libad_official_code, "fetch_official_code") as fetch:
            code = fetch_libad_official_code.main(["--status"])
        self.assertEqual(code, 0)
        fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
