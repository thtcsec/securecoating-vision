"""CI JUnit gate: artifact-gated skips are allowed; unexplained skips are not."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_ci_junit import failure_count, unexpected_skip_names  # noqa: E402


SAMPLE = """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" errors="0" failures="{failures}" skipped="{skipped}" tests="3">
    <testcase classname="tests.test_api.TestAPI" name="test_health" time="0.01"/>
    <testcase classname="tests.test_api.TestAPI" name="test_inspect_sample_name_detects_defect" time="0.01">
      <skipped type="pytest.skip" message="ONNX model or test image missing">ONNX model or test image missing</skipped>
    </testcase>
    {extra}
  </testsuite>
</testsuites>
"""


class TestCiJunit(unittest.TestCase):
    def _write(self, failures: int, extra: str = "") -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "junit.xml"
        skipped = 1 + (1 if "skipped" in extra else 0)
        path.write_text(
            SAMPLE.format(failures=failures, skipped=skipped, extra=extra),
            encoding="utf-8",
        )
        return path

    def test_artifact_gated_skip_is_allowed(self):
        path = self._write(failures=0)
        self.assertEqual(failure_count(path), 0)
        self.assertEqual(unexpected_skip_names(path), [])

    def test_unexplained_skip_is_rejected(self):
        extra = """
    <testcase classname="tests.test_api.TestAPI" name="test_hidden" time="0.01">
      <skipped type="pytest.skip" message="later"/>
    </testcase>
"""
        path = self._write(failures=0, extra=extra)
        names = unexpected_skip_names(path)
        self.assertEqual(len(names), 1)
        self.assertIn("test_hidden", names[0])

    def test_failures_are_counted(self):
        extra = """
    <testcase classname="tests.test_api.TestAPI" name="test_broken" time="0.01">
      <failure message="boom">boom</failure>
    </testcase>
"""
        path = self._write(failures=1, extra=extra)
        self.assertEqual(failure_count(path), 1)

    def test_ci_workflow_allows_artifact_gated_skips(self):
        workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertIn("scripts/check_ci_junit.py", workflow)
        self.assertNotIn("skipped == 0", workflow)
        self.assertIn("SECURECOATING_ENV: test", workflow)
        self.assertIn("OPENBLAS_NUM_THREADS: \"1\"", workflow)


if __name__ == "__main__":
    unittest.main()
