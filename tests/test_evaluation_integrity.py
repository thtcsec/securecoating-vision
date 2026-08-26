"""Evaluation provenance and leakage safeguards."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_evaluation import assert_no_dataset_overlap


class TestEvaluationIntegrity(unittest.TestCase):
    def test_hash_overlap_is_rejected_even_when_filename_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "evaluation"
            reference = root / "reference" / "images" / "val"
            evaluation.mkdir(parents=True)
            reference.mkdir(parents=True)
            (evaluation / "renamed.png").write_bytes(b"same image bytes")
            (reference / "original.png").write_bytes(b"same image bytes")
            with self.assertRaises(RuntimeError):
                assert_no_dataset_overlap(str(evaluation), str(root / "reference"))


if __name__ == "__main__":
    unittest.main()
