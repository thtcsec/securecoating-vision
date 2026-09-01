"""Official multimodal index lists mounted triples only; never protocol fixtures."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from libad import dataset as libad_dataset


class TestLibadOfficialIndex(unittest.TestCase):
    def tearDown(self):
        libad_dataset.reset_official_sample_index_cache()

    def test_unmounted_index_is_empty_not_a_fixture(self):
        libad_dataset.reset_official_sample_index_cache()
        payload = libad_dataset.list_official_samples(offset=0, limit=12)
        if not payload["official_dataset_present"]:
            self.assertEqual(payload["total"], 0)
            self.assertEqual(payload["items"], [])
            self.assertEqual(payload["source"], "not_mounted")
        self.assertFalse(payload["comparable_to_paper"])
        self.assertFalse(payload["image_payloads_included"])

    def test_index_and_view_use_official_files_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "LIBAD" / "1_crack" / "anomaly"
            sample_dir.mkdir(parents=True)
            gray = np.full((32, 32), 120, dtype=np.uint8)
            for suffix in ("A", "B", "L"):
                ok = cv2.imwrite(str(sample_dir / f"unit1{suffix}.tiff"), gray)
                self.assertTrue(ok)
            config = {
                "paths": {
                    "dataset_root": "LIBAD",
                    "splits_root": "splits",
                    "official_artifact_manifest": "missing.json",
                }
            }
            libad_dataset.reset_official_sample_index_cache()
            with patch.object(libad_dataset, "PROJECT_ROOT", root), patch(
                "libad.protocol.PROJECT_ROOT", root
            ):
                payload = libad_dataset.list_official_samples(
                    offset=0, limit=12, config=config
                )
                self.assertTrue(payload["official_dataset_present"])
                self.assertEqual(payload["total"], 1)
                self.assertEqual(payload["items"][0]["sample_id"], "unit1")
                self.assertEqual(payload["items"][0]["defect_group"], "crack")
                self.assertEqual(payload["items"][0]["label"], "anomaly")
                frame = libad_dataset.load_official_sample_view(
                    "unit1", "xray_l", config=config
                )
                self.assertEqual(tuple(frame.shape[:2]), (32, 32))
                self.assertEqual(frame.ndim, 2)
                with self.assertRaises(FileNotFoundError):
                    libad_dataset.load_official_sample_view(
                        "../secret", "vis_a", config=config
                    )


if __name__ == "__main__":
    unittest.main()
