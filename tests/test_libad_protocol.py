"""Official tree hashing skips cache dirs and does not stampede by default."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from libad.dataset import dataset_status
from libad.protocol import OFFICIAL_SPLIT_SEEDS, sha256_tree, tree_stat_fingerprint


class TestLibadProtocol(unittest.TestCase):
    def test_tree_hash_skips_preview_cache_dirs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "keep").mkdir()
            (root / "keep" / "a.bin").write_bytes(b"alpha")
            preview = root / "_preview64" / "nested"
            preview.mkdir(parents=True)
            (preview / "cached.bin").write_bytes(b"should-not-hash")
            incoming = root / "_incoming"
            incoming.mkdir()
            (incoming / "tmp.bin").write_bytes(b"tmp")
            hashed = sha256_tree(root)
            fingerprint = tree_stat_fingerprint(root)
            self.assertEqual(fingerprint["n_files"], 1)
            self.assertEqual(fingerprint["total_bytes"], 5)
            only_keep = Path(directory + "-keep")
            only_keep.mkdir()
            (only_keep / "keep").mkdir()
            (only_keep / "keep" / "a.bin").write_bytes(b"alpha")
            self.assertEqual(hashed, sha256_tree(only_keep))

    def test_default_status_trusts_fingerprint_without_live_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "LIBAD"
            splits = root / "splits"
            normal = dataset / "01_scratch" / "normal"
            normal.mkdir(parents=True)
            (normal / "train-1A.tiff").write_bytes(b"img")
            (normal / "train-1B.tiff").write_bytes(b"img")
            (normal / "train-1L.tiff").write_bytes(b"img")
            (normal / "val-1A.tiff").write_bytes(b"img")
            (normal / "val-1B.tiff").write_bytes(b"img")
            (normal / "val-1L.tiff").write_bytes(b"img")
            (normal / "test-1A.tiff").write_bytes(b"img")
            (normal / "test-1B.tiff").write_bytes(b"img")
            (normal / "test-1L.tiff").write_bytes(b"img")
            splits.mkdir()
            split_payload = json.dumps({
                "train": ["train-1"],
                "val": ["val-1"],
                "test": ["test-1"],
            })
            for seed in OFFICIAL_SPLIT_SEEDS:
                (splits / f"{seed}.json").write_text(split_payload, encoding="utf-8")
            dataset_fp = tree_stat_fingerprint(dataset)
            splits_fp = tree_stat_fingerprint(splits)
            manifest = root / "official_artifact_manifest.json"
            manifest.write_text(json.dumps({
                "dataset_tree_sha256": "a" * 64,
                "splits_tree_sha256": "b" * 64,
                "dataset_n_files": dataset_fp["n_files"],
                "dataset_total_bytes": dataset_fp["total_bytes"],
                "splits_n_files": splits_fp["n_files"],
                "splits_total_bytes": splits_fp["total_bytes"],
                "official_split_seeds": list(OFFICIAL_SPLIT_SEEDS),
                "comparable_to_paper": False,
            }), encoding="utf-8")
            cfg = {
                "paths": {
                    "dataset_root": str(dataset),
                    "splits_root": str(splits),
                    "official_artifact_manifest": str(manifest),
                }
            }
            with patch("libad.dataset.sha256_tree") as hashed:
                status = dataset_status(config=cfg)
                hashed.assert_not_called()
            self.assertTrue(status["official_artifact_manifest_verified"])
            self.assertTrue(status["official_protocol_complete"])
            self.assertFalse(status["tree_hash_verified_live"])
            self.assertFalse(status["comparable_to_paper"])
            self.assertEqual(status["dataset_tree_sha256"], "a" * 64)


if __name__ == "__main__":
    unittest.main()
