"""Official 10-seed protocol: no random split, four required experiments."""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from libad.evaluate import evaluate_official_splits
from libad.dataset import dataset_status, load_official_split
from libad.protocol import OFFICIAL_SPLIT_SEEDS, sha256_tree


class TestLibadEvaluate(unittest.TestCase):
    @staticmethod
    def _write_official_sample(folder: Path, sample_id: str) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        image = np.full((8, 8), 127, dtype=np.uint8)
        for suffix in ("A", "B", "L"):
            if not cv2.imwrite(str(folder / f"{sample_id}{suffix}.tiff"), image):
                raise RuntimeError("Failed to create test TIFF")

    def test_refuses_a_non_official_seed(self):
        from libad.dataset import load_split

        with self.assertRaises(ValueError):
            load_split(123)

    def test_ten_official_seeds_and_four_experiments(self):
        report = evaluate_official_splits(
            seeds=OFFICIAL_SPLIT_SEEDS[:2],
            experiments=("unimodal_vis", "unimodal_xray_l", "multimodal", "securecoating_gate"),
            allow_fixture=True,
        )
        self.assertEqual(report["official_split_seeds"], list(OFFICIAL_SPLIT_SEEDS[:2]))
        self.assertFalse(report["comparable_to_paper"])
        self.assertEqual(report["evidence_class"], "protocol_fixture")
        self.assertFalse(report["official_protocol_complete"])
        self.assertIsNone(report["hashes"]["official_dataset_tree_sha256"])
        self.assertTrue(report["hashes"]["protocol_fixture_definition"])
        for name in ("unimodal_vis", "unimodal_xray_l", "multimodal", "securecoating_gate"):
            self.assertIn(name, report["experiments"])
            self.assertIn("academic", report["experiments"][name])
        industrial = report["experiments"]["securecoating_gate"]["industrial"]
        for metric in (
            "automatic_decision_coverage",
            "hold_rate",
            "escape_rate",
            "selective_risk",
        ):
            self.assertIn(metric, industrial)
        for split_record in report["split_records"]["securecoating_gate"]:
            self.assertGreater(
                split_record["industrial"]["n_pass"], 0,
                f"Validation calibration rejected every normal for seed {split_record['seed']}",
            )
        self.assertIn("does not claim DA-Core", report["local_contribution"])
        self.assertTrue(report["predictions"])

    def test_official_inputs_still_do_not_claim_paper_comparability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "LIBAD"
            splits = root / "splits"
            normal = dataset / "01_scratch" / "normal"
            for sample_id in ("train-1", "val-1", "test-1"):
                self._write_official_sample(normal, sample_id)
            splits.mkdir()
            (splits / "347.json").write_text(
                json.dumps({
                    "train": ["train-1"],
                    "val": ["val-1"],
                    "test": ["test-1"],
                }),
                encoding="utf-8",
            )
            cfg = {"paths": {"dataset_root": str(dataset), "splits_root": str(splits)}}
            split = load_official_split(347, config=cfg)
            self.assertIsNotNone(split)
            self.assertEqual(split.source, "libad_structured_inputs")
            self.assertFalse(split.comparable_to_paper)

            status = dataset_status(config=cfg)
            self.assertFalse(status["official_protocol_complete"])
            self.assertFalse(status["input_structure_complete"])
            self.assertEqual(status["valid_official_split_seeds"], [347])
            self.assertFalse(status["comparable_to_paper"])

    def test_overlapping_official_split_ids_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "LIBAD"
            splits = root / "splits"
            normal = dataset / "01_scratch" / "normal"
            for sample_id in ("shared", "val-1"):
                self._write_official_sample(normal, sample_id)
            splits.mkdir()
            (splits / "347.json").write_text(
                json.dumps({
                    "train": ["shared"],
                    "val": ["val-1"],
                    "test": ["shared"],
                }),
                encoding="utf-8",
            )
            cfg = {"paths": {"dataset_root": str(dataset), "splits_root": str(splits)}}
            self.assertIsNone(load_official_split(347, config=cfg))

    def test_all_ten_splits_require_matching_tree_hash_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "LIBAD"
            splits = root / "splits"
            normal = dataset / "01_scratch" / "normal"
            for sample_id in ("train-1", "val-1", "test-1"):
                self._write_official_sample(normal, sample_id)
            splits.mkdir()
            split_payload = json.dumps({
                "train": ["train-1"],
                "val": ["val-1"],
                "test": ["test-1"],
            })
            for seed in OFFICIAL_SPLIT_SEEDS:
                (splits / f"{seed}.json").write_text(split_payload, encoding="utf-8")
            manifest = root / "official_artifact_manifest.json"
            manifest.write_text(json.dumps({
                "dataset_tree_sha256": sha256_tree(dataset),
                "splits_tree_sha256": sha256_tree(splits),
                "official_split_seeds": list(OFFICIAL_SPLIT_SEEDS),
            }), encoding="utf-8")
            cfg = {"paths": {
                "dataset_root": str(dataset),
                "splits_root": str(splits),
                "official_artifact_manifest": str(manifest),
            }}
            status = dataset_status(config=cfg)
            self.assertTrue(status["input_structure_complete"])
            self.assertTrue(status["official_artifact_manifest_verified"])
            self.assertTrue(status["official_protocol_complete"])

            manifest.write_text(json.dumps({
                "dataset_tree_sha256": "0" * 64,
                "splits_tree_sha256": sha256_tree(splits),
                "official_split_seeds": list(OFFICIAL_SPLIT_SEEDS),
            }), encoding="utf-8")
            status = dataset_status(config=cfg)
            self.assertFalse(status["official_artifact_manifest_verified"])
            self.assertFalse(status["official_protocol_complete"])
