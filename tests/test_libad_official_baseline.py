"""Official DINOv3/DA-Core harness stays honest about paper comparability."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from libad.official_baseline import (
    PAPER_BACKBONE_VARIANT,
    aggregate_experiment_log,
    build_report,
    build_run_command,
    dinov3_access_status,
    paper_config_match,
    run_card,
)


class TestLibadOfficialBaseline(unittest.TestCase):
    def test_paper_config_match_requires_base_variant(self):
        self.assertTrue(
            paper_config_match(
                {
                    "backbone_family": "convnext",
                    "dino_version": "v3",
                    "backbone_variant": PAPER_BACKBONE_VARIANT,
                    "coreset_selection_method": "density_fps",
                    "f_coreset": 0.05,
                    "coreset_density_weight": 0.7,
                }
            )
        )
        self.assertFalse(
            paper_config_match(
                {
                    "backbone_family": "convnext",
                    "dino_version": "v3",
                    "backbone_variant": "tiny",
                    "coreset_selection_method": "density_fps",
                    "f_coreset": 0.05,
                    "coreset_density_weight": 0.7,
                }
            )
        )

    def test_build_run_command_points_at_local_libad_paths(self):
        cmd = build_run_command(
            seed=347,
            modality="vis_xray_l",
            backbone_variant="tiny",
            batch_size=1,
            precision="fp16",
            root_path=Path("data/libad/LIBAD"),
            split_dir=Path("data/libad/splits"),
            results_csv=Path("results/log.csv"),
            results_md=Path("results/log.md"),
        )
        normalized = " ".join(part.replace("\\", "/") for part in cmd)
        self.assertIn("run.py", normalized)
        self.assertIn("density_fps", normalized)
        self.assertIn("vis_xray_l", normalized)
        self.assertIn("data/libad/LIBAD", normalized)
        self.assertIn("data/libad/splits", normalized)
        self.assertNotIn("--root_path dataset/LIBAD", normalized)
        self.assertIn("--coreset_device cpu", normalized)
        self.assertIn("--distance_chunk_size 8192", normalized)
        self.assertNotIn("--save_bank", normalized)
        self.assertNotIn("--save_raw_scores", normalized)

    def test_laptop_defaults_are_conservative(self):
        from libad.official_baseline import laptop_runner_defaults

        defaults = laptop_runner_defaults()
        self.assertEqual(defaults["modalities"], ["vis_xray_l"])
        self.assertEqual(defaults["batch_size"], 1)
        self.assertEqual(defaults["coreset_device"], "cpu")
        self.assertFalse(defaults["save_bank"])
        self.assertGreaterEqual(defaults["pause_seconds"], 1)

    def test_aggregate_and_report_mark_tiny_as_not_paper_comparable(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "log.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "seed",
                        "modality",
                        "backbone_variant",
                        "coreset_selection_method",
                        "image_auroc",
                        "image_aupr",
                        "image_best_f1",
                        "image_fpr95",
                    ],
                )
                writer.writeheader()
                for seed in (347, 725):
                    writer.writerow(
                        {
                            "seed": seed,
                            "modality": "vis_xray_l",
                            "backbone_variant": "tiny",
                            "coreset_selection_method": "density_fps",
                            "image_auroc": "0.80",
                            "image_aupr": "0.90",
                            "image_best_f1": "0.85",
                            "image_fpr95": "0.55",
                        }
                    )
            agg = aggregate_experiment_log(
                csv_path,
                seeds=(347, 725),
                modalities=("vis_xray_l",),
                backbone_variant="tiny",
            )
            report = build_report(
                config={
                    "backbone_family": "convnext",
                    "dino_version": "v3",
                    "backbone_variant": "tiny",
                    "coreset_selection_method": "density_fps",
                    "f_coreset": 0.05,
                    "coreset_density_weight": 0.7,
                    "modalities": ["vis_xray_l"],
                    "seeds": [347, 725],
                },
                aggregation=agg,
                dataset_meta={"official_protocol_complete": True},
                code_meta={"present": True, "commit": "abc"},
                run_records=[],
            )
            self.assertFalse(report["comparable_to_paper"])
            self.assertIn("tiny", " ".join(report["paper_comparability_blockers"]))
            self.assertEqual(report["experiments"]["multimodal"]["n_splits"], 2)
            self.assertAlmostEqual(report["experiments"]["multimodal"]["academic"]["auroc"]["mean"], 0.80)

    def test_run_card_and_access_status_are_serializable(self):
        # Offline-safe: probe=False must not touch the network.
        card = run_card(backbone_variant="tiny", probe=False)
        access = dinov3_access_status(probe=False)
        json.dumps(card)
        json.dumps(access)
        self.assertIn("facebook/dinov3-convnext-tiny-pretrain-lvd1689m", access["gated_models"])
        self.assertIn("hf_auth_present", access)
        self.assertIn("model_access_ok", access)
        self.assertFalse(access["model_access_ok"])  # no probe => not verified
        self.assertEqual(access["probes"], [])
        self.assertIn("paper_comparability_blockers", card)
        blockers = " ".join(card["paper_comparability_blockers"])
        if access["hf_auth_present"]:
            self.assertIn("not verified", blockers)
            self.assertNotIn("Hugging Face auth missing", blockers)
        else:
            self.assertIn("Hugging Face auth missing", blockers)

    def test_build_run_command_supports_dinov2_interim(self):
        cmd = build_run_command(
            seed=347,
            modality="vis_xray_l",
            backbone_variant="small",
            batch_size=1,
            precision="fp16",
            root_path=Path("data/libad/LIBAD"),
            split_dir=Path("data/libad/splits"),
            results_csv=Path("results/log.csv"),
            results_md=Path("results/log.md"),
            backbone_family="vit",
            dino_version="v2",
        )
        normalized = " ".join(cmd)
        self.assertIn("--dino_version", normalized)
        self.assertIn("v2", normalized)
        self.assertIn("--backbone_family", normalized)
        self.assertIn("vit", normalized)
        self.assertIn("small", normalized)


if __name__ == "__main__":
    unittest.main()
