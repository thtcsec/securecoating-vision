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
    OFFICIAL_CODE_BACKBONE_VARIANT,
    PAPER_BACKBONE_VARIANT,
    aggregate_experiment_log,
    append_harness_ledger_row,
    build_report,
    build_run_command,
    dinov3_access_status,
    official_code_core_match,
    paper_config_match,
    paper_spec_match,
    run_card,
)


LEDGER_FIELDS = [
    "schema_version",
    "run_id",
    "config_sha256",
    "started_at_utc",
    "finished_at_utc",
    "seed",
    "modality",
    "dino_version",
    "backbone_family",
    "backbone_variant",
    "extractor_precision",
    "coreset_selection_method",
    "coreset_density_weight",
    "f_coreset",
    "returncode",
    "image_auroc",
    "image_aupr",
    "image_best_f1",
    "image_fpr95",
]


def _write_ledger(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class TestLibadOfficialBaseline(unittest.TestCase):
    def test_paper_spec_is_vit_s16_not_convnext_base(self):
        self.assertTrue(
            paper_spec_match(
                {
                    "backbone_family": "vit",
                    "dino_version": "v3",
                    "backbone_variant": "small",
                    "coreset_selection_method": "density_fps",
                    "f_coreset": 0.05,
                    "coreset_density_weight": 0.7,
                    "image_score_method": "max",
                }
            )
        )
        self.assertFalse(
            paper_spec_match(
                {
                    "backbone_family": "vit",
                    "dino_version": "v3",
                    "backbone_variant": "small",
                    "coreset_selection_method": "density_fps",
                    "f_coreset": 0.05,
                    "coreset_density_weight": 0.7,
                    "image_score_method": "topk",
                }
            )
        )
        self.assertFalse(
            paper_spec_match(
                {
                    "backbone_family": "convnext",
                    "dino_version": "v3",
                    "backbone_variant": "base",
                    "coreset_selection_method": "density_fps",
                    "f_coreset": 0.05,
                    "coreset_density_weight": 0.7,
                    "image_score_method": "max",
                }
            )
        )
        self.assertTrue(
            official_code_core_match(
                {
                    "backbone_family": "convnext",
                    "dino_version": "v3",
                    "backbone_variant": OFFICIAL_CODE_BACKBONE_VARIANT,
                    "coreset_selection_method": "density_fps",
                    "f_coreset": 0.05,
                    "coreset_density_weight": 0.7,
                }
            )
        )
        # Legacy alias points at official-code core, not textual paper.
        self.assertEqual(PAPER_BACKBONE_VARIANT, OFFICIAL_CODE_BACKBONE_VARIANT)
        self.assertTrue(
            paper_config_match(
                {
                    "backbone_family": "convnext",
                    "dino_version": "v3",
                    "backbone_variant": "base",
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
            run_id="abc123",
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
        self.assertIn("run_id=abc123", normalized)

    def test_laptop_defaults_are_conservative(self):
        from libad.official_baseline import laptop_runner_defaults

        defaults = laptop_runner_defaults()
        self.assertEqual(defaults["modalities"], ["vis_xray_l"])
        self.assertEqual(defaults["batch_size"], 1)
        self.assertEqual(defaults["coreset_device"], "cpu")
        self.assertFalse(defaults["save_bank"])
        self.assertGreaterEqual(defaults["pause_seconds"], 1)

    def test_stale_csv_cannot_inflate_without_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "stale.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "seed",
                        "modality",
                        "dino_version",
                        "backbone_family",
                        "backbone_variant",
                        "coreset_selection_method",
                        "f_coreset",
                        "coreset_density_weight",
                        "image_auroc",
                        "image_aupr",
                        "image_best_f1",
                        "image_fpr95",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "seed": 347,
                        "modality": "vis_xray_l",
                        "dino_version": "v2",
                        "backbone_family": "vit",
                        "backbone_variant": "base",
                        "coreset_selection_method": "density_fps",
                        "f_coreset": "0.05",
                        "coreset_density_weight": "0.7",
                        "image_auroc": "0.99",
                        "image_aupr": "0.99",
                        "image_best_f1": "0.99",
                        "image_fpr95": "0.01",
                    }
                )
            # No run_id / accepted_cells → refuse rows.
            empty = aggregate_experiment_log(
                csv_path,
                seeds=(347, 725),
                modalities=("vis_xray_l",),
                backbone_variant="base",
                dino_version="v3",
                backbone_family="convnext",
            )
            self.assertEqual(empty["n_matching_rows"], 0)
            self.assertEqual(empty["experiments"]["multimodal"]["n_splits"], 0)

    def test_failed_process_cannot_be_paper_comparable(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.csv"
            run_id = "run_failed_guard"
            _write_ledger(
                ledger,
                [
                    {
                        "schema_version": "securecoating-harness-ledger/v1",
                        "run_id": run_id,
                        "config_sha256": "x",
                        "started_at_utc": "2026-09-06T00:00:00Z",
                        "finished_at_utc": "2026-09-06T00:01:00Z",
                        "seed": 347,
                        "modality": "vis_xray_l",
                        "dino_version": "v3",
                        "backbone_family": "vit",
                        "backbone_variant": "small",
                        "extractor_precision": "fp16",
                        "coreset_selection_method": "density_fps",
                        "coreset_density_weight": "0.7",
                        "f_coreset": "0.05",
                        "returncode": 0,
                        "image_auroc": "0.90",
                        "image_aupr": "0.95",
                        "image_best_f1": "0.91",
                        "image_fpr95": "0.50",
                    }
                ],
            )
            agg = aggregate_experiment_log(
                ledger,
                seeds=(347,),
                modalities=("vis_xray_l",),
                backbone_variant="small",
                dino_version="v3",
                backbone_family="vit",
                run_id=run_id,
                accepted_cells=[(347, "vis_xray_l")],
            )
            report = build_report(
                config={
                    "backbone_family": "vit",
                    "dino_version": "v3",
                    "backbone_variant": "small",
                    "coreset_selection_method": "density_fps",
                    "f_coreset": 0.05,
                    "coreset_density_weight": 0.7,
                    "modalities": ["vis_xray_l"],
                    "seeds": [347],
                },
                aggregation=agg,
                dataset_meta={"official_protocol_complete": True},
                code_meta={"present": True, "commit": "abc"},
                run_records=[{"seed": 347, "modality": "vis_xray_l", "returncode": 1}],
                run_id=run_id,
            )
            self.assertFalse(report["comparable_to_paper"])
            self.assertEqual(report["status"], "FAILED")
            self.assertEqual(report["claim_class"], "INVALID")
            self.assertTrue(report["dataset_protocol_complete"])
            self.assertTrue(report["experiment_coverage_complete"])

    def test_empty_modalities_not_ok(self):
        report = build_report(
            config={
                "backbone_family": "vit",
                "dino_version": "v3",
                "backbone_variant": "small",
                "coreset_selection_method": "density_fps",
                "f_coreset": 0.05,
                "coreset_density_weight": 0.7,
                "modalities": [],
                "seeds": [],
            },
            aggregation={"experiments": {}, "run_id": "x"},
            dataset_meta={"official_protocol_complete": True},
            code_meta={"present": True},
            run_records=[],
            run_id="x",
        )
        self.assertFalse(report["comparable_to_paper"])
        self.assertNotEqual(report["status"], "OK")
        self.assertFalse(report["experiment_coverage_complete"])

    def test_aggregate_and_report_mark_tiny_as_not_paper_comparable(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.csv"
            run_id = "tiny_run"
            rows = []
            for seed in (347, 725):
                rows.append(
                    {
                        "schema_version": "securecoating-harness-ledger/v1",
                        "run_id": run_id,
                        "config_sha256": "x",
                        "started_at_utc": "2026-09-06T00:00:00Z",
                        "finished_at_utc": "2026-09-06T00:01:00Z",
                        "seed": seed,
                        "modality": "vis_xray_l",
                        "dino_version": "v3",
                        "backbone_family": "convnext",
                        "backbone_variant": "tiny",
                        "extractor_precision": "fp16",
                        "coreset_selection_method": "density_fps",
                        "coreset_density_weight": "0.7",
                        "f_coreset": "0.05",
                        "returncode": 0,
                        "image_auroc": "0.80",
                        "image_aupr": "0.90",
                        "image_best_f1": "0.85",
                        "image_fpr95": "0.55",
                    }
                )
            _write_ledger(ledger, rows)
            accepted = [(347, "vis_xray_l"), (725, "vis_xray_l")]
            agg = aggregate_experiment_log(
                ledger,
                seeds=(347, 725),
                modalities=("vis_xray_l",),
                backbone_variant="tiny",
                dino_version="v3",
                backbone_family="convnext",
                run_id=run_id,
                accepted_cells=accepted,
            )
            records = [
                {"seed": 347, "modality": "vis_xray_l", "returncode": 0},
                {"seed": 725, "modality": "vis_xray_l", "returncode": 0},
            ]
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
                run_records=records,
                run_id=run_id,
            )
            self.assertFalse(report["comparable_to_paper"])
            self.assertEqual(report["claim_class"], "ADAPTED_REPRODUCTION")
            self.assertEqual(report["experiments"]["multimodal"]["n_splits"], 2)
            self.assertAlmostEqual(report["experiments"]["multimodal"]["academic"]["auroc"]["mean"], 0.80)
            self.assertIn("DINO v3", report["experiments"]["multimodal"]["purpose"])
            self.assertEqual(report["status"], "OK")

    def test_aggregate_purpose_names_dinov2_honestly(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.csv"
            run_id = "dinov2"
            _write_ledger(
                ledger,
                [
                    {
                        "schema_version": "securecoating-harness-ledger/v1",
                        "run_id": run_id,
                        "config_sha256": "x",
                        "started_at_utc": "2026-09-06T00:00:00Z",
                        "finished_at_utc": "2026-09-06T00:01:00Z",
                        "seed": 347,
                        "modality": "vis_xray_l",
                        "dino_version": "v2",
                        "backbone_family": "vit",
                        "backbone_variant": "small",
                        "extractor_precision": "fp16",
                        "coreset_selection_method": "density_fps",
                        "coreset_density_weight": "0.7",
                        "f_coreset": "0.05",
                        "returncode": 0,
                        "image_auroc": "0.85",
                        "image_aupr": "0.95",
                        "image_best_f1": "0.88",
                        "image_fpr95": "0.71",
                    }
                ],
            )
            agg = aggregate_experiment_log(
                ledger,
                seeds=(347,),
                modalities=("vis_xray_l",),
                backbone_variant="small",
                dino_version="v2",
                backbone_family="vit",
                run_id=run_id,
                accepted_cells=[(347, "vis_xray_l")],
            )
            purpose = agg["experiments"]["multimodal"]["purpose"]
            self.assertIn("DINO v2", purpose)
            self.assertNotIn("DINOv3", purpose)
            self.assertNotIn("DINO v3", purpose)

    def test_run_card_and_access_status_are_serializable(self):
        card = run_card(backbone_variant="tiny", probe=False)
        access = dinov3_access_status(probe=False)
        json.dumps(card)
        json.dumps(access)
        self.assertIn("facebook/dinov3-convnext-tiny-pretrain-lvd1689m", access["gated_models"])
        self.assertIn("paper_spec_match", card)
        self.assertFalse(card["paper_spec_match"])
        self.assertEqual(card["paper_code_consistency"], "mismatch")
        self.assertFalse(access["model_access_ok"])
        self.assertEqual(access["probes"], [])

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

    def test_append_harness_ledger_row_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.csv"
            append_harness_ledger_row(
                path,
                {
                    "schema_version": "securecoating-harness-ledger/v1",
                    "run_id": "r1",
                    "seed": 347,
                    "modality": "vis_xray_l",
                    "dino_version": "v3",
                    "backbone_family": "vit",
                    "backbone_variant": "small",
                    "coreset_selection_method": "density_fps",
                    "coreset_density_weight": 0.7,
                    "f_coreset": 0.05,
                    "returncode": 0,
                    "image_auroc": 0.1,
                },
            )
            self.assertTrue(path.is_file())
            text = path.read_text(encoding="utf-8")
            self.assertIn("run_id", text)
            self.assertIn("r1", text)


if __name__ == "__main__":
    unittest.main()
