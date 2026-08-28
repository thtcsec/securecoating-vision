"""Release gates for checked-in identity, claims, and generated evidence artifacts."""

from __future__ import annotations

import json
import unittest
import xml.etree.ElementTree as ET
import zipfile
import ast
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from libad.protocol import git_source_provenance  # noqa: E402
from scripts.run_external_coatingvision_demo import _portable_evidence_path  # noqa: E402
from scripts.build_submission import main as build_submission_main  # noqa: E402


def _docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    return "".join(root.itertext())


class TestRepositoryEvidenceArtifacts(unittest.TestCase):
    def test_external_demo_accepts_output_paths_outside_repository(self):
        external_path = Path("C:/securecoating-manual/output.png")
        self.assertEqual(
            _portable_evidence_path(external_path),
            "C:/securecoating-manual/output.png",
        )

    def test_submission_cli_rejects_unknown_arguments_without_building(self):
        with self.assertRaises(SystemExit) as raised:
            build_submission_main(["--not-a-real-option"])
        self.assertEqual(raised.exception.code, 2)

    def test_public_identity_matches_single_source_of_truth(self):
        identity = yaml.safe_load((ROOT / "configs/project_identity.yaml").read_text(encoding="utf-8"))
        sources = [
            (ROOT / "README.md").read_text(encoding="utf-8"),
            (ROOT / "docs/presentation_pitch.md").read_text(encoding="utf-8"),
            _docx_text(ROOT / "Al + Materials Competition Application Form.docx"),
        ]
        for source in sources:
            self.assertIn(identity["brand"], source)
            self.assertIn(identity["registered_title"], source)
            self.assertIn(identity["tagline"], source)
            self.assertNotIn("82 passing tests", source)
            self.assertNotIn("85 passing tests", source)

    def test_libad_artifacts_never_claim_paper_comparability(self):
        paths = [
            ROOT / "reports/libad/libad_benchmark.json",
            *sorted((ROOT / "reports/libad_demo").glob("*.json")),
        ]
        self.assertTrue(paths)
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            serialized = json.dumps(payload, allow_nan=False)
            self.assertNotIn('"comparable_to_paper": true', serialized, path.as_posix())
        benchmark = json.loads(paths[0].read_text(encoding="utf-8"))
        self.assertEqual(benchmark["evidence_class"], "protocol_fixture")
        self.assertFalse(benchmark["comparable_to_paper"])
        self.assertFalse(benchmark["official_protocol_complete"])

    def test_four_demo_cases_and_identity_screen_are_consistent(self):
        manifest = json.loads(
            (ROOT / "reports/libad_demo/demo_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            [item["decision"]["action"] for item in manifest],
            ["PASS", "REJECT", "REJECT", "HOLD"],
        )
        case_four = manifest[3]
        self.assertEqual(case_four["decision"]["contract_holds"], [])
        self.assertEqual(case_four["decision"]["vis_state"], "UNCERTAIN")
        self.assertEqual(case_four["decision"]["xray_state"], "NORMAL")
        self.assertEqual(case_four["calibration_state"], "VERIFIED")
        for item in manifest:
            self.assertEqual(item["certificate"]["roll_id"], item["identity"]["roll_id"])
            self.assertEqual(item["certificate"]["batch_id"], item["identity"]["batch_id"])
            self.assertEqual(item["certificate"]["part_id"], item["identity"]["part_id"])

    def test_coatingvision_hold_artifacts_are_fail_closed_and_bounded(self):
        paths = sorted((ROOT / "reports/coatingvision_gallery").glob("*/coatingvision_model_output.json"))
        paths.extend([
            ROOT / "reports/coatingvision_real_demo/coatingvision_model_output.json",
            ROOT / "reports/external_coatingvision_demo/coatingvision_model_output.json",
        ])
        self.assertGreaterEqual(len(paths), 3)
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            pipeline = payload["seven_stage_pipeline"]
            self.assertEqual(pipeline["raw_detections_count"], len(payload["detections"]), path.as_posix())
            self.assertEqual(pipeline["raw_detections_count"], len(pipeline["raw_detections"]), path.as_posix())
            if pipeline["overall_verdict"] == "HOLD":
                self.assertFalse(pipeline["standards_compliant"], path.as_posix())
                self.assertEqual(pipeline["plc_gate_action"], "HOLD", path.as_posix())
                self.assertNotEqual(
                    pipeline["root_cause_report"]["severity_level"], "NOMINAL", path.as_posix()
                )
            for detection in payload["detections"]:
                self.assertNotIn("mask", detection)
                self.assertIsInstance(detection.get("mask_foreground_pixels"), int)
            json.dumps(payload, allow_nan=False)

    def test_submission_whitelist_carries_authoritative_test_manifest(self):
        source = (ROOT / "scripts/build_submission.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        whitelist = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "WHITELIST"
                for target in node.targets
            ):
                whitelist = node.value
                break
        self.assertIsInstance(whitelist, ast.List)
        string_items = {
            item.value for item in whitelist.elts
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        }
        self.assertIn("reports/test_manifest.json", string_items)
        self.assertIn("configs/project_identity.yaml", string_items)
        self.assertIn("reports/libad/libad_benchmark.json", string_items)
        self.assertIn("reports/libad_demo/demo_manifest.json", string_items)

    def test_test_manifest_discloses_dirty_source_provenance(self):
        manifest = json.loads(
            (ROOT / "reports/test_manifest.json").read_text(encoding="utf-8")
        )
        current = git_source_provenance(ROOT)
        self.assertEqual(manifest["working_tree_dirty"], current["working_tree_dirty"])
        self.assertEqual(manifest["source_diff_sha256"], current["source_diff_sha256"])


if __name__ == "__main__":
    unittest.main()
