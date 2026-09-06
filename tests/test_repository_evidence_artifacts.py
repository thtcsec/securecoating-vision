"""Release gates for checked-in identity, claims, and generated evidence artifacts."""

from __future__ import annotations

import json
import hashlib
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


def _pptx_text(path: Path) -> str:
    ns = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    chunks = []
    with zipfile.ZipFile(path) as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if name.startswith("ppt/slides/slide") and name.endswith(".xml")
        )
        for name in names:
            root = ET.fromstring(archive.read(name))
            chunks.extend(node.text or "" for node in root.iter(f"{ns}t"))
    return " ".join(chunks)


class TestRepositoryEvidenceArtifacts(unittest.TestCase):
    def test_real_demo_manifest_hashes_every_checked_in_sample(self):
        demo_root = ROOT / "data/demo_real"
        manifest = json.loads((demo_root / "manifest.json").read_text(encoding="utf-8"))
        records = {record["filename"]: record for record in manifest["samples"]}
        images = sorted((demo_root / "images").glob("*.jpg"))
        self.assertEqual(set(records), {path.name for path in images})
        self.assertGreaterEqual(len(images), 80)
        self.assertEqual(manifest.get("split"), "test")
        self.assertEqual(manifest.get("license"), "CC BY 4.0")
        for path in images:
            self.assertEqual(records[path.name]["source_type"], "REAL_OPTICAL")
            self.assertEqual(records[path.name]["license"], "CC BY 4.0")
            self.assertEqual(records[path.name].get("split"), "test")
            self.assertEqual(
                records[path.name]["sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    def test_container_build_uses_cpu_lock_and_excludes_runtime_mounts(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        docker_lock = (ROOT / "requirements-docker-lock.txt").read_text(encoding="utf-8")
        dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn("COPY requirements-docker-lock.txt", dockerfile)
        self.assertIn("--mount=from=builder,source=/wheels,target=/wheels", dockerfile)
        self.assertNotIn("COPY --from=builder /wheels /wheels", dockerfile)
        self.assertIn("libgl1 libglib2.0-0t64", dockerfile)
        self.assertIn("torch==2.13.0+cpu", docker_lock)
        self.assertIn("torchvision==0.28.0+cpu", docker_lock)
        self.assertNotIn("cuda-toolkit", docker_lock.lower())
        self.assertIn("data/**", dockerignore)
        self.assertIn("outputs/**", dockerignore)
        self.assertIn("./data:/app/data", compose)
        dashboard_service = compose.split("  dashboard:", 1)[1].split("\nnetworks:", 1)[0]
        self.assertNotIn("\n    volumes:", dashboard_service)
        self.assertNotIn("SECURECOATING_FACTORY_SECRET", dashboard_service)
        self.assertNotIn("SECURECOATING_OPERATOR_CREDENTIALS", dashboard_service)
        self.assertNotIn("MODEL_CONFIG", dashboard_service)
        self.assertNotIn("CALIBRATION_CONFIG", dashboard_service)
        streamlit_config = (ROOT / ".streamlit/config.toml").read_text(encoding="utf-8")
        self.assertIn('toolbarMode = "minimal"', streamlit_config)
        self.assertIn('primaryColor = "#00E5FF"', streamlit_config)
        self.assertIn("./outputs:/app/outputs", compose)
        self.assertIn("SECURECOATING_HARDWARE_PROFILE:", compose)
        gpu_compose = (ROOT / "docker-compose.gpu.yml").read_text(encoding="utf-8")
        self.assertIn("Dockerfile.gpu", gpu_compose)
        self.assertNotIn("gpus:", gpu_compose)
        self.assertIn("driver: nvidia", gpu_compose)
        self.assertIn("capabilities: [gpu]", gpu_compose)
        self.assertIn('SECURECOATING_DASHBOARD_SANDBOX: "false"', compose)
        self.assertNotIn('["CMD", "curl"', compose)
        dashboard_source = (ROOT / "dashboard/app.py").read_text(encoding="utf-8")
        self.assertNotIn("predictor = get_predictor()", dashboard_source)
        self.assertIn("if not DASHBOARD_SANDBOX_ENABLED:", dashboard_source)
        self.assertIn("stSidebarCollapsedControl", dashboard_source)
        production_source = (ROOT / "dashboard/production_console.py").read_text(encoding="utf-8")
        self.assertNotIn("TRIGGER 7-STAGE", production_source)
        self.assertNotIn("Sensor Simulation Target", production_source)
        self.assertNotIn("Send Parameter Offset", production_source)
        self.assertNotIn("RUN 90s", production_source)
        self.assertNotIn("LIBAD", production_source)
        self.assertIn("Operate", production_source)
        self.assertIn("software loopback", production_source)
        self.assertIn("Diagnose", production_source)
        self.assertIn("Traceability", production_source)
        self.assertIn("Multimodal", production_source)
        self.assertIn("replay-grid", production_source)
        self.assertIn("gallery-grid", production_source)
        self.assertIn("Published class mix", production_source)
        self.assertNotIn("st.image", production_source)
        multimodal_source = (ROOT / "dashboard/multimodal_lane.py").read_text(encoding="utf-8")
        self.assertIn("LIBAD", multimodal_source)
        self.assertIn("official_dataset_present", multimodal_source)
        self.assertIn("libad_samples", multimodal_source)
        api_source = (ROOT / "src/api/main.py").read_text(encoding="utf-8")
        self.assertIn('X-Multimodal-Source": "official-release"', api_source)
        self.assertIn("_attach_inspection_artifacts", api_source)
        self.assertIn("_list_inspection_artifact_names", api_source)
        self.assertIn("_dataset_view_cache", api_source)
        self.assertIn("_control_state_from_parts", api_source)
        self.assertIn("comparable_to_paper", multimodal_source)
        self.assertNotIn("libad_demo", multimodal_source)
        self.assertIn('key="ops_view"', production_source)
        self.assertIn("st.radio", production_source)
        self.assertIn("horizontal=True", production_source)
        self.assertNotIn("st.sidebar.radio", production_source)
        self.assertIn("view-rail-label", production_source)
        self.assertIn("Views stay on this rail", production_source)
        self.assertIn("_kv_panel", production_source)
        self.assertIn("inspection_images", production_source)
        self.assertIn("dataset_images", production_source)
        self.assertIn("Show letterboxed model input", production_source)
        self.assertIn("render_dataset_pager", production_source)
        self.assertNotIn("Preview page", production_source)
        self.assertNotIn("st.segmented_control", production_source)
        self.assertIn("def render_live_strip", production_source)
        self.assertIn("Inference Engine", production_source)
        self.assertNotIn("Inference Health", production_source)
        self.assertIn("PUBLISHED COATINGVISION CLASS", production_source)
        self.assertIn("AI DETECTOR CLASS", production_source)
        self.assertIn('key="ops_reload"', production_source)
        self.assertIn('st.rerun(scope="app")', production_source)
        self.assertNotIn("st.tabs", production_source)
        self.assertNotIn("Bắt đầu", production_source)
        self.assertNotIn("run_every", production_source)

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
        docx_path = next(
            (
                path
                for path in (
                    ROOT / "Al + Materials Competition Application Form.docx",
                    ROOT / "AI+Materials_Competition_Application_Form.docx",
                )
                if path.is_file()
            ),
            None,
        )
        self.assertIsNotNone(docx_path, "application form DOCX missing")
        sources = [
            (ROOT / "README.md").read_text(encoding="utf-8"),
            (ROOT / "docs/presentation_pitch.md").read_text(encoding="utf-8"),
            _docx_text(docx_path),
            _pptx_text(ROOT / "SecureCoating-Vision_Final_Defense_6min.pptx"),
        ]
        for source in sources:
            self.assertIn(identity["brand"], source)
            self.assertIn(identity["registered_title"], source)
            self.assertIn(identity["tagline"], source)
            self.assertNotIn("82 passing tests", source)
            self.assertNotIn("85 passing tests", source)
        docx = _docx_text(docx_path)
        self.assertNotIn("YOLOv8-seg", docx)
        self.assertIn("YOLO26n", docx)
        self.assertIn("History", docx)
        self.assertIn("Multimodal", docx)

    def test_libad_artifacts_never_claim_paper_comparability(self):
        paths = [
            ROOT / "reports/libad/libad_benchmark.json",
            *sorted((ROOT / "reports/libad_demo").glob("*.json")),
        ]
        extra = [
            ROOT / "reports/libad/official_mount_hashes.json",
            ROOT / "reports/libad/official_local_adapter.json",
        ]
        paths.extend(path for path in extra if path.is_file())
        self.assertTrue(paths)
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            serialized = json.dumps(payload, allow_nan=False)
            self.assertNotIn('"comparable_to_paper": true', serialized, path.as_posix())
        benchmark = json.loads((ROOT / "reports/libad/libad_benchmark.json").read_text(encoding="utf-8"))
        self.assertEqual(benchmark["evidence_class"], "protocol_fixture")
        self.assertFalse(benchmark["comparable_to_paper"])
        self.assertFalse(benchmark["official_protocol_complete"])
        official = ROOT / "reports/libad/official_local_adapter.json"
        if official.is_file():
            payload = json.loads(official.read_text(encoding="utf-8"))
            self.assertFalse(payload["comparable_to_paper"])
            self.assertNotEqual(payload["evidence_class"], "protocol_fixture")
        interim = ROOT / "reports/libad/official_dinov2_dacore_interim.json"
        if interim.is_file():
            payload = json.loads(interim.read_text(encoding="utf-8"))
            purpose = payload["experiments"]["multimodal"]["purpose"]
            self.assertIn("DINO v2", purpose)
            self.assertNotIn("DINOv3", purpose)
            json.dumps(payload, allow_nan=False)
        failed = ROOT / "reports/libad/official_dinov3_dacore.json"
        if failed.is_file():
            payload = json.loads(failed.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("status"), "FAILED")
            self.assertFalse(payload["comparable_to_paper"])
            json.dumps(payload, allow_nan=False)
        hashes = ROOT / "reports/libad/official_mount_hashes.json"
        if hashes.is_file():
            payload = json.loads(hashes.read_text(encoding="utf-8"))
            self.assertFalse(payload["comparable_to_paper"])
            self.assertEqual(payload["official_split_seeds"], [347, 725, 1245, 4012, 4589, 5021, 5678, 6234, 6789, 7345])

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
        # Full-repo development galleries stay in git; competition ZIP may omit them.
        existing = [path for path in paths if path.is_file()]
        if len(existing) < 3:
            self.skipTest("development CoatingVision gallery artifacts not present")
        for path in existing:
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
        self.assertIn("src/inference/hardware_profile.py", string_items)
        self.assertIn("docs/hardware_profiles.md", string_items)
        self.assertIn("docker-compose.gpu.yml", string_items)
        self.assertIn("Dockerfile.gpu", string_items)
        self.assertIn(".github/workflows/ci.yml", string_items)
        self.assertIn("tests/test_hardware_profile.py", string_items)
        self.assertIn("reports/submission_manifest.json", string_items)
        self.assertIn("TREE_DIRS", source)
        self.assertIn("verify_packed_artifact", source)
        self.assertIn("executed contract tests", source)
        self.assertGreaterEqual(source.count("tests/test_evaluation_integrity.py"), 2)
        self.assertIn("assert_test_snapshot_certifies_head", source)
        self.assertIn("test snapshot does not certify current HEAD", source)
        self.assertIn("CLAIM_HASH_PREFIXES", source)
        self.assertIn("reports/defense_gifs", source)
        self.assertIn("docs/industrialization_path.md", source)
        self.assertIn("ZIP_ARCNAMES", source)
        self.assertNotIn("logo.png", string_items)
        self.assertNotIn("official_dinov3_dacore.json", source)
        self.assertNotIn("reports/pilot_validation_dossier.md", source)
        self.assertNotIn("reports/coatingvision_gallery", source)
        self.assertIn("tests/test_defense_gifs.py", source)
        # Packed contract gate must execute defense GIF regeneration, not only ship the GIFs.
        self.assertGreaterEqual(source.count("tests/test_defense_gifs.py"), 2)
        self.assertIn("scripts/build_synthetic_evaluation_manifest.py", string_items)
        self.assertIn("data/evaluation/README.md", source)
        self.assertIn("data/evaluation/reference", source)
        self.assertIn("reports/synthetic_evaluation_manifest.json", source)
        self.assertIn("data/coatingvision_real_test", source)
        self.assertIn("NOTICE.md", source)
        self.assertIn("official_dinov2_dacore_interim_results.csv", source)
        self.assertIn("reports/pytest_junit.xml", source)
        self.assertIn("reports/pytest_output.txt", source)
        self.assertNotIn('"reports/dataset_manifest.json"', source)
        self.assertNotIn("Ensures NO ground-truth labels", source)
        self.assertNotIn("NO ground truth labels", source)
        self.assertIn("no training/private ground-truth labels", source)
        self.assertIn("Synthetic evaluation-fixture labels are explicitly allowed", source)
        self.assertIn("reports/libad/libad_benchmark.json", string_items)
        self.assertIn("reports/libad/official_mount_hashes.json", string_items)
        self.assertIn("scripts/record_libad_official_manifest.py", string_items)
        self.assertIn("src/libad/official_code.py", string_items)
        self.assertIn("src/libad/official_baseline.py", string_items)
        self.assertIn("scripts/run_libad_official_baseline.py", string_items)
        self.assertIn("tests/test_libad_official_baseline.py", string_items)
        self.assertIn("tests/test_libad_protocol.py", string_items)
        self.assertIn("reports/libad_demo/demo_manifest.json", string_items)
        self.assertIn("SecureCoating-Vision_Final_Defense_6min.pptx", string_items)
        self.assertIn("dashboard/production_console.py", string_items)
        self.assertIn("dashboard/multimodal_lane.py", string_items)
        self.assertIn("tests/test_libad_official_index.py", string_items)
        self.assertIn("tests/test_onnx_postprocess.py", string_items)
        self.assertIn("scripts/generate_defense_slides.py", string_items)
        self.assertIn("scripts/generate_defense_gifs.py", string_items)
        self.assertIn("scripts/check_ci_junit.py", string_items)
        self.assertIn("tests/test_defense_gifs.py", string_items)
        self.assertIn("reports/defense_gifs/coating_surface_heldout.png", source)
        self.assertIn("--skip-tests", source)
        self.assertNotIn("Evidence_Aware.pptx", source)
        self.assertNotIn("Coating_Surface_Evidence.pptx", source)

    def test_official_defense_pptx_does_not_display_forbidden_factory_numbers(self):
        text = _pptx_text(ROOT / "SecureCoating-Vision_Final_Defense_6min.pptx")
        self.assertNotIn("99.4%", text)
        self.assertNotIn("0.02 ppm", text)
        self.assertNotIn("P99.9 SLA", text)

    def test_defense_slide_generator_does_not_print_forbidden_factory_numbers(self):
        source = (ROOT / "scripts/generate_defense_slides.py").read_text(encoding="utf-8")
        self.assertNotIn("99.4% mAP", source)
        self.assertNotIn("0.02 ppm", source)
        self.assertNotIn("−84% scrap", source)
        self.assertNotIn("-84% scrap", source)
        self.assertNotIn("P99.9 SLA", source)
        self.assertNotIn("Coating_Surface_Evidence", source)
        self.assertNotIn("Evidence_Aware", source)
        self.assertNotIn("3c3f2773", source)
        self.assertIn("coatingvision_real_test_metrics.json", source)
        self.assertIn("_load_rgb_metrics", source)
        self.assertIn("coating_surface_heldout.png", source)
        self.assertIn("Prof. Kris Singh", source)
        self.assertIn("Trinh Hoang Tu · HUFLIT", source)
        self.assertNotIn("logo.png", source)
        self.assertNotIn("MSE Lab", source)
        self.assertTrue((ROOT / "reports/defense_gifs/coating_surface_heldout.png").is_file())
        with zipfile.ZipFile(ROOT / "SecureCoating-Vision_Final_Defense_6min.pptx") as archive:
            media_names = [name for name in archive.namelist() if name.startswith("ppt/media/")]
            # Title slide is typography-only; logo.png must not appear as product branding.
            if (ROOT / "logo.png").is_file():
                logo_bytes = (ROOT / "logo.png").read_bytes()
                media = [archive.read(name) for name in media_names]
                self.assertNotIn(logo_bytes, media)
        leftover_decks = (
            "SecureCoating-Vision_Final_Defense_6min_Evidence_Aware.pptx",
            "SecureCoating-Vision_Final_Defense_6min_Coating_Surface_Evidence.pptx",
            "SecureCoating-Vision_Final_Defense_6min_Fixed.pptx",
        )
        for name in leftover_decks:
            self.assertFalse((ROOT / name).is_file(), name)

    def test_pitch_and_slides_match_official_local_adapter_status(self):
        pitch = (ROOT / "docs/presentation_pitch.md").read_text(encoding="utf-8")
        slides = (ROOT / "scripts/generate_defense_slides.py").read_text(encoding="utf-8")
        self.assertNotIn("official data execution is still pending", pitch)
        self.assertNotIn("official-data validation is pending", pitch)
        self.assertIn("official_local_adapter.json", pitch)
        self.assertIn("official_dinov2_dacore_interim.json", pitch)
        self.assertIn("0.856", pitch)
        self.assertIn("0.716", pitch)
        self.assertIn("paper-comparable", pitch)
        self.assertIn("0.856", slides)
        self.assertIn("0.716", slides)
        self.assertNotIn(
            "Official 4.84 GB dataset and 10 splits are not in this runtime",
            slides,
        )

    def test_default_dataset_yaml_is_labelled_synthetic_only(self):
        text = (ROOT / "configs/dataset.yaml").read_text(encoding="utf-8")
        self.assertIn("coatingvision_real_detect.yaml", text)
        self.assertIn("Synthetic development data only", text)
        self.assertNotIn("Procedurally generated synthetic coating defect images", text)

    def test_test_manifest_discloses_dirty_source_provenance(self):
        manifest = json.loads(
            (ROOT / "reports/test_manifest.json").read_text(encoding="utf-8")
        )
        current = git_source_provenance(ROOT)
        if current["working_tree_dirty"] is None:
            # Packaged ZIP has no .git — judges verify reports/submission_manifest.json instead.
            submission = ROOT / "reports/submission_manifest.json"
            self.assertTrue(
                submission.is_file(),
                "Packaged artifacts must ship reports/submission_manifest.json",
            )
            payload = json.loads(submission.read_text(encoding="utf-8"))
            self.assertIn("source_commit", payload)
            self.assertIn("files", payload)
            self.assertGreater(len(payload["files"]), 10)
            return
        if current["working_tree_dirty"]:
            # Local developer WIP is expected; packed-artifact verify runs on a clean unzip.
            return
        self.assertEqual(manifest["working_tree_dirty"], False)
        self.assertIsNone(manifest["source_diff_sha256"])

    def test_both_readmes_share_the_generated_test_manifest_contract(self):
        for name in ("README.md", "README_CN.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertEqual(text.count("<!-- TEST_MANIFEST:START -->"), 1, name)
            self.assertEqual(text.count("<!-- TEST_MANIFEST:END -->"), 1, name)
            self.assertIn("reports/test_manifest.json", text, name)
        recorder = (ROOT / "scripts/record_test_manifest.py").read_text(encoding="utf-8")
        self.assertIn('README_CN.md', recorder)
        self.assertIn("当前软件验证快照", recorder)

    def test_agpl_license_notice_and_readme_identity(self):
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertTrue(
            ("GNU AFFERO GENERAL PUBLIC LICENSE" in license_text)
            or ("AGPL" in license_text),
            "LICENSE must contain AGPL / GNU Affero text",
        )
        notice = (ROOT / "NOTICE.md").read_text(encoding="utf-8")
        self.assertIn("Ultralytics", notice)
        self.assertIn("BSD 3-Clause", notice)
        self.assertIn("CC BY 4.0", notice)
        pitch = (ROOT / "docs" / "presentation_pitch.md").read_text(encoding="utf-8")
        self.assertNotIn("3c3f2773", pitch)
        self.assertIn("d1db7823", pitch)
        status = (ROOT / "docs" / "implementation_status.md").read_text(encoding="utf-8")
        self.assertNotIn("open release blocker", status)
        self.assertIn("production-deployment security acceptance gate", status)
        for name in ("README.md", "README_CN.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertNotRegex(
                text,
                r"(?i)\bMIT\b.*\b(license|许可)",
                f"{name} must not claim MIT for project source",
            )
            self.assertNotIn("License: MIT", text, name)
            self.assertNotIn("[MIT License]", text, name)

    def test_ci_and_compose_keep_security_gates_enabled(self):
        workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn("persist-credentials: false", workflow)
        self.assertIn("docker-compose.gpu.yml config", workflow)
        self.assertIn("no-new-privileges:true", compose)
        self.assertIn("max-size: \"10m\"", compose)
        self.assertIn("stop_grace_period: 20s", compose)


if __name__ == "__main__":
    unittest.main()
