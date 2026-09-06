"""
Build a clean submission ZIP with strict whitelist.
Ensures no training/private ground-truth labels are included.
Synthetic evaluation-fixture labels are explicitly allowed for reproducibility.
Also excludes __pycache__, .env, and build artifacts.

Usage:
    python scripts/build_submission.py
    python scripts/build_submission.py --skip-tests
    python scripts/build_submission.py --require-live-api
"""

import argparse
import os
import subprocess
import zipfile
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

OUTPUT_ZIP = "SecureCoatingVision_Submission.zip"

# Strict whitelist of files/dirs to include
WHITELIST = [
    # Core source code
    "src/api/main.py",
    "src/inference/__init__.py" if os.path.exists("src/inference/__init__.py") else None,
    "src/inference/onnx_engine.py",
    "src/inference/yolo_engine.py",
    "src/inference/predictor.py",
    "src/inference/hardware_profile.py",
    "src/inference/postprocess.py",
    "src/inference/sensor_fusion.py",
    "src/inference/failsafe.py",
    "src/inference/electrode_metrology.py",
    "src/inference/multi_stage_pipeline.py",
    "src/industrial/__init__.py",
    "src/industrial/protocol_manager.py",
    "src/industrial/web_synchronizer.py",
    "src/industrial/optical_budget.py",
    "src/industrial/latency_budget.py",
    "src/industrial/spc_spatial_diagnostics.py",
    "src/industrial/modbus_loopback.py",
    "src/traceability/quality_memory.py",
    "src/traceability/root_cause_engine.py",
    "src/traceability/roll_certificate.py",
    "src/training/train_yolo.py",
    "src/training/train_baseline.py",
    "src/libad/__init__.py",
    "src/libad/protocol.py",
    "src/libad/features.py",
    "src/libad/memory.py",
    "src/libad/scorer.py",
    "src/libad/metrics.py",
    "src/libad/evidence_gate.py",
    "src/libad/dataset.py",
    "src/libad/certificate.py",
    "src/libad/demo_cases.py",
    "src/libad/evaluate.py",
    "src/libad/official_code.py",
    "src/libad/official_baseline.py",
    "src/evaluation/evaluate.py",
    "src/evaluation/dataset_manifest.py",
    "src/evaluation/coatingvision_grouping.py",
    "src/utils/__init__.py",
    # Dashboard
    "dashboard/app.py",
    "dashboard/api_client.py",
    "dashboard/production_console.py",
    "dashboard/multimodal_lane.py",
    "dashboard/sandbox_console.py",
    "dashboard/__init__.py",
    ".streamlit/config.toml",
    # Tests
    "tests/test_failsafe.py",
    "tests/test_sensor_fusion.py",
    "tests/test_postprocess.py",
    "tests/test_onnx_engine.py",
    "tests/test_onnx_postprocess.py",
    "tests/test_predictor.py",
    "tests/test_hardware_profile.py",
    "tests/test_api.py",
    "tests/test_web_synchronizer.py",
    "tests/test_electrode_metrology.py",
    "tests/test_root_cause_engine.py",
    "tests/test_roll_certificate.py",
    "tests/test_multi_stage_pipeline.py",
    "tests/test_optical_budget.py",
    "tests/test_latency_budget.py",
    "tests/test_spc_spatial_diagnostics.py",
    "tests/test_safety_contracts.py",
    "tests/test_modbus_loopback.py",
    "tests/test_industrial_env_overrides.py",
    "tests/test_download_libad.py",
    "tests/test_coatingvision_grouping.py",
    "tests/test_quality_memory_faults.py",
    "tests/test_evaluation_integrity.py",
    "tests/test_dashboard_api_client.py",
    "tests/test_dashboard_app.py",
    "tests/test_libad_evidence_gate.py",
    "tests/test_libad_metrics.py",
    "tests/test_libad_memory.py",
    "tests/test_libad_demo_cases.py",
    "tests/test_libad_evaluate.py",
    "tests/test_libad_official_index.py",
    "tests/test_libad_features.py",
    "tests/test_libad_official_code.py",
    "tests/test_libad_official_baseline.py",
    "tests/test_libad_protocol.py",
    "tests/test_defense_gifs.py",
    "tests/test_repository_evidence_artifacts.py",
    "tests/test_zip_safe_coatingvision_real_test.py",
    "tests/test_coatingvision_manifest_regenerator.py",
    "tests/test_production_startup.py",
    "tests/test_ci_junit.py",
    # Configs
    "configs/app.yaml",
    "configs/model.yaml",
    "configs/dataset.yaml",
    "configs/evaluation.yaml",
    "configs/coatingvision_real_detect.yaml" if os.path.exists("configs/coatingvision_real_detect.yaml") else None,
    "configs/coatingvision_real_test.yaml" if os.path.exists("configs/coatingvision_real_test.yaml") else None,
    "configs/project_identity.yaml",
    "configs/libad.yaml",
    "configs/calibration.yaml",
    # Scripts
    "scripts/run_evaluation.py",
    "scripts/build_synthetic_evaluation_manifest.py",
    "scripts/run_ultralytics_validation.py",
    "scripts/generate_synthetic_coating_defects.py",
    "scripts/prepare_synthetic_dataset.py",
    "scripts/download_dataset.py",
    "scripts/build_submission.py",
    "scripts/run_pilot_validation_dossier.py",
    "scripts/smoke_live.py",
    "scripts/run_modbus_loopback_trigger.py",
    "scripts/verify_inject.py",
    "scripts/backup_quality_db.py",
    "scripts/run_api.ps1",
    "scripts/run_dashboard.ps1",
    "scripts/download_libad.py",
    "scripts/run_libad_benchmark.py",
    "scripts/run_libad_official_baseline.py",
    "scripts/run_libad_demo.py",
    "scripts/run_libad_live.py",
    "scripts/fetch_libad_official_code.py",
    "scripts/record_libad_official_manifest.py",
    "scripts/record_test_manifest.py",
    "scripts/check_ci_junit.py",
    "scripts/evaluate_coatingvision_real.py" if os.path.exists("scripts/evaluate_coatingvision_real.py") else None,
    "scripts/build_coatingvision_test_bundle_manifest.py" if os.path.exists("scripts/build_coatingvision_test_bundle_manifest.py") else None,
    "scripts/polish_application_form_docx.py" if os.path.exists("scripts/polish_application_form_docx.py") else None,
    "scripts/generate_coatingvision_evidence_views.py" if os.path.exists("scripts/generate_coatingvision_evidence_views.py") else None,
    "scripts/prepare_coatingvision_detection_dataset.py" if os.path.exists("scripts/prepare_coatingvision_detection_dataset.py") else None,
    "scripts/run_external_coatingvision_demo.py" if os.path.exists("scripts/run_external_coatingvision_demo.py") else None,
    "scripts/verify_coatingvision_dataset.py" if os.path.exists("scripts/verify_coatingvision_dataset.py") else None,
    "scripts/train_coatingvision_real.py" if os.path.exists("scripts/train_coatingvision_real.py") else None,
    "reports/coatingvision_real_test_metrics.json" if os.path.exists("reports/coatingvision_real_test_metrics.json") else None,
    "reports/external_coatingvision_demo/coatingvision_model_output.json" if os.path.exists("reports/external_coatingvision_demo/coatingvision_model_output.json") else None,
    "reports/external_coatingvision_demo/coatingvision_model_output.png" if os.path.exists("reports/external_coatingvision_demo/coatingvision_model_output.png") else None,
    # Documentation
    "README.md",
    "README_CN.md",
    "LICENSE",
    "NOTICE.md",
    "logo.png",
    "reports/analysis_report.md",
    "reports/analysis_report_template.md",
    "reports/pilot_validation_dossier.md",
    "docs/system_architecture.md",
    "docs/inspection_workflow.md",
    "docs/scoring_rubric_mapping.md",
    "docs/presentation_pitch.md",
    "docs/libad_validation_extension.md",
    "docs/implementation_status.md",
    "docs/hardware_profiles.md",
    "scripts/generate_defense_slides.py",
    "scripts/generate_defense_gifs.py",
    "SecureCoating-Vision_Final_Defense_6min.pptx",
    "Al + Materials Competition Application Form.docx",
    # Deployment
    "Dockerfile",
    "Dockerfile.gpu",
    "docker-compose.yml",
    "docker-compose.gpu.yml",
    ".github/workflows/ci.yml",
    "requirements.txt",
    "requirements-docker.txt",
    "requirements-docker-lock.txt",
    "requirements-gpu.txt",
    "requirements-lock.txt",
    "requirements-core.txt",
    ".dockerignore",
    ".env.example",
    # Data and ZIP-safe evaluation fixtures.
    # Training/private labels are excluded;
    # synthetic evaluator labels are intentionally included.
    "data/README.md",
    "data/evaluation/README.md" if os.path.exists("data/evaluation/README.md") else None,
    "reports/libad/libad_benchmark.json",
    "reports/libad/libad_predictions.json",
    "reports/libad/official_mount_hashes.json",
    "reports/libad/official_local_adapter.json" if os.path.exists("reports/libad/official_local_adapter.json") else None,
    "outputs/official_local_adapter_predictions.json" if os.path.exists("outputs/official_local_adapter_predictions.json") else None,
    "reports/libad/official_dinov2_dacore_interim.json" if os.path.exists("reports/libad/official_dinov2_dacore_interim.json") else None,
    "reports/libad/official_dinov2_dacore_interim_results.csv" if os.path.exists("reports/libad/official_dinov2_dacore_interim_results.csv") else None,
    "reports/libad/official_dinov3_dacore.json" if os.path.exists("reports/libad/official_dinov3_dacore.json") else None,
    "reports/libad/official_dinov3_run_card.json" if os.path.exists("reports/libad/official_dinov3_run_card.json") else None,
    "reports/submission_manifest.json",
    "reports/libad_demo/demo_manifest.json",
    "reports/libad_demo/final_screen.json",
    "reports/defense_gifs/rgb_hold_replay.gif" if os.path.exists("reports/defense_gifs/rgb_hold_replay.gif") else None,
    "reports/defense_gifs/libad_gate.gif" if os.path.exists("reports/defense_gifs/libad_gate.gif") else None,
    "reports/defense_gifs/coating_surface_heldout.png" if os.path.exists("reports/defense_gifs/coating_surface_heldout.png") else None,
    "reports/evaluation_results.json" if os.path.exists("reports/evaluation_results.json") else None,
    "reports/evaluation_results.csv" if os.path.exists("reports/evaluation_results.csv") else None,
    "reports/ultralytics_validation_results.json" if os.path.exists("reports/ultralytics_validation_results.json") else None,
    "reports/synthetic_evaluation_manifest.json" if os.path.exists("reports/synthetic_evaluation_manifest.json") else None,
    "reports/model_sha256.txt" if os.path.exists("reports/model_sha256.txt") else None,
    "reports/test_manifest.json",
    "reports/pytest_junit.xml" if os.path.exists("reports/pytest_junit.xml") else None,
    "reports/pytest_output.txt" if os.path.exists("reports/pytest_output.txt") else None,
    "reports/environment.txt" if os.path.exists("reports/environment.txt") else None,
    "reports/evaluation_command.txt" if os.path.exists("reports/evaluation_command.txt") else None,
    # Model artifacts (REQUIRED)
    "outputs/model.onnx",
    "outputs/best.pt",
]
# Directories to include recursively (images, manifests, evidence JSON/PNG)
TREE_DIRS = [
    ("data/test_set/images", "data/test_set/images"),
    ("data/demo_real", "data/demo_real"),
    ("data/coatingvision_real_test", "data/coatingvision_real_test"),
    ("data/evaluation/images", "data/evaluation/images"),
    ("data/evaluation/labels", "data/evaluation/labels"),
    ("data/evaluation/reference", "data/evaluation/reference"),
    ("reports/libad_demo", "reports/libad_demo"),
    ("reports/coatingvision_gallery", "reports/coatingvision_gallery"),
    ("reports/coatingvision_real_demo", "reports/coatingvision_real_demo"),
    ("reports/external_coatingvision_demo", "reports/external_coatingvision_demo"),
    ("reports/coatingvision_visual_evidence", "reports/coatingvision_visual_evidence"),
]
TREE_ALLOWED_SUFFIXES = (".jpg", ".jpeg", ".png", ".txt", ".json", ".md", ".gif")
# Legacy flat IMAGE_DIRS name kept for tests that may reference the idea.
IMAGE_DIRS = TREE_DIRS

# Explicitly EXCLUDED (safety check)
BLACKLIST_PATTERNS = [
    "__pycache__",
    ".pyc",
    ".pyo",
    ".env",
    ".git",
    ".venv",
    "labels",  # Training/private labels forbidden; synthetic evaluation + ZIP-safe CoatingVision test labels allowed
    "coating_defects",  # full training data stays local
    "coatingvision_real_detect",  # full local detect tree stays local; ZIP uses data/coatingvision_real_test
    "runs/",
    "yolo_training",
    ".ipynb_checkpoints",
    "node_modules",
    "*.db",
    "*.zip",
]

# Files that must exist inside the packed ZIP or the artifact is not submittable.
ARTIFACT_REQUIRED_PATHS = [
    "src/inference/hardware_profile.py",
    "src/inference/predictor.py",
    "src/inference/onnx_engine.py",
    "src/inference/yolo_engine.py",
    "docs/hardware_profiles.md",
    "docker-compose.gpu.yml",
    "Dockerfile.gpu",
    ".github/workflows/ci.yml",
    "reports/test_manifest.json",
    "reports/pytest_junit.xml",
    "reports/pytest_output.txt",
    "reports/submission_manifest.json",
    "outputs/model.onnx",
    "outputs/best.pt",
    "data/demo_real/manifest.json",
    "data/coatingvision_real_test/README.md",
    "data/coatingvision_real_test/manifest.json",
    "configs/coatingvision_real_test.yaml",
    "NOTICE.md",
    "reports/libad/official_dinov2_dacore_interim_results.csv",
    "data/evaluation/README.md",
    "reports/coatingvision_visual_evidence/image_1548_optical_raw.png",
    "reports/coatingvision_visual_evidence/image_1548_contrast_clahe.png",
    "reports/coatingvision_visual_evidence/image_1548_yolo_candidate_zoom.png",
    "scripts/generate_defense_gifs.py",
    "scripts/build_synthetic_evaluation_manifest.py",
    "scripts/evaluate_coatingvision_real.py",
    "tests/test_evaluation_integrity.py",
    "tests/test_defense_gifs.py",
]


def is_blacklisted(path):
    # Exact / segment-aware exclusions (avoid false positives like .env.example)
    parts = path.replace("\\", "/").split("/")
    name = parts[-1] if parts else path
    if name in {".env", ".git", ".venv"}:
        return True
    if "__pycache__" in parts or name.endswith((".pyc", ".pyo")):
        return True
    if name.endswith(".db") or name.endswith(".zip"):
        return True
    if (
        ("labels" in parts and "evaluation" not in parts and "coatingvision_real_test" not in parts)
        or "coating_defects" in parts
        or "coatingvision_real_detect" in parts
        or "yolo_training" in parts
    ):
        return True
    if "runs" in parts or ".ipynb_checkpoints" in parts or "node_modules" in parts:
        return True
    return False


def _iter_tree_files(src_dir: str):
    full_dir = os.path.join(PROJECT_ROOT, src_dir)
    if not os.path.isdir(full_dir):
        return
    for root, _dirs, files in os.walk(full_dir):
        for fname in sorted(files):
            if not fname.lower().endswith(TREE_ALLOWED_SUFFIXES):
                continue
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, PROJECT_ROOT).replace("\\", "/")
            yield fpath, rel


def _git_head_sha() -> str:
    try:
        commit_run = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if commit_run.returncode == 0 and commit_run.stdout.strip():
            return commit_run.stdout.strip()
    except OSError:
        pass
    return "unknown"


def assert_test_snapshot_certifies_head() -> dict:
    """Abort unless reports/test_manifest.json certifies the current git HEAD.

    Release rule: the packed ZIP must not claim '265 tests validate submitted source'
    unless the recorded snapshot commit equals HEAD and was taken on a clean tree.
    """
    import json

    path = os.path.join(PROJECT_ROOT, "reports", "test_manifest.json")
    if not os.path.isfile(path):
        print("  [ERROR] ABORT BUILD: reports/test_manifest.json missing.")
        sys.exit(1)
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    head = _git_head_sha()
    snap = str(manifest.get("commit_sha") or "")
    if not snap or snap == "unknown" or head == "unknown" or snap != head:
        print(
            "  [ERROR] ABORT BUILD: test snapshot does not certify current HEAD\n"
            f"    test_manifest.commit_sha = {snap!r}\n"
            f"    git HEAD                 = {head!r}"
        )
        sys.exit(1)
    if bool(manifest.get("working_tree_dirty")):
        print(
            "  [ERROR] ABORT BUILD: test snapshot was recorded on a dirty working tree.\n"
            "    Re-run on a clean tree: scripts/record_test_manifest.py then rebuild."
        )
        sys.exit(1)
    print(f"  Provenance lock OK: test_manifest.commit_sha == HEAD ({head[:12]}…)")
    return manifest


# Paths hashed into submission_manifest for judge verification (exclude the manifest itself).
CLAIM_HASH_PREFIXES = (
    "src/",
    "configs/",
    "scripts/",
    "dashboard/",
    "tests/",
    "docs/",
    "reports/",
    "outputs/",
    "data/coatingvision_real_test/",
)
CLAIM_HASH_EXACT = {
    "docker-compose.yml",
    "docker-compose.gpu.yml",
    "Dockerfile",
    "Dockerfile.gpu",
    ".github/workflows/ci.yml",
    "README.md",
    "README_CN.md",
    "LICENSE",
    "NOTICE.md",
    "requirements.txt",
    "requirements-core.txt",
    "requirements-lock.txt",
    "requirements-docker.txt",
    "requirements-docker-lock.txt",
    "requirements-gpu.txt",
    ".env.example",
    "SecureCoating-Vision_Final_Defense_6min.pptx",
    "Al + Materials Competition Application Form.docx",
    "data/demo_real/manifest.json",
}


def write_submission_manifest(added_files):
    """Record artifact-level hashes judges can verify without .git."""
    import hashlib
    import json
    from datetime import datetime, timezone

    test_manifest = assert_test_snapshot_certifies_head()
    certified = str(test_manifest.get("commit_sha") or "")

    files = {}
    for rel in sorted(set(added_files)):
        rel_n = rel.replace("\\", "/")
        if rel_n == "reports/submission_manifest.json":
            continue  # avoid self-reference
        if not rel_n.startswith(CLAIM_HASH_PREFIXES) and rel_n not in CLAIM_HASH_EXACT:
            continue
        full = os.path.join(PROJECT_ROOT, rel_n)
        if not os.path.isfile(full):
            continue
        digest = hashlib.sha256()
        with open(full, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        files[rel_n] = digest.hexdigest()

    head = _git_head_sha()
    if certified != head:
        print(
            "  [ERROR] ABORT BUILD: certified test commit drifted from HEAD during packaging\n"
            f"    certified={certified!r} head={head!r}"
        )
        sys.exit(1)

    payload = {
        "schema_version": "securecoating-submission-manifest/v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_commit": certified,
        "test_manifest_commit": certified,
        "test_manifest_working_tree_dirty": False,
        "certified_by_test_manifest": True,
        "file_count": len(files),
        "files": files,
        "note": (
            "Artifact provenance for the packed ZIP. Judges should hash claim-bearing "
            "files in the unzipped archive against this map (source, configs, models, "
            "reports, predictions, deck). source_commit equals test_manifest.commit_sha. "
            "Do not require a .git directory. This file is excluded from its own hash map."
        ),
    }
    out = os.path.join(PROJECT_ROOT, "reports", "submission_manifest.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return out


def verify_packed_artifact(zip_path: str) -> None:
    """Unzip the just-built archive and execute the release contract tests."""
    import tempfile

    print("  VERIFYING PACKED ARTIFACT (unzip → import → execute contract tests):")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        missing = [path for path in ARTIFACT_REQUIRED_PATHS if path not in names]
        if missing:
            print("  [ERROR] ZIP missing required paths:")
            for path in missing:
                print(f"    - {path}")
            sys.exit(1)
        demo_imgs = [n for n in names if n.startswith("data/demo_real/images/") and n.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(demo_imgs) < 1:
            print("  [ERROR] ZIP missing data/demo_real/images/*")
            sys.exit(1)
        evidence_json = [
            n for n in names
            if n.endswith("coatingvision_model_output.json")
            and (
                n.startswith("reports/coatingvision_gallery/")
                or n.startswith("reports/coatingvision_real_demo/")
                or n.startswith("reports/external_coatingvision_demo/")
            )
        ]
        if len(evidence_json) < 3:
            print(f"  [ERROR] ZIP needs >=3 coatingvision evidence JSON files, found {len(evidence_json)}")
            sys.exit(1)
        with tempfile.TemporaryDirectory(prefix="scv_zip_verify_") as tmp:
            zf.extractall(tmp)
            env = os.environ.copy()
            env["PYTHONPATH"] = os.path.join(tmp, "src")
            import_probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from inference.hardware_profile import resolve_inference_profile\n"
                        "from inference.predictor import CoatingPredictor\n"
                        "print('import_ok', resolve_inference_profile is not None, CoatingPredictor is not None)"
                    ),
                ],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
            )
            if import_probe.returncode != 0:
                print("  [ERROR] Packed artifact import failed:")
                print(import_probe.stdout)
                print(import_probe.stderr)
                sys.exit(1)
            executed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "--tb=line",
                    "tests/test_hardware_profile.py",
                    "tests/test_predictor.py",
                    "tests/test_libad_evidence_gate.py",
                    "tests/test_evaluation_integrity.py",
                    "tests/test_repository_evidence_artifacts.py",
                    "tests/test_defense_gifs.py",
                ],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
            )
            print(executed.stdout)
            if executed.returncode != 0:
                print("  [ERROR] Packed artifact contract tests failed:")
                print(executed.stderr)
                sys.exit(1)
            print("  Packed artifact import + executed contract tests: OK")


def build_zip(require_live_api=False, skip_tests=False):
    print(f"Building submission ZIP: {OUTPUT_ZIP}")
    print(f"Project root: {PROJECT_ROOT}")
    print()

    required = ["outputs/model.onnx", "outputs/best.pt", "data/test_set/images"]
    missing_required = []
    for req in required:
        path = os.path.join(PROJECT_ROOT, req)
        if req.endswith("images"):
            if not os.path.isdir(path) or not any(
                f.lower().endswith((".jpg", ".jpeg", ".png")) for f in os.listdir(path)
            ):
                missing_required.append(req)
        elif not os.path.isfile(path):
            missing_required.append(req)
    if missing_required:
        print("  [ERROR] Required artifacts missing:")
        for m in missing_required:
            print(f"    - {m}")
        sys.exit(1)
    print("  Required artifacts present: outputs/model.onnx, outputs/best.pt, test_set images")

    if skip_tests:
        print(
            "  SKIPPING UNIT TESTS (--skip-tests). Using the current reports/test_manifest.json.\n"
            "  Final release builds should NOT use --skip-tests.\n"
        )
    else:
        print("  RUNNING UNIT TESTS:")
        manifest_run = subprocess.run(
            [sys.executable, "scripts/record_test_manifest.py"],
            cwd=PROJECT_ROOT,
        )
        if manifest_run.returncode != 0:
            print("\n  [ERROR] Unit tests failed! Submission build aborted.")
            sys.exit(1)
        print("  All unit tests PASSED. Test manifest written to reports/test_manifest.json\n")

    assert_test_snapshot_certifies_head()

    print("  RUNNING LIVE INJECT CHECK (optional if API offline):")
    live = subprocess.run(
        [sys.executable, "scripts/verify_inject.py"],
        cwd=PROJECT_ROOT,
    )
    if live.returncode != 0:
        if require_live_api:
            print("  [ERROR] Live inject check is required and did not pass. Build aborted.")
            return 1
        print("  [WARN] Live inject check failed or API offline — archive will not be marked release-ready.")
    else:
        print("  Live inject check PASSED.\n")

    if os.path.exists(OUTPUT_ZIP):
        os.remove(OUTPUT_ZIP)

    added_files = []
    skipped_files = []
    planned = []

    for filepath in WHITELIST:
        if filepath is None:
            continue
        full_path = os.path.join(PROJECT_ROOT, filepath)
        if os.path.exists(full_path) and os.path.isfile(full_path):
            if is_blacklisted(filepath):
                skipped_files.append(filepath)
                continue
            planned.append((full_path, filepath.replace("\\", "/")))
        else:
            skipped_files.append(f"(missing) {filepath}")

    for src_dir, _zip_dir in TREE_DIRS:
        for fpath, rel in _iter_tree_files(src_dir):
            if is_blacklisted(rel):
                skipped_files.append(rel)
                continue
            planned.append((fpath, rel))

    # Deduplicate by arcname
    seen = set()
    unique_planned = []
    for fpath, rel in planned:
        if rel in seen:
            continue
        seen.add(rel)
        unique_planned.append((fpath, rel))

    write_submission_manifest([rel for _fpath, rel in unique_planned])
    manifest_path = os.path.join(PROJECT_ROOT, "reports", "submission_manifest.json")
    if os.path.isfile(manifest_path) and "reports/submission_manifest.json" not in seen:
        unique_planned.append((manifest_path, "reports/submission_manifest.json"))

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for fpath, rel in unique_planned:
            zf.write(fpath, rel)
            added_files.append(rel)

    zip_size = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)

    print(f"  Added: {len(added_files)} files")
    print(f"  Skipped: {len(skipped_files)} files")
    print(f"  ZIP size: {zip_size:.1f} MB")
    print()

    # Safety verification
    print("  SAFETY CHECK:")
    with zipfile.ZipFile(OUTPUT_ZIP, 'r') as zf:
        all_names = zf.namelist()
        has_train_labels_leak = any("coating_defects/labels" in n.lower() for n in all_names)
        has_pycache = any("__pycache__" in n for n in all_names)
        has_env = any(".env" in n and not n.endswith(".example") for n in all_names)
        has_gt_leak = any("ground_truth_train" in n.lower() for n in all_names)

        has_onnx = any(n.endswith("outputs/model.onnx") or n == "outputs/model.onnx" for n in all_names)
        has_pt = any(n.endswith("outputs/best.pt") or n == "outputs/best.pt" for n in all_names)
        img_count = sum(1 for n in all_names if n.startswith("data/test_set/images/") and n.lower().endswith((".jpg", ".jpeg", ".png")))
        eval_lbl_count = sum(1 for n in all_names if n.startswith("data/evaluation/labels/") and n.endswith(".txt"))
        has_yolo_engine = any(n.endswith("yolo_engine.py") for n in all_names)

        print(f"    Train Labels Leak:  {'FAIL' if has_train_labels_leak else 'CLEAN'}")
        print(f"    __pycache__:        {'FAIL' if has_pycache else 'CLEAN'}")
        print(f"    .env secrets:       {'FAIL' if has_env else 'CLEAN'}")
        print(f"    model.onnx:         {'OK' if has_onnx else 'FAIL'}")
        print(f"    best.pt:            {'OK' if has_pt else 'FAIL'}")
        print(f"    yolo_engine.py:     {'OK' if has_yolo_engine else 'FAIL'}")
        print(f"    test images:        {img_count}")
        print(f"    evaluation labels:  {eval_lbl_count}")

        if has_train_labels_leak or has_pycache or has_env or has_gt_leak or not has_onnx or not has_pt or img_count < 10 or eval_lbl_count < 10:
            print("\n  WARNING: ZIP failed safety / completeness checks!")
            sys.exit(1)
        else:
            print("\n  All safety checks PASSED.")

    verify_packed_artifact(OUTPUT_ZIP)

    if live.returncode == 0:
        print(f"\n  READY TO SUBMIT: {OUTPUT_ZIP} ({zip_size:.1f} MB)")
    else:
        print(
            f"\n  ARCHIVE BUILT WITH WARNINGS: {OUTPUT_ZIP} ({zip_size:.1f} MB). "
            "Start the API and pass --require-live-api before release."
        )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the evidence-gated SecureCoating-Vision submission archive."
    )
    parser.add_argument(
        "--require-live-api",
        action="store_true",
        help="Abort unless scripts/verify_inject.py passes against a running API.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help=(
            "Pack without re-running pytest. Still requires test_manifest.commit_sha "
            "== HEAD and working_tree_dirty=false. Prefer omitting this flag for final release."
        ),
    )
    args = parser.parse_args(argv)
    return build_zip(require_live_api=args.require_live_api, skip_tests=args.skip_tests)


if __name__ == "__main__":
    raise SystemExit(main())
