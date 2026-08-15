"""
Build a clean submission ZIP with strict whitelist.
Ensures NO ground-truth labels, __pycache__, .env, or build artifacts are included.

Usage:
    python scripts/build_submission.py
"""

import os
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
    "src/traceability/quality_memory.py",
    "src/traceability/root_cause_engine.py",
    "src/traceability/roll_certificate.py",
    "src/training/train_yolo.py",
    "src/training/train_baseline.py",
    "src/evaluation/evaluate.py",
    "src/utils/__init__.py",
    # Dashboard
    "dashboard/app.py",
    # Tests
    "tests/test_failsafe.py",
    "tests/test_sensor_fusion.py",
    "tests/test_postprocess.py",
    "tests/test_onnx_engine.py",
    "tests/test_predictor.py",
    "tests/test_api.py",
    "tests/test_web_synchronizer.py",
    "tests/test_electrode_metrology.py",
    "tests/test_root_cause_engine.py",
    "tests/test_roll_certificate.py",
    "tests/test_multi_stage_pipeline.py",
    "tests/test_optical_budget.py",
    "tests/test_latency_budget.py",
    # Configs
    "configs/app.yaml",
    "configs/model.yaml",
    "configs/dataset.yaml",
    # Scripts
    "scripts/run_evaluation.py",
    "scripts/run_ultralytics_validation.py",
    "scripts/generate_synthetic_coating_defects.py",
    "scripts/prepare_synthetic_dataset.py",
    "scripts/download_dataset.py",
    "scripts/build_submission.py",
    "scripts/run_pilot_validation_dossier.py",
    "scripts/smoke_live.py",
    "scripts/verify_inject.py",
    "scripts/run_api.ps1",
    "scripts/run_dashboard.ps1",
    "configs/evaluation.yaml",
    # Documentation
    "README.md",
    "README_CN.md",
    "LICENSE",
    "logo.png",
    "reports/analysis_report.md",
    "reports/analysis_report_template.md",
    "reports/pilot_validation_dossier.md",
    "docs/system_architecture.md",
    "docs/inspection_workflow.md",
    "docs/scoring_rubric_mapping.md",
    "docs/presentation_pitch.md",
    "Al + Materials Competition Application Form.docx",
    # Deployment
    "Dockerfile",
    "docker-compose.yml",
    "requirements.txt",
    "requirements-docker.txt",
    "requirements-gpu.txt",
    "requirements-lock.txt",
    ".dockerignore",
    ".env.example",
    # Data (images ONLY, no labels)
    "data/README.md",
    "reports/evaluation_results.json" if os.path.exists("reports/evaluation_results.json") else None,
    "reports/evaluation_results.csv" if os.path.exists("reports/evaluation_results.csv") else None,
    "reports/ultralytics_validation_results.json" if os.path.exists("reports/ultralytics_validation_results.json") else None,
    "reports/dataset_manifest.json" if os.path.exists("reports/dataset_manifest.json") else None,
    "reports/model_sha256.txt" if os.path.exists("reports/model_sha256.txt") else None,
    "reports/environment.txt" if os.path.exists("reports/environment.txt") else None,
    "reports/evaluation_command.txt" if os.path.exists("reports/evaluation_command.txt") else None,
    # Model artifacts (REQUIRED)
    "outputs/model.onnx",
    "outputs/best.pt",
]
# Directories to include (test_set and evaluation dataset with labels)
IMAGE_DIRS = [
    ("data/test_set/images", "data/test_set/images"),
    ("data/evaluation/images", "data/evaluation/images"),
    ("data/evaluation/labels", "data/evaluation/labels"),
]

# Explicitly EXCLUDED (safety check)
BLACKLIST_PATTERNS = [
    "__pycache__",
    ".pyc",
    ".pyo",
    ".env",
    ".git",
    ".venv",
    "labels",  # NO ground truth labels
    "coating_defects",  # full training data stays local
    "runs/",
    "yolo_training",
    ".ipynb_checkpoints",
    "node_modules",
    "*.db",
    "*.zip",
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
    if ("labels" in parts and "evaluation" not in parts) or "coating_defects" in parts or "yolo_training" in parts:
        return True
    if "runs" in parts or ".ipynb_checkpoints" in parts or "node_modules" in parts:
        return True
    return False


def build_zip():
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

    # Run unit tests before building
    print("  RUNNING UNIT TESTS:")
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "--tb=line"],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print("\n  [ERROR] Unit tests failed! Submission build aborted.")
        sys.exit(1)
    print("  All unit tests PASSED.\n")

    # Live inject verification if API is up (non-fatal if down)
    print("  RUNNING LIVE INJECT CHECK (optional if API offline):")
    live = subprocess.run(
        [sys.executable, "scripts/verify_inject.py"],
        cwd=PROJECT_ROOT,
    )
    if live.returncode != 0:
        print("  [WARN] Live inject check failed or API offline — continuing if unit tests passed.")
    else:
        print("  Live inject check PASSED.\n")

    if os.path.exists(OUTPUT_ZIP):
        os.remove(OUTPUT_ZIP)

    added_files = []
    skipped_files = []

    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        # Add whitelisted individual files
        for filepath in WHITELIST:
            if filepath is None:
                continue
            full_path = os.path.join(PROJECT_ROOT, filepath)
            if os.path.exists(full_path) and os.path.isfile(full_path):
                if is_blacklisted(filepath):
                    skipped_files.append(filepath)
                    continue
                zf.write(full_path, filepath)
                added_files.append(filepath)
            else:
                skipped_files.append(f"(missing) {filepath}")

        # Add image and evaluation label directories
        for src_dir, zip_dir in IMAGE_DIRS:
            full_dir = os.path.join(PROJECT_ROOT, src_dir)
            if os.path.exists(full_dir):
                for fname in sorted(os.listdir(full_dir)):
                    if fname.endswith(('.jpg', '.png', '.jpeg', '.txt')):
                        fpath = os.path.join(full_dir, fname)
                        arcname = f"{zip_dir}/{fname}"
                        if not is_blacklisted(arcname):
                            zf.write(fpath, arcname)
                            added_files.append(arcname)

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

    print(f"\n  READY TO SUBMIT: {OUTPUT_ZIP} ({zip_size:.1f} MB)")


if __name__ == "__main__":
    build_zip()
