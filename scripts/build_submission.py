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
    "src/inference/predictor.py",
    "src/inference/postprocess.py",
    "src/inference/sensor_fusion.py",
    "src/inference/failsafe.py",
    "src/industrial/__init__.py",
    "src/industrial/protocol_manager.py",
    "src/traceability/quality_memory.py",
    "src/training/train_yolo.py",
    "src/training/train_baseline.py",
    "src/evaluation/evaluate.py",
    "src/utils/__init__.py",
    # Dashboard
    "dashboard/app.py",
    # Configs
    "configs/app.yaml",
    "configs/model.yaml",
    "configs/dataset.yaml",
    # Scripts
    "scripts/run_evaluation.py",
    "scripts/prepare_real_dataset.py",
    "scripts/download_dataset.py",
    "scripts/build_submission.py",
    # Documentation
    "README.md",
    "README_CN.md",
    "LICENSE",
    "logo.png",
    "reports/analysis_report.md",
    "reports/analysis_report_template.md",
    "docs/system_architecture.md",
    "docs/inspection_workflow.md",
    "docs/scoring_rubric_mapping.md",
    # Deployment
    "Dockerfile",
    "docker-compose.yml",
    "requirements.txt",
    "requirements-docker.txt",
    ".dockerignore",
    # Data (images ONLY, no labels)
    "data/README.md",
    "data/sample_metadata.csv",
    "data/test_set/README.md",
    # Model artifact
    "outputs/model.onnx",
]

# Directories to include (images only from test_set)
IMAGE_DIRS = [
    ("data/test_set/images", "data/test_set/images"),
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
    for pattern in BLACKLIST_PATTERNS:
        if pattern in path:
            return True
    return False


def build_zip():
    print(f"Building submission ZIP: {OUTPUT_ZIP}")
    print(f"Project root: {PROJECT_ROOT}")
    print()

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

        # Add image directories (no labels!)
        for src_dir, zip_dir in IMAGE_DIRS:
            full_dir = os.path.join(PROJECT_ROOT, src_dir)
            if os.path.exists(full_dir):
                for fname in sorted(os.listdir(full_dir)):
                    if fname.endswith(('.jpg', '.png', '.jpeg')):
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
        has_labels = any("label" in n.lower() for n in all_names)
        has_pycache = any("__pycache__" in n for n in all_names)
        has_env = any(".env" in n and not n.endswith(".example") for n in all_names)
        has_gt = any("ground" in n.lower() or "annotation" in n.lower() for n in all_names)

        print(f"    Labels present:     {'FAIL' if has_labels else 'CLEAN'}")
        print(f"    __pycache__:        {'FAIL' if has_pycache else 'CLEAN'}")
        print(f"    .env secrets:       {'FAIL' if has_env else 'CLEAN'}")
        print(f"    Ground truth leak:  {'FAIL' if has_gt else 'CLEAN'}")

        if has_labels or has_pycache or has_env or has_gt:
            print("\n  WARNING: ZIP contains prohibited content!")
            sys.exit(1)
        else:
            print("\n  All safety checks PASSED.")

    print(f"\n  READY TO SUBMIT: {OUTPUT_ZIP} ({zip_size:.1f} MB)")


if __name__ == "__main__":
    build_zip()
