"""Protocol constants, attribution, and hash helpers for the LIBAD adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

OFFICIAL_SPLIT_SEEDS = (347, 725, 1245, 4012, 4589, 5021, 5678, 6234, 6789, 7345)

LIBAD_CITATION = {
    "title": "LIBAD: A Multimodal Anomaly Detection Benchmark for Li-Ion Battery Electrode Manufacturing",
    "authors": ["Wenbo Sui", "Daniel Lichau", "Harold Phelippeau", "Zhao Liu"],
    "year": 2026,
    "arxiv": "2608.07958",
    "url": "https://arxiv.org/abs/2608.07958",
    "dataset_url": "https://huggingface.co/datasets/Evenrose/LIBAD",
    "code_url": "https://github.com/evenrose/LIBAD",
    "license": "CC BY 4.0",
    "da_core_attribution": (
        "DA-Core is the density-aware coreset baseline published by the LIBAD authors. "
        "SecureCoating-Vision does not claim DA-Core as an original algorithm."
    ),
}

LIBAD_PAPER_RESULT_NOTE = (
    "Sui et al. report that the best LIBAD setting still has FPR95 of 54.3% at "
    "AUROC 86.7%, AUPR 95.7%, and F1-max 90.6%. The authors conclude that this "
    "false-positive rate is too high for direct deployment and call out modality "
    "disagreement and closed-loop process control as open problems. "
    "LIBAD addresses how to detect; SecureCoating-Vision addresses when a "
    "detection is safe enough to act on."
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


SKIP_TREE_DIR_NAMES = ("_preview64", "_incoming", ".cache")


def _tree_files(root: Path) -> list[Path]:
    files = []
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        if any(part in SKIP_TREE_DIR_NAMES for part in item.relative_to(root).parts):
            continue
        files.append(item)
    return files


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path, progress: bool = False) -> Optional[str]:
    """Content hash for a directory tree, including normalized relative paths."""
    root = Path(path)
    if not root.is_dir():
        return None
    files = _tree_files(root)
    if not files:
        return None
    digest = hashlib.sha256()
    for index, item in enumerate(files, 1):
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
        if progress and (index % 100 == 0 or index == len(files)):
            print(f"  hashed {index}/{len(files)} {root.name}", flush=True)
    return digest.hexdigest()


def tree_stat_fingerprint(path: Path) -> Optional[Dict[str, int]]:
    """Cheap integrity check: file count and total bytes. Does not read contents."""
    root = Path(path)
    if not root.is_dir():
        return None
    files = _tree_files(root)
    if not files:
        return None
    return {
        "n_files": len(files),
        "total_bytes": int(sum(item.stat().st_size for item in files)),
    }


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256_bytes(encoded)


def load_yaml(relative_path: str) -> Dict[str, Any]:
    path = PROJECT_ROOT / relative_path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_project_identity() -> Dict[str, Any]:
    return load_yaml("configs/project_identity.yaml")


def load_libad_config() -> Dict[str, Any]:
    return load_yaml("configs/libad.yaml")


def git_commit_sha(repo_root: Optional[Path] = None) -> str:
    root = repo_root or PROJECT_ROOT
    git_dir = root / ".git"
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=False,
            capture_output=True,
            text=True,
        )
        sha = (result.stdout or "").strip()
        if result.returncode == 0 and len(sha) == 40:
            return sha
    except OSError:
        pass
    if git_dir.is_file():
        return "unknown"
    return "unknown"


def git_source_provenance(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Identify the HEAD commit and any uncommitted implementation state.

    Generated reports must not imply that dirty source came from a clean commit.
    Report artifacts are deliberately excluded to avoid self-referential hashes.
    """
    root = repo_root or PROJECT_ROOT
    pathspec = [
        "src", "scripts", "configs", "dashboard",
        ":(exclude)scripts/record_test_manifest.py",
        ":(exclude)scripts/build_submission.py",
        "Dockerfile", "docker-compose.yml", "requirements.txt",
        "requirements-core.txt", "requirements-docker.txt",
        "requirements-gpu.txt", "requirements-lock.txt",
    ]
    try:
        import subprocess

        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all", "--", *pathspec],
            cwd=str(root),
            check=False,
            capture_output=True,
        )
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--", *pathspec],
            cwd=str(root),
            check=False,
            capture_output=True,
        )
        if status.returncode != 0 or diff.returncode != 0:
            raise OSError("git provenance command failed")
        status_bytes = status.stdout or b""
        digest = hashlib.sha256()
        digest.update(status_bytes)
        digest.update(diff.stdout or b"")
        for raw_line in status_bytes.splitlines():
            line = raw_line.decode("utf-8", errors="surrogateescape")
            if line.startswith("?? "):
                untracked = root / line[3:]
                if untracked.is_file():
                    digest.update(line[3:].encode("utf-8", errors="surrogateescape"))
                    digest.update(bytes.fromhex(sha256_file(untracked)))
        dirty = bool(status_bytes.strip())
        return {
            "commit": git_commit_sha(root),
            "working_tree_dirty": dirty,
            "source_diff_sha256": digest.hexdigest() if dirty else None,
        }
    except OSError:
        return {
            "commit": git_commit_sha(root),
            "working_tree_dirty": None,
            "source_diff_sha256": None,
        }


def hash_existing_files(paths: Iterable[str]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for relative in paths:
        path = PROJECT_ROOT / relative
        out[relative.replace("\\", "/")] = sha256_file(path) if path.is_file() else None
    return out


def _mean_std(block: Any, name: str) -> Optional[Dict[str, float]]:
    if not isinstance(block, dict):
        return None
    item = block.get(name)
    if not isinstance(item, dict) or "mean" not in item:
        return None
    payload = {"mean": float(item["mean"])}
    if item.get("std") is not None:
        payload["std"] = float(item["std"])
    return payload


def official_local_adapter_summary() -> Optional[Dict[str, Any]]:
    """Checked-in numpy 10-seed summary. Never a paper-table claim."""
    path = PROJECT_ROOT / "reports" / "libad" / "official_local_adapter.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if payload.get("comparable_to_paper") is True:
        return None
    experiments = payload.get("experiments") if isinstance(payload.get("experiments"), dict) else {}

    def academic(name: str) -> Dict[str, Any]:
        block = (experiments.get(name) or {}).get("academic") if isinstance(experiments.get(name), dict) else {}
        out: Dict[str, Any] = {}
        for metric in ("auroc", "aupr", "f1_max", "fpr95"):
            parsed = _mean_std(block, metric)
            if parsed:
                out[metric] = parsed
        return out

    industrial = (
        (experiments.get("securecoating_gate") or {}).get("industrial")
        if isinstance(experiments.get("securecoating_gate"), dict)
        else {}
    )
    gate: Dict[str, Any] = {}
    if isinstance(industrial, dict):
        for key in (
            "hold_rate",
            "escape_rate",
            "selective_risk",
            "automatic_decision_coverage",
        ):
            parsed = _mean_std(industrial, key)
            if parsed:
                gate[key] = parsed
    return {
        "source": "reports/libad/official_local_adapter.json",
        "evidence_class": payload.get("evidence_class"),
        "comparable_to_paper": False,
        "feature_backbone": payload.get("feature_backbone"),
        "official_protocol_complete": bool(payload.get("official_protocol_complete")),
        "unimodal_vis": academic("unimodal_vis"),
        "unimodal_xray_l": academic("unimodal_xray_l"),
        "multimodal": academic("multimodal"),
        "securecoating_gate": gate,
    }


def dataset_root_from_config(config: Optional[Dict[str, Any]] = None) -> Path:
    cfg = config or load_libad_config()
    return PROJECT_ROOT / cfg["paths"]["dataset_root"]


def splits_root_from_config(config: Optional[Dict[str, Any]] = None) -> Path:
    cfg = config or load_libad_config()
    return PROJECT_ROOT / cfg["paths"]["splits_root"]


def official_libad_present(config: Optional[Dict[str, Any]] = None) -> bool:
    root = dataset_root_from_config(config)
    if not root.is_dir():
        return False
    groups = [path for path in root.iterdir() if path.is_dir()]
    return len(groups) >= 1
