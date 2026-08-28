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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def hash_existing_files(paths: Iterable[str]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for relative in paths:
        path = PROJECT_ROOT / relative
        out[relative.replace("\\", "/")] = sha256_file(path) if path.is_file() else None
    return out


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
