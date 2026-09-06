"""Status of an optional clone of the authors' evenrose/LIBAD runner.

Cloning that repository does not make local scores paper-comparable. The authors'
DINOv3/DA-Core path still needs their 4.84 GB dataset, their weights, and their
runtime. This module only records whether the source tree is present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from libad.protocol import PROJECT_ROOT, git_commit_sha

OFFICIAL_CODE_URL = "https://github.com/evenrose/LIBAD.git"
OFFICIAL_CODE_RELATIVE = "third_party/evenrose-libad"


def official_code_root(repo_root: Optional[Path] = None) -> Path:
    return (repo_root or PROJECT_ROOT) / OFFICIAL_CODE_RELATIVE


def official_code_status(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    root = official_code_root(repo_root)
    present = root.is_dir() and any(root.iterdir())
    commit = None
    if present:
        sha = git_commit_sha(root)
        commit = sha if sha and sha != "unknown" else None
    return {
        # Runtime checkout on this machine (may be true in a developer tree).
        "present": bool(present),
        # Historical/run-time checkout used when an experiment was recorded.
        "present_at_run": bool(present),
        # Submission ZIP never packs third_party/evenrose-libad (gitignored / blacklisted).
        "present_in_submission": False,
        "path": OFFICIAL_CODE_RELATIVE,
        "path_at_run": OFFICIAL_CODE_RELATIVE if present else None,
        "fetch_script": "scripts/fetch_libad_official_code.py",
        "url": OFFICIAL_CODE_URL,
        "commit": commit,
        "runnable_as_paper_baseline": False,
        "note": (
            "evenrose/LIBAD belongs to Sui et al. Presence of their source does not "
            "make this repository's numpy adapter comparable to the published table. "
            "Textual PAPER_SPEC is DINOv3 ViT-S/16 + DA-FPS + max-NN via "
            "scripts/run_libad_official_baseline.py --paper-config. Common upstream "
            "ConvNeXt-base defaults are official-code core (ADAPTED), not PAPER_EXACT; "
            "ConvNeXt also needs HF gated-weight access and the hash-verified 4.84 GB mount. "
            "The submission archive does not include third_party/evenrose-libad; re-fetch "
            "with scripts/fetch_libad_official_code.py when needed."
        ),
    }


def submission_official_code_meta(code_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Normalize official_code blocks written into checked-in / packed reports."""
    base = dict(code_meta or {})
    present_at_run = bool(base.get("present_at_run", base.get("present", False)))
    commit = base.get("commit")
    return {
        "present": present_at_run,  # backward-compatible: means present when the run was recorded
        "present_at_run": present_at_run,
        "present_in_submission": False,
        "path": OFFICIAL_CODE_RELATIVE,
        "path_at_run": OFFICIAL_CODE_RELATIVE if present_at_run else None,
        "fetch_script": "scripts/fetch_libad_official_code.py",
        "url": base.get("url") or OFFICIAL_CODE_URL,
        "commit": commit,
        "runnable_as_paper_baseline": False,
        "note": base.get("note")
        or (
            "evenrose/LIBAD belongs to Sui et al. Recorded as present_at_run when the "
            "experiment executed; not packed in the submission ZIP. Re-fetch with "
            "scripts/fetch_libad_official_code.py for local reproduction."
        ),
    }
