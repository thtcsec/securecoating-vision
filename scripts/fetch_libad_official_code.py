"""Clone the authors' evenrose/LIBAD source tree. Does not download 4.84 GB data."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from libad.official_code import (  # noqa: E402
    OFFICIAL_CODE_RELATIVE,
    OFFICIAL_CODE_URL,
    official_code_root,
    official_code_status,
)


def _git(args: Sequence[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        capture_output=True,
        text=True,
    )


def fetch_official_code(dest: Optional[Path] = None, update: bool = False) -> dict:
    dest = dest or official_code_root()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and any(dest.iterdir()):
        if update:
            pull = _git(["pull", "--ff-only"], cwd=dest)
            if pull.returncode != 0:
                raise RuntimeError(pull.stderr.strip() or "git pull failed")
        status = official_code_status()
        status["updated"] = bool(update)
        status["cloned"] = False
        return status
    clone = _git(["clone", "--depth", "1", OFFICIAL_CODE_URL, str(dest)])
    if clone.returncode != 0:
        raise RuntimeError(clone.stderr.strip() or f"git clone {OFFICIAL_CODE_URL} failed")
    status = official_code_status()
    status["updated"] = False
    status["cloned"] = True
    return status


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="git pull if the clone already exists")
    parser.add_argument("--status", action="store_true", help="report clone status without fetching")
    args = parser.parse_args(argv)
    if args.status:
        status = official_code_status()
        print(
            f"official_code present={status['present']} "
            f"path={status['path']} commit={status['commit']}"
        )
        return 0
    try:
        status = fetch_official_code(update=args.update)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        f"official_code cloned={status.get('cloned')} present={status['present']} "
        f"commit={status['commit']} comparable_to_paper=false"
    )
    print(
        "This clone is the authors' runner, not a SecureCoating-Vision algorithm. "
        "It cannot produce paper-comparable numbers without the gated 4.84 GB dataset."
    )
    print(f"Checkout: {PROJECT_ROOT / OFFICIAL_CODE_RELATIVE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
