"""Compatibility entry point for the authoritative real-data validation.

The top-level checkpoint is a two-class CoatingVision detector. This wrapper
forwards to ``evaluate_coatingvision_real.py`` so the model, test split,
manifest hashes, class map, and output report cannot silently diverge.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_coatingvision_real.py"),
        "--weights",
        str(ROOT / "outputs" / "best.pt"),
        "--imgsz",
        "512",
        "--device",
        "0",
    ]
    raise SystemExit(subprocess.call(command, cwd=ROOT))
