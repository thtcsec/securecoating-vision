"""Run the LIBAD 10-split validation-extension harness."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from libad.evaluate import evaluate_official_splits  # noqa: E402
from libad.protocol import OFFICIAL_SPLIT_SEEDS  # noqa: E402

BELOW_NORMAL_PRIORITY_CLASS = 0x00004000


def _low_io_priority() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if os.name != "nt":
        return
    try:
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS)
    except OSError:
        return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-official", action="store_true")
    parser.add_argument("--seeds", default="", help="Optional comma-separated official seeds.")
    parser.add_argument(
        "--out",
        default="reports/libad/libad_benchmark.json",
        help="JSON report path.",
    )
    args = parser.parse_args()
    _low_io_priority()
    seeds = OFFICIAL_SPLIT_SEEDS
    if args.seeds.strip():
        seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
        unknown = [seed for seed in seeds if seed not in OFFICIAL_SPLIT_SEEDS]
        if unknown:
            raise SystemExit(f"Non-official seeds are refused: {unknown}")
    report = evaluate_official_splits(
        seeds=seeds,
        allow_fixture=not args.require_official,
    )
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(report)
    if out_path.name == "libad_benchmark.json":
        predictions_path = out_path.with_name("libad_predictions.json")
    else:
        predictions_path = PROJECT_ROOT / "outputs" / f"{out_path.stem}_predictions.json"
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(json.dumps(report["predictions"], indent=2), encoding="utf-8")
    serializable["predictions_path"] = str(predictions_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    serializable["predictions"] = f"{len(report['predictions'])} rows written to predictions file"
    out_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(json.dumps({k: serializable[k] for k in ("evidence_class", "comparable_to_paper", "experiments", "hashes")}, indent=2))
    print(f"Wrote {out_path}")
    if args.require_official and report["evidence_class"] == "protocol_fixture":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
