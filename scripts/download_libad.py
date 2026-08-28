"""Record how to obtain the official LIBAD release. Does not silently download 4.84 GB."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from libad.protocol import LIBAD_CITATION  # noqa: E402


INSTRUCTIONS = f"""
LIBAD official dataset (CC BY 4.0)
Paper: {LIBAD_CITATION['url']}
Dataset: {LIBAD_CITATION['dataset_url']}
Code: {LIBAD_CITATION['code_url']}

This adapter never claims DA-Core as a SecureCoating-Vision algorithm.

Place the extracted archives at:

  data/libad/LIBAD/
  data/libad/splits/

Expected layout:

  data/libad/LIBAD/1_wrinkling/{{normal,anomaly}}/*{{A,B,X,L}}.tiff
  data/libad/splits/<official split files for seeds {', '.join(str(s) for s in (347, 725, 1245, 4012, 4589, 5021, 5678, 6234, 6789, 7345))}>

Then:

  python scripts/run_libad_benchmark.py --require-official

Without the official 4.84 GB release, the harness runs a protocol fixture and
labels the report evidence_class=protocol_fixture, comparable_to_paper=false.
""".strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Show LIBAD download/layout instructions.")
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="Acknowledge CC BY 4.0. This script still does not auto-download 4.84 GB.",
    )
    args = parser.parse_args()
    print(INSTRUCTIONS)
    target = PROJECT_ROOT / "data" / "libad"
    target.mkdir(parents=True, exist_ok=True)
    readme = target / "README.md"
    if not readme.exists():
        readme.write_text(INSTRUCTIONS + "\n", encoding="utf-8")
    if args.accept_license:
        print("\nLicense acknowledgement recorded locally. Download the archives from Hugging Face manually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
