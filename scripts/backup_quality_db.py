"""Create a consistent online backup of the traceability SQLite database."""

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traceability.quality_memory import QualityMemory


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=os.environ.get("SECURECOATING_DB_PATH", "data/quality_history.db"),
    )
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    memory = QualityMemory(args.source)
    if not memory.backup_to(args.destination):
        print(f"Backup failed: {memory.last_error}", file=sys.stderr)
        return 1
    print(f"Consistent SQLite backup created: {os.path.abspath(args.destination)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
