"""Record official LIBAD dataset/splits tree hashes once.

Default API/pytest status checks a cheap file-count/byte fingerprint against
this manifest. Live SHA-256 rehash of 4.84 GB is opt-in:

  SECURECOATING_LIBAD_VERIFY_TREES=1
  or dataset_status(verify_trees=True)

comparable_to_paper stays false. This does not run DINOv3/DA-Core.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from libad.dataset import dataset_status  # noqa: E402
from libad.protocol import (  # noqa: E402
    OFFICIAL_SPLIT_SEEDS,
    load_libad_config,
    sha256_tree,
    tree_stat_fingerprint,
)

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


def record_official_manifest(verify_after: bool = False) -> dict:
    cfg = load_libad_config()
    dataset_root = PROJECT_ROOT / cfg["paths"]["dataset_root"]
    splits_root = PROJECT_ROOT / cfg["paths"]["splits_root"]
    if not dataset_root.is_dir() or not splits_root.is_dir():
        raise SystemExit(
            f"Official LIBAD mount missing. Expected {dataset_root} and {splits_root}."
        )
    print(f"Fingerprinting {dataset_root} ...", flush=True)
    dataset_fp = tree_stat_fingerprint(dataset_root)
    print(f"Fingerprinting {splits_root} ...", flush=True)
    splits_fp = tree_stat_fingerprint(splits_root)
    if dataset_fp is None or splits_fp is None:
        raise SystemExit("Official LIBAD trees are empty after skipping cache dirs.")
    print(
        f"Hashing dataset tree ({dataset_fp['n_files']} files, "
        f"{dataset_fp['total_bytes']} bytes) ...",
        flush=True,
    )
    dataset_hash = sha256_tree(dataset_root, progress=True)
    print(
        f"Hashing splits tree ({splits_fp['n_files']} files, "
        f"{splits_fp['total_bytes']} bytes) ...",
        flush=True,
    )
    splits_hash = sha256_tree(splits_root, progress=True)
    if not dataset_hash or not splits_hash:
        raise SystemExit("Tree hashing failed.")
    payload = {
        "dataset_tree_sha256": dataset_hash,
        "splits_tree_sha256": splits_hash,
        "dataset_n_files": dataset_fp["n_files"],
        "dataset_total_bytes": dataset_fp["total_bytes"],
        "splits_n_files": splits_fp["n_files"],
        "splits_total_bytes": splits_fp["total_bytes"],
        "official_split_seeds": list(OFFICIAL_SPLIT_SEEDS),
        "comparable_to_paper": False,
        "feature_backbone": cfg["features"]["backbone"],
        "hashed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": (
            "Local numpy_patch_descriptor hashes of the official CC BY 4.0 mount. "
            "This is not a paper-comparable DINOv3/DA-Core run."
        ),
    }
    runtime_path = PROJECT_ROOT / cfg["paths"]["official_artifact_manifest"]
    reports_path = PROJECT_ROOT / "reports" / "libad" / "official_mount_hashes.json"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    reports_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2) + "\n"
    runtime_path.write_text(encoded, encoding="utf-8")
    reports_path.write_text(encoded, encoding="utf-8")
    print(f"Wrote {runtime_path}")
    print(f"Wrote {reports_path}")
    status = dataset_status(verify_trees=verify_after)
    print(
        json.dumps(
            {
                "official_protocol_complete": status["official_protocol_complete"],
                "official_artifact_manifest_verified": status[
                    "official_artifact_manifest_verified"
                ],
                "tree_hash_verified_live": status["tree_hash_verified_live"],
                "comparable_to_paper": status["comparable_to_paper"],
                "mounted_sample_count": status["mounted_sample_count"],
            },
            indent=2,
        )
    )
    payload["status"] = {
        "official_protocol_complete": status["official_protocol_complete"],
        "official_artifact_manifest_verified": status["official_artifact_manifest_verified"],
        "tree_hash_verified_live": status["tree_hash_verified_live"],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-trees",
        action="store_true",
        help="Re-hash after writing. Slow on the 4.84 GB mount; default is fingerprint only.",
    )
    args = parser.parse_args()
    _low_io_priority()
    record_official_manifest(verify_after=args.verify_trees)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
