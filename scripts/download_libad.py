"""Obtain or report the official LIBAD release. Never silently download 4.84 GB."""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from libad.protocol import LIBAD_CITATION  # noqa: E402

HF_DATASET_ID = "Evenrose/LIBAD"
ARCHIVE_FILES = ("LIBAD.zip", "splits.zip")
GATED_EXIT = 2
LICENSE_EXIT = 3
DEPENDENCY_EXIT = 4

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

Download is gated on Hugging Face. This script does not start the 4.84 GB
transfer unless you pass --download --accept-license. A Hugging Face token
with accepted dataset terms is required; unauthenticated requests receive 401.
huggingface_hub is an optional dependency: pip install huggingface_hub
""".strip()


class DownloadError(RuntimeError):
    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


def huggingface_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    return token.strip() if token and token.strip() else None


def _import_hub():
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
    except ImportError as exc:
        raise DownloadError(
            "huggingface_hub is not installed. pip install huggingface_hub",
            DEPENDENCY_EXIT,
        ) from exc
    try:
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    except ImportError:  # pragma: no cover - older hub layouts
        from huggingface_hub.utils import GatedRepoError, HfHubHTTPError
    return hf_hub_url, get_hf_file_metadata, hf_hub_download, GatedRepoError, HfHubHTTPError


def probe_official_archives(token: Optional[str] = None) -> Dict[str, Any]:
    """HEAD the official zip files. Does not download archive bodies."""
    hf_hub_url, get_hf_file_metadata, _, GatedRepoError, HfHubHTTPError = _import_hub()
    auth = token if token else False
    files: Dict[str, Any] = {}
    try:
        for filename in ARCHIVE_FILES:
            url = hf_hub_url(HF_DATASET_ID, filename=filename, repo_type="dataset")
            metadata = get_hf_file_metadata(url, token=auth)
            files[filename] = {
                "accessible": True,
                "size": getattr(metadata, "size", None),
                "commit_hash": getattr(metadata, "commit_hash", None),
            }
    except GatedRepoError as exc:
        raise DownloadError(
            "Official LIBAD archives are gated. Accept the dataset terms on "
            f"{LIBAD_CITATION['dataset_url']} and set HF_TOKEN. ({exc})",
            GATED_EXIT,
        ) from exc
    except HfHubHTTPError as exc:
        code = getattr(getattr(exc, "response", None), "status_code", None)
        exit_code = GATED_EXIT if code in {401, 403} else 1
        raise DownloadError(f"Hugging Face rejected LIBAD metadata: {exc}", exit_code) from exc
    return {
        "repo_id": HF_DATASET_ID,
        "gated": False,
        "authenticated": bool(token),
        "files": files,
    }


def download_official_archives(
    dest_root: Path,
    token: Optional[str] = None,
    incoming_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Download and extract LIBAD.zip plus splits.zip into dest_root."""
    probe = probe_official_archives(token=token)
    _, _, hf_hub_download, GatedRepoError, HfHubHTTPError = _import_hub()
    dest_root = Path(dest_root)
    incoming = Path(incoming_dir or dest_root / "_incoming")
    incoming.mkdir(parents=True, exist_ok=True)
    auth = token if token else False
    saved: Dict[str, str] = {}
    try:
        for filename in ARCHIVE_FILES:
            path = hf_hub_download(
                repo_id=HF_DATASET_ID,
                filename=filename,
                repo_type="dataset",
                token=auth,
                local_dir=str(incoming),
            )
            saved[filename] = str(path)
    except GatedRepoError as exc:
        raise DownloadError(
            "Official LIBAD download is gated until Hugging Face terms are accepted "
            f"and HF_TOKEN is set. ({exc})",
            GATED_EXIT,
        ) from exc
    except HfHubHTTPError as exc:
        code = getattr(getattr(exc, "response", None), "status_code", None)
        exit_code = GATED_EXIT if code in {401, 403} else 1
        raise DownloadError(f"Hugging Face rejected LIBAD download: {exc}", exit_code) from exc
    layout = extract_official_archives(
        Path(saved["LIBAD.zip"]),
        Path(saved["splits.zip"]),
        dest_root,
    )
    return {**probe, "downloaded": saved, **layout}


def extract_official_archives(libad_zip: Path, splits_zip: Path, dest_root: Path) -> Dict[str, str]:
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    _safe_extract(libad_zip, dest_root)
    _safe_extract(splits_zip, dest_root)
    dataset_dir = dest_root / "LIBAD"
    splits_dir = dest_root / "splits"
    if not dataset_dir.is_dir():
        raise DownloadError(f"LIBAD.zip did not produce {dataset_dir}")
    if not splits_dir.is_dir():
        raise DownloadError(f"splits.zip did not produce {splits_dir}")
    return {
        "dataset_root": str(dataset_dir),
        "splits_root": str(splits_dir),
    }


def _safe_extract(archive: Path, dest_root: Path) -> None:
    dest_root = dest_root.resolve()
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            target = (dest_root / info.filename).resolve()
            if dest_root not in target.parents and target != dest_root:
                raise DownloadError(f"Refusing zip path outside destination: {info.filename}")
        handle.extractall(dest_root)


def write_instructions(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    readme = target / "README.md"
    if not readme.exists():
        readme.write_text(INSTRUCTIONS + "\n", encoding="utf-8")


def print_status() -> None:
    from libad.dataset import dataset_status, list_official_samples

    status = dataset_status()
    samples = list_official_samples(offset=0, limit=0)
    print("\nOfficial mount status")
    print(f"  present: {status.get('official_dataset_present')}")
    print(f"  protocol_complete: {status.get('official_protocol_complete')}")
    print(f"  comparable_to_paper: {status.get('comparable_to_paper')}")
    print(f"  mounted_complete_triples: {samples.get('total')}")
    for blocker in status.get("comparability_blockers") or []:
        print(f"  blocker: {blocker}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Show or fetch official LIBAD archives.")
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="Acknowledge CC BY 4.0 before any archive download.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download LIBAD.zip and splits.zip after license acknowledgement. ~4.84 GB.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="HEAD the official zip files without downloading archive bodies.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print whether the official release is mounted. Does not download files.",
    )
    args = parser.parse_args(argv)
    target = PROJECT_ROOT / "data" / "libad"
    write_instructions(target)
    if not (args.probe or args.download or args.status or args.accept_license):
        print(INSTRUCTIONS)
        return 0
    try:
        if args.download and not args.accept_license:
            raise DownloadError(
                "Refusing to download 4.84 GB without --accept-license.",
                LICENSE_EXIT,
            )
        if args.probe or args.download:
            if args.download:
                result = download_official_archives(target, token=huggingface_token())
                print("\nDownloaded and extracted official LIBAD archives.")
            else:
                result = probe_official_archives(token=huggingface_token())
                print("\nOfficial LIBAD zip metadata is accessible.")
            for filename, info in (result.get("files") or {}).items():
                print(f"  {filename}: size={info.get('size')} commit={info.get('commit_hash')}")
            if result.get("dataset_root"):
                print(f"  dataset_root: {result['dataset_root']}")
                print(f"  splits_root: {result['splits_root']}")
        elif args.accept_license:
            print(
                "\nLicense acknowledgement recorded locally. "
                "Re-run with --download --accept-license after Hugging Face access is granted."
            )
        if args.status or args.download:
            print_status()
    except DownloadError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return exc.exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
