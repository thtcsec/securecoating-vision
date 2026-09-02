"""Obtain or report the official LIBAD release. Never silently download 4.84 GB.

Low-I/O defaults: HTTP (not Xet), cache on the destination volume, one archive at
a time, extract then delete zips, BelowNormal process priority, no tree hash.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from libad.protocol import LIBAD_CITATION  # noqa: E402

HF_DATASET_ID = "Evenrose/LIBAD"
ARCHIVE_FILES = ("splits.zip", "LIBAD.zip")  # tiny file first
GATED_EXIT = 2
LICENSE_EXIT = 3
DEPENDENCY_EXIT = 4
SPACE_EXIT = 5
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
EXTRACT_PROGRESS_EVERY = 250
HEADROOM_SLACK_BYTES = 2 * 1024 * 1024 * 1024
RETRY_DELAYS_SECONDS = (15, 30, 45, 60, 90, 120)

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

  python scripts/record_libad_official_manifest.py
  python scripts/run_libad_benchmark.py --require-official --out reports/libad/official_local_adapter.json

The fixture report at reports/libad/libad_benchmark.json stays protocol_fixture.
Official local-adapter numbers belong in official_local_adapter.json and remain
comparable_to_paper=false. Paper comparison requires the authors' DINOv3/DA-Core
runner, not this numpy adapter.

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


def configure_low_io_env() -> None:
    """Keep Hub I/O on one volume and disable parallel disk writers."""
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"


def lower_process_priority() -> bool:
    if os.name != "nt":
        return False
    try:
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        return bool(
            ctypes.windll.kernel32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS)
        )
    except OSError:
        return False


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
    configure_low_io_env()
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


def _required_headroom_bytes(probe: Dict[str, Any]) -> int:
    zip_total = 0
    for info in (probe.get("files") or {}).values():
        size = info.get("size")
        if size:
            zip_total += int(size)
    # Peak is zip-on-disk plus extracted tree before the archive is deleted.
    return int(zip_total * 2) + HEADROOM_SLACK_BYTES


def require_disk_headroom(dest_root: Path, needed: int) -> int:
    dest_root.mkdir(parents=True, exist_ok=True)
    free = int(shutil.disk_usage(dest_root).free)
    if free < needed:
        raise DownloadError(
            f"Refusing download: {free} free bytes, need {needed} "
            "(zip + extract + slack, same volume).",
            SPACE_EXIT,
        )
    return free


def _download_one(hf_hub_download, auth, incoming: Path, filename: str, probe: Dict[str, Any]) -> str:
    expected = (probe.get("files") or {}).get(filename, {}).get("size")
    local_zip = incoming / filename
    if local_zip.is_file() and expected and local_zip.stat().st_size == int(expected):
        print(f"Reusing complete {filename} ({expected} bytes)", flush=True)
        return str(local_zip)
    last_error: Optional[BaseException] = None
    delays = (0,) + tuple(RETRY_DELAYS_SECONDS)
    for attempt, delay in enumerate(delays, 1):
        if delay:
            print(f"Waiting {delay}s before retry {attempt - 1} of {filename}", flush=True)
            time.sleep(delay)
        try:
            print(
                f"HTTP get {filename} (attempt {attempt}/{len(delays)}, resume if Hub kept the temp)",
                flush=True,
            )
            path = hf_hub_download(
                repo_id=HF_DATASET_ID,
                filename=filename,
                repo_type="dataset",
                token=auth,
                local_dir=str(incoming),
            )
            return str(path)
        except Exception as exc:  # noqa: BLE001 - Hub raises httpx/httpcore on drops
            if "Gated" in type(exc).__name__:
                raise
            code = getattr(getattr(exc, "response", None), "status_code", None)
            if code in {401, 403}:
                raise
            last_error = exc
            print(f"{filename} attempt {attempt} failed: {exc}", flush=True)
    raise DownloadError(
        f"Failed to download {filename} after {len(delays)} attempts: {last_error}",
        1,
    ) from last_error


def download_official_archives(
    dest_root: Path,
    token: Optional[str] = None,
    incoming_dir: Optional[Path] = None,
    cleanup_archives: bool = True,
    light_status: bool = True,
) -> Dict[str, Any]:
    """Download and extract LIBAD.zip plus splits.zip into dest_root."""
    configure_low_io_env()
    lower_process_priority()
    probe = probe_official_archives(token=token)
    needed = _required_headroom_bytes(probe)
    free = require_disk_headroom(dest_root, needed)
    print(
        f"Low-I/O download: Xet off, BelowNormal, sequential archives. "
        f"free={free} needed={needed}",
        flush=True,
    )
    _, _, hf_hub_download, GatedRepoError, HfHubHTTPError = _import_hub()
    dest_root = Path(dest_root)
    incoming = Path(incoming_dir or dest_root / "_incoming")
    incoming.mkdir(parents=True, exist_ok=True)
    auth = token if token else False
    saved: Dict[str, str] = {}
    splits_dir = dest_root / "splits"
    try:
        if splits_dir.is_dir() and any(splits_dir.iterdir()):
            print("splits/ already present; skipping splits.zip", flush=True)
            saved["splits.zip"] = str(splits_dir)
        else:
            print("Downloading splits.zip first (small sanity check)", flush=True)
            saved["splits.zip"] = _download_one(
                hf_hub_download, auth, incoming, "splits.zip", probe
            )
            print("Extracting splits.zip", flush=True)
            _safe_extract(Path(saved["splits.zip"]), dest_root)
            if cleanup_archives:
                Path(saved["splits.zip"]).unlink(missing_ok=True)
                print("Deleted splits.zip after extract", flush=True)

        print("Downloading LIBAD.zip (large, single HTTP stream)", flush=True)
        saved["LIBAD.zip"] = _download_one(
            hf_hub_download, auth, incoming, "LIBAD.zip", probe
        )
        print("Extracting LIBAD.zip sequentially", flush=True)
        _safe_extract(Path(saved["LIBAD.zip"]), dest_root)
        if cleanup_archives:
            Path(saved["LIBAD.zip"]).unlink(missing_ok=True)
            print("Deleted LIBAD.zip after extract", flush=True)
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

    dataset_dir = dest_root / "LIBAD"
    splits_dir = dest_root / "splits"
    if not dataset_dir.is_dir():
        raise DownloadError(f"LIBAD.zip did not produce {dataset_dir}")
    if not splits_dir.is_dir():
        raise DownloadError(f"splits.zip did not produce {splits_dir}")
    if cleanup_archives:
        _cleanup_incoming(incoming, {})
    return {
        **probe,
        "downloaded": saved,
        "dataset_root": str(dataset_dir),
        "splits_root": str(splits_dir),
        "cleanup_archives": cleanup_archives,
        "light_status": light_status,
    }


def extract_official_archives(libad_zip: Path, splits_zip: Path, dest_root: Path) -> Dict[str, str]:
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {splits_zip.name} (small)", flush=True)
    _safe_extract(splits_zip, dest_root)
    print(f"Extracting {libad_zip.name} (large, sequential members)", flush=True)
    _safe_extract(libad_zip, dest_root)
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
        infos = handle.infolist()
        for index, info in enumerate(infos, 1):
            target = (dest_root / info.filename).resolve()
            if dest_root not in target.parents and target != dest_root:
                raise DownloadError(f"Refusing zip path outside destination: {info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.is_file() and target.stat().st_size == info.file_size:
                continue
            handle.extract(info, dest_root)
            if index % EXTRACT_PROGRESS_EVERY == 0 or index == len(infos):
                print(f"  {archive.name}: {index}/{len(infos)} members", flush=True)


def _cleanup_incoming(incoming: Path, saved: Dict[str, str]) -> None:
    for path in saved.values():
        zip_path = Path(path)
        if zip_path.is_file():
            zip_path.unlink()
            print(f"Deleted archive {zip_path.name} after extract", flush=True)
    cache = incoming / ".cache"
    if cache.exists():
        shutil.rmtree(cache, ignore_errors=True)
    if incoming.exists() and not any(incoming.iterdir()):
        incoming.rmdir()


def write_instructions(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    readme = target / "README.md"
    if not readme.exists():
        readme.write_text(INSTRUCTIONS + "\n", encoding="utf-8")


def print_status(light: bool = False) -> None:
    from libad.dataset import dataset_status, list_official_samples
    from libad.protocol import official_libad_present

    if light:
        present = official_libad_present()
        splits = PROJECT_ROOT / "data" / "libad" / "splits"
        print("\nOfficial mount status (light, no tree hash)")
        print(f"  present: {present}")
        print(f"  splits_dir: {splits.is_dir()}")
        print("  comparable_to_paper: False")
        print("  skipped full sample index and SHA-256 tree to avoid a disk stampede")
        return
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
    parser.add_argument(
        "--full-status",
        action="store_true",
        help="Walk the mounted tree after download. Avoid immediately after extract.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep zip files after extract (uses extra disk).",
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
                result = download_official_archives(
                    target,
                    token=huggingface_token(),
                    cleanup_archives=not args.keep_archives,
                    light_status=not args.full_status,
                )
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
            print_status(light=args.download and not args.full_status)
    except DownloadError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return exc.exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
