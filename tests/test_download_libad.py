"""Official LIBAD download is explicit, license-gated, and never silent."""

import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.download_libad as download_libad


class _FakeMetadata:
    size = 12
    commit_hash = "abc123"


class TestDownloadLibad(unittest.TestCase):
    def test_download_without_license_is_refused(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(sys, "stdout", stdout), patch.object(sys, "stderr", stderr):
            code = download_libad.main(["--download"])
        self.assertEqual(code, download_libad.LICENSE_EXIT)
        self.assertIn("accept-license", stderr.getvalue().lower())

    def test_probe_maps_gated_access_to_exit_2(self):
        def boom(*_args, **_kwargs):
            raise download_libad.DownloadError("gated dataset", download_libad.GATED_EXIT)

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(download_libad, "probe_official_archives", side_effect=boom):
            with patch.object(sys, "stdout", stdout), patch.object(sys, "stderr", stderr):
                code = download_libad.main(["--probe"])
        self.assertEqual(code, download_libad.GATED_EXIT)
        self.assertIn("gated dataset", stderr.getvalue())

    def test_extracts_official_zip_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            libad_zip = root / "LIBAD.zip"
            splits_zip = root / "splits.zip"
            dest = root / "libad"
            with zipfile.ZipFile(libad_zip, "w") as handle:
                handle.writestr("LIBAD/1_wrinkling/normal/unitA.tiff", b"vis")
            with zipfile.ZipFile(splits_zip, "w") as handle:
                handle.writestr("splits/347.json", b"{}")
            layout = download_libad.extract_official_archives(libad_zip, splits_zip, dest)
            self.assertTrue((dest / "LIBAD" / "1_wrinkling" / "normal" / "unitA.tiff").is_file())
            self.assertTrue((dest / "splits" / "347.json").is_file())
            self.assertEqual(Path(layout["dataset_root"]), dest / "LIBAD")

    def test_status_only_does_not_download(self):
        stdout = io.StringIO()
        with patch.object(download_libad, "download_official_archives") as download:
            with patch.object(download_libad, "probe_official_archives") as probe:
                with patch.object(download_libad, "print_status"):
                    with patch.object(sys, "stdout", stdout):
                        code = download_libad.main(["--status"])
        self.assertEqual(code, 0)
        download.assert_not_called()
        probe.assert_not_called()

    def test_headroom_refuses_when_disk_is_too_small(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            with patch.object(download_libad.shutil, "disk_usage") as usage:
                usage.return_value = type("Usage", (), {"free": 10})()
                with self.assertRaises(download_libad.DownloadError) as raised:
                    download_libad.require_disk_headroom(dest, needed=100)
        self.assertEqual(raised.exception.exit_code, download_libad.SPACE_EXIT)

    def test_download_one_retries_transient_failures(self):
        calls = {"n": 0}

        def flaky(*_args, **_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("peer closed")
            return "ok.zip"

        with tempfile.TemporaryDirectory() as directory, patch.object(
            download_libad, "RETRY_DELAYS_SECONDS", (0,)
        ):
            path = download_libad._download_one(
                flaky, False, Path(directory), "LIBAD.zip", {"files": {}}
            )
        self.assertEqual(path, "ok.zip")
        self.assertEqual(calls["n"], 2)

    def test_cleanup_removes_archives_after_extract(self):
        with tempfile.TemporaryDirectory() as directory:
            incoming = Path(directory) / "incoming"
            incoming.mkdir()
            zip_path = incoming / "LIBAD.zip"
            zip_path.write_bytes(b"zip")
            download_libad._cleanup_incoming(incoming, {"LIBAD.zip": str(zip_path)})
            self.assertFalse(zip_path.exists())
            self.assertFalse(incoming.exists())


if __name__ == "__main__":
    unittest.main()
