"""Run pytest once and write the single test-count manifest used by docs/submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "reports" / "test_manifest.json"
MARKER_START = "<!-- TEST_MANIFEST:START -->"
MARKER_END = "<!-- TEST_MANIFEST:END -->"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    sha = (result.stdout or "").strip()
    return sha if result.returncode == 0 and sha else "unknown"


def _load_identity() -> dict:
    try:
        import yaml

        with (PROJECT_ROOT / "configs" / "project_identity.yaml").open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except Exception:
        return {"release_version": "2.0.0"}


def _replace_marked_block(path: Path, body: str) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER_START not in text or MARKER_END not in text:
        return
    start = text.index(MARKER_START)
    end = text.index(MARKER_END) + len(MARKER_END)
    replacement = f"{MARKER_START}\n{body.rstrip()}\n{MARKER_END}"
    path.write_text(text[:start] + replacement + text[end:], encoding="utf-8")


def record_test_manifest(pytest_args: list[str] | None = None) -> dict:
    junit_path = PROJECT_ROOT / "reports" / "pytest_junit.xml"
    junit_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "pytest", "-q", "--tb=line", f"--junitxml={junit_path}"]
    if pytest_args:
        command.extend(pytest_args)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
    log = (completed.stdout or "") + (completed.stderr or "")
    root = ET.parse(junit_path).getroot() if junit_path.is_file() else None
    # pytest may emit testsuites or testsuite as the root.
    suite = root
    if root is not None and root.tag == "testsuites":
        suites = list(root)
        collected = sum(int(item.attrib.get("tests", "0")) for item in suites)
        failures = sum(int(item.attrib.get("failures", "0")) for item in suites)
        errors = sum(int(item.attrib.get("errors", "0")) for item in suites)
        skipped = sum(int(item.attrib.get("skipped", "0")) for item in suites)
        duration = sum(float(item.attrib.get("time", "0")) for item in suites)
    elif root is not None:
        collected = int(root.attrib.get("tests", "0"))
        failures = int(root.attrib.get("failures", "0"))
        errors = int(root.attrib.get("errors", "0"))
        skipped = int(root.attrib.get("skipped", "0"))
        duration = float(root.attrib.get("time", "0"))
    else:
        collected = failures = errors = skipped = 0
        duration = 0.0
    failed = failures + errors
    passed = max(collected - failed - skipped, 0)
    identity = _load_identity()
    manifest = {
        "release_version": identity.get("release_version", "2.0.0"),
        "release_label": identity.get("release_label", ""),
        "commit_sha": _git_sha(),
        "python_version": platform.python_version(),
        "collected": collected,
        "passed": passed,
        "skipped": skipped,
        "failed": failed,
        "duration_seconds": round(duration, 2),
        "log_sha256": _sha256_bytes(log.encode("utf-8", errors="replace")),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "pytest_returncode": completed.returncode,
        "command": command,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    readme_body = (
        f"The current software validation snapshot is {passed} passing tests "
        f"with {skipped} skips and {failed} failures "
        f"(commit `{manifest['commit_sha'][:12]}`, Python {manifest['python_version']}, "
        f"{manifest['duration_seconds']}s). The authoritative record is "
        f"[reports/test_manifest.json](reports/test_manifest.json). "
        "This does not constitute evidence of factory performance, physical PLC behavior, "
        "safety-rated E-stop operation, or production qualification."
    )
    status_body = (
        "```text\n"
        f"{sys.executable} -m pytest -q\n"
        f"{passed} passed"
        + (f", {skipped} skipped" if skipped else "")
        + (f", {failed} failed" if failed else "")
        + f" in {manifest['duration_seconds']}s\n"
        f"python {manifest['python_version']}\n"
        f"commit {manifest['commit_sha']}\n"
        f"log_sha256 {manifest['log_sha256']}\n"
        "```"
    )
    _replace_marked_block(PROJECT_ROOT / "README.md", readme_body)
    _replace_marked_block(PROJECT_ROOT / "docs" / "implementation_status.md", status_body)
    print(json.dumps(manifest, indent=2))
    if completed.returncode != 0:
        sys.stderr.write(log)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pytest_args", nargs="*")
    args = parser.parse_args()
    manifest = record_test_manifest(args.pytest_args)
    return int(manifest["pytest_returncode"])


if __name__ == "__main__":
    raise SystemExit(main())
