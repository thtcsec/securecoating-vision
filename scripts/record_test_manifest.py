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
SOURCE_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from libad.protocol import git_source_provenance  # noqa: E402

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
    source_provenance = git_source_provenance(PROJECT_ROOT)
    # Tests verify that the checked-in evidence refers to the exact source tree
    # under test. Refresh only those provenance fields before pytest so the
    # validation does not depend on a stale manifest from an earlier run.
    try:
        preflight_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        preflight_manifest = {}
    preflight_manifest.update(
        {
            "working_tree_dirty": source_provenance["working_tree_dirty"],
            "source_diff_sha256": source_provenance["source_diff_sha256"],
        }
    )
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(preflight_manifest, indent=2), encoding="utf-8")
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
        "working_tree_dirty": source_provenance["working_tree_dirty"],
        "source_diff_sha256": source_provenance["source_diff_sha256"],
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
        f"(commit `{manifest['commit_sha'][:12]}`, working tree "
        f"{'dirty' if manifest['working_tree_dirty'] else 'clean'}, "
        f"source diff `{(manifest['source_diff_sha256'] or 'none')[:12]}`, "
        f"Python {manifest['python_version']}, "
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
        f"working_tree_dirty {manifest['working_tree_dirty']}\n"
        f"source_diff_sha256 {manifest['source_diff_sha256']}\n"
        f"log_sha256 {manifest['log_sha256']}\n"
        "```"
    )
    _replace_marked_block(PROJECT_ROOT / "README.md", readme_body)
    readme_cn_body = (
        f"当前软件验证快照为 {passed} 项测试通过、{skipped} 项跳过、{failed} 项失败"
        f"（提交 `{manifest['commit_sha'][:12]}`，工作树"
        f"{'脏' if manifest['working_tree_dirty'] else '干净'}，"
        f"源码差异 `{(manifest['source_diff_sha256'] or 'none')[:12]}`，"
        f"Python {manifest['python_version']}，耗时 {manifest['duration_seconds']} 秒）。"
        f"权威记录为 [reports/test_manifest.json](reports/test_manifest.json)。"
        "该记录仅证明当前软件测试结果，不代表工厂性能、真实 PLC 行为、"
        "安全等级急停认证或生产资质。"
    )
    _replace_marked_block(PROJECT_ROOT / "README_CN.md", readme_cn_body)
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
