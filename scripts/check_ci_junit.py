"""Fail CI on test failures or unexplained skips; allow artifact-gated skips."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ALLOWED_SKIP_MARKERS = (
    "missing",
    "not present",
    "not found",
)


def _suite_nodes(root: ET.Element) -> list[ET.Element]:
    if root.tag == "testsuites":
        return list(root)
    return [root]


def unexpected_skip_names(junit_path: Path) -> list[str]:
    root = ET.parse(junit_path).getroot()
    unexpected = []
    for suite in _suite_nodes(root):
        for case in suite.iter("testcase"):
            skipped = case.find("skipped")
            if skipped is None:
                continue
            reason = f"{skipped.get('message', '')} {skipped.text or ''}".lower()
            if any(marker in reason for marker in ALLOWED_SKIP_MARKERS):
                continue
            classname = case.get("classname", "")
            name = case.get("name", "")
            unexpected.append(f"{classname}.{name}: {reason.strip() or 'no skip reason'}")
    return unexpected


def failure_count(junit_path: Path) -> int:
    root = ET.parse(junit_path).getroot()
    total = 0
    for suite in _suite_nodes(root):
        total += int(suite.attrib.get("failures", "0"))
        total += int(suite.attrib.get("errors", "0"))
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("junit_path")
    args = parser.parse_args()
    path = Path(args.junit_path)
    if not path.is_file():
        print(f"JUnit report missing: {path}", file=sys.stderr)
        return 2
    failed = failure_count(path)
    if failed:
        print(f"{failed} failing/errored tests in {path}", file=sys.stderr)
        return 1
    unexpected = unexpected_skip_names(path)
    if unexpected:
        print("Unexpected skipped tests:", file=sys.stderr)
        for item in unexpected:
            print(f"  {item}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
