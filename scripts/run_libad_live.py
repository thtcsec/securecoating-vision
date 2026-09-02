"""Run the LIBAD adapter live: probe, demo, 10-split fixture benchmark, optional API.

Official 4.84 GB archives stay gated until Hugging Face terms are accepted.
This script never starts that download. comparable_to_paper stays false.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from libad.dataset import dataset_status  # noqa: E402
from libad.demo_cases import run_all_demo_cases  # noqa: E402
from libad.evaluate import evaluate_official_splits  # noqa: E402
from libad.official_code import official_code_status  # noqa: E402
from libad.protocol import OFFICIAL_SPLIT_SEEDS  # noqa: E402


def _free_tcp_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _probe() -> Dict[str, Any]:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "download_libad.py"), "--probe"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    return {
        "exit_code": result.returncode,
        "stdout": (result.stdout or "").strip(),
        "stderr": (result.stderr or "").strip(),
        "gated": result.returncode == 2,
    }


def _wait_health(client: httpx.Client, timeout_seconds: float = 180.0) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            response = client.get("/health")
            if response.status_code in {200, 503}:
                return response.json()
            last_error = f"HTTP {response.status_code}"
        except Exception as exc:  # noqa: BLE001 - startup race
            last_error = str(exc)
        time.sleep(0.5)
    raise TimeoutError(f"API did not become reachable: {last_error}")


def _run_api_demo(host: str, port: int) -> Dict[str, Any]:
    workdir = Path(tempfile.mkdtemp(prefix="securecoating_libad_live_"))
    env = os.environ.copy()
    env.update(
        {
            "SECURECOATING_ENV": "development",
            "SECURECOATING_ALLOW_UNAUTHENTICATED_DEMO": "true",
            "SECURECOATING_ENABLE_SENSOR_SIMULATION": "true",
            "SECURECOATING_DB_PATH": str(workdir / "quality.db"),
            "SECURECOATING_INSPECTION_ARTIFACT_DIR": str(workdir / "artifacts"),
            "SECURECOATING_DATASET_CATALOG_CACHE": str(workdir / "catalog.json"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    log_path = workdir / "uvicorn.log"
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.api.main:app",
            "--host",
            host,
            "--port",
            str(port),
            "--workers",
            "1",
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    payload: Dict[str, Any] = {
        "api": f"http://{host}:{port}",
        "workdir": str(workdir),
        "pid": process.pid,
        "cases": {},
    }
    try:
        with httpx.Client(base_url=payload["api"], timeout=60.0) as client:
            payload["health"] = _wait_health(client)
            protocol = client.get("/api/libad/protocol")
            payload["protocol"] = {
                "http_status": protocol.status_code,
                "body": protocol.json() if protocol.headers.get("content-type", "").startswith("application/json") else protocol.text[:500],
            }
            for case_id in (1, 2, 3, 4):
                response = client.get(f"/api/libad/demo/{case_id}")
                body = response.json()
                payload["cases"][str(case_id)] = {
                    "http_status": response.status_code,
                    "action": body.get("decision", {}).get("action") if isinstance(body, dict) else None,
                    "comparable_to_paper": body.get("comparable_to_paper") if isinstance(body, dict) else None,
                }
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        log_handle.close()
        payload["uvicorn_log"] = str(log_path)
    return payload


def _compact_experiments(report: Dict[str, Any]) -> Dict[str, Any]:
    compact = {}
    for name, payload in report.get("experiments", {}).items():
        academic = payload.get("academic") or {}
        item = {
            "n_splits": payload.get("n_splits"),
            "auroc_mean": (academic.get("auroc") or {}).get("mean"),
        }
        if "industrial" in payload:
            industrial = payload["industrial"]
            item["n_pass_mean"] = (industrial.get("n_pass") or {}).get("mean")
            item["hold_rate_mean"] = (industrial.get("hold_rate") or {}).get("mean")
        compact[name] = item
    return compact


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-official-code", action="store_true")
    parser.add_argument("--with-api", action="store_true")
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=0)
    parser.add_argument(
        "--seeds",
        default="",
        help="Optional comma-separated official seeds. Default is all 10.",
    )
    parser.add_argument(
        "--report",
        default="outputs/libad_live_run.json",
        help="JSON evidence path. Default is gitignored under outputs/",
    )
    parser.add_argument(
        "--write-reports",
        action="store_true",
        help="Also refresh reports/libad/*.json and reports/libad_demo/",
    )
    args = parser.parse_args(argv)

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)

    seeds = OFFICIAL_SPLIT_SEEDS
    if args.seeds.strip():
        seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())

    report: Dict[str, Any] = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparable_to_paper": False,
        "evidence_class_expected": "protocol_fixture unless official archives are mounted",
        "steps": {},
    }

    report["steps"]["dataset_status"] = dataset_status()
    report["steps"]["probe"] = _probe()
    report["steps"]["official_code_before"] = official_code_status()

    if args.fetch_official_code:
        fetch = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "fetch_libad_official_code.py")],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        report["steps"]["fetch_official_code"] = {
            "exit_code": fetch.returncode,
            "stdout": (fetch.stdout or "").strip(),
            "stderr": (fetch.stderr or "").strip(),
        }
    report["steps"]["official_code_after"] = official_code_status()

    demo = run_all_demo_cases()
    report["steps"]["demo"] = {
        str(item["case_id"]): {
            "name": item["case_name"],
            "action": item["decision"]["action"],
            "expected": item["expected_action"],
            "comparable_to_paper": item["comparable_to_paper"],
        }
        for item in demo
    }
    demo_ok = [item["decision"]["action"] for item in demo] == ["PASS", "REJECT", "REJECT", "HOLD"]
    report["steps"]["demo"]["script_ok"] = demo_ok

    benchmark = evaluate_official_splits(seeds=seeds, allow_fixture=True)
    serializable = dict(benchmark)
    serializable.pop("predictions", None)
    report["steps"]["benchmark"] = {
        "evidence_class": benchmark["evidence_class"],
        "comparable_to_paper": benchmark["comparable_to_paper"],
        "official_protocol_complete": benchmark["official_protocol_complete"],
        "feature_backbone": benchmark["feature_backbone"],
        "n_predictions": len(benchmark["predictions"]),
        "latency_ms_total": benchmark["latency_ms_total"],
        "experiments": _compact_experiments(benchmark),
        "paper_comparability_blockers": benchmark["paper_comparability_blockers"],
    }

    if args.write_reports:
        refresh = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_libad_benchmark.py"),
                "--seeds",
                ",".join(str(seed) for seed in seeds),
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        demo_refresh = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "run_libad_demo.py")],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        report["steps"]["write_reports"] = {
            "benchmark_exit": refresh.returncode,
            "demo_exit": demo_refresh.returncode,
        }

    if args.with_api:
        port = int(args.api_port or _free_tcp_port(args.api_host))
        report["steps"]["api"] = _run_api_demo(args.api_host, port)

    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "probe_exit": report["steps"]["probe"]["exit_code"],
        "demo": {k: v["action"] for k, v in report["steps"]["demo"].items() if k.isdigit()},
        "benchmark_evidence_class": report["steps"]["benchmark"]["evidence_class"],
        "comparable_to_paper": False,
        "official_code_present": report["steps"]["official_code_after"]["present"],
        "api_cases": report["steps"].get("api", {}).get("cases"),
    }, indent=2))

    if not demo_ok:
        return 2
    if benchmark["comparable_to_paper"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
