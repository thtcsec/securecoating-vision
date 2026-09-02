"""Run live Modbus TCP triggers through the real API, not pytest mocks.

Starts a localhost software gateway, boots uvicorn with mock_mode=false, then
fires RESET, one inspection (fail-closed HOLD), and E-stop over TCP. This is
not vendor PLC HIL and not a factory-qualified gate.
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
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from industrial.modbus_loopback import EVIDENCE_CLASS, SoftwareModbusGateway  # noqa: E402


def _free_tcp_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


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


def _control(client: httpx.Client, action: str, confirmation: str, reason: str, snapshot_id: str = "") -> dict:
    response = client.post(
        "/api/operations/control",
        data={
            "action": action,
            "operator_id": "OP_LOOPBACK",
            "confirmation": confirmation,
            "reason": reason,
            "snapshot_id": snapshot_id,
            "idempotency_key": f"CMD_LOOPBACK_{action}_{uuid.uuid4().hex[:10]}",
        },
    )
    payload = response.json()
    payload["_http_status"] = response.status_code
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=0, help="0 selects an ephemeral port")
    parser.add_argument(
        "--image",
        default="data/demo_real/images/image_822.jpg",
        help="Attributed real optical image for the inspection trigger",
    )
    parser.add_argument(
        "--report",
        default="outputs/modbus_loopback_live_trigger.json",
        help="JSON evidence path. Default is gitignored under outputs/",
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Leave API and Modbus gateway running until Ctrl+C",
    )
    args = parser.parse_args(argv)

    image_path = (PROJECT_ROOT / args.image).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Inspection image is missing: {image_path}")

    api_port = int(args.api_port or _free_tcp_port(args.api_host))
    workdir = Path(tempfile.mkdtemp(prefix="securecoating_modbus_live_"))
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)

    gateway = SoftwareModbusGateway(host="127.0.0.1")
    process = None
    client = None
    log_handle = None
    report = {
        "evidence_class": EVIDENCE_CLASS,
        "vendor_hil": False,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "api": f"http://{args.api_host}:{api_port}",
        "modbus": None,
        "steps": [],
    }
    try:
        gateway.start()
        report["modbus"] = {
            "host": gateway.host,
            "port": gateway.port,
            "command_register": gateway.command_register,
            "ack_register": gateway.ack_register,
        }
        env = os.environ.copy()
        env.update(
            {
                "SECURECOATING_ENV": "development",
                "SECURECOATING_ALLOW_UNAUTHENTICATED_DEMO": "true",
                "SECURECOATING_ENABLE_SENSOR_SIMULATION": "true",
                "SECURECOATING_INDUSTRIAL_MOCK_MODE": "false",
                "SECURECOATING_COMMAND_CHANNEL": "modbus",
                "SECURECOATING_PLC_IP": gateway.host,
                "SECURECOATING_MODBUS_PORT": str(gateway.port),
                "SECURECOATING_MODBUS_TRUSTED_GATEWAY": "true",
                "SECURECOATING_ACK_TIMEOUT_SECONDS": "1.5",
                "SECURECOATING_INDUSTRIAL_EVIDENCE_CLASS": EVIDENCE_CLASS,
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
                args.api_host,
                "--port",
                str(api_port),
                "--workers",
                "1",
                "--log-level",
                "info",
            ],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        report["uvicorn_log"] = str(log_path)
        client = httpx.Client(base_url=f"http://{args.api_host}:{api_port}", timeout=120.0)
        health = _wait_health(client)
        report["steps"].append({"name": "health", "payload": health})
        state = client.get("/api/industrial/state").json()
        report["steps"].append({"name": "state_before_reset", "payload": state})
        if state.get("connection", {}).get("mock_mode"):
            raise RuntimeError("API still has industrial mock_mode=true; loopback was not wired")
        if not state.get("connection", {}).get("transport_ready"):
            raise RuntimeError(f"Modbus transport is not ready: {state}")

        snapshot = client.get("/api/operations/snapshot", params={"scope": "live"}).json()
        reset = _control(
            client,
            "RESET",
            "CONFIRM RESET",
            "software loopback reset after unverified startup latch",
            snapshot["snapshot_id"],
        )
        report["steps"].append({"name": "reset", "payload": reset})
        if not reset.get("acknowledged") or reset.get("status") != "ACKNOWLEDGED":
            raise RuntimeError(f"RESET was not TCP-acknowledged: {reset}")

        active = client.get("/api/roll/active").json()
        batch_id = active["summary"]["batch_id"]
        with image_path.open("rb") as handle:
            inspect = client.post(
                "/api/inspect",
                data={
                    "batch_id": batch_id,
                    "part_id": f"LOOPBACK_{uuid.uuid4().hex[:10]}",
                },
                files={"image": (image_path.name, handle, "image/jpeg")},
            )
        inspect_body = inspect.json()
        inspect_body["_http_status"] = inspect.status_code
        report["steps"].append({"name": "inspect", "payload": inspect_body})
        if inspect.status_code not in {200, 503}:
            raise RuntimeError(f"Inspection HTTP {inspect.status_code}: {inspect_body}")

        estop = _control(
            client,
            "EMERGENCY_STOP",
            "CONFIRM EMERGENCY_STOP",
            "software loopback emergency stop after live TCP sequence",
        )
        report["steps"].append({"name": "emergency_stop", "payload": estop})
        if not estop.get("acknowledged"):
            raise RuntimeError(f"E-stop was not TCP-acknowledged: {estop}")

        signals = client.get("/api/industrial/signals", params={"limit": 20}).json()
        state_after = client.get("/api/industrial/state").json()
        report["steps"].append({"name": "signals", "payload": signals})
        report["steps"].append({"name": "state_after", "payload": state_after})
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["ok"] = True
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(
            {
                "ok": True,
                "evidence_class": EVIDENCE_CLASS,
                "vendor_hil": False,
                "api": report["api"],
                "modbus": report["modbus"],
                "reset": reset.get("status"),
                "inspect_gate": inspect_body.get("gate_action"),
                "estop": estop.get("status"),
                "report": str(report_path),
            },
            indent=2,
        ))
        if args.keep_alive:
            print("Keep-alive: Ctrl+C to stop the API and Modbus gateway.", flush=True)
            process.wait()
        return 0
    except Exception as exc:
        report["ok"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        if log_handle is not None:
            log_handle.flush()
        log_path = workdir / "uvicorn.log"
        if log_path.is_file():
            report["uvicorn_tail"] = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"ok": False, "error": report["error"], "report": str(report_path)}, indent=2))
        return 1
    finally:
        if client is not None:
            client.close()
        if process is not None and process.poll() is None and not args.keep_alive:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        if log_handle is not None:
            log_handle.close()
        gateway.stop()


if __name__ == "__main__":
    raise SystemExit(main())
