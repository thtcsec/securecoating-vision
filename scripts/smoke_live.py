"""Bounded live smoke check against an already-running API."""
import argparse
import os
import sys
import time
import uuid
import httpx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default=os.environ.get("SECURECOATING_API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.environ.get("SECURECOATING_API_KEY", ""))
    args = parser.parse_args()
    headers = {"x-api-key": args.api_key} if args.api_key else {}
    c = httpx.Client(base_url=args.api_url, timeout=120.0, headers=headers)
    health_response = c.get("/health")
    if health_response.status_code not in {200, 503}:
        health_response.raise_for_status()
    h = health_response.json()
    print("HEALTH", h["status"], h["primary_engine"], h["device"])

    active = c.get("/api/roll/active")
    active.raise_for_status()
    active_batch = active.json()["summary"]["batch_id"]

    samples_response = c.get("/api/samples")
    samples_response.raise_for_status()
    samples = samples_response.json()["samples"][:2]
    latencies = []
    for i, name in enumerate(samples):
        t0 = time.time()
        r = c.post(
            "/api/inspect",
            data={
                "batch_id": active_batch,
                "part_id": f"SMOKE_{i}_{uuid.uuid4().hex[:10]}",
                "sample_name": name,
                "thermal_online": "true",
                "profiler_online": "false",
            },
        )
        r.raise_for_status()
        b = r.json()
        http_ms = (time.time() - t0) * 1000
        latencies.append(b["latency_ms"])
        dets = len(b["defects_found"])
        print(
            f"{name}: {r.status_code} {b['engine']} "
            f"infer={b['latency_ms']:.1f}ms dets={dets} "
            f"pass={b['passed']} fallback={b['fallback_active']} http={http_ms:.0f}ms"
        )
    if latencies:
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        print(f"observed_samples={len(samples)} p50_infer_ms={p50:.1f} (not a benchmark)")
    print("spc", c.get(f"/api/batch/{active_batch}/spc").json()["status"])
    print("estop", c.post("/api/industrial/estop", data={"reason": "smoke"}).json())
    print("reset", c.post("/api/industrial/reset").json())
    # upload path
    with open("data/test_set/images/defect_val_00000.jpg", "rb") as f:
        up = c.post(
            "/api/inspect",
            data={"batch_id": active_batch, "part_id": f"SMOKE_UPLOAD_{uuid.uuid4().hex[:10]}"},
            files={"image": ("x.jpg", f, "image/jpeg")},
        )
    up.raise_for_status()
    print("upload", up.status_code, up.json()["engine"], len(up.json()["defects_found"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
