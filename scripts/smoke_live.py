"""Live smoke check against running API (uses project .venv)."""
import time
import httpx

API = "http://127.0.0.1:8000"


def main():
    c = httpx.Client(base_url=API, timeout=120.0)
    h = c.get("/health").json()
    print("HEALTH", h["status"], h["primary_engine"], h["device"])

    samples = c.get("/api/samples").json()["samples"][:8]
    ok = 0
    latencies = []
    for i, name in enumerate(samples):
        t0 = time.time()
        r = c.post(
            "/api/inspect",
            data={
                "batch_id": "CHK",
                "part_id": f"P{i}",
                "sample_name": name,
                "thermal_online": "true",
                "profiler_online": "false",
            },
        )
        b = r.json()
        http_ms = (time.time() - t0) * 1000
        latencies.append(b["latency_ms"])
        dets = len(b["defects_found"])
        print(
            f"{name}: {r.status_code} {b['engine']} "
            f"infer={b['latency_ms']:.1f}ms dets={dets} "
            f"pass={b['passed']} fallback={b['fallback_active']} http={http_ms:.0f}ms"
        )
        if r.status_code == 200 and dets >= 1:
            ok += 1

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    print(f"detect_ok {ok}/{len(samples)}  p50_infer_ms={p50:.1f}")
    print("spc", c.get("/api/batch/CHK/spc").json()["status"])
    print("estop", c.post("/api/industrial/estop", data={"reason": "smoke"}).json())
    print("reset", c.post("/api/industrial/reset").json())
    # upload path
    with open("data/test_set/images/defect_val_00000.jpg", "rb") as f:
        up = c.post(
            "/api/inspect",
            data={"batch_id": "CHK", "part_id": "UPLOAD"},
            files={"image": ("x.jpg", f, "image/jpeg")},
        )
    print("upload", up.status_code, up.json()["engine"], len(up.json()["defects_found"]))


if __name__ == "__main__":
    main()
