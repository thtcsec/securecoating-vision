"""Thorough live verification: inject samples, multipart upload, degrade, dashboard."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
API = "http://127.0.0.1:8000"
DASH = "http://127.0.0.1:8501"
errors: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        errors.append(name)


def main() -> int:
    c = httpx.Client(base_url=API, timeout=120.0)

    try:
        h = c.get("/health").json()
        check(
            "health",
            h.get("status") == "HEALTHY" and h.get("yolo_available") is True,
            str(h),
        )
    except Exception as exc:
        check("health", False, str(exc))
        print("API down — abort remaining inspect tests")
        print("FAILS", errors)
        return 1

    s = c.get("/api/samples").json()
    check("samples_list", s.get("count", 0) >= 40, f"count={s.get('count')}")

    class_imgs = {
        "scratch": "defect_val_00000.jpg",
        "void": "defect_val_00001.jpg",
        "blister": "defect_val_00002.jpg",
        "delamination": "defect_val_00003.jpg",
    }
    for cls, name in class_imgs.items():
        r = c.post(
            "/api/inspect",
            data={
                "batch_id": "INJECT_CHK",
                "part_id": f"CLS_{cls}",
                "sample_name": name,
                "thermal_online": "true",
                "profiler_online": "true",
            },
        )
        body = r.json() if r.status_code == 200 else {}
        names = {d["class_name"] for d in body.get("defects_found", [])}
        check(
            f"sample_inject_{cls}",
            r.status_code == 200 and cls in names and str(body.get("run_id", "")).startswith("RUN_"),
            f"status={r.status_code} classes={names} eng={body.get('engine')} "
            f"ms={body.get('latency_ms')} run={body.get('run_id')}",
        )

    img_path = ROOT / "data" / "test_set" / "images" / "defect_val_00000.jpg"
    with img_path.open("rb") as f:
        r = c.post(
            "/api/inspect",
            data={"batch_id": "INJECT_CHK", "part_id": "UPLOAD1"},
            files={"image": (img_path.name, f, "image/jpeg")},
        )
    body = r.json() if r.status_code == 200 else {}
    dets = body.get("defects_found", [])
    check(
        "multipart_jpeg_upload",
        r.status_code == 200 and len(dets) >= 1 and dets[0].get("confidence") is not None,
        f"n={len(dets)} conf={dets[0].get('confidence') if dets else None}",
    )

    # second upload format: PNG
    canvas = np.random.randint(140, 200, (640, 640, 3), dtype=np.uint8)
    cv2.line(canvas, (40, 80), (500, 100), (20, 20, 20), 10)
    ok_enc, buf = cv2.imencode(".png", canvas)
    assert ok_enc
    r = c.post(
        "/api/inspect",
        data={"batch_id": "INJECT_CHK", "part_id": "SYNTH_PNG"},
        files={"image": ("synth.png", buf.tobytes(), "image/png")},
    )
    check("multipart_png_upload", r.status_code == 200, f"status={r.status_code}")

    r = c.post(
        "/api/inspect",
        data={
            "batch_id": "INJECT_CHK",
            "part_id": "DEG",
            "sample_name": "defect_val_00000.jpg",
            "thermal_online": "false",
            "profiler_online": "false",
        },
    )
    body = r.json()
    check(
        "degraded_fallback",
        r.status_code == 200 and body.get("fallback_active") is True,
        f"fallback={body.get('fallback_active')} status={body.get('status')}",
    )

    lat = c.get("/api/metrics/latency", params={"batch_id": "INJECT_CHK"}).json()
    check("latency_metrics", lat.get("count", 0) > 0 and "p50_ms" in lat, str(lat))
    stats = c.get("/api/batch/INJECT_CHK/stats").json()
    check("batch_stats", stats.get("total", 0) > 0, f"total={stats.get('total')}")

    r404 = c.post("/api/inspect", data={"sample_name": "nope_missing.jpg"})
    check("missing_sample_404", r404.status_code == 404, f"status={r404.status_code}")

    rbad = c.post(
        "/api/inspect",
        files={"image": ("bad.bin", b"notanimage", "application/octet-stream")},
    )
    check("bad_upload_rejected", rbad.status_code in (400, 422), f"status={rbad.status_code}")

    try:
        d = httpx.get(DASH, timeout=10.0)
        check("dashboard_up", d.status_code == 200 and len(d.text) > 500, f"len={len(d.text)}")
    except Exception as exc:
        check("dashboard_up", False, str(exc))

    # local predictor path (same stack dashboard uses when API toggle off)
    sys.path.insert(0, str(ROOT / "src"))
    import yaml
    from inference.predictor import CoatingPredictor

    cfg = yaml.safe_load((ROOT / "configs" / "model.yaml").read_text(encoding="utf-8"))
    pred = CoatingPredictor(cfg)
    image = cv2.imread(str(img_path))
    out = pred.predict(image, thermal=None, height=None)
    check(
        "local_predictor_inject",
        out.get("detections") and "YOLO" in str(out.get("engine", "")),
        f"engine={out.get('engine')} n={len(out.get('detections', []))}",
    )

    print("---")
    if errors:
        print(f"FAILS ({len(errors)}): {errors}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
