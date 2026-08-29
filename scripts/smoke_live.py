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
    parser.add_argument(
        "--image",
        default="data/demo_real/images/image_1548.jpg",
        help="Attributed real optical image to upload.",
    )
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

    catalog_response = c.get("/api/dataset/catalog", params={"offset": 0, "limit": 24})
    catalog_response.raise_for_status()
    catalog = catalog_response.json()
    if not catalog.get("provenance_verified"):
        raise RuntimeError("Demo dataset provenance/hash verification failed")
    print(
        "DATASET",
        catalog.get("dataset_name"),
        catalog.get("dataset_license"),
        f"samples={catalog.get('total')}",
    )
    first_sample = (catalog.get("items") or [None])[0]
    if not first_sample:
        raise RuntimeError("Verified demo catalog is empty")
    preview = c.get(f"/api/dataset/images/{first_sample['filename']}")
    preview.raise_for_status()
    if preview.headers.get("x-dataset-sha256") != first_sample.get("sha256"):
        raise RuntimeError("Dataset preview hash header does not match catalog")
    if not preview.content.startswith(b"\xff\xd8\xff"):
        raise RuntimeError("Dataset preview is not a valid JPEG stream")
    print("DATASET_PREVIEW", first_sample["filename"], len(preview.content), "bytes", "HASH_OK")

    t0 = time.time()
    with open(args.image, "rb") as f:
        up = c.post(
            "/api/inspect",
            data={
                "batch_id": active_batch,
                "part_id": f"SMOKE_REAL_{uuid.uuid4().hex[:10]}",
            },
            files={"image": (os.path.basename(args.image), f, "image/jpeg")},
        )
    up.raise_for_status()
    result = up.json()
    http_ms = (time.time() - t0) * 1000
    print(
        "INSPECTION",
        result["run_id"],
        result["engine"],
        f"infer={result['latency_ms']:.1f}ms",
        f"detections={len(result['defects_found'])}",
        f"gate={result['gate_action']}",
        f"passed={result['passed']}",
        f"http={http_ms:.0f}ms",
    )
    for view in ("raw", "input", "overlay"):
        artifact = c.get(
            f"/api/inspections/{result['run_id']}/image",
            params={"view": view},
        )
        artifact.raise_for_status()
        if artifact.headers.get("content-type", "").split(";", 1)[0] != "image/jpeg":
            raise RuntimeError(f"{view} artifact has an unexpected content type")
        if not artifact.content.startswith(b"\xff\xd8\xff"):
            raise RuntimeError(f"{view} artifact is not a valid JPEG stream")
        print("ARTIFACT", view, len(artifact.content), "bytes", "JPEG_OK")
    print("spc", c.get(f"/api/batch/{active_batch}/spc").json()["status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
