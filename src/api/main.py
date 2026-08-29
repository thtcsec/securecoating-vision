"""
SecureCoating-Vision: FastAPI REST API
=======================================
Research REST API for fail-closed multi-sensor coating inspection experiments.

Integrates:
- ONNX Runtime inference engine (YOLOv8-seg)
- Multi-source sensor fusion (RGB + Thermal + 3D Height)
- Fail-safe graceful degradation
- Industrial protocol signaling (OPC UA / Modbus TCP)
- Quality traceability memory (SQLite)

Endpoints:
- GET  /health                    -> System health check
- GET  /api/operations/snapshot   -> Single-timestamp operations snapshot
- POST /api/operations/control    -> Authenticated confirm-audit control
- POST /api/inspect               -> Run full inspection pipeline
- GET  /api/batch/{id}/stats      -> Batch quality statistics
- GET  /api/batch/{id}/spc        -> SPC alarm status
- GET  /api/industrial/state      -> PLC register state
- GET  /api/industrial/signals    -> Signal history
- POST /api/industrial/estop      -> Disabled legacy control (410)
- POST /api/industrial/reset      -> Disabled legacy control (410)
- GET  /api/system/health-report  -> Detailed system health
"""

import os
import sys
import time
import uuid
import secrets
import re
import asyncio
import base64
import hashlib
import json
import threading
from pathlib import Path
from datetime import datetime, timezone
from io import BytesIO
import yaml
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("SecureCoatingVision.API")

# Adjust pathing for module imports
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
# Ensure working directory is project root for config file access
PROJECT_ROOT = os.path.dirname(SRC_DIR)
os.chdir(PROJECT_ROOT)
from inference.predictor import CoatingPredictor
from inference.postprocess import extract_defects_from_mask, grade_coating
from inference.sensor_fusion import SensorFusionManager
from inference.failsafe import FailSafeManager
from traceability.quality_memory import QualityMemory
from industrial.protocol_manager import IndustrialProtocolManager
from industrial.web_synchronizer import WebSynchronizer, RollMetadata
from inference.electrode_metrology import ElectrodeMetrologyEngine
from inference.multi_stage_pipeline import MultiStageIndustrialPipeline
from traceability.root_cause_engine import RootCauseDiagnosticEngine
from traceability.roll_certificate import RollCertificateGenerator

# --- Application Setup ---
ENVIRONMENT = os.environ.get("SECURECOATING_ENV", "production").lower()
EXPOSE_API_DOCS = (
    ENVIRONMENT != "production"
    or os.environ.get("SECURECOATING_EXPOSE_DOCS", "false").lower() in {"1", "true", "yes"}
)
app = FastAPI(
    title="SecureCoating-Vision API",
    description=(
        "Fail-closed coating inspection API with RGB YOLO/ONNX, simulated "
        "thermal/profilometry adapters, and an optional LIBAD VIS+X-rayL evidence lane."
    ),
    version="2.0.0",
    docs_url="/docs" if EXPOSE_API_DOCS else None,
    redoc_url="/redoc" if EXPOSE_API_DOCS else None,
    openapi_url="/openapi.json" if EXPOSE_API_DOCS else None,
)

# CORS middleware for dashboard access. Wildcard origins are unsafe with credentials.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "SECURECOATING_ALLOWED_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "X-Operator-Token"],
)
TRUSTED_HOSTS = [
    host.strip()
    for host in os.environ.get(
        "SECURECOATING_TRUSTED_HOSTS", "localhost,127.0.0.1,api,testserver"
    ).split(",")
    if host.strip()
]
app.add_middleware(TrustedHostMiddleware, allowed_hosts=TRUSTED_HOSTS)

# Authentication is fail-closed. A deliberate local-only test/demo override is
# required to run without an API key.
API_KEY = os.environ.get("SECURECOATING_API_KEY", "").strip()
ALLOW_UNAUTHENTICATED_DEMO = (
    ENVIRONMENT in {"development", "test"}
    and os.environ.get("SECURECOATING_ALLOW_UNAUTHENTICATED_DEMO", "false").lower()
    in {"1", "true", "yes"}
)
REQUIRE_API_KEY = not ALLOW_UNAUTHENTICATED_DEMO or bool(API_KEY)
if ENVIRONMENT == "production" and not os.environ.get("SECURECOATING_FACTORY_SECRET", "").strip():
    raise RuntimeError("SECURECOATING_FACTORY_SECRET must be configured in production")
if ENVIRONMENT == "production" and not API_KEY:
    raise RuntimeError("SECURECOATING_API_KEY must be configured in production")


@app.middleware("http")
async def optional_api_key_guard(request: Request, call_next):
    if request.method in {"POST", "PUT", "PATCH"}:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_REQUEST_SIZE:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "Request body exceeds the configured safety limit"},
                    )
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length"})
    if REQUIRE_API_KEY and (
        request.url.path.startswith("/api/") or request.url.path == "/health"
    ):
        if not API_KEY:
            return JSONResponse(
                status_code=503,
                content={"detail": "API authentication is not configured"},
            )
        provided = request.headers.get("x-api-key", "")
        if not secrets.compare_digest(provided, API_KEY):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing X-API-Key"},
            )
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


MAX_INFLIGHT_INSPECTIONS = max(
    1, min(int(os.environ.get("SECURECOATING_MAX_INFLIGHT_INSPECTIONS", "1")), 8)
)
INSPECTION_CAPACITY_WAIT_SECONDS = max(
    0.001,
    min(float(os.environ.get("SECURECOATING_CAPACITY_WAIT_SECONDS", "0.05")), 5.0),
)
inspection_capacity = asyncio.Semaphore(MAX_INFLIGHT_INSPECTIONS)


@app.middleware("http")
async def inspection_backpressure(request: Request, call_next):
    if request.url.path not in {"/api/inspect", "/api/pipeline/multi-stage"}:
        return await call_next(request)
    acquired = False
    try:
        await asyncio.wait_for(
            inspection_capacity.acquire(), timeout=INSPECTION_CAPACITY_WAIT_SECONDS
        )
        acquired = True
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=429,
            content={"detail": "Inspection capacity is busy; request was not queued"},
            headers={"Retry-After": "1"},
        )
    try:
        return await call_next(request)
    finally:
        if acquired:
            inspection_capacity.release()

# --- Configuration Loading ---
CONFIG_PATH = os.environ.get("MODEL_CONFIG", "configs/model.yaml")
APP_CONFIG_PATH = os.environ.get("APP_CONFIG", "configs/app.yaml")
CALIBRATION_CONFIG_PATH = os.environ.get("CALIBRATION_CONFIG", "configs/calibration.yaml")

with open(CONFIG_PATH, "r") as f:
    model_config = yaml.safe_load(f)
with open(APP_CONFIG_PATH, "r") as f:
    app_config = yaml.safe_load(f)
with open(CALIBRATION_CONFIG_PATH, "r") as f:
    calibration_config = yaml.safe_load(f)

PIXEL_SIZE_MM = float(calibration_config["pixel_size_x_mm"])
CALIBRATION_VERIFIED = bool(calibration_config.get("verified", False))

DEMO_DATASET_ROOT = Path(PROJECT_ROOT, "data", "demo_real").resolve()
TEST_SET_DIR = str(DEMO_DATASET_ROOT / "images")
DEMO_DATASET_MANIFEST = DEMO_DATASET_ROOT / "manifest.json"
_default_artifact_dir = Path(PROJECT_ROOT, "outputs", "inspection_artifacts").resolve()
INSPECTION_ARTIFACT_DIR = Path(
    os.environ.get("SECURECOATING_INSPECTION_ARTIFACT_DIR", str(_default_artifact_dir))
).resolve()
if ENVIRONMENT == "production" and _default_artifact_dir.parent not in (
    INSPECTION_ARTIFACT_DIR,
    *INSPECTION_ARTIFACT_DIR.parents,
):
    raise RuntimeError("Production inspection artifact directory must remain under outputs/")
MAX_INSPECTION_ARTIFACTS = max(
    10, min(int(os.environ.get("SECURECOATING_MAX_INSPECTION_ARTIFACTS", "200")), 5000)
)
REQUIRE_INSPECTION_ARTIFACTS = os.environ.get(
    "SECURECOATING_REQUIRE_INSPECTION_ARTIFACTS",
    "true" if ENVIRONMENT == "production" else "false",
).lower() in {"1", "true", "yes"}
_artifact_lock = threading.Lock()
_dataset_catalog_lock = threading.Lock()
_dataset_catalog_cache: Optional[tuple[Dict[str, Any], ...]] = None
_dataset_catalog_identity: Dict[str, Any] = {}
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
CONTROL_POLICY = {
    "endpoint": "/api/operations/control",
    "snapshot_ttl_seconds": 30,
    "actions": {
        "EMERGENCY_STOP": {
            "confirmation": "CONFIRM EMERGENCY_STOP",
            "effect": "Latch local interlock and request PLC E-stop",
        },
        "RESET": {
            "confirmation": "CONFIRM RESET",
            "effect": "Request PLC reset; latch clears only after confirmation",
            "requires_fresh_snapshot": True,
        },
        "INFERENCE_RESET": {
            "confirmation": "CONFIRM INFERENCE_RESET",
            "effect": "Arm one trained-model recovery probe after an inference fault",
            "requires_fresh_snapshot": True,
        },
    },
    "forbidden": [
        "PLC_PARAMETER_WRITE",
        "RECIPE_SLIDER",
        "DEFECT_INJECTION",
        "LIBAD_DEMO",
    ],
}

SNAPSHOT_TTL_SECONDS = max(
    5, min(int(os.environ.get("SECURECOATING_SNAPSHOT_TTL_SECONDS", "30")), 300)
)
CONTROL_POLICY["snapshot_ttl_seconds"] = SNAPSHOT_TTL_SECONDS
_snapshot_registry: Dict[str, Dict[str, Any]] = {}
_snapshot_lock = threading.Lock()
_control_lock = threading.Lock()


def _load_operator_credentials() -> List[Dict[str, Any]]:
    """Load hashed operator tokens without keeping plaintext credentials in memory."""
    raw = os.environ.get("SECURECOATING_OPERATOR_CREDENTIALS", "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("SECURECOATING_OPERATOR_CREDENTIALS must be valid JSON") from exc
    if not isinstance(parsed, list):
        raise RuntimeError("SECURECOATING_OPERATOR_CREDENTIALS must be a JSON list")
    credentials = []
    for entry in parsed:
        if not isinstance(entry, dict):
            raise RuntimeError("Each operator credential must be an object")
        operator_id = str(entry.get("operator_id", ""))
        token_sha256 = str(entry.get("token_sha256", "")).lower()
        roles = entry.get("roles", [])
        _validate_identifier(operator_id, "operator_id")
        if not re.fullmatch(r"[0-9a-f]{64}", token_sha256):
            raise RuntimeError("Operator token_sha256 must be a 64-character lowercase hex digest")
        if not isinstance(roles, list) or not roles:
            raise RuntimeError("Operator credentials must contain at least one role")
        credentials.append(
            {"operator_id": operator_id, "token_sha256": token_sha256, "roles": set(map(str, roles))}
        )
    return credentials


def _validate_identifier(value: str, field_name: str) -> str:
    if not IDENTIFIER_PATTERN.fullmatch(value or ""):
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be 1-128 characters from the approved identifier set",
        )
    return value


OPERATOR_CREDENTIALS = _load_operator_credentials()
CONTROL_AUTH_READY = ENVIRONMENT != "production" or bool(OPERATOR_CREDENTIALS)
_ACTION_ROLES = {
    "EMERGENCY_STOP": {"operator", "safety_reset", "maintenance"},
    "RESET": {"safety_reset"},
    "INFERENCE_RESET": {"maintenance"},
}


def _authenticate_control_operator(request: Request, claimed_operator_id: str, action: str) -> str:
    """Resolve the operator from a hashed token; never trust a form identity in production."""
    _validate_identifier(claimed_operator_id, "operator_id")
    if not OPERATOR_CREDENTIALS:
        if ENVIRONMENT == "production":
            raise HTTPException(
                status_code=503,
                detail="Operator control authentication is not configured",
            )
        return claimed_operator_id

    token = request.headers.get("x-operator-token", "")
    if not token:
        raise HTTPException(status_code=401, detail="Missing X-Operator-Token")
    token_digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    credential = next(
        (
            item
            for item in OPERATOR_CREDENTIALS
            if secrets.compare_digest(token_digest, item["token_sha256"])
        ),
        None,
    )
    if credential is None:
        raise HTTPException(status_code=401, detail="Invalid operator credential")
    if not secrets.compare_digest(claimed_operator_id, credential["operator_id"]):
        raise HTTPException(status_code=403, detail="Operator identity does not match credential")
    if not (credential["roles"] & _ACTION_ROLES[action]):
        raise HTTPException(status_code=403, detail="Operator role is not authorized for this action")
    return credential["operator_id"]


def _decode_upload(file_bytes: bytes) -> np.ndarray:
    """Validate image headers/dimensions before allocating the decoded array."""
    try:
        with Image.open(BytesIO(file_bytes)) as header:
            if header.format not in ALLOWED_IMAGE_FORMATS:
                raise HTTPException(status_code=415, detail="Unsupported image format")
            width, height = header.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="Decoded image dimensions exceed the safety limit",
                )
            header.verify()
    except HTTPException:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid or unsafe image payload") from exc

    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image")
    if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
        raise HTTPException(status_code=413, detail="Decoded image dimensions exceed the safety limit")
    return image


async def _read_upload_limited(upload: UploadFile) -> bytes:
    """Read an upload incrementally so the payload limit is enforced before allocation."""
    if upload.content_type and upload.content_type.lower() not in ALLOWED_IMAGE_MIME_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported image media type")
    payload = bytearray()
    total = 0
    chunk_size = 1024 * 1024
    while True:
        chunk = await upload.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded image exceeds maximum payload limit of {MAX_UPLOAD_SIZE // (1024 * 1024)}MB."
            )
        payload.extend(chunk)
    return bytes(payload)


def _load_sample_image(sample_name: str) -> np.ndarray:
    """Load an attributed real demo image by basename only."""
    safe_name = os.path.basename(sample_name)
    if safe_name != sample_name or len(sample_name) > 255:
        raise HTTPException(status_code=422, detail="sample_name must be a bounded basename")
    path = os.path.join(TEST_SET_DIR, safe_name)
    if not os.path.isfile(path):
        raise HTTPException(
            status_code=404,
            detail=f"Sample '{safe_name}' not found under data/demo_real/images",
        )
    image = cv2.imread(path)
    if image is None:
        raise HTTPException(status_code=400, detail=f"Failed to read sample '{safe_name}'")
    return image


def _synthetic_optical(h: int, w: int, simulate_defect: Optional[str]) -> np.ndarray:
    """Camera-simulation canvas used when no real image is supplied."""
    optical = np.ones((h, w, 3), dtype=np.uint8) * 180
    noise = np.random.randint(-8, 8, (h, w, 3), dtype=np.int16)
    optical = np.clip(optical.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if simulate_defect == "scratch":
        cv2.line(optical, (150, 300), (800, 320), (40, 40, 40), thickness=6)
    elif simulate_defect == "void":
        cv2.circle(optical, (500, 600), 80, (80, 60, 80), -1)
    elif simulate_defect == "blister":
        cv2.circle(optical, (400, 450), 60, (210, 215, 210), -1)
    elif simulate_defect == "delamination":
        pts = np.array([[400, 400], [600, 380], [650, 550], [420, 580]], dtype=np.int32)
        cv2.fillPoly(optical, [pts], (100, 90, 110))
    return optical


MAX_UPLOAD_SIZE = int(os.environ.get("SECURECOATING_MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
MAX_REQUEST_SIZE = MAX_UPLOAD_SIZE + 1024 * 1024  # bounded multipart envelope overhead
MAX_IMAGE_PIXELS = int(os.environ.get("SECURECOATING_MAX_IMAGE_PIXELS", 8_000_000))
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "BMP", "TIFF"}
ALLOWED_IMAGE_MIME_TYPES = {
    "image/jpeg", "image/png", "image/bmp", "image/tiff", "application/octet-stream"
}
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


ARTIFACT_VIEWS = ("raw", "input", "overlay")


def _inspection_artifact_path(run_id: str, view: str = "overlay") -> Path:
    if not re.fullmatch(r"RUN_[A-F0-9]{12}", run_id):
        raise ValueError("Invalid inspection run identifier")
    if view not in ARTIFACT_VIEWS:
        raise ValueError("Invalid inspection artifact view")
    # Keep the historical overlay filename stable for existing deployments.
    suffix = "" if view == "overlay" else f"_{view}"
    return INSPECTION_ARTIFACT_DIR / f"{run_id}{suffix}.jpg"


def _bounded_preview(image: np.ndarray, max_edge: int = 1280) -> np.ndarray:
    """Return a bounded BGR preview without changing the acquired source array."""
    height, width = image.shape[:2]
    scale = min(1.0, max_edge / max(height, width))
    if scale >= 1.0:
        return image.copy()
    return cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _model_input_preview(image: np.ndarray) -> np.ndarray:
    """Render the letterboxed optical tensor geometry used by the YOLO paths."""
    image_size = int(model_config.get("inference", {}).get("imgsz", 640))
    height, width = image.shape[:2]
    scale = min(image_size / height, image_size / width)
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    canvas = np.full((image_size, image_size, 3), 114, dtype=np.uint8)
    top = (image_size - resized_height) // 2
    left = (image_size - resized_width) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return canvas


def _encode_artifact_jpeg(image: np.ndarray) -> Optional[bytes]:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok or encoded.nbytes > 3 * 1024 * 1024:
        return None
    return encoded.tobytes()


def _write_inspection_artifact(
    image: np.ndarray,
    defects: list[Dict[str, Any]],
    run_id: str,
) -> bool:
    """Persist bounded raw/input/overlay previews as one run-scoped bundle."""
    raw_preview = _bounded_preview(image)
    canvas = raw_preview.copy()
    height, width = image.shape[:2]
    scale = canvas.shape[1] / width

    for defect in defects:
        bbox = defect.get("bbox") or []
        if len(bbox) != 4:
            continue
        x, y, box_width, box_height = (int(round(float(v) * scale)) for v in bbox)
        x2 = min(canvas.shape[1] - 1, max(x + 1, x + box_width))
        y2 = min(canvas.shape[0] - 1, max(y + 1, y + box_height))
        x = max(0, min(x, canvas.shape[1] - 1))
        y = max(0, min(y, canvas.shape[0] - 1))
        cv2.rectangle(canvas, (x, y), (x2, y2), (0, 220, 255), 2)
        confidence = defect.get("confidence")
        label = str(defect.get("class_name", "unknown"))
        if isinstance(confidence, (int, float)):
            label += f" {confidence:.3f}"
        cv2.putText(
            canvas,
            label,
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )

    encoded_artifacts = {
        "raw": _encode_artifact_jpeg(raw_preview),
        "input": _encode_artifact_jpeg(_model_input_preview(image)),
        "overlay": _encode_artifact_jpeg(canvas),
    }
    if any(payload is None for payload in encoded_artifacts.values()):
        return False

    with _artifact_lock:
        INSPECTION_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        targets = {
            view: _inspection_artifact_path(run_id, view)
            for view in ARTIFACT_VIEWS
        }
        temporaries = {
            view: target.with_suffix(target.suffix + ".tmp")
            for view, target in targets.items()
        }
        committed = []
        try:
            for view in ARTIFACT_VIEWS:
                with open(temporaries[view], "wb") as handle:
                    handle.write(encoded_artifacts[view])
                    handle.flush()
                    os.fsync(handle.fileno())
            for view in ARTIFACT_VIEWS:
                os.replace(temporaries[view], targets[view])
                committed.append(targets[view])
        except OSError:
            for target in committed:
                target.unlink(missing_ok=True)
            raise
        finally:
            for temporary in temporaries.values():
                temporary.unlink(missing_ok=True)

        overlays = sorted(
            INSPECTION_ARTIFACT_DIR.glob("RUN_*.jpg"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        overlays = [
            path for path in overlays
            if not path.stem.endswith(("_raw", "_input"))
        ]
        for expired_overlay in overlays[MAX_INSPECTION_ARTIFACTS:]:
            expired_run_id = expired_overlay.stem
            for view in ARTIFACT_VIEWS:
                _inspection_artifact_path(expired_run_id, view).unlink(missing_ok=True)
    return True


def _dataset_catalog(offset: int = 0, limit: int = 24) -> Dict[str, Any]:
    """Return bounded demo metadata without image payloads or host paths."""
    global _dataset_catalog_cache, _dataset_catalog_identity
    with _dataset_catalog_lock:
        if _dataset_catalog_cache is None:
            names = []
            if os.path.isdir(TEST_SET_DIR):
                with os.scandir(TEST_SET_DIR) as entries:
                    for entry in entries:
                        if entry.is_file() and entry.name.lower().endswith(
                            (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
                        ):
                            names.append(entry.name)
                            if len(names) >= 50_000:
                                break
            try:
                manifest = json.loads(DEMO_DATASET_MANIFEST.read_text(encoding="utf-8"))
                manifest_records = {
                    str(record["filename"]): record
                    for record in manifest.get("samples", [])
                    if isinstance(record, dict) and record.get("filename")
                }
            except (OSError, ValueError, TypeError):
                manifest = {}
                manifest_records = {}

            catalog_records = []
            all_verified = bool(names)
            for name in sorted(names):
                record = manifest_records.get(name, {})
                expected_hash = str(record.get("sha256") or "").lower()
                path = Path(TEST_SET_DIR, name)
                actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                hash_verified = bool(expected_hash) and secrets.compare_digest(
                    expected_hash, actual_hash
                )
                all_verified = all_verified and hash_verified
                catalog_records.append({
                    "filename": name,
                    "source_type": record.get("source_type", "UNVERIFIED"),
                    "capture_stage": record.get("capture_stage", "unknown"),
                    "source_id": record.get("source_id"),
                    "license": record.get("license"),
                    "sha256": actual_hash,
                    "hash_verified": hash_verified,
                })
            all_verified = all_verified and len(manifest_records) == len(names)
            _dataset_catalog_cache = tuple(catalog_records)
            _dataset_catalog_identity = {
                "dataset_name": manifest.get("dataset_name", "Unverified local dataset"),
                "dataset_source": manifest.get("source"),
                "dataset_license": manifest.get("license"),
                "provenance_verified": all_verified,
            }
        records = _dataset_catalog_cache
        identity = dict(_dataset_catalog_identity)
    page = records[offset:offset + limit]
    return {
        "total": len(records),
        "offset": offset,
        "limit": limit,
        "returned": len(page),
        "items": list(page),
        **identity,
        "truncated_at_source": len(records) >= 50_000,
        "image_payloads_included": False,
    }


def _attach_confidences(defects: list, detections: list) -> list:
    """Attach confidence only when class and bbox overlap identify the same instance."""
    if not detections:
        for d in defects:
            d.setdefault("confidence", None)
        return defects

    matched_dets = set()
    for d in defects:
        defect_box = d.get("bbox", [])
        if len(defect_box) != 4:
            d["confidence"] = None
            continue
        dx1, dy1, dw, dh = map(float, defect_box)
        dx2, dy2 = dx1 + max(0.0, dw), dy1 + max(0.0, dh)
        d_class = d.get("class_name", "")

        best_conf = None
        best_iou = 0.0
        best_idx = -1

        for i, det in enumerate(detections):
            if i in matched_dets:
                continue
            if det.get("class_name", "") != d_class:
                continue

            box = det.get("box", [0, 0, 0, 0])
            if len(box) == 4:
                bx1, by1, bx2, by2 = map(float, box)
                ix1, iy1 = max(dx1, bx1), max(dy1, by1)
                ix2, iy2 = min(dx2, bx2), min(dy2, by2)
                intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                union = max(0.0, dw * dh) + max(0.0, (bx2 - bx1) * (by2 - by1)) - intersection
                iou = intersection / union if union > 0 else 0.0
                if iou >= 0.10 and iou > best_iou:
                    best_iou = iou
                    best_conf = float(det.get("confidence", 0.0))
                    best_idx = i

        if best_idx >= 0:
            matched_dets.add(best_idx)
            d["confidence"] = round(best_conf, 4) if best_conf is not None else None
        else:
            d["confidence"] = None

    return defects


# --- Initialize Core Engines ---
# 1. Predictor (ONNX YOLOv8-seg primary, PyTorch fusion fallback)
predictor = CoatingPredictor(model_config)

# 2. Sensor Fusion Manager
SIMULATION_MODE = (
    ENVIRONMENT in {"development", "test"}
    and os.environ.get("SECURECOATING_ENABLE_SENSOR_SIMULATION", "true").lower()
    in {"1", "true", "yes"}
)
fusion_manager = SensorFusionManager(
    target_size=(1024, 1024),
    enable_mock=SIMULATION_MODE,
)

# 3. Fail-Safe Manager
failsafe = FailSafeManager(
    max_inference_timeout_ms=5000.0,
    enable_frame_validation=True,
    enable_auto_recovery=True
)
if not (predictor.yolo_available or predictor.onnx_available):
    failsafe.latch_inference_interlock("No trained inference model is loaded")
if not SIMULATION_MODE:
    failsafe.health.rgb_sensor_ok = False
    failsafe.health.thermal_sensor_ok = False
    failsafe.health.profiler_sensor_ok = False
    failsafe._update_system_state()

# 4. Quality Memory (SQLite traceability)
db_path = os.environ.get(
    "SECURECOATING_DB_PATH",
    app_config.get("paths", {}).get("db_path", "data/quality_history.db"),
)
quality_mem = QualityMemory(db_path)

# 5. Industrial Protocol Manager (OPC UA / Modbus TCP)
industrial_config = app_config.get("industrial_io", {})
industrial_config = dict(industrial_config)
if "SECURECOATING_INDUSTRIAL_MOCK_MODE" in os.environ:
    industrial_config["mock_mode"] = os.environ["SECURECOATING_INDUSTRIAL_MOCK_MODE"].lower() in {
        "1", "true", "yes"
    }
if ENVIRONMENT == "production" and industrial_config.get("mock_mode", True):
    raise RuntimeError("Industrial mock mode is forbidden in production")
industrial_mgr = IndustrialProtocolManager(industrial_config)

# 6. Web Motion & Continuous Roll Synchronizer
web_synchronizer = WebSynchronizer()

# 7. Physics-Informed Battery Electrode Metrology
electrode_metrology = ElectrodeMetrologyEngine(pixel_to_mm_ratio=PIXEL_SIZE_MM)

# 8. AI Closed-Loop Root-Cause Diagnostics
root_cause_engine = RootCauseDiagnosticEngine()

# 9. Multi-Stage Industrial Pipeline Orchestrator
multi_stage_pipeline = MultiStageIndustrialPipeline(
    predictor=predictor,
    fusion_manager=fusion_manager,
    failsafe_manager=failsafe,
    industrial_manager=industrial_mgr,
    web_synchronizer=web_synchronizer,
    pixel_to_mm_ratio=PIXEL_SIZE_MM,
    calibration_verified=CALIBRATION_VERIFIED or SIMULATION_MODE,
    fov_width_mm=float(calibration_config["fov_width_mm"]),
    fov_length_m=float(calibration_config["fov_length_m"]),
)


# --- Response Models ---
class InspectionResponse(BaseModel):
    part_id: str
    batch_id: str
    run_id: str
    status: str
    passed: bool
    reject_reasons: list = Field(default_factory=list)
    latency_ms: float
    defects_found: list = Field(default_factory=list)
    fallback_active: bool
    system_state: str
    gate_action: str
    engine: str
    model_version: str = "2.0.0"
    detections: list = Field(default_factory=list)


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    status: str
    device: str
    model_version: str
    onnx_available: bool
    yolo_available: bool = False
    primary_engine: str = "unknown"
    system_state: str
    sensors: Dict[str, str]


# --- Endpoints ---
@app.get("/live")
def liveness_check():
    """Process liveness only; safety/readiness state is reported by /health."""
    return {"status": "ALIVE"}


def _build_health_payload():
    """Build one authoritative readiness payload for health and operations views."""
    health = failsafe.get_health_report()
    industrial_state = industrial_mgr.get_plc_state()
    ready = (
        health["system_state"] == "OPTIMAL"
        and not industrial_state["interlock_latched"]
        and quality_mem.healthy
        and (predictor.yolo_available or predictor.onnx_available)
        and industrial_mgr.transport_ready
        and CONTROL_AUTH_READY
    )
    payload = {
        "status": "HEALTHY" if ready else "DEGRADED",
        "device": str(predictor.device),
        "model_version": model_config.get("model", {}).get("version", "2.0.0"),
        "onnx_available": predictor.onnx_available,
        "yolo_available": predictor.yolo_available,
        "primary_engine": (
            predictor.yolo_engine.active_provider
            if predictor.yolo_available
            else (
                predictor.onnx_engine.active_provider
                if predictor.onnx_available
                else "NONE"
            )
        ),
        "system_state": health["system_state"],
        "sensors": health["sensors"],
        "industrial_interlock_latched": industrial_state["interlock_latched"],
        "traceability_ok": quality_mem.healthy,
        "simulation_mode": SIMULATION_MODE,
        "industrial_transport_ready": industrial_mgr.transport_ready,
        "control_auth_ready": CONTROL_AUTH_READY,
    }
    return payload, ready


def _build_roll_certificate_payload(roll_id: str, batch_id: str, electrode_type: str) -> Dict[str, Any]:
    """Build a certificate from the current roll/batch snapshot without inventing missing stats."""
    snapshot = web_synchronizer.get_roll_snapshot(roll_id, batch_id=batch_id)
    if snapshot is None:
        raise HTTPException(
            status_code=404,
            detail=f"No active roll/batch snapshot for {roll_id}/{batch_id}",
        )
    summary = snapshot["summary"]
    defect_records = snapshot["defect_records"]
    batch_stats = quality_mem.get_batch_stats(batch_id)
    if batch_stats.get("status") == "ERROR":
        raise HTTPException(status_code=503, detail="Traceability database unavailable")
    spc = quality_mem.check_spc_alarms(batch_id)
    root_cause = root_cause_engine.diagnose_batch(defect_records)
    cert = RollCertificateGenerator.build_certificate(
        roll_id=roll_id,
        batch_id=batch_id,
        inspected_length_m=summary["inspected_length_m"],
        total_length_m=summary["total_roll_length_m"],
        defect_records=defect_records,
        spc_status=spc.get("status", "UNKNOWN"),
        root_cause_summary=root_cause.primary_root_cause,
        electrode_type=electrode_type,
        total_inspections=batch_stats["total"] if batch_stats["total"] > 0 else None,
        failed_inspections=batch_stats["failed"] if batch_stats["total"] > 0 else None,
        held_inspections=batch_stats.get("held", 0) if batch_stats["total"] > 0 else None,
        metric_provenance=(
            f"SQLite batch={batch_id}; roll ledger={roll_id}; "
            f"unresolved_holds={batch_stats.get('held', 0)}"
        ),
    )
    return cert.to_dict()


@app.get("/health", response_model=HealthResponse)
def health_check():
    """System health check with component status."""
    payload, ready = _build_health_payload()
    return JSONResponse(
        status_code=200 if ready else 503,
        content=payload,
    )


@app.post("/api/inspect", response_model=InspectionResponse)
async def inspect(
    background_tasks: BackgroundTasks,
    batch_id: Optional[str] = Form(None),
    part_id: str = Form("PART_000"),
    simulate_defect: Optional[str] = Form(None),
    sample_name: Optional[str] = Form(None),
    thermal_online: bool = Form(True),
    profiler_online: bool = Form(True),
    image: Optional[UploadFile] = File(None),
):
    """
    Full inspection pipeline: acquire -> fuse -> infer -> grade -> signal.

    Image sources (priority):
    1. multipart `image` upload
    2. `sample_name` from the attributed real optical demo subset
    3. camera simulation canvas (+ optional simulate_defect overlay)
    """
    using_real_image = False
    active_batch_id = web_synchronizer.roll.batch_id
    batch_id = batch_id or active_batch_id
    _validate_identifier(batch_id, "batch_id")
    _validate_identifier(part_id, "part_id")
    if simulate_defect not in {None, "scratch", "void", "blister", "delamination"}:
        raise HTTPException(status_code=422, detail="Unsupported simulate_defect value")
    if not SIMULATION_MODE and (simulate_defect is not None or sample_name is not None):
        raise HTTPException(
            status_code=404,
            detail="Demo sample and defect simulation inputs are disabled in production",
        )
    if not SIMULATION_MODE and (image is None or not image.filename):
        raise HTTPException(
            status_code=422,
            detail="Production mode requires an explicit acquired image payload",
        )
    if batch_id != active_batch_id:
        raise HTTPException(
            status_code=409,
            detail=f"Batch {batch_id} is not active for roll {web_synchronizer.roll.roll_id}",
        )

    if image is not None and image.filename:
        raw_bytes = await _read_upload_limited(image)
        optical = _decode_upload(raw_bytes)
        using_real_image = True
    elif sample_name:
        optical = _load_sample_image(sample_name)
        using_real_image = True
    else:
        optical = _synthetic_optical(1024, 1024, simulate_defect)

    h, w = optical.shape[:2]
    frame_ctx = web_synchronizer.advance_motion(dt_seconds=0.0)

    # 2. Multi-source sensor fusion
    effective_thermal_online = thermal_online if SIMULATION_MODE else False
    effective_profiler_online = profiler_online if SIMULATION_MODE else False
    fusion_result = await run_in_threadpool(
        fusion_manager.fuse,
        rgb_image=optical,
        thermal_online=effective_thermal_online,
        profiler_online=effective_profiler_online,
    )

    thermal = fusion_result.thermal_frame
    height = fusion_result.height_frame

    # 3. Update fail-safe sensor status from THIS request (non-sticky intent)
    failsafe.health.thermal_sensor_ok = fusion_result.sensor_status.thermal_online
    failsafe.health.profiler_sensor_ok = fusion_result.sensor_status.profiler_online
    failsafe.health.rgb_sensor_ok = True
    failsafe._update_system_state()

    # 4. Safe prediction with fail-safe wrapping
    result = await run_in_threadpool(
        failsafe.safe_predict, predictor, optical, thermal, height
    )

    # 5. Postprocess: extract physical defect measurements
    seg_mask = result.get("segmentation_mask", np.zeros((h, w), dtype=np.uint8))
    defects = extract_defects_from_mask(seg_mask, height, pixel_to_mm_ratio=PIXEL_SIZE_MM)

    # Camera-sim only: if synthetic overlay was not detected, plant a visible region
    # so grading/PLC demo still works without claiming false ONNX detections.
    if (
        not using_real_image
        and simulate_defect
        and len(defects) == 0
        and not (predictor.yolo_available or predictor.onnx_available)
    ):
        class_map = {"scratch": 1, "void": 2, "blister": 3, "delamination": 4}
        cid = class_map.get(simulate_defect, 1)
        cv2.circle(seg_mask, (w // 2, h // 2), min(h, w) // 10, int(cid), -1)
        defects = extract_defects_from_mask(seg_mask, height, pixel_to_mm_ratio=PIXEL_SIZE_MM)

    # 6. Grade the coating against quality rules
    grading_rules = model_config.get("inference", {}).get("grading", {})
    grade = grade_coating(defects, grading_rules)

    # 7. Authoritative safety gate. Grading output is never allowed to
    # override inference/model/sensor/PLC/traceability readiness.
    result_state = result.get("system_state", "EMERGENCY")
    safety_reasons = []
    if result_state != "OPTIMAL" or not failsafe.decision_permitted():
        safety_reasons.append(f"Inspection state is {result_state}; automatic gate decision forbidden")
    if result.get("untrained_fallback", False) or not (
        predictor.yolo_available or predictor.onnx_available
    ):
        safety_reasons.append("No trained and verified inference model is ready")
    if result.get("error_reason"):
        safety_reasons.append(str(result["error_reason"]))
    if industrial_mgr.interlock_latched:
        safety_reasons.append("PLC safety interlock is already latched")
    if not industrial_mgr.transport_ready:
        safety_reasons.append("Industrial transport security/readback configuration is not ready")
    if not (CALIBRATION_VERIFIED or SIMULATION_MODE):
        safety_reasons.append("Factory coordinate calibration is missing or unverified")

    # 8. Persist the inspection before issuing any normal PLC gate command.
    max_len = max([d["length_mm"] for d in defects]) if defects else 0.0
    max_area = max([d["area_mm2"] for d in defects]) if defects else 0.0
    peak_h = max([d["peak_height_um"] for d in defects]) if defects else 0.0
    primary_class = defects[0]["class_name"] if defects else "none"
    detections = result.get("detections", [])
    defects = _attach_confidences(defects, detections)
    run_id = f"RUN_{uuid.uuid4().hex[:12].upper()}"
    for defect in defects:
        defect["run_id"] = run_id
    trace_written = quality_mem.add_entry(
        batch_id=batch_id,
        part_id=part_id,
        has_defect=not grade["passed"],
        defect_class=primary_class,
        max_length=max_len,
        max_area=max_area,
        peak_height=peak_h,
        latency=result.get("latency_ms", 0.0),
        fallback=result.get("fallback_active", False),
        model_version=result.get("model_version", predictor.model_version),
        run_id=run_id,
        roll_id=web_synchronizer.roll.roll_id,
        # The authoritative decision is written only after PLC confirmation.
        # PENDING cannot be mistaken for a released PASS if finalization fails.
        gate_action="PENDING",
        system_state=result_state,
        inspection_valid=not safety_reasons,
        error_reason="; ".join(safety_reasons) or None,
    )
    if trace_written:
        try:
            artifact_written = _write_inspection_artifact(optical, defects, run_id)
        except (OSError, ValueError, cv2.error) as exc:
            artifact_written = False
            logger.error("Inspection artifact write failed: %s", exc)
        if REQUIRE_INSPECTION_ARTIFACTS and not artifact_written:
            safety_reasons.append(
                "Inspection evidence artifact write failed; automatic gate decision forbidden"
            )
        try:
            for defect in defects:
                bbox = defect.get("bbox") or []
                if (CALIBRATION_VERIFIED or SIMULATION_MODE) and len(bbox) == 4:
                    box_x, box_y, box_w, box_h = bbox
                    coordinate = web_synchronizer.map_defect_to_physical_coordinate(
                        frame_ctx,
                        pixel_x_td=float(box_x) + float(box_w) / 2.0,
                        pixel_y_md=float(box_y) + float(box_h) / 2.0,
                        frame_width_px=w,
                        frame_height_px=h,
                        fov_width_mm=float(calibration_config["fov_width_mm"]),
                        fov_length_m=float(calibration_config["fov_length_m"]),
                    )
                    web_synchronizer.record_defect_on_roll(defect, coordinate)
                else:
                    web_synchronizer.record_unlocalized_defect(defect, frame_ctx)
        except (KeyError, TypeError, ValueError) as exc:
            logger.error("Defect-ledger write failed: %s", exc)
            safety_reasons.append("Defect-ledger write failed; automatic gate decision forbidden")
    if not trace_written:
        trace_reason = quality_mem.last_error
        if trace_reason.startswith("Duplicate inspection identity rejected"):
            safety_reasons.append(trace_reason)
        else:
            safety_reasons.append("Traceability write failed; automatic gate decision forbidden")

    # 9. Industrial protocol signaling (PLC/MES)
    industrial_result = await run_in_threadpool(
        industrial_mgr.process_inspection_result,
        part_id=part_id,
        batch_id=batch_id,
        defects=defects,
        grade_result=grade,
        safety_permitted=not safety_reasons,
        safety_reasons=safety_reasons,
    )
    final_gate_action = industrial_result["gate_action"]
    if trace_written and not quality_mem.update_decision(
        run_id,
        final_gate_action,
        result_state,
        inspection_valid=not safety_reasons,
        error_reason="; ".join(safety_reasons) or None,
    ):
        safety_reasons.append("Traceability finalization failed; HOLD latched")
        hold_signal = industrial_mgr.trigger_hold(part_id, batch_id, safety_reasons)
        final_gate_action = "HOLD"
        industrial_result.update({
            "gate_action": "HOLD",
            "signal_id": hold_signal.signal_id,
            "rejection_reasons": safety_reasons,
        })

    final_passed = bool(
        grade["passed"]
        and not safety_reasons
        and result_state == "OPTIMAL"
        and final_gate_action == "PASS"
    )
    response_reasons = list(grade["reject_reasons"])
    for reason in safety_reasons:
        if reason not in response_reasons:
            response_reasons.append(reason)

    sanitized_detections = [
        {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}
        for d in detections if isinstance(d, dict)
    ]

    return {
        "part_id": part_id,
        "batch_id": batch_id,
        "run_id": run_id,
        "status": result.get("status", "Unknown"),
        "passed": final_passed,
        "reject_reasons": response_reasons,
        "latency_ms": round(result.get("latency_ms", 0.0), 2),
        "defects_found": defects,
        "fallback_active": result.get("fallback_active", False),
        "system_state": result.get("system_state", "OPTIMAL"),
        "gate_action": final_gate_action,
        "engine": result.get("engine", "unknown"),
        "model_version": result.get("model_version", predictor.model_version),
        "detections": sanitized_detections,
    }


@app.get("/api/samples")
def list_demo_samples(
    offset: int = Query(0, ge=0, le=10_000),
    limit: int = Query(100, ge=1, le=100),
):
    """List attributed real optical demo images available in simulation mode."""
    if not SIMULATION_MODE:
        raise HTTPException(status_code=404, detail="Demo samples are disabled in production")
    catalog = _dataset_catalog(offset=offset, limit=limit)
    samples = [item["filename"] for item in catalog["items"]]
    return {
        **catalog,
        "samples": samples,
        "count": catalog["total"],
        "directory": "data/demo_real/images",
    }


@app.get("/api/dataset/catalog")
def get_dataset_catalog(
    offset: int = Query(0, ge=0, le=10_000),
    limit: int = Query(24, ge=1, le=100),
):
    """Return bounded dataset basenames; never return image payloads or host paths."""
    return _dataset_catalog(offset=offset, limit=limit)


@app.get("/api/inspections/{run_id}/image")
def get_inspection_image(
    run_id: str,
    view: Literal["raw", "input", "overlay"] = Query("overlay"),
):
    """Return one authenticated bounded view for the active roll/batch identity."""
    try:
        path = _inspection_artifact_path(run_id, view)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    identity = quality_mem.get_inspection_identity(run_id)
    if identity.get("status") == "ERROR":
        raise HTTPException(status_code=503, detail="Traceability database unavailable")
    if identity.get("status") != "OK":
        raise HTTPException(status_code=404, detail="Inspection run not found")
    if (
        identity.get("roll_id") != web_synchronizer.roll.roll_id
        or identity.get("batch_id") != web_synchronizer.roll.batch_id
    ):
        raise HTTPException(status_code=404, detail="Inspection run is outside the active roll/batch")
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Inspection {view} image artifact is unavailable",
        )
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/metrics/latency")
def latency_metrics(
    batch_id: Optional[str] = None,
    limit: int = Query(200, ge=1, le=1000),
):
    """Latency p50/p95 vs 35ms competition target."""
    result = quality_mem.get_latency_stats(batch_id=batch_id, limit=limit)
    if result.get("status") == "ERROR":
        raise HTTPException(status_code=503, detail="Traceability database unavailable")
    return result


@app.get("/api/batch/{batch_id}/stats")
def get_batch_stats(batch_id: str):
    """Retrieve aggregated quality indicators for MES integration."""
    _validate_identifier(batch_id, "batch_id")
    if batch_id != web_synchronizer.roll.batch_id:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} is not active")
    stats = quality_mem.get_batch_stats(batch_id)
    if stats.get("status") == "ERROR":
        raise HTTPException(status_code=503, detail="Traceability database unavailable")
    return stats


@app.get("/api/batch/{batch_id}/spc")
def get_batch_spc(batch_id: str):
    """Check Statistical Process Control alarm state."""
    _validate_identifier(batch_id, "batch_id")
    if batch_id != web_synchronizer.roll.batch_id:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} is not active")
    result = quality_mem.check_spc_alarms(batch_id)
    if result.get("status") == "UNKNOWN":
        raise HTTPException(status_code=503, detail="Traceability database unavailable")
    return result


@app.get("/api/industrial/state")
def get_industrial_state():
    """Get current PLC register state and OPC UA node values."""
    return industrial_mgr.get_plc_state()


@app.get("/api/industrial/signals")
def get_signal_history(limit: int = Query(50, ge=1, le=500)):
    """Get recent industrial signal history."""
    return industrial_mgr.get_signal_history(limit=limit)


def _control_state_digest(state: Dict[str, Any]) -> str:
    canonical = json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _current_control_state() -> Dict[str, Any]:
    health, ready = _build_health_payload()
    roll = web_synchronizer.get_roll_snapshot(web_synchronizer.roll.roll_id)
    industrial = industrial_mgr.get_plc_state()
    summary = (roll or {}).get("summary") or {}
    disposition = (
        "EMERGENCY_STOP"
        if health["system_state"] == "EMERGENCY"
        else ("AUTOMATIC_DECISIONS_ALLOWED" if ready else "HOLD_REQUIRED")
    )
    return {
        "roll_id": summary.get("roll_id"),
        "batch_id": summary.get("batch_id"),
        "system_state": health["system_state"],
        "line_disposition": disposition,
        "industrial_interlock_latched": industrial.get("interlock_latched"),
        "inference_interlock_latched": failsafe.inference_interlock_latched,
        "industrial_transport_ready": health["industrial_transport_ready"],
        "traceability_ok": health["traceability_ok"],
        "trained_model_ready": bool(health["yolo_available"] or health["onnx_available"]),
    }


def _register_control_snapshot(snapshot_id: str) -> None:
    now = time.monotonic()
    record = {
        "expires_at": now + SNAPSHOT_TTL_SECONDS,
        "state_digest": _control_state_digest(_current_control_state()),
    }
    with _snapshot_lock:
        expired = [key for key, value in _snapshot_registry.items() if value["expires_at"] <= now]
        for key in expired:
            _snapshot_registry.pop(key, None)
        while len(_snapshot_registry) >= 256:
            _snapshot_registry.pop(next(iter(_snapshot_registry)))
        _snapshot_registry[snapshot_id] = record


def _consume_fresh_snapshot(snapshot_id: str) -> None:
    if not snapshot_id:
        raise HTTPException(status_code=409, detail="A fresh operations snapshot is required")
    now = time.monotonic()
    with _snapshot_lock:
        record = _snapshot_registry.pop(snapshot_id, None)
    if record is None:
        raise HTTPException(status_code=409, detail="Snapshot is unknown, expired, or already consumed")
    if record["expires_at"] <= now:
        raise HTTPException(status_code=409, detail="Snapshot has expired")
    if not secrets.compare_digest(record["state_digest"], _control_state_digest(_current_control_state())):
        raise HTTPException(status_code=409, detail="Authoritative control state changed after the snapshot")


@app.get("/api/operations/snapshot")
def get_operations_snapshot(signal_limit: int = Query(25, ge=1, le=100)):
    """Return one read-only operations snapshot for the production dashboard."""
    health, ready = _build_health_payload()
    roll = web_synchronizer.get_roll_snapshot(web_synchronizer.roll.roll_id)
    if roll is None:
        raise HTTPException(status_code=503, detail="Active roll state is unavailable")

    batch_id = roll["summary"]["batch_id"]
    roll_id = roll["summary"]["roll_id"]
    stats = quality_mem.get_batch_stats(batch_id)
    spc = quality_mem.check_spc_alarms(batch_id)
    recent_inspections = quality_mem.get_recent_inspections(batch_id, limit=20)
    for inspection in recent_inspections.get("records", []):
        run_id = inspection.get("run_id")
        artifact_views = {}
        try:
            for view in ARTIFACT_VIEWS:
                available = bool(
                    run_id and _inspection_artifact_path(run_id, view).is_file()
                )
                artifact_views[view] = {
                    "available": available,
                    "endpoint": (
                        f"/api/inspections/{run_id}/image?view={view}"
                        if available else None
                    ),
                }
            image_available = artifact_views["overlay"]["available"]
        except ValueError:
            image_available = False
            artifact_views = {
                view: {"available": False, "endpoint": None}
                for view in ARTIFACT_VIEWS
            }
        inspection["image_available"] = image_available
        inspection["image_endpoint"] = (
            f"/api/inspections/{run_id}/image" if image_available else None
        )
        inspection["artifacts"] = artifact_views
    industrial = industrial_mgr.get_plc_state()
    signals = industrial_mgr.get_signal_history(limit=signal_limit)
    control_audit = quality_mem.get_recent_control_audits(limit=25)

    if health["system_state"] == "EMERGENCY":
        disposition = "EMERGENCY_STOP"
    elif not ready:
        disposition = "HOLD_REQUIRED"
    else:
        disposition = "AUTOMATIC_DECISIONS_ALLOWED"

    readiness_reasons = []
    if health["system_state"] != "OPTIMAL":
        readiness_reasons.append(f"system_state={health['system_state']}")
    offline_sensors = [
        name for name, state in health["sensors"].items() if state != "ONLINE"
    ]
    if offline_sensors:
        readiness_reasons.append("offline_sensors=" + ",".join(offline_sensors))
    if health["industrial_interlock_latched"]:
        readiness_reasons.append("industrial_interlock_latched")
    if not health["industrial_transport_ready"]:
        readiness_reasons.append("industrial_transport_not_ready")
    if not health["traceability_ok"]:
        readiness_reasons.append("traceability_not_ready")
    if not (health["yolo_available"] or health["onnx_available"]):
        readiness_reasons.append("trained_model_not_ready")
    if not health["control_auth_ready"]:
        readiness_reasons.append("operator_control_auth_not_ready")

    certificate_payload = None
    certificate_error = None
    try:
        certificate_payload = _build_roll_certificate_payload(
            roll_id,
            batch_id,
            web_synchronizer.roll.coating_type,
        )
    except HTTPException as exc:
        certificate_error = exc.detail if isinstance(exc.detail, str) else str(exc.detail)

    snapshot_id = f"OPS_{uuid.uuid4().hex[:12].upper()}"
    snapshot = {
        "schema_version": "1.1",
        "snapshot_id": snapshot_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "consistency": "single_api_response_non_transactional_components",
        "surface": "operations",
        "readiness": health,
        "ready": ready,
        "line_disposition": disposition,
        "readiness_reasons": readiness_reasons,
        "roll": roll,
        "quality": {
            "stats": stats,
            "spc": spc,
            "recent_inspections": recent_inspections,
        },
        "dataset_catalog": _dataset_catalog(offset=0, limit=24),
        "industrial": industrial,
        "signals": signals,
        "traceability": {
            "certificate_status": "OK" if certificate_payload is not None else "UNAVAILABLE",
            "certificate": certificate_payload,
            "certificate_error": certificate_error,
        },
        "control_audit": control_audit,
        "control_policy": CONTROL_POLICY,
    }
    _register_control_snapshot(snapshot_id)
    return snapshot


def _dispatch_confirmed_control(action: str, reason: str) -> Dict[str, Any]:
    if action == "EMERGENCY_STOP":
        signal = industrial_mgr.emergency_stop(reason)
        return {
            "result_status": "SIMULATED" if industrial_mgr.mock_mode else (
                "ACKNOWLEDGED" if signal.acknowledged else "UNCONFIRMED"
            ),
            "http_status": 200 if signal.acknowledged or industrial_mgr.mock_mode else 503,
            "signal_id": signal.signal_id,
            "interlock_latched": industrial_mgr.interlock_latched,
            "acknowledged": signal.acknowledged,
            "mock_mode": industrial_mgr.mock_mode,
        }
    if action == "RESET":
        signal = industrial_mgr.reset_line()
        cleared = not industrial_mgr.interlock_latched
        return {
            "result_status": "SIMULATED" if industrial_mgr.mock_mode and cleared else (
                "ACKNOWLEDGED" if signal.acknowledged and cleared else "UNCONFIRMED"
            ),
            "http_status": 200 if cleared else 503,
            "signal_id": signal.signal_id,
            "interlock_latched": industrial_mgr.interlock_latched,
            "acknowledged": signal.acknowledged,
            "mock_mode": industrial_mgr.mock_mode,
        }
    accepted = failsafe.request_inference_reset_probe()
    return {
        "result_status": "RESET_PROBE_ARMED" if accepted else "RESET_BLOCKED",
        "http_status": 202 if accepted else 409,
        "signal_id": None,
        "inference_interlock_latched": failsafe.inference_interlock_latched,
        "acknowledged": None,
        "reset_probe_accepted": accepted,
        "mock_mode": industrial_mgr.mock_mode,
    }


def _latch_control_failure(reason: str) -> None:
    """Best-effort local HOLD/E-stop latch when command integrity becomes uncertain."""
    failsafe.latch_inference_interlock(reason)
    try:
        industrial_mgr.emergency_stop(reason)
    except Exception:
        logger.exception("Failed to dispatch secondary E-stop while latching control failure")


@app.post("/api/operations/control")
def operations_control(
    request: Request,
    action: str = Form(...),
    operator_id: str = Form(...),
    confirmation: str = Form(...),
    reason: str = Form(...),
    snapshot_id: str = Form(""),
    idempotency_key: str = Form(...),
):
    """Authenticated operator control: exact confirmation phrase, then durable audit."""
    _validate_identifier(idempotency_key, "idempotency_key")
    if snapshot_id:
        _validate_identifier(snapshot_id, "snapshot_id")
    policy = CONTROL_POLICY["actions"].get(action)
    if policy is None:
        raise HTTPException(status_code=422, detail="Unsupported control action")
    expected = policy["confirmation"]
    if not secrets.compare_digest(confirmation.strip(), expected):
        raise HTTPException(
            status_code=409,
            detail=f"Confirmation must match exactly: {expected}",
        )
    reason = reason.strip()
    if len(reason) < 8:
        raise HTTPException(status_code=422, detail="Reason must be at least 8 characters")
    if action == "INFERENCE_RESET" and not (predictor.yolo_available or predictor.onnx_available):
        raise HTTPException(
            status_code=409,
            detail="No trained model is available for a reset probe",
        )
    operator_id = _authenticate_control_operator(request, operator_id, action)
    request_digest = hashlib.sha256(
        json.dumps(
            {
                "action": action,
                "operator_id": operator_id,
                "reason": reason,
                "snapshot_id": snapshot_id or None,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    with _control_lock:
        try:
            prior = quality_mem.get_control_audit_by_idempotency(idempotency_key)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Control audit lookup failed") from exc
        if prior is not None:
            details = prior["details"]
            if not secrets.compare_digest(str(details.get("request_digest", "")), request_digest):
                raise HTTPException(status_code=409, detail="Idempotency key was used for another command")
            if prior["result_status"] == "DISPATCHING":
                raise HTTPException(
                    status_code=409,
                    detail="Prior command outcome is unresolved; keep the line in HOLD",
                )
            return JSONResponse(
                status_code=int(details.get("http_status", 503)),
                content={
                    "audit_id": prior["audit_id"],
                    "action": prior["action"],
                    "operator_id": prior["operator_id"],
                    "snapshot_id": prior["snapshot_id"],
                    "status": prior["result_status"],
                    "signal_id": prior["signal_id"],
                    "acknowledged": details.get("acknowledged"),
                    "mock_mode": details.get("mock_mode"),
                    "interlock_latched": (
                        details.get("interlock_latched")
                        if isinstance(details.get("interlock_latched"), bool)
                        else industrial_mgr.interlock_latched
                    ),
                    "inference_interlock_latched": details.get(
                        "inference_interlock_latched", failsafe.inference_interlock_latched
                    ),
                    "idempotent_replay": True,
                    "reset_probe_accepted": details.get("reset_probe_accepted"),
                },
            )

        if policy.get("requires_fresh_snapshot"):
            _consume_fresh_snapshot(snapshot_id)

        audit_id = f"AUD_{uuid.uuid4().hex[:12].upper()}"
        recorded = quality_mem.record_control_audit(
            audit_id=audit_id,
            operator_id=operator_id,
            action=action,
            reason=reason,
            confirmation=expected,
            snapshot_id=snapshot_id or None,
            idempotency_key=idempotency_key,
            result_status="DISPATCHING",
            details={"policy_effect": policy["effect"], "request_digest": request_digest},
        )
        if not recorded:
            _latch_control_failure("Control audit write failed before dispatch")
            raise HTTPException(
                status_code=503,
                detail="Control audit write failed; fail-closed HOLD/E-stop was requested",
            )

        try:
            outcome = _dispatch_confirmed_control(action, reason)
        except Exception as exc:
            logger.exception("Control dispatch raised an exception")
            _latch_control_failure("Control dispatch exception; outcome is uncertain")
            outcome = {
                "result_status": "DISPATCH_EXCEPTION",
                "http_status": 503,
                "signal_id": None,
                "acknowledged": False,
                "mock_mode": industrial_mgr.mock_mode,
                "interlock_latched": industrial_mgr.interlock_latched,
                "inference_interlock_latched": failsafe.inference_interlock_latched,
                "error_type": type(exc).__name__,
            }
        final_details = {
            "policy_effect": policy["effect"],
            "request_digest": request_digest,
            "http_status": outcome["http_status"],
            "acknowledged": outcome.get("acknowledged"),
            "reset_probe_accepted": outcome.get("reset_probe_accepted"),
            "mock_mode": outcome.get("mock_mode"),
            "interlock_latched": outcome.get(
                "interlock_latched", industrial_mgr.interlock_latched
            ),
            "inference_interlock_latched": outcome.get("inference_interlock_latched"),
            "error_type": outcome.get("error_type"),
        }
        finalized = quality_mem.finalize_control_audit(
            audit_id,
            outcome["result_status"],
            signal_id=outcome.get("signal_id"),
            details=final_details,
        )
        if not finalized:
            _latch_control_failure("Control audit finalization failed after dispatch")
            raise HTTPException(
                status_code=503,
                detail="Control was dispatched but audit finalization failed; treat the line as HOLD",
            )

        return JSONResponse(
            status_code=outcome["http_status"],
            content={
                "audit_id": audit_id,
                "action": action,
                "operator_id": operator_id,
                "snapshot_id": snapshot_id or None,
                "status": outcome["result_status"],
                "signal_id": outcome.get("signal_id"),
                "acknowledged": outcome.get("acknowledged"),
                "reset_probe_accepted": outcome.get("reset_probe_accepted"),
                "mock_mode": outcome.get("mock_mode"),
                "interlock_latched": industrial_mgr.interlock_latched,
                "inference_interlock_latched": failsafe.inference_interlock_latched,
                "idempotent_replay": False,
            },
        )


@app.post("/api/industrial/estop")
def emergency_stop(reason: str = Form("API trigger")):
    """Legacy unaudited control is deliberately disabled."""
    raise HTTPException(status_code=410, detail="Use POST /api/operations/control")


@app.post("/api/industrial/reset")
def reset_line():
    """Legacy unaudited control is deliberately disabled."""
    raise HTTPException(status_code=410, detail="Use POST /api/operations/control")


@app.post("/api/system/inference-reset")
def reset_inference_interlock():
    """Legacy unaudited control is deliberately disabled."""
    raise HTTPException(status_code=410, detail="Use POST /api/operations/control")


@app.get("/api/system/health-report")
def system_health_report():
    """Detailed system health report including all subsystems."""
    health = failsafe.get_health_report()
    fusion_status = {
        "thermal_online": fusion_manager.sensor_status.thermal_online,
        "profiler_online": fusion_manager.sensor_status.profiler_online,
        "degradation_level": fusion_manager.sensor_status.degradation_level,
    }
    return {
        "failsafe": health,
        "fusion": fusion_status,
        "industrial": industrial_mgr.get_plc_state(),
        "model": {
            "yolo_loaded": predictor.yolo_available,
            "yolo_provider": (
                predictor.yolo_engine.active_provider if predictor.yolo_available else "N/A"
            ),
            "onnx_loaded": predictor.onnx_available,
            "onnx_provider": (
                predictor.onnx_engine.active_provider if predictor.onnx_available else "N/A"
            ),
            "pytorch_device": str(predictor.device),
            "model_version": predictor.model_version,
        }
    }


# =========================================================================
# NEW INDUSTRIAL ENDPOINTS (FINAL ROUND ENHANCEMENTS)
# =========================================================================

@app.post("/api/pipeline/multi-stage")
def run_multi_stage_pipeline(
    part_id: Optional[str] = Form(None),
    sample_name: Optional[str] = Form(None),
    cross_web_pos_mm: float = Form(325.0),
    enable_plc_signal: bool = Form(True)
):
    """
    Execute the complete 7-stage multi-modal industrial inspection workflow:
    1. Web Motion & Encoder Synchronization
    2. Multi-Modal Physical Acquisition (Brightfield, Darkfield, Lock-in Thermography, 3D Laser)
    3. Homography Registration & 5-Channel Fusion
    4. Configured YOLO/ONNX instance segmentation
    5. Prototype battery-electrode metrology policy
    6. Hardware Rejection Interlock
    7. AI Closed-Loop Equipment Diagnostics & Parameter Tuning
    """
    if not SIMULATION_MODE:
        raise HTTPException(
            status_code=404,
            detail="The synthetic multi-stage demonstration endpoint is disabled in production",
        )
    try:
        sid = sample_name or part_id or f"PART_SAMPLE_{int(time.time()*1000)}"
        result = multi_stage_pipeline.execute_inspection(
            sample_id=sid,
            cross_web_pos_mm=cross_web_pos_mm,
            enable_plc_signal=enable_plc_signal,
        )
        return {
            "status": "success",
            "data": result.to_summary_dict()
        }
    except Exception as e:
        logger.error(f"Multi-stage inspection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/libad/protocol")
def libad_protocol():
    """Disclose the LIBAD validation-extension contract without replacing RGB inference."""
    from libad.dataset import dataset_status
    from libad.protocol import LIBAD_CITATION, LIBAD_PAPER_RESULT_NOTE, load_project_identity

    identity = load_project_identity()
    return {
        "brand": identity["brand"],
        "registered_title": identity["registered_title"],
        "tagline": identity["tagline"],
        "citation": LIBAD_CITATION,
        "paper_result_note": LIBAD_PAPER_RESULT_NOTE,
        "dataset": dataset_status(),
        "local_contribution": (
            "Evidence-gated PASS/REJECT/HOLD. DA-Core is the LIBAD authors' baseline, "
            "not a SecureCoating-Vision algorithm."
        ),
        "existing_rgb_path": "unchanged YOLOv8-seg/ONNX surface localization",
        "simulated_adapters": ["thermal", "profilometry"],
        "real_multimodal_lane": ["vis", "xray_l"],
    }


@app.get("/api/libad/demo/{case_id}")
def libad_demo(case_id: int):
    """Four staged industrial cases for the VIS/X-rayL evidence lane."""
    if not SIMULATION_MODE:
        raise HTTPException(
            status_code=404,
            detail="LIBAD protocol-fixture demo is disabled in production",
        )
    if case_id not in {1, 2, 3, 4}:
        raise HTTPException(status_code=422, detail="Demo case must be 1, 2, 3, or 4")
    from libad.demo_cases import run_libad_demo_case

    result = run_libad_demo_case(case_id)
    encoded_frames: Dict[str, str] = {}
    for name, frame in result["frames"].items():
        ok, payload = cv2.imencode(".png", frame)
        if not ok:
            raise HTTPException(status_code=500, detail=f"Failed to encode {name} demo frame")
        encoded_frames[name] = base64.b64encode(payload.tobytes()).decode("ascii")
    response = {key: value for key, value in result.items() if key != "frames"}
    response["frames_png_base64"] = encoded_frames
    return response



@app.get("/api/roll/{roll_id}/map")
def get_roll_defect_map(roll_id: str):
    """Retrieve 2D defect coordinates across the full jumbo roll for digital twin visualization."""
    _validate_identifier(roll_id, "roll_id")
    snapshot = web_synchronizer.get_roll_snapshot(roll_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"Roll {roll_id} is not the active roll")
    return {
        **snapshot,
        "total_records": len(snapshot["defect_records"]),
    }


@app.get("/api/roll/active")
def get_active_roll():
    """Return the authoritative active roll identity and atomic snapshot."""
    snapshot = web_synchronizer.get_roll_snapshot(web_synchronizer.roll.roll_id)
    if snapshot is None:
        raise HTTPException(status_code=503, detail="Active roll state is unavailable")
    return {**snapshot, "total_records": len(snapshot["defect_records"])}


@app.get("/api/roll/{roll_id}/certificate")
def get_roll_certificate(
    roll_id: str,
    batch_id: Optional[str] = None,
    electrode_type: str = "Cathode_LFP"
):
    """
    Generate a provisional tamper-evident quality manifest with SHA-256/HMAC.
    """
    _validate_identifier(roll_id, "roll_id")
    batch_id = batch_id or web_synchronizer.roll.batch_id
    _validate_identifier(batch_id, "batch_id")
    return _build_roll_certificate_payload(roll_id, batch_id, electrode_type)


@app.post("/api/diagnostics/root-cause")
def get_root_cause_diagnostics(batch_id: Optional[str] = None):
    """
    AI Closed-Loop Diagnostics: attributes coating defects to upstream equipment
    and recommends parameter adjustments (Slot-Die gap, Oven Zone temperatures, Mixer vacuum).
    """
    batch_id = batch_id or web_synchronizer.roll.batch_id
    _validate_identifier(batch_id, "batch_id")
    if batch_id != web_synchronizer.roll.batch_id:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} is not active")
    report = root_cause_engine.diagnose_batch(web_synchronizer.roll_defect_map)
    return report.to_dict()


@app.get("/api/spc/passport")
def get_digital_battery_passport():
    """
    Generate a provisional telemetry manifest. This is not a regulatory DPP
    and does not claim Cpk/Ppk without subgroup measurement data.
    """
    passport = multi_stage_pipeline.get_gigafactory_spc_summary()
    return passport


@app.get("/api/spc/slitting-yield")
def get_slitting_yield_plan():
    """
    Smart Slitting Yield Optimizer:
    Evaluates EV/ESS grade recovery yield and calculates optimal splice cutting positions.
    """
    passport = multi_stage_pipeline.get_gigafactory_spc_summary()
    return {
        "slitting_optimization": passport.get("slitting_optimization", {}),
        "lane_cpk_summary": passport.get("six_sigma_quality_summary", {}),
        "mechanical_anomalies": passport.get("mechanical_health_diagnostics", [])
    }


if __name__ == "__main__":
    host = app_config.get("server", {}).get("host", "0.0.0.0")
    port = app_config.get("server", {}).get("port", 8000)
    uvicorn.run(app, host=host, port=port)
