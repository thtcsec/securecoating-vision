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
- POST /api/inspect               -> Run full inspection pipeline
- GET  /api/batch/{id}/stats      -> Batch quality statistics
- GET  /api/batch/{id}/spc        -> SPC alarm status
- GET  /api/industrial/state      -> PLC register state
- GET  /api/industrial/signals    -> Signal history
- POST /api/industrial/estop      -> Emergency stop
- POST /api/industrial/reset      -> Reset line after E-Stop
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
from io import BytesIO
import yaml
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
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
    allow_headers=["Content-Type", "X-API-Key"],
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

TEST_SET_DIR = os.path.join(PROJECT_ROOT, "data", "test_set", "images")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _validate_identifier(value: str, field_name: str) -> str:
    if not IDENTIFIER_PATTERN.fullmatch(value or ""):
        raise HTTPException(
            status_code=422,
            detail=f"{field_name} must be 1-128 characters from the approved identifier set",
        )
    return value


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
    """Load a demo image from data/test_set/images (basename only)."""
    safe_name = os.path.basename(sample_name)
    if safe_name != sample_name or len(sample_name) > 255:
        raise HTTPException(status_code=422, detail="sample_name must be a bounded basename")
    path = os.path.join(TEST_SET_DIR, safe_name)
    if not os.path.isfile(path):
        raise HTTPException(
            status_code=404,
            detail=f"Sample '{safe_name}' not found under data/test_set/images",
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


@app.get("/health", response_model=HealthResponse)
def health_check():
    """System health check with component status."""
    health = failsafe.get_health_report()
    industrial_state = industrial_mgr.get_plc_state()
    ready = (
        health["system_state"] == "OPTIMAL"
        and not industrial_state["interlock_latched"]
        and quality_mem.healthy
        and (predictor.yolo_available or predictor.onnx_available)
        and industrial_mgr.transport_ready
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
    }
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
    2. `sample_name` from data/test_set/images (e.g. defect_val_00000.jpg)
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
    if trace_written and not quality_mem.update_decision(run_id, final_gate_action, result_state):
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
def list_demo_samples():
    """List demo images available under data/test_set/images."""
    if not SIMULATION_MODE:
        raise HTTPException(status_code=404, detail="Demo samples are disabled in production")
    if not os.path.isdir(TEST_SET_DIR):
        return {"samples": [], "directory": TEST_SET_DIR}
    samples = sorted(
        f
        for f in os.listdir(TEST_SET_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )
    return {"samples": samples, "count": len(samples), "directory": "data/test_set/images"}


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


@app.post("/api/industrial/estop")
def emergency_stop(reason: str = Form("API trigger")):
    """Trigger emergency stop on production line."""
    signal = industrial_mgr.emergency_stop(reason)
    return JSONResponse(
        status_code=200 if signal.acknowledged or industrial_mgr.mock_mode else 503,
        content={
            "status": "SIMULATED" if industrial_mgr.mock_mode else (
                "ACKNOWLEDGED" if signal.acknowledged else "UNCONFIRMED"
            ),
            "action": "EMERGENCY_STOP",
            "reason": reason,
            "signal_id": signal.signal_id,
            "interlock_latched": industrial_mgr.interlock_latched,
        },
    )


@app.post("/api/industrial/reset")
def reset_line():
    """Reset production line after emergency stop."""
    signal = industrial_mgr.reset_line()
    cleared = not industrial_mgr.interlock_latched
    return JSONResponse(
        status_code=200 if cleared else 503,
        content={
            "status": "SIMULATED" if industrial_mgr.mock_mode and cleared else (
                "ACKNOWLEDGED" if signal.acknowledged and cleared else "UNCONFIRMED"
            ),
            "action": "RESET",
            "signal_id": signal.signal_id,
            "interlock_latched": industrial_mgr.interlock_latched,
        },
    )


@app.post("/api/system/inference-reset")
def reset_inference_interlock():
    """Authorize one trained-model recovery probe after an inference fault."""
    if not (predictor.yolo_available or predictor.onnx_available):
        raise HTTPException(status_code=409, detail="No trained model is available for a reset probe")
    accepted = failsafe.request_inference_reset_probe()
    return JSONResponse(
        status_code=202 if accepted else 409,
        content={
            "status": "RESET_PROBE_ARMED" if accepted else "RESET_BLOCKED",
            "inference_interlock_latched": failsafe.inference_interlock_latched,
        },
    )


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
        metric_provenance=f"SQLite batch={batch_id}; roll ledger={roll_id}",
    )
    return cert.to_dict()


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
