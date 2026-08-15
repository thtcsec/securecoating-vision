"""
SecureCoating-Vision: FastAPI REST API
=======================================
Production-grade REST API for multi-sensor fusion coating inspection.

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
import uuid
import yaml
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import numpy as np
import cv2

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
app = FastAPI(
    title="SecureCoating-Vision API",
    description=(
        "Multi-sensor fusion coating inspection system with ONNX Runtime inference, "
        "sensor fusion, fail-safe degradation, and industrial PLC signaling."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional API key (set SECURECOATING_API_KEY to enable)
API_KEY = os.environ.get("SECURECOATING_API_KEY", "").strip()


@app.middleware("http")
async def optional_api_key_guard(request: Request, call_next):
    if API_KEY and request.url.path.startswith("/api/"):
        provided = request.headers.get("x-api-key", "")
        if provided != API_KEY:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing X-API-Key"},
            )
    return await call_next(request)

# --- Configuration Loading ---
CONFIG_PATH = os.environ.get("MODEL_CONFIG", "configs/model.yaml")
APP_CONFIG_PATH = os.environ.get("APP_CONFIG", "configs/app.yaml")

with open(CONFIG_PATH, "r") as f:
    model_config = yaml.safe_load(f)
with open(APP_CONFIG_PATH, "r") as f:
    app_config = yaml.safe_load(f)

TEST_SET_DIR = os.path.join(PROJECT_ROOT, "data", "test_set", "images")


def _decode_upload(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes to BGR uint8."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image")
    return image


def _load_sample_image(sample_name: str) -> np.ndarray:
    """Load a demo image from data/test_set/images (basename only)."""
    safe_name = os.path.basename(sample_name)
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


def _attach_confidences(defects: list, detections: list) -> list:
    """Attach model confidence scores onto graded defect records."""
    if not detections:
        for d in defects:
            d.setdefault("confidence", None)
        return defects
    by_class: Dict[str, float] = {}
    for det in detections:
        name = det.get("class_name", "")
        conf = float(det.get("confidence", 0.0))
        by_class[name] = max(by_class.get(name, 0.0), conf)
    for d in defects:
        d["confidence"] = round(by_class.get(d.get("class_name", ""), 0.0), 4) or None
    return defects


# --- Initialize Core Engines ---
# 1. Predictor (ONNX YOLOv8-seg primary, PyTorch fusion fallback)
predictor = CoatingPredictor(model_config)

# 2. Sensor Fusion Manager
fusion_manager = SensorFusionManager(
    target_size=(1024, 1024),
    enable_mock=True
)

# 3. Fail-Safe Manager
failsafe = FailSafeManager(
    max_inference_timeout_ms=5000.0,
    enable_frame_validation=True,
    enable_auto_recovery=True
)

# 4. Quality Memory (SQLite traceability)
db_path = app_config.get("paths", {}).get("db_path", "data/quality_history.db")
quality_mem = QualityMemory(db_path)

# 5. Industrial Protocol Manager (OPC UA / Modbus TCP)
industrial_config = app_config.get("industrial_io", {})
industrial_mgr = IndustrialProtocolManager(industrial_config)

# 6. Web Motion & Continuous Roll Synchronizer
web_synchronizer = WebSynchronizer()

# 7. Physics-Informed Battery Electrode Metrology
electrode_metrology = ElectrodeMetrologyEngine(pixel_to_mm_ratio=0.1)

# 8. AI Closed-Loop Root-Cause Diagnostics
root_cause_engine = RootCauseDiagnosticEngine()

# 9. Multi-Stage Industrial Pipeline Orchestrator
multi_stage_pipeline = MultiStageIndustrialPipeline(
    predictor=predictor,
    fusion_manager=fusion_manager,
    failsafe_manager=failsafe,
    industrial_manager=industrial_mgr,
    web_synchronizer=web_synchronizer,
    pixel_to_mm_ratio=0.1
)


# --- Response Models ---
class InspectionResponse(BaseModel):
    part_id: str
    batch_id: str
    run_id: str
    status: str
    passed: bool
    reject_reasons: list
    latency_ms: float
    defects_found: list
    fallback_active: bool
    system_state: str
    gate_action: str
    engine: str
    model_version: str = "2.0.0"
    detections: list = []


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
@app.get("/health", response_model=HealthResponse)
def health_check():
    """System health check with component status."""
    health = failsafe.get_health_report()
    return {
        "status": "HEALTHY" if health["system_state"] != "OFFLINE" else "DEGRADED",
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
                else "PyTorchFusion"
            )
        ),
        "system_state": health["system_state"],
        "sensors": health["sensors"]
    }


@app.post("/api/inspect", response_model=InspectionResponse)
async def inspect(
    background_tasks: BackgroundTasks,
    batch_id: str = Form("BATCH_DEFAULT"),
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

    if image is not None and image.filename:
        optical = _decode_upload(await image.read())
        using_real_image = True
    elif sample_name:
        optical = _load_sample_image(sample_name)
        using_real_image = True
    else:
        optical = _synthetic_optical(1024, 1024, simulate_defect)

    h, w = optical.shape[:2]

    # 2. Multi-source sensor fusion
    fusion_result = fusion_manager.fuse(
        rgb_image=optical,
        thermal_online=thermal_online,
        profiler_online=profiler_online,
    )

    thermal = fusion_result.thermal_frame
    height = fusion_result.height_frame

    # 3. Update fail-safe sensor status from THIS request (non-sticky intent)
    failsafe.health.thermal_sensor_ok = thermal_online
    failsafe.health.profiler_sensor_ok = profiler_online
    failsafe.health.rgb_sensor_ok = True
    failsafe._update_system_state()

    # 4. Safe prediction with fail-safe wrapping
    result = failsafe.safe_predict(predictor, optical, thermal, height)

    # 5. Postprocess: extract physical defect measurements
    seg_mask = result.get("segmentation_mask", np.zeros((h, w), dtype=np.uint8))
    defects = extract_defects_from_mask(seg_mask, height, pixel_to_mm_ratio=0.1)

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
        defects = extract_defects_from_mask(seg_mask, height, pixel_to_mm_ratio=0.1)

    # 6. Grade the coating against quality rules
    grading_rules = model_config.get("inference", {}).get("grading", {})
    grade = grade_coating(defects, grading_rules)

    # 7. Industrial protocol signaling (PLC/MES)
    industrial_result = industrial_mgr.process_inspection_result(
        part_id=part_id,
        batch_id=batch_id,
        defects=defects,
        grade_result=grade,
    )

    # 8. Log to Quality Memory database
    max_len = max([d["length_mm"] for d in defects]) if defects else 0.0
    max_area = max([d["area_mm2"] for d in defects]) if defects else 0.0
    peak_h = max([d["peak_height_um"] for d in defects]) if defects else 0.0
    primary_class = (
        simulate_defect
        or (defects[0]["class_name"] if defects else "none")
    )
    detections = result.get("detections", [])
    defects = _attach_confidences(defects, detections)
    run_id = f"RUN_{uuid.uuid4().hex[:12].upper()}"

    quality_mem.add_entry(
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
    )

    return {
        "part_id": part_id,
        "batch_id": batch_id,
        "run_id": run_id,
        "status": result.get("status", "Unknown"),
        "passed": grade["passed"],
        "reject_reasons": grade["reject_reasons"],
        "latency_ms": round(result.get("latency_ms", 0.0), 2),
        "defects_found": defects,
        "fallback_active": result.get("fallback_active", False),
        "system_state": result.get("system_state", "OPTIMAL"),
        "gate_action": industrial_result["gate_action"],
        "engine": result.get("engine", "unknown"),
        "model_version": result.get("model_version", predictor.model_version),
        "detections": detections,
    }


@app.get("/api/samples")
def list_demo_samples():
    """List demo images available under data/test_set/images."""
    if not os.path.isdir(TEST_SET_DIR):
        return {"samples": [], "directory": TEST_SET_DIR}
    samples = sorted(
        f
        for f in os.listdir(TEST_SET_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )
    return {"samples": samples, "count": len(samples), "directory": "data/test_set/images"}


@app.get("/api/metrics/latency")
def latency_metrics(batch_id: Optional[str] = None, limit: int = 200):
    """Latency p50/p95 vs 35ms competition target."""
    return quality_mem.get_latency_stats(batch_id=batch_id, limit=limit)


@app.get("/api/batch/{batch_id}/stats")
def get_batch_stats(batch_id: str):
    """Retrieve aggregated quality indicators for MES integration."""
    stats = quality_mem.get_batch_stats(batch_id)
    if stats["total"] == 0:
        raise HTTPException(status_code=404, detail=f"No inspection data for batch {batch_id}")
    return stats


@app.get("/api/batch/{batch_id}/spc")
def get_batch_spc(batch_id: str):
    """Check Statistical Process Control alarm state."""
    return quality_mem.check_spc_alarms(batch_id)


@app.get("/api/industrial/state")
def get_industrial_state():
    """Get current PLC register state and OPC UA node values."""
    return industrial_mgr.get_plc_state()


@app.get("/api/industrial/signals")
def get_signal_history(limit: int = 50):
    """Get recent industrial signal history."""
    return industrial_mgr.get_signal_history(limit=limit)


@app.post("/api/industrial/estop")
def emergency_stop(reason: str = Form("API trigger")):
    """Trigger emergency stop on production line."""
    industrial_mgr.emergency_stop(reason)
    return {"status": "E-STOP ACTIVATED", "reason": reason}


@app.post("/api/industrial/reset")
def reset_line():
    """Reset production line after emergency stop."""
    industrial_mgr.reset_line()
    return {"status": "LINE RESET", "message": "Production line resumed"}


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
    4. Edge AI TensorRT/ONNX Instance Segmentation
    5. Physics-Informed Battery Electrode Metrology & T/CIAPS 0006 / QC/T 743 Audit
    6. Hardware Rejection Interlock
    7. AI Closed-Loop Equipment Diagnostics & Parameter Tuning
    """
    try:
        sample_name = sample_id or f"PART_SAMPLE_{int(time.time()*1000)}"
        result = pipeline.execute_inspection(sample_id=sample_name)
        return {
            "status": "success",
            "data": result.to_summary_dict()
        }
    except Exception as e:
        logger.error(f"Multi-stage inspection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/roll/{roll_id}/map")
def get_roll_defect_map(roll_id: str):
    """Retrieve 2D defect coordinates across the full jumbo roll for digital twin visualization."""
    summary = web_synchronizer.get_roll_defect_summary()
    return {
        "summary": summary,
        "defect_records": web_synchronizer.roll_defect_map,
        "total_records": len(web_synchronizer.roll_defect_map)
    }


@app.get("/api/roll/{roll_id}/certificate")
def get_roll_certificate(
    roll_id: str,
    batch_id: str = "BATCH_2026_08A",
    electrode_type: str = "Cathode_LFP"
):
    """
    Generate an official, tamper-evident Battery Electrode Quality Inspection Certificate
    with SHA-256 cryptographic verification and GB/T 38823-2020 compliance audit.
    """
    summary = web_synchronizer.get_roll_defect_summary()
    spc = quality_mem.check_spc_alarms(batch_id)
    root_cause = root_cause_engine.diagnose_batch(web_synchronizer.roll_defect_map)

    cert = RollCertificateGenerator.build_certificate(
        roll_id=roll_id,
        batch_id=batch_id,
        inspected_length_m=summary["inspected_length_m"],
        total_length_m=summary["total_roll_length_m"],
        defect_records=web_synchronizer.roll_defect_map,
        spc_status=spc.get("status", "IN CONTROL"),
        root_cause_summary=root_cause.primary_root_cause,
        electrode_type=electrode_type
    )
    return cert.to_dict()


@app.post("/api/diagnostics/root-cause")
def get_root_cause_diagnostics(batch_id: str = "BATCH_2026_08A"):
    """
    AI Closed-Loop Diagnostics: attributes coating defects to upstream equipment
    and recommends parameter adjustments (Slot-Die gap, Oven Zone temperatures, Mixer vacuum).
    """
    report = root_cause_engine.diagnose_batch(web_synchronizer.roll_defect_map)
    return report.to_dict()


if __name__ == "__main__":
    host = app_config.get("server", {}).get("host", "0.0.0.0")
    port = app_config.get("server", {}).get("port", 8000)
    uvicorn.run(app, host=host, port=port)
