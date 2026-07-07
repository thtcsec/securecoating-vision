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
import yaml
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.predictor import CoatingPredictor
from inference.postprocess import extract_defects_from_mask, grade_coating
from inference.sensor_fusion import SensorFusionManager
from inference.failsafe import FailSafeManager
from traceability.quality_memory import QualityMemory
from industrial.protocol_manager import IndustrialProtocolManager

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

# --- Configuration Loading ---
CONFIG_PATH = os.environ.get("MODEL_CONFIG", "configs/model.yaml")
APP_CONFIG_PATH = os.environ.get("APP_CONFIG", "configs/app.yaml")

with open(CONFIG_PATH, "r") as f:
    model_config = yaml.safe_load(f)
with open(APP_CONFIG_PATH, "r") as f:
    app_config = yaml.safe_load(f)

# --- Initialize Core Engines ---
# 1. Predictor (PyTorch + ONNX fallback)
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


# --- Response Models ---
class InspectionResponse(BaseModel):
    part_id: str
    batch_id: str
    status: str
    passed: bool
    reject_reasons: list
    latency_ms: float
    defects_found: list
    fallback_active: bool
    system_state: str
    gate_action: str
    engine: str


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    status: str
    device: str
    model_version: str
    onnx_available: bool
    system_state: str
    sensors: Dict[str, str]


# --- Endpoints ---
@app.get("/health", response_model=HealthResponse)
def health_check():
    """System health check with component status."""
    health = failsafe.get_health_report()
    onnx_loaded = (
        predictor.onnx_engine is not None and predictor.onnx_engine.is_loaded
    )
    return {
        "status": "HEALTHY" if health["system_state"] != "OFFLINE" else "DEGRADED",
        "device": str(predictor.device),
        "model_version": model_config.get("model", {}).get("version", "2.0.0"),
        "onnx_available": onnx_loaded,
        "system_state": health["system_state"],
        "sensors": health["sensors"]
    }


@app.post("/api/inspect", response_model=InspectionResponse)
def inspect(
    background_tasks: BackgroundTasks,
    batch_id: str = Form("BATCH_DEFAULT"),
    part_id: str = Form("PART_000"),
    simulate_defect: Optional[str] = Form(None),
    thermal_online: bool = Form(True),
    profiler_online: bool = Form(True)
):
    """
    Full inspection pipeline: acquire -> fuse -> infer -> grade -> signal.
    
    Processes a simulated coating inspection with multi-sensor fusion,
    YOLOv8 segmentation, quality grading, and PLC reject signaling.
    """
    # 1. Simulate sensor acquisition (mock camera frame)
    h, w = 1024, 1024
    optical = np.ones((h, w, 3), dtype=np.uint8) * 180

    # Add base texture
    noise = np.random.randint(-8, 8, (h, w, 3), dtype=np.int16)
    optical = np.clip(optical.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Optional defect simulation overlays
    if simulate_defect == "scratch":
        cv2.line(optical, (150, 300), (800, 320), (40, 40, 40), thickness=6)
    elif simulate_defect == "void":
        cv2.circle(optical, (500, 600), 80, (80, 60, 80), -1)
    elif simulate_defect == "blister":
        cv2.circle(optical, (400, 450), 60, (210, 215, 210), -1)
    elif simulate_defect == "delamination":
        pts = np.array([[400, 400], [600, 380], [650, 550], [420, 580]], dtype=np.int32)
        cv2.fillPoly(optical, [pts], (100, 90, 110))

    # 2. Multi-source sensor fusion
    fusion_result = fusion_manager.fuse(
        rgb_image=optical,
        thermal_online=thermal_online,
        profiler_online=profiler_online
    )

    thermal = fusion_result.thermal_frame
    height = fusion_result.height_frame

    # 3. Update fail-safe sensor status
    failsafe.health.thermal_sensor_ok = thermal_online
    failsafe.health.profiler_sensor_ok = profiler_online

    # 4. Safe prediction with fail-safe wrapping
    result = failsafe.safe_predict(predictor, optical, thermal, height)

    # 5. Postprocess: extract physical defect measurements
    seg_mask = result.get("segmentation_mask", np.zeros((h, w), dtype=np.uint8))
    defects = extract_defects_from_mask(seg_mask, height, pixel_to_mm_ratio=0.1)

    # Override with simulated defect if model didn't detect (for demo)
    if simulate_defect and len(defects) == 0:
        class_map = {"scratch": 1, "void": 2, "blister": 3, "delamination": 4}
        cid = class_map.get(simulate_defect, 1)
        cv2.circle(seg_mask, (512, 512), 100, int(cid), -1)
        defects = extract_defects_from_mask(seg_mask, height, pixel_to_mm_ratio=0.1)
        # Fallback stub if still empty
        if len(defects) == 0:
            defects.append({
                "defect_id": f"{simulate_defect}_0",
                "class_id": cid,
                "class_name": simulate_defect,
                "bbox": [412, 412, 200, 200],
                "length_mm": 8.5 if simulate_defect == "scratch" else 2.0,
                "width_mm": 1.0,
                "area_mm2": 5.2 if simulate_defect in ["void", "delamination"] else 2.0,
                "peak_height_um": 180.0 if simulate_defect == "blister" else 0.0
            })

    # 6. Grade the coating against quality rules
    grading_rules = model_config.get("inference", {}).get("grading", {})
    grade = grade_coating(defects, grading_rules)

    # 7. Industrial protocol signaling (PLC/MES)
    industrial_result = industrial_mgr.process_inspection_result(
        part_id=part_id,
        batch_id=batch_id,
        defects=defects,
        grade_result=grade
    )

    # 8. Log to Quality Memory database
    max_len = max([d["length_mm"] for d in defects]) if defects else 0.0
    max_area = max([d["area_mm2"] for d in defects]) if defects else 0.0
    peak_h = max([d["peak_height_um"] for d in defects]) if defects else 0.0

    quality_mem.add_entry(
        batch_id=batch_id,
        part_id=part_id,
        has_defect=not grade["passed"],
        defect_class=simulate_defect or "none",
        max_length=max_len,
        max_area=max_area,
        peak_height=peak_h,
        latency=result.get("latency_ms", 0.0),
        fallback=result.get("fallback_active", False)
    )

    return {
        "part_id": part_id,
        "batch_id": batch_id,
        "status": result.get("status", "Unknown"),
        "passed": grade["passed"],
        "reject_reasons": grade["reject_reasons"],
        "latency_ms": round(result.get("latency_ms", 0.0), 2),
        "defects_found": defects,
        "fallback_active": result.get("fallback_active", False),
        "system_state": result.get("system_state", "OPTIMAL"),
        "gate_action": industrial_result["gate_action"],
        "engine": result.get("engine", "PyTorch (Fusion)")
    }


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
            "onnx_loaded": predictor.onnx_engine is not None and predictor.onnx_engine.is_loaded,
            "onnx_provider": predictor.onnx_engine.active_provider if predictor.onnx_engine else "N/A",
            "pytorch_device": str(predictor.device),
        }
    }


if __name__ == "__main__":
    host = app_config.get("server", {}).get("host", "0.0.0.0")
    port = app_config.get("server", {}).get("port", 8000)
    uvicorn.run(app, host=host, port=port)
