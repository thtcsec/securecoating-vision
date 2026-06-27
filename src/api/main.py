import os
import sys
import yaml
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from pydantic import BaseModel
import numpy as np
import cv2

# Adjust pathing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.predictor import CoatingPredictor
from inference.postprocess import extract_defects_from_mask, grade_coating
from traceability.quality_memory import QualityMemory

app = FastAPI(
    title="SecureCoating-Vision API",
    description="REST API for multi-sensor fusion coating inspection and defect tracking.",
    version="1.0.0"
)

# Load configuration paths
CONFIG_PATH = os.environ.get("MODEL_CONFIG", "configs/model.yaml")
APP_CONFIG_PATH = os.environ.get("APP_CONFIG", "configs/app.yaml")

with open(CONFIG_PATH, "r") as f:
    model_config = yaml.safe_load(f)
with open(APP_CONFIG_PATH, "r") as f:
    app_config = yaml.safe_load(f)

# Initialize engines
predictor = CoatingPredictor(model_config)
db_path = app_config.get("paths", {}).get("db_path", "data/quality_history.db")
quality_mem = QualityMemory(db_path)

# Mock industrial output signal handler
def send_plc_reject_pulse(part_id: str, reason: str):
    """Simulates sending a rejection trigger via Modbus TCP / OPC UA registers."""
    logger_enabled = app_config.get("industrial_io", {}).get("enabled", True)
    if logger_enabled:
        plc_ip = app_config["industrial_io"]["plc_ip"]
        reject_reg = app_config["industrial_io"]["modbus"]["register_reject"]
        print(f"[INDUSTRIAL I/O] Signal sent to PLC at {plc_ip}: REJECT Part {part_id}. Reason: {reason}.")
        print(f"[INDUSTRIAL I/O] Modbus Reg [{reject_reg}] set to 1 (Reject Gate Triggered)")

class InspectionResponse(BaseModel):
    part_id: str
    batch_id: str
    status: str
    passed: bool
    reject_reasons: list
    latency_ms: float
    defects_found: list
    fallback_active: bool

@app.get("/health")
def health_check():
    return {
        "status": "HEALTHY",
        "device": str(predictor.device),
        "model_version": model_config.get("model", {}).get("version", "1.0.0"),
        "fallback_status": "ENABLED"
    }

@app.post("/api/inspect", response_model=InspectionResponse)
def inspect(
    background_tasks: BackgroundTasks,
    batch_id: str = Form("BATCH_DEFAULT"),
    part_id: str = Form("PART_000"),
    simulate_defect: str = Form(None)
):
    """
    Accepts inspect queries and processes the active sensor frames.
    Note: For simplified demo/dashboard compatibility, we generate mock data matching
    the request parameters or simulate a defect type if requested.
    """
    # 1. Acquire inputs (Simulating input acquisition from camera caches)
    h, w = 1024, 1024
    optical = np.ones((h, w, 3), dtype=np.uint8) * 180 # Clean background
    thermal = np.ones((h, w), dtype=np.float32) * 40.0
    height = np.zeros((h, w), dtype=np.float32)
    
    # Optional simulation overlays
    if simulate_defect == "scratch":
        cv2.line(optical, (200, 300), (800, 320), (50, 50, 50), thickness=6)
    elif simulate_defect == "void":
        cv2.circle(thermal, (500, 600), 80, 28.0, -1)
    elif simulate_defect == "blister":
        cv2.circle(height, (350, 450), 50, 180.0, -1)
    elif simulate_defect == "delamination":
        # Multi-sensor defect
        cv2.circle(thermal, (600, 200), 120, 25.0, -1)
        cv2.circle(height, (600, 200), 120, 12.0, -1)

    # 2. Run prediction network
    result = predictor.predict(optical, thermal, height)
    
    # 3. Postprocess and extract physical defect sizes
    defects = extract_defects_from_mask(result["segmentation_mask"], height, pixel_to_mm_ratio=0.1)
    
    # Override/ensure class matches if simulated
    if simulate_defect and len(defects) == 0:
        # Fallback generator for stubs
        defects.append({
            "defect_id": f"{simulate_defect}_0",
            "class_id": 1,
            "class_name": simulate_defect,
            "bbox": [100, 100, 50, 50],
            "length_mm": 8.5 if simulate_defect == "scratch" else 1.0,
            "width_mm": 1.0,
            "area_mm2": 5.2 if simulate_defect in ["void", "delamination"] else 1.0,
            "peak_height_um": 180.0 if simulate_defect == "blister" else 0.0
        })

    # 4. Grade the defects according to configuration rules
    grading_rules = model_config.get("inference", {}).get("grading", {})
    grade = grade_coating(defects, grading_rules)
    
    # 5. Log details to Quality Memory
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
        latency=result["latency_ms"],
        fallback=result["fallback_active"]
    )
    
    # 6. Industrial Automation rejection loop
    if not grade["passed"]:
        reason_str = ", ".join(grade["reject_reasons"])
        background_tasks.add_task(send_plc_reject_pulse, part_id, reason_str)

    return {
        "part_id": part_id,
        "batch_id": batch_id,
        "status": result["status"],
        "passed": grade["passed"],
        "reject_reasons": grade["reject_reasons"],
        "latency_ms": result["latency_ms"],
        "defects_found": defects,
        "fallback_active": result["fallback_active"]
    }

@app.get("/api/batch/{batch_id}/stats")
def get_batch_stats(batch_id: str):
    """Endpoint for MES query to retrieve aggregated quality indicators."""
    stats = quality_mem.get_batch_stats(batch_id)
    if stats["total"] == 0:
        raise HTTPException(status_code=404, detail=f"No inspection data found for batch {batch_id}")
    return stats

@app.get("/api/batch/{batch_id}/spc")
def get_batch_spc(batch_id: str):
    """Endpoint to check process control alarm state."""
    return quality_mem.check_spc_alarms(batch_id)

if __name__ == "__main__":
    host = app_config.get("server", {}).get("host", "0.0.0.0")
    port = app_config.get("server", {}).get("port", 8000)
    uvicorn.run(app, host=host, port=port)
