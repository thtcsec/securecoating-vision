"""
SecureCoating-Vision: Streamlit Quality Inspection Dashboard
=============================================================
Interactive real-time dashboard for coating defect inspection.

Features:
- 3-column multi-source sensor display (RGB, Thermal, 3D Height)
- Defect overlay visualization with segmentation mask
- SPC (Statistical Process Control) line charts
- Batch trend analytics
- Digital recalibration interface (homography sliders)
- Industrial I/O monitoring panel
- Quality traceability log viewer
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import yaml
import time
import datetime

# Page configuration (must be first Streamlit call)
st.set_page_config(
    page_title="SecureCoating-Vision Console",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium dark theme
st.markdown("""
<style>
    .reportview-container { background: #0F1219; }
    .metric-card {
        background: #181D28;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #2B3548;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        text-align: center;
    }
    .metric-title {
        font-size: 13px;
        color: #8C9BAE;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 28px;
        color: #00F2FE;
        font-weight: 700;
    }
    .status-optimal { color: #00E676; font-weight: 700; }
    .status-degraded { color: #FFD600; font-weight: 700; }
    .status-fail { color: #FF1744; font-weight: 700; }
    .sensor-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-online { background: #1B5E20; color: #00E676; }
    .badge-offline { background: #B71C1C; color: #FF8A80; }
</style>
""", unsafe_allow_html=True)

# Path resolution for module imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)
os.chdir(PROJECT_ROOT)

from inference.predictor import CoatingPredictor
from inference.postprocess import extract_defects_from_mask, grade_coating
from inference.sensor_fusion import SensorFusionManager
from inference.failsafe import FailSafeManager
from traceability.quality_memory import QualityMemory
from industrial.protocol_manager import IndustrialProtocolManager

# Load configuration
with open("configs/model.yaml", "r") as f:
    model_config = yaml.safe_load(f)
with open("configs/app.yaml", "r") as f:
    app_config = yaml.safe_load(f)

# Initialize core components (cached for performance)
@st.cache_resource
def init_predictor():
    return CoatingPredictor(model_config)

@st.cache_resource
def init_fusion():
    return SensorFusionManager(target_size=(1024, 1024), enable_mock=True)

@st.cache_resource
def init_failsafe():
    return FailSafeManager(max_inference_timeout_ms=5000.0)

@st.cache_resource
def init_industrial():
    return IndustrialProtocolManager(app_config.get("industrial_io", {}))

predictor = init_predictor()
fusion_mgr = init_fusion()
failsafe = init_failsafe()
industrial_mgr = init_industrial()

DB_PATH = app_config.get("paths", {}).get("db_path", "data/quality_history.db")
quality_mem = QualityMemory(DB_PATH)

# ============================================================
# APP HEADER
# ============================================================
st.markdown(
    "<h1 style='text-align:center; color:#FFFFFF; font-weight:800;'>"
    "🔍 SecureCoating-Vision</h1>",
    unsafe_allow_html=True
)
if predictor.yolo_available:
    engine_badge = f"YOLO ready ({predictor.yolo_engine.active_provider})"
elif predictor.onnx_available:
    engine_badge = f"ONNX ready ({predictor.onnx_engine.active_provider})"
else:
    engine_badge = "No weights — PyTorch fusion fallback"
st.markdown(
    "<p style='text-align:center; color:#8C9BAE; font-size:15px; margin-bottom:8px;'>"
    "Multi-Sensor Fusion Inspection Console &middot; Real-Time Quality Analytics "
    "&middot; Industrial I/O Control</p>",
    unsafe_allow_html=True
)
st.caption(
    f"Model v{predictor.model_version} · Engine: {engine_badge}"
)

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.markdown("<h2 style='color:#FFF;'>System Control Panel</h2>", unsafe_allow_html=True)
active_batch = st.sidebar.text_input("Active Batch ID", value="BATCH_2026_07A")
part_num = st.sidebar.number_input("Starting Part Number", min_value=1, value=10, step=1)

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='color:#FFF;'>Backend</h3>", unsafe_allow_html=True)
use_api_backend = st.sidebar.toggle("Use REST API backend", value=True)
api_base = st.sidebar.text_input("API Base URL", value="http://127.0.0.1:8000")
if use_api_backend:
    try:
        import requests as _req
        health = _req.get(f"{api_base.rstrip('/')}/health", timeout=3).json()
        st.sidebar.success(
            f"API: {health.get('status')} · {health.get('primary_engine', '?')}"
        )
        lat = _req.get(f"{api_base.rstrip('/')}/api/metrics/latency", timeout=3).json()
        if lat.get("count", 0) > 0:
            st.sidebar.caption(
                f"Latency p50={lat['p50_ms']}ms · p95={lat['p95_ms']}ms · "
                f"≤35ms: {lat['within_target_pct']}%"
            )
    except Exception as exc:
        st.sidebar.warning(f"API unreachable — falling back to local engine ({exc.__class__.__name__})")
        use_api_backend = False

# Sensor health toggles
st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='color:#FFF;'>Hardware Health Simulation</h3>", unsafe_allow_html=True)
thermal_connected = st.sidebar.toggle("LWIR Thermal Camera Online", value=True)
depth_connected = st.sidebar.toggle("3D Laser Profiler Online", value=True)

# Defect simulation
st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='color:#FFF;'>Image Source</h3>", unsafe_allow_html=True)
image_source = st.sidebar.radio(
    "Input Mode:",
    ["Upload Image", "Camera Simulation"],
    index=0
)

uploaded_file = None
selected_test_sample = None
if image_source == "Upload Image":
    uploaded_file = st.sidebar.file_uploader(
        "Upload coating/surface image",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        help="Upload a real coating surface image for AI inspection"
    )
    test_dir = os.path.join(PROJECT_ROOT, "data", "test_set", "images")
    test_samples = []
    if os.path.isdir(test_dir):
        test_samples = sorted(
            f for f in os.listdir(test_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
    selected_test_sample = st.sidebar.selectbox(
        "Or inject from test_set:",
        options=["(none)"] + test_samples,
        index=1 if test_samples else 0,
        help="Pick a labeled demo image then click Trigger Physical Scan",
    )
    if selected_test_sample and selected_test_sample != "(none)" and uploaded_file is None:
        st.session_state["demo_sample_path"] = os.path.join(test_dir, selected_test_sample)
        st.session_state["_last_demo_path"] = st.session_state["demo_sample_path"]
        st.sidebar.caption(f"Ready: `{selected_test_sample}` — press Trigger to inspect")
    st.sidebar.caption("Tip: upload your own photo OR pick a test_set image above")

if image_source == "Camera Simulation":
    st.sidebar.markdown("<h4 style='color:#FFF;'>Defect Injection</h4>", unsafe_allow_html=True)
    selected_defect = st.sidebar.selectbox(
        "Simulate Defect Type:",
        ["None (Pass)", "Scratch", "Void (Sub-surface)", "Blister (Height)", "Delamination"]
    )
else:
    selected_defect = "None (Pass)"

# Calibration interface
st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='color:#FFF;'>Digital Recalibration</h3>", unsafe_allow_html=True)
st.sidebar.caption("Adjust homography parameters for sensor alignment")
cal_rotation = st.sidebar.slider("Rotation (deg)", -2.0, 2.0, 0.3, 0.1)
cal_tx = st.sidebar.slider("Translation X (px)", -10.0, 10.0, 2.0, 0.5)
cal_ty = st.sidebar.slider("Translation Y (px)", -10.0, 10.0, -1.5, 0.5)
cal_scale = st.sidebar.slider("Scale Factor", 0.95, 1.05, 1.0, 0.005)

# Apply calibration to fusion manager
angle_rad = np.radians(cal_rotation)
cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
H_custom = np.array([
    [cos_a * cal_scale, -sin_a, cal_tx],
    [sin_a, cos_a * cal_scale, cal_ty],
    [0.0, 0.0, 1.0]
], dtype=np.float32)
fusion_mgr.update_calibration(h_thermal=H_custom, h_profiler=H_custom)

# ============================================================
# SESSION STATE
# ============================================================
if "part_counter" not in st.session_state:
    st.session_state.part_counter = part_num
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "inspection_history" not in st.session_state:
    st.session_state.inspection_history = []

# ============================================================
# TRIGGER INSPECTION
# ============================================================
trigger_btn = st.sidebar.button("⚡ Trigger Physical Scan", use_container_width=True)

if trigger_btn:
    st.session_state.part_counter += 1
    part_id = f"PART_{st.session_state.part_counter:04d}"

    # 1. Acquire image (real upload or simulated)
    h, w = 1024, 1024
    defect_label = "none"

    if uploaded_file is not None and image_source == "Upload Image":
        # Read uploaded real image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        optical = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if optical is None:
            st.sidebar.error("Failed to decode image!")
            st.stop()
        # Keep native resolution (letterboxed inside ONNX engine); cap oversized frames
        max_side = 1280
        oh, ow = optical.shape[:2]
        scale = min(1.0, max_side / max(oh, ow))
        if scale < 1.0:
            optical = cv2.resize(
                optical,
                (int(ow * scale), int(oh * scale)),
                interpolation=cv2.INTER_AREA,
            )
        h, w = optical.shape[:2]
        defect_label = "unknown"  # Real image - let model decide
        uploaded_file.seek(0)  # Reset for potential re-read
    elif st.session_state.get("demo_sample_path"):
        # Load demo sample from dataset
        demo_path = st.session_state.pop("demo_sample_path")
        st.session_state["_last_demo_path"] = demo_path
        optical = cv2.imread(demo_path)
        if optical is not None:
            max_side = 1280
            oh, ow = optical.shape[:2]
            scale = min(1.0, max_side / max(oh, ow))
            if scale < 1.0:
                optical = cv2.resize(
                    optical,
                    (int(ow * scale), int(oh * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            h, w = optical.shape[:2]
        else:
            h, w = 1024, 1024
            optical = np.random.randint(160, 195, (h, w, 3), dtype=np.uint8)
        defect_label = "unknown"
    else:
        # Generate realistic simulated surface (textured, not flat gray)
        # Base: metallic coating surface with grain texture
        optical = np.random.randint(160, 195, (h, w, 3), dtype=np.uint8)
        # Add horizontal machining lines (realistic for rolled coating)
        for y in range(0, h, 4):
            line_intensity = np.random.randint(-12, 12)
            optical[y:y+2, :] = np.clip(
                optical[y:y+2, :].astype(np.int16) + line_intensity, 0, 255
            ).astype(np.uint8)
        # Add subtle gradient (illumination variation)
        gradient = np.linspace(0.92, 1.08, w).reshape(1, -1, 1)
        optical = np.clip(optical.astype(np.float32) * gradient, 0, 255).astype(np.uint8)
        # Gaussian blur for realistic camera focus
        optical = cv2.GaussianBlur(optical, (3, 3), 0.5)

        # Apply defect overlay
        if "Scratch" in selected_defect:
            # Realistic scratch: thin dark line with slight irregularity
            pts = []
            x, y_pos = 100, np.random.randint(200, 800)
            for i in range(50):
                pts.append([x + i * 16, y_pos + np.random.randint(-3, 3)])
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(optical, [pts], False, (35, 30, 28), thickness=np.random.randint(2, 6))
            defect_label = "scratch"
        elif "Void" in selected_defect:
            # Realistic void: dark circular region with soft edges
            cx, cy = np.random.randint(300, 700), np.random.randint(300, 700)
            radius = np.random.randint(40, 90)
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(mask, (cx, cy), radius, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (21, 21), 5.0)
            for c in range(3):
                optical[:, :, c] = np.clip(
                    optical[:, :, c].astype(np.float32) - mask * 80, 0, 255
                ).astype(np.uint8)
            defect_label = "void"
        elif "Blister" in selected_defect:
            # Realistic blister: bright raised spot with rim
            cx, cy = np.random.randint(300, 700), np.random.randint(300, 700)
            radius = np.random.randint(30, 70)
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(mask, (cx, cy), radius, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (15, 15), 4.0)
            for c in range(3):
                optical[:, :, c] = np.clip(
                    optical[:, :, c].astype(np.float32) + mask * 50, 0, 255
                ).astype(np.uint8)
            # Dark rim around blister
            cv2.circle(optical, (cx, cy), radius + 5, (120, 115, 110), 2)
            defect_label = "blister"
        elif "Delamination" in selected_defect:
            # Realistic delamination: irregular dark patch with texture change
            cx, cy = np.random.randint(300, 700), np.random.randint(300, 700)
            num_pts = np.random.randint(6, 10)
            pts = []
            for k in range(num_pts):
                angle = 2 * np.pi * k / num_pts
                r = np.random.randint(60, 130)
                pts.append([int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))])
            pts = np.array(pts, dtype=np.int32)
            # Fill with different texture (peeled coating)
            overlay = optical.copy()
            cv2.fillPoly(overlay, [pts], (90, 85, 80))
            # Add noise inside delamination area
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            noise_patch = np.random.randint(-20, 20, (h, w, 3), dtype=np.int16)
            overlay = np.clip(overlay.astype(np.int16) + noise_patch * (mask[:, :, None] > 0), 0, 255).astype(np.uint8)
            # Blend
            alpha = 0.7
            optical = cv2.addWeighted(optical, 1 - alpha * (mask[:, :, None] > 0).astype(np.float32).mean(),
                                      overlay, alpha * (mask[:, :, None] > 0).astype(np.float32).mean(), 0)
            optical = np.clip(optical, 0, 255).astype(np.uint8)
            cv2.polylines(optical, [pts], True, (60, 55, 50), 2)
            defect_label = "delamination"

    # 2. Multi-source fusion
    fusion_result = fusion_mgr.fuse(
        rgb_image=optical,
        thermal_online=thermal_connected,
        profiler_online=depth_connected
    )
    thermal = fusion_result.thermal_frame
    height_map = fusion_result.height_frame

    api_payload = None
    if use_api_backend:
        try:
            import requests as _req
            form = {
                "batch_id": active_batch,
                "part_id": part_id,
                "thermal_online": str(thermal_connected).lower(),
                "profiler_online": str(depth_connected).lower(),
            }
            files = None
            demo_path = st.session_state.get("_last_demo_path")
            if image_source == "Upload Image" and uploaded_file is not None:
                uploaded_file.seek(0)
                files = {"image": (uploaded_file.name, uploaded_file.read(), uploaded_file.type or "image/jpeg")}
            elif demo_path:
                form["sample_name"] = os.path.basename(demo_path)
            elif defect_label not in ("none", "unknown"):
                form["simulate_defect"] = defect_label
            resp = _req.post(
                f"{api_base.rstrip('/')}/api/inspect",
                data=form,
                files=files,
                timeout=120,
            )
            resp.raise_for_status()
            api_payload = resp.json()
        except Exception as api_err:
            st.sidebar.error(f"API inspect failed: {api_err}")
            api_payload = None

    # 3. Run inference with fail-safe (local overlay / fallback if API down)
    failsafe.health.thermal_sensor_ok = thermal_connected
    failsafe.health.profiler_sensor_ok = depth_connected
    result = failsafe.safe_predict(predictor, optical, thermal, height_map)
    if api_payload:
        # Prefer API as source of truth for grading / PLC / engine metadata
        result = {
            **result,
            "latency_ms": api_payload.get("latency_ms", result.get("latency_ms", 0.0)),
            "engine": api_payload.get("engine", result.get("engine")),
            "fallback_active": api_payload.get("fallback_active", result.get("fallback_active")),
            "system_state": api_payload.get("system_state", result.get("system_state")),
            "model_version": api_payload.get("model_version", result.get("model_version")),
            "detections": api_payload.get("detections", result.get("detections", [])),
            "run_id": api_payload.get("run_id"),
        }

    # 4. Post-process segmentation mask (trust YOLO/ONNX — do NOT invent CV masks)
    seg_mask = result.get("segmentation_mask", np.zeros((h, w), dtype=np.uint8)).copy()

    # Legacy classic-CV fill only when no trained model is loaded AND camera-sim injected a defect
    if (
        np.max(seg_mask) == 0
        and defect_label not in ("none", "unknown")
        and not (predictor.yolo_available or predictor.onnx_available)
    ):
        # If model output is empty, generate segmentation from image analysis
        # This provides meaningful visualization even before model is trained on real data
        gray = cv2.cvtColor(optical, cv2.COLOR_BGR2GRAY)
        
        # Adaptive threshold to find defect regions
        blurred = cv2.GaussianBlur(gray, (11, 11), 2.0)
        
        # Detect dark anomalies (scratches, voids, delamination)
        dark_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 15
        )
        # Detect bright anomalies (blisters, raised areas)
        bright_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -10
        )
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dark_clean = cv2.morphologyEx(dark_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        dark_clean = cv2.morphologyEx(dark_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
        bright_clean = cv2.morphologyEx(bright_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Edge-based scratch detection
        edges = cv2.Canny(gray, 50, 150)
        edges_dilated = cv2.dilate(edges, np.ones((3, 3)), iterations=1)
        
        # Assign classes: 1=scratch (edges), 2=void (dark), 3=blister (bright), 4=delamination (large dark)
        # Find contours and classify by shape
        contours_dark, _ = cv2.findContours(dark_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_bright, _ = cv2.findContours(bright_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours_dark:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect_ratio = max(cw, ch) / (min(cw, ch) + 1e-6)
            
            if aspect_ratio > 4:  # Long thin = scratch
                cv2.drawContours(seg_mask, [cnt], -1, 1, -1)
            elif area > 5000:  # Large area = delamination
                cv2.drawContours(seg_mask, [cnt], -1, 4, -1)
            else:  # Medium dark = void
                cv2.drawContours(seg_mask, [cnt], -1, 2, -1)
        
        for cnt in contours_bright:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            cv2.drawContours(seg_mask, [cnt], -1, 3, -1)  # Bright = blister

    # Prefer API defects/grading; otherwise grade locally from the segmentation mask
    if api_payload and api_payload.get("defects_found") is not None:
        defects = api_payload["defects_found"]
    else:
        defects = extract_defects_from_mask(seg_mask, height_map, pixel_to_mm_ratio=0.1)
        model_dets = result.get("detections") or []
        if model_dets:
            by_class = {}
            for det in model_dets:
                name = det.get("class_name", "")
                by_class[name] = max(by_class.get(name, 0.0), float(det.get("confidence", 0.0)))
            for d in defects:
                d["confidence"] = round(by_class.get(d.get("class_name", ""), 0.0), 4) or None

    # 5. Grade coating
    rules = model_config.get("inference", {}).get("grading", {})
    if api_payload:
        grade = {
            "passed": api_payload.get("passed", True),
            "reject_reasons": api_payload.get("reject_reasons", []),
            "action": "PASS" if api_payload.get("passed", True) else "REJECT",
        }
        industrial_result = {"gate_action": api_payload.get("gate_action", "HOLD")}
    else:
        grade = grade_coating(defects, rules)
        industrial_result = industrial_mgr.process_inspection_result(
            part_id=part_id,
            batch_id=active_batch,
            defects=defects,
            grade_result=grade
        )

    # 7. Log to DB (skip when API already persisted the inspection)
    max_len = max([d["length_mm"] for d in defects]) if defects else 0.0
    max_area = max([d["area_mm2"] for d in defects]) if defects else 0.0
    peak_h = max([d["peak_height_um"] for d in defects]) if defects else 0.0

    if not api_payload:
        quality_mem.add_entry(
            batch_id=active_batch,
            part_id=part_id,
            has_defect=not grade["passed"],
            defect_class=defect_label if defect_label != "unknown" else (
                defects[0]["class_name"] if defects else "none"
            ),
            max_length=max_len,
            max_area=max_area,
            peak_height=peak_h,
            latency=result.get("latency_ms", 0.0),
            fallback=result.get("fallback_active", False),
            model_version=result.get("model_version", predictor.model_version),
            run_id=result.get("run_id"),
        )

    # Store result in session
    st.session_state.last_result = {
        "part_id": part_id,
        "run_id": result.get("run_id") or (api_payload or {}).get("run_id"),
        "passed": grade["passed"],
        "reject_reasons": grade["reject_reasons"],
        "defects": defects,
        "latency_ms": result.get("latency_ms", 0.0),
        "fallback_active": result.get("fallback_active", False),
        "optical_img": optical,
        "thermal_img": thermal,
        "height_img": height_map,
        "seg_mask": seg_mask,
        "gate_action": industrial_result["gate_action"],
        "engine": result.get("engine", "PyTorch"),
        "system_state": result.get("system_state", "OPTIMAL"),
    }

    # Add to inspection history for trend chart
    st.session_state.inspection_history.append({
        "part_id": part_id,
        "timestamp": datetime.datetime.now(),
        "has_defect": not grade["passed"],
        "defect_class": defect_label,
        "latency_ms": result.get("latency_ms", 0.0),
        "gate_action": industrial_result["gate_action"],
    })
    # Keep last 100 entries
    if len(st.session_state.inspection_history) > 100:
        st.session_state.inspection_history = st.session_state.inspection_history[-100:]

# ============================================================
# TOP METRICS ROW
# ============================================================
stats = quality_mem.get_batch_stats(active_batch)
spc = quality_mem.check_spc_alarms(active_batch)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-title">Inspected</div>
        <div class="metric-value">{stats["total"]}</div>
    </div>""", unsafe_allow_html=True)
with col2:
    rate = stats["pass_rate"]
    color = "status-optimal" if rate >= 95 else ("status-degraded" if rate >= 85 else "status-fail")
    st.markdown(f"""<div class="metric-card">
        <div class="metric-title">Pass Rate</div>
        <div class="metric-value {color}">{rate}%</div>
    </div>""", unsafe_allow_html=True)
with col3:
    lat = stats["avg_latency_ms"]
    st.markdown(f"""<div class="metric-card">
        <div class="metric-title">Avg Latency</div>
        <div class="metric-value" style="color:#E040FB;">{lat} ms</div>
    </div>""", unsafe_allow_html=True)
with col4:
    spc_status = spc["status"]
    spc_color = "status-optimal" if spc_status == "IN CONTROL" else (
        "status-degraded" if spc_status == "WARNING" else "status-fail")
    st.markdown(f"""<div class="metric-card">
        <div class="metric-title">SPC Status</div>
        <div class="metric-value {spc_color}">{spc_status}</div>
    </div>""", unsafe_allow_html=True)
with col5:
    sys_state = failsafe.system_state.value
    sys_color = "status-optimal" if sys_state == "OPTIMAL" else (
        "status-degraded" if sys_state == "DEGRADED" else "status-fail")
    st.markdown(f"""<div class="metric-card">
        <div class="metric-title">System</div>
        <div class="metric-value {sys_color}">{sys_state}</div>
    </div>""", unsafe_allow_html=True)

if spc["status"] != "IN CONTROL":
    st.warning(f"⚠️ **SPC Alert:** {spc['message']}")

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# MAIN TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📺 Live Inspection",
    "📈 SPC Analytics",
    "🏭 Industrial I/O",
    "🗄️ Quality Logs"
])

# ------ TAB 1: LIVE INSPECTION VIEW ------
with tab1:
    if st.session_state.last_result is None:
        st.info("No scan triggered yet. Use '⚡ Trigger Physical Scan' in the sidebar.")
    else:
        res = st.session_state.last_result

        # Verdict banner with professional styling
        if res["passed"]:
            st.markdown("""<div style="background:linear-gradient(135deg,#1B5E20,#2E7D32);
                padding:15px 25px;border-radius:10px;margin-bottom:20px;">
                <span style="font-size:24px;color:#fff;font-weight:700;">
                PASS</span>
                <span style="color:#A5D6A7;margin-left:15px;">No critical defects detected. Part cleared for next stage.</span>
            </div>""", unsafe_allow_html=True)
        else:
            n_violations = len(res["reject_reasons"])
            st.markdown(f"""<div style="background:linear-gradient(135deg,#B71C1C,#D32F2F);
                padding:15px 25px;border-radius:10px;margin-bottom:20px;">
                <span style="font-size:24px;color:#fff;font-weight:700;">
                REJECT</span>
                <span style="color:#FFCDD2;margin-left:15px;">{n_violations} quality violation(s) — Part diverted to sorting gate.</span>
            </div>""", unsafe_allow_html=True)

        # Main layout: large image left, info right
        main_c1, main_c2 = st.columns([3, 2])

        with main_c1:
            # Professional detection overlay with bboxes + labels
            seg = res["seg_mask"]
            overlay = res["optical_img"].copy()
            class_names = {1: "SCRATCH", 2: "VOID", 3: "BLISTER", 4: "DELAMINATION"}
            class_colors = {
                1: (0, 255, 255),    # Cyan
                2: (255, 100, 255),  # Magenta
                3: (100, 255, 100),  # Green
                4: (80, 80, 255),    # Red
            }

            if np.max(seg) > 0:
                # Draw semi-transparent mask overlay
                mask_overlay = np.zeros_like(overlay)
                for class_id, color in class_colors.items():
                    mask_region = (seg == class_id).astype(np.uint8)
                    if np.any(mask_region):
                        mask_overlay[mask_region == 1] = color

                overlay = cv2.addWeighted(overlay, 0.7, mask_overlay, 0.3, 0)

                # Draw bounding boxes + labels for top defects (professional YOLO style)
                top_defects = sorted(res["defects"], key=lambda d: d.get("area_mm2", 0), reverse=True)[:10]
                for defect in top_defects:
                    bbox = defect.get("bbox", None)
                    if bbox is None:
                        continue
                    x, y, w, h = bbox
                    cls_id = defect.get("class_id", 1)
                    cls_name = class_names.get(cls_id, "DEFECT")
                    color = class_colors.get(cls_id, (255, 255, 255))

                    # Draw box
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                    # Label background
                    label = f"{cls_name} {defect.get('area_mm2', 0):.1f}mm2"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(overlay, (x, y - th - 8), (x + tw + 4, y), color, -1)
                    cv2.putText(overlay, label, (x + 2, y - 4),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            st.image(
                cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                caption=f"AI Detection Result — {res['part_id']}",
                use_container_width=True
            )

        with main_c2:
            # Inspection summary card
            st.markdown("##### Inspection Report")
            st.markdown(f"""
            | Parameter | Value |
            |-----------|-------|
            | Part ID | `{res['part_id']}` |
            | Run ID | `{res.get('run_id') or '—'}` |
            | Verdict | **{'PASS' if res['passed'] else 'REJECT'}** |
            | Defects Found | {len(res['defects'])} |
            | Inference Engine | {res['engine']} |
            | Latency | {res['latency_ms']:.1f} ms |
            | Sensor Mode | {'Optimal (3-source)' if not res['fallback_active'] else 'Degraded'} |
            | Gate Signal | {res['gate_action']} |
            """)

            if not res["passed"] and res["reject_reasons"]:
                st.markdown("##### Rejection Reasons")
                for i, reason in enumerate(res["reject_reasons"][:5], 1):
                    st.markdown(f"- {reason}")
                if len(res["reject_reasons"]) > 5:
                    st.caption(f"... +{len(res['reject_reasons'])-5} more")

            # Class legend
            st.markdown("##### Defect Class Legend")
            st.markdown("""
            <span style="color:#00FFFF;">&#9632;</span> SCRATCH &nbsp;&nbsp;
            <span style="color:#FF64FF;">&#9632;</span> VOID &nbsp;&nbsp;
            <span style="color:#64FF64;">&#9632;</span> BLISTER &nbsp;&nbsp;
            <span style="color:#5050FF;">&#9632;</span> DELAMINATION
            """, unsafe_allow_html=True)

        # Sensor modalities row (smaller, below main view)
        st.markdown("---")
        st.markdown("##### Sensor Modalities")
        s1, s2, s3 = st.columns(3)
        with s1:
            st.image(
                cv2.cvtColor(res["optical_img"], cv2.COLOR_BGR2RGB),
                caption="RGB Camera",
                use_container_width=True
            )
        with s2:
            if res["thermal_img"] is not None:
                therm_colored = fusion_mgr.get_colorized_thermal(res["thermal_img"])
                st.image(cv2.cvtColor(therm_colored, cv2.COLOR_BGR2RGB),
                         caption="LWIR Thermal", use_container_width=True)
            else:
                st.error("LWIR: OFFLINE")
        with s3:
            if res["height_img"] is not None:
                height_colored = fusion_mgr.get_colorized_height(res["height_img"])
                st.image(cv2.cvtColor(height_colored, cv2.COLOR_BGR2RGB),
                         caption="3D Profilometer", use_container_width=True)
            else:
                st.error("3D: OFFLINE")

        # Defect details (collapsible)
        with st.expander(f"Defect Details ({len(res['defects'])} regions)", expanded=False):
            if len(res["defects"]) == 0:
                st.success("No anomalies detected.")
            else:
                total_defects = len(res["defects"])
                df_defects = pd.DataFrame(res["defects"])
                display_cols = [
                    "defect_id", "class_name", "confidence",
                    "length_mm", "area_mm2", "peak_height_um",
                ]
                available_cols = [c for c in display_cols if c in df_defects.columns]
                if total_defects > 15:
                    df_defects = df_defects.sort_values("area_mm2", ascending=False).head(15)
                st.dataframe(df_defects[available_cols], use_container_width=True)

# ------ TAB 2: SPC ANALYTICS ------
with tab2:
    st.markdown("### Statistical Process Control & Batch Trend Analysis")

    history = st.session_state.inspection_history
    if len(history) < 2:
        st.info("Trigger multiple scans to populate SPC analytics.")
    else:
        df_hist = pd.DataFrame(history)

        # Defect rate over time (rolling window)
        df_hist["defect_int"] = df_hist["has_defect"].astype(int)
        window = min(10, len(df_hist))
        df_hist["defect_rate_pct"] = (
            df_hist["defect_int"].rolling(window=window, min_periods=1).mean() * 100
        )
        df_hist["latency_rolling"] = (
            df_hist["latency_ms"].rolling(window=window, min_periods=1).mean()
        )

        # Layout: two charts side by side
        chart_c1, chart_c2 = st.columns(2)

        with chart_c1:
            st.markdown("#### Defect Rate Trend (SPC)")
            chart_data = pd.DataFrame({
                "Part": range(len(df_hist)),
                "Defect Rate (%)": df_hist["defect_rate_pct"].values,
                "UCL (10%)": [10.0] * len(df_hist),
                "Warning (5%)": [5.0] * len(df_hist),
            }).set_index("Part")
            st.line_chart(chart_data, height=300)

        with chart_c2:
            st.markdown("#### Inference Latency Trend")
            lat_data = pd.DataFrame({
                "Part": range(len(df_hist)),
                "Latency (ms)": df_hist["latency_rolling"].values,
                "Target (35ms)": [35.0] * len(df_hist),
            }).set_index("Part")
            st.line_chart(lat_data, height=300)

        # Defect class distribution
        st.markdown("#### Defect Class Distribution")
        defect_only = df_hist[df_hist["has_defect"] == True]
        if len(defect_only) > 0:
            dist = defect_only["defect_class"].value_counts()
            st.bar_chart(dist)
        else:
            st.success("No defects recorded in current session.")

        # Summary metrics
        st.markdown("#### Session Summary")
        sum_c1, sum_c2, sum_c3, sum_c4 = st.columns(4)
        with sum_c1:
            st.metric("Total Inspected", len(df_hist))
        with sum_c2:
            defect_count = df_hist["defect_int"].sum()
            st.metric("Total Defects", int(defect_count))
        with sum_c3:
            avg_lat = df_hist["latency_ms"].mean()
            st.metric("Avg Latency", f"{avg_lat:.1f} ms")
        with sum_c4:
            reject_count = len(df_hist[df_hist["gate_action"] == "REJECT"])
            st.metric("Parts Rejected", reject_count)

# ------ TAB 3: INDUSTRIAL I/O ------
with tab3:
    st.markdown("### Industrial Communication Monitor")
    st.caption("Simulated OPC UA & Modbus TCP interface to PLC sorting gate")

    # PLC State Display
    plc_state = industrial_mgr.get_plc_state()

    io_c1, io_c2 = st.columns(2)

    with io_c1:
        st.markdown("#### Modbus TCP Registers")
        modbus_regs = plc_state.get("modbus_registers", {})
        reg_df = pd.DataFrame([
            {"Register": k, "Value": v} for k, v in modbus_regs.items()
        ])
        st.dataframe(reg_df, use_container_width=True, hide_index=True)

    with io_c2:
        st.markdown("#### OPC UA Nodes")
        opc_nodes = plc_state.get("opc_ua_nodes", {})
        node_df = pd.DataFrame([
            {"Node ID": k, "Value": str(v)} for k, v in opc_nodes.items()
        ])
        st.dataframe(node_df, use_container_width=True, hide_index=True)

    # Connection info
    conn = plc_state.get("connection", {})
    st.markdown(f"""
    **Connection Details:**
    - PLC IP: `{conn.get('plc_ip', 'N/A')}`
    - OPC UA Endpoint: `{conn.get('opc_endpoint', 'N/A')}`
    - Modbus Port: `{conn.get('modbus_port', 'N/A')}`
    - Status: `{conn.get('status', 'Unknown')}`
    """)

    # Signal History
    st.markdown("#### Recent Signals")
    signals = industrial_mgr.get_signal_history(limit=20)
    if signals:
        sig_df = pd.DataFrame(signals)
        st.dataframe(sig_df, use_container_width=True, hide_index=True)
    else:
        st.info("No signals sent yet.")

    # Emergency controls
    st.markdown("---")
    e_c1, e_c2 = st.columns(2)
    with e_c1:
        if st.button("🚨 EMERGENCY STOP", type="primary", use_container_width=True):
            industrial_mgr.emergency_stop("Dashboard manual trigger")
            st.error("E-STOP ACTIVATED! Line halted.")
    with e_c2:
        if st.button("🔄 Reset Line", use_container_width=True):
            industrial_mgr.reset_line()
            st.success("Production line reset and resumed.")

# ------ TAB 4: QUALITY LOGS ------
with tab4:
    st.markdown("### Quality Traceability Database")
    st.caption(f"SQLite database: `{DB_PATH}`")

    # Query batch data
    batch_stats = quality_mem.get_batch_stats(active_batch)

    if batch_stats["total"] == 0:
        st.info(f"No records found for batch '{active_batch}'. Run some inspections first.")
    else:
        # Batch overview
        st.markdown(f"#### Batch: `{active_batch}`")
        log_c1, log_c2, log_c3, log_c4 = st.columns(4)
        with log_c1:
            st.metric("Total Parts", batch_stats["total"])
        with log_c2:
            st.metric("Passed", batch_stats["passed"])
        with log_c3:
            st.metric("Failed", batch_stats["failed"])
        with log_c4:
            st.metric("Pass Rate", f"{batch_stats['pass_rate']}%")

        # Defect distribution
        if batch_stats.get("defect_distribution"):
            st.markdown("#### Defect Distribution")
            dist_df = pd.DataFrame([
                {"Defect Class": k, "Count": v}
                for k, v in batch_stats["defect_distribution"].items()
            ])
            st.bar_chart(dist_df.set_index("Defect Class"))

    # Raw DB query viewer
    st.markdown("---")
    st.markdown("#### Raw Inspection Records")
    try:
        import sqlite3
        if os.path.exists(DB_PATH):
            conn = sqlite3.connect(DB_PATH)
            df_raw = pd.read_sql_query(
                f"SELECT * FROM inspections WHERE batch_id = ? ORDER BY timestamp DESC LIMIT 50",
                conn,
                params=(active_batch,)
            )
            conn.close()
            if len(df_raw) > 0:
                st.dataframe(df_raw, use_container_width=True, hide_index=True)
            else:
                st.info("No records to display.")
        else:
            st.warning("Database file not yet created. Run an inspection to initialize.")
    except Exception as e:
        st.error(f"Error reading database: {e}")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#555; font-size:12px;'>"
    "SecureCoating-Vision v2.0 | 2026 AI + Materials Competition | "
    "Multi-Sensor Fusion Inspection System</p>",
    unsafe_allow_html=True
)
