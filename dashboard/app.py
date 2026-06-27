import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import requests
import yaml
import sqlite3
import datetime

# Page styling configurations
st.set_page_config(
    page_title="SecureCoating-Vision Console",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling rules (CSS injection)
st.markdown("""
<style>
    .reportview-container {
        background: #0F1219;
    }
    .metric-card {
        background: #181D28;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #2B3548;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        text-align: center;
    }
    .metric-title {
        font-size: 14px;
        color: #8C9BAE;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 32px;
        color: #00F2FE;
        font-weight: 700;
    }
    .status-optimal {
        color: #00E676;
        font-weight: 700;
    }
    .status-degraded {
        color: #FFD600;
        font-weight: 700;
    }
    .status-fail {
        color: #FF1744;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Path resolutions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.predictor import CoatingPredictor
from inference.postprocess import extract_defects_from_mask, grade_coating
from traceability.quality_memory import QualityMemory

# Load configuration parameters
with open("configs/model.yaml", "r") as f:
    model_config = yaml.safe_load(f)
with open("configs/app.yaml", "r") as f:
    app_config = yaml.safe_load(f)

# DB Access Helper
DB_PATH = app_config.get("paths", {}).get("db_path", "data/quality_history.db")
quality_mem = QualityMemory(DB_PATH)

# Initialize Predictor
@st.cache_resource
def get_predictor():
    return CoatingPredictor(model_config)

predictor = get_predictor()

# App Header
st.markdown("<h1 style='text-align: center; color: #FFFFFF; font-weight: 800;'>🔍 SecureCoating-Vision</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8C9BAE; font-size: 16px; margin-bottom: 30px;'>Multi-Sensor Fusion Inspection Console & Traceability Analytics</p>", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("<h2 style='color: #FFFFFF;'>System Control Panel</h2>", unsafe_allow_html=True)
active_batch = st.sidebar.text_input("Active Batch ID", value="BATCH_2026_06A")
part_num = st.sidebar.number_input("Starting Part Number", min_value=1, value=10, step=1)

# Sensor statuses override simulator
st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='color: #FFFFFF;'>Hardware Health Override</h3>", unsafe_allow_html=True)
thermal_connected = st.sidebar.toggle("LWIR Thermal Camera Online", value=True)
depth_connected = st.sidebar.toggle("3D Laser Profiler Online", value=True)

# Defect simulator trigger
st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='color: #FFFFFF;'>Simulate Defects</h3>", unsafe_allow_html=True)
selected_defect = st.sidebar.selectbox(
    "Select Defect type to introduce:",
    ["None (Pass)", "Scratch", "Void (Sub-surface)", "Blister (Height deviation)", "Delamination (Fused Area)"]
)

# ----------------- INFERENCE CONTROLLER -----------------
# State parameters
if "part_counter" not in st.session_state:
    st.session_state.part_counter = part_num
if "last_result" not in st.session_state:
    st.session_state.last_result = None

trigger_btn = st.sidebar.button("⚡ Trigger Physical Scan", use_container_width=True)

if trigger_btn:
    st.session_state.part_counter += 1
    part_id = f"PART_{st.session_state.part_counter:04d}"
    
    # 1. Setup mock canvas
    h, w = 1024, 1024
    optical = np.ones((h, w, 3), dtype=np.uint8) * 190
    
    # Optional base textures to look real
    cv2.putText(optical, "COATED SURFACE BASE", (350, 512), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (170,170,170), 2)
    
    # Sensor status checking
    thermal = np.ones((h, w), dtype=np.float32) * 42.5 if thermal_connected else None
    height = np.zeros((h, w), dtype=np.float32) if depth_connected else None
    
    defect_label = "none"
    # Overlays based on select
    if "Scratch" in selected_defect:
        cv2.line(optical, (150, 200), (700, 240), (40, 40, 40), thickness=8)
        defect_label = "scratch"
    elif "Void" in selected_defect and thermal is not None:
        cv2.circle(thermal, (600, 500), 90, 31.0, -1)
        defect_label = "void"
    elif "Blister" in selected_defect and height is not None:
        cv2.circle(height, (400, 650), 70, 220.0, -1)
        defect_label = "blister"
    elif "Delamination" in selected_defect:
        if thermal is not None:
            cv2.circle(thermal, (512, 512), 150, 22.0, -1)
        if height is not None:
            cv2.circle(height, (512, 512), 150, 35.0, -1)
        defect_label = "delamination"

    # 2. Run inference
    result = predictor.predict(optical, thermal, height)
    
    # Make sure mock defect class displays if simulator override is enabled
    if defect_label != "none":
        # Draw fake segmentation circle inside mask
        mask = result["segmentation_mask"]
        class_id = {"scratch": 1, "void": 2, "blister": 3, "delamination": 4}[defect_label]
        cv2.circle(mask, (512, 512), 100, class_id, -1)
        
    defects = extract_defects_from_mask(result["segmentation_mask"], height, pixel_to_mm_ratio=0.1)
    
    # Evaluate grading rules
    rules = model_config.get("inference", {}).get("grading", {})
    grade = grade_coating(defects, rules)
    
    # 3. Log to DB
    max_len = max([d["length_mm"] for d in defects]) if defects else 0.0
    max_area = max([d["area_mm2"] for d in defects]) if defects else 0.0
    peak_h = max([d["peak_height_um"] for d in defects]) if defects else 0.0
    
    quality_mem.add_entry(
        batch_id=active_batch,
        part_id=part_id,
        has_defect=not grade["passed"],
        defect_class=defect_label,
        max_length=max_len,
        max_area=max_area,
        peak_height=peak_h,
        latency=result["latency_ms"],
        fallback=result["fallback_active"]
    )
    
    st.session_state.last_result = {
        "part_id": part_id,
        "passed": grade["passed"],
        "reject_reasons": grade["reject_reasons"],
        "defects": defects,
        "latency_ms": result["latency_ms"],
        "fallback_active": result["fallback_active"],
        "optical_img": optical,
        "thermal_img": thermal,
        "height_img": height,
        "seg_mask": result["segmentation_mask"]
    }

# ----------------- METRICS DASHBOARD SECTION -----------------
# Retrieve current batch stats
stats = quality_mem.get_batch_stats(active_batch)
spc = quality_mem.check_spc_alarms(active_batch)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Inspected Count</div>
        <div class="metric-value">{stats["total"]}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    color_class = "status-optimal" if stats["pass_rate"] >= 95.0 else "status-degraded"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Batch Pass Rate</div>
        <div class="metric-value {color_class}">{stats["pass_rate"]}%</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Avg Latency</div>
        <div class="metric-value" style="color:#E040FB;">{stats["avg_latency_ms"]} ms</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    spc_status = spc["status"]
    spc_color = "status-optimal" if spc_status == "IN CONTROL" else ("status-degraded" if spc_status == "WARNING" else "status-fail")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">SPC Status</div>
        <div class="metric-value {spc_color}">{spc_status}</div>
    </div>
    """, unsafe_allow_html=True)

# Warn operators on SPC flags
if spc["status"] != "IN CONTROL":
    st.warning(f"⚠️ **SPC Notification:** {spc['message']}")

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- CORE TABS -----------------
tab1, tab2, tab3 = st.tabs(["📺 Live Inspection View", "📈 Batch Trend & SPC Analytics", "🗄️ Quality Logs DB"])

with tab1:
    if st.session_state.last_result is None:
        st.info("No physical scan triggered yet. Use 'Trigger Physical Scan' in the sidebar controls to begin inspection.")
    else:
        res = st.session_state.last_result
        
        # Upper banner details
        banner_col1, banner_col2, banner_col3 = st.columns(3)
        with banner_col1:
            st.metric("Inspection Target ID", res["part_id"])
        with banner_col2:
            status_text = "PASS" if res["passed"] else "REJECT"
            st.metric("Evaluation Verdict", status_text, delta=None, delta_color="normal")
        with banner_col3:
            degrade_text = "Fallback (Degraded Mode)" if res["fallback_active"] else "Optimal (Multi-Source)"
            st.metric("Sensors Execution", degrade_text)
            
        if not res["passed"]:
            st.error(f"❌ **Rejection Reasons:** {', '.join(res['reject_reasons'])}")
            
        # Modal display mapping
        st.markdown("<h4 style='color: #FFFFFF;'>Source Modalities & Defect Mapping</h4>", unsafe_allow_html=True)
        img_col1, img_col2, img_col3, img_col4 = st.columns(4)
        
        with img_col1:
            st.image(res["optical_img"], caption="Optical Camera (RGB)", use_column_width=True)
        with img_col2:
            if res["thermal_img"] is not None:
                # Map temperature to colored feed
                therm_norm = cv2.normalize(res["thermal_img"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                therm_colored = cv2.applyColorMap(therm_norm, cv2.COLORMAP_JET)
                st.image(therm_colored, caption="LWIR Thermal Feed (Heat Map)", use_column_width=True)
            else:
                st.error("LWIR Thermal: NO SIGNAL")
        with img_col3:
            if res["height_img"] is not None:
                height_norm = cv2.normalize(res["height_img"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                height_colored = cv2.applyColorMap(height_norm, cv2.COLORMAP_VIRIDIS)
                st.image(height_colored, caption="3D Height profile", use_column_width=True)
            else:
                st.error("3D Profiler: NO SIGNAL")
        with img_col4:
            # Color map overlay of classes on background
            mask_colored = cv2.applyColorMap(res["seg_mask"] * 50, cv2.COLORMAP_BONE)
            st.image(mask_colored, caption="Inference Segmentation Mask Output", use_column_width=True)

        # Defect contour findings list
        st.markdown("<h4 style='color: #FFFFFF;'>Feature Extraction Summary</h4>", unsafe_allow_html=True)
        if len(res["defects"]) == 0:
            st.success("No anomalies detected in active inspection windows.")
        else:
            df_defects = pd.DataFrame(res["defects"])
            st.dataframe(df_defects[["defect_id", "class_name", "length_mm", "width_mm", "area_mm2", "peak_height_um"]], use_container_width=True)

with tab2:
    st.markdown("<h4 style='color: #FFFFFF;'>Batch Diagnostics & Parameter Trends</h4>", unsafe_allow_html=True)
    
    # Load all records for chart plotting
    conn = sqlite3.connect(DB_PATH)
    df_history = pd.read_sql_query(f"SELECT * FROM inspections WHERE batch_id = '{active_batch}' ORDER BY id ASC", conn)
    conn.close()
    
    if df_history.empty:
        st.info("No records found for current batch trends.")
    else:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("##### Defect Class Prevalence")
            defect_counts = df_history[df_history["has_defect"] == 1]["defect_class"].value_counts()
            if not defect_counts.empty:
                st.bar_chart(defect_counts)
            else:
                st.success("Zero defects recorded in this batch run.")
                
        with col_c2:
            st.markdown("##### Latency Execution Speed (ms)")
            st.line_chart(df_history["latency_ms"])

with tab3:
    st.markdown("<h4 style='color: #FFFFFF;'>Traceability Database Records</h4>", unsafe_allow_html=True)
    conn = sqlite3.connect(DB_PATH)
    df_all = pd.read_sql_query("SELECT * FROM inspections ORDER BY timestamp DESC", conn)
    conn.close()
    
    if df_all.empty:
        st.info("Quality memory database table is currently empty.")
    else:
        st.dataframe(df_all, use_container_width=True)
