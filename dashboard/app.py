"""
SecureCoating-Vision: SCADA Industrial Quality Inspection Terminal
===================================================================
Read-only research dashboard for local simulation and inspection visualization.

Features:
1. 7-Stage Industrial Pipeline Visualizer with Sub-Millisecond Timing Breakdown
2. Synchronized 4-Way Multi-Modal Sensor Split (Brightfield, Darkfield, Thermography Phase, 3D Laser)
3. Interactive 3D Defect Topography Surface Mesh (Plotly 3D)
4. 1,200m Jumbo Roll Digital Twin Defect Map (Waterfall View)
5. Physics-Informed Battery Electrode Metrology (GB/T 38823-2020, Micro-Short Hazard Index)
6. AI Closed-Loop Equipment Parameter Tuning Feedback (Slot-Die, Drying Oven, Mixer)
7. Industrial Telemetry (OPC UA Node Tree, Modbus TCP Hex Table, Latency Jitter)
8. Cryptographic Tamper-Evident Digital Roll Quality Certificate (SHA-256)
"""

import os
import sys
import time
import datetime
import yaml
import numpy as np
import cv2
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from api_client import InspectionApiClient

# Page configuration
st.set_page_config(
    page_title="SecureCoating-Vision | SCADA Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom SCADA Dark Theme CSS
st.markdown("""
<style>
    .stApp { background-color: #0A0D14; }
    .scada-panel {
        background: linear-gradient(180deg, #131824 0%, #0E121B 100%);
        border: 1px solid #222C3E;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        margin-bottom: 15px;
    }
    .scada-metric-card {
        background: #151B28;
        border-left: 4px solid #00F2FE;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .metric-label {
        font-size: 11px;
        color: #7E8DA4;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .metric-val {
        font-size: 22px;
        font-weight: 800;
        color: #FFFFFF;
        font-family: 'SF Pro Display', -apple-system, sans-serif;
    }
    .status-optimal { color: #00E676; }
    .status-warning { color: #FFD600; }
    .status-danger  { color: #FF1744; }
    .stage-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 700;
        margin-right: 4px;
    }
    .stage-done { background: #1B5E20; color: #00E676; }
    .stage-active { background: #01579B; color: #40C4FF; }
</style>
""", unsafe_allow_html=True)

# Path setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
os.chdir(PROJECT_ROOT)

from inference.predictor import CoatingPredictor
from inference.sensor_fusion import SensorFusionManager
from inference.failsafe import FailSafeManager
from inference.electrode_metrology import ElectrodeMetrologyEngine
from inference.multi_stage_pipeline import MultiStageIndustrialPipeline
from industrial.protocol_manager import IndustrialProtocolManager
from industrial.web_synchronizer import WebSynchronizer, RollMetadata
from traceability.quality_memory import QualityMemory
from traceability.root_cause_engine import RootCauseDiagnosticEngine
from traceability.roll_certificate import RollCertificateGenerator

# Load Configuration
with open("configs/model.yaml", "r") as f:
    model_config = yaml.safe_load(f)
with open("configs/app.yaml", "r") as f:
    app_config = yaml.safe_load(f)

# Singleton Resource Initializers
@st.cache_resource
def get_predictor():
    return CoatingPredictor(model_config)

@st.cache_resource
def get_fusion_mgr():
    return SensorFusionManager(target_size=(1024, 1024), enable_mock=True)

@st.cache_resource
def get_failsafe_mgr():
    return FailSafeManager()

@st.cache_resource
def get_industrial_mgr():
    # Dashboard is a read-only visualization sandbox. PLC ownership belongs to
    # the API process; never create a second live PLC command channel here.
    return IndustrialProtocolManager({"enabled": False, "mock_mode": True})

@st.cache_resource
def get_web_sync():
    return WebSynchronizer()

@st.cache_resource
def get_metrology_engine():
    return ElectrodeMetrologyEngine(pixel_to_mm_ratio=0.1)

@st.cache_resource
def get_root_cause_engine():
    return RootCauseDiagnosticEngine()

predictor = get_predictor()
fusion_mgr = get_fusion_mgr()
failsafe = get_failsafe_mgr()
industrial_mgr = get_industrial_mgr()
web_sync = get_web_sync()
metrology_engine = get_metrology_engine()
root_cause_engine = get_root_cause_engine()

pipeline = MultiStageIndustrialPipeline(
    predictor=predictor,
    fusion_manager=fusion_mgr,
    failsafe_manager=failsafe,
    industrial_manager=industrial_mgr,
    web_synchronizer=web_sync,
    pixel_to_mm_ratio=0.1
)

db_path = app_config.get("paths", {}).get("db_path", "data/quality_history.db")
quality_mem = QualityMemory(db_path)

api_client = InspectionApiClient(
    base_url=os.environ.get("SECURECOATING_API_URL", "http://127.0.0.1:8000"),
    api_key=os.environ.get("SECURECOATING_API_KEY", ""),
)
api_online = True
api_error = ""
try:
    api_health = api_client.health()
    api_roll_id = os.environ.get("SECURECOATING_ACTIVE_ROLL_ID", "ROLL_2026_CATL_001")
    api_roll = api_client.roll_map(api_roll_id)
    api_summary = api_roll["summary"]
    api_batch_id = api_summary["batch_id"]
    api_stats = api_client.batch_stats(api_batch_id)
    api_spc = api_client.batch_spc(api_batch_id)
    api_plc_state = api_client.industrial_state()
    api_passport = api_client.passport()
except (OSError, ValueError, KeyError, requests.RequestException) as exc:
    api_online = False
    api_error = str(exc)
    api_health = {}
    api_roll = {}
    api_summary = web_sync.get_roll_defect_summary()
    api_batch_id = web_sync.roll.batch_id
    api_stats = quality_mem.get_batch_stats(api_batch_id)
    api_spc = quality_mem.check_spc_alarms(api_batch_id)
    api_plc_state = industrial_mgr.get_plc_state()
    api_passport = pipeline.get_gigafactory_spc_summary()

# Initialize Session State
if "inspection_history" not in st.session_state:
    st.session_state.inspection_history = []
if "last_pipeline_result" not in st.session_state:
    st.session_state.last_pipeline_result = None

# =========================================================================
# HEADER & SCADA TELEMETRY BAR
# =========================================================================
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #222C3E; padding-bottom:12px; margin-bottom:15px;">
    <div>
        <h2 style="color:#FFFFFF; margin:0; font-weight:800; font-size:24px;">⚡ SecureCoating-Vision | SCADA Industrial Inspection Terminal</h2>
        <span style="color:#8C9BAE; font-size:12px;">Multi-Modal Inline Battery Electrode Quality Metrology &middot; Plant QA Spec & GB 38031-2025 Baseline</span>
    </div>
    <div>
        <span class="stage-badge stage-warn">● LOCAL SIMULATION</span>
        <span class="stage-badge" style="background:#263238; color:#90A4AE;">PLC CONTROL DISABLED</span>
        <span class="stage-badge" style="background:#1A237E; color:#8C9EFF;">YOLOv8-seg FP16</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Top 6 Telemetry Metrics
roll_summary = api_summary
stats = api_stats
spc = api_spc
if not api_online:
    st.warning(f"Authoritative API unavailable; showing isolated local sandbox data: {api_error}")

m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    st.markdown(f"""<div class="scada-metric-card">
        <div class="metric-label">Line Velocity</div>
        <div class="metric-val">{web_sync.line_speed_m_s:.2f} <span style="font-size:14px; color:#8C9BAE;">m/s</span></div>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""<div class="scada-metric-card" style="border-left-color:#00E676;">
        <div class="metric-label">Roll Progress</div>
        <div class="metric-val">{roll_summary['inspected_length_m']:.1f} / {roll_summary['total_roll_length_m']:.0f}m</div>
    </div>""", unsafe_allow_html=True)
with m3:
    pass_rate = stats["pass_rate"] if stats["pass_rate"] is not None else 0.0
    color_cls = "status-optimal" if pass_rate >= 95 else ("status-warning" if pass_rate >= 85 else "status-danger")
    st.markdown(f"""<div class="scada-metric-card" style="border-left-color:#FFD600;">
        <div class="metric-label">Yield Pass Rate</div>
        <div class="metric-val {color_cls}">{pass_rate}%</div>
    </div>""", unsafe_allow_html=True)
with m4:
    lat = stats["avg_latency_ms"] if stats["avg_latency_ms"] is not None else 0.0
    st.markdown(f"""<div class="scada-metric-card" style="border-left-color:#E040FB;">
        <div class="metric-label">Pipeline Latency</div>
        <div class="metric-val" style="color:#E040FB;">{lat} <span style="font-size:14px; color:#8C9BAE;">ms</span></div>
    </div>""", unsafe_allow_html=True)
with m5:
    spc_st = spc["status"]
    spc_color = "status-optimal" if spc_st == "IN CONTROL" else ("status-warning" if spc_st == "WARNING" else "status-danger")
    st.markdown(f"""<div class="scada-metric-card" style="border-left-color:#FF1744;">
        <div class="metric-label">SPC Quality State</div>
        <div class="metric-val {spc_color}">{spc_st}</div>
    </div>""", unsafe_allow_html=True)
with m6:
    sys_st = failsafe.system_state.value
    sys_col = "status-optimal" if sys_st == "OPTIMAL" else "status-warning"
    st.markdown(f"""<div class="scada-metric-card" style="border-left-color:#00E5FF;">
        <div class="metric-label">Fail-Safe Health</div>
        <div class="metric-val {sys_col}">{sys_st}</div>
    </div>""", unsafe_allow_html=True)

# =========================================================================
# SIDEBAR: ROLL RECIPE & SCAN TRIGGER
# =========================================================================
st.sidebar.markdown("<h3 style='color:#FFF;'>🏭 Production Recipe</h3>", unsafe_allow_html=True)
active_roll = st.sidebar.text_input("Active Roll ID", value="ROLL_2026_CATL_001")
electrode_type = st.sidebar.selectbox(
    "Electrode Chemistry",
    ["Cathode_LFP (LiFePO4)", "Cathode_NMC811", "Anode_Graphite_Silicon"],
    index=0
)
line_speed_slider = st.sidebar.slider("Line Velocity (m/s)", 0.5, 3.0, 1.8, 0.1)
web_sync.set_line_speed(line_speed_slider)

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='color:#FFF;'>⚡ Trigger Inspection</h3>", unsafe_allow_html=True)

defect_injection = st.sidebar.selectbox(
    "Sensor Simulation Target",
    ["Random Physical Anomaly", "Scratch (Doctor Blade Nick)", "Void (Slurry Degassing)", "Blister (Solvent Boil)", "Delamination (Skinning)", "Nominal (Clean Surface)"]
)

cross_pos = st.sidebar.slider("Cross-Web Position (mm)", 0.0, 650.0, 325.0, 10.0)

if st.sidebar.button("🚀 TRIGGER 7-STAGE INSPECTION", type="primary", use_container_width=True):
    # Map defect injection
    sim_map = {
        "Scratch (Doctor Blade Nick)": "scratch",
        "Void (Slurry Degassing)": "void",
        "Blister (Solvent Boil)": "blister",
        "Delamination (Skinning)": "delamination",
        "Nominal (Clean Surface)": "none",
        "Random Physical Anomaly": "random"
    }
    target_defect = sim_map.get(defect_injection, "random")
    opt_m, th_m, h_m = fusion_mgr.generate_mock_frame(defect_type=target_defect)
    
    part_id = f"PART_{int(time.time() * 1000)}"
    result = pipeline.execute_inspection(
        sample_id=part_id,
        optical_rgb=opt_m,
        thermal_raw=th_m,
        height_map=h_m,
        cross_web_pos_mm=cross_pos,
        enable_plc_signal=False,
    )
    st.session_state.last_pipeline_result = result
    
    st.rerun()

# =========================================================================
# MAIN SCADA TABS
# =========================================================================
tabs = st.tabs([
    "📺 Live Multi-Stage Station",
    "🏔️ 3D Defect Topography",
    "📜 1,200m Roll Digital Twin",
    "🔬 Battery Metrology & GB/T 38031",
    "⚙️ AI Closed-Loop Diagnostics",
    "🏭 Hardware & Protocol Telemetry",
    "📊 Gigafactory SPC & Slitting Yield",
    "🗄️ Traceability & Quality Certificate"
])

res = st.session_state.last_pipeline_result

# -------------------------------------------------------------------------
# TAB 1: LIVE MULTI-STAGE INSPECTION STATION
# -------------------------------------------------------------------------
with tabs[0]:
    if res is None:
        st.info("👈 Click **'TRIGGER 7-STAGE INSPECTION'** in the sidebar to run the multi-modal inline inspection workflow.")
    else:
        # Pipeline 7-Stage Progress Bar
        st.markdown(f"""
        <div class="scada-panel" style="padding:10px 16px;">
            <div style="font-size:12px; color:#8C9BAE; font-weight:700; margin-bottom:6px;">7-STAGE INDUSTRIAL EXECUTION TIMELINE (Total: {res.total_pipeline_latency_ms:.2f} ms)</div>
            <div style="display:flex; justify-content:space-between; font-size:11px;">
                <span class="stage-badge stage-done">1. Web Sync ({res.stage_latencies_ms.get('stage_1_web_sync_ms',0):.1f}ms)</span>
                <span class="stage-badge stage-done">2. Sensor Acq ({res.stage_latencies_ms.get('stage_2_acquisition_ms',0):.1f}ms)</span>
                <span class="stage-badge stage-done">3. Homography ({res.stage_latencies_ms.get('stage_3_fusion_ms',0):.1f}ms)</span>
                <span class="stage-badge stage-done">4. TensorRT AI ({res.stage_latencies_ms.get('stage_4_ai_inference_ms',0):.1f}ms)</span>
                <span class="stage-badge stage-done">5. Metrology ({res.stage_latencies_ms.get('stage_5_metrology_ms',0):.1f}ms)</span>
                <span class="stage-badge stage-done">6. PLC Gate ({res.stage_latencies_ms.get('stage_6_decision_plc_ms',0):.1f}ms)</span>
                <span class="stage-badge stage-done">7. Diagnostics ({res.stage_latencies_ms.get('stage_7_root_cause_ms',0):.1f}ms)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Verdict Header
        if res.overall_verdict == "PASS":
            st.success(f"✅ **PASS** | Quality Grade: **{res.quality_tier}** | Plant QA Spec & GB 38031-2025 Baseline | Roll Location: **X = {res.roll_coordinate['linear_pos_m']}m**, Lane: **{res.roll_coordinate['lane_id']}**")
        else:
            st.error(f"🚨 **REJECT** | Quality Grade: **{res.quality_tier}** | PLC Gate Action: **{res.plc_gate_action}** | Violations: {', '.join(res.rejection_reasons)}")

        # 4-Way Camera Synchronized View
        st.markdown("<h4 style='color:#FFF; margin-top:10px;'>Multi-Modal Synchronized Sensor Ingestion</h4>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.caption("📷 Optical Brightfield (RGB)")
            # Overlay contours on optical
            disp_opt = cv2.cvtColor(res.optical_brightfield, cv2.COLOR_BGR2RGB)
            st.image(disp_opt, use_container_width=True)
            
        with c2:
            st.caption("🔬 Optical Darkfield Scatter")
            disp_df = cv2.cvtColor(res.optical_darkfield, cv2.COLOR_BGR2RGB)
            st.image(disp_df, use_container_width=True)
            
        with c3:
            st.caption("🌡️ Lock-in Thermography Phase (Δφ)")
            disp_th = cv2.applyColorMap(res.thermal_diffusivity_phase, cv2.COLORMAP_INFERNO)
            st.image(cv2.cvtColor(disp_th, cv2.COLOR_BGR2RGB), use_container_width=True)
            
        with c4:
            st.caption("📐 Confocal 3D Laser Profile (Z-Height)")
            norm_h = cv2.normalize(res.height_topography_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            disp_h = cv2.applyColorMap(norm_h, cv2.COLORMAP_JET)
            st.image(cv2.cvtColor(disp_h, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Defect Segmentation Overlay & Physical Table
        st.markdown("<h4 style='color:#FFF; margin-top:15px;'>Defect Segmentation & Metrology Ledger</h4>", unsafe_allow_html=True)
        if res.defect_metrology:
            df_defects = pd.DataFrame(res.defect_metrology)
            cols_to_show = ["defect_id", "class_name", "length_mm", "width_mm", "area_mm2", "peak_height_um", "micro_short_hazard_index", "battery_failure_mode", "quality_tier"]
            st.dataframe(df_defects[cols_to_show], use_container_width=True)
        else:
            st.info("Zero defects detected in this zone. Surface within nominal production tolerances.")

# -------------------------------------------------------------------------
# TAB 2: INTERACTIVE 3D DEFECT TOPOGRAPHY
# -------------------------------------------------------------------------
with tabs[1]:
    st.markdown("<h4 style='color:#FFF;'>Interactive 3D Defect Topography Surface Reconstruction</h4>", unsafe_allow_html=True)
    if res is None:
        st.info("Trigger an inspection to view 3D topography.")
    else:
        # Downsample height map for fast interactive 3D rendering
        h_map = res.height_topography_map
        h_down = cv2.resize(h_map, (128, 128))
        
        # Plotly 3D Surface
        x_grid = np.linspace(0, 20, 128)  # 20mm FOV
        y_grid = np.linspace(0, 20, 128)
        
        fig3d = go.Figure(data=[go.Surface(
            z=h_down,
            x=x_grid,
            y=y_grid,
            colorscale='Viridis',
            colorbar=dict(title="Height (µm)")
        )])
        
        fig3d.update_layout(
            title=f"3D Topography Mesh (Peak: {np.max(h_map):.1f} µm, Valley: {np.min(h_map):.1f} µm)",
            autosize=True,
            height=550,
            scene=dict(
                xaxis_title="X Width (mm)",
                yaxis_title="Y Length (mm)",
                zaxis_title="Coating Profile (µm)",
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            margin=dict(l=10, r=10, b=10, t=40),
            paper_bgcolor="#0E121B"
        )
        st.plotly_chart(fig3d, use_container_width=True)
        
        # Cross-sectional slice
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            mid_y = h_down.shape[0] // 2
            fig_slice = px.line(
                x=x_grid,
                y=h_down[mid_y, :],
                labels={"x": "Cross-Section Distance (mm)", "y": "Height (µm)"},
                title="Cross-Sectional Height Profile Slice (Y = 10.0 mm)"
            )
            fig_slice.update_layout(paper_bgcolor="#0E121B", plot_bgcolor="#131824", font_color="#FFF")
            st.plotly_chart(fig_slice, use_container_width=True)
            
        with col_s2:
            st.markdown(f"""
            <div class="scada-panel" style="margin-top:25px;">
                <h5 style="color:#FFF;">3D Topography Metrics</h5>
                <p><b>Peak Protrusion (+Z):</b> {np.max(h_map):.2f} µm</p>
                <p><b>Max Depression (-Z):</b> {abs(min(0.0, float(np.min(h_map)))):.2f} µm</p>
                <p><b>Surface Roughness Ra:</b> {np.mean(np.abs(h_map - np.mean(h_map))):.2f} µm</p>
                <p><b>Separator Puncture Clearance:</b> {max(0.0, 14.0 - np.max(h_map)):.1f} µm (Threshold: 14.0 µm)</p>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------------------------------
# TAB 3: 1,200m JUMBO ROLL DIGITAL TWIN (WATERFALL VIEW)
# -------------------------------------------------------------------------
with tabs[2]:
    st.markdown("<h4 style='color:#FFF;'>Continuous 1,200m Roll Defect Map (Digital Twin Waterfall View)</h4>", unsafe_allow_html=True)
    roll_defects = api_roll.get("defect_records", []) if api_online else web_sync.roll_defect_map
    
    if not roll_defects:
        st.info("No defects logged on the active roll yet. Run inspections to populate the digital twin.")
    else:
        df_roll = pd.DataFrame(roll_defects)
        
        # 2D Roll Map Scatter
        fig_map = px.scatter(
            df_roll,
            x="linear_pos_m",
            y="cross_pos_mm",
            color="class_name",
            size="area_mm2",
            hover_data=["defect_id", "lane_id", "peak_height_um", "severity"],
            title=f"Roll Defect Map: {roll_summary['roll_id']} (Total Length: {roll_summary['total_roll_length_m']:.0f}m, Web Width: 650mm)",
            labels={"linear_pos_m": "Roll Length X (Meters)", "cross_pos_mm": "Cross-Web Width Y (mm)"},
            range_x=[0, max(100.0, roll_summary['inspected_length_m'] + 10.0)],
            range_y=[0, 650.0]
        )
        fig_map.update_layout(paper_bgcolor="#0E121B", plot_bgcolor="#131824", font_color="#FFF", height=450)
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Slitting Lane Quality Breakdown
        c_lane1, c_lane2 = st.columns(2)
        with c_lane1:
            lane_counts = df_roll["lane_id"].value_counts().reset_index()
            lane_counts.columns = ["Slitting Lane", "Defects Count"]
            fig_lane = px.bar(lane_counts, x="Slitting Lane", y="Defects Count", title="Defects Distribution across Slitting Lanes (1-4)", color="Slitting Lane")
            fig_lane.update_layout(paper_bgcolor="#0E121B", plot_bgcolor="#131824", font_color="#FFF")
            st.plotly_chart(fig_lane, use_container_width=True)
            
        with c_lane2:
            st.markdown(f"""
            <div class="scada-panel" style="margin-top:20px;">
                <h5 style="color:#FFF;">Slitter Downstream Recommendations</h5>
                <p><b>Lane 1 (0-162.5mm):</b> {'🔴 Flag for manual inspection' if roll_summary['defects_by_lane'].get(1,0) > 3 else '🟢 Prime Grade'}</p>
                <p><b>Lane 2 (162.5-325mm):</b> {'🔴 Flag for manual inspection' if roll_summary['defects_by_lane'].get(2,0) > 3 else '🟢 Prime Grade'}</p>
                <p><b>Lane 3 (325-487.5mm):</b> {'🔴 Flag for manual inspection' if roll_summary['defects_by_lane'].get(3,0) > 3 else '🟢 Prime Grade'}</p>
                <p><b>Lane 4 (487.5-650mm):</b> {'🔴 Flag for manual inspection' if roll_summary['defects_by_lane'].get(4,0) > 3 else '🟢 Prime Grade'}</p>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------------------------------
# TAB 4: BATTERY METROLOGY & STANDARDS AUDIT (GB/T 38031 & Plant QA Spec)
# -------------------------------------------------------------------------
with tabs[3]:
    st.markdown("<h4 style='color:#FFF;'>Quantitative Battery Safety Metrology (Plant QA Specification & GB/T 38031)</h4>", unsafe_allow_html=True)
    if res is None:
        st.info("Trigger an inspection to evaluate electrochemical safety metrics.")
    else:
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            max_hazard = max([d.get("micro_short_hazard_index", 0.0) for d in res.defect_metrology], default=0.0)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=max_hazard * 100.0,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Micro-Short Hazard Index (%) [Post-Calendering]"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#FF1744" if max_hazard > 0.8 else ("#FFD600" if max_hazard > 0.4 else "#00E676")},
                    'steps': [
                        {'range': [0, 40], 'color': "#1B5E20"},
                        {'range': [40, 80], 'color': "#F57F17"},
                        {'range': [80, 100], 'color': "#B71C1C"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            fig_gauge.update_layout(paper_bgcolor="#0E121B", font_color="#FFF", height=320)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with col_g2:
            st.markdown("""
            <div class="scada-panel">
                <h5 style="color:#FFF;">Plant Engineering QA & Safety Baseline Checklist</h5>
                <p><b>Surface Defects:</b> Max Scratch Length &le; 5.0mm, Void Area &le; 1.5mm²</p>
                <p><b>Delamination:</b> Prototype policy rejects configured delamination evidence; no escape-rate claim.</p>
                <p><b>Particle Protrusion:</b> Post-calendering height &lt; 11.0 µm (Safety margin vs 14µm separator)</p>
                <p><b>Edge Margin:</b> Uncoated foil margin 20.0 ± 1.0 mm, Waviness &le; 0.5 mm</p>
            </div>
            """, unsafe_allow_html=True)
            if res.standards_compliant:
                st.success("✅ **STATUS:** Fully Compliant with Plant QA Spec & GB/T 38031 Safety Baseline")
            else:
                st.error(f"❌ **NON-COMPLIANT:** {len(res.rejection_reasons)} clause violation(s) detected")

# -------------------------------------------------------------------------
# TAB 5: AI CLOSED-LOOP DIAGNOSTICS & EQUIPMENT FEEDBACK
# -------------------------------------------------------------------------
with tabs[4]:
    st.markdown("<h4 style='color:#FFF;'>AI Closed-Loop Root-Cause Diagnostics & Upstream Tuning Feedback</h4>", unsafe_allow_html=True)
    if res is None:
        st.info("Trigger an inspection to view equipment diagnostic feedback.")
    else:
        rc = res.root_cause_report
        st.markdown(f"""
        <div class="scada-panel" style="border-left:5px solid {'#FF1744' if rc['severity_level']=='CRITICAL' else '#00E676'};">
            <h4 style="color:#FFFFFF; margin-top:0;">Attributed Root Cause: <span style="color:#00F2FE;">{rc['primary_root_cause']}</span></h4>
            <p><b>Affected Upstream Equipment:</b> {rc['affected_equipment']}</p>
            <p><b>Evidence Level:</b> {rc.get('evidence_level', 'HEURISTIC')} &middot; <b>Defect Signature:</b> {rc['defect_signature']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h5 style='color:#FFF;'>Recommended Closed-Loop Equipment Adjustments</h5>", unsafe_allow_html=True)
        if rc.get("action_items"):
            for item in rc["action_items"]:
                with st.expander(f"⚙️ {item['equipment']} &rarr; Adjust {item['parameter']} ({item['delta']})", expanded=True):
                    st.write(f"**Action Urgency:** `{item['urgency']}`")
                    st.write(f"**Engineering Rationale:** {item['rationale']}")
                    if st.button(f"📲 Send Parameter Offset ({item['delta']}) to PLC", key=f"btn_{item['parameter']}"):
                        st.error("PLC parameter writes are disabled in the dashboard. Use an authenticated, audited control workflow.")
        else:
            st.success("All upstream equipment operating within nominal closed-loop setpoints.")

# -------------------------------------------------------------------------
# TAB 6: INDUSTRIAL HARDWARE & REAL-TIME BUDGET TELEMETRY
# -------------------------------------------------------------------------
with tabs[5]:
    st.markdown("<h4 style='color:#FFF;'>Industrial Hardware, Optical Budget & P99.9 Latency Determinism</h4>", unsafe_allow_html=True)
    plc_state = api_plc_state
    opt_budget = pipeline.optical_budget.compute_budget(line_speed_m_s=web_sync.line_speed_m_s)
    lat_budget = pipeline.latency_budget.compute_budget(line_speed_m_s=web_sync.line_speed_m_s)
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown(f"""
        <div class="scada-panel">
            <h5 style="color:#FFF;">📷 Optical & Line-Scan Throughput Budget (v = {web_sync.line_speed_m_s} m/s)</h5>
            <p><b>Spatial Resolution:</b> {opt_budget['spatial_resolution_x_um_per_px']} µm/px &middot; Coverage: {opt_budget['total_optical_coverage_mm']} mm</p>
            <p><b>Required Line Rate:</b> <b>{opt_budget['required_line_rate_khz']} kHz</b> (Period: {opt_budget['line_period_us']} µs)</p>
            <p><b>Recommended Exposure:</b> <b>{opt_budget['recommended_exposure_time_us']} µs</b> (Motion Blur: <b>{opt_budget['motion_blur_pixels']} px</b> &le; 0.5px)</p>
            <p><b>Modeled D95 Detectability:</b> <b>{opt_budget['d95_detectability_threshold_um']} µm</b> &middot; Raw bandwidth: {opt_budget['raw_bandwidth_mb_s']} MB/s</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="scada-panel">
            <h5 style="color:#FFF;">Modbus TCP Register Telemetry</h5>
            <p><b>Target PLC Address:</b> {plc_state['connection']['plc_ip']}:{plc_state['connection']['modbus_port']}</p>
            <p><b>Rejection Register:</b> <code>{plc_state['modbus_registers'].get('HR_1001', 0)}</code></p>
            <p><b>Connection State:</b> {plc_state['connection']['status']} (dashboard control disabled)</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_p2:
        st.markdown(f"""
        <div class="scada-panel">
            <h5 style="color:#FFF;">⚡ Camera-to-Ejector Deterministic Latency Budget</h5>
            <p><b>Modeled P50 Latency:</b> {lat_budget['p50_total_ms']} ms &middot; <b>Modeled P95:</b> {lat_budget['p95_total_ms']} ms</p>
            <p><b>Modeled P99.9:</b> <b style="color:#00F2FE;">{lat_budget['p99_9_total_ms']} ms</b> (not hardware-bench verified)</p>
            <p><b>Min Required Distance:</b> {lat_budget['min_required_ejector_distance_mm']} mm &rarr; <b>Installed Distance:</b> {lat_budget['installed_ejector_distance_mm']} mm</p>
            <p><b>Theoretical Margin Ratio:</b> <b>{lat_budget['safety_margin_ratio']}x</b>; hardware verification required.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="scada-panel">
            <h5 style="color:#FFF;">OPC UA Node Namespace Tree</h5>
            <p><code>ns=2;s=Device/Inspection/Verdict</code> &rarr; <b>{res.overall_verdict if res else 'NOMINAL'}</b></p>
            <p><code>ns=2;s=Device/Inspection/QualityTier</code> &rarr; <b>{res.quality_tier if res else 'GRADE_A'}</b></p>
            <p><code>ns=2;s=Device/Motion/LineVelocity</code> &rarr; <b>{web_sync.line_speed_m_s:.2f} m/s</b></p>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------------------
# TAB 7: GIGAFACTORY SPC, SPATIAL FFT & SLITTING YIELD OPTIMIZER
# -------------------------------------------------------------------------
with tabs[6]:
    st.markdown("<h4 style='color:#FFF;'>Experimental Defect Density, Spatial Periodicity & Slitting Model</h4>", unsafe_allow_html=True)
    passport_data = api_passport
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown(f"""
        <div class="scada-panel">
            <h5 style="color:#FFF;">Per-Lane Density (Cpk/Ppk require subgroup data)</h5>
        """, unsafe_allow_html=True)
        spc_dict = passport_data.get("six_sigma_quality_summary", {})
        for lane_k, lane_v in spc_dict.items():
            tier_col = "#00E676" if "TIER_1" in lane_v["tier"] else ("#FFD600" if "TIER_2" in lane_v["tier"] else "#FF1744")
            st.markdown(f"""
            <p><b>{lane_k.upper()}:</b> Cpk = <b style="color:{tier_col};">{lane_v['cpk']}</b> &middot; Density: <b>{lane_v['defect_density_per_100m']} def/100m</b> &rarr; <span style="color:{tier_col};"><code>{lane_v['tier']}</code></span></p>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Spatial FFT Mechanical Anomaly Report
        anomalies = passport_data.get("mechanical_health_diagnostics", [])
        st.markdown("<div class='scada-panel'><h5 style='color:#FFF;'>🔍 Spatial FFT Mechanical Diagnostics</h5>", unsafe_allow_html=True)
        if anomalies:
            for an in anomalies:
                st.markdown(f"""
                <p>⚠️ <b>Periodic Pattern:</b> Repeating every <b>{an['periodic_wavelength_mm']} mm</b> (Confidence: {an['confidence']*100:.1f}%)<br/>
                &rarr; <b>Matched Component:</b> <code style="color:#00F2FE;">{an['source_component']}</code><br/>
                &rarr; <b>Action:</b> {an['action']}</p>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ Zero repeating periodic spatial patterns detected (All Guide Rollers & Slurry Pumps Nominal).")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_s2:
        slit = passport_data.get("slitting_optimization", {})
        st.markdown(f"""
        <div class="scada-panel">
            <h5 style="color:#FFF;">✂️ Smart Slitting Yield & Economic Recovery Plan</h5>
            <p><b>Modeled Usable Recovery:</b> <b style="font-size:18px;">{slit.get('recovery_yield_pct', 0.0)}%</b></p>
            <p><b>EV Grade Output Area:</b> <b style="color:#00E676;">{slit.get('ev_grade_area_m2', 0)} m²</b> (High-Power Traction Cells)</p>
            <p><b>ESS Grade Output Area:</b> <b style="color:#FFD600;">{slit.get('ess_grade_area_m2', 0)} m²</b> (Grid Energy Storage Cells)</p>
            <p><b>Scrap Area:</b> <b style="color:#FF1744;">{slit.get('scrap_area_m2', 0)} m²</b></p>
            <p><b>Recommended Splice Positions:</b> <code>{slit.get('splice_locations_md_m', [])} m</code></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="scada-panel">
            <h5 style="color:#FFF;">🇪🇺 EU Battery Digital Product Passport (DPP) Header</h5>
            <p><b>Regulation:</b> <code>{passport_data.get('passport_standard')}</code></p>
            <p><b>Cleanroom Dew Point:</b> <b>{passport_data.get('traceability_genealogy',{}).get('cleanroom_dew_point_celsius')} °C</b></p>
            <p><b>Slurry Viscosity:</b> <b>{passport_data.get('traceability_genealogy',{}).get('slurry_viscosity_mpa_s')} mPa·s</b></p>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------------------
# TAB 8: QUALITY LOGS & DIGITAL ROLL CERTIFICATE
# -------------------------------------------------------------------------
with tabs[7]:
    st.markdown("<h4 style='color:#FFF;'>Provisional Local Quality Manifest (HMAC-SHA256)</h4>", unsafe_allow_html=True)
    
    cert = RollCertificateGenerator.build_certificate(
        roll_id=web_sync.roll.roll_id,
        batch_id=web_sync.roll.batch_id,
        inspected_length_m=web_sync.current_pos_m,
        total_length_m=web_sync.roll.total_length_m,
        defect_records=list(web_sync.roll_defect_map),
        spc_status=spc.get("status", "IN CONTROL"),
        root_cause_summary=res.root_cause_report['primary_root_cause'] if res else "Nominal",
        electrode_type=electrode_type,
        total_inspections=stats["total"] if stats["total"] else None,
        failed_inspections=stats["failed"] if stats["total"] else None,
        metric_provenance="Local dashboard simulation; not a production certificate",
    )
    
    st.markdown(f"""
    <div class="scada-panel" style="border-left:4px solid #00E676;">
        <p><b>Certificate ID:</b> <code>{cert.certificate_id}</code></p>
        <p><b>Payload Digest (SHA-256):</b> <code>{cert.payload_hash_sha256}</code></p>
        <p><b>Cryptographic HMAC Signature:</b> <code style="color:#00F2FE;">{cert.hmac_digital_signature}</code> (Algorithm: <code>{cert.signature_algorithm}</code>)</p>
        <p><b>Signature Verification:</b> <span>{'VALID FOR THIS PROCESS' if cert.verify_signature() else 'SIGNATURE MISMATCH'}</span></p>
        <p><b>Overall Quality Grade:</b> <span class="status-optimal">{cert.overall_quality_grade}</span> &middot; <b>Pass Rate:</b> {cert.pass_rate_pct:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "📥 Download Certificate (Markdown)",
            data=cert.to_markdown(),
            file_name=f"{cert.certificate_id}.md",
            mime="text/markdown",
            use_container_width=True
        )
    with col_dl2:
        st.download_button(
            "📥 Export Certificate (JSON Manifest)",
            data=pd.Series(cert.to_dict()).to_json(indent=2),
            file_name=f"{cert.certificate_id}.json",
            mime="application/json",
            use_container_width=True
        )
        
    with st.expander("👁️ Preview Full Certificate Document", expanded=False):
        st.markdown(cert.to_markdown())
