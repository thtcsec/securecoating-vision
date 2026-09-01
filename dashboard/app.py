"""
SecureCoating-Vision dashboard entrypoint.

Production renders a SCADA operations console from one API snapshot.
The research sandbox is opt-in for development/test only.
"""

import os
import sys
import time

# Streamlit executes this file as a script, so the repository root is not
# guaranteed to be importable. Bootstrap it before importing dashboard modules.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for path in (PROJECT_ROOT, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import requests
import streamlit as st
import yaml
from dashboard.api_client import InspectionApiClient

st.set_page_config(
    page_title="SecureCoating-Vision | Line Operations",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    html, body, [class*="css"], .stApp, .stMarkdown, .stCaption {
        font-family: "Segoe UI", "SF Pro Display", system-ui, sans-serif;
    }
    .stApp { background-color: #0A0D14; color: #F4F7FB; }
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2rem;
        max-width: 1440px;
    }
    section[data-testid="stSidebar"] {
        background: #0C1018;
        border-right: 1px solid #222C3E;
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
    /* Hide the in-sidebar collapse chevron so identity is not lost by accident. */
    section[data-testid="stSidebar"] [data-testid="stBaseButton-headerNoPadding"],
    section[data-testid="stSidebar"] [data-testid="stExpandSidebarButton"],
    section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
    section[data-testid="stSidebar"] button[kind="header"],
    section[data-testid="stSidebar"] button[kind="headerNoPadding"] {
        display: none !important;
    }
    /* If the rail is already collapsed, keep the reopen control visible. */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        position: fixed !important;
        left: 0.75rem !important;
        top: 0.75rem !important;
        z-index: 2147483647 !important;
        pointer-events: auto !important;
    }
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"] button {
        background: #151B28 !important;
        border: 1px solid #00E5FF !important;
        color: #00E5FF !important;
        min-width: 44px !important;
        min-height: 44px !important;
    }
    .view-rail-label {
        color: #7E8DA4;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin: 2px 0 6px 0;
    }
    div[data-testid="stMain"] div[data-testid="stRadio"] div[role="radiogroup"] {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        background: #131824;
        border: 1px solid #222C3E;
        border-radius: 10px;
        padding: 8px 10px;
        margin: 0 0 8px 0;
    }
    div[data-testid="stMain"] div[data-testid="stRadio"] div[role="radiogroup"] label {
        background: #151B28;
        border: 1px solid #222C3E;
        border-radius: 8px;
        padding: 8px 14px;
        margin: 0;
    }
    div[data-testid="stMain"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        background: #0E3A45;
        border-color: #00E5FF;
        color: #00E5FF;
    }
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
        padding: 14px 18px;
        margin-bottom: 10px;
        min-height: 96px;
        box-sizing: border-box;
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
        line-height: 1.18;
        color: #FFFFFF;
        overflow-wrap: anywhere;
        word-break: break-word;
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
    div[data-testid="stButtonGroup"] {
        background: #131824;
        border: 1px solid #222C3E;
        border-radius: 10px;
        padding: 6px 8px;
        margin: 0 0 8px 0;
    }
    div[data-testid="stButtonGroup"] > div {
        flex-wrap: wrap;
        row-gap: 4px;
    }
    div[data-testid="stButtonGroup"] button {
        min-height: 38px;
    }
    div[data-testid="stButtonGroup"] [aria-checked="true"] {
        background: #0E3A45 !important;
        color: #00E5FF !important;
        border-color: #00E5FF !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: #151B28;
        border: 1px solid #222C3E;
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 6px;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: #0E3A45;
        border-color: #00E5FF;
        color: #00E5FF;
    }
    .dataset-page-status {
        text-align: center;
        color: #FFFFFF;
        font-size: 14px;
        line-height: 1.35;
        padding-top: 4px;
    }
    .dataset-page-status span {
        color: #8C9BAE;
        font-size: 12px;
    }
    .vision-flow {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 8px 0 16px 0;
    }
    .vision-step {
        position: relative;
        background: linear-gradient(145deg, #151B28 0%, #101620 100%);
        border: 1px solid #26344A;
        border-radius: 10px;
        padding: 12px 14px;
        min-height: 88px;
    }
    .vision-step:not(:last-child)::after {
        content: "→";
        position: absolute;
        right: -14px;
        top: 30px;
        z-index: 2;
        color: #00E5FF;
        font-size: 18px;
        font-weight: 900;
    }
    .vision-step-no {
        color: #00E5FF;
        font-size: 10px;
        font-weight: 900;
        letter-spacing: 1.2px;
    }
    .vision-step-title {
        color: #FFFFFF;
        font-size: 15px;
        font-weight: 800;
        margin-top: 4px;
    }
    .vision-step-copy {
        color: #8C9BAE;
        font-size: 11px;
        margin-top: 4px;
    }
    .evidence-title {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 2px solid #243149;
        padding: 0 2px 7px 2px;
        margin-bottom: 8px;
    }
    .evidence-title b { color: #FFFFFF; }
    .evidence-chip {
        border: 1px solid #00E5FF;
        border-radius: 999px;
        color: #00E5FF;
        font-size: 9px;
        font-weight: 900;
        letter-spacing: .8px;
        padding: 3px 7px;
        white-space: nowrap;
    }
    .ai-verdict {
        background: linear-gradient(90deg, #111B24 0%, #17202D 100%);
        border: 1px solid #2C3E57;
        border-left: 5px solid #FFB300;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 12px 0 16px 0;
    }
    .ai-verdict-label {
        color: #8C9BAE;
        font-size: 10px;
        font-weight: 900;
        letter-spacing: 1.1px;
    }
    .ai-verdict-main {
        color: #FFFFFF;
        font-size: 20px;
        font-weight: 900;
        margin-top: 4px;
    }
    .ai-verdict-detail {
        color: #B4C0D0;
        font-size: 12px;
        margin-top: 4px;
    }
    div[data-testid="stImage"] img {
        border-radius: 8px;
        border: 1px solid #26344A;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.35);
        background: #0B0F16;
    }
    div[data-testid="stImage"] button {
        display: none !important;
    }
    div[data-testid="stButton"] button {
        background: #151B28;
        border: 1px solid #2A3A52;
        color: #E8EEF6;
        border-radius: 8px;
    }
    div[data-testid="stButton"] button:hover {
        border-color: #00E5FF;
        color: #00E5FF;
    }
    div[data-testid="stButton"] button:disabled {
        opacity: 0.45;
        border-color: #222C3E;
        color: #7E8DA4;
    }
    div[data-testid="stSelectbox"] > div,
    div[data-testid="stTextInput"] > div > div,
    div[data-baseweb="select"] {
        background: #151B28 !important;
        border-radius: 8px;
        border-color: #2A3A52 !important;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #222C3E;
        border-radius: 8px;
        overflow: hidden;
    }
    div[data-testid="stAlert"] {
        background: #151B28;
        border: 1px solid #26344A;
        border-radius: 8px;
        color: #D5DEE9;
    }
    div[data-testid="stMetric"] {
        background: #151B28;
        border: 1px solid #222C3E;
        border-radius: 8px;
        padding: 10px 12px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 22px !important;
        overflow-wrap: anywhere;
    }
    [data-testid="stHeaderActionElements"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    #MainMenu, footer, .stDeployButton { display: none !important; }
    header[data-testid="stHeader"] {
        background: transparent;
        min-height: 3rem;
    }
    .replay-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 14px;
        margin: 6px 0 12px 0;
    }
    .replay-cell {
        background: linear-gradient(180deg, #131824 0%, #0E121B 100%);
        border: 1px solid #26344A;
        border-radius: 10px;
        padding: 10px 10px 12px 10px;
    }
    .replay-cell img {
        width: 100%;
        display: block;
        border-radius: 8px;
        border: 1px solid #26344A;
        background: #0B0F16;
    }
    .replay-empty {
        min-height: 140px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #8C9BAE;
        font-size: 13px;
        text-align: center;
        background: #10161F;
        border-radius: 8px;
        border: 1px dashed #2A3A52;
        padding: 16px;
    }
    .class-chip {
        display: inline-block;
        background: #152033;
        border: 1px solid #2C4A6E;
        color: #9ED8FF;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
        padding: 3px 10px;
        margin: 8px 6px 0 0;
        letter-spacing: 0.3px;
    }
    .mix-row {
        display: grid;
        grid-template-columns: 140px 1fr 64px;
        gap: 10px;
        align-items: center;
        margin: 6px 0;
        color: #B4C0D0;
        font-size: 12px;
    }
    .mix-track {
        height: 8px;
        background: #10161F;
        border: 1px solid #26344A;
        border-radius: 999px;
        overflow: hidden;
    }
    .mix-fill {
        display: block;
        height: 100%;
        background: linear-gradient(90deg, #00E5FF 0%, #7C4DFF 100%);
    }
    .gallery-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 8px 0 14px 0;
    }
    .gallery-card {
        background: linear-gradient(180deg, #131824 0%, #0E121B 100%);
        border: 1px solid #26344A;
        border-radius: 10px;
        padding: 8px;
    }
    .gallery-card.selected { border-color: #00E5FF; }
    .gallery-card img {
        width: 100%;
        display: block;
        border-radius: 7px;
        border: 1px solid #26344A;
        background: #0B0F16;
        aspect-ratio: 1;
        object-fit: cover;
    }
    .gallery-name {
        color: #9ED8FF;
        font-size: 11px;
        font-weight: 700;
        margin-top: 8px;
        overflow-wrap: anywhere;
    }
    @media (max-width: 1100px) {
        .vision-flow, .replay-grid, .gallery-grid { grid-template-columns: 1fr 1fr; }
        .vision-step::after { display: none; }
    }
    div[data-testid="stSkeleton"] { display: none !important; }
    /* Streamlit fades the tree on every rerun (tab switch + 5s live strip). */
    .stApp, .stApp * {
        animation: none !important;
        animation-duration: 0s !important;
        animation-delay: 0s !important;
        transition: none !important;
        transition-duration: 0s !important;
    }
    [data-stale="true"] {
        opacity: 1 !important;
        filter: none !important;
    }

</style>
""", unsafe_allow_html=True)

DASHBOARD_SANDBOX_ENABLED = (
    os.environ.get("SECURECOATING_ENV", "production").lower() in {"development", "test"}
    and os.environ.get("SECURECOATING_DASHBOARD_SANDBOX", "false").lower()
    in {"1", "true", "yes"}
)

with open(os.path.join(PROJECT_ROOT, "configs", "app.yaml"), "r") as f:
    app_config = yaml.safe_load(f)

api_client = InspectionApiClient(
    base_url=os.environ.get("SECURECOATING_API_URL", "http://127.0.0.1:8000"),
    api_key=os.environ.get("SECURECOATING_API_KEY", ""),
)

if not DASHBOARD_SANDBOX_ENABLED:
    from dashboard.production_console import render_live_strip, render_production_console

    force_reload = bool(st.session_state.pop("ops_force_reload", False))

    def _retain_full_snapshot_fields(previous, live):
        """Live telemetry may omit catalog/certificate; keep the last full payload."""
        if not isinstance(previous, dict) or not isinstance(live, dict):
            return live
        if live.get("snapshot_scope") != "live":
            return live
        merged = dict(live)
        if previous.get("dataset_catalog") is not None:
            merged["dataset_catalog"] = previous["dataset_catalog"]
        previous_trace = previous.get("traceability")
        live_trace = live.get("traceability") or {}
        if live_trace.get("certificate_status") == "DEFERRED_TO_FULL_SNAPSHOT" and previous_trace:
            merged["traceability"] = previous_trace
        return merged

    def _load_ops_snapshot():
        last_error = None
        for attempt in range(3):
            try:
                return api_client.operations_snapshot()
            except (OSError, ValueError, KeyError, requests.RequestException) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.0)
        if last_error is None:
            raise RuntimeError("Operations snapshot retry exhausted")
        raise last_error

    if force_reload or "ops_snapshot" not in st.session_state:
        try:
            st.session_state["ops_snapshot"] = _load_ops_snapshot()
        except (OSError, ValueError, KeyError, requests.RequestException):
            if "ops_snapshot" not in st.session_state:
                st.error(
                    "Authoritative API unavailable. Production dashboard refuses local fallback."
                )
                st.stop()
            st.warning("Snapshot reload failed. Last authoritative snapshot remains on screen.")

    @st.fragment(run_every="5s")
    def refresh_live_telemetry() -> None:
        try:
            snapshot = _retain_full_snapshot_fields(
                st.session_state.get("ops_snapshot"),
                api_client.operations_snapshot(
                    timeout_seconds=api_client.timeout_seconds,
                    scope="live",
                ),
            )
            st.session_state["ops_snapshot"] = snapshot
        except (OSError, ValueError, KeyError, requests.RequestException):
            st.warning("Live telemetry refresh failed. Last authoritative snapshot remains on screen.")
            snapshot = st.session_state["ops_snapshot"]
        render_live_strip(snapshot)

    @st.fragment
    def render_operator_views() -> None:
        render_production_console(st.session_state["ops_snapshot"], api_client)

    refresh_live_telemetry()
    render_operator_views()
    st.stop()

from dashboard.sandbox_console import render_sandbox_console
from industrial.optical_budget import OpticalThroughputBudgetEngine
from industrial.latency_budget import LatencyBudgetEngine
from inference.predictor import CoatingPredictor
from inference.sensor_fusion import SensorFusionManager
from inference.failsafe import FailSafeManager
from inference.electrode_metrology import ElectrodeMetrologyEngine
from inference.multi_stage_pipeline import MultiStageIndustrialPipeline
from industrial.protocol_manager import IndustrialProtocolManager
from industrial.web_synchronizer import WebSynchronizer
from traceability.quality_memory import QualityMemory
from traceability.root_cause_engine import RootCauseDiagnosticEngine

with open(os.path.join(PROJECT_ROOT, "configs", "model.yaml"), "r") as f:
    model_config = yaml.safe_load(f)


@st.cache_resource
def get_sandbox_resources():
    local_predictor = CoatingPredictor(model_config)
    local_fusion = SensorFusionManager(target_size=(1024, 1024), enable_mock=True)
    local_failsafe = FailSafeManager()
    local_industrial = IndustrialProtocolManager({"enabled": False, "mock_mode": True})
    local_web_sync = WebSynchronizer()
    local_metrology = ElectrodeMetrologyEngine(pixel_to_mm_ratio=0.1)
    local_root_cause = RootCauseDiagnosticEngine()
    local_pipeline = MultiStageIndustrialPipeline(
        predictor=local_predictor,
        fusion_manager=local_fusion,
        failsafe_manager=local_failsafe,
        industrial_manager=local_industrial,
        web_synchronizer=local_web_sync,
        pixel_to_mm_ratio=0.1,
    )
    db_path = os.environ.get(
        "SECURECOATING_DB_PATH",
        app_config.get("paths", {}).get("db_path", "data/quality_history.db"),
    )
    if not os.path.isabs(db_path):
        db_path = os.path.join(PROJECT_ROOT, db_path)
    return (
        local_predictor, local_fusion, local_failsafe, local_industrial,
        local_web_sync, local_metrology, local_root_cause, local_pipeline,
        QualityMemory(db_path),
    )


(
    predictor, fusion_mgr, failsafe, industrial_mgr, web_sync,
    metrology_engine, root_cause_engine, pipeline, quality_mem,
) = get_sandbox_resources()
optical_budget = OpticalThroughputBudgetEngine()
latency_budget = LatencyBudgetEngine(optical_engine=optical_budget)

api_online = True
api_error = ""
try:
    snapshot = api_client.operations_snapshot()
    api_health = snapshot["readiness"]
    api_roll = snapshot["roll"]
    api_summary = api_roll["summary"]
    api_roll_id = api_summary["roll_id"]
    api_batch_id = api_summary["batch_id"]
    api_stats = snapshot["quality"]["stats"]
    api_spc = snapshot["quality"]["spc"]
    api_plc_state = snapshot["industrial"]
    api_passport = api_client.passport()
    api_libad_protocol = api_client.libad_protocol()
except (OSError, ValueError, KeyError, requests.RequestException) as exc:
    api_online = False
    api_error = str(exc)
    api_health = {}
    api_roll = {}
    api_summary = web_sync.get_roll_defect_summary()
    api_roll_id = None
    api_batch_id = web_sync.roll.batch_id
    api_stats = quality_mem.get_batch_stats(api_batch_id)
    api_spc = quality_mem.check_spc_alarms(api_batch_id)
    api_plc_state = industrial_mgr.get_plc_state()
    api_passport = pipeline.get_gigafactory_spc_summary()
    api_libad_protocol = {
        "dataset": {
            "official_protocol_complete": False,
            "comparable_to_paper": False,
        }
    }

render_sandbox_console(
    api_online=api_online,
    api_error=api_error,
    api_client=api_client,
    api_health=api_health,
    api_roll=api_roll,
    api_summary=api_summary,
    api_roll_id=api_roll_id,
    api_batch_id=api_batch_id,
    api_stats=api_stats,
    api_spc=api_spc,
    api_plc_state=api_plc_state,
    api_passport=api_passport,
    api_libad_protocol=api_libad_protocol,
    fusion_mgr=fusion_mgr,
    failsafe=failsafe,
    industrial_mgr=industrial_mgr,
    web_sync=web_sync,
    pipeline=pipeline,
    quality_mem=quality_mem,
    optical_budget=optical_budget,
    latency_budget=latency_budget,
)
