"""
SecureCoating-Vision dashboard entrypoint.

Production renders a SCADA operations console from one API snapshot.
The research sandbox is opt-in for development/test only.
"""

import os
import sys

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

os.chdir(PROJECT_ROOT)

DASHBOARD_SANDBOX_ENABLED = (
    os.environ.get("SECURECOATING_ENV", "production").lower() in {"development", "test"}
    and os.environ.get("SECURECOATING_DASHBOARD_SANDBOX", "false").lower()
    in {"1", "true", "yes"}
)

with open("configs/app.yaml", "r") as f:
    app_config = yaml.safe_load(f)

api_client = InspectionApiClient(
    base_url=os.environ.get("SECURECOATING_API_URL", "http://127.0.0.1:8000"),
    api_key=os.environ.get("SECURECOATING_API_KEY", ""),
)

if not DASHBOARD_SANDBOX_ENABLED:
    from dashboard.production_console import render_production_console

    try:
        snapshot = api_client.operations_snapshot()
    except (OSError, ValueError, KeyError, requests.RequestException) as exc:
        st.error(
            "Authoritative API unavailable. Production dashboard refuses local fallback. "
            f"Details: {exc}"
        )
        st.stop()
    render_production_console(snapshot, api_client)
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

with open("configs/model.yaml", "r") as f:
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
