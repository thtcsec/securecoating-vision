"""Isolated research/sandbox dashboard. Not the production operations surface."""

from __future__ import annotations

import base64
import json
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


def render_sandbox_console(
    *,
    api_online: bool,
    api_error: str,
    api_client: Any,
    api_health: Dict[str, Any],
    api_roll: Dict[str, Any],
    api_summary: Dict[str, Any],
    api_roll_id: Optional[str],
    api_batch_id: str,
    api_stats: Dict[str, Any],
    api_spc: Dict[str, Any],
    api_plc_state: Dict[str, Any],
    api_passport: Dict[str, Any],
    api_libad_protocol: Dict[str, Any],
    fusion_mgr: Any,
    failsafe: Any,
    industrial_mgr: Any,
    web_sync: Any,
    pipeline: Any,
    quality_mem: Any,
    optical_budget: Any,
    latency_budget: Any,
) -> None:
    from traceability.roll_certificate import RollCertificateGenerator

    if "inspection_history" not in st.session_state:
        st.session_state.inspection_history = []
    if "last_pipeline_result" not in st.session_state:
        st.session_state.last_pipeline_result = None
    if "libad_demo" not in st.session_state:
        st.session_state.libad_demo = None

    dashboard_badges = (
        '<span class="stage-badge stage-done">● AUTHORITATIVE API</span>'
        '<span class="stage-badge" style="background:#263238; color:#90A4AE;">SANDBOX SURFACE</span>'
        if api_online
        else
        '<span class="stage-badge" style="background:#F57F17; color:#000;">● ISOLATED SANDBOX</span>'
        '<span class="stage-badge" style="background:#263238; color:#90A4AE;">PLC CONTROL DISABLED</span>'
    )
    st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #222C3E; padding-bottom:12px; margin-bottom:15px;">
    <div>
        <h2 style="color:#FFFFFF; margin:0; font-weight:800; font-size:24px;">SecureCoating-Vision | Research Sandbox</h2>
        <span style="color:#8C9BAE; font-size:12px;">Simulator and evidence-demo surface. This is not the production operations console.</span>
    </div>
    <div>
        {dashboard_badges}
    </div>
</div>
""", unsafe_allow_html=True)

    roll_summary = api_summary
    stats = api_stats
    spc = api_spc
    if not api_online:
        st.warning(f"Authoritative API unavailable; showing isolated local sandbox data: {api_error}")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.markdown(f"""<div class="scada-metric-card">
            <div class="metric-label">Line Velocity</div>
            <div class="metric-val">{float(roll_summary.get('line_speed_m_s', 0.0)):.2f} <span style="font-size:14px; color:#8C9BAE;">m/s</span></div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="scada-metric-card" style="border-left-color:#00E676;">
            <div class="metric-label">Roll Progress</div>
            <div class="metric-val">{roll_summary['inspected_length_m']:.1f} / {roll_summary['total_roll_length_m']:.0f}m</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        pass_rate = stats["pass_rate"] if stats.get("pass_rate") is not None else None
        if pass_rate is None:
            pass_html = "NO DATA"
            color_cls = ""
        else:
            color_cls = "status-optimal" if pass_rate >= 95 else ("status-warning" if pass_rate >= 85 else "status-danger")
            pass_html = f"{pass_rate}%"
        st.markdown(f"""<div class="scada-metric-card" style="border-left-color:#FFD600;">
            <div class="metric-label">Yield Pass Rate</div>
            <div class="metric-val {color_cls}">{pass_html}</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        lat = stats["avg_latency_ms"] if stats.get("avg_latency_ms") is not None else None
        lat_html = "NO DATA" if lat is None else f'{lat} <span style="font-size:14px; color:#8C9BAE;">ms</span>'
        st.markdown(f"""<div class="scada-metric-card" style="border-left-color:#E040FB;">
            <div class="metric-label">Pipeline Latency</div>
            <div class="metric-val" style="color:#E040FB;">{lat_html}</div>
        </div>""", unsafe_allow_html=True)
    with m5:
        spc_st = spc.get("status", "NO DATA")
        spc_color = "status-optimal" if spc_st == "IN CONTROL" else ("status-warning" if spc_st == "WARNING" else "status-danger")
        st.markdown(f"""<div class="scada-metric-card" style="border-left-color:#FF1744;">
            <div class="metric-label">SPC Quality State</div>
            <div class="metric-val {spc_color}">{spc_st}</div>
        </div>""", unsafe_allow_html=True)
    with m6:
        sandbox_state = failsafe.system_state.value if failsafe is not None else "UNKNOWN"
        sys_st = api_health.get("system_state", sandbox_state)
        sys_col = "status-optimal" if sys_st == "OPTIMAL" else "status-warning"
        st.markdown(f"""<div class="scada-metric-card" style="border-left-color:#00E5FF;">
            <div class="metric-label">Fail-Safe Health</div>
            <div class="metric-val {sys_col}">{sys_st}</div>
        </div>""", unsafe_allow_html=True)

    st.sidebar.markdown("<h3 style='color:#FFF;'>Sandbox Recipe</h3>", unsafe_allow_html=True)
    st.sidebar.caption("These controls mutate the isolated sandbox only. They do not write a production PLC.")
    electrode_type = st.sidebar.selectbox(
        "Electrode Chemistry",
        ["Cathode_LFP (LiFePO4)", "Cathode_NMC811", "Anode_Graphite_Silicon"],
        index=0,
    )
    line_speed_slider = st.sidebar.slider("Line Velocity (m/s)", 0.5, 3.0, 1.8, 0.1)
    web_sync.set_line_speed(line_speed_slider)

    st.sidebar.markdown("---")
    st.sidebar.markdown("<h3 style='color:#FFF;'>Sandbox inspection</h3>", unsafe_allow_html=True)
    defect_injection = st.sidebar.selectbox(
        "Sensor Simulation Target",
        ["Random Physical Anomaly", "Scratch (Doctor Blade Nick)", "Void (Slurry Degassing)", "Blister (Solvent Boil)", "Delamination (Skinning)", "Nominal (Clean Surface)"],
    )
    cross_pos = st.sidebar.slider("Cross-Web Position (mm)", 0.0, 650.0, 325.0, 10.0)
    if st.sidebar.button("RUN SANDBOX 7-STAGE INSPECTION", type="primary", width="stretch"):
        sim_map = {
            "Scratch (Doctor Blade Nick)": "scratch",
            "Void (Slurry Degassing)": "void",
            "Blister (Solvent Boil)": "blister",
            "Delamination (Skinning)": "delamination",
            "Nominal (Clean Surface)": "none",
            "Random Physical Anomaly": "random",
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

    st.sidebar.markdown("---")
    st.sidebar.markdown("<h3 style='color:#FFF;'>90s LIBAD Evidence Demo</h3>", unsafe_allow_html=True)
    st.sidebar.caption(
        "Deterministic protocol-fixture VIS + X-rayL cases. Official LIBAD inputs are "
        "reported separately; thermal/profilometry remain simulated adapters."
    )
    libad_case = st.sidebar.radio(
        "Demo case",
        [1, 2, 3, 4],
        format_func=lambda value: {
            1: "1 · Normal agreement → PASS",
            2: "2 · Surface defect → REJECT",
            3: "3 · Internal X-ray → REJECT",
            4: "4 · Disagreement → HOLD",
        }[value],
    )
    if st.sidebar.button("RUN 90s EVIDENCE CASE", width="stretch"):
        try:
            if api_online:
                st.session_state.libad_demo = api_client.libad_demo(int(libad_case))
            else:
                from libad.demo_cases import run_libad_demo_case

                st.session_state.libad_demo = run_libad_demo_case(int(libad_case))
            st.rerun()
        except (OSError, ValueError, requests.RequestException) as exc:
            st.error(f"LIBAD demo endpoint failed; no local fallback was substituted: {exc}")

    tabs = st.tabs([
        "90s Evidence Lane",
        "Live Multi-Stage Station",
        "3D Defect Topography",
        "1,200m Roll Digital Twin",
        "Prototype Battery Metrology",
        "AI Closed-Loop Diagnostics",
        "Hardware & Protocol Telemetry",
        "Prototype SPC & Slitting Scenario",
        "Traceability & Quality Certificate",
    ])

    res = st.session_state.last_pipeline_result
    demo = st.session_state.libad_demo

    with tabs[0]:
        st.markdown(
            "<h4 style='color:#FFF;'>Evidence-Gated Multimodal Inspection for Battery Electrode Manufacturing</h4>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Staged protocol-fixture validation for a LIBAD-compatible VIS + X-rayL contract. "
            "Thermal and 3D profilometry remain simulated interface adapters."
        )
        if not api_libad_protocol.get("dataset", {}).get("official_protocol_complete", False):
            st.warning(
                "Official LIBAD dataset + all 10 splits are not verified in this runtime. "
                "These staged cases are protocol_fixture evidence and are not paper-comparable."
            )
        if demo is None:
            st.info("Use RUN 90s EVIDENCE CASE in the sidebar. Only four situations are staged.")
        else:
            if "frames_png_base64" in demo:
                def decode_demo_frame(name):
                    payload = base64.b64decode(demo["frames_png_base64"][name], validate=True)
                    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise ValueError(f"API returned an invalid {name} PNG frame")
                    return frame

                vis = decode_demo_frame("vis_bgr")
                xray = decode_demo_frame("xray_bgr")
            else:
                vis = demo["frames"]["vis_bgr"]
                xray = demo["frames"]["xray_bgr"]
            action = demo["decision"]["action"]
            vis_col, xray_col = st.columns(2)
            with vis_col:
                st.caption("VIS (visible-light surface)")
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), width="stretch")
            with xray_col:
                st.caption("X-rayL (inline-compatible density)")
                xray_gray = cv2.cvtColor(xray, cv2.COLOR_BGR2GRAY)
                st.image(cv2.cvtColor(cv2.applyColorMap(xray_gray, cv2.COLORMAP_BONE), cv2.COLOR_BGR2RGB), width="stretch")
            if action == "PASS":
                st.success(f"Case {demo['case_id']} — {demo['case_name']}: **PASS**")
            elif action == "HOLD":
                st.warning(f"Case {demo['case_id']} — {demo['case_name']}: **HOLD** — manual QA required")
            else:
                st.error(f"Case {demo['case_id']} — {demo['case_name']}: **REJECT**")
            st.write(demo["decision"]["reason"])
            cert = demo["certificate"]
            st.markdown("#### Closing identity / certificate / PLC screen")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Roll ID", cert["roll_id"])
            c2.metric("Batch ID", cert["batch_id"])
            c3.metric("Part ID", cert["part_id"])
            c4.metric("Decision", cert["decision"])
            s1, s2, s3 = st.columns(3)
            s1.metric("VIS score", f"{cert['vis_score']:.3f}" if cert["vis_score"] is not None else "n/a")
            s2.metric("X-rayL score", f"{cert['xray_score']:.3f}" if cert["xray_score"] is not None else "n/a")
            s3.metric("Calibration", cert["calibration_state"])
            st.code(
                f"model_hash: {cert['model_hash']}\n"
                f"commit: {cert['commit_hash']}\n"
                f"reason: {cert['decision_reason']}\n"
                f"certificate: {cert['hmac_digital_signature']}\n"
                f"PLC: {cert['plc_state']}",
                language="text",
            )

    with tabs[1]:
        if res is None:
            st.info("Run a sandbox 7-stage inspection from the sidebar to populate this view.")
        else:
            st.markdown(f"""
            <div class="scada-panel" style="padding:10px 16px;">
                <div style="font-size:12px; color:#8C9BAE; font-weight:700; margin-bottom:6px;">7-STAGE INDUSTRIAL EXECUTION TIMELINE (Total: {res.total_pipeline_latency_ms:.2f} ms)</div>
                <div style="display:flex; justify-content:space-between; font-size:11px;">
                    <span class="stage-badge stage-done">1. Web Sync ({res.stage_latencies_ms.get('stage_1_web_sync_ms',0):.1f}ms)</span>
                    <span class="stage-badge stage-done">2. Sensor Acq ({res.stage_latencies_ms.get('stage_2_acquisition_ms',0):.1f}ms)</span>
                    <span class="stage-badge stage-done">3. Homography ({res.stage_latencies_ms.get('stage_3_fusion_ms',0):.1f}ms)</span>
                    <span class="stage-badge stage-done">4. AI Inference ({res.stage_latencies_ms.get('stage_4_ai_inference_ms',0):.1f}ms)</span>
                    <span class="stage-badge stage-done">5. Metrology ({res.stage_latencies_ms.get('stage_5_metrology_ms',0):.1f}ms)</span>
                    <span class="stage-badge stage-done">6. PLC Gate ({res.stage_latencies_ms.get('stage_6_decision_plc_ms',0):.1f}ms)</span>
                    <span class="stage-badge stage-done">7. Diagnostics ({res.stage_latencies_ms.get('stage_7_root_cause_ms',0):.1f}ms)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if res.overall_verdict == "PASS":
                st.success(f"**PASS** | Quality Grade: **{res.quality_tier}** | Roll Location: **X = {res.roll_coordinate['linear_pos_m']}m**, Lane: **{res.roll_coordinate['lane_id']}**")
            elif res.overall_verdict == "HOLD":
                st.warning(f"**HOLD** | Quality Grade: **{res.quality_tier}** | PLC Gate Action: **{res.plc_gate_action}** | {', '.join(res.rejection_reasons)}")
            else:
                st.error(f"**REJECT** | Quality Grade: **{res.quality_tier}** | PLC Gate Action: **{res.plc_gate_action}** | Violations: {', '.join(res.rejection_reasons)}")
            st.markdown("<h4 style='color:#FFF; margin-top:10px;'>Multi-Modal Synchronized Sensor Ingestion</h4>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.caption("Optical Brightfield (RGB)")
                st.image(cv2.cvtColor(res.optical_brightfield, cv2.COLOR_BGR2RGB), width="stretch")
            with c2:
                st.caption("Optical Darkfield Scatter")
                st.image(cv2.cvtColor(res.optical_darkfield, cv2.COLOR_BGR2RGB), width="stretch")
            with c3:
                st.caption("Lock-in Thermography Phase")
                disp_th = cv2.applyColorMap(res.thermal_diffusivity_phase, cv2.COLORMAP_INFERNO)
                st.image(cv2.cvtColor(disp_th, cv2.COLOR_BGR2RGB), width="stretch")
            with c4:
                st.caption("Confocal 3D Laser Profile")
                norm_h = cv2.normalize(res.height_topography_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                disp_h = cv2.applyColorMap(norm_h, cv2.COLORMAP_JET)
                st.image(cv2.cvtColor(disp_h, cv2.COLOR_BGR2RGB), width="stretch")
            st.markdown("<h4 style='color:#FFF; margin-top:15px;'>Defect Segmentation & Metrology Ledger</h4>", unsafe_allow_html=True)
            if res.defect_metrology:
                df_defects = pd.DataFrame(res.defect_metrology)
                cols_to_show = ["defect_id", "class_name", "length_mm", "width_mm", "area_mm2", "peak_height_um", "micro_short_hazard_index", "battery_failure_mode", "quality_tier"]
                st.dataframe(df_defects[cols_to_show], width="stretch")
            elif res.raw_detections:
                st.warning(
                    f"{len(res.raw_detections)} raw detector proposals are present, but metrology/"
                    "release is blocked. This is not a zero-anomaly claim."
                )
            else:
                st.info("No metrology objects in this zone. Absence of metrology is not a production PASS.")

    with tabs[2]:
        st.markdown("<h4 style='color:#FFF;'>Interactive 3D Defect Topography Surface Reconstruction</h4>", unsafe_allow_html=True)
        if res is None:
            st.info("Run a sandbox inspection to view 3D topography.")
        else:
            h_map = res.height_topography_map
            h_down = cv2.resize(h_map, (128, 128))
            x_grid = np.linspace(0, 20, 128)
            y_grid = np.linspace(0, 20, 128)
            fig3d = go.Figure(data=[go.Surface(
                z=h_down, x=x_grid, y=y_grid, colorscale="Viridis",
                colorbar=dict(title="Height (µm)"),
            )])
            fig3d.update_layout(
                title=f"3D Topography Mesh (Peak: {np.max(h_map):.1f} µm, Valley: {np.min(h_map):.1f} µm)",
                autosize=True, height=550,
                scene=dict(
                    xaxis_title="X Width (mm)", yaxis_title="Y Length (mm)",
                    zaxis_title="Coating Profile (µm)", aspectmode="manual",
                    aspectratio=dict(x=1, y=1, z=0.5),
                ),
                margin=dict(l=10, r=10, b=10, t=40), paper_bgcolor="#0E121B",
            )
            st.plotly_chart(fig3d, width="stretch")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                mid_y = h_down.shape[0] // 2
                fig_slice = px.line(
                    x=x_grid, y=h_down[mid_y, :],
                    labels={"x": "Cross-Section Distance (mm)", "y": "Height (µm)"},
                    title="Cross-Sectional Height Profile Slice (Y = 10.0 mm)",
                )
                fig_slice.update_layout(paper_bgcolor="#0E121B", plot_bgcolor="#131824", font_color="#FFF")
                st.plotly_chart(fig_slice, width="stretch")
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

    with tabs[3]:
        st.markdown("<h4 style='color:#FFF;'>Continuous 1,200m Roll Defect Map (Digital Twin Waterfall View)</h4>", unsafe_allow_html=True)
        roll_defects = api_roll.get("defect_records", []) if api_online else web_sync.roll_defect_map
        if not roll_defects:
            st.info("No defects logged on the active roll yet.")
        else:
            df_roll = pd.DataFrame(roll_defects)
            fig_map = px.scatter(
                df_roll, x="linear_pos_m", y="cross_pos_mm", color="class_name", size="area_mm2",
                hover_data=["defect_id", "lane_id", "peak_height_um", "severity"],
                title=f"Roll Defect Map: {roll_summary['roll_id']} (Total Length: {roll_summary['total_roll_length_m']:.0f}m, Web Width: 650mm)",
                labels={"linear_pos_m": "Roll Length X (Meters)", "cross_pos_mm": "Cross-Web Width Y (mm)"},
                range_x=[0, max(100.0, roll_summary["inspected_length_m"] + 10.0)],
                range_y=[0, 650.0],
            )
            fig_map.update_layout(paper_bgcolor="#0E121B", plot_bgcolor="#131824", font_color="#FFF", height=450)
            st.plotly_chart(fig_map, width="stretch")
            c_lane1, c_lane2 = st.columns(2)
            with c_lane1:
                lane_counts = df_roll["lane_id"].value_counts().reset_index()
                lane_counts.columns = ["Slitting Lane", "Defects Count"]
                fig_lane = px.bar(lane_counts, x="Slitting Lane", y="Defects Count", title="Defects Distribution across Slitting Lanes (1-4)", color="Slitting Lane")
                fig_lane.update_layout(paper_bgcolor="#0E121B", plot_bgcolor="#131824", font_color="#FFF")
                st.plotly_chart(fig_lane, width="stretch")
            with c_lane2:
                st.markdown(f"""
                <div class="scada-panel" style="margin-top:20px;">
                    <h5 style="color:#FFF;">Slitter Downstream Recommendations</h5>
                    <p><b>Lane 1 (0-162.5mm):</b> {'Flag for manual inspection' if roll_summary['defects_by_lane'].get(1,0) > 3 else 'Prime Grade'}</p>
                    <p><b>Lane 2 (162.5-325mm):</b> {'Flag for manual inspection' if roll_summary['defects_by_lane'].get(2,0) > 3 else 'Prime Grade'}</p>
                    <p><b>Lane 3 (325-487.5mm):</b> {'Flag for manual inspection' if roll_summary['defects_by_lane'].get(3,0) > 3 else 'Prime Grade'}</p>
                    <p><b>Lane 4 (487.5-650mm):</b> {'Flag for manual inspection' if roll_summary['defects_by_lane'].get(4,0) > 3 else 'Prime Grade'}</p>
                </div>
                """, unsafe_allow_html=True)

    with tabs[4]:
        st.markdown("<h4 style='color:#FFF;'>Prototype Battery Metrology (unqualified engineering thresholds)</h4>", unsafe_allow_html=True)
        if res is None:
            st.info("Run a sandbox inspection to evaluate electrochemical safety metrics.")
        else:
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                max_hazard = max([d.get("micro_short_hazard_index", 0.0) for d in res.defect_metrology], default=0.0)
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=max_hazard * 100.0,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Micro-Short Hazard Index (%) [Post-Calendering]"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#FF1744" if max_hazard > 0.8 else ("#FFD600" if max_hazard > 0.4 else "#00E676")},
                        "steps": [
                            {"range": [0, 40], "color": "#1B5E20"},
                            {"range": [40, 80], "color": "#F57F17"},
                            {"range": [80, 100], "color": "#B71C1C"},
                        ],
                        "threshold": {"line": {"color": "white", "width": 4}, "thickness": 0.75, "value": 80},
                    },
                ))
                fig_gauge.update_layout(paper_bgcolor="#0E121B", font_color="#FFF", height=320)
                st.plotly_chart(fig_gauge, width="stretch")
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
                    st.success("Prototype policy checks passed for this simulated inspection; regulatory compliance is not established.")
                else:
                    st.error(f"**NON-COMPLIANT:** {len(res.rejection_reasons)} clause violation(s) detected")

    with tabs[5]:
        st.markdown("<h4 style='color:#FFF;'>AI Closed-Loop Root-Cause Diagnostics & Upstream Tuning Feedback</h4>", unsafe_allow_html=True)
        if res is None:
            st.info("Run a sandbox inspection to view equipment diagnostic feedback.")
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
            st.caption("Sandbox diagnostics are heuristic. PLC parameter writes are not offered on this surface.")
            if rc.get("action_items"):
                for item in rc["action_items"]:
                    with st.expander(f"{item['equipment']} → Adjust {item['parameter']} ({item['delta']})", expanded=True):
                        st.write(f"**Action Urgency:** `{item['urgency']}`")
                        st.write(f"**Engineering Rationale:** {item['rationale']}")
            else:
                st.info("No heuristic action item is available; equipment condition is NOT MEASURED here.")

    with tabs[6]:
        st.markdown("<h4 style='color:#FFF;'>Industrial Hardware, Optical Budget & P99.9 Latency Determinism</h4>", unsafe_allow_html=True)
        plc_state = api_plc_state
        displayed_line_speed = float(roll_summary.get("line_speed_m_s", 0.0))
        opt_budget = optical_budget.compute_budget(line_speed_m_s=displayed_line_speed)
        lat_budget = latency_budget.compute_budget(line_speed_m_s=displayed_line_speed)
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown(f"""
            <div class="scada-panel">
                <h5 style="color:#FFF;">Modeled Optical Budget (v = {displayed_line_speed} m/s)</h5>
                <p><b>Spatial Resolution:</b> {opt_budget['spatial_resolution_x_um_per_px']} µm/px · Coverage: {opt_budget['total_optical_coverage_mm']} mm</p>
                <p><b>Required Line Rate:</b> <b>{opt_budget['required_line_rate_khz']} kHz</b> (Period: {opt_budget['line_period_us']} µs)</p>
                <p><b>Recommended Exposure:</b> <b>{opt_budget['recommended_exposure_time_us']} µs</b> (Motion Blur: <b>{opt_budget['motion_blur_pixels']} px</b> ≤ 0.5px)</p>
                <p><b>Modeled D95 Detectability:</b> <b>{opt_budget['d95_detectability_threshold_um']} µm</b> · Raw bandwidth: {opt_budget['raw_bandwidth_mb_s']} MB/s</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="scada-panel">
                <h5 style="color:#FFF;">Modbus TCP Register Telemetry</h5>
                <p><b>Target PLC Address:</b> {plc_state['connection']['plc_ip']}:{plc_state['connection']['modbus_port']}</p>
                <p><b>Rejection Register:</b> <code>{plc_state['modbus_registers'].get('HR_1001', 0)}</code></p>
                <p><b>Connection State:</b> {plc_state['connection']['status']} (sandbox control disabled)</p>
            </div>
            """, unsafe_allow_html=True)
        with col_p2:
            st.markdown(f"""
            <div class="scada-panel">
                <h5 style="color:#FFF;">Camera-to-Ejector Deterministic Latency Budget</h5>
                <p><b>Modeled P50 Latency:</b> {lat_budget['p50_total_ms']} ms · <b>Modeled P95:</b> {lat_budget['p95_total_ms']} ms</p>
                <p><b>Modeled P99.9:</b> <b style="color:#00F2FE;">{lat_budget['p99_9_total_ms']} ms</b> (not hardware-bench verified)</p>
                <p><b>Min Required Distance:</b> {lat_budget['min_required_ejector_distance_mm']} mm → <b>Installed Distance:</b> {lat_budget['installed_ejector_distance_mm']} mm</p>
                <p><b>Theoretical Margin Ratio:</b> <b>{lat_budget['safety_margin_ratio']}x</b>; hardware verification required.</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="scada-panel">
                <h5 style="color:#FFF;">OPC UA Node Namespace Tree</h5>
                <p><code>ns=2;s=Device/Inspection/Verdict</code> → <b>{res.overall_verdict if res else 'NO INSPECTION'}</b></p>
                <p><code>ns=2;s=Device/Inspection/QualityTier</code> → <b>{res.quality_tier if res else 'NO INSPECTION'}</b></p>
                <p><code>ns=2;s=Device/Motion/LineVelocity</code> → <b>{displayed_line_speed:.2f} m/s</b></p>
            </div>
            """, unsafe_allow_html=True)

    with tabs[7]:
        st.markdown("<h4 style='color:#FFF;'>Experimental Defect Density, Spatial Periodicity & Slitting Model</h4>", unsafe_allow_html=True)
        passport_data = api_passport
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("""<div class="scada-panel"><h5 style="color:#FFF;">Per-Lane Density (Cpk/Ppk require subgroup data)</h5>""", unsafe_allow_html=True)
            spc_dict = passport_data.get("six_sigma_quality_summary", {})
            for lane_k, lane_v in spc_dict.items():
                tier_col = "#00E676" if "TIER_1" in lane_v["tier"] else ("#FFD600" if "TIER_2" in lane_v["tier"] else "#FF1744")
                st.markdown(f"""<p><b>{lane_k.upper()}:</b> Cpk = <b style="color:{tier_col};">{lane_v['cpk']}</b> · Density: <b>{lane_v['defect_density_per_100m']} def/100m</b> → <span style="color:{tier_col};"><code>{lane_v['tier']}</code></span></p>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            anomalies = passport_data.get("mechanical_health_diagnostics", [])
            st.markdown("<div class='scada-panel'><h5 style='color:#FFF;'>Spatial FFT Mechanical Diagnostics</h5>", unsafe_allow_html=True)
            if anomalies:
                for an in anomalies:
                    st.markdown(f"""<p>Periodic Pattern: Repeating every <b>{an['periodic_wavelength_mm']} mm</b> (Confidence: {an['confidence']*100:.1f}%)<br/>→ <b>Matched Component:</b> <code style="color:#00F2FE;">{an['source_component']}</code><br/>→ <b>Action:</b> {an['action']}</p>""", unsafe_allow_html=True)
            else:
                st.info("No periodic pattern is available in the current data; equipment health is not inferred.")
            st.markdown("</div>", unsafe_allow_html=True)
        with col_s2:
            slit = passport_data.get("slitting_optimization", {})
            st.markdown(f"""
            <div class="scada-panel">
                <h5 style="color:#FFF;">Smart Slitting Yield & Economic Recovery Plan</h5>
                <p><b>Modeled Usable Recovery:</b> <b style="font-size:18px;">{slit.get('recovery_yield_pct', 0.0)}%</b></p>
                <p><b>EV Grade Output Area:</b> <b style="color:#00E676;">{slit.get('ev_grade_area_m2', 0)} m²</b></p>
                <p><b>ESS Grade Output Area:</b> <b style="color:#FFD600;">{slit.get('ess_grade_area_m2', 0)} m²</b></p>
                <p><b>Scrap Area:</b> <b style="color:#FF1744;">{slit.get('scrap_area_m2', 0)} m²</b></p>
                <p><b>Recommended Splice Positions:</b> <code>{slit.get('splice_locations_md_m', [])} m</code></p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="scada-panel">
                <h5 style="color:#FFF;">Provisional Traceability Manifest (not an EU DPP)</h5>
                <p><b>Regulation:</b> <code>{passport_data.get('passport_standard')}</code></p>
                <p><b>Cleanroom Dew Point:</b> <b>{passport_data.get('traceability_genealogy',{}).get('cleanroom_dew_point_celsius')} °C</b></p>
                <p><b>Slurry Viscosity:</b> <b>{passport_data.get('traceability_genealogy',{}).get('slurry_viscosity_mpa_s')} mPa·s</b></p>
            </div>
            """, unsafe_allow_html=True)

    with tabs[8]:
        st.markdown("<h4 style='color:#FFF;'>Provisional Tamper-Evident Quality Manifest (HMAC-SHA256)</h4>", unsafe_allow_html=True)
        if api_online and api_roll_id:
            cert_payload = api_client.certificate(api_roll_id, api_batch_id)
            cert_id = cert_payload["certificate_id"]
            cert_markdown = None
            signature_status = "API-SIGNED PAYLOAD; verify with the protected factory secret"
        else:
            cert = RollCertificateGenerator.build_certificate(
                roll_id=web_sync.roll.roll_id,
                batch_id=web_sync.roll.batch_id,
                inspected_length_m=web_sync.current_pos_m,
                total_length_m=web_sync.roll.total_length_m,
                defect_records=list(web_sync.roll_defect_map),
                spc_status=spc.get("status", "UNKNOWN"),
                root_cause_summary=res.root_cause_report["primary_root_cause"] if res else "UNKNOWN",
                electrode_type=electrode_type,
                total_inspections=stats["total"] if stats.get("total") else None,
                failed_inspections=stats["failed"] if stats.get("total") else None,
                held_inspections=stats.get("held", 0) if stats.get("total") else None,
                metric_provenance="Isolated dashboard sandbox; not a production certificate",
            )
            cert_payload = cert.to_dict()
            cert_id = cert.certificate_id
            cert_markdown = cert.to_markdown()
            signature_status = "VALID IN THIS SANDBOX PROCESS" if cert.verify_signature() else "SIGNATURE MISMATCH"
        st.markdown(f"""
        <div class="scada-panel" style="border-left:4px solid #00E676;">
            <p><b>Certificate ID:</b> <code>{cert_id}</code></p>
            <p><b>Payload Digest (SHA-256):</b> <code>{cert_payload['payload_hash_sha256']}</code></p>
            <p><b>HMAC-SHA256 tag:</b> <code style="color:#00F2FE;">{cert_payload['hmac_digital_signature']}</code> (Algorithm: <code>{cert_payload['signature_algorithm']}</code>)</p>
            <p><b>Signature Status:</b> <span>{signature_status}</span></p>
            <p><b>Overall Quality Grade:</b> <span>{cert_payload['overall_quality_grade']}</span> · <b>Pass Rate:</b> {f"{float(cert_payload['pass_rate_pct']):.1f}%" if cert_payload['pass_rate_pct'] is not None else "UNVERIFIED"}</p>
        </div>
        """, unsafe_allow_html=True)
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "Download Certificate (Markdown)",
                data=cert_markdown or json.dumps(cert_payload, indent=2),
                file_name=f"{cert_id}.{'md' if cert_markdown else 'json'}",
                mime="text/markdown" if cert_markdown else "application/json",
                width="stretch",
            )
        with col_dl2:
            st.download_button(
                "Export Certificate (JSON Manifest)",
                data=json.dumps(cert_payload, indent=2),
                file_name=f"{cert_id}.json",
                mime="application/json",
                width="stretch",
            )
        with st.expander("Preview Full Certificate Document", expanded=False):
            if cert_markdown:
                st.markdown(cert_markdown)
            else:
                st.json(cert_payload)
