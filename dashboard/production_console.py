"""Production operations console: snapshot telemetry plus confirm-audit control."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
import uuid
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import streamlit as st


REQUIRED_SNAPSHOT_KEYS = (
    "schema_version",
    "snapshot_id",
    "generated_at_utc",
    "line_disposition",
    "readiness",
    "roll",
    "quality",
    "industrial",
)


def _display_or_none(value: Any, suffix: str = "") -> str:
    if value is None or value == "":
        return "NO DATA"
    return f"{value}{suffix}"


def _metric_card(label: str, value: str, border: str = "#00F2FE", extra_class: str = "") -> None:
    st.markdown(
        f"""<div class="scada-metric-card" style="border-left-color:{border};">
            <div class="metric-label">{label}</div>
            <div class="metric-val {extra_class}">{value}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_production_console(snapshot: Dict[str, Any], api_client: Any) -> None:
    missing = [key for key in REQUIRED_SNAPSHOT_KEYS if key not in snapshot]
    if missing:
        st.error(
            "Operations snapshot is incomplete; production console will not invent replacements. "
            f"Missing: {', '.join(missing)}"
        )
        st.stop()

    readiness = snapshot["readiness"]
    roll = snapshot["roll"]
    summary = roll.get("summary") or {}
    quality = snapshot["quality"]
    stats = quality.get("stats") or {}
    spc = quality.get("spc") or {}
    industrial = snapshot["industrial"]
    signals = snapshot.get("signals") or []
    if isinstance(signals, dict):
        signals = signals.get("records", [])
    disposition = snapshot["line_disposition"]
    reasons = snapshot.get("readiness_reasons") or []
    policy = snapshot.get("control_policy") or {}
    actions = policy.get("actions") or {}
    traceability = snapshot.get("traceability") or {}
    certificate = traceability.get("certificate")
    audits = (snapshot.get("control_audit") or {}).get("records") or []
    recent = (quality.get("recent_inspections") or {}).get("records") or []
    snapshot_ttl = int(policy.get("snapshot_ttl_seconds", 30))
    try:
        generated_at = datetime.fromisoformat(str(snapshot["generated_at_utc"]).replace("Z", "+00:00"))
        snapshot_age = max(0.0, (datetime.now(timezone.utc) - generated_at).total_seconds())
    except (TypeError, ValueError):
        snapshot_age = float("inf")
    snapshot_stale = snapshot_age > snapshot_ttl

    disposition_color = {
        "AUTOMATIC_DECISIONS_ALLOWED": "#1B5E20",
        "HOLD_REQUIRED": "#F57F17",
        "EMERGENCY_STOP": "#B71C1C",
    }.get(disposition, "#263238")

    st.markdown(
        f"""
<div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #222C3E; padding-bottom:12px; margin-bottom:15px;">
    <div>
        <h2 style="color:#FFFFFF; margin:0; font-weight:800; font-size:24px;">SecureCoating-Vision | Line Operations</h2>
        <span style="color:#8C9BAE; font-size:12px;">Authoritative snapshot {html.escape(str(snapshot['snapshot_id']))} at {html.escape(str(snapshot['generated_at_utc']))}</span>
    </div>
    <div>
        <span class="stage-badge stage-done">AUTHORITATIVE API</span>
        <span class="stage-badge" style="background:{disposition_color}; color:#FFFFFF;">{html.escape(str(disposition))}</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    pass_rate = stats.get("pass_rate")
    latency = stats.get("avg_latency_ms")
    spc_status = spc.get("status") or "NO DATA"
    sys_state = readiness.get("system_state") or "UNKNOWN"
    inspected = summary.get("inspected_length_m")
    total_len = summary.get("total_roll_length_m")
    line_speed = summary.get("line_speed_m_s")

    pass_class = ""
    if isinstance(pass_rate, (int, float)):
        pass_class = (
            "status-optimal" if pass_rate >= 95 else ("status-warning" if pass_rate >= 85 else "status-danger")
        )
    spc_class = (
        "status-optimal"
        if spc_status == "IN CONTROL"
        else ("status-warning" if spc_status == "WARNING" else "status-danger")
    )
    sys_class = "status-optimal" if sys_state == "OPTIMAL" else "status-warning"

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        _metric_card(
            "Line Velocity",
            "NO DATA" if line_speed is None else f"{float(line_speed):.2f} <span style='font-size:14px; color:#8C9BAE;'>m/s</span>",
        )
    with m2:
        progress = (
            "NO DATA"
            if inspected is None or total_len is None
            else f"{float(inspected):.1f} / {float(total_len):.0f}m"
        )
        _metric_card("Roll Progress", progress, "#00E676")
    with m3:
        _metric_card(
            "Resolved Pass Rate",
            "NO DATA" if pass_rate is None else f"{pass_rate}%",
            "#FFD600",
            pass_class,
        )
    with m4:
        _metric_card(
            "Mean Pipeline Latency",
            "NO DATA" if latency is None else f"{latency} <span style='font-size:14px; color:#8C9BAE;'>ms</span>",
            "#E040FB",
        )
    with m5:
        _metric_card("SPC Quality State", html.escape(str(spc_status)), "#FF1744", spc_class)
    with m6:
        _metric_card("Fail-Safe Health", html.escape(str(sys_state)), "#00E5FF", sys_class)

    st.sidebar.markdown("<h3 style='color:#FFF;'>Active identity</h3>", unsafe_allow_html=True)
    st.sidebar.code(
        f"roll: {summary.get('roll_id', 'UNKNOWN')}\n"
        f"batch: {summary.get('batch_id', 'UNKNOWN')}\n"
        f"snapshot: {snapshot['snapshot_id']}",
        language="text",
    )
    st.sidebar.caption("Identity is taken from the API snapshot. Recipe sliders are not part of production.")
    if st.sidebar.button("Reload snapshot", width="stretch"):
        st.rerun()

    operate, diagnose, traceability_tab = st.tabs(["Operate", "Diagnose", "Traceability"])

    with operate:
        st.caption(
            "Operate shows one API snapshot: line disposition, quality counts, PLC registers, "
            "and recent inspections. Control requires operator identity, a reason, and an exact confirmation phrase."
        )
        if reasons:
            st.warning("Readiness blockers: " + "; ".join(map(str, reasons)))
        else:
            st.success("No readiness blockers were reported in this snapshot.")
        if snapshot_stale:
            st.error(
                f"Snapshot is stale ({snapshot_age:.1f}s; control TTL {snapshot_ttl}s). "
                "Reload before issuing any reset command."
            )
        else:
            st.caption(f"Snapshot age: {snapshot_age:.1f}s / {snapshot_ttl}s control TTL")

        plc_col, insp_col = st.columns(2)
        with plc_col:
            connection = industrial.get("connection") or {}
            st.markdown(
                f"""
<div class="scada-panel">
    <h5 style="color:#FFF; margin-top:0;">PLC telemetry</h5>
    <p><b>Connection status:</b> {html.escape(str(connection.get('status', 'UNKNOWN')))}</p>
    <p><b>Endpoint:</b> <code>{html.escape(str(connection.get('plc_ip', 'UNKNOWN')))}:{html.escape(str(connection.get('modbus_port', 'UNKNOWN')))}</code></p>
    <p><b>Interlock latched:</b> {html.escape(str(industrial.get('interlock_latched')))}</p>
    <p><b>Interlock reason:</b> {html.escape(str(industrial.get('interlock_reason') or 'none reported'))}</p>
</div>
""",
                unsafe_allow_html=True,
            )
            st.json(industrial.get("modbus_registers") or {"status": "NO DATA"})
        with insp_col:
            st.markdown("##### Recent inspections")
            if not recent:
                st.info("No inspection rows are present for this batch in the snapshot.")
            else:
                st.dataframe(pd.DataFrame(recent), width="stretch", hide_index=True)

        st.markdown("##### Industrial signals")
        if not signals:
            st.info("No industrial signals were included in this snapshot.")
        else:
            st.dataframe(pd.DataFrame(signals), width="stretch", hide_index=True)

        st.markdown("##### Authenticated control")
        if not actions:
            st.error("Snapshot did not include a control policy; the console will not invent control actions.")
        else:
            operator_id = st.text_input("Operator ID", value="", placeholder="e.g. OP_SHIFT_A")
            operator_token = st.text_input(
                "Operator token",
                value="",
                type="password",
                help="Validated by the API; the dashboard does not derive operator identity from this token.",
            )
            action = st.selectbox("Action", list(actions.keys()))
            expected = actions[action]["confirmation"]
            reason = st.text_area("Reason (min 8 characters)", value="")
            confirmation = st.text_input(
                f"Type exactly: {expected}",
                value="",
                placeholder=expected,
            )
            control_disabled = snapshot_stale or not bool(readiness.get("control_auth_ready", True))
            if st.button("Submit control command", type="primary", disabled=control_disabled):
                try:
                    result = api_client.operations_control(
                        action=action,
                        operator_id=operator_id,
                        confirmation=confirmation,
                        reason=reason,
                        snapshot_id=snapshot["snapshot_id"],
                        idempotency_key=f"CMD_{uuid.uuid4().hex.upper()}",
                        operator_token=operator_token,
                    )
                except Exception:
                    st.error("Control request failed before an authoritative audit result was returned.")
                else:
                    if result.get("detail"):
                        st.error(result["detail"])
                    else:
                        st.json(result)
                        if result.get("mock_mode"):
                            st.warning("PLC delivery is SIMULATED in this runtime. This is not a hardware ACK.")
                        if result.get("status") in {"UNCONFIRMED", "RESET_BLOCKED"}:
                            st.error("Command was not confirmed. Treat the line as HOLD.")

    with diagnose:
        st.caption(
            "Diagnose reports readiness, sensor, model, and transport fields from the same snapshot. "
            "It does not estimate Cpk, optical budgets, or equipment health from defect density."
        )
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("##### Readiness")
            st.json(
                {
                    "ready": snapshot.get("ready"),
                    "line_disposition": disposition,
                    "readiness_reasons": reasons,
                    "simulation_mode": readiness.get("simulation_mode"),
                    "industrial_transport_ready": readiness.get("industrial_transport_ready"),
                    "traceability_ok": readiness.get("traceability_ok"),
                    "industrial_interlock_latched": readiness.get("industrial_interlock_latched"),
                }
            )
        with d2:
            st.markdown("##### Sensors and model")
            st.json(
                {
                    "sensors": readiness.get("sensors"),
                    "system_state": readiness.get("system_state"),
                    "primary_engine": readiness.get("primary_engine"),
                    "onnx_available": readiness.get("onnx_available"),
                    "yolo_available": readiness.get("yolo_available"),
                    "device": readiness.get("device"),
                }
            )
        st.markdown("##### OPC UA nodes reported by the API")
        st.json(industrial.get("opc_ua_nodes") or {"status": "NO DATA"})

    with traceability_tab:
        st.caption(
            "Traceability uses the snapshot roll ledger, batch stats, certificate payload, and control audit. "
            "Certificate root-cause text is the generator's heuristic summary, not a measured plant diagnosis."
        )
        defects = roll.get("defect_records") or []
        if not defects:
            st.info("The snapshot roll ledger contains no defect records.")
        else:
            df_roll = pd.DataFrame(defects)
            localized = df_roll.dropna(subset=["linear_pos_m", "cross_pos_mm"])
            if localized.empty:
                st.warning(
                    "Defects were retained, but no physical roll coordinates are shown because "
                    "the factory calibration is unverified. Pixel bounding boxes remain in the signed ledger."
                )
            else:
                fig_map = px.scatter(
                    localized,
                    x="linear_pos_m",
                    y="cross_pos_mm",
                    color="class_name" if "class_name" in localized.columns else None,
                    size="area_mm2" if "area_mm2" in localized.columns else None,
                    hover_data=[col for col in ("defect_id", "lane_id", "peak_height_um") if col in localized.columns],
                    title=f"Roll ledger {summary.get('roll_id', 'UNKNOWN')}",
                    labels={"linear_pos_m": "Machine direction (m)", "cross_pos_mm": "Cross-web (mm)"},
                )
                fig_map.update_layout(paper_bgcolor="#0E121B", plot_bgcolor="#131824", font_color="#FFF", height=420)
                st.plotly_chart(fig_map, width="stretch")
            st.dataframe(df_roll, width="stretch", hide_index=True)

        s1, s2, s3 = st.columns(3)
        s1.metric("Inspections", _display_or_none(stats.get("total")))
        s2.metric("PASS / REJECT / HOLD", f"{stats.get('passed', 0)} / {stats.get('failed', 0)} / {stats.get('held', 0)}")
        s3.metric("Stats status", stats.get("status") or "NO DATA")

        if traceability.get("certificate_status") != "OK" or certificate is None:
            st.error(
                "Certificate payload is unavailable in this snapshot. "
                f"{traceability.get('certificate_error') or 'No certificate was returned.'}"
            )
        else:
            pass_rate_field = _display_or_none(certificate.get("pass_rate_pct"), "%")
            st.markdown(
                f"""
<div class="scada-panel" style="border-left:4px solid #00E676;">
    <p><b>Certificate ID:</b> <code>{html.escape(str(certificate.get('certificate_id')))}</code></p>
    <p><b>Payload digest:</b> <code>{html.escape(str(certificate.get('payload_hash_sha256')))}</code></p>
    <p><b>HMAC:</b> <code>{html.escape(str(certificate.get('hmac_digital_signature')))}</code></p>
    <p><b>Quality grade:</b> {html.escape(str(certificate.get('overall_quality_grade')))} · <b>Pass rate field:</b> {html.escape(str(pass_rate_field))}</p>
    <p><b>Metric provenance:</b> {html.escape(str(certificate.get('metric_provenance')))}</p>
</div>
""",
                unsafe_allow_html=True,
            )
            st.download_button(
                "Download certificate JSON from this snapshot",
                data=json.dumps(certificate, indent=2),
                file_name=f"{certificate.get('certificate_id', snapshot['snapshot_id'])}.json",
                mime="application/json",
                width="stretch",
            )

        st.markdown("##### Control audit")
        audit_status = (snapshot.get("control_audit") or {}).get("status")
        if audit_status == "ERROR":
            st.error("Control audit could not be read from traceability storage.")
        elif not audits:
            st.info("No control-audit rows are present yet.")
        else:
            st.dataframe(pd.DataFrame(audits), width="stretch", hide_index=True)
