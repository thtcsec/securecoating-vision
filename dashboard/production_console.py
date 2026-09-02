"""Production operations console: snapshot telemetry plus confirm-audit control."""

from __future__ import annotations

import base64
import html
import json
from datetime import datetime, timezone
import uuid
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.multimodal_lane import render_multimodal_lane


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


def _format_kv_value(value: Any) -> str:
    if value is None or value == "":
        return "NO DATA"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dict):
        if not value:
            return "NO DATA"
        return ", ".join(f"{key}={_format_kv_value(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        if not value:
            return "NO DATA"
        return ", ".join(_format_kv_value(item) for item in value)
    return str(value)


def _kv_panel(title: str, rows: Any) -> None:
    if not isinstance(rows, dict):
        rows = {"value": rows}
    items: list[str] = []
    for key, value in rows.items():
        items.append(
            "<p><b>"
            + html.escape(str(key))
            + ":</b> <code>"
            + html.escape(_format_kv_value(value))
            + "</code></p>"
        )
    st.markdown(
        '<div class="scada-panel"><h5 style="color:#FFF; margin-top:0;">'
        + html.escape(title)
        + "</h5>"
        + "".join(items)
        + "</div>",
        unsafe_allow_html=True,
    )


def _evidence_title(title: str, chip: str) -> None:
    st.markdown(
        (
            f'<div class="evidence-title"><b>{html.escape(title)}</b>'
            f'<span class="evidence-chip">{html.escape(chip)}</span></div>'
        ),
        unsafe_allow_html=True,
    )


def _jpeg_data_uri(payload: bytes) -> str:
    return "data:image/jpeg;base64," + base64.standard_b64encode(payload).decode("ascii")


def _replay_grid(panels: Iterable[Dict[str, Any]], columns: int = 2) -> None:
    """One HTML grid so replay does not inherit Streamlit image-widget chrome."""
    panel_list = list(panels)
    count = max(1, min(int(columns), 3))
    cells: list[str] = []
    for panel in panel_list:
        title = html.escape(str(panel.get("title") or ""))
        chip = html.escape(str(panel.get("chip") or ""))
        payload = panel.get("payload")
        empty = html.escape(str(panel.get("empty") or "No evidence"))
        if isinstance(payload, (bytes, bytearray)) and payload.startswith(b"\xff\xd8\xff"):
            body = f'<img src="{_jpeg_data_uri(bytes(payload))}" alt="{title}" />'
        else:
            body = f'<div class="replay-empty">{empty}</div>'
        cells.append(
            '<div class="replay-cell">'
            f'<div class="evidence-title"><b>{title}</b>'
            f'<span class="evidence-chip">{chip}</span></div>{body}</div>'
        )
    st.markdown(
        f'<div class="replay-grid" style="grid-template-columns:repeat({count},minmax(0,1fr));">'
        f'{"".join(cells)}</div>',
        unsafe_allow_html=True,
    )


def _class_chips(labels: Optional[list[Any]], present: bool = False) -> str:
    flags = [str(label) for label in (labels or []) if label]
    if flags:
        return "".join(
            f'<span class="class-chip">{html.escape(flag.replace("_", " "))}</span>'
            for flag in flags
        )
    if present:
        return '<span class="class-chip">NO POSITIVE FLAG</span>'
    return '<span class="class-chip">NO PUBLISHED CLASS ROW</span>'


def _ai_class_chips(defect_name: Any) -> str:
    name = str(defect_name or "").strip()
    if name and name.lower() not in {"none", "null"}:
        return f'<span class="class-chip">{html.escape(name.replace("_", " "))}</span>'
    return '<span class="class-chip">NO AI DETECTION</span>'


def _mix_bars(counts: Dict[str, Any], denominator: int) -> str:
    total = max(1, int(denominator))
    rows = []
    for name, raw in counts.items():
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 0
        pct = min(100.0, round(100.0 * value / total, 1))
        rows.append(
            '<div class="mix-row">'
            f'<div>{html.escape(str(name).replace("_", " "))}</div>'
            f'<div class="mix-track"><span class="mix-fill" style="width:{pct}%;"></span></div>'
            f'<div>{value} · {pct}%</div></div>'
        )
    return "".join(rows)


def _gallery_grid(cards: Iterable[Dict[str, Any]]) -> None:
    cells: list[str] = []
    for card in cards:
        filename = html.escape(str(card.get("filename") or ""))
        selected = " selected" if card.get("selected") else ""
        payload = card.get("payload")
        badge = html.escape(str(card.get("badge") or ""))
        if isinstance(payload, (bytes, bytearray)) and payload.startswith(b"\xff\xd8\xff"):
            body = f'<img src="{_jpeg_data_uri(bytes(payload))}" alt="{filename}" />'
        else:
            body = '<div class="replay-empty">Preview unavailable</div>'
        cells.append(
            f'<div class="gallery-card{selected}">{body}'
            f'<div class="gallery-name">{filename}</div>'
            f'<div class="evidence-chip">{badge}</div></div>'
        )
    st.markdown(f'<div class="gallery-grid">{"".join(cells)}</div>', unsafe_allow_html=True)


def _snapshot_age_seconds(snapshot: Dict[str, Any]) -> float:
    try:
        generated_at = datetime.fromisoformat(str(snapshot["generated_at_utc"]).replace("Z", "+00:00"))
        return max(0.0, (datetime.now(timezone.utc) - generated_at).total_seconds())
    except (TypeError, ValueError, KeyError):
        return float("inf")


OPS_VIEWS = (
    "Guide",
    "Operate",
    "History",
    "Dataset",
    "Diagnose",
    "Traceability",
    "Multimodal",
)


def _page_count(total: int, page_size: int) -> int:
    return max(1, (max(0, int(total)) + page_size - 1) // page_size)


def render_dataset_pager(total: int, page_size: int, state_key: str = "dataset_gallery_page") -> int:
    """Prev/Next pager. Avoids a wrapping selectbox for 100+ pages."""
    pages = _page_count(total, page_size)
    page = int(st.session_state.get(state_key, 1) or 1)
    page = min(max(1, page), pages)
    st.session_state[state_key] = page

    first_col, prev_col, status_col, next_col, last_col = st.columns([1, 1, 2.4, 1, 1])
    with first_col:
        if st.button("First", disabled=page <= 1, width="stretch", key=f"{state_key}_first"):
            st.session_state[state_key] = 1
            st.rerun()
    with prev_col:
        if st.button("Previous", disabled=page <= 1, width="stretch", key=f"{state_key}_prev"):
            st.session_state[state_key] = page - 1
            st.rerun()
    with status_col:
        start = (page - 1) * page_size + 1 if total else 0
        end = min(page * page_size, total)
        st.markdown(
            f"<div class='dataset-page-status'>Page <b>{page}</b> / {pages}"
            f"<br><span>Original frames {start}–{end} of {total}</span></div>",
            unsafe_allow_html=True,
        )
    with next_col:
        if st.button("Next", disabled=page >= pages, width="stretch", key=f"{state_key}_next"):
            st.session_state[state_key] = page + 1
            st.rerun()
    with last_col:
        if st.button("Last", disabled=page >= pages, width="stretch", key=f"{state_key}_last"):
            st.session_state[state_key] = pages
            st.rerun()
    return int(st.session_state[state_key])


def render_live_strip(snapshot: Dict[str, Any]) -> None:
    """Header and six metric cards. Safe to refresh on a timer without remounting views."""
    missing = [key for key in REQUIRED_SNAPSHOT_KEYS if key not in snapshot]
    if missing:
        st.error(
            "Operations snapshot is incomplete; live telemetry will not invent replacements. "
            f"Missing: {', '.join(missing)}"
        )
        return

    readiness = snapshot["readiness"]
    summary = (snapshot["roll"] or {}).get("summary") or {}
    stats = (snapshot["quality"] or {}).get("stats") or {}
    spc = (snapshot["quality"] or {}).get("spc") or {}
    disposition = snapshot["line_disposition"]
    policy = snapshot.get("control_policy") or {}
    snapshot_ttl = int(policy.get("snapshot_ttl_seconds", 30))
    snapshot_age = _snapshot_age_seconds(snapshot)
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
        <span style="color:#8C9BAE; font-size:12px;">Authoritative snapshot {html.escape(str(snapshot['snapshot_id']))} at {html.escape(str(snapshot['generated_at_utc']))} · live strip {snapshot_age:.0f}s / {snapshot_ttl}s</span>
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
    simulation_mode = bool(readiness.get("simulation_mode"))

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
    model_ready = bool(
        readiness.get("yolo_available")
        or readiness.get("onnx_available")
        or readiness.get("trained_model_ready")
    )
    engine_name = str(readiness.get("primary_engine") or "NONE").strip() or "NONE"
    if model_ready:
        engine_label = html.escape(engine_name)
        engine_class = "status-optimal"
    else:
        engine_label = "NO MODEL"
        engine_class = "status-danger"

    m1, m2, m3 = st.columns(3)
    with m1:
        _metric_card(
            "Validated Demo Speed" if simulation_mode else "Line Velocity",
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
    m4, m5, m6 = st.columns(3)
    with m4:
        _metric_card(
            "Mean Pipeline Latency",
            "NO DATA" if latency is None else f"{latency} <span style='font-size:14px; color:#8C9BAE;'>ms</span>",
            "#E040FB",
        )
    with m5:
        _metric_card("SPC Quality State", html.escape(str(spc_status)), "#FF1744", spc_class)
    with m6:
        _metric_card("Inference Engine", engine_label, "#00E5FF", engine_class)
    st.caption(
        "Inference Engine is YOLO/ONNX load status, not a PASS authorization. "
        f"Line sensors and the safety gate remain fail-closed independently (system_state={sys_state})."
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
    catalog = snapshot.get("dataset_catalog") or {}
    snapshot_ttl = int(policy.get("snapshot_ttl_seconds", 30))
    snapshot_age = _snapshot_age_seconds(snapshot)
    snapshot_stale = snapshot_age > snapshot_ttl

    st.sidebar.markdown("<h3 style='color:#FFF;'>Active identity</h3>", unsafe_allow_html=True)
    st.sidebar.code(
        f"roll: {summary.get('roll_id', 'UNKNOWN')}\n"
        f"batch: {summary.get('batch_id', 'UNKNOWN')}\n"
        f"snapshot: {snapshot['snapshot_id']}",
        language="text",
    )
    st.sidebar.caption(
        "Identity is taken from the API snapshot. Recipe sliders are not part of production. "
        "This rail can be restored with the cyan chevron at the top-left if it is collapsed."
    )
    if st.sidebar.button("Reload snapshot", width="stretch", key="ops_reload"):
        st.session_state["ops_force_reload"] = True
        st.rerun(scope="app")

    st.markdown('<div class="view-rail-label">Views</div>', unsafe_allow_html=True)
    view = st.radio(
        "Operations view",
        options=OPS_VIEWS,
        index=2,
        key="ops_view",
        horizontal=True,
        label_visibility="collapsed",
    )
    if view is None:
        view = "Guide"
    st.caption("Views stay on this rail if the left identity panel is collapsed. The live strip above stays on every page.")

    if view == "Guide":
        st.markdown("### Get started in 2 minutes")
        st.info(
            "This runtime is fail-closed. Development simulation, a mock PLC, a localhost "
            "software loopback, or unverified calibration always keeps the line disposition "
            "at HOLD_REQUIRED. Detection evidence remains visible, but it cannot authorize "
            "a physical line decision."
        )
        g1, g2 = st.columns(2)
        with g1:
            st.markdown(
                """
#### Image path

1. **Original capture** — optical surface JPEG, retained unchanged.
2. **Published mask** — CoatingVision pixel label, shown only when the file bytes match the public set.
3. **Label heatmap** — distance transform of that published mask, not model confidence.
4. **AI overlay** — predicted regions and boxes on the same capture.
5. **Published vs AI class** — Argonne CoatingVision flags beside the local detector class; they are not fused.
6. **Safety gate** — model + sensors + calibration + database + PLC decide `PASS`, `REJECT`, or `HOLD`.

> `HOLD` means there is not enough evidence for an automatic release. It is not the same as “no defect found”.
"""
            )
        with g2:
            st.markdown(
                """
#### Which view to open

- **Operate:** line disposition, PLC telemetry, confirm-audit control.
- **History:** original → published mask → label heatmap → AI overlay, then published class vs AI class.
- **Dataset:** original CoatingVision JPEGs, published pixel masks, and published classification flags.
- **Diagnose:** why the snapshot is `DEGRADED` or `HOLD`.
- **Traceability:** certificate, roll/batch identity, and control audit.
- **Multimodal:** VIS + X-ray evidence lane. Official electrode release only; this console does not substitute a protocol fixture.

The left identity rail is not the view switcher. Collapsing it hides roll/batch identity only.
"""
            )
        st.markdown("#### Create a live inspection")
        st.code(
            "docker exec coating_inspection_api python scripts/smoke_live.py "
            "--api-url http://localhost:8000",
            language="powershell",
        )
        st.caption(
            "Uses a packed CoatingVision optical frame, writes a new inspection, and checks "
            "retained JPEG artifacts. Then open History; the snapshot refreshes about every 5 seconds."
        )
        st.markdown("#### What the dataset is")
        st.markdown(
            "Frames in this app are **acquired optical surface images** from "
            "[CoatingVision](https://doi.org/10.6084/m9.figshare.29260121.v1) "
            "(Sampath et al., Argonne; CC BY 4.0) — close-up coating JPEGs, not "
            "synthetic renders and not factory-floor photographs. Open **Dataset** "
            "to page through the original files. If the local Figshare archive is "
            "mounted (`data/external/coatingvision`, ~2227 JPEGs) the library uses "
            "that; otherwise it falls back to the 88-frame git test split."
        )

    elif view == "Operate":
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
            evidence_class = connection.get("evidence_class") or "unspecified"
            st.markdown(
                f"""
<div class="scada-panel">
    <h5 style="color:#FFF; margin-top:0;">PLC telemetry</h5>
    <p><b>Connection status:</b> {html.escape(str(connection.get('status', 'UNKNOWN')))}</p>
    <p><b>Command channel:</b> {html.escape(str(connection.get('command_channel') or 'unspecified'))}</p>
    <p><b>Evidence class:</b> {html.escape(str(evidence_class))}</p>
    <p><b>Endpoint:</b> <code>{html.escape(str(connection.get('plc_ip', 'UNKNOWN')))}:{html.escape(str(connection.get('modbus_port', 'UNKNOWN')))}</code></p>
    <p><b>Interlock latched:</b> {html.escape(str(industrial.get('interlock_latched')))}</p>
    <p><b>Interlock reason:</b> {html.escape(str(industrial.get('interlock_reason') or 'none reported'))}</p>
</div>
""",
                unsafe_allow_html=True,
            )
            if (
                str(connection.get("status")) == "SOFTWARE_LOOPBACK"
                or evidence_class == "software_loopback_not_vendor_hil"
            ):
                st.warning(
                    "PLC ACK is a localhost software loopback. This is not vendor hardware-in-the-loop."
                )
            elif connection.get("mock_mode"):
                st.warning("PLC delivery is SIMULATED in this runtime. This is not a hardware ACK.")
            _kv_panel(
                "Modbus registers",
                industrial.get("modbus_registers") or {"status": "NO DATA"},
            )
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
            operator_id = st.text_input(
                "Operator ID",
                placeholder="e.g. OP_SHIFT_A",
                key="ops_operator_id",
            )
            operator_token = st.text_input(
                "Operator token",
                type="password",
                help="Validated by the API; the dashboard does not derive operator identity from this token.",
                key="ops_operator_token",
            )
            action = st.selectbox("Action", list(actions.keys()), key="ops_action")
            expected = actions[action]["confirmation"]
            reason = st.text_area("Reason (min 8 characters)", key="ops_reason")
            confirmation = st.text_input(
                f"Type exactly: {expected}",
                placeholder=expected,
                key="ops_confirmation",
            )
            control_disabled = snapshot_stale or not bool(readiness.get("control_auth_ready", True))
            if st.button("Submit control command", type="primary", disabled=control_disabled, key="ops_submit"):
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
                        _kv_panel("Control audit result", result)
                        if result.get("mock_mode"):
                            st.warning("PLC delivery is SIMULATED in this runtime. This is not a hardware ACK.")
                        elif str((industrial.get("connection") or {}).get("status")) == "SOFTWARE_LOOPBACK":
                            st.warning(
                                "Control ACK is a localhost software loopback. This is not vendor HIL."
                            )
                        if result.get("status") in {"UNCONFIRMED", "RESET_BLOCKED"}:
                            st.error("Command was not confirmed. Treat the line as HOLD.")

    elif view == "History":
        st.caption(
            "Persisted inspection history is authoritative for the active batch. "
            "Published-label views appear only when the acquired bytes match a CoatingVision file. "
            "The event table below is a reconstruction from persisted fields, not raw PLC logs."
        )
        st.markdown("### Visual inspection replay")
        st.markdown(
            """
<div class="vision-flow">
  <div class="vision-step"><div class="vision-step-no">01 · REAL CAPTURE</div><div class="vision-step-title">Original optical JPEG</div><div class="vision-step-copy">Unmodified CoatingVision or camera frame</div></div>
  <div class="vision-step"><div class="vision-step-no">02 · PUBLISHED LABEL</div><div class="vision-step-title">Pixel mask</div><div class="vision-step-copy">Official segmentation geometry, not AI</div></div>
  <div class="vision-step"><div class="vision-step-no">03 · LABEL HEATMAP</div><div class="vision-step-title">Mask intensity</div><div class="vision-step-copy">Distance transform of the published mask</div></div>
  <div class="vision-step"><div class="vision-step-no">04 · AI INFERENCE</div><div class="vision-step-title">Predicted regions</div><div class="vision-step-copy">Model overlay on the same capture</div></div>
</div>
""",
            unsafe_allow_html=True,
        )
        if not recent:
            st.info("No persisted inspection is available for the active batch.")
        else:
            options = {
                (
                    f"{row.get('part_id')} · {row.get('gate_action')} · "
                    f"{str(row.get('run_id') or 'NO RUN ID')}"
                ): row
                for row in recent
            }
            selected_label = st.selectbox("Inspection run", list(options.keys()), key="history_run")
            selected = options[selected_label]
            run_id = selected.get("run_id")
            try:
                selected_latency = f"{float(selected.get('latency_ms')):.2f}"
            except (TypeError, ValueError):
                selected_latency = "NO DATA"
            selected_defects = [
                defect
                for defect in (roll.get("defect_records") or [])
                if defect.get("run_id") == run_id
            ]
            top_defect = max(
                selected_defects,
                key=lambda item: float(item.get("confidence") or 0.0),
                default={},
            )
            defect_name = top_defect.get("class_name") or selected.get("defect_class") or "none"
            published_flags = selected.get("published_classes")
            has_published_row = isinstance(published_flags, list)
            if selected.get("image_available") and run_id:
                image_cache = st.session_state.setdefault("inspection_image_cache", {})
                artifacts = selected.get("artifacts") or {
                    "overlay": {"available": True}
                }
                replay_views = ("raw", "mask", "heatmap", "overlay")
                missing = [
                    view
                    for view in replay_views
                    if bool((artifacts.get(view) or {}).get("available"))
                    and f"{run_id}:{view}" not in image_cache
                ]
                if missing:
                    try:
                        fetched = api_client.inspection_images(run_id, missing)
                    except Exception:
                        fetched = {view: None for view in missing}
                    for view, payload in fetched.items():
                        image_cache[f"{run_id}:{view}"] = payload
                        while len(image_cache) > 40:
                            image_cache.pop(next(iter(image_cache)))

                def _payload(view_name: str):
                    available = bool((artifacts.get(view_name) or {}).get("available"))
                    if not available:
                        return None
                    return image_cache.get(f"{run_id}:{view_name}")

                _replay_grid(
                    [
                        {
                            "title": "1. Original capture",
                            "chip": "REAL JPEG",
                            "payload": _payload("raw"),
                            "empty": "Original frame was not retained for this historical run.",
                        },
                        {
                            "title": "2. Published mask",
                            "chip": "PIXEL LABEL",
                            "payload": _payload("mask"),
                            "empty": "No published pixel mask is linked to this acquired frame.",
                        },
                        {
                            "title": "3. Label heatmap",
                            "chip": "PUBLISHED",
                            "payload": _payload("heatmap"),
                            "empty": "Label heatmap is unavailable unless the acquired bytes match CoatingVision.",
                        },
                        {
                            "title": "4. AI overlay",
                            "chip": "MODEL",
                            "payload": _payload("overlay"),
                            "empty": "Detection overlay was not retained for this run.",
                        },
                    ]
                )
                st.caption(
                    "Heatmap is a distance transform of the published CoatingVision mask. "
                    "It is not a model-confidence map. AI overlay is the independent prediction."
                )
                if st.checkbox("Show letterboxed model input", key=f"ops_show_input_{run_id}"):
                    input_key = f"{run_id}:input"
                    if bool((artifacts.get("input") or {}).get("available")) and input_key not in image_cache:
                        try:
                            fetched_input = api_client.inspection_images(run_id, ["input"])
                            image_cache[input_key] = fetched_input.get("input")
                        except Exception:
                            image_cache[input_key] = None
                        while len(image_cache) > 40:
                            image_cache.pop(next(iter(image_cache)))
                    input_payload = image_cache.get(input_key)
                    if input_payload:
                        st.markdown(
                            f'<img src="{_jpeg_data_uri(input_payload)}" alt="Model input" '
                            f'style="max-width:360px;border-radius:8px;border:1px solid #26344A;" />',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("Model input was not retained for this run.")

                confidence = top_defect.get("confidence", selected.get("peak_confidence"))
                confidence_text = (
                    f"{float(confidence) * 100:.1f}%" if isinstance(confidence, (int, float)) else "NO DATA"
                )
                gate_action = html.escape(str(selected.get("gate_action") or "UNKNOWN"))
                verdict_color = "#00E676" if gate_action == "PASS" else ("#FF1744" if gate_action == "REJECT" else "#FFB300")
                sample_note = (
                    f" · sample {html.escape(str(selected.get('sample_name')))}"
                    if selected.get("sample_name")
                    else ""
                )
                st.markdown(
                    f"""
<div class="ai-verdict" style="border-left-color:{verdict_color};">
  <div class="ai-verdict-label">AI RESULT → SAFETY GATE</div>
  <div class="ai-verdict-main">{html.escape(str(defect_name)).upper()} · {confidence_text} → {gate_action}</div>
  <div class="ai-verdict-detail">Model {html.escape(str(selected.get('model_version') or 'UNKNOWN'))} · inference {selected_latency} ms · run {html.escape(str(run_id))}{sample_note}. Detection is visible even when the independent safety gate holds the line.</div>
</div>
""",
                    unsafe_allow_html=True,
                )
            else:
                st.info("No retained inspection image bundle is available for this historical row.")

            st.markdown(
                '<div class="metric-label">PUBLISHED COATINGVISION CLASS</div>'
                + _class_chips(published_flags if has_published_row else None, has_published_row)
                + '<div class="metric-label" style="margin-top:10px;">AI DETECTOR CLASS</div>'
                + _ai_class_chips(defect_name),
                unsafe_allow_html=True,
            )
            st.caption(
                "Published flags are Argonne CoatingVision classification labels. "
                "AI class names are the local detector map. They are shown side by side, not fused."
            )

            if selected_defects:
                st.markdown("##### Matched detections")
                detection_columns = [
                    "defect_id", "class_name", "confidence", "bbox",
                    "area_mm2", "length_mm", "width_mm", "coordinate_verified",
                ]
                st.dataframe(
                    pd.DataFrame(selected_defects)[
                        [column for column in detection_columns if column in selected_defects[0]]
                    ],
                    width="stretch",
                    hide_index=True,
                )

            event_rows = [
                {
                    "event": "INSPECTION_PERSISTED",
                    "status": "OK",
                    "detail": f"part={selected.get('part_id')} run={run_id}",
                },
                {
                    "event": "INFERENCE_COMPLETED",
                    "status": selected.get("system_state") or "UNKNOWN",
                    "detail": (
                        f"model={selected.get('model_version')} latency={selected_latency}ms "
                        f"defect={selected.get('defect_class')}"
                    ),
                },
                {
                    "event": "SAFETY_VALIDATION",
                    "status": "VALID" if selected.get("inspection_valid") else "FAIL_CLOSED",
                    "detail": selected.get("error_reason") or "No safety blocker persisted",
                },
                {
                    "event": "GATE_DISPOSITION",
                    "status": selected.get("gate_action") or "UNKNOWN",
                    "detail": f"system_state={selected.get('system_state')}",
                },
            ]
            st.markdown("##### Decision event log")
            st.dataframe(pd.DataFrame(event_rows), width="stretch", hide_index=True)

    elif view == "Dataset":
        st.markdown("### Dataset Library")
        library_scope = catalog.get("library_scope") or "checked_in_test_split"
        source_type = "VERIFIED" if catalog.get("provenance_verified") else "UNVERIFIED"
        doi_url = catalog.get("doi_url") or "https://doi.org/10.6084/m9.figshare.29260121.v1"
        archive_count = int(catalog.get("archive_count") or 0)
        mask_count = int(catalog.get("segmentation_mask_count") or 0)
        class_count = int(catalog.get("classification_label_count") or 0)
        checked_in = int(catalog.get("checked_in_count") or 0)
        parent_archive = int(catalog.get("parent_archive_images") or 2227)
        if library_scope == "local_figshare_archive":
            scope_copy = (
                f"This machine has the **local Figshare detection archive** "
                f"({archive_count or parent_archive} unmodified JPEGs). "
                f"Git also hashes the {checked_in or 88}-frame held-out test split."
            )
        else:
            scope_copy = (
                f"Full Figshare archive is **not mounted**. Showing the checked-in "
                f"test split ({int(catalog.get('total') or 0)} original JPEGs). "
                f"Parent public set is {parent_archive} images; 581 labeled pairs are train/eval only."
            )
        st.info(
            "These are **original CoatingVision optical surface frames** (Sampath et al., "
            f"Argonne National Laboratory, [Figshare DOI]({doi_url}), CC BY 4.0) — "
            "close-up coating photographs, not synthetic canvases and not line overviews. "
            + scope_copy
        )
        d1, d2, d3 = st.columns(3)
        with d1:
            _metric_card("Original frames", str(int(catalog.get("total") or 0)))
        with d2:
            _metric_card("Figshare archive", str(archive_count or parent_archive), "#00E676")
        with d3:
            _metric_card("Hashed test split", str(checked_in or int(catalog.get("total") or 0)), "#E040FB")
        d4, d5, d6 = st.columns(3)
        with d4:
            _metric_card(
                "Published masks",
                str(mask_count) if mask_count else "NOT MOUNTED",
                "#FFB300",
            )
        with d5:
            _metric_card(
                "Published class rows",
                str(class_count) if class_count else "NOT MOUNTED",
                "#7C4DFF",
            )
        with d6:
            _metric_card("Provenance", source_type, "#00E5FF")
        st.markdown(f"**Dataset:** {catalog.get('dataset_name') or 'Unknown dataset'}")
        st.caption(str(catalog.get("dataset_source") or doi_url))

        mix_counts = catalog.get("classification_flag_counts") or {}
        mix_total = int(catalog.get("total") or 0)
        if mix_counts and mix_total:
            st.markdown("#### Published class mix")
            st.markdown(_mix_bars(mix_counts, mix_total), unsafe_allow_html=True)
            st.caption(
                f"Multi-label flags can sum above 100%. "
                f"{int(catalog.get('classification_multi_label_count') or 0)} frames have more than one flag; "
                f"{int(catalog.get('classification_empty_positive_count') or 0)} labeled rows have no positive flag. "
                "The detector class map is independent (surface_crack / delamination_crack on 581 box-labeled pairs)."
            )

        filter_labels = {
            "All published flags": None,
            "Surface Crack": "Surface_Crack",
            "Delamination": "Delamination",
            "Pinhole": "Pinhole",
            "unclassified": "unclassified",
            "No positive flag": "none",
        }
        selected_filter = st.selectbox(
            "Show frames with",
            list(filter_labels.keys()),
            key="dataset_class_filter",
        )
        class_flag = filter_labels[selected_filter]
        if st.session_state.get("dataset_class_filter_applied") != class_flag:
            st.session_state["dataset_class_filter_applied"] = class_flag
            st.session_state.pop("dataset_gallery_catalog_cache", None)
            st.session_state["dataset_page"] = 1

        gallery_page_size = 12
        gallery_catalog_cache = st.session_state.setdefault("dataset_gallery_catalog_cache", {})
        if 1 not in gallery_catalog_cache:
            try:
                gallery_catalog_cache[1] = api_client.dataset_catalog(
                    offset=0,
                    limit=gallery_page_size,
                    class_flag=class_flag,
                )
            except Exception:
                gallery_catalog_cache[1] = None
        probe_catalog = gallery_catalog_cache.get(1) or {}
        gallery_total = int(probe_catalog.get("total") or 0)
        gallery_page = render_dataset_pager(gallery_total, gallery_page_size)
        if gallery_page not in gallery_catalog_cache:
            try:
                gallery_catalog_cache[gallery_page] = api_client.dataset_catalog(
                    offset=(gallery_page - 1) * gallery_page_size,
                    limit=gallery_page_size,
                    class_flag=class_flag,
                )
            except Exception:
                gallery_catalog_cache[gallery_page] = None
            while len(gallery_catalog_cache) > 12:
                gallery_catalog_cache.pop(next(iter(gallery_catalog_cache)))
        gallery_catalog = gallery_catalog_cache.get(gallery_page) or probe_catalog

        gallery_items = (gallery_catalog.get("items") or [])[:gallery_page_size]
        dataset_image_cache = st.session_state.setdefault("dataset_image_cache", {})
        if not gallery_items:
            st.info("No dataset images are available on this page.")
        else:
            focus_options = [str(item.get("filename") or "") for item in gallery_items if item.get("filename")]
            default_focus = st.session_state.get("dataset_focus_filename")
            if default_focus not in focus_options:
                default_focus = focus_options[0]
            focus_name = st.selectbox(
                "Focus frame",
                focus_options,
                index=focus_options.index(default_focus),
                key="dataset_focus_select",
            )
            st.session_state["dataset_focus_filename"] = focus_name
            focus_item = next((item for item in gallery_items if item.get("filename") == focus_name), None)
            focus_evidence_cache = st.session_state.setdefault(
                "dataset_focus_evidence_cache", {}
            )
            if focus_name:
                needed_views = []
                if focus_name not in dataset_image_cache:
                    needed_views.append("original")
                for evidence_view in ("mask", "heatmap"):
                    if f"{focus_name}:{evidence_view}" not in focus_evidence_cache:
                        needed_views.append(evidence_view)
                if needed_views:
                    try:
                        fetched = api_client.dataset_evidence(focus_name, needed_views)
                    except Exception:
                        fetched = {view: None for view in needed_views}
                    if "original" in fetched:
                        dataset_image_cache[focus_name] = fetched.get("original")
                    for evidence_view in ("mask", "heatmap"):
                        if evidence_view in fetched:
                            focus_evidence_cache[f"{focus_name}:{evidence_view}"] = fetched.get(
                                evidence_view
                            )
                            while len(focus_evidence_cache) > 24:
                                focus_evidence_cache.pop(next(iter(focus_evidence_cache)))
            if focus_item and dataset_image_cache.get(focus_name):
                st.markdown("#### Pixel-level evidence explorer")
                _replay_grid(
                    [
                        {
                            "title": "1. Original optical JPEG",
                            "chip": "SOURCE",
                            "payload": dataset_image_cache[focus_name],
                            "empty": "Original JPEG is unavailable.",
                        },
                        {
                            "title": "2. Published pixel mask",
                            "chip": "LABEL",
                            "payload": focus_evidence_cache.get(f"{focus_name}:mask"),
                            "empty": "No segmentation label for this frame.",
                        },
                        {
                            "title": "3. Label heatmap",
                            "chip": "PUBLISHED",
                            "payload": focus_evidence_cache.get(f"{focus_name}:heatmap"),
                            "empty": "No published heatmap for this frame.",
                        },
                    ],
                    columns=3,
                )
                st.markdown(
                    '<div class="metric-label">PUBLISHED CLASSIFICATION</div>'
                    + _class_chips(
                        focus_item.get("published_classes"),
                        bool(focus_item.get("has_published_classes")),
                    ),
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Heatmap and class flags come from the official CoatingVision labels, not from the detector."
                )

            gallery_cards = []
            missing_thumbs = [
                str(item.get("filename") or "")
                for item in gallery_items
                if item.get("filename")
                and f"{item.get('filename')}:thumb" not in dataset_image_cache
            ]
            if missing_thumbs:
                try:
                    fetched_thumbs = api_client.dataset_images(missing_thumbs, "thumb")
                except Exception:
                    fetched_thumbs = {name: None for name in missing_thumbs}
                for filename, payload in fetched_thumbs.items():
                    dataset_image_cache[f"{filename}:thumb"] = (
                        payload if payload else dataset_image_cache.get(filename)
                    )
                    while len(dataset_image_cache) > 48:
                        dataset_image_cache.pop(next(iter(dataset_image_cache)))
            for item in gallery_items:
                filename = str(item.get("filename") or "")
                thumb_key = f"{filename}:thumb"
                if item.get("hash_verified"):
                    badge = "HASH VERIFIED"
                elif item.get("in_checked_in_split"):
                    badge = "TEST SPLIT"
                else:
                    badge = "FIGSHARE"
                gallery_cards.append(
                    {
                        "filename": filename,
                        "payload": dataset_image_cache.get(thumb_key) or dataset_image_cache.get(filename),
                        "selected": filename == focus_name,
                        "badge": badge,
                    }
                )
            st.markdown("#### Library page")
            _gallery_grid(gallery_cards)

    elif view == "Diagnose":
        st.caption(
            "Diagnose reports readiness, sensor, model, and transport fields from the same snapshot. "
            "It does not estimate Cpk, optical budgets, or equipment health from defect density."
        )
        d1, d2 = st.columns(2)
        with d1:
            _kv_panel(
                "Readiness",
                {
                    "ready": snapshot.get("ready"),
                    "line_disposition": disposition,
                    "readiness_reasons": reasons,
                    "simulation_mode": readiness.get("simulation_mode"),
                    "calibration_verified": readiness.get("calibration_verified"),
                    "industrial_mock_mode": readiness.get("industrial_mock_mode"),
                    "industrial_transport_ready": readiness.get("industrial_transport_ready"),
                    "traceability_ok": readiness.get("traceability_ok"),
                    "industrial_interlock_latched": readiness.get("industrial_interlock_latched"),
                },
            )
        with d2:
            _kv_panel(
                "Sensors and model",
                {
                    "sensors": readiness.get("sensors"),
                    "system_state": readiness.get("system_state"),
                    "primary_engine": readiness.get("primary_engine"),
                    "onnx_available": readiness.get("onnx_available"),
                    "yolo_available": readiness.get("yolo_available"),
                    "device": readiness.get("device"),
                    "model_artifact_sha256": readiness.get("model_artifact_sha256"),
                    "onnx_artifact_sha256": readiness.get("onnx_artifact_sha256"),
                },
            )
        _kv_panel("Throughput evidence", snapshot.get("throughput") or {"status": "NO DATA"})
        _kv_panel(
            "OPC UA nodes reported by the API",
            industrial.get("opc_ua_nodes") or {"status": "NO DATA"},
        )

    elif view == "Traceability":
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
                key="ops_cert_download",
            )

        st.markdown("##### Control audit")
        audit_status = (snapshot.get("control_audit") or {}).get("status")
        if audit_status == "ERROR":
            st.error("Control audit could not be read from traceability storage.")
        elif not audits:
            st.info("No control-audit rows are present yet.")
        else:
            st.dataframe(pd.DataFrame(audits), width="stretch", hide_index=True)

    elif view == "Multimodal":
        render_multimodal_lane(api_client)
