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


def _snapshot_age_seconds(snapshot: Dict[str, Any]) -> float:
    try:
        generated_at = datetime.fromisoformat(str(snapshot["generated_at_utc"]).replace("Z", "+00:00"))
        return max(0.0, (datetime.now(timezone.utc) - generated_at).total_seconds())
    except (TypeError, ValueError, KeyError):
        return float("inf")


OPS_VIEWS = ("Guide", "Operate", "History", "Dataset", "Diagnose", "Traceability")


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

    m1, m2, m3 = st.columns(3)
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
        _metric_card("Fail-Safe Health", html.escape(str(sys_state)), "#00E5FF", sys_class)


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
    st.sidebar.caption("Identity is taken from the API snapshot. Recipe sliders are not part of production.")
    if st.sidebar.button("Reload snapshot", width="stretch", key="ops_reload"):
        st.session_state["ops_force_reload"] = True
        st.rerun(scope="app")

    st.sidebar.markdown("<h3 style='color:#FFF; margin-top:18px;'>Views</h3>", unsafe_allow_html=True)
    view = st.sidebar.radio(
        "Operations view",
        options=OPS_VIEWS,
        key="ops_view",
        label_visibility="collapsed",
    )
    if view is None:
        view = "Guide"
    st.sidebar.caption("One view at a time. The live strip above stays on every page.")

    if view == "Guide":
        st.markdown("### Get started in 2 minutes")
        st.info(
            "This runtime is fail-closed. Without a real PLC, plant sensors, and verified "
            "calibration, the gate stays HOLD and health stays DEGRADED. That is a safe "
            "state, not a frozen app."
        )
        g1, g2 = st.columns(2)
        with g1:
            st.markdown(
                """
#### Image path

1. **Acquired frame** — optical surface frame from camera or upload.
2. **Model input** — resized/letterboxed geometry the model actually saw.
3. **Detection result** — predicted boxes and confidences on the acquired frame.
4. **Safety gate** — model + sensors + calibration + database + PLC decide `PASS`, `REJECT`, or `HOLD`.

> `HOLD` means there is not enough evidence for an automatic release. It is not the same as “no defect found”.
"""
            )
        with g2:
            st.markdown(
                """
#### Which view to open

- **Operate:** line disposition, PLC telemetry, confirm-audit control.
- **History:** one run with acquired / model-input / overlay plus the decision log.
- **Dataset:** original CoatingVision JPEGs with First / Previous / Next / Last paging, DOI, license, and SHA-256 for the git test split.
- **Diagnose:** why the snapshot is `DEGRADED` or `HOLD`.
- **Traceability:** certificate, roll/batch identity, and control audit.
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
            "all three JPEG artifacts. Then open History; the snapshot refreshes about every 5 seconds."
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
                        st.json(result)
                        if result.get("mock_mode"):
                            st.warning("PLC delivery is SIMULATED in this runtime. This is not a hardware ACK.")
                        if result.get("status") in {"UNCONFIRMED", "RESET_BLOCKED"}:
                            st.error("Command was not confirmed. Treat the line as HOLD.")

    elif view == "History":
        st.caption(
            "Persisted inspection history is authoritative for the active batch. Bounded acquired, "
            "model-input, and overlay previews are loaded on demand; the event table below is a "
            "reconstruction from persisted fields, not raw PLC logs."
        )
        st.markdown("##### Inspection result")
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
            if selected.get("image_available") and run_id:
                image_cache = st.session_state.setdefault("inspection_image_cache", {})
                artifacts = selected.get("artifacts") or {
                    "overlay": {"available": True}
                }

                def _load_artifact(artifact_view: str):
                    available = bool((artifacts.get(artifact_view) or {}).get("available"))
                    cache_key = f"{run_id}:{artifact_view}"
                    if available and cache_key not in image_cache:
                        try:
                            image_cache[cache_key] = api_client.inspection_image(run_id, artifact_view)
                        except Exception:
                            image_cache[cache_key] = None
                        while len(image_cache) > 24:
                            image_cache.pop(next(iter(image_cache)))
                    return available, image_cache.get(cache_key)

                hero, side = st.columns([2, 1])
                with hero:
                    st.markdown("**Original acquired frame**")
                    available, payload = _load_artifact("raw")
                    if available and payload:
                        st.image(
                            payload,
                            caption="Unmodified optical JPEG retained for this run — not a synthetic canvas",
                            width="stretch",
                        )
                    elif available:
                        st.error("The API reported acquired-frame evidence, but fetch failed.")
                    else:
                        st.info("Original frame was not retained for this historical run.")
                with side:
                    for artifact_view, title, caption in (
                        ("input", "Model input", "Letterboxed geometry the detector saw"),
                        ("overlay", "Detection overlay", "Predicted boxes on the acquired frame"),
                    ):
                        st.markdown(f"**{title}**")
                        available, payload = _load_artifact(artifact_view)
                        if available and payload:
                            st.image(payload, caption=caption, width="stretch")
                        elif available:
                            st.error(f"The API reported {artifact_view} evidence, but fetch failed.")
                        else:
                            st.info("Not retained for this historical run.")
                st.caption(
                    f"{selected.get('part_id')} · {selected.get('gate_action')} · "
                    f"{selected_latency} ms · original / input / overlay are bounded JPEG evidence"
                )
            else:
                st.info("No retained inspection image bundle is available for this historical row.")

            selected_defects = [
                defect
                for defect in (roll.get("defect_records") or [])
                if defect.get("run_id") == run_id
            ]
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
        source_type = "VERIFIED REAL OPTICAL" if catalog.get("provenance_verified") else "UNVERIFIED"
        doi_url = catalog.get("doi_url") or "https://doi.org/10.6084/m9.figshare.29260121.v1"
        archive_count = int(catalog.get("archive_count") or 0)
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
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.metric("Original frames here", int(catalog.get("total") or 0))
        with d2:
            st.metric("Figshare archive", archive_count or parent_archive)
        with d3:
            st.metric("Hashed test split", checked_in or int(catalog.get("total") or 0))
        with d4:
            st.metric("Provenance", source_type)
        st.markdown(f"**Dataset:** {catalog.get('dataset_name') or 'Unknown dataset'}")
        st.caption(str(catalog.get("dataset_source") or doi_url))

        gallery_page_size = 12
        gallery_total = int(catalog.get("total") or 0)
        gallery_page = render_dataset_pager(gallery_total, gallery_page_size)
        gallery_catalog_cache = st.session_state.setdefault("dataset_gallery_catalog_cache", {})
        if gallery_page not in gallery_catalog_cache:
            try:
                gallery_catalog_cache[gallery_page] = api_client.dataset_catalog(
                    offset=(gallery_page - 1) * gallery_page_size,
                    limit=gallery_page_size,
                )
            except Exception:
                gallery_catalog_cache[gallery_page] = None
            while len(gallery_catalog_cache) > 12:
                gallery_catalog_cache.pop(next(iter(gallery_catalog_cache)))
        gallery_catalog = gallery_catalog_cache.get(gallery_page) or {}

        gallery_items = (gallery_catalog.get("items") or [])[:gallery_page_size]
        dataset_image_cache = st.session_state.setdefault("dataset_image_cache", {})
        if not gallery_items:
            st.info("No dataset images are available on this page.")
        else:
            focus_name = st.session_state.get("dataset_focus_filename")
            focus_item = next((item for item in gallery_items if item.get("filename") == focus_name), None)
            if focus_item and dataset_image_cache.get(focus_name):
                st.markdown("#### Original JPEG (unmodified)")
                st.image(
                    dataset_image_cache[focus_name],
                    caption=(
                        f"{focus_name} · {focus_item.get('source_id')} · "
                        "this is the source optical frame, not a model overlay"
                    ),
                    width="stretch",
                )

            for row_start in range(0, len(gallery_items), 4):
                gallery_columns = st.columns(4)
                for column, item in zip(gallery_columns, gallery_items[row_start:row_start + 4]):
                    filename = str(item.get("filename") or "")
                    with column:
                        if filename and filename not in dataset_image_cache:
                            try:
                                dataset_image_cache[filename] = api_client.dataset_image(filename)
                            except Exception:
                                dataset_image_cache[filename] = None
                            while len(dataset_image_cache) > 36:
                                dataset_image_cache.pop(next(iter(dataset_image_cache)))
                        if dataset_image_cache.get(filename):
                            st.image(
                                dataset_image_cache[filename],
                                caption=filename,
                                width="stretch",
                            )
                        else:
                            st.error(f"Preview unavailable: {filename or 'unknown image'}")
                        if st.button("Open original", key=f"dataset_focus_{filename}", width="stretch"):
                            st.session_state["dataset_focus_filename"] = filename
                            st.rerun()
                        if item.get("hash_verified"):
                            badge = "HASH VERIFIED · test split"
                        elif item.get("in_checked_in_split"):
                            badge = "TEST SPLIT · UNVERIFIED"
                        else:
                            badge = "FIGSHARE ORIGINAL"
                        st.caption(
                            f"{badge} · {item.get('source_id') or 'source id missing'}"
                        )
            st.markdown("#### Page records")
            st.dataframe(pd.DataFrame(gallery_items), width="stretch", hide_index=True)

    elif view == "Diagnose":
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
