"""Read-only VIS + X-ray evidence lane for production operations.

Official comparable numbers require the published LIBAD release plus the
authors' splits. This view never substitutes protocol-fixture canvases as
plant evidence, and it never issues a line command.
"""

from __future__ import annotations

import html
from typing import Any

import streamlit as st


def render_multimodal_lane(api_client: Any) -> None:
    st.markdown("### External multimodal lane")
    st.caption(
        "Aligned visible-light (two faces) and inline-compatible X-ray observations "
        "for battery electrode manufacturing. This lane is independent of the "
        "optical YOLO/ONNX surface path. Protocol-fixture demos stay in the "
        "research sandbox and are not shown here."
    )
    try:
        payload = api_client.libad_protocol()
    except Exception as exc:
        st.error(
            "The multimodal protocol endpoint failed. The console will not invent "
            f"a local substitute. {exc}"
        )
        return
    if not isinstance(payload, dict):
        st.error("Multimodal protocol payload is not an object.")
        return

    dataset = payload.get("dataset") if isinstance(payload.get("dataset"), dict) else {}
    citation = payload.get("citation") if isinstance(payload.get("citation"), dict) else {}
    present = bool(dataset.get("official_dataset_present"))
    complete = bool(dataset.get("official_protocol_complete"))
    comparable = bool(dataset.get("comparable_to_paper"))
    blockers = [
        str(item)
        for item in (dataset.get("comparability_blockers") or [])
        if item
    ]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            _status_card(
                "Official release",
                "MOUNTED" if present else "NOT MOUNTED",
                "#00E676" if present else "#FFB300",
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _status_card(
                "10-seed protocol",
                "COMPLETE" if complete else "INCOMPLETE",
                "#00E676" if complete else "#FFB300",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            _status_card(
                "Paper-table comparable",
                "YES" if comparable else "NO",
                "#00E676" if comparable else "#FF1744",
            ),
            unsafe_allow_html=True,
        )

    paper_url = html.escape(str(citation.get("url") or "https://arxiv.org/abs/2608.07958"))
    dataset_url = html.escape(
        str(citation.get("dataset_url") or "https://huggingface.co/datasets/Evenrose/LIBAD")
    )
    title = html.escape(
        str(
            citation.get("title")
            or "LIBAD: A Multimodal Anomaly Detection Benchmark for Li-Ion Battery Electrode Manufacturing"
        )
    )
    st.markdown(
        f"""
<div class="scada-panel">
  <div class="metric-label">ATTRIBUTION</div>
  <p style="color:#FFFFFF; font-weight:700; margin:8px 0 6px 0;">{title}</p>
  <p style="color:#B4C0D0; font-size:13px; margin:0;">
    Sui et al., arXiv 2608.07958, CC BY 4.0.
    Paper: <a href="{paper_url}" target="_blank" rel="noreferrer">{paper_url}</a>.
    Dataset: <a href="{dataset_url}" target="_blank" rel="noreferrer">{dataset_url}</a>.
    DA-Core belongs to those authors. This runtime only adds an evidence-gated
    PASS/REJECT/HOLD decision layer.
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    note = str(payload.get("paper_result_note") or "").strip()
    if note:
        st.info(note)

    report = payload.get("local_adapter_report") if isinstance(payload.get("local_adapter_report"), dict) else None
    if report and report.get("comparable_to_paper") is not True:
        multimodal = report.get("multimodal") if isinstance(report.get("multimodal"), dict) else {}
        gate = report.get("securecoating_gate") if isinstance(report.get("securecoating_gate"), dict) else {}
        auroc = multimodal.get("auroc") if isinstance(multimodal.get("auroc"), dict) else {}
        fpr = multimodal.get("fpr95") if isinstance(multimodal.get("fpr95"), dict) else {}
        hold = gate.get("hold_rate") if isinstance(gate.get("hold_rate"), dict) else {}
        escape = gate.get("escape_rate") if isinstance(gate.get("escape_rate"), dict) else {}
        st.markdown("#### Local numpy 10-seed (not the paper table)")
        st.caption(
            "Checked-in report at reports/libad/official_local_adapter.json. "
            "CPU numpy_patch_descriptor on the hash-verified official splits. "
            "Do not quote these as DINOv3/DA-Core numbers. Mount status is the cards above."
        )
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Multimodal AUROC", _fmt_mean(auroc.get("mean")))
        with m2:
            st.metric("Multimodal FPR95", _fmt_mean(fpr.get("mean")))
        with m3:
            st.metric("Gate HOLD", _fmt_mean(hold.get("mean"), percent=True))
        with m4:
            st.metric("Gate escape", _fmt_mean(escape.get("mean"), percent=True))

    interim = (
        payload.get("authors_interim_report")
        if isinstance(payload.get("authors_interim_report"), dict)
        else None
    )
    if interim and interim.get("comparable_to_paper") is not True:
        multimodal = interim.get("multimodal") if isinstance(interim.get("multimodal"), dict) else {}
        auroc = multimodal.get("auroc") if isinstance(multimodal.get("auroc"), dict) else {}
        fpr = multimodal.get("fpr95") if isinstance(multimodal.get("fpr95"), dict) else {}
        aupr = multimodal.get("aupr") if isinstance(multimodal.get("aupr"), dict) else {}
        f1 = multimodal.get("f1_max") if isinstance(multimodal.get("f1_max"), dict) else {}
        st.markdown("#### Authors' runner interim (not paper-comparable)")
        st.caption(
            str(interim.get("note") or "")
            + " Source: reports/libad/official_dinov2_dacore_interim.json. "
            f"n_splits={interim.get('n_splits')}. "
            "DINOv3 ConvNeXt remains gated/unfinished on this host."
        )
        i1, i2, i3, i4 = st.columns(4)
        with i1:
            st.metric("Interim AUROC", _fmt_mean(auroc.get("mean")))
        with i2:
            st.metric("Interim FPR95", _fmt_mean(fpr.get("mean")))
        with i3:
            st.metric("Interim AUPR", _fmt_mean(aupr.get("mean")))
        with i4:
            st.metric("Interim F1-max", _fmt_mean(f1.get("mean")))

    if not present:
        st.warning(
            "Official VIS + X-ray files are not mounted at the configured dataset root. "
            "Place the published release under data/libad/LIBAD (with matching splits "
            "and artifact manifest) before this view can show real multimodal frames. "
            "The console will not render synthetic electrode canvases as plant evidence."
        )
        return

    if not complete:
        st.warning(
            "Official files are present, but the 10 train/val/test split seeds are not "
            "file-complete. Multimodal scoring stays HOLD until the protocol is complete."
        )
    if not comparable:
        st.caption(
            "Local scoring uses a numpy patch descriptor, not the authors' DINOv3/DA-Core "
            "implementation. Do not quote paper-table AUROC/FPR from this runtime."
        )
    if blockers:
        st.markdown("##### Comparability blockers")
        for blocker in blockers:
            st.markdown(f"- {blocker}")

    contribution = str(payload.get("local_contribution") or "").strip()
    if contribution:
        st.caption(contribution)

    try:
        samples = api_client.libad_samples(offset=0, limit=12)
    except Exception as exc:
        st.error(f"Official sample index failed. {exc}")
        return
    items = samples.get("items") if isinstance(samples, dict) else None
    if not isinstance(items, list) or not items:
        st.info(
            "The official root is mounted, but no complete VIS-A / VIS-B / X-rayL "
            "triples were indexed yet."
        )
        return

    st.markdown("#### Official mounted frames")
    st.caption(
        f"{int(samples.get('total') or 0)} complete triples on disk. "
        "These are published release files, not protocol fixtures."
    )
    names = [str(item.get("sample_id") or "") for item in items if item.get("sample_id")]
    selected_id = st.selectbox("Official sample", names, key="mm_sample")
    selected = next((item for item in items if item.get("sample_id") == selected_id), {})
    st.caption(
        f"Group {selected.get('defect_group') or 'UNKNOWN'} · "
        f"{str(selected.get('label') or 'unknown').upper()}"
    )
    image_cache = st.session_state.setdefault("multimodal_image_cache", {})
    panels = []
    for view_name, title in (
        ("vis_a", "Visible-light face A"),
        ("vis_b", "Visible-light face B"),
        ("xray_l", "Inline-compatible X-ray L"),
    ):
        cache_key = f"{selected_id}:{view_name}"
        if cache_key not in image_cache:
            try:
                image_cache[cache_key] = api_client.libad_sample_image(selected_id, view_name)
            except Exception:
                image_cache[cache_key] = None
            while len(image_cache) > 24:
                image_cache.pop(next(iter(image_cache)))
        panels.append(
            {
                "title": title,
                "chip": "OFFICIAL",
                "payload": image_cache.get(cache_key),
                "empty": "This official view is unavailable.",
            }
        )
    from dashboard.production_console import _replay_grid

    _replay_grid(panels, columns=3)


def _fmt_mean(value: Any, percent: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if percent:
        return f"{number:.1%}"
    return f"{number:.3f}"


def _status_card(label: str, value: str, border: str) -> str:
    return (
        f'<div class="scada-metric-card" style="border-left-color:{html.escape(border)};">'
        f'<div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-val">{html.escape(value)}</div></div>'
    )
