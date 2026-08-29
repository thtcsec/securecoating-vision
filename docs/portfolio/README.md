# SecureCoating-Vision portfolio media

## Hero visual

`securecoating-vision-hero.png` is an AI-generated conceptual render for portfolio presentation. It is not a photograph of a deployed factory line and must be labelled accordingly.

Suggested caption:

> Conceptual visualization of a fail-closed computer-vision inspection cell for roll-to-roll electrode coating. The implemented prototype combines FastAPI inference, PLC-oriented interlocks, signed traceability, and an authoritative operations dashboard.

## Runtime screenshots

For evidence screenshots, use the live Docker dashboard at `http://127.0.0.1:8501/` and keep the following context visible:

- **Operate:** `HOLD_REQUIRED`, readiness blockers, PLC transport state, and the recent-inspection row.
- **Diagnose:** authoritative readiness plus sensor/model state; do not crop out `DEGRADED` or offline hardware.
- **Traceability:** the four retained defect records, calibration warning, `UNVERIFIED` certificate grade, and `NO DATA` pass-rate field.

Do not present the generated hero as implementation evidence. Do not present a university, laboratory, or company logo unless permission and the precise relationship can be documented.
