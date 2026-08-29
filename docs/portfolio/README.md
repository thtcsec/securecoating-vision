# SecureCoating-Vision portfolio media

These are live captures of the Docker dashboard at `http://127.0.0.1:8501/`.
Do not present a university, laboratory, or company logo unless permission and
the precise relationship can be documented.

## Dataset vs dashboard library

| Surface | Count | What it is |
|---|---|---|
| `data/external/coatingvision/detection` | ~2227 JPEGs | Public CoatingVision archive (gitignored) |
| `data/coatingvision_real_detect` | 581 labeled pairs | Deterministic train/val/test copy (gitignored) |
| Dashboard Dataset Library / `data/demo_real` | 88 JPEGs | Held-out **test split**, SHA-256 + DOI, CC BY 4.0 |

The library is real optical frames, not synthetic. It is not an independent
factory metric set.

## Runtime screenshots

- **`runtime-guide.png`:** default Guide: fail-closed HOLD, PASS/REJECT/HOLD, tab map.
- **`runtime-operate.png`:** `HOLD_REQUIRED`, readiness blockers, PLC telemetry.
- **`runtime-history.png`:** acquired frame → model input → detection overlay on a real CoatingVision smoke run.
- **`runtime-history-log.png`:** matched detections plus `SAFETY_VALIDATION FAIL_CLOSED` → `GATE_DISPOSITION HOLD`.
- **`runtime-dataset.png`:** Dataset Library, **88** indexed images, `VERIFIED REAL OPTICAL`, Figshare DOI, page 1/15.
- **`runtime-diagnose.png`:** readiness JSON (`ready: false`, offline thermal/profiler, latched interlock).
- **`runtime-traceability.png` / `runtime-certificate.png`:** retained defects, inspection counts, certificate ID, payload digest, HMAC, `UNVERIFIED` grade.
