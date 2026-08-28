# LIBAD Validation Extension

This repository adds a **real multimodal evidence lane** without replacing the existing RGB YOLOv8-seg/ONNX prototype.

## What changed, and what did not

The original software-and-safety contract remains:

- YOLOv8-seg / ONNX for known surface-defect localization
- FastAPI, dashboard, traceability memory, HMAC certificates
- PASS / REJECT / HOLD
- OPC UA / Modbus contracts, calibration checks, model/dataset/commit hashes

LIBAD is an **external validation extension**, not a topic change:

> The initial prototype validated the software and safety contract using RGB inspection and simulated secondary modalities. Following preliminary qualification, we added external validation on a newly released real multimodal battery-electrode benchmark using aligned visible-light and inline-compatible X-ray data.

Honest modality split:

- **Real multimodal validation:** VIS + inline-compatible X-rayL (LIBAD, CC BY 4.0)
- **Simulation-based interface validation:** thermal and profilometry adapters

## Attribution

[LIBAD](https://arxiv.org/abs/2608.07958) and **DA-Core** belong to Sui, Lichau, Phelippeau, and Liu (2026). SecureCoating-Vision does **not** claim DA-Core as an original algorithm. The local contribution is the evidence-gated industrial decision layer:

```text
VIS / X-rayL
  -> PatchCore or DA-Core baseline (authors' detector)
  -> modality-level anomaly scores
  -> SecureCoating Evidence Gate
  -> PASS / REJECT / HOLD
  -> traceability + certificate + PLC contract
```

Sui et al. report that even the best LIBAD setting still has FPR95 of 54.3% with AUROC 86.7%, AUPR 95.7%, and F1-max 90.6%. They conclude that this false-positive rate is too high for direct deployment and call out modality disagreement and closed-loop process control as open problems.

LIBAD answers **how to detect**. SecureCoating-Vision answers **when a detection is safe enough to act on**.

## Required experiments

Use the 10 official split seeds only: `347, 725, 1245, 4012, 4589, 5021, 5678, 6234, 6789, 7345`.

| Experiment | VIS | X-rayL | Purpose |
|---|---|---|---|
| Unimodal surface | yes |  | Surface baseline |
| Unimodal density |  | yes | Internal/density baseline |
| Multimodal | yes | yes | Fusion value |
| SecureCoating Gate | yes | yes | PASS/REJECT/HOLD safety |

Reported metrics: AUROC, AUPR, F1-max, FPR95 (mean ± std over 10 splits), latency, model hash, dataset manifest hash, commit hash, raw prediction JSON.

Industrial metrics:

- Automatic Decision Coverage = (N_PASS + N_REJECT) / N
- HOLD Rate = N_HOLD / N
- Escape Rate = (anomalies incorrectly PASS) / (total anomalies)
- Selective Risk = (errors among automatic decisions) / (N_PASS + N_REJECT)

Without the official 4.84 GB dataset the harness runs a protocol fixture and labels the report `comparable_to_paper: false`.

```powershell
.venv\Scripts\python.exe scripts/download_libad.py
.venv\Scripts\python.exe scripts/run_libad_benchmark.py
.venv\Scripts\python.exe scripts/run_libad_demo.py
```

## 90-second demo

1. Normal agreement → PASS
2. Surface-visible defect → REJECT (surface evidence)
3. Internally visible X-ray anomaly → REJECT (complementary X-ray)
4. Disagreement / stale / unverified calibration → HOLD

The closing screen shows roll, batch, part, modality scores, calibration state, model/commit hashes, decision reason, certificate signature, and PLC state.
