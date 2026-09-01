# LIBAD Validation Extension

This repository adds a **LIBAD-compatible multimodal evidence adapter** without replacing the existing two-class RGB YOLO/ONNX detector. The official dataset is real multimodal data; the checked-in demo and benchmark artifacts are deterministic protocol fixtures.

## What changed, and what did not

The original software-and-safety contract remains:

- YOLO / ONNX object detection for known surface-defect localization
- FastAPI, dashboard, traceability memory, HMAC certificates
- PASS / REJECT / HOLD
- OPC UA / Modbus contracts, calibration checks, model/dataset/commit hashes

LIBAD is an **external validation extension**, not a topic change:

> The initial prototype validated the software and safety contract using RGB inspection and simulated secondary modalities. We then added an adapter and evidence gate for a newly released real multimodal battery-electrode benchmark. Official external validation remains pending until its dataset, all 10 splits, and implementation hashes are recorded.

Honest modality split:

- **Target external validation inputs:** real VIS + inline-compatible X-rayL (official LIBAD, CC BY 4.0; not checked into this repository)
- **Checked-in LIBAD artifacts:** deterministic protocol fixtures, always `comparable_to_paper: false`
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

Without the official 4.84 GB dataset, all 10 valid split manifests, and `data/libad/official_artifact_manifest.json` containing matching dataset/splits tree SHA-256 values, the harness runs a protocol fixture or labels structured inputs unverified. Even with verified official inputs, this local numpy patch descriptor remains `comparable_to_paper: false`; paper comparison requires the authors' official DINOv3/DA-Core implementation and recorded hashes.

```powershell
.venv\Scripts\python.exe scripts/download_libad.py
.venv\Scripts\python.exe scripts/run_libad_benchmark.py
.venv\Scripts\python.exe scripts/run_libad_demo.py
```

## 90-second demo

1. Normal agreement → PASS
2. Surface-visible defect → REJECT (surface evidence)
3. Internally visible X-ray anomaly → REJECT (complementary X-ray)
4. Near-threshold VIS/X-rayL disagreement with valid contracts → HOLD

The closing screen shows roll, batch, part, modality scores, calibration state, model/commit hashes, decision reason, certificate signature, and PLC state.
