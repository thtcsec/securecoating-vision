# LIBAD Validation Extension

This repository adds a **LIBAD-compatible multimodal evidence adapter** without replacing the existing two-class RGB YOLO/ONNX detector. The official dataset is real multimodal data; the checked-in demo and benchmark artifacts are deterministic protocol fixtures.

## What changed, and what did not

The original software-and-safety contract remains:

- YOLO / ONNX object detection for known surface-defect localization
- FastAPI, dashboard, traceability memory, HMAC certificates
- PASS / REJECT / HOLD
- OPC UA / Modbus contracts, calibration checks, model/dataset/commit hashes

LIBAD is an **external validation extension**, not a topic change:

> The initial prototype validated the software and safety contract using RGB inspection and simulated secondary modalities. We then mounted official LIBAD, hashed the trees, and ran the 10-seed local numpy adapter. Authors' DINOv3+DA-Core is launched via `scripts/run_libad_official_baseline.py` once gated HF model access is available.

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

Without the official 4.84 GB dataset, all 10 valid split files, and a hash-recorded artifact manifest, the harness runs a protocol fixture or labels structured inputs unverified. Default status checks a cheap file-count/byte fingerprint against `data/libad/official_artifact_manifest.json` (copied to `reports/libad/official_mount_hashes.json`) so pytest and the API do not SHA-256 4.84 GB on every call. Live rehash is `SECURECOATING_LIBAD_VERIFY_TREES=1` or `dataset_status(verify_trees=True)`. Even with verified official inputs, this local numpy patch descriptor remains `comparable_to_paper: false`; paper comparison requires the authors' official DINOv3/DA-Core implementation and recorded hashes. Checked-in `reports/libad/libad_benchmark.json` stays `evidence_class: protocol_fixture`. Official local-adapter numbers belong in `reports/libad/official_local_adapter.json`.

```powershell
.venv\Scripts\python.exe scripts/download_libad.py --probe
.venv\Scripts\python.exe scripts/download_libad.py --download --accept-license
.venv\Scripts\python.exe scripts/record_libad_official_manifest.py
.venv\Scripts\python.exe scripts/run_libad_benchmark.py --require-official --out reports/libad/official_local_adapter.json
.venv\Scripts\python.exe scripts/run_libad_official_baseline.py --run-card
.venv\Scripts\python.exe scripts/run_libad_official_baseline.py --smoke
.venv\Scripts\python.exe scripts/run_libad_official_baseline.py --allow-heavy --modalities vis_xray_l
.venv\Scripts\python.exe scripts/run_libad_official_baseline.py --smoke --dino-version v2 --backbone-family vit --backbone-variant small --out reports/libad/official_dinov2_dacore_interim.json
.venv\Scripts\python.exe scripts/run_libad_benchmark.py
.venv\Scripts\python.exe scripts/run_libad_demo.py
.venv\Scripts\python.exe scripts/run_libad_live.py --fetch-official-code --with-api --write-reports
```

`--probe` HEADs the official zip files and does not download 4.84 GB. `--download` requires `--accept-license` plus a Hugging Face token after accepting the dataset terms. Unauthenticated requests receive HTTP 401 (`GatedRepoError`). Even after a successful mount, this repository's numpy patch descriptor remains `comparable_to_paper: false`. Textual PAPER_SPEC for DA-Core (arXiv:2608.07958) is frozen **DINOv3 ViT-S/16 + max-NN**; common upstream `run.py` defaults often resolve to ConvNeXt-base — the harness records `paper_code_consistency: mismatch` and will not auto-claim PAPER_EXACT for ConvNeXt-base (that path is at most `OFFICIAL_CODE_CORE_ADAPTED`). Authors' DINOv3 ConvNeXt weights are separately gated on Hugging Face; set `HF_TOKEN` in `.env` after accepting `facebook/dinov3-convnext-*` terms. Default harness uses a **laptop profile** (ConvNeXt-tiny, batch 1, coreset on CPU, no bank dumps, cool-down pauses, `--allow-heavy` required for multi-cell runs). `--paper-config` forces textual PAPER_SPEC (**ViT / v3 / small / max-NN**) and also requires `--allow-heavy`.

## 90-second demo

1. Normal agreement → PASS
2. Surface-visible defect → REJECT (surface evidence)
3. Internally visible X-ray anomaly → REJECT (complementary X-ray)
4. Near-threshold VIS/X-rayL disagreement with valid contracts → HOLD

The closing screen shows roll, batch, part, modality scores, calibration state, model/commit hashes, decision reason, certificate signature, and PLC state.
