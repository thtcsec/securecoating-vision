# SecureCoating-Vision

<p align="center">
  <img src="logo.png" alt="SecureCoating-Vision" width="450">
</p>

[![Track 4 Finalist](https://img.shields.io/badge/Tsinghua%20MSE%202026-Track%204%20Finalist-C8102E.svg)](README.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Team 71** · Trịnh Hoàng Tú — HUFLIT  
**Advisor:** Prof. Kris Singh — SRII / Visiting Professor, Tsinghua University  
**Competition:** 2026 Global AI + Materials Innovation Application Competition · Track 4 (AI + Materials Testing and Characterization)

**Registered title:** SecureCoating Vision: A High-Throughput and Zero-Trust Edge-Cloud Pipeline for Inline Battery Electrode Defect Inspection and Traceable Quality Decisions

**Tagline:** Evidence-Gated Multimodal Inspection for Battery Electrode Manufacturing

SecureCoating-Vision is an **industrial computer-vision research prototype** for real-image coating-defect detection, fail-closed inspection decisions, simulated roll synchronization, PLC protocol integration, and traceability experiments.

It is **not production-qualified**. The repository does not contain a factory calibration certificate, PLC hardware-in-the-loop evidence, an independent roll-disjoint test set, or plant safety approval. Do not connect it directly to a production gate or treat the software E-stop as a safety-rated E-stop.

## Scope and Evidence

The repository contains a fail-closed inspection prototype, traceability experiments, simulated industrial I/O, and reproducible software checks. Implemented behavior, test evidence, simulation boundaries, and external validation requirements are recorded in the [implementation status ledger](docs/implementation_status.md).

The initial prototype validated the software and safety contract using RGB inspection and simulated secondary modalities. The repository now contains a validation adapter and 10-seed harness for [LIBAD](https://arxiv.org/abs/2608.07958), whose official release contains aligned visible-light and inline-compatible X-ray data from real roll-to-roll electrode manufacturing. Git does not contain the 4.84 GB mount; a local copy can be hashed with `scripts/record_libad_official_manifest.py`. Checked-in `reports/libad/libad_benchmark.json` remains a deterministic protocol fixture. Official-input numpy numbers live in `reports/libad/official_local_adapter.json` and stay `comparable_to_paper: false`. This is a validation extension, not a change of topic. Details are in [docs/libad_validation_extension.md](docs/libad_validation_extension.md).

<!-- TEST_MANIFEST:START -->
The current software validation snapshot is 224 passing tests with 0 skips and 0 failures (commit `5ee634b23499`, working tree clean, source diff `none`, Python 3.11.9, 47.98s). The authoritative record is [reports/test_manifest.json](reports/test_manifest.json). This does not constitute evidence of factory performance, physical PLC behavior, safety-rated E-stop operation, or production qualification.
<!-- TEST_MANIFEST:END -->

## Current safety contract

- Only an `OPTIMAL` inference result from a loaded trained model may produce an automatic PASS/REJECT decision.
- Timeout, inference error, missing model, missing sensor, unverified production calibration, traceability failure, PLC communication failure, modality disagreement, stale evidence, or a latched interlock produces `HOLD`.
- Mock OPC UA/Modbus operations are labelled `SIMULATED`; they are never reported as PLC acknowledgements.
- Real PLC commands use exactly one configured command-owner protocol and require a unique command sequence plus a matching PLC ACK sequence. OPC UA requires `SignAndEncrypt`; plaintext Modbus is refused unless an explicitly trusted gateway is configured.
- The production dashboard reads one `/api/operations/snapshot` payload. An English Guide explains the workflow and fail-closed states. History loads an HTML replay of original JPEG → published pixel mask → published-label heatmap → AI overlay, then shows Argonne CoatingVision classification flags beside the local detector class (side by side, not fused). Dataset Library pages hash-verified metadata, published masks/heatmaps, classification chips, and an honest published-class mix (multi-label; Surface_Crack dominates this public set). A disk catalog cache avoids rescanning thousands of Figshare files after the first warmup. Dataset thumbnails and published evidence views are reused in process after the first hash-verified encode. History and Dataset fetch retained JPEGs in parallel instead of one request at a time. The live strip reports YOLO/ONNX engine load separately from fail-closed sensors; a loaded model is not a PASS authorization. Views stay on a main-page rail so collapsing the left identity panel cannot hide navigation; a cyan chevron restores that panel if it is already collapsed. Live telemetry uses a lighter snapshot scope and keeps the last catalog/certificate payload. Inspection artifact presence is taken from one directory listing, not a per-file exists() scan. The Multimodal view shows official VIS + X-ray frames only when that release is mounted as complete file triples; protocol fixtures are not substituted as plant X-ray. Simulator, recipe sliders, defect injection, and the LIBAD evidence demo exist only in the explicit development/test sandbox. Operator control is limited to authenticated confirm-audit actions (E-stop, reset, inference reset); the dashboard never writes recipe offsets to a PLC.

These properties are covered by software tests, but physical actuator behavior still requires vendor-specific HIL and safety validation.

## Modality honesty

- **RGB / two-class YOLO detector / ONNX:** primary inference path for `surface_crack` and `delamination_crack` on real CoatingVision optical images. The PyTorch and ONNX artifacts share the same class map and are hash-pinned in `configs/model.yaml`.
- **LIBAD VIS + X-rayL:** adapter for official real multimodal inputs. Checked-in fixture artifacts stay protocol fixtures; a local official mount is gitignored. PatchCore/DA-Core are attributed baselines of Sui et al.; the local numpy/OpenCV descriptor is CPU-only, not their official DINOv3 implementation, and is never paper-comparable. The local contribution is the evidence gate.
- **Thermal and profilometry:** simulated or injected interface adapters for registration, fail-closed degradation, and contract tests. They are not plant-instrument measurements in this repository.

## Status and Upgrade Plan

The staged implementation checklist, evidence matrix, release gates, and next-step plan are maintained in [docs/implementation_status.md](docs/implementation_status.md).

For a running local API in explicit development simulation mode, use the existing smoke script:

```powershell
.venv\Scripts\python.exe scripts/smoke_live.py
```

## Model and metric evidence

The authoritative model report is [reports/coatingvision_real_test_metrics.json](reports/coatingvision_real_test_metrics.json): public real optical data, a fixed 88-image image-disjoint test split, precision 0.645, recall 0.642, mAP50 0.633, and mAP50-95 0.354. It is explicitly **not** factory roll-disjoint evidence. Older synthetic segmentation reports are retained only as development baselines and must not be presented as current-model evidence.

The repository deliberately makes no claim of 99.4% mAP, zero escapes, ppm performance, factory yield improvement, production latency, Six Sigma capability, or standards certification. Such claims require an immutable model hash, roll-disjoint dataset manifest, raw measurements, methodology, confidence intervals, and independent review.

`scripts/run_pilot_validation_dossier.py` now requires a hash-verified evidence manifest and will not create pilot values by itself.

## Configuration required for production-mode startup

The API defaults to fail-closed production mode. At minimum configure:

```powershell
$env:SECURECOATING_ENV = "production"
$env:SECURECOATING_API_KEY = "<secret from your secret manager>"
$env:SECURECOATING_FACTORY_SECRET = "<certificate signing secret>"
$env:SECURECOATING_OPERATOR_CREDENTIALS = '[{"operator_id":"OP_SHIFT_A","token_sha256":"<sha256-of-operator-token>","roles":["operator","safety_reset"]}]'
$env:SECURECOATING_INDUSTRIAL_MOCK_MODE = "false"
```

You must also replace the simulation-only values in `configs/calibration.yaml`, configure the selected PLC command owner and its command/ACK sequence contract, and provision OPC UA client/server certificates. Set `modbus.trusted_gateway: true` only after the OT network control has been independently verified.

Production operator controls are disabled unless `SECURECOATING_OPERATOR_CREDENTIALS` contains hashed tokens. `operator` may request E-stop, `safety_reset` may reset the line, and `maintenance` may arm an inference recovery probe. Never store plaintext operator tokens in the repository.

For an explicit unauthenticated local test sandbox only:

```powershell
$env:SECURECOATING_ENV = "development"
$env:SECURECOATING_ALLOW_UNAUTHENTICATED_DEMO = "true"
$env:SECURECOATING_ENABLE_SENSOR_SIMULATION = "true"
$env:SECURECOATING_DASHBOARD_SANDBOX = "true"
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Never expose that mode outside a trusted developer workstation.

## Run

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts/record_test_manifest.py
.venv\Scripts\python.exe scripts/run_libad_demo.py
.venv\Scripts\python.exe scripts/run_libad_benchmark.py
.venv\Scripts\python.exe scripts/run_libad_live.py --fetch-official-code --with-api --write-reports
.venv\Scripts\python.exe scripts/download_libad.py --probe
.venv\Scripts\python.exe scripts/record_libad_official_manifest.py
.venv\Scripts\python.exe scripts/run_libad_benchmark.py --require-official --out reports/libad/official_local_adapter.json
.venv\Scripts\python.exe scripts/verify_coatingvision_dataset.py --grouping
.venv\Scripts\python.exe scripts/run_modbus_loopback_trigger.py
.venv\Scripts\python.exe -m compileall -q src dashboard scripts tests
.venv\Scripts\python.exe -m pip check
docker compose config
```

The dashboard is API-authoritative by default. To run its explicitly isolated local sandbox:

```powershell
.venv\Scripts\streamlit.exe run dashboard/app.py
```

## Evaluation

Evaluation requires a real ONNX artifact. Missing models, malformed labels, undecodable images, and labels absent for negative samples are fatal. Use an empty label file for a verified negative image.

```powershell
.venv\Scripts\python.exe scripts/run_evaluation.py `
  --dataset-dir <independent-roll-test-set> `
  --dataset-manifest <immutable-roll-disjoint-manifest.json> `
  --model-path <exported-model.onnx> `
  --ultralytics-model-path <source-model.pt> `
  --ultralytics-data-config <matching-dataset.yaml>
```

The manifest must contain non-empty `train`, `val`, and `test` lists. Each entry must include `path`, `roll_id`, and SHA-256; test entries must additionally include `label_path` and `label_sha256`. Roll IDs may not cross splits, and the test image/label lists must exactly match the evaluation directories. Before publishing results, prove by hash that the evaluation set does not overlap train/validation data and record the model, dataset, manifest, and commit hashes.

## Training

- `src/training/train_yolo.py` requires an explicit model identifier/path and refuses to synthesize an empty dataset unless `--allow-synthetic` is supplied.
- `src/training/train_baseline.py` requires a CSV manifest referencing real RGB, thermal, height and mask artifacts; missing files are fatal.
- `scripts/prepare_synthetic_dataset.py` creates development data only. Synthetic results must be labelled as synthetic.

## Deployment limits

The supplied Compose file runs one API worker because PLC, inference and active-roll ownership are stateful. Scaling requires an external transactional state/event service and a single PLC-command owner. Containers run as a non-root user with dropped capabilities and bind only to loopback by default.

`requirements-lock.txt` pins the local runtime while `requirements-docker-lock.txt` pins CPU-only PyTorch variants for the container. They are direct-version locks, not hash-locked fully resolved environments. A release still requires CI-generated locks with hashes plus a container build/SBOM scan for the target platform.

## License

Source code is provided under the [MIT License](LICENSE). Dataset licenses and factory safety approvals remain the deployer's responsibility.
