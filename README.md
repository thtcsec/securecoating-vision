# SecureCoating Vision

[![Track 4 Finalist 2026](https://img.shields.io/badge/Track%204%20Finalist-2026-C8102E.svg)](README.md)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)

[中文说明](README_CN.md)

**Evidence-Gated AI Inspection for Battery Electrode Manufacturing**

**Team 71** · Global AI + Materials Innovation Application Competition 2026  
**Track 4** · AI + Materials Testing & Characterization  
**Trinh Hoang Tu** · HUFLIT  
**Advisor:** Prof. Kris Singh — Visiting Professor, Tsinghua University; Founder & CEO, SRII  
(Advisory scope: innovation, industrialization, and final-defense guidance; not a detector co-author)

**Competition title:** SecureCoating-Vision: Evidence-Gated Multimodal AI for Battery Electrode Inspection and Traceable Quality Decisions

**Tagline:** Evidence-Gated Multimodal Inspection for Battery Electrode Manufacturing

SecureCoating-Vision is an **industrial computer-vision research prototype** by Team 71 (HUFLIT). Production security hardening such as mTLS, RBAC, secret rotation, and OT segmentation remains part of the industrialization roadmap and is not claimed complete.

It is **not production-qualified**. The repository does not contain a factory calibration certificate, PLC hardware-in-the-loop evidence, an independent roll-disjoint test set, or plant safety approval. Do not connect it directly to a production gate or treat the software E-stop as a safety-rated E-stop.

## What is actually implemented?

- RGB inspection on real CoatingVision test evidence (image-disjoint; hash-pinned weights)
- Evidence / readiness gate before automatic disposition
- PASS / REJECT / HOLD software semantics
- PLC command/ACK contract verified in software
- Traceability hooks (roll/batch/part + certificate snapshot)
- Reproducibility manifests and packed pytest evidence
- LIBAD multimodal validation adapter (not paper-comparable DINOv3/DA-Core)

## What is NOT claimed?

- Factory-qualified or plant-approved deployment
- Roll-disjoint validation of the current RGB headline metric
- Physical safety-rated E-stop
- Completed production mTLS / RBAC / OT segmentation
- Electrochemical performance prediction
- Paper-comparable LIBAD DINOv3 reproduction

Deployment phases and industrial value are summarized in [docs/industrialization_path.md](docs/industrialization_path.md). The Track 4 inspection-system proposal identity is [docs/Inspection_System_Proposal.md](docs/Inspection_System_Proposal.md).

## Scope and Evidence

The repository contains a fail-closed inspection prototype, traceability experiments, simulated industrial I/O, and reproducible software checks. Implemented behavior, test evidence, simulation boundaries, and external validation requirements are recorded in the [implementation status ledger](docs/implementation_status.md).

The initial prototype validated the software and safety contract using RGB inspection and simulated secondary modalities. The repository now contains a validation adapter and 10-seed harness for [LIBAD](https://arxiv.org/abs/2608.07958), whose official release contains aligned visible-light and inline-compatible X-ray data from real roll-to-roll electrode manufacturing. Git does not contain the gated official LIBAD download archives (~4.84 GB compressed); after extraction, the local dataset tree can be hash-recorded with `scripts/record_libad_official_manifest.py` (see `reports/libad/official_mount_hashes.json`). Checked-in `reports/libad/libad_benchmark.json` remains a deterministic protocol fixture. Official-input numpy numbers live in `reports/libad/official_local_adapter.json` (10 official splits; `comparable_to_paper: false`; use as motivation that FPR remains too high for unsupervised line authority, not as a trophy metric). An authors' runner interim DINOv2 smoke lives in `reports/libad/official_dinov2_dacore_interim.json` (~AUROC 0.856 / FPR95 0.716, 1 seed; Q&A backup only). Textual PAPER_SPEC is DINOv3 ViT-S/16 (paper↔code mismatch vs common ConvNeXt upstream defaults). The LIBAD lane extends validation of the same battery-electrode inspection and quality-decision problem. Details are in [docs/libad_validation_extension.md](docs/libad_validation_extension.md).

<!-- TEST_MANIFEST:START -->
The current software validation snapshot is 283 passing tests with 0 skips and 0 failures (commit `17e418d1f5bf`, working tree clean, source diff `none`, Python 3.11.9, 89.21s). The authoritative record is [reports/test_manifest.json](reports/test_manifest.json). This does not constitute evidence of factory performance, physical PLC behavior, safety-rated E-stop operation, or production qualification.
<!-- TEST_MANIFEST:END -->

## Verified local demo path

The default Compose deployment is the conservative CPU path:

```powershell
docker compose up --build -d
```

Open `http://127.0.0.1:8501`. The live runtime card must report the runtime that
is actually active, for example `EDGE · ONNX CPU · FP32 · 512px`. History shows
the evidence sequence on the same acquired frame: original JPEG → published
pixel mask → published-label heatmap → independent AI overlay. A model result
does not override the safety gate; missing physical sensors, PLC readiness, or
verified calibration remains `HOLD_REQUIRED`.

For an NVIDIA host with NVIDIA Container Toolkit:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

Do not describe the GPU path as TensorRT unless the live card reports TensorRT.
When TensorRT libraries are unavailable, the tested RTX path falls back to YOLO
CUDA FP16 and records the reason. See [adaptive inference profiles](docs/hardware_profiles.md).

## Data inventory and evidence classes

| Source | Role | Evidence boundary |
|---|---|---|
| Argonne CoatingVision | Real optical images, published masks and multi-label classes | Current detector report is image-disjoint, not factory roll-disjoint |
| LIBAD official mount | Real aligned VIS + X-rayL validation input | Local adapter is not the authors' DINOv3/DA-Core and is not paper-comparable |
| Synthetic paired coating data | Development, fault injection and contract tests | Never presented as real plant performance |
| Thermal/profilometry adapters | Registration and fail-closed interface tests | Simulated/injected unless factory instruments are connected and calibrated |

Dataset pages and reports preserve these evidence classes rather than combining
public labels, synthetic artifacts and model predictions into one unsupported
metric.

The checked-in `data/evaluation` set is a **synthetic evaluator fixture**. Its
ground-truth labels are included solely to reproduce evaluator behavior; they
are not training labels and are not the source of the reported CoatingVision
RGB performance metrics. The corresponding `reports/evaluation_results.json`
is explicitly marked `SYNTHETIC_EVALUATOR_FIXTURE` and must not be presented as
the real two-class RGB detector result. Its mask values are bbox-derived proxy
masks for detection-only output, not semantic or instance-segmentation
evidence. Reproduce it with the complete command in
`reports/evaluation_command.txt`.

This synthetic fixture reuses development images by design, so it is not an
independent benchmark. Its ZIP-safe reference stubs live under
`data/evaluation/reference/` (not `coating_defects`). Its manifest and labels
exist to exercise the evaluator contract, not to support a leakage-safe model
claim.

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

The authoritative model report is [reports/coatingvision_real_test_metrics.json](reports/coatingvision_real_test_metrics.json): public real optical data, a fixed 88-image image-disjoint test split (seed 71), precision 0.645, recall 0.642, mAP50 0.634, and mAP50-95 0.354. It is explicitly **not** factory roll-disjoint evidence. Older synthetic segmentation reports are retained only as development baselines and must not be presented as current-model evidence.

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

Production secrets must each contain at least 32 bytes. Production startup also
rejects wildcard CORS origins, wildcard trusted hosts, malformed origins, and
non-HTTPS non-loopback origins. API responses—including authentication and body
limit failures—carry no-store, anti-framing, MIME-sniffing, restrictive CSP,
permissions-policy, and cross-origin resource-policy headers. HSTS is emitted
only when the request is already HTTPS; TLS termination remains the deployer's
responsibility.

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
.venv\Scripts\python.exe scripts/run_libad_official_baseline.py --run-card
.venv\Scripts\python.exe scripts/run_libad_official_baseline.py --smoke
.venv\Scripts\python.exe scripts/run_libad_official_baseline.py --allow-heavy --modalities vis_xray_l
.venv\Scripts\python.exe scripts/run_libad_live.py --fetch-official-code --with-api --write-reports
.venv\Scripts\python.exe scripts/download_libad.py --probe
.venv\Scripts\python.exe scripts/record_libad_official_manifest.py
.venv\Scripts\python.exe scripts/run_libad_benchmark.py --require-official --out reports/libad/official_local_adapter.json
.venv\Scripts\python.exe scripts/verify_coatingvision_dataset.py --grouping
.venv\Scripts\python.exe scripts/generate_defense_gifs.py
.venv\Scripts\python.exe scripts/generate_defense_slides.py
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

Inference serving is hardware-adaptive. The CPU Compose image resolves `auto` to
the ONNX-first `edge` profile; NVIDIA hosts can use the GPU overlay for
CUDA/TensorRT-capable `balanced` or `performance` routing. Live health telemetry
reports the requested and active profile, provider, actual precision, image size,
and fallback reason. See [docs/hardware_profiles.md](docs/hardware_profiles.md).

The Compose services use a non-root image user, read-only root filesystem,
dropped Linux capabilities, `no-new-privileges`, bounded PID/file-descriptor
limits, bounded temporary filesystems, graceful shutdown, and rotated local
JSON logs. Ports bind to loopback. These controls reduce software risk; they do
not replace network segmentation, a reverse proxy/TLS boundary, image signing,
SBOM review, vulnerability scanning, backups, or factory HIL validation.

`requirements-lock.txt` pins the local runtime while `requirements-docker-lock.txt` pins CPU-only PyTorch variants for the container. They are direct-version locks, not hash-locked fully resolved environments. A release still requires CI-generated locks with hashes plus a container build/SBOM scan for the target platform.

## License

Project source is provided under the [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE) because shipped Ultralytics YOLO26 weights inherit Ultralytics AGPL-3.0 terms when redistributed without Ultralytics Enterprise. Third-party attributions (Ultralytics, CoatingVision, LIBAD/DA-Core) are in [NOTICE.md](NOTICE.md). Dataset licenses and factory safety approvals remain the deployer's responsibility.
