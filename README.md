# SecureCoating-Vision

**Registered title:** SecureCoating Vision: A High-Throughput and Zero-Trust Edge-Cloud Pipeline for Inline Battery Electrode Defect Inspection and Traceable Quality Decisions

**Tagline:** Evidence-Gated Multimodal Inspection for Battery Electrode Manufacturing

SecureCoating-Vision is an **industrial computer-vision research prototype** for coating-defect segmentation, fail-closed inspection decisions, simulated roll synchronization, PLC protocol integration, and traceability experiments.

It is **not production-qualified**. The repository does not contain a factory calibration certificate, PLC hardware-in-the-loop evidence, an independent roll-disjoint test set, or plant safety approval. Do not connect it directly to a production gate or treat the software E-stop as a safety-rated E-stop.

## Scope and Evidence

The repository contains a fail-closed inspection prototype, traceability experiments, simulated industrial I/O, and reproducible software checks. Implemented behavior, test evidence, simulation boundaries, and external validation requirements are recorded in the [implementation status ledger](docs/implementation_status.md).

The initial prototype validated the software and safety contract using RGB inspection and simulated secondary modalities. Following that qualification path, the repository adds an external validation lane on [LIBAD](https://arxiv.org/abs/2608.07958): aligned visible-light and inline-compatible X-ray data from real roll-to-roll electrode manufacturing. This is a validation extension, not a change of topic. Details are in [docs/libad_validation_extension.md](docs/libad_validation_extension.md).

<!-- TEST_MANIFEST:START -->
The current software validation snapshot is 109 passing tests with 0 skips and 0 failures (commit `a4adb9fad8f5`, Python 3.11.9, 40.34s). The authoritative record is [reports/test_manifest.json](reports/test_manifest.json). This does not constitute evidence of factory performance, physical PLC behavior, safety-rated E-stop operation, or production qualification.
<!-- TEST_MANIFEST:END -->

## Current safety contract

- Only an `OPTIMAL` inference result from a loaded trained model may produce an automatic PASS/REJECT decision.
- Timeout, inference error, missing model, missing sensor, unverified production calibration, traceability failure, PLC communication failure, modality disagreement, stale evidence, or a latched interlock produces `HOLD`.
- Mock OPC UA/Modbus operations are labelled `SIMULATED`; they are never reported as PLC acknowledgements.
- Real PLC commands use exactly one configured command-owner protocol and require a unique command sequence plus a matching PLC ACK sequence. OPC UA requires `SignAndEncrypt`; plaintext Modbus is refused unless an explicitly trusted gateway is configured.
- The dashboard reads the API as its authoritative source. Local fallback is available only when `SECURECOATING_DASHBOARD_SANDBOX=true` in development/test, and it never owns a live PLC channel.

These properties are covered by software tests, but physical actuator behavior still requires vendor-specific HIL and safety validation.

## Modality honesty

- **RGB / YOLO-seg / ONNX:** primary inference path for known surface defects.
- **LIBAD VIS + X-rayL:** real multimodal validation extension. PatchCore/DA-Core scores are the published baseline of Sui et al.; SecureCoating-Vision does not claim DA-Core as its own algorithm. The local contribution is the evidence gate.
- **Thermal and profilometry:** simulated or injected interface adapters for registration, fail-closed degradation, and contract tests. They are not plant-instrument measurements in this repository.

## Current safety contract

- Only an `OPTIMAL` inference result from a loaded trained model may produce an automatic PASS/REJECT decision.
- Timeout, inference error, missing model, missing sensor, unverified production calibration, traceability failure, PLC communication failure, or a latched interlock produces `HOLD`.
- Mock OPC UA/Modbus operations are labelled `SIMULATED`; they are never reported as PLC acknowledgements.
- Real PLC commands use exactly one configured command-owner protocol and require a unique command sequence plus a matching PLC ACK sequence. OPC UA requires `SignAndEncrypt`; plaintext Modbus is refused unless an explicitly trusted gateway is configured.
- The dashboard reads the API as its authoritative source. Local fallback is available only when `SECURECOATING_DASHBOARD_SANDBOX=true` in development/test, and it never owns a live PLC channel.

These properties are covered by software tests, but physical actuator behavior still requires vendor-specific HIL and safety validation.

## Status and Upgrade Plan

The staged implementation checklist, evidence matrix, release gates, and next-step plan are maintained in [docs/implementation_status.md](docs/implementation_status.md).

For a running local API in explicit development simulation mode, use the existing smoke script:

```powershell
.venv\Scripts\python.exe scripts/smoke_live.py
```

## Model and metric evidence

Tracked reports under `reports/` are development artifacts, not production qualification results. The existing `data/evaluation` files overlap the tracked training validation split; therefore those results must not be presented as an independent test-set estimate.

The repository deliberately makes no claim of 99.4% mAP, zero escapes, ppm performance, factory yield improvement, production latency, Six Sigma capability, or standards certification. Such claims require an immutable model hash, roll-disjoint dataset manifest, raw measurements, methodology, confidence intervals, and independent review.

`scripts/run_pilot_validation_dossier.py` now requires a hash-verified evidence manifest and will not create pilot values by itself.

## Configuration required for production-mode startup

The API defaults to fail-closed production mode. At minimum configure:

```powershell
$env:SECURECOATING_ENV = "production"
$env:SECURECOATING_API_KEY = "<secret from your secret manager>"
$env:SECURECOATING_FACTORY_SECRET = "<certificate signing secret>"
$env:SECURECOATING_INDUSTRIAL_MOCK_MODE = "false"
```

You must also replace the simulation-only values in `configs/calibration.yaml`, configure the selected PLC command owner and its command/ACK sequence contract, and provision OPC UA client/server certificates. Set `modbus.trusted_gateway: true` only after the OT network control has been independently verified.

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

The manifest must contain non-empty `train`, `val`, and `test` lists. Each entry must include `path`, `roll_id`, and SHA-256; roll IDs may not cross splits, and the test list must exactly match the evaluation image directory. Before publishing results, prove by hash that the evaluation set does not overlap train/validation data and record the model, dataset, manifest, and commit hashes.

## Training

- `src/training/train_yolo.py` requires an explicit model identifier/path and refuses to synthesize an empty dataset unless `--allow-synthetic` is supplied.
- `src/training/train_baseline.py` requires a CSV manifest referencing real RGB, thermal, height and mask artifacts; missing files are fatal.
- `scripts/prepare_synthetic_dataset.py` creates development data only. Synthetic results must be labelled as synthetic.

## Deployment limits

The supplied Compose file runs one API worker because PLC, inference and active-roll ownership are stateful. Scaling requires an external transactional state/event service and a single PLC-command owner. Containers run as a non-root user with dropped capabilities and bind only to loopback by default.

`requirements-lock.txt` pins direct runtime requirements, but it is not a hash-locked, fully resolved cross-platform lock. A release build still requires a CI-generated lock with hashes and a successful container build/SBOM scan for the target platform.

## License

Source code is provided under the [MIT License](LICENSE). Dataset licenses and factory safety approvals remain the deployer's responsibility.
