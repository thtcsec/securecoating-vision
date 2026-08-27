# Implementation Status and Upgrade Plan

Last reviewed: 2026-08-27

This document is the project status ledger. A feature is marked **verified** only when its behavior is covered by a reproducible repository test or a recorded manual command. Simulation is never evidence of physical PLC or factory qualification.

## Status Legend

- [x] **Verified in repository**: implementation exists and automated tests pass.
- [x] **Verified manually in local simulation**: API/manual flow was executed, but external hardware is not involved.
- [ ] **Pending external validation**: requires PLC/HIL, factory instruments, independent data, or deployment infrastructure.
- [ ] **Planned**: not implemented yet.

## Phase 0: Evidence and Scope Integrity

- [x] README labels the repository as a research prototype, not a production-qualified system.
- [x] Benchmark/evaluation data overlap is disclosed; current evaluation is not presented as independent generalization evidence.
- [x] Pilot dossier generation requires a hash-verified evidence manifest.
- [x] Synthetic data and simulated PLC operations are explicitly labelled.
- [x] Model, dataset, and commit hashes are required before publishing independent results.

## Phase 1: Safety and Control-Plane Hardening

- [x] Automatic PASS/REJECT requires `OPTIMAL`, trained model readiness, verified calibration, healthy traceability, and ready industrial transport.
- [x] Missing sensor, inference error, timeout, database failure, calibration failure, interlock, and PLC communication failure produce `HOLD`.
- [x] Mock PLC operations are labelled `SIMULATED` and never claim real ACK.
- [x] Exactly one configured protocol owns PLC commands; command sequence and ACK sequence are distinct.
- [x] Real OPC UA commands require configured SignAndEncrypt, command readback, and matching PLC ACK sequence.
- [x] Real Modbus commands require an explicitly trusted gateway, response validation, command readback, and matching ACK sequence.
- [x] E-stop and HOLD latch local state; reset clears the latch only after both channels confirm.
- [x] Stateful API runs with one worker by default.
- [x] API control endpoints require authentication unless an explicit local development override is enabled.
- [x] Upload body, media type, image header, pixel count, and decoded image limits are enforced.
- [x] Timeout circuit prevents a new inference while a previous timed-out worker is still running.
- [x] Runtime model errors/timeouts latch the inference circuit; an authenticated reset permits one recovery probe.
- [x] Production PLC state starts latched/unverified after process restart until a sequence-acknowledged reset.
- [ ] Verify physical HOLD/reject gate behavior with vendor PLC HIL.
- [ ] Verify safety-rated E-stop and hardwired interlock behavior with the plant safety owner.
- [ ] Add mTLS/RBAC and OT network segmentation for the target deployment.

## Phase 2: Correctness and Traceability

- [x] Roll and batch identifiers are validated before map/certificate retrieval.
- [x] Inspection ingestion rejects non-active batches instead of attaching them to the active roll.
- [x] Certificate signatures cover the complete canonical payload, including nested defect map entries.
- [x] Certificate output marks provisional metrics as `UNVERIFIED`.
- [x] SQLite writes report failure instead of silently returning healthy defaults.
- [x] SQLite connections close deterministically; indexes cover batch/timestamp and run ID queries.
- [x] Duplicate `(batch_id, part_id)` identities are transactionally claimed before PLC signaling.
- [x] Trace rows remain `PENDING` until PLC outcome finalization and cannot retain a planned PASS after a finalization failure.
- [x] SQLite online backup uses the backup API, atomic replace, and integrity check.
- [x] Confidence matching uses class-aware bounding-box IoU rather than a default origin point.
- [x] Coordinate mapping uses the actual input frame dimensions.
- [x] Calibration is loaded from an explicit configuration artifact.
- [x] HOLD inspections are excluded from resolved PASS/REJECT rate calculations.
- [x] API latency and signal history limits have bounded request validation.
- [ ] Add durable multi-roll persistence and transactional roll lifecycle ownership.
- [ ] Add power-loss recovery, backup/restore, retention, and disk-full validation for SQLite.
- [ ] Define part identity/duplicate policy for certificate pass-rate calculations.

## Phase 3: Evaluation and Model Integrity

- [x] Evaluation rejects missing model, malformed labels, undecodable images, and missing negative labels.
- [x] Evaluation has integrity tests for dataset/model provenance constraints.
- [x] Training scripts require explicit real data manifests; missing artifacts are fatal.
- [x] Training baseline supports deterministic seeds and records manifest metadata.
- [x] Missing or untrained fallback models cannot be treated as production-ready inference.
- [x] ONNX post-processing has regression coverage for mask bounds and confidence output.
- [x] Box and mask evaluation metrics use one shared prediction-to-ground-truth instance match.
- [x] Evaluation requires a hash-verified manifest with non-overlapping train/val/test roll IDs.
- [ ] Supply a real independent roll-disjoint manifest and publish metrics from it.
- [ ] Run clean-surface false-positive, hard-negative, and defect false-negative suites.
- [ ] Measure POD, escape rate, confidence intervals, and performance on independent factory data.
- [ ] Add ONNX-vs-source-model parity and multi-resolution calibration tests.

## Phase 4: Deployment and Operations

- [x] Docker uses Python 3.11, direct dependency pins, a non-root runtime user, dropped capabilities, and loopback bindings by default.
- [x] Compose uses one API worker because the current roll/PLC owner is in-process.
- [x] Liveness is separated from readiness/health.
- [x] Environment examples distinguish local simulation from production configuration.
- [x] Dashboard is read-only and does not create a second live PLC command owner.
- [x] Dashboard reads authoritative health, roll, batch, SPC, PLC, and passport telemetry from the API when available.
- [x] Dashboard API client has bounded requests and explicit HTTP/JSON error handling.
- [x] Production dashboard refuses automatic local fallback; sandbox fallback is explicit development/test opt-in.
- [x] Dashboard AppTest renders without exceptions in the explicit sandbox.
- [x] Dependency resolver dry-run succeeds; Pillow/GitPython/Torch pins were advanced to audited fixed versions.
- [x] GPU development environment verified Torch 2.13.0+cu130 with the RTX 4050.
- [ ] Build and run the image with Docker Desktop or CI; local Docker daemon was unavailable during the 2026-08-27 review.
- [ ] Add CI for tests, compile, dependency checks, Docker build, and release artifact checks.
- [ ] Add browser/Compose smoke coverage against the built target image.
- [x] API inspection endpoints have bounded in-flight concurrency and reject excess work instead of building an unbounded queue.
- [ ] Add target-hardware operational metrics and alerting.

## Test Evidence Recorded

The latest repository validation completed on 2026-08-27:

```text
.venv\Scripts\python.exe -m pytest -q
82 passed in 30.84s

.venv\Scripts\python.exe -m compileall -q src dashboard scripts tests
OK

.venv\Scripts\python.exe -m pip check
No broken requirements found

.venv\Scripts\python.exe -m pip_audit -r requirements-lock.txt
No known vulnerabilities found

.venv\Scripts\python.exe -m pip install --dry-run -r requirements-lock.txt
Resolved successfully

docker compose config
OK

git diff --check
OK (line-ending conversion notices only)
```

The local live Uvicorn smoke run on port 8011 verified API liveness/readiness, clean PASS, real-image REJECT, degraded-sensor HOLD, duplicate-part HOLD, E-stop latch, simulated reset, and upload inspection. PLC statuses were `SIMULATED`, never physical ACK. The observed timings are smoke diagnostics, not benchmarks.

The default evaluation command was also run and correctly refused to publish metrics because all 50 evaluation images overlap the development validation set by SHA-256.

## Release Gates

A release may be called **research/demo-ready** only when automated tests and evidence checks pass. It must not be called **production-ready** until all of the following are complete:

1. PLC vendor HIL verifies command sequencing, readback, timeout, duplicate, stale, and partial-channel failures.
2. Safety owner verifies hardwired E-stop/HOLD interlocks and recovery procedures.
3. Independent roll-disjoint data verifies model performance without leakage.
4. Factory calibration and sensor freshness are verified on target instruments.
5. Container build/runtime and rollback/backup procedures pass in the target environment.
6. Security review approves authentication, certificates, key rotation, RBAC, and OT segmentation.

## Next Execution Plan

1. **Next engineering slice**: verify API-backed dashboard telemetry against a running Compose stack and add browser smoke coverage.
2. **Next integration slice**: run a PLC simulator/HIL matrix for OPC UA/Modbus readback and failure modes.
3. **Next evidence slice**: create an immutable roll-disjoint manifest and rerun evaluation; publish only metrics produced by that manifest.
4. **Next deployment slice**: restore Docker daemon, build the locked image, run liveness/readiness checks as non-root, and document rollback.
5. **Release decision**: keep the classification at research prototype until every external gate above has attached evidence.
