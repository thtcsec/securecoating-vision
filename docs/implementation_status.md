# Implementation Status and Upgrade Plan

Last reviewed: 2026-09-02

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
- [x] A localhost Modbus TCP software loopback confirms command/readback/ACK over a real socket. This is not vendor PLC HIL.
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
- [x] CoatingVision public metadata was inventoried: class flags and filenames only, no factory roll/coil/web IDs, so a roll-disjoint split cannot be constructed from this release.
- [ ] Supply a real independent roll-disjoint manifest and publish metrics from it.
- [x] LIBAD adapter, evidence gate, four industrial metrics, and 10 official-seed harness exist in repository tests.
- [x] Official LIBAD Hugging Face zip files were probed; unauthenticated access returns HTTP 401 / gated until terms are accepted and `HF_TOKEN` is set.
- [x] Official LIBAD tree hashes for the local numpy adapter are recorded by `scripts/record_libad_official_manifest.py` (runtime manifest gitignored; copy in `reports/libad/official_mount_hashes.json`). Default status uses a cheap fingerprint so pytest/API do not SHA-256 4.84 GB on every call.
- [ ] Run the authors' DINOv3/DA-Core implementation on the hash-verified mount before publishing any paper-comparable metrics.
- [ ] Run clean-surface false-positive, hard-negative, and defect false-negative suites.
- [ ] Measure POD, escape rate, confidence intervals, and performance on independent factory data.
- [ ] Add ONNX-vs-source-model parity and multi-resolution calibration tests.

## Phase 4: Deployment and Operations

- [x] Docker uses Python 3.11, direct dependency pins, a non-root runtime user, dropped capabilities, and loopback bindings by default.
- [x] Compose uses one API worker because the current roll/PLC owner is in-process.
- [x] Liveness is separated from readiness/health.
- [x] Environment examples distinguish local simulation from production configuration.
- [x] Dashboard production surface is snapshot-only (line/quality/PLC/traceability) and does not render simulator controls.
- [x] Dashboard sandbox is an explicit development/test opt-in for simulator and LIBAD evidence demo.
- [x] Operator control uses authenticated confirm-audit workflow; recipe/PLC parameter writes are not offered.
- [x] Dashboard API client has bounded requests and explicit HTTP/JSON error handling.
- [x] Production dashboard refuses automatic local fallback; sandbox fallback is explicit development/test opt-in.
- [x] Inspection history retains bounded authenticated overlay artifacts, run-scoped detections, persisted decision context, and a paginated metadata-only dataset catalog.
- [x] Dashboard AppTest covers production snapshot console and isolated sandbox render.
- [x] Dependency resolver dry-run succeeds; Pillow/GitPython/Torch pins were advanced to audited fixed versions.
- [x] GPU development environment verified Torch 2.13.0+cu130 with the RTX 4050.
- [x] Docker Desktop build and runtime smoke passed on 2026-08-28 with Python 3.11.16, CPU inference, non-root/read-only containers, and loopback-only published ports.
- [x] GitHub Actions compiles sources, checks dependencies, and runs pytest; artifact-gated ONNX skips are allowed, unexplained skips are not.
- [ ] Add CI Docker image build and release artifact checks.
- [x] Manual browser/Compose smoke verified the API-backed dashboard and a real CoatingVision image through the built API image.
- [ ] Docker Scout still reports 2 Critical and 2 High findings in Debian's essential `perl-base` 5.40.1-6, with no fixed version reported; production release remains blocked pending a fixed/minimal base or documented security acceptance.
- [x] API inspection endpoints have bounded in-flight concurrency and reject excess work instead of building an unbounded queue.
- [ ] Add target-hardware operational metrics and alerting.

## Test Evidence Recorded

The latest repository validation is recorded by `scripts/record_test_manifest.py` and must not be typed by hand in multiple documents.

<!-- TEST_MANIFEST:START -->
```text
D:\tu_projects\securecoating-vision\.venv\Scripts\python.exe -m pytest -q
241 passed in 62.34s
python 3.11.9
commit 2a182d4e2f75d9b7dee62024d375da470cd62740
working_tree_dirty False
source_diff_sha256 None
log_sha256 93dd6b628e697324f77452195f14291c7aaac03c30721d52a0b31e1631bbf784
```
<!-- TEST_MANIFEST:END -->

The local live Uvicorn smoke run on port 8011 verified API liveness/readiness, clean PASS, real-image REJECT, degraded-sensor HOLD, duplicate-part HOLD, E-stop latch, simulated reset, and upload inspection. PLC statuses were `SIMULATED`, never physical ACK. The observed timings are smoke diagnostics, not benchmarks.

The Docker production-mode smoke used an isolated temporary database and real CoatingVision test image. It verified authenticated readiness returned 503 while sensors/PLC/calibration were unavailable, inference completed on CPU, the final action stayed `HOLD`, and PLC signal history recorded `acknowledged=false`. Docker Scout reduced from 5 Critical/43 High on the old base to 2 Critical/2 High on the current image; the remaining no-fix `perl-base` findings are an open release blocker, not a clean security result.

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

1. **Next evidence slice**: present the git 6-minute deck (`SecureCoating-Vision_Final_Defense_6min.pptx`) in Slideshow so the looping GIFs animate, and submit the rebuilt `SecureCoatingVision_Submission.zip` before the 10 September 2026 organizer deadline; authors' DINOv3 remains optional and non-blocking.
2. **Next engineering slice**: keep GitHub Actions green; add Compose/browser smoke only when the runner has enough RAM.
3. **Next integration slice**: vendor PLC HIL remains open; localhost Modbus command/ACK now runs through the live API (`scripts/run_modbus_loopback_trigger.py`).
4. **Next deployment slice**: replace or refresh the base when the remaining `perl-base` findings are fixable, generate an SBOM, and exercise rollback/backup on the target host.
5. **Release decision**: keep the classification at research prototype until every external gate above has attached evidence.
