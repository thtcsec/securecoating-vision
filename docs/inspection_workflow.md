# Inspection Workflow and Safety Contract

## Current request path

1. The API authenticates protected routes and bounds request/image size.
2. It validates the active batch and part identifier, then decodes an uploaded/demo image.
3. In development/test, secondary sensor maps may be simulated. In production, this repository has no real secondary-sensor adapter, so those sensors remain unavailable and automatic release is prohibited.
4. The fail-safe manager validates the RGB frame and runs inference behind a response deadline. A timeout returns promptly but cannot terminate an in-process native/GPU call; the circuit remains latched until an authenticated reset authorizes one recovery probe.
5. A missing/untrained model, invalid frame, sensor failure, unverified calibration, database failure, PLC interlock, unready PLC transport, stale evidence, or LIBAD modality disagreement selects `HOLD`.
6. The database first records `PENDING`. A unique `(batch_id, part_id)` claim prevents duplicate command issuance.
7. Exactly one configured PLC protocol owns commands. Software writes a unique command sequence and treats the operation as acknowledged only after the configured ACK sequence matches. Mock mode is always labelled `SIMULATED` and `acknowledged=false`.
8. The trace row is finalized after the PLC result. If finalization fails, `HOLD` is latched and the stored row cannot be mistaken for `PASS` because it remains `PENDING`.
9. The optional LIBAD evidence lane scores aligned VIS and X-rayL with a PatchCore/DA-Core memory baseline, then applies the same fail-closed contract. Detection cannot self-release. The production dashboard does not host this lane.
10. The production dashboard reads `/api/operations/snapshot` once per render. Operator E-stop, reset, and inference-reset go through `/api/operations/control` only after API-key authentication, an exact confirmation phrase, and a durable control-audit write.

## Important limits

- Bounding-box/mask dimensions are converted with configured calibration values, but the tracked calibration is not factory-verified.
- Prototype grading thresholds are engineering configuration, not GB/T, ISO, ASTM, or regulatory certification.
- Readback/ACK software tests do not prove physical actuator movement. PLC vendor HIL must test stale ACK, duplicate sequence, timeout, partial write, restart, network loss, and gate-position feedback.
- The timeout is a response deadline, not guaranteed compute cancellation. Production isolation should move inference into a supervised process/service that can be terminated and restarted safely.
