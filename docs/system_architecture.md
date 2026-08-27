# System Architecture: Implemented vs Target

SecureCoating-Vision is a single-process research prototype. The default Compose deployment deliberately uses one API worker because inference circuit state, PLC ownership, and the active roll ledger are held in process memory.

## Implemented software path

```text
HTTP image/demo input
  -> bounded image validation
  -> simulated or unavailable secondary-sensor fusion
  -> trained YOLO/ONNX inference (otherwise HOLD)
  -> mask post-processing and prototype metrology
  -> SQLite PENDING trace record
  -> one configured PLC command owner
  -> explicit command-sequence / ACK-sequence check
  -> finalized trace decision, or latched HOLD on failure
```

The dashboard is read-only with respect to the PLC. In normal mode it obtains health, active-roll, batch, SPC, PLC, passport, and certificate data from the API. An isolated local simulation fallback exists only when explicitly enabled in development/test.

## Not implemented or not verified

- No GigE Vision/LWIR/profilometer acquisition adapter or authoritative sensor timestamp/freshness source exists.
- No hardware encoder or deterministic multi-camera trigger integration is present.
- The tracked calibration artifact is explicitly unverified and simulation-only.
- TensorRT execution is not established by the current repository evidence.
- PLC execution semantics, physical gate timing, E-stop safety integrity, and reset behavior are not HIL-qualified.
- Active-roll state and PLC interlock state are not durable across process restart.
- SQLite backup, restore, retention, and disk-full recovery remain release gates.
- Multi-worker/multi-instance API deployment is unsafe until state and command ownership are externalized.

## Target production architecture

A production design needs a vendor-qualified acquisition service, an isolated inference worker with a killable deadline, a durable roll/event store, exactly one PLC command owner, mTLS/RBAC at the API boundary, OT segmentation, observability, and an independently tested safety PLC/hardwired stop chain. Those are roadmap requirements, not implemented features.
