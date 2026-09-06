# System Architecture: Implemented vs Target

SecureCoating-Vision is a single-process research prototype. The default Compose deployment deliberately uses one API worker because inference circuit state, PLC ownership, and the active roll ledger are held in process memory.

## Implemented software path

```text
HTTP image/demo input
  -> bounded image validation
  -> two-class RGB YOLO/ONNX detector, with simulated thermal/profilometry adapters
  -> optional LIBAD VIS + X-rayL evidence lane (PatchCore/DA-Core scores, authors' baseline)
  -> evidence gate: PASS / REJECT / HOLD
  -> SQLite PENDING trace record
  -> one configured PLC command owner
  -> explicit command-sequence / ACK-sequence check
  -> finalized trace decision, or latched HOLD on failure
```

The production dashboard consumes one `/api/operations/snapshot` response for line, quality, PLC, and traceability fields. Recipe sliders, defect injection, 7-stage simulation, and the LIBAD evidence lane exist only in the explicit development/test sandbox. Operator commands use `/api/operations/control` with API-key authentication, an exact confirmation phrase, and a durable audit row. The dashboard never writes equipment recipe offsets to a PLC.

## Not implemented or not verified

- No GigE Vision/LWIR/profilometer acquisition adapter or authoritative sensor timestamp/freshness source exists. Thermal and profilometry remain simulated interface adapters.
- LIBAD VIS + X-rayL is an external validation adapter. Official-data evidence for the local numpy adapter requires the 4.84 GB dataset, all 10 split files, and a recorded tree-hash manifest. That still is not the authors' DINOv3 implementation, so reports remain `comparable_to_paper: false`.
- DA-Core is not claimed as a SecureCoating-Vision algorithm.
- No hardware encoder or deterministic multi-camera trigger integration is present.
- The tracked calibration artifact is explicitly unverified and simulation-only.
- TensorRT execution is not established by the current repository evidence.
- PLC execution semantics, physical gate timing, E-stop safety integrity, and reset behavior are not HIL-qualified.
- Physical PLC actuation and the SQLite final-state write are not atomic. ACK success with a failed DB finalization latches HOLD / leaves the row `PENDING`, but plant recovery for that race remains a HIL/factory concern (research-prototype limitation).
- Active-roll state and PLC interlock state are not durable across process restart.
- SQLite backup, restore, retention, and disk-full recovery remain release gates.
- Multi-worker/multi-instance API deployment is unsafe until state and command ownership are externalized.

## Target production architecture

A production design needs a vendor-qualified acquisition service, an isolated inference worker with a killable deadline, a durable roll/event store, exactly one PLC command owner, mTLS/RBAC at the API boundary, OT segmentation, observability, and an independently tested safety PLC/hardwired stop chain. Those are roadmap requirements, not implemented features.
