# Evidence Mapping

This file maps claims only to implemented artifacts. It is not a competition score prediction.

| Area | Implemented evidence | Remaining gate |
|---|---|---|
| Detection | `src/inference/yolo_engine.py`, `src/inference/onnx_engine.py`, post-processing tests | Independent roll-disjoint evaluation, false-positive/false-negative suite, model parity |
| LIBAD validation extension | `src/libad/`, 10 official seeds, industrial gate metrics, hash-recorded local numpy 10-seed in `reports/libad/official_local_adapter.json` | Authors' DINOv3/DA-Core run if paper-table comparison is required; local numpy remains non-comparable |
| Fail-closed decisions | `src/inference/failsafe.py`, `src/libad/evidence_gate.py`, `src/api/main.py`, safety contract tests | Killable inference process and target-hardware fault injection |
| PLC signaling | `src/industrial/protocol_manager.py`; mock/negative ACK tests | Vendor PLC command/ACK mapping and physical HIL |
| Traceability | `src/traceability/quality_memory.py`, `web_synchronizer.py`, `roll_certificate.py` | Durable roll lifecycle, backup/restore, retention and restart recovery |
| Security | Production auth/config startup checks, bounded uploads, CORS/host allowlists | mTLS, RBAC, secret rotation, OT threat model and penetration review |
| Dashboard | Production snapshot console (`Operate`/`Diagnose`/`Traceability`); sandbox is opt-in | Browser/Compose smoke on the target host; do not start Docker if RAM is tight |
| Reproducibility | Direct dependency pins, deterministic training options, artifact validation | Fully resolved hash lock, SBOM, signed image and CI build |

The tracked demonstration evaluation overlaps development data and is not independent performance evidence. Do not use stale report values as proof of factory performance.
