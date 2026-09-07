# Industrialization path

SecureCoating-Vision is a Track 4 research prototype by **Team 71 / Trinh Hoang Tu (HUFLIT)**, with academic advising from Prof. Kris Singh. This note states what is validated today and how the work would move toward industrial use. It is not a factory qualification certificate and does not invent revenue figures.

## Current validated prototype

Already demonstrated in software with immutable evidence in this repository:

- **RGB inspection** on real Argonne CoatingVision optical images (fixed image-disjoint test split; see `reports/coatingvision_real_test_metrics.json`).
- **Evidence / readiness gate** that blocks automatic release when sensors, calibration, inference, or communication are not ready.
- **PASS / REJECT / HOLD** disposition semantics (HOLD is a controlled software state, not an inferred PASS).
- **PLC command / ACK contract** verified in software (unique command sequence; matching ACK; mock I/O labelled `SIMULATED`).
- **Traceability** hooks for roll/batch/part identity and HMAC-SHA256 authenticated certificate snapshots.
- **LIBAD external multimodal validation** as an adapter lane (local numpy adapter and authors' DINOv2 interim remain `comparable_to_paper: false`).
- **Reproducibility manifests** (`reports/test_manifest.json`, `reports/submission_manifest.json`, dataset/model hashes).

The current artifact implements inspection, evidence gating, and fail-closed contracts. Production security hardening such as mTLS, RBAC, secret rotation, and OT segmentation remains part of the industrialization roadmap and is not claimed complete.

## Deployment path

### Phase 1 — HIL

Connect a real industrial PLC, production-representative camera, and line simulator. Prove command ownership, ACK timing, interlock latching, and fail-closed behavior against vendor hardware — not only unit tests.

### Phase 2 — pilot line

Collect an independent **roll-disjoint** dataset, complete plant calibration, and run a supervised pilot beside the existing inspection stack. Publish only hash-verified metrics from that split. No automatic production disposition without operator/process ownership.

### Phase 3 — production

MES integration, edge inference appliance hardening, mTLS/RBAC and OT network segmentation, and safety validation appropriate to the host plant. Software E-stop remains a software request path until a safety-rated hardwired stop is engineered and certified by the plant owner.

## Industrial value

- Reduce **untraceable AI decisions** by binding every disposition to evidence and identity.
- Prevent **low-confidence automatic release** when readiness contracts fail.
- Retain **auditable evidence** for quality disposition, audit, and post-event review.

## Business model / technology transfer

A realistic transfer path is a **software + edge inference appliance + integration layer**, or licensing/integration with battery electrode coating lines through an existing automation partner. Commercial terms would follow successful HIL and pilot gates; this repository does not claim ARR, yield savings, or a sold product.

## Explicit non-claims

Not claimed here: factory qualification, roll-disjoint validation of the current RGB headline metric, physical safety-rated E-stop, completed production Zero-Trust controls, or electrochemical performance prediction.
