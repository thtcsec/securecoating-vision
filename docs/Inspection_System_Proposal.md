# Inspection System Proposal（检测系统方案）

**Project:** SecureCoating-Vision  
**Competition title:** SecureCoating-Vision: Evidence-Gated Multimodal AI for Battery Electrode Inspection and Traceable Quality Decisions  
**Team:** 71 · Trinh Hoang Tu · HUFLIT  
**Track:** 4 — AI + Materials Testing & Characterization  
**Role of this file:** Track 4 *Inspection System Proposal* identity for the competition archive. It does not invent new claims; it points judges to the authoritative documents and evidence already in the submission.

## 1. Problem

Battery electrode coating defects become scrap, rework, or untraceable cell risk. Academic detectors answer *how to detect*. A coating line needs *when a detection is safe enough to authorize a disposition*.

Detail: [docs/presentation_pitch.md](presentation_pitch.md), [docs/scoring_rubric_mapping.md](scoring_rubric_mapping.md)

## 2. Current architecture and boundary

The current prototype implements optical inspection, evidence gating, PLC command/ACK contracts in software, and traceability hooks. Production security hardening such as mTLS, RBAC, secret rotation, and OT segmentation remains part of the industrialization roadmap and is **not** claimed complete. The repository does **not** claim factory qualification.

Detail: [docs/system_architecture.md](system_architecture.md), [README.md](../README.md)

## 3. Inspection pipeline

1. Acquire optical frame (real CoatingVision evidence in this release).  
2. Run two-class detector (`surface_crack`, `delamination_crack`) via YOLO26n / ONNX.  
3. Apply post-processing and readiness checks.  
4. Emit **PASS / REJECT / HOLD** only through the evidence gate.  
5. Record roll/batch/part identity and certificate snapshot fields.

Detail: [docs/inspection_workflow.md](inspection_workflow.md)

## 4. Evidence / readiness gate

Model output cannot self-authorize release. Missing sensors, invalid calibration, inference faults, stale or unconfirmed industrial communication, traceability faults, or latched interlocks produce **HOLD**.

Detail: [docs/inspection_workflow.md](inspection_workflow.md), software contracts under `src/inference/` and `src/industrial/`

## 5. Industrial control boundary

- Real PLC path: unique command sequence + matching ACK; mock I/O labelled `SIMULATED`.  
- Software E-stop latches local interlock and requests PLC channels; it is **not** a safety-rated hardwired stop.  
- Operator confirm-audit actions are authenticated.

Detail: [docs/system_architecture.md](system_architecture.md), [docs/hardware_profiles.md](hardware_profiles.md)

## 6. Hardware and runtime profiles

Inference serving is hardware-adaptive (`edge` / `balanced` / `performance` / `auto`). Live UI must report the **active** runtime, not the requested intent. TensorRT is claimed only when the live card reports TensorRT.

Detail: [docs/hardware_profiles.md](hardware_profiles.md)

## 7. Validation evidence in this submission

| Lane | Authoritative artifact | Boundary |
|---|---|---|
| RGB detection | `reports/coatingvision_real_test_metrics.json` | mAP50 0.634 on a fixed 88-image public CoatingVision **image-disjoint** test split (seed 71; **not** factory roll-disjoint) |
| Inference timing | same report → `speed_ms_per_image.inference` (CPU model-only) | Model inference only; not camera/PLC/line throughput |
| Software tests | `reports/test_manifest.json`, `reports/pytest_*` | Authoritative software-test count, environment, and execution evidence are recorded in `reports/test_manifest.json` and `reports/pytest_*`; this document intentionally does not duplicate a mutable test count |
| Multimodal extension | `reports/libad/official_local_adapter.json` | Official-input local adapter over 10 splits (`comparable_to_paper: false`); use as motivation that FPR remains too high for unsupervised line authority — not a trophy metric |
| Demo loops | `reports/defense_gifs/` | Presentation artifacts from checked-in evidence |

### Performance-Prediction Scope Boundary

SecureCoating-Vision covers the **inspection-to-quality-decision** and **traceability** segment of the Track 4 materials-testing pipeline: image acquisition → defect/anomaly detection → evidence gate → PASS/REJECT/HOLD → traceability. It does **not** claim experimentally validated electrochemical or material performance prediction. Downstream property prediction is a future validation stage that requires plant-linked property labels; it is not a current defense metric and is not treated as a missing Track 4 requirement.

Status ledger: [docs/implementation_status.md](implementation_status.md)

## 8. Deployment roadmap

Phase 1 HIL → Phase 2 pilot (roll-disjoint + plant calibration) → Phase 3 production (MES, Zero-Trust controls, safety validation).

Detail: [docs/industrialization_path.md](industrialization_path.md)

## 9. Explicit limitations

Not claimed: factory qualification, roll-disjoint RGB headline validation, physical safety-rated E-stop, completed production Zero-Trust controls, paper-comparable LIBAD DINOv3 reproduction, or electrochemical performance prediction.

## Cross-reference map (BTC checklist)

| Track 4 ask | Where it lives in this ZIP |
|---|---|
| 大赛作品申报书 / Application Form | `AI+Materials_Competition_Application_Form.docx` |
| 检测系统方案 / Inspection System Proposal | **this file** |
| 算法模型 / Algorithm & model | `outputs/best.pt`, `outputs/model.onnx`, `src/inference/` |
| 测试数据集 / Test dataset | `data/coatingvision_real_test/` |
| 分析报告 / Analysis report | `reports/analysis_report.md` → authoritative metrics JSON |
| Industrialization narrative | `docs/industrialization_path.md` |
