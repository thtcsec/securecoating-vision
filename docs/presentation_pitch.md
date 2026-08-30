# 6-Minute Final Defense Pitch Script & Q&A Preparation Guide
**Competition:** 2026 Global AI + Materials Innovation Application Competition (全球AI+材料创新应用大赛)  
**Track:** Track 4 — AI + Materials Testing and Characterization (AI + 材料检测与表征)  
**Registered title:** SecureCoating Vision: A High-Throughput and Zero-Trust Edge-Cloud Pipeline for Inline Battery Electrode Defect Inspection and Traceable Quality Decisions  
**Tagline:** Evidence-Gated Multimodal Inspection for Battery Electrode Manufacturing  
**Brand:** SecureCoating-Vision  
**Format:** 6-Minute Presentation + 2-Minute Q&A  

Do not introduce a fourth title. Slide 1 of `SecureCoating-Vision_Final_Defense_6min.pptx` uses the registered finalist-list title exactly, then the tagline.

Use that deck for judges. `*_Evidence_Aware.pptx` still reads as multi-sensor fusion; do not lead with that subtitle.

---

## 0. What judges must hear in 30 seconds

Say this, then stop talking until they look at the first defect image:

> Electrode coating defects are a **materials characterization** problem: a pinhole or scratch on a 1,200 m roll becomes scrap or an untraceable cell. Academic detectors answer *how to detect*. A coating line needs *when a detection is safe enough to act on*. SecureCoating-Vision is not a new YOLO. It is an **evidence gate**: PASS and REJECT only when the optical evidence, calibration, traceability write, and PLC ACK agree; otherwise **HOLD**.

| Judge look-for | Where it lives in 6 minutes | One-line proof |
|---|---|---|
| Problem framing | 0:00–0:35 | Scrap / escape vs unsafe automatic PASS |
| Novelty | 0:35–2:00 | Gate on top of YOLO + attributed LIBAD/DA-Core; HOLD is the contribution |
| Quantitative evidence | 2:45–4:20 | CoatingVision held-out: mAP50 0.63 / P 0.64 / R 0.64; CPU pipeline ~312 ms; LIBAD paper FPR95 **54.3%** / F1-max 90.6% as the deployment gap |
| Materials relevance | throughout | Roll/batch identity, coating-surface frames, SPC on the coating process — not generic object detection |
| Industrial impact | 2:00–3:35 | Fail-closed, false-accept blocked by HOLD, confirm-audit control, HMAC certificate; **no factory yield claim** |
| Demo storytelling | 5:10–6:00 | Acquired frame → model input → overlay → PASS/REJECT/HOLD |
| Reproducibility / limits | last 20 s + Q&A | DOI + hashes + 171 tests; thermal/profiler simulated; official LIBAD 4.84 GB run pending |

Do **not** say 99.4% mAP, ≤35 ms TensorRT, real thermal/laser plant instruments, or zero escapes.

---

## 1. 6-Minute Presentation Deck & Timing Structure

```
+-------------------------------------------------------------------------------------+
|  Minute 1: Problem — electrode scrap, escapes, and unsafe automatic release        |
|  Minute 2: Architecture — RGB contract first, then LIBAD-compatible evidence lane |
|  Minute 3: Evidence gate — HOLD turns uncertainty into a controlled industrial state|
|  Minute 4: Operations console, confirm-audit control, certificate, PLC contracts |
|  Minute 5: LIBAD numbers vs SecureCoating operational metrics                      |
|  Minute 6: Four-case demo close and honest roadmap                                 |
+-------------------------------------------------------------------------------------+
```

### Minute 1: The Industrial Pain Point
*   **Opening (30 s):** Use the paragraph in section 0. Then: "I am Trịnh Hoàng Tú, Team 71."
*   **Core Challenge:** Electrode coating defects drive scrap and can escape into cells. A detector that ranks anomalies well can still be unsafe to act on if false positives are high or modalities disagree.
*   **The Gap:** Academic detection answers *how to detect*. A factory needs *when a detection is safe enough to act on*.

### Minute 2: Two Evidence Lanes, One Safety Contract
*   **Lane A, already built:** YOLOv8-seg/ONNX on RGB, FastAPI, production operations dashboard, PASS/REJECT/HOLD, OPC UA/Modbus, HMAC certificates. Thermal and profilometry are **simulated interface adapters**.
*   **Lane B, validation extension:** LIBAD defines aligned VIS and inline-compatible X-rayL inputs plus 10 official splits. This repository implements the adapter and evidence gate; official data execution is still pending.
*   **One sentence of progress:** "The initial prototype validated the software and safety contract using RGB inspection and simulated secondary modalities. We then added a LIBAD-compatible validation harness; the checked-in run is a protocol fixture, not a paper-comparable result."
*   **Not a topic change.** YOLO stays. DA-Core stays attributed to Sui et al.

### Minute 3: The Evidence Gate
*   Model output cannot self-release production.
*   Modality disagreement, stale sensor, invalid calibration, failed traceability write, or mismatched/stale PLC ACK → **HOLD**.
*   **PASS** only when evidence and the communication contract both pass.
*   This is industrial AI: uncertainty becomes an operationally controlled state, not a hidden false-positive rate.

### Minute 4: Operations Console, Traceability, and Control-Plane
*   Production UI: **Guide / Operate / History / Dataset / Diagnose / Traceability**, filled from one `/api/operations/snapshot`.
*   History is the demo sentence: acquired frame → letterboxed model input → detection overlay → gate action.
*   Recipe sliders, defect injection, 7-stage demo, LIBAD evidence cases, and “send to PLC” live only in the explicit sandbox.
*   Operator E-stop, reset, and inference reset require API authentication, an exact confirmation phrase, and a durable control-audit row. Mock PLC remains labelled `SIMULATED`.
*   Every decision carries roll, batch, and part identity. Trace rows stay PENDING until PLC finalization.
*   Certificates HMAC-sign the canonical payload.

### Minute 5: Why LIBAD Does Not End the Story
*   **Our RGB lane (research, not factory qualification):** CoatingVision held-out test-split, seed 71, DOI 10.6084/m9.figshare.29260121.v1 — mAP50 0.63, precision 0.64, recall 0.64. Live CPU mean pipeline latency ~312 ms.
*   Sui et al. report that even the best LIBAD setting still has **FPR95 54.3%** with AUROC 86.7%, AUPR 95.7%, F1-max 90.6%. They say this false-positive rate is too high for direct deployment.
*   Place those numbers **side by side**: a strong F1 does not authorize automatic release. The harness adds Automatic Decision Coverage, HOLD Rate, Escape Rate, and Selective Risk. Official-data numbers for those four have not yet been generated in this repository.
*   We do not hide a high FPR. We measure how much of it is converted into HOLD.

### Minute 6: Four Cases, One Closing Screen
1. Normal agreement → PASS
2. Surface-visible defect → REJECT, surface evidence
3. Internally visible X-ray anomaly → REJECT, complementary X-ray evidence
4. Near-threshold modality disagreement → HOLD, manual QA
*   Closing screen: Roll ID, Batch ID, Part ID, modality scores, calibration state, model hash, decision reason, certificate signature, PLC state.
*   **Close:** "LIBAD detects. SecureCoating-Vision decides when detection is safe enough to act on."

---

## 2. Anticipated 2-Minute Q&A Defense Script

### Q1: "Are thermal and 3D sensors real?"
*   **Answer:** "No. Thermal and profilometry are simulated or injected adapters used to validate registration and fail-closed behavior. The target real multimodal inputs are official LIBAD visible-light plus inline-compatible X-rayL, but the checked-in demo is a protocol fixture and official-data validation is pending."

### Q2: "Did you invent DA-Core?"
*   **Answer:** "No. DA-Core is the LIBAD authors' memory-bank baseline. Our contribution is the evidence-gated PASS/REJECT/HOLD layer that sits on top of those modality scores."

### Q3: "Your FPR is still high. Are you hiding it?"
*   **Answer:** "We publish FPR95. The industrial claim is not that FPR vanished. It is that uncertain or disagreed evidence cannot become an automatic PASS or REJECT. HOLD is the controlled state."

### Q4: "Is this just YOLO plus a few sensors?"
*   **Answer:** "YOLO localizes known surface defects. Thermal and the laser profiler are simulated adapters in this prototype. The contribution is the fail-closed decision layer: detection cannot self-release. That is the materials-testing problem Track 4 actually grades."

### Q5: "Why not rebuild the model?"
*   **Answer:** "The bottleneck the paper itself names is not another detector. It is modality disagreement and closed-loop control. That is already the safety contract of this repository."
