# 6-Minute Final Defense Pitch Script & Q&A Preparation Guide
**Competition:** 2026 Global AI + Materials Innovation Application Competition (全球AI+材料创新应用大赛)  
**Track:** Track 4 — AI + Materials Testing and Characterization (AI + 材料检测与表征)  
**Registered title:** SecureCoating Vision: A High-Throughput and Zero-Trust Edge-Cloud Pipeline for Inline Battery Electrode Defect Inspection and Traceable Quality Decisions  
**Tagline:** Evidence-Gated Multimodal Inspection for Battery Electrode Manufacturing  
**Brand:** SecureCoating-Vision  
**Format:** 6-Minute Presentation + 2-Minute Q&A  

Do not introduce a fourth title. Slide 1 of `SecureCoating-Vision_Final_Defense_6min.pptx` uses the registered finalist-list title exactly, then the tagline.

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
*   **Opening:** "Honorable judges, I am Trịnh Hoàng Tú. The registered title is High-Throughput and Zero-Trust Edge-Cloud Pipeline for inline battery electrode inspection. The tagline is Evidence-Gated Multimodal Inspection."
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
*   Production UI has three surfaces only: **Operate / Diagnose / Traceability**, filled from one `/api/operations/snapshot`.
*   Recipe sliders, defect injection, 7-stage demo, LIBAD evidence cases, and “send to PLC” live only in the explicit sandbox.
*   Operator E-stop, reset, and inference reset require API authentication, an exact confirmation phrase, and a durable control-audit row. Mock PLC remains labelled `SIMULATED`.
*   Every decision carries roll, batch, and part identity. Trace rows stay PENDING until PLC finalization.
*   Certificates HMAC-sign the canonical payload.

### Minute 5: Why LIBAD Does Not End the Story
*   Sui et al. report that even the best LIBAD setting still has **FPR95 54.3%** with AUROC 86.7%, AUPR 95.7%, F1-max 90.6%. They say this false-positive rate is too high for direct deployment.
*   The harness can record academic metrics on the 10 official splits and adds four SecureCoating metrics: Automatic Decision Coverage, HOLD Rate, Escape Rate, and Selective Risk. Those official-data numbers have not yet been generated in this repository.
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

### Q4: "Why not rebuild the model?"
*   **Answer:** "The bottleneck the paper itself names is not another detector. It is modality disagreement and closed-loop control. That is already the safety contract of this repository."
