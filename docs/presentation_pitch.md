# 6-Minute Final Defense Pitch Script & Q&A Preparation Guide
**Competition:** 2026 Global AI + Materials Innovation Application Competition (全球AI+材料创新应用大赛)  
**Track:** Track 4 — AI + Materials Testing and Characterization (AI + 材料检测与表征)  
**Competition title:** SecureCoating-Vision: Evidence-Gated Multimodal AI for Battery Electrode Inspection and Traceable Quality Decisions  
**Tagline:** Evidence-Gated Multimodal Inspection for Battery Electrode Manufacturing  
**Brand:** SecureCoating-Vision  
**Format:** 6-Minute Presentation + 2-Minute Q&A  

Do not introduce a fourth title. Present `SecureCoating-Vision_Final_Defense_6min.pptx` (git): clean typography title slide for Team 71 / HUFLIT — no organizer lab logo as product branding. Do not present leftover `*_Fixed.pptx` or `*_Evidence_Aware.pptx` decks.

**Slide 1 identity (small, contestant first):** Team 71 · Track 4; Trinh Hoang Tu · HUFLIT; Advisor line smaller — Prof. Kris Singh · Visiting Professor, Tsinghua University · Founder & CEO, SRII.

Organizer update: final materials deadline is **10 September 2026**. Play the two looping GIFs in minute 6 (Slideshow mode so GIF animates).

---

## 0. What judges must hear in 30 seconds

Say this, then stop talking until they look at the first defect image:

> Electrode coating defects are a **materials characterization** problem: a pinhole or scratch on a 1,200 m roll becomes scrap or an untraceable cell. Academic detectors answer *how to detect*. A coating line needs *when a detection is safe enough to act on*. SecureCoating-Vision is not a new YOLO. It is an **evidence gate**: PASS and REJECT only when the optical evidence, calibration, traceability write, and PLC ACK agree; otherwise **HOLD**.

| Judge look-for | Where it lives in 6 minutes | One-line proof |
|---|---|---|
| Problem framing | 0:00–0:35 | Scrap / escape vs unsafe automatic PASS |
| Novelty | 0:35–2:00 | Gate on top of YOLO + attributed LIBAD/DA-Core; HOLD is the contribution |
| Quantitative evidence | 2:45–4:20 | CoatingVision 88-image image-disjoint split: mAP50 0.634 / P 0.645 / R 0.642 / mAP50-95 0.354; model-only CPU inference from `reports/coatingvision_real_test_metrics.json`; not roll-disjoint; weights/dataset hashes shown |
| Materials relevance | throughout | Roll/batch identity, coating-surface frames, SPC on the coating process — not generic object detection |
| Industrial impact | 2:00–3:35 | Fail-closed HOLD; confirm-audit control; buy path shadow→HIL→decision assistance→per-line; **no factory yield / ROI claim** |
| Demo storytelling | 5:10–6:00 | Looping GIFs: RGB HOLD replay + LIBAD fixture PASS/REJECT/REJECT/HOLD |
| Reproducibility / limits | last 20 s + Q&A | DOI + model/dataset hashes + current test manifest; thermal/profiler simulated; roll-disjoint/HIL evidence pending |

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
*   **Lane A, already built:** two-class YOLO26n detector/ONNX on real CoatingVision RGB images, FastAPI, production operations dashboard, PASS/REJECT/HOLD, OPC UA/Modbus, HMAC certificates. Thermal and profilometry are **simulated interface adapters**.
*   **Lane B, validation extension:** LIBAD defines aligned VIS and inline-compatible X-rayL inputs plus 10 official splits. Stage line: "We extended validation to LIBAD's real aligned VIS and X-rayL data over its 10 official splits. The current adapter is an external validation experiment—not a claim of reproducing the authors' DA-Core paper results." Motivation (not a trophy metric): even a stronger anomaly detector still leaves an FPR operating point unsuitable for direct line authority — that is why detection is separated from disposition. Checked-in `reports/libad/official_local_adapter.json` is the local numpy adapter on official inputs (`comparable_to_paper: false`). Authors' DINOv2 one-seed interim (`reports/libad/official_dinov2_dacore_interim.json`) is **Q&A backup only**, not the main story. Paper table remains DINOv3/DA-Core (Sui et al.). Demo cases stay protocol fixtures for CI.
*   **One sentence of progress:** "RGB proved the software contract on real optical evidence. LIBAD extends the same inspection-and-disposition problem onto real aligned VIS+X-rayL. High false-positive rates are the reason HOLD exists — not a scoreboard win."
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
*   Physical PLC actuation and the SQLite final-state write are **not atomic**; if ACK succeeds but DB finalization fails, plant deployments need HIL/factory recovery (research-prototype limitation).
*   Certificates attach a tamper-evident **HMAC-SHA256 authenticated tag** to the canonical payload (API field `hmac_digital_signature` for compatibility; not a public-key digital signature).

### Minute 5: Why LIBAD Does Not End the Story
*   **Our RGB lane (research, not factory qualification):** CoatingVision fixed 88-image **image-disjoint** test split, seed 71, DOI 10.6084/m9.figshare.29260121.v1 — mAP50 0.634, precision 0.645, recall 0.642, mAP50-95 0.354; **not** factory roll-disjoint.
*   Show the checkpoint SHA-256 prefix and dataset-tree SHA-256 prefix from `reports/coatingvision_real_test_metrics.json` (currently weights `f72a8f2b…`, dataset-tree `d1db7823…`).
*   **Official LIBAD local adapter (motivation, not trophy):** mean multimodal AUROC / FPR95 over 10 official splits live in `reports/libad/official_local_adapter.json` — use them to show that anomaly scores alone are not safe line authority (`comparable_to_paper: false`; non-paper-comparable). Paper DA-Core still reports FPR95 54.3% at AUROC 86.7% — same lesson. Authors' DINOv2 interim one-seed smoke is Q&A backup only. The prediction artifact's 19,680 rows are evaluation rows across lanes, **not** 19,680 independent multimodal samples.
*   Local throughput evidence is presented only as measured inference timing. Camera exposure, transport, PLC ACK, and target-hardware HIL remain outside that number.

### Minute 6: Real Detection, Safe Demo Disposition
Let the two GIFs loop in Slideshow. Do not narrate every frame.
1. Left: CoatingVision `image_1548` optical → CLAHE → YOLO overlay → **HOLD**.
2. Right: LIBAD fixture **PASS / REJECT / REJECT / HOLD** (`comparable_to_paper: false`).
3. Close: "The model finds defects. The evidence gate controls when the line may act."

PyTorch and ONNX share the same two-class map. Development simulation + mock PLC + unverified calibration is why the RGB loop ends on HOLD, not PASS.

**Oral (~10–15 s) — compute / reproducibility (if asked or if closing needs one sentence):**
> Optional cloud and HPC channels were evaluated during development. The authoritative evidence path is independently reproducible on a local workstation: GPU-capable workloads ran on a laptop RTX 4050; the documented CPU inference benchmark is model-only timing measured separately. The prototype does not require competition-specific infrastructure.

---

## 2. Anticipated 2-Minute Q&A Defense Script

### Q1: "Are thermal and 3D sensors real?"
*   **Answer:** "No. Thermal and profilometry are simulated or injected adapters used to validate registration and fail-closed behavior. The real multimodal inputs are official LIBAD VIS plus inline-compatible X-rayL. We mounted that release, hashed the trees, ran all 10 official splits with a CPU numpy descriptor, and an authors' DINOv2 interim smoke. That is still not the textual PAPER_SPEC DINOv3 ViT-S/16 table. The 90-second demo cases remain protocol fixtures so CI stays deterministic."

### Q2: "Did you invent DA-Core?"
*   **Answer:** "No. DA-Core is the LIBAD authors' memory-bank baseline. Our contribution is the evidence-gated PASS/REJECT/HOLD layer that sits on top of those modality scores."

### Q3: "Your FPR is still high. Are you hiding it?"
*   **Answer:** "We publish FPR95. On official LIBAD the local numpy multimodal FPR95 is about 0.84; the authors' DINOv2 interim smoke is about AUROC 0.856 / FPR95 0.716 on one seed; the paper's DINOv3 setting is still 54.3%. The claim is software fail-closed semantics: uncertain or disagreed evidence cannot become an automatic PASS or REJECT. That is not yet a factory-qualified safe operating point — local adapter escape remains ~51% at ~5% HOLD."

*   **Likely Q:** "If HOLD is the answer, why is HOLD rate only ~5% while escape rate is ~51%?"
*   **Answer:** "That operating point is not a proposed plant set-point. It is an experimental gate policy used to evaluate PASS/REJECT/HOLD semantics on the local numpy adapter. The escape rate shows this threshold is not deployment-acceptable — which is why the repository remains a research prototype, not factory-qualified. Next work is a locked risk–coverage calibration curve."

*   **Likely Q:** "Is production Zero-Trust / mTLS / RBAC complete?"
*   **Answer:** "No. Production security hardening such as mTLS, RBAC, secret rotation, and OT segmentation remains on the industrialization roadmap. This artifact validates edge inspection, evidence, traceability, and fail-closed control contracts."

*   **Likely Q:** "Is the certificate HMAC a digital signature?"
*   **Answer:** "No. It is an HMAC-SHA256 authentication tag (shared-secret integrity), not a public-key digital signature. The JSON field remains `hmac_digital_signature` for API compatibility."

*   **Likely Q:** "What if the PLC ACK succeeds but the database write fails?"
*   **Answer:** "Physical actuation and SQLite finalization are not atomic in this prototype. Software latches HOLD and leaves the row PENDING so it cannot look like PASS, but a plant deployment still needs HIL/factory recovery for that race."

### Q4: "Is this just YOLO plus a few sensors?"
*   **Answer:** "YOLO localizes known surface defects. Thermal and the laser profiler are simulated adapters in this prototype. The contribution is the fail-closed decision layer: detection cannot self-release. That is central to the published Track 4 scope."

### Q4b: "Challenge 1 says performance prediction — where is it?"
*   **Answer:** "SecureCoating covers the inspection-to-quality-decision segment of that pipeline. Material or electrochemical performance prediction is the next validation stage once plant-linked property labels exist. We will not invent cell-performance numbers from coating images."

### Q4c: "What about your DINOv2 / DINOv3 numbers?"
*   **Answer:** "Backup only. The stage claim is LIBAD as external validation of the same disposition problem, not paper reproduction. Authors' DINOv2 one-seed interim and the paper DINOv3/DA-Core table stay in the reports; they are not the 45-second story."

### Q5: "Why not rebuild the model?"
*   **Answer:** "The bottleneck the paper itself names is not another detector. It is modality disagreement and closed-loop control. That is already the safety contract of this repository."

### Q6: "Who buys this, and how does it enter a factory?"
*   **Answer:** "Shadow inspection first, then PLC/HIL validation, then controlled decision assistance, then per-line deployment. The commercial model is per-line integration plus calibration, model monitoring, and traceability support — not a fake TAM slide."

### Q7: "Did you use competition computing resources?"
*   **Answer:** "Not for the final authoritative evidence path. We kept the release independently reproducible on commodity local hardware: GPU-capable workloads on a laptop RTX 4050, with the CPU inference benchmark measured separately as model-only timing. Cloud and HPC remain optional acceleration for larger future experiments, not a requirement to reproduce this prototype."

### Q7b: "Why not use them?"
*   **Answer:** "Access and environment constraints for some platforms—especially for an international participant—made a local, deterministic evidence path the safer choice for reproducibility. We do not frame that as organizer compute failure; cloud/HPC stay optional future acceleration. Do not name specific platforms as failing unless a judge asks."
