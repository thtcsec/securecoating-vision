# 6-Minute Final Defense Pitch Script & Q&A Preparation Guide
**Competition:** 2026 Global AI + Materials Innovation Application Competition (全球AI+材料创新应用大赛)  
**Track:** Track 4 — AI + Materials Testing and Characterization (AI + 材料检测与表征)  
**Project:** SecureCoating-Vision — Multi-Sensor Fusion AI Platform for Inline Battery Electrode Inspection  
**Format:** 6-Minute Presentation + 2-Minute Q&A  

---

## 1. 6-Minute Presentation Deck & Timing Structure

```
+-------------------------------------------------------------------------------+
|  Minute 1: Problem & Industry Pain Point (Electrode Scrap & Defect Escapes)  |
|  Minute 2: Architectural Solution (Multi-Source Fusion RGB + Thermal + 3D)   |
|  Minute 3: Edge-to-Cloud Pipeline & ONNX Optimization                         |
|  Minute 4: Traceability Memory & OPC UA Industrial Integration                |
|  Minute 5: Quantitative Baseline & Failure Analysis                           |
|  Minute 6: Market Potential & Implementation Roadmap                           |
+-------------------------------------------------------------------------------+
```

### Minute 1: The Industrial Pain Point
*   **Opening:** "Good morning honorable judges and professors. I am Trịnh Hoàng Tú representing SecureCoating-Vision."
*   **Core Challenge:** In EV battery fabrication, electrode coating & drying account for over 20% of scrap costs. Sub-surface adhesion voids and micro-scratches propagate into severe thermal runaway risks during cell assembly.
*   **The Gap:** Traditional inspection uses single optical cameras with zero sub-surface visibility, treating defect detection as an isolated offline demo rather than an inline traceable quality decision system.

### Minute 2: Multi-Sensor Fusion Architecture
*   **Solution:** SecureCoating-Vision introduces a multi-modal inspection pipeline:
    1. **High-Res RGB Optical:** Captures surface cracks & contamination.
    2. **LWIR Thermal Camera:** Detects sub-surface thermal dissipation variations (voids & wet spots).
    3. **3D Laser Profilometer:** Gauges absolute coating thickness profiles.
*   **Alignment:** Digital homography alignment aligns spatial coordinates across modalities before feature fusion.

### Minute 3: Edge Computing & Low Latency Optimization
*   **Model Pipeline:** Powered by YOLOv8n-seg compiled into ONNX Runtime engine.
*   **Latency Positioning:** The repository contains modeled budgets and historical development artifacts, but no currently accepted hardware benchmark manifest. Do not quote GPU, end-to-end, or camera-to-ejector latency until the target-hardware run is hash-recorded.

### Minute 4: Industrial Traceability & Fail-Safe Integration
*   **Quality Memory:** Converts raw pixel masks into actionable industrial decisions. Every frame is tagged with `Batch_ID`, `Roll_ID`, and spatial offsets stored in a relational Quality Memory DB.
*   **Fail-Safe Behavior:** A required sensor, model, inference, database, calibration, or PLC fault latches/requests `HOLD`; the prototype does not claim uninterrupted production throughput.
*   **PLC Signaling:** Local demonstrations simulate command sequences. Physical action and ACK timing remain pending vendor PLC HIL.

### Minute 5: Quantitative Baseline & Failure Analysis
*   **Evidence Status:** "The tracked 50-image demonstration result is invalid as independent performance evidence because its images overlap a development validation split. Its JSON artifact reports box precision 50.0%, recall 66.67%, and F1 57.14% at IoU 0.50; these values must not be presented as factory generalization metrics."
*   **Next Gate:** Publish metrics only from an immutable roll-disjoint test manifest with model/dataset/commit hashes and confidence intervals.

### Minute 6: Market Impact & Business Strategy
*   **Deployment Strategy:** High-probability "Private Pilot" model—integrating as a software-overlay on existing industrial camera gateways.
*   **Commercial Model:** Implementation fee + annual site license for drift detection and quality memory analytics.
*   **Conclusion:** "SecureCoating-Vision bridges deep learning vision models with real-world factory control. Thank you, and I look forward to your questions."

---

## 2. Anticipated 2-Minute Q&A Defense Script

### Q1: "Are your LWIR Thermal and 3D Profilometer inputs based on real physical sensors or synthetic data?"
*   **Answer:** "For this competition prototype, the 2D optical pipeline runs on synthetic defect images generated procedurally with seed 42, while the secondary thermal and depth streams are simulated metadata feeds designed to validate our homography alignment and fail-safe degradation algorithms. In production, these will connect directly to GigE Vision LWIR and USB 3.0 Laser Profilometer hardware."

### Q2: "Why does Delamination currently have low recall on your demonstration evaluation subset?"
*   **Answer:** "Delamination patches in our procedural synthetic dataset exhibit high texture contrast variation, causing the model to misclassify them as large voids or blisters. This establishes an honest baseline and highlights the exact fine-tuning target needed when collecting live factory delamination samples."

### Q3: "How does the system ensure data privacy for sensitive manufacturing recipes?"
*   **Answer:** "All heavy raw image processing remains strictly within local edge gateway boundaries. Only aggregated defect metadata (counts, sizes, batch summaries) is transmitted to cloud registries over TLS-encrypted channels."
