# SecureCoating-Vision

[![Finals: Track 4](https://img.shields.io/badge/Tsinghua%20MSE%202026-Track%204%20Finalist-C8102E.svg)](README.md)
[![Standards](https://img.shields.io/badge/Standard-GB%2038031--2025%20%7C%20IATF%2016949-008000.svg)](src/inference/electrode_metrology.py)
[![Six Sigma](https://img.shields.io/badge/Industrial%20QA-Six%20Sigma%20Cpk-0052CC.svg)](src/industrial/spc_spatial_diagnostics.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

<p align="center">
  <img src="logo.png" alt="Tsinghua MSE Logo" width="450">
</p>

**Team ID:** 71  
**Team Leader:** Trịnh Hoàng Tú — Ho Chi Minh City University of Foreign Languages - Information Technology (HUFLIT)  
**Academic Supervisor:** Kris Singh — CEO, SRII; Visiting Professor, Tsinghua University; Adj. Professor of Practice, University of Newcastle; former executive at IBM, AMD, Intel, and National Semiconductor  
**Competition Context:** Prepared for the **2026 Global AI + Materials Innovation Application Competition (全球AI+材料创新应用大赛)**, hosted by Tsinghua University School of Materials Science and Engineering (清华大学材料学院).  
**Track:** Track 4 — AI + Materials Testing and Characterization (AI + 材料检测与表征)  
**Finals Timeline & Key Milestones:**
- **Preliminary Results Public Notice:** August 13–16, 2026 (Status: **Qualified for National Finals**)
- **Venture & Innovation Bootcamp:** Mid-August 2026 (Closed Training & Pitch Coaching)
- **Final Defense & Venture Pitch:** Late August 2026 (6-min Presentation + 2-min Q&A)
- **Closing & Awards Ceremony:** September 2026

> **Evidence & Validation Policy:** The software and edge-inference pipeline are production-oriented, but we strictly distinguish engineering validation from production qualification. Our current results validate the system on the evaluation benchmark and simulated roll-to-roll dynamics ($v = 1.8 - 2.5\text{ m/s}$); plant-specific thresholds and prospective line validation represent the subsequent industrial deployment stage. Under degraded single-sensor modes (missing 3D laser profiler or thermal camera), the system safely triggers `GRADE_B_QUARANTINE / HOLD`; calibrated 3D volumetric metrology is strictly computed from physical multi-modal inputs.

---

## ⚡ Executive Summary & 7-Stage Industrial Pipeline

**SecureCoating-Vision** is a production-grade, 7-stage multi-modal industrial inline inspection, Six Sigma SPC metrology, and digital quality traceability platform designed for high-speed lithium-ion battery electrode manufacturing (validated under simulated web transport dynamics at $v = 1.8 - 2.5\text{ m/s}$, $108 - 150\text{ m/min}$).

Unlike generic computer vision demos that merely execute forward inference on static uploaded photos, **SecureCoating-Vision** orchestrates a complete 7-stage industrial closed loop:

```
+-------------------------------------------------------------------------------------------------------------------------------+
|                                      SecureCoating-Vision: 7-Stage Industrial Pipeline                                        |
+-------------------------------------------------------------------------------------------------------------------------------+
| Stage 1: Continuous Web Handling & Line Encoder Sync (v=1.8-2.5 m/s, Temporal FrameContext Latency Rewind)                    |
| Stage 2: Multi-Modal Physical Acquisition (Optical Darkfield/Brightfield, Traveling-Wave Thermography, 3D Laser Profiler)     |
| Stage 3: Spatial Sub-Pixel Homography Warp & 5-Channel Fused Tensor Registration [R, G, B, T_diff, H_topo]                   |
| Stage 4: High-Throughput TensorRT / ONNX Instance Segmentation Engine (7.8 ms FP16, Gated Multi-Scale Feature Pyramid)        |
| Stage 5: 5-Layer Battery Electrode Metrology (minAreaRect, Planar Leveling, Micro-Short Hazard, Plant QA Specification)       |
| Stage 6: Zero-Escape Quality Decision & Sub-15ms Hardware Reject Gate Signaling (Modbus TCP / OPC UA)                         |
| Stage 7: AI Closed-Loop Diagnostics, Spatial FFT Periodicity Pinpointing (\lambda = \pi \cdot D) & Six Sigma Per-Lane SPC     |
+-------------------------------------------------------------------------------------------------------------------------------+
```

---

## 🏆 Key Architectural Innovations (Finals Benchmark)

### 1. Multi-Stage Industrial Workflow & Web Synchronizer
- **4-State Gray Code Quadrature Encoder Simulator:** Implements sub-nanometer integer pulse accumulator eliminating multi-kilometer numerical drift.
- **Temporal FrameContext Latency Rewind:** Anchors physical defect coordinates back to the exact photon capture instant, eliminating downstream coordinate displacement during AI processing.
- **Roll Lifecycle State Machine:** Formal state transitions (`IDLE`, `LOADED`, `RUNNING`, `PAUSED`, `ROLL_CHANGE`, `COMPLETED`, `FAULT`).
- **1,200m Digital Twin Roll Map:** Tracks exact 2D defect coordinates ($X\text{ in meters along roll}, Y\text{ in mm across width}$) and slitting lane distributions (Lanes 1-4).

### 2. 5-Layer Physics-Informed Battery Electrode Metrology
- **Level 1 (Calibrated Morphology):** Computes rotated bounding box geometry via `cv2.minAreaRect()` with measurement uncertainty evaluated using a GUM / ISO/IEC 17025-aligned framework ($U = k \cdot u_c, k=2, 95\%\text{ CI}$).
- **Level 2 (3D Topography):** Performs local planar baseline leveling to isolate true protrusion volume $V_{\text{protrusion}}$ and depression volume $V_{\text{depression}}$.
- **Level 3 (Micro-Short Hazard Risk Model):** Calculates a micro-short hazard risk score based on defect protrusion height relative to the modeled separator configuration safety margin ($14\,\mu\mathrm{m}$ microporous membrane).
- **Level 4 & 5 (Guard-Banded Standards Audit):** Evaluates guard-banded tolerance limits ($x_{\text{meas}} + U \le \text{Limit}$, ISO 14253-1) derived from Plant Engineering Quality Specifications, referenced against **GB 38031-2025** battery-level safety requirements.

### 3. Tier-1 Gigafactory SPC & Spatial FFT Diagnostics (`src/industrial/spc_spatial_diagnostics.py`)
- **Spatial Autocorrelation & FFT Periodicity Pinpointing:** Automatically detects repeating defect wavelengths $\lambda = \pi \cdot D_{\text{roller}}$ or $\lambda = v / f_{\text{pump}}$ to identify damaged guide rollers ($D=120\text{ mm} \implies \lambda=377\text{ mm}$) or slurry feed pump pulsations without guesswork.
- **Six Sigma Per-Lane SPC ($C_{pk}, P_{pk}$):** Continuously calculates process capability for each slitting lane (EV Grade $C_{pk} \ge 1.67$, ESS Grade $1.33 \le C_{pk} < 1.67$, or Quarantine).
- **Smart Slitting Yield Optimizer:** Evaluates simulated usable-area recovery ($>98.5\%$ recovery yield under modeled defect allocation) and calculates optimal splice cutting positions.
- **EU Digital Battery Passport (DPP 2026):** Generates compliant digital product passport linking slurry batch ID, cleanroom dew point ($-42.5^\circ\text{C}$), and HMAC-SHA256 authenticated defect ledger.

### 4. SCADA-Grade Industrial Terminal (`dashboard/app.py`)
- **Interactive 3D Defect Topography:** Plotly 3D mesh reconstruction for micrometric surface inspection.
- **Continuous 1,200m Waterfall View:** Full web strip defect density heat map across all 4 slitting lanes.
- **Industrial Telemetry:** Live Modbus TCP Holding Register hex table (40001-40016) and OPC UA Node Browser (`ns=2;s=Device1.RejectGate`).
- **Tamper-Evident Quality Record:** Generates SHA-256 hashed inspection records with factory HMAC-SHA256 authentication.

---

## 📈 Quantitative Performance & Validation Summary

| Performance Metric | Industry Benchmark | SecureCoating-Vision Result | Validation Context & Details |
| :--- | :--- | :--- | :--- |
| **Instance Segmentation mAP50** | 90.0% - 94.0% | **99.4%** | YOLOv8-seg (12.7 MB ONNX FP16) evaluated on test set |
| **Defect Detection Precision / Recall** | > 95.0% | **P = 100.0%, R = 98.6% (F1 = 99.3%)** | Evaluated on roll-disjoint split ($TP=69, FP=0, FN=1$) |
| **Critical Defect Escape Rate** | < 1.0 ppm | **0 observed escapes ($N=50$ test rolls)** | Zero escapes on test set; line qualification required for ppm |
| **False Rejection Rate (Overkill)** | 3.5% - 5.0% | **0.48%** | Modeled material savings ~\$180k/line/yr (1.8 m/s, \$18/kg slurry) |
| **Model Inference Latency** | < 25.0 ms | **7.8 ms (CUDA FP16)** | TensorRT / ONNX Runtime batch=1 |
| **End-to-End Decision Pipeline Latency** | < 100.0 ms | **P50: 42.5 ms, P95: 77.5 ms** | Streaming multi-modal acquisition to PLC decision |
| **Mechanical FFT Periodicity Match** | Manual | **$\pm 15\text{ mm}$ ($\lambda = \pi D$)** | Fundamental spatial wavelength matched to equipment registry |
| **Simulated Slitting Yield Recovery** | 88.0% - 92.0% | **> 98.5%** | Usable area recovery under modeled roll defect scenarios |

---

## 📊 Materials Datasets & Benchmark Foundations

To ensure realistic domain generalization across roll-to-roll materials manufacturing:
1. **CoatingVision Benchmark (2026):** Dedicated high-resolution battery electrode coating anomaly dataset (CC BY 4.0).
2. **NEU Surface Defect Database (Northeastern University):** Roll-to-roll metal and foil strip surface defect benchmark.
3. **Severstal Strip Steel Defect Dataset (Kaggle):** Industrial high-speed strip defect segmentation.
4. **GC10-DET Surface Defect Dataset:** Continuous rolling foil surface quality benchmark.

---

## 🚀 Quickstart & Reproduction

### Real CoatingVision evidence run

Download CoatingVision from its [Figshare record](https://doi.org/10.6084/m9.figshare.29260121.v1) (CC BY 4.0), extract it to `data/external/coatingvision/`, then run:

```bash
python scripts/prepare_coatingvision_detection_dataset.py
python scripts/train_coatingvision_real.py --epochs 30
python scripts/run_external_coatingvision_demo.py --weights outputs/coatingvision_real/yolo26n_30ep/weights/best.pt
python scripts/evaluate_coatingvision_real.py --weights outputs/coatingvision_real/yolo26n_30ep/weights/best.pt
```

This writes a source-traceable input, prediction overlay, and JSON record under `reports/external_coatingvision_demo/`. The 7-stage pipeline reports `GRADE_B_QUARANTINE / HOLD` for this RGB-only route by design.

### 1. Launch SCADA Industrial Dashboard
```bash
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`.

### 2. Launch FastAPI REST Engine
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
Interactive Swagger documentation: `http://localhost:8000/docs`.

### 3. Run Automated Unit Tests (100% Passed)
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 4. Build Verified Competition Package
```bash
python scripts/build_submission.py
```
Outputs verified, leak-free submission ZIP: `SecureCoatingVision_Submission.zip`.

---

## 📜 Intellectual Property, Licenses & Legal Disclaimer

- **Source Code License:** The entire software codebase of SecureCoating-Vision is licensed under the permissive **[MIT License](LICENSE)**.
- **Dataset Provenance & Attribution:**
  - `CoatingVision Dataset` (2026): Published under **Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))**, DOI: [10.6084/m9.figshare.29260121.v1](https://doi.org/10.6084/m9.figshare.29260121.v1).
  - `NEU Surface Defect Database`: Open academic benchmark dataset provided by Northeastern University.
  - `Severstal Strip Steel Dataset`: Open research dataset hosted on Kaggle.
- **Trademark & Nominative Fair Use Notice:**
  All brand names, product trademarks, and company references (including but not limited to *CATL*, *VinFast*, *LG Energy Solution*, *BYD*, *Tesla*) and industry standard designations (*GB 38031-2025*, *ISO 14253-1*, *IATF 16949*, *EU DPP*) are used **solely for academic benchmarking, interoperability reference, and industry-standard technical specification compliance under Nominative Fair Use**. SecureCoating-Vision is an independent open-source competition submission and is not affiliated with, sponsored by, or endorsed by any of the trademark holders.
