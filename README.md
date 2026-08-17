# SecureCoating-Vision

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](docker-compose.yml)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)](src/api/main.py)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-orange.svg)](dashboard/app.py)
[![Plotly 3D](https://img.shields.io/badge/3D%20Topography-Plotly-blueviolet.svg)](dashboard/app.py)
[![Standards](https://img.shields.io/badge/Standard-T%2FCIAPS%200006--2020-brightgreen.svg)](src/inference/electrode_metrology.py)

<p align="center">
  <img src="logo.png" alt="Tsinghua MSE Logo" width="450">
</p>

**Team ID:** 71  
**Team Leader:** Trịnh Hoàng Tú (HUFLIT)  
**Academic Supervisor:** Kris Singh — CEO, SRII; Visiting Professor, Tsinghua University; Adj. Professor of Practice, University of Newcastle; former executive at IBM, AMD, Intel, and National Semiconductor  
**Competition Context:** Prepared for the **2026 Global AI + Materials Innovation Application Competition (全球AI+材料创新应用大赛)**, hosted by Tsinghua University School of Materials Science and Engineering (清华大学材料学院).  
**Track:** Track 4 — AI + Materials Testing and Characterization (AI + 材料检测与表征)  
**Status:** **FINALS (全国总决赛入围)** — Defense Pitch: Late August 2026 (6-min presentation + 2-min Q&A)

> Evidence policy: the bundled model results are a prototype baseline. The repository now includes a reproducible real-data route using CoatingVision optical electrode images. RGB-only runs are explicitly held for QA; no calibrated 3D metrology, multimodal-fusion claim, or production release is inferred without those physical sensor inputs.

---

## ⚡ Executive Summary & Industrial System Architecture

**SecureCoating-Vision** is a production-grade, 7-stage multi-modal industrial inline inspection and traceable quality decision platform designed for high-speed lithium-ion battery electrode coating lines ($v = 1.8 - 2.5\text{ m/s}$).

Unlike generic computer vision demos that merely execute forward inference on static uploaded photos, **SecureCoating-Vision** orchestrates a complete 7-stage industrial workflow:

```
+-------------------------------------------------------------------------------------------------------------------------------+
|                                      SecureCoating-Vision: 7-Stage Industrial Pipeline                                        |
+-------------------------------------------------------------------------------------------------------------------------------+
| Stage 1: Continuous Web Handling & Line Encoder Sync (v=1.8 m/s, X-meter / Y-mm coordinate mapping)                           |
| Stage 2: Multi-Modal Physical Acquisition (Optical Darkfield/Brightfield, Traveling-Wave Thermography, 3D Confocal Profiler)  |
| Stage 3: Spatial Sub-Pixel Homography Warp & 5-Channel Fused Tensor Registration [R, G, B, T_diff, H_topo]                   |
| Stage 4: High-Throughput TensorRT / ONNX Instance Segmentation Engine (FP16, Gated Multi-Scale Feature Pyramid)              |
| Stage 5: Physics-Informed Battery Electrode Metrology (3D Volumetrics, Micro-Short Hazard Index, T/CIAPS 0006 Compliance)    |
| Stage 6: Multi-Tier Quality Decision Engine (Grade A/B/C/D & Sub-10ms Hardware Reject Gate Signaling via Modbus/OPC UA)       |
| Stage 7: Closed-Loop AI Root-Cause Diagnostics & Upstream Equipment Parameter Tuning Feedback (\Delta h, \Delta T, \Delta Q) |
+-------------------------------------------------------------------------------------------------------------------------------+
```

---

## 🏆 Key Architectural Innovations (Finals Benchmark)

### 1. Multi-Stage Industrial Workflow & Web Synchronizer
- **Quadrature Encoder Simulator:** Implements sub-pulse fractional residual accumulation to eliminate multi-kilometer drift.
- **Roll Lifecycle State Machine:** Formal transitions (`IDLE`, `LOADED`, `RUNNING`, `PAUSED`, `ROLL_CHANGE`, `COMPLETED`, `FAULT`).
- **1,200m Digital Twin Roll Map:** Tracks exact 2D defect coordinates ($X\text{ in meters along roll}, Y\text{ in mm across width}$) and slitting lane distributions (Lanes 1-4).

### 2. Deep Physics-Informed Battery Electrode Metrology
- **Micro-Short Circuit Hazard Index ($0.0 - 1.0$):** Evaluates defect protrusion height against the polyethylene separator safety barrier ($14\,\mu\mathrm{m}$). Flags critical cell puncture risk.
- **Areal Mass Loading Variance ($\mathrm{mg/cm^2}$):** Computes active material density fluctuations causing local overpotentials and lithium plating.
- **T/CIAPS 0006-2020 & QC/T 743 Compliance Engine:** Automatically audits surface defects, void areas, and delamination against national lithium-ion battery manufacturing standards.

### 3. AI Closed-Loop Equipment Diagnostics & Parameter Feedback
- **Doctor Blade / Slot-Die Coater:** Longitudinal scratches $\to$ Triggers ultrasonic slot-die wash and blade lateral servo indexing ($\Delta y = +2.5\text{ mm}$).
- **Planetary Slurry Mixer:** Cyclic voids $\to$ Recommends de-aerator vacuum increase ($\Delta P = -8.0\text{ kPa}$).
- **Multi-Zone Drying Oven:** Blistering / skinning delamination $\to$ Recommends Zone 1 temperature ramp reduction ($\Delta T_1 = -4.0^\circ\mathrm{C}$) and exhaust damper adjustment ($+8\%$).

### 4. SCADA-Grade Industrial Terminal (`dashboard/app.py`)
- **Interactive 3D Defect Topography:** Plotly 3D mesh reconstruction for micrometric surface inspection.
- **Continuous 1,200m Waterfall View:** Full web strip defect density heat map.
- **Industrial Telemetry:** Live Modbus TCP Holding Register hex table (40001-40016) and OPC UA Node Browser (`ns=2;s=Inspection/Verdict`).
- **Cryptographic Quality Certificate:** Issues SHA-256 signed digital certificates for zero-tamper automotive battery pack traceability (IATF 16949).

---

## 📊 Materials Datasets & Benchmark Foundations

To ensure realistic domain generalization across roll-to-roll materials manufacturing:
1. **CoatingVision Benchmark (2026):** Dedicated high-resolution battery electrode coating anomaly dataset.
2. **NEU Surface Defect Database (Northeastern University):** Roll-to-roll metal and foil strip surface defect benchmark (scratches, inclusions, crazing, patches).
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

This writes a source-traceable input, prediction overlay, and JSON record under `reports/external_coatingvision_demo/`. The 7-stage pipeline reports `DEGRADED / HOLD` for this RGB-only route by design.

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
