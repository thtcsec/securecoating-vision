# SecureCoating-Vision

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](docker-compose.yml)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)](src/api/main.py)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-orange.svg)](dashboard/app.py)

<p align="center">
  <img src="logo.png" alt="Tsinghua MSE Logo" width="450">
</p>

**Author:** Trịnh Hoàng Tú (HUFLIT)  
**External Advisor:** Kris Singh — Founder and CEO, SRII; Visiting Professor, Tsinghua University  
**Competition Context:** Prepared for the **2026 Global AI + Materials Innovation Application Competition (全球AI+材料创新应用大赛)**, hosted by Tsinghua University School of Materials Science and Engineering (清华大学材料学院) and co-hosts.  
**Track:** Track 4 — AI + Materials Testing and Characterization (AI + 材料检测与表征)  
**Submission Deadline:** July 31, 2026 (Final Defense: Late August 2026 — 6-min pitch + 2-min Q&A)

> [!NOTE]
> This project is developed for the 2026 Global AI + Materials Competition. Computing resources are aligned with official competition power supporters, including Volcengine (北京火山引擎), Alibaba Cloud (阿里云), Paratera (并行科技), Sugon (曙光智算), and Ctyun.

A production-inspired prototype for a multi-source fusion vision inspection system, designed to explore and demonstrate the feasibility of detecting, classifying, and tracking coating defects (e.g., voids, scratches, thin areas, and adhesion failures) in real-time.

---

## 1. Detection Accuracy & Efficiency (30% of Evaluation)
This project targets high detection accuracy and low-latency processing through a planned dual-stage inference pipeline.

### Target Performance Metrics (Design Specifications)
*   **Defect Detection Accuracy (mAP@0.5:0.95):** Target of $\ge 92.5\%$ across critical defect classes (Scratch, Void, Blister, Delamination, Under-coating) under evaluation.
*   **Defect Recall Rate:** Target of $\ge 98.2\%$ for critical safety defects, designed to reduce defect escape risk to near-zero before downstream cell assembly steps.
*   **Inference Latency:** Design target of $\le 35\text{ ms} $ per inspection zone (combining sensor calibration, fusion, and model execution).
*   **Inspection Throughput:** Aimed at supporting production line speeds of up to $2.0\text{ m/s}$ under simulated continuous rolling inspection constraints.

### Proposed Architectural Optimizations
*   **TensorRT & ONNX Runtime Acceleration:** Planned model compilation to TensorRT engines with FP16/INT8 quantization to ensure deterministic execution times.
*   **Asynchronous Processing Pipeline:** Frame capture, multi-source alignment, inference, and simulated industrial signaling are structured to run in separate, non-blocking threads.
*   **Automated Triggering:** Designed for hardware synchronization via digital input pulses (simulated optical encoder triggers) to ensure frames are captured at exact physical intervals.

---

## 2. Technical Completeness & Robustness (25% of Evaluation)
Designed as a production-inspired prototype with proactive health checks, automated error handling, and robust exception processing.

### Robustness & Fault Tolerance (Prototype Design)
*   **Sensor Disconnection Fail-Safe:** If any secondary sensor (e.g., thermal or 3D profiler) fails or disconnects, the prototype auto-degrades gracefully to single-source optical detection with reduced features, logs a diagnostic warning, and continues processing.
*   **Automated Calibration Recovery:** The system design supports spatial recalibration to detect camera/sensor shifting caused by mechanical vibrations and applies digital homography correction on the fly.
*   **Data Validation & Quality Gates:** Preprocessing includes statistical anomaly filters to reject frames affected by severe lens glare, dust accumulation, or sudden illumination changes.

### Logging & Diagnostics
*   **Diagnostic Logging:** Log levels are dynamically controlled. Simulated errors automatically raise warnings on the dashboard and log alert flags.
*   **Container Health Monitoring:** Containers are monitored by standard Docker restart policies to evaluate startup and recovery after unexpected interruptions.

---

## 3. Multi-Source Data Fusion (20% of Evaluation)
Coatings can have sub-surface defects (like voids) invisible to optical cameras, or surface scratches invisible to thermal cameras. This system is designed to support multi-source fusion across three complementary sensing modalities:

```
+------------------+     +--------------------+
| 2D High-Res RGB  | --> | Spatial Alignment  | \
+------------------+     +--------------------+  \
+------------------+     +--------------------+   \  +-----------------------+
|  LWIR Thermal    | --> | Homography Matrix  | ---->| Pixel & Feature Fusion| --> Defect Output
+------------------+     +--------------------+   /  +-----------------------+
+------------------+     +--------------------+  /
| 3D Laser Profile | --> | Height Calibration | /
+------------------+     +--------------------+
```

### Proposed Fusion Methodology
1.  **Optical RGB (High Spatial Resolution):** Captures fine-grain surface cracks, scratches, and dust contamination (active in current prototype).
2.  **Long-Wave Infrared - LWIR (Sub-Surface Thermal):** Designed to capture heat dissipation variations to identify sub-surface adhesion voids, wet spots, and thickness deviations (represented via simulated/metadata feeds in this prototype).
3.  **3D Profilometer (Height & Topography):** Designed to measure precise coating thickness profiles and detect physical height variations (blisters/bumps) (represented via simulated/metadata feeds in this prototype).
4.  **Hybrid Fusion Network:** Architecture structured to combine modalities at both the **Pixel-Level** (concatenation after spatial registration) and **Feature-Level** (cross-attention maps inside the deep learning backbone) to evaluate detection feasibility under multi-sensor conditions.

---

## 4. Industrial Application Adaptability (10% of Evaluation)
Our system is designed to demonstrate compatibility with standard Smart Factory ecosystems.

### Protocols & Connectivity
*   **Simulated Industrial I/O:** PLC integration (OPC UA & Modbus TCP) is represented through simulated signaling interfaces, demonstrating how inspection decisions would trigger physical reject sorting gates in production.
*   **REST API Integration:** Exposes standard endpoints for MES (Manufacturing Execution Systems) to query simulated real-time quality parameters, current batch details, and historical logs.

### Traceability & Diagnostics
*   **Quality Memory Module:** Logs inspection data to a local database and applies statistical process control (SPC) rules to flag potential anomalies, demonstrating how operators would receive early preventive warnings (e.g., simulated nozzle blockage trends).
*   **Calibration Interface:** The dashboard prototype includes manual target controls to demonstrate how operators would quickly recalibrate sensor alignment during line changeovers.

---

## 5. Demo and Submission Package (15% of Evaluation)
This repository is packaged for rapid evaluation, featuring a clean local setup, containerized services, and an interactive QA dashboard.

### Submission Package Checklist
*   `README.md` & `README_CN.md`: Direct mapping to competition tracks, deadlines, and scoring criteria.
*   `Al + Materials Competition Application Form.docx`: Official filled competition application form.
*   `docs/`: Architectural drawings, workflows, and `presentation_pitch.md` (6-Minute Final Defense Script + 2-Minute Q&A prep).
*   `src/`: Modular Python pipeline implementing ONNX inference, fail-safe degradation, traceability, and industrial PLC signaling.
*   `scripts/`: Automation scripts including `generate_synthetic_coating_defects.py` (contour-derived polygon generator with exact class quotas & 20% negative baseline samples) and `build_submission.py`.
*   `dashboard/`: Streamlit interactive dashboard showing real-time defect maps, multi-source inputs, and quality graphs.
*   `configs/`: Configurable thresholds, network parameters, and inference settings.
*   `docker-compose.yml`: Zero-config startup template.

### Quick Start Guide

#### 1. Running Locally (Development Mode)
```bash
# Clone the repository
git clone https://github.com/thtcsec/securecoating-vision.git
cd securecoating-vision

# Create / use project virtualenv on the project drive (recommended — avoids filling system disk)
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

# Install dependencies INTO .venv only
pip install -r requirements.txt
# GPU PyTorch (if needed):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Start the REST API
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
# Or: powershell -File scripts/run_api.ps1

# In a separate terminal (with .venv activated), start the Streamlit Dashboard
streamlit run dashboard/app.py
# Or: powershell -File scripts/run_dashboard.ps1
```

> **Important:** Always activate `.venv` before `pip install`. Do not install into the global/user Python site-packages (especially on Windows `AppData` on C:).
#### 2. Running in Production (Docker)
```bash
docker-compose up --build
```
Once started:
*   **Inference API:** Available at `http://localhost:8000` (interactive API documentation at `http://localhost:8000/docs`)
*   **Quality Dashboard:** Available at `http://localhost:8501`

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
