# SecureCoating-Vision

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](docker-compose.yml)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)](src/api/main.py)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-orange.svg)](dashboard/app.py)

**Author:** Trịnh Hoàng Tú  
**Submission:** Submitted to the 2026 AI + Materials Competition (organized by Tsinghua University)  
**Track:** Track 4: AI + Materials Testing and Characterization  

> [!NOTE]
> This project is developed for the 2026 AI + Materials Competition. Computing resources may be adapted to supported platforms such as Volcengine, Alibaba Cloud, Paratera, Sugon, or Ctyun, subject to account availability and competition rules.

An industrial-grade, multi-source fusion vision inspection system designed to detect, classify, and track coating defects (e.g., voids, scratches, thin areas, and adhesion failures) in real-time. Built specifically to meet the high precision and robustness requirements of advanced coating manufacturing lines.

---

## 1. Detection Accuracy & Efficiency (30% of Evaluation)
This system achieves high detection accuracy and low-latency processing through a highly optimized dual-stage inference pipeline.

### Target Performance Metrics
*   **Defect Detection Accuracy (mAP@0.5:0.95):** $\ge 92.5\%$ across all critical defect classes (Scratch, Void, Blister, Delamination, Under-coating).
*   **Defect Recall Rate:** $\ge 98.2\%$ for critical safety defects, ensuring zero escapes to the next manufacturing step.
*   **Inference Latency:** $\le 35\text{ ms}$ per inspection zone (combining sensor calibration, fusion, and model execution).
*   **Inspection Throughput:** Fully supports production line speeds of up to $2.0\text{ m/s}$ with continuous rolling inspection.

### Architectural Optimizations
*   **TensorRT & ONNX Runtime Acceleration:** Models are compiled to TensorRT engines with FP16/INT8 quantization, ensuring deterministic execution times.
*   **Asynchronous Processing Pipeline:** Frame capture, multi-source alignment, inference, and industrial PLC signaling run in separate, non-blocking threads.
*   **Automated Triggering:** Integrated hardware synchronization via digital input pulses (optical encoder triggers) ensures frames are captured at exact physical intervals.

---

## 2. Technical Completeness & Robustness (25% of Evaluation)
Designed for continuous $24/7$ factory operation with proactive health checks, automatic error recovery, and robust exception handling.

### Robustness & Fault Tolerance
*   **Sensor Disconnection Fail-Safe:** If any secondary sensor (e.g., thermal or 3D profiler) fails or disconnects, the system auto-degrades to single-source optical detection with reduced features, flags a warning, and continues inspection instead of crashing.
*   **Automated Calibration Recovery:** Built-in spatial recalibration detects camera/sensor shifting caused by mechanical vibrations and applies digital homography correction on the fly.
*   **Data Validation & Quality Gates:** All inputs are passed through statistical anomaly filters to reject frames affected by severe lens glare, dust accumulation, or sudden illumination changes.

### Production-Ready Logging & Diagnostics
*   **Syslog-Compatible Logging:** Log levels are dynamically controlled. Critical errors automatically raise sirens on the dashboard and trigger PLC alert flags.
*   **Automatic Crash Recovery:** Containers are monitored by systemd/Docker restart policies to guarantee sub-second startup after unexpected OS interrupts.

---

## 3. Multi-Source Data Fusion (20% of Evaluation)
Coatings can have sub-surface defects (like voids) invisible to optical cameras, or surface scratches invisible to thermal cameras. This system fuses three complementary sensing modalities:

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

### Fusion Methodology
1.  **Optical RGB (High Spatial Resolution):** Captures fine-grain surface cracks, scratches, and dust contamination.
2.  **Long-Wave Infrared - LWIR (Sub-Surface Thermal):** Captures heat dissipation variations to identify sub-surface adhesion voids, wet spots, and thickness deviations.
3.  **3D Profilometer (Height & Topography):** Measures precise coating thickness profiles and detects physical height variations (blisters/bumps).
4.  **Hybrid Fusion Network:** Combines modalities at both the **Pixel-Level** (concatenation after spatial registration) and **Feature-Level** (cross-attention maps inside the deep learning backbone) to ensure high detection accuracy under varying conditions.

---

## 4. Industrial Application Adaptability (10% of Evaluation)
Our system is built to fit seamlessly into existing Smart Factory ecosystems.

### Protocols & Connectivity
*   **OPC UA & Modbus TCP:** Natively communicates inspection decisions directly to PLCs (Siemens S7, Beckhoff, etc.) to trigger physical reject sorting gates within $100\text{ ms}$ of inspection.
*   **REST API Integration:** Exposes standard endpoints for MES (Manufacturing Execution Systems) to query real-time quality parameters, current batch details, and historical data.

### Traceability & Maintenance
*   **Quality Memory Module:** Tracks rolling defects and uses statistical process control (SPC) to trigger preventive alerts before defects exceed control limits (e.g., notifying maintenance if nozzle clog patterns are identified).
*   **Low-Overhead Calibration:** Simple, semi-automated UI tools allow operators to recalibrate sensor alignment targets in less than 5 minutes during line changeovers.

---

## 5. Demo and Submission Package (15% of Evaluation)
This repository is packaged for rapid evaluation, featuring a clean local setup, containerized services, and an interactive QA dashboard.

### Submission Package Checklist
*   `README.md`: Direct mapping to competition tracks and scores.
*   `docs/`: Deep-dive architectural drawings and workflows.
*   `src/`: Modular Python pipeline implementing model inference, training, evaluation, and industrial communication.
*   `dashboard/`: Streamlit interactive dashboard showing real-time defect maps, multi-source inputs, and quality graphs.
*   `configs/`: Configurable thresholds, network parameters, and inference settings.
*   `docker-compose.yml`: Zero-config startup template.

### Quick Start Guide

#### 1. Running Locally (Development Mode)
```bash
# Clone the repository
git clone https://github.com/thtcsec/securecoating-vision.git
cd securecoating-vision

# Install dependencies
pip install -r requirements.txt

# Start the REST API
python src/api/main.py

# In a separate terminal, start the Streamlit Dashboard
streamlit run dashboard/app.py
```

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
