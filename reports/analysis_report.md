# SecureCoating-Vision: Performance Analysis Report

**Project:** Multi-Sensor Fusion AI Platform for Inline Coating Defect Inspection  
**Track:** Track 4 — AI + Materials Testing and Characterization  
**Author:** Trịnh Hoàng Tú (HUFLIT)  
**Academic Supervisor:** Kris Singh (SRII; Visiting Professor, Tsinghua University)  
**Date:** July 2026  

---

## 1. Executive Summary

SecureCoating-Vision is structured as a **production-inspired prototype** designed to demonstrate the technical feasibility of real-time inline battery electrode inspection. On our included **demonstration evaluation subset** (`data/evaluation/`, 50 images, 30 GT instances), the ONNX model achieves **54.1% Precision**, **66.7% Recall**, and **59.7% F1-Score** at IoU≥0.50 threshold, with **44.21% Mask mAP@0.50** computed via official Ultralytics segmentation validation. ONNX inference latency is **8.7 ms on GPU (RTX 4050)** and **43.91 ms on CPU**.

> [!NOTE]
> All quantitative metrics below are directly reproducible by executing `python scripts/run_evaluation.py` or `python scripts/run_ultralytics_validation.py`. Results are automatically exported to `reports/evaluation_results.json`, `reports/evaluation_results.csv`, and `reports/ultralytics_validation_results.json`. Real-line factory deployment will require fine-tuning on physical inline sensor feeds.

---

## 2. Model Architecture & Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | YOLOv8n-seg (Nano segmentation) |
| Backbone | CSPDarknet53-Nano (Planned feature-level cross-attention extension) |
| Parameters | 3,258,844 |
| GFLOPs | 11.3 |
| Input Resolution | 640 × 640 |
| Training Epochs | 50 |
| Batch Size | 4 |
| Optimizer | AdamW (lr=0.00125, momentum=0.9) |
| Export Format | ONNX (opset 12) |
| Model Size | 12.7 MB (ONNX) |

---

## 3. Included Demonstration Evaluation Subset Description

| Subset Path | Total Images | Positive Samples | Negative Clean Samples | GT Instances | Source |
|-------------|--------------|------------------|------------------------|--------------|--------|
| `data/evaluation/` | 50 | 30 | 20 (40%) | 30 | Procedurally generated synthetic images |

### Class Breakdown (Demonstration Evaluation Subset)
| Class ID | Name | Ground Truth Instances | Description |
|----------|------|------------------------|-------------|
| 0 | Scratch | 6 | Linear surface damage, cracks |
| 1 | Void | 10 | Sub-surface pores, inclusions |
| 2 | Blister | 8 | Raised coating bumps |
| 3 | Delamination | 6 | Coating separation/peeling |

---

## 4. Reproducible Evaluation & mAP Metrics

### 4.1 Overall Metrics (`python scripts/run_evaluation.py`)

| Metric | Measured Value | Design Specification Target |
|--------|----------------|-----------------------------|
| Precision | 54.1% | ≥95.0% |
| Recall | 66.7% | Target near-zero defect escape |
| F1-Score | 59.7% | ≥96.5% |
| True Positives (TP) | 20 | — |
| False Positives (FP) | 17 | — |
| False Negatives (FN) | 10 | — |

### 4.2 Official Ultralytics mAP Validation (`python scripts/run_ultralytics_validation.py`)

| Metric Type | mAP @ 0.50 | mAP @ 0.50:0.95 |
|-------------|------------|------------------|
| Bounding Box Detection | 49.01% | 43.18% |
| Instance Segmentation Mask | 44.21% | 35.91% |

### 4.3 Per-Class Performance Breakdown

| Class | TP | FP | FN | Precision | Recall | Failure Case Analysis |
|-------|----|----|----|-----------|--------|-----------------------|
| Scratch | 2 | 0 | 4 | 100.0% | 33.3% | Thin 2px lines require fine confidence thresholding |
| Void | 10 | 6 | 0 | 62.5% | 100.0% | Excellent recall (100%), mild FP on dark noise spots |
| Blister | 8 | 4 | 0 | 66.7% | 100.0% | Excellent recall (100%), mild FP on glare highlights |
| Delamination | 0 | 7 | 6 | 0.0% | 0.0% | Contrast threshold mismatch on large patch polygons |

---

## 5. Inference & Pipeline Latency

| Benchmark Context | Provider | Mean Latency | Median | P95 | Throughput |
|-------------------|----------|--------------|--------|-----|------------|
| ONNX Inference Latency (GPU) | CUDAExecutionProvider | 8.70 ms | 8.50 ms | 11.20 ms | 115 FPS |
| ONNX Inference Latency (CPU) | CPUExecutionProvider | 43.91 ms | 44.09 ms | 46.67 ms | 22.8 FPS |
| **End-to-End Pipeline (GPU)** | **CUDA + Postprocess + DB** | **12.50 ms** | **12.10 ms** | **15.00 ms** | **80 FPS** |

---

## 6. System Scope & Industrial API Endpoints

### 6.1 System Scope Breakdown
- **Implemented & Benchmarked:** RGB Optical YOLO Segmentation, Fail-safe degradation engine, OPC UA/Modbus TCP signal emulation, SQLite Traceability DB.
- **Emulated Framework:** Multi-sensor LWIR Thermal & 3D Profilometer height maps metadata fusion framework.

### 6.2 Industrial REST API Endpoints (FastAPI)
The system exposes **11 business endpoints**:
1. `GET /health` — Service health check
2. `POST /api/inspect` — Run inline inspection pipeline
3. `GET /api/samples` — List demo evaluation samples
4. `GET /api/metrics/latency` — Latency statistics
5. `GET /api/batch/{id}/stats` — Batch quality metrics
6. `GET /api/batch/{id}/spc` — SPC alarm status
7. `GET /api/industrial/state` — PLC register status
8. `GET /api/industrial/signals` — Signal history log
9. `POST /api/industrial/estop` — Emergency stop trigger
10. `POST /api/industrial/reset` — Reset line state
11. `GET /api/system/health-report` — Detailed diagnostic report
