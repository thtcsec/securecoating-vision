# SecureCoating-Vision: Performance Analysis Report

**Project:** Multi-Sensor Fusion AI Platform for Inline Coating Defect Inspection  
**Track:** Track 4 — AI + Materials Testing and Characterization  
**Author:** Trịnh Hoàng Tú (HUFLIT)  
**Academic Supervisor:** Kris Singh (SRII; Visiting Professor, Tsinghua University)  
**Date:** July 2026  

---

## 1. Executive Summary

SecureCoating-Vision is structured as a **production-inspired prototype** designed to demonstrate the technical feasibility of real-time inline battery electrode inspection. On our standalone model evaluation subset (`data/evaluation/`, 50 images, 30 GT instances), the ONNX model achieves **54.1% Precision**, **66.7% Recall**, and **59.7% F1-Score** at IoU≥0.50 threshold, with a standalone ONNX inference latency of **46.5 ms per frame on CPU** and **8.7 ms on GPU (RTX 4050)**.

> [!NOTE]
> All quantitative metrics below are directly reproducible by executing `python scripts/run_evaluation.py`. Results are automatically exported to `reports/evaluation_results.json` and `reports/evaluation_results.csv`. Real-line factory deployment will require fine-tuning on physical inline sensor feeds.

---

## 2. Model Architecture & Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | YOLOv8n-seg (Nano segmentation) |
| Backbone | CSPDarknet53-Nano |
| Parameters | 3,258,844 |
| GFLOPs | 11.3 |
| Input Resolution | 640 × 640 |
| Training Epochs | 50 |
| Batch Size | 4 |
| Optimizer | AdamW (lr=0.00125, momentum=0.9) |
| Export Format | ONNX (opset 12, FP16) |
| Model Size | 12.7 MB (ONNX) |

---

## 3. Evaluation Dataset Description

| Subset Path | Total Images | Positive Samples | Negative Clean Samples | GT Instances |
|-------------|--------------|------------------|------------------------|--------------|
| `data/evaluation/` | 50 | 30 | 20 (40%) | 30 |

### Class Breakdown (Evaluation Subset)
| Class ID | Name | Ground Truth Instances | Description |
|----------|------|------------------------|-------------|
| 0 | Scratch | 6 | Linear surface damage, cracks |
| 1 | Void | 10 | Sub-surface pores, inclusions |
| 2 | Blister | 8 | Raised coating bumps |
| 3 | Delamination | 6 | Coating separation/peeling |

---

## 4. Reproducible Evaluation Metrics

### 4.1 Overall Metrics (`python scripts/run_evaluation.py`)

| Metric | Measured Value | Design Target |
|--------|----------------|---------------|
| Precision | 54.1% | ≥95.0% |
| Recall | 66.7% | Target near-zero defect escape |
| F1-Score | 59.7% | ≥96.5% |
| True Positives (TP) | 20 | — |
| False Positives (FP) | 17 | — |
| False Negatives (FN) | 10 | — |

### 4.2 Per-Class Performance Breakdown

| Class | TP | FP | FN | Precision | Recall | Failure Case Analysis |
|-------|----|----|----|-----------|--------|-----------------------|
| Scratch | 2 | 0 | 4 | 100.0% | 33.3% | Thin 2px lines require fine confidence thresholding |
| Void | 10 | 6 | 0 | 62.5% | 100.0% | Excellent recall (100%), mild FP on dark noise spots |
| Blister | 8 | 4 | 0 | 66.7% | 100.0% | Excellent recall (100%), mild FP on glare highlights |
| Delamination | 0 | 7 | 6 | 0.0% | 0.0% | Contrast threshold mismatch on large patch polygons |

### 4.3 Honest Baseline & Failure Analysis
- **Void & Blister (100% Recall):** The model achieves 100% recall on surface voids and raised blisters, eliminating false negatives for these critical defect categories.
- **Scratch (33.3% Recall):** Ultra-thin scratches (2-3px) require lower confidence thresholds (`conf_thresh=0.25`) to increase sensitivity.
- **Delamination (0% Recall):** Delamination patches in the synthetic validation set exhibit high texture contrast variation, causing the model to misclassify them as large voids or blisters. This highlights the need for fine-tuning on real factory delamination samples.

---

## 5. Inference & Pipeline Latency

| Benchmark Context | Provider | Mean Latency | Median | P95 | Throughput |
|-------------------|----------|--------------|--------|-----|------------|
| ONNX Model (GPU FP16) | CUDAExecutionProvider | 8.7 ms | 8.5 ms | 11.2 ms | 115 FPS |
| ONNX Model (CPU) | CPUExecutionProvider | 46.5 ms | 46.2 ms | 51.6 ms | 21.5 FPS |
| **End-to-End Pipeline (GPU)** | **CUDA + Postprocess + DB** | **12.5 ms** | **12.1 ms** | **15.0 ms** | **80 FPS** |

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
