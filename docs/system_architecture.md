# System Architecture

This document describes the hardware and software architecture of the **SecureCoating-Vision** system.

## 1. Hardware Architecture & Sensor Layout

The physical inspection station is installed directly on the coating production line. The architecture utilizes three distinct imaging modalities synchronized with the conveyor's movement.

```
                  +---------------------------------+
                  |      Industrial PLC Controller  | <--- Encoder Pulses
                  +---------------------------------+
                                   |
                  +----------------+----------------+
                  |  Hardware Trigger (Pulse Sync)  |
                  +----------------+----------------+
                                   |
       +---------------------------+---------------------------+
       |                           |                           |
       v                           v                           v
+--------------+            +--------------+            +--------------+
| Optical Cam  |            |  LWIR Camera |            |  3D Profiler |
| (RGB 2D)     |            |  (Thermal)   |            |  (Laser)     |
+--------------+            +--------------+            +--------------+
       |                           |                           |
       v                           v                           v
  [10GbE / GigE]              [GigE Vision]                 [USB 3.0]
       |                           |                           |
       +---------------------------+---------------------------+
                                   |
                                   v
                  +---------------------------------+
                  |      Edge Inference Computer    |
                  |     (NVIDIA Jetson / RTX GPU)   |
                  +---------------------------------+
```

### Sensor Specifications & Roles:
1.  **Optical RGB Camera:** 2D high-resolution GigE Vision camera. Captures high-frequency spatial details (scratches, pinholes, contamination, hair).
2.  **LWIR Thermal Camera:** Long-Wave Infrared sensor detecting heat signatures. Active heating/cooling elements before the inspection zone generate thermal differentials. Sub-surface voids and delaminations manifest as local thermal impedance variations (hot/cold spots).
3.  **3D Laser Line Profiler:** Laser triangulation profiler. Captures precise depth/height profiles to gauge absolute coating thickness and detect bumps/blisters.

---

## 2. Software Architecture

The software architecture is modular, event-driven, and designed for sub-40ms execution times.

```
+-------------------------------------------------------------------------------------------------+
|                                    Edge Inference Service                                       |
|                                                                                                 |
|  +------------------+      +---------------------+      +----------------+                      |
|  |  Data Capture    | ---> | Multi-Source Fusion | ---> | TensorRT/ONNX  |                      |
|  |  (Async Workers) |      | (Spatial alignment) |      | Inference Eng. |                      |
|  +------------------+      +---------------------+      +----------------+                      |
|                                                                 |                               |
|                                                                 v                               |
|  +------------------+      +---------------------+      +----------------+                      |
|  |  Industrial I/O  | <--- | Post-Processing     | <--- | Defect Map &   |                      |
|  |  (OPC UA/Modbus) |      | (NMS, Classification|      | Segmentation   |                      |
|  +------------------+      +---------------------+      +----------------+                      |
+-------------------------------------------------------------------------------------------------+
                                      |
                                      v
                      +-------------------------------+
                      |      Quality Database &       |
                      |      MES REST API Service     |
                      +-------------------------------+
                                      |
                                      v
                      +-------------------------------+
                      |      Streamlit Dashboard      |
                      +-------------------------------+
```

### Component Details

*   **Data Capture Module:** Uses multi-threaded queues to capture images from RGB, LWIR, and 3D sensors. It ensures frames are temporally matched based on hardware encoder timestamps.
*   **Fusion & Alignment Module:** Performs spatial registration using pre-calculated homography matrices. Transforms thermal and 3D profiling maps to align precisely with the optical coordinate system.
*   **Inference Engine:** Runs the fused images through a multi-branch convolutional/transformer network optimized via TensorRT for low latency.
*   **Post-processing & Grading Module:** Filters raw predictions using Non-Maximum Suppression (NMS), calculates physical defect sizes (in mm), and applies the scoring rubric to make a pass/fail decision.
*   **Industrial Communication Module:** A TCP client communicating directly with the PLC via OPC UA or Modbus protocol to trigger immediate rejection hardware for failed parts.
*   **dashboard/app.py:** Real-time user interface showing live feed, historical trends, and quality reports.
