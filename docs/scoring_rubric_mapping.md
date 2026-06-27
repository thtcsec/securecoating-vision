# Scoring Rubric Mapping (Track 4)

This document maps the **Track 4 Evaluation Criteria** to the specific implementation files, modules, and directories in this repository.

| Rubric Category | Weight | Target Criteria | Codebase Implementation Reference |
| :--- | :--- | :--- | :--- |
| **Detection Accuracy & Efficiency** | 30% | - Defect classification/recall<br>- Pipeline latency ($\le 35\text{ ms}$)<br>- Throughput & automation | - `[src/inference/predictor.py](file:///d:/tu_projects/securecoating-vision/src/inference/predictor.py)`: Multithreaded execution, TensorRT/ONNX stub configurations.<br>- `[src/evaluation/evaluate.py](file:///d:/tu_projects/securecoating-vision/src/evaluation/evaluate.py)`: Accuracy, precision, recall, and throughput benchmarks.<br>- `[configs/model.yaml](file:///d:/tu_projects/securecoating-vision/configs/model.yaml)`: Optimization configs. |
| **Technical Completeness & Robustness** | 25% | - Error handling<br>- Sensor fail-safes<br>- Input data validation<br>- Auto-calibration recovery | - `[src/inference/predictor.py](file:///d:/tu_projects/securecoating-vision/src/inference/predictor.py)` (line checks for sensor status & fallback).<br>- `[src/inference/postprocess.py](file:///d:/tu_projects/securecoating-vision/src/inference/postprocess.py)`: Edge check rejection & spatial correction.<br>- `[src/api/main.py](file:///d:/tu_projects/securecoating-vision/src/api/main.py)`: API error boundaries, logging configuration. |
| **Multi-Source Data Fusion** | 20% | - Spatial homography registration<br>- Multi-modal channel weighting<br>- Combination of RGB, thermal, and 3D profiler | - `[src/inference/predictor.py](file:///d:/tu_projects/securecoating-vision/src/inference/predictor.py)`: Homography warps, multi-channel stacking, and channel-attention model stub.<br>- `[data/sample_metadata.csv](file:///d:/tu_projects/securecoating-vision/data/sample_metadata.csv)`: Fused image mappings. |
| **Industrial Application Adaptability** | 10% | - OPC UA / PLC signaling<br>- MES integration APIs<br>- Long-term quality memory and SPC | - `[src/api/main.py](file:///d:/tu_projects/securecoating-vision/src/api/main.py)`: FastAPI end points exposing metrics for MES.<br>- `[src/traceability/quality_memory.py](file:///d:/tu_projects/securecoating-vision/src/traceability/quality_memory.py)`: SPC chart calculations, database logs, trend analysis. |
| **Presentation & Submission Package** | 15% | - Interactive demo dashboard<br>- Clean documentation<br>- Fast containerized deployment | - `[dashboard/app.py](file:///d:/tu_projects/securecoating-vision/dashboard/app.py)`: Streamlit real-time dashboard UI.<br>- `[docker-compose.yml](file:///d:/tu_projects/securecoating-vision/docker-compose.yml)`: Container composition.<br>- `[README.md](file:///d:/tu_projects/securecoating-vision/README.md)`: Rubric-aligned explanations. |

---

## Code References Detail

### 1. Accuracy & Latency (`src/inference/predictor.py`)
Optimized for multi-threaded sensor reading and hardware acceleration (e.g., using `TensorRT`). Defect boundaries are post-processed inside `src/inference/postprocess.py` using fast vector calculations in NumPy.

### 2. Sensor Fallback Mechanism (`src/inference/predictor.py`)
The class `CoatingPredictor` validates whether all sensor inputs (RGB, thermal, 3D profiler) are alive. If one fails, the pipeline degrades gracefully to single-sensor execution and records a diagnostic alert.

### 3. Spatial Image Calibration (`src/inference/predictor.py`)
Includes placeholder functions for calculating the spatial mapping homography between different sensor frame sizes and camera heights.

### 4. PLC Modbus & OPC UA Mocking (`src/api/main.py`)
Under the endpoint `/api/inspect`, the API automatically calls a mock industrial controller to signal the pneumatic sorting gates.

### 5. Quality Memory & MES API (`src/traceability/quality_memory.py`)
Tracks batch statistics to identify trends like nozzle wear or heater degradation, providing early warning capabilities.
