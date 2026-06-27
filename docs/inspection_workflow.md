# Inspection Workflow

This document details the step-by-step workflow of the **SecureCoating-Vision** system, from physical triggering to defect classification and automated sorting.

## Workflow Overview

```
 [Conveyor Motion]
        |
        v
 [Encoder Trigger] ---> [Camera Capture (RGB, Thermal, 3D)]
                               |
                               v
                       [Preprocessing] (Denoise, Exposure Correction)
                               |
                               v
                       [Spatial Alignment] (Homography Matrix Warp)
                               |
                               v
                       [Feature Fusion] (Pixel concat + Channel attention)
                               |
                               v
                       [Model Inference] (UNet/YOLO Multi-Task)
                               |
                               v
                       [Post-Processing] (NMS & Defect Sizing)
                               |
                               +-----------------------------+
                               |                             |
                               v                             v
                      [PLC Reject Command]         [Quality DB & Traceability]
                      (If defect size exceeds      (Saves data for SPC charts)
                       threshold)
```

---

## Detailed Inspection Steps

### Step 1: Hardware Triggering & Acquisition
*   As the product travels on the conveyor, an incremental encoder counts pulses.
*   Upon reaching a predefined distance interval (e.g., every 500 mm), the encoder triggers a digital output pulse.
*   This pulse is broadcast simultaneously to the RGB camera, LWIR thermal camera, and 3D Laser Profiler, ensuring less than $1\text{ ms}$ synchronization error.

### Step 2: Preprocessing & Data Validation
*   **Glance Check:** The system verifies raw image intensity histograms. If lens obstruction (extreme dust or grease) is detected, a maintenance alarm is flagged.
*   **Thermal Normalization:** Raw thermal readings are mapped to temperature values in degrees Celsius using pre-calibrated sensor equations.
*   **3D Height Correction:** Triangulation maps are converted to height offsets (in micrometers) relative to a reference structural baseline.

### Step 3: Spatial Registration & Homography Mapping
*   Because the cameras are mounted at slightly different physical angles and distances, their fields of view differ.
*   A pre-calculated $3\times3$ homography matrix is applied to warp the Thermal image and 3D Height map into the coordinate grid of the high-resolution RGB camera.
*   Result: All three inputs represent exact pixel-to-pixel matches.

### Step 4: Multi-Source Sensor Fusion
*   The aligned inputs are stacked into a 5-channel tensor: `[Red, Green, Blue, Thermal Intensity, Height Offset]`.
*   A lightweight Channel Attention module dynamically weights the channels:
    *   For surface cracks, RGB features are weighted higher.
    *   For sub-surface delamination, the Thermal channel receives higher weights.
    *   For thickness variations and bubbles, the Height channel dominates.

### Step 5: Model Inference
*   The fused 5-channel tensor is passed to a unified deep learning network (such as a multi-task segmentation and detection architecture).
*   The model predicts two outputs:
    1.  **Defect Bounding Boxes:** Class label and confidence score.
    2.  **Segmentation Mask:** High-precision boundary of the defect region.

### Step 6: Post-Processing & Grading Rubric
*   Defect instances are extracted from the segmentation mask.
*   For each defect, physical metrics are computed:
    *   **Length / Width:** Computed using the pixel-to-millimeter ratio.
    *   **Maximum Depth / Height:** Extracted from the 3D laser profiler values.
*   The defect is mapped against the scoring rubric rules (e.g., Scratches $> 5\text{ mm}$ or Voids $> 2\text{ mm}^2$ are marked as critical failures).

### Step 7: Industrial Signaling & Logging
*   **PLC Sorting Gate:** If a critical failure is identified, an OPC UA / Modbus signal is sent to the PLC within $10\text{ ms}$, prompting the pneumatic reject gate to discard the part at the end of the line.
*   **MES Database Update:** Image metadata, defect metrics, timestamp, and camera telemetry are written to the Quality Database for end-to-end traceability.
