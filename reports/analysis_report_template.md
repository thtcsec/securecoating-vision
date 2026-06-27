# Coating Inspection Quality & Analysis Report

**Report ID:** {report_id}  
**Execution Timestamp:** {timestamp}  
**Inspection Run Batch ID:** {batch_id}  

---

## 1. Executive Summary
*   **Total Parts Inspected:** {total_parts}
*   **Passed Parts:** {passed_parts} ({pass_rate}%)
*   **Failed / Rejected Parts:** {failed_parts} ({fail_rate}%)
*   **System Average Latency:** {avg_latency_ms} ms

---

## 2. Defect Analysis Breakdowns
Below is the distribution of defects categorized by class type:

| Defect Class | Occurrences | Percentage | Primary Sensor Modality Detected |
| :--- | :---: | :---: | :--- |
| **Scratch** | {scratch_count} | {scratch_pct}% | Optical (RGB) |
| **Void** | {void_count} | {void_pct}% | Thermal (LWIR) |
| **Blister** | {blister_count} | {blister_pct}% | 3D Laser Profiler |
| **Delamination** | {delamination_count} | {delamination_pct}% | Mixed (Thermal & 3D) |

### Pareto Defect Distribution
```
Defects
  ^
  |  ===================
  |  *                 *
  |  *   {scratch_cnt} *   ===================
  |  *                 *   *                 *
  |  *                 *   *   {void_cnt}    *   ===================
  |  *                 *   *                 *   *  {delamin_cnt}  *
  +--------------------------------------------------------------------> Classes
        Scratch                 Void               Delamination
```

---

## 3. Industrial Process Diagnostics (SPC)
*   **Control Status:** {control_status} (IN CONTROL / WARNING / OUT OF CONTROL)
*   **Nozzle Pattern Warning:** {nozzle_warning} (YES/NO) - *Triggered when voids or stripes show recurrent spatial coordinates suggesting nozzle blockage.*
*   **Thermal Consistency Check:** {thermal_consistency} (NORMAL / DRIFT DETECTED)

---

## 4. Key Recommendations & Action Items
*   {recommendation_1}
*   {recommendation_2}

*Report generated automatically by SecureCoating-Vision Traceability System.*
