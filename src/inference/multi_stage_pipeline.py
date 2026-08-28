"""
SecureCoating-Vision: 7-Stage Multi-Modal Industrial Pipeline Orchestrator
==========================================================================
End-to-End industrial inspection pipeline executing:
1. Web Motion & Quadrature Encoder Pulse Synchronization (A/B fractional residual tracking)
2. Traveling-Wave Thermography & Multi-Modal Acquisition (Darkfield/Brightfield, Thermal Diffusivity Inversion, 3D Laser)
3. Sub-Pixel Homography Registration & 5-Channel Tensor Concatenation
4. Configured YOLO / ONNX Instance Segmentation
5. Prototype battery-electrode metrology policy (not standards certification)
6. Fail-closed PASS / REJECT / HOLD decision and PLC protocol contract
7. Prototype rule-based diagnostics, spatial periodicity analysis, and per-lane summaries
"""

import time
import math
import logging
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

from industrial.web_synchronizer import WebSynchronizer, WebCoordinate, RollMetadata
from industrial.protocol_manager import IndustrialProtocolManager
from industrial.optical_budget import OpticalThroughputBudgetEngine
from industrial.latency_budget import LatencyBudgetEngine
from industrial.spc_spatial_diagnostics import (
    SpatialDiagnosticsEngine, GigafactorySPCEngine, DigitalBatteryPassportGenerator,
    SpatialPeriodicAnomaly, LaneSPCResult, SlittingYieldPlan
)
from inference.sensor_fusion import SensorFusionManager
from inference.failsafe import FailSafeManager
from inference.predictor import CoatingPredictor
from inference.electrode_metrology import ElectrodeMetrologyEngine, DefectMetrology
from traceability.root_cause_engine import RootCauseDiagnosticEngine, RootCauseReport

logger = logging.getLogger("SecureCoatingVision.Pipeline")


def serialize_detection_metadata(detection: Dict[str, Any]) -> Dict[str, Any]:
    """Bounded JSON metadata for one detector proposal.

    Instance masks can be megapixel numpy arrays. They remain available to the
    in-process metrology path but must not be copied into API/artifact summaries.
    """
    record: Dict[str, Any] = {}
    for key, value in detection.items():
        if key == "mask":
            if value is not None:
                mask = np.asarray(value)
                record["mask_shape"] = [int(v) for v in mask.shape]
                record["mask_foreground_pixels"] = int(np.count_nonzero(mask))
            continue
        if isinstance(value, np.ndarray):
            record[key] = value.tolist()
        elif isinstance(value, np.generic):
            record[key] = value.item()
        elif isinstance(value, tuple):
            record[key] = list(value)
        else:
            record[key] = value
    return record


@dataclass
class MultiStageInspectionResult:
    """Full telemetry and metrology outcome of the 7-stage inspection pipeline."""
    sample_id: str
    roll_coordinate: Dict[str, Any]
    
    # Timing breakdown (latency in milliseconds per stage)
    stage_latencies_ms: Dict[str, float]
    total_pipeline_latency_ms: float
    
    # Sensor Frames (numpy arrays)
    optical_brightfield: np.ndarray
    optical_darkfield: np.ndarray
    thermal_diffusivity_phase: np.ndarray
    height_topography_map: np.ndarray
    segmentation_mask: np.ndarray
    
    # AI & Metrology Findings
    raw_detections: List[Dict[str, Any]]
    defect_metrology: List[Dict[str, Any]]
    
    # Quality & Hardware Decision
    overall_verdict: str                # 'PASS', 'REJECT', or 'HOLD'
    quality_tier: str                   # 'GRADE_A', 'GRADE_B_QUARANTINE', 'GRADE_C_REJECT'
    rejection_reasons: List[str]
    standards_compliant: bool
    plc_gate_action: str                # 'PASS', 'REJECT', 'HOLD'
    
    # Closed-Loop AI Diagnostics
    root_cause_report: Dict[str, Any]
    
    # Execution Metadata
    system_health: str                  # 'OPTIMAL', 'DEGRADED', 'OFFLINE'
    measurement_policy: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary for API/Dashboard responses."""
        return {
            "sample_id": self.sample_id,
            "roll_coordinate": self.roll_coordinate,
            "total_latency_ms": round(self.total_pipeline_latency_ms, 2),
            "stage_latencies_ms": {k: round(v, 2) for k, v in self.stage_latencies_ms.items()},
            "overall_verdict": self.overall_verdict,
            "quality_tier": self.quality_tier,
            "standards_compliant": self.standards_compliant,
            "rejection_reasons": self.rejection_reasons,
            "plc_gate_action": self.plc_gate_action,
            "raw_detections_count": len(self.raw_detections),
            "defects_count": len(self.defect_metrology),
            "defect_metrology": self.defect_metrology,
            "raw_detections": [
                serialize_detection_metadata(item) for item in self.raw_detections
            ],
            "measurement_policy": self.measurement_policy,
            "root_cause_report": self.root_cause_report,
            "system_health": self.system_health,
            "timestamp": self.timestamp,
        }


class MultiStageIndustrialPipeline:
    """
    Research inspection orchestrator unifying the seven prototype stages.
    """

    def __init__(
        self,
        predictor: CoatingPredictor,
        fusion_manager: Optional[SensorFusionManager] = None,
        failsafe_manager: Optional[FailSafeManager] = None,
        industrial_manager: Optional[IndustrialProtocolManager] = None,
        web_synchronizer: Optional[WebSynchronizer] = None,
        pixel_to_mm_ratio: float = 0.1,
        calibration_verified: bool = False,
        fov_width_mm: Optional[float] = None,
        fov_length_m: Optional[float] = None,
    ):
        self.predictor = predictor
        self.fusion = fusion_manager or SensorFusionManager(target_size=(1024, 1024), enable_mock=True)
        self.failsafe = failsafe_manager or FailSafeManager()
        self.industrial = industrial_manager or IndustrialProtocolManager({"enabled": True, "mock_mode": True})
        self.web_sync = web_synchronizer or WebSynchronizer()
        self.metrology = ElectrodeMetrologyEngine(pixel_to_mm_ratio=pixel_to_mm_ratio)
        self.root_cause = RootCauseDiagnosticEngine()
        self.optical_budget = OpticalThroughputBudgetEngine()
        self.latency_budget = LatencyBudgetEngine()
        self.spatial_diagnostics = SpatialDiagnosticsEngine()
        self.spc_engine = GigafactorySPCEngine(num_lanes=self.web_sync.roll.num_lanes)
        self.pixel_to_mm = pixel_to_mm_ratio
        self.calibration_verified = calibration_verified
        self.fov_width_mm = fov_width_mm
        self.fov_length_m = fov_length_m

    def _simulate_traveling_wave_thermography(
        self,
        base_thermal: np.ndarray,
        downstream_distance_mm: float = 150.0
    ) -> np.ndarray:
        """
        Simulates Traveling-Wave Thermography on moving web.
        Thermal excitation occurs at X0. LWIR camera inspects at downstream distance L_drift.
        Calculates local thermal diffusivity alpha = k / (rho * cp) from spatial temperature decay.
        """
        if base_thermal is None or base_thermal.size == 0:
            return np.zeros((1024, 1024), dtype=np.uint8)
            
        norm = cv2.normalize(base_thermal, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        dx = cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=3)
        thermal_gradient = np.sqrt(dx**2 + dy**2)
        
        # Spatial phase lag delta phi based on traveling wave diffusion
        phase_map = np.arctan2(dy, dx + 1e-6) * (1.0 + thermal_gradient * 2.0)
        phase_normalized = cv2.normalize(phase_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return phase_normalized

    def _simulate_darkfield_scatter(self, optical_rgb: np.ndarray) -> np.ndarray:
        """High-angle darkfield scatter simulation for micro-scratches."""
        if optical_rgb is None:
            return np.zeros((1024, 1024, 3), dtype=np.uint8)
        gray = cv2.cvtColor(optical_rgb, cv2.COLOR_BGR2GRAY) if len(optical_rgb.shape) == 3 else optical_rgb
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        scatter = np.uint8(np.clip(np.abs(laplacian) * 3.5, 0, 255))
        darkfield_bgr = cv2.applyColorMap(scatter, cv2.COLORMAP_VIRIDIS)
        return darkfield_bgr

    def execute_inspection(
        self,
        sample_id: str,
        optical_rgb: Optional[np.ndarray] = None,
        thermal_raw: Optional[np.ndarray] = None,
        height_map: Optional[np.ndarray] = None,
        cross_web_pos_mm: float = 325.0,
        enable_plc_signal: bool = True
    ) -> MultiStageInspectionResult:
        """
        Executes the full 7-stage industrial inspection workflow.
        """
        timings: Dict[str, float] = {}
        t_start_total = time.perf_counter()

        # =========================================================================
        # STAGE 1: Continuous Web Handling & Line Encoder Synchronization
        # =========================================================================
        t0 = time.perf_counter()
        frame_ctx = self.web_sync.advance_motion(dt_seconds=0.055)
        coord = self.web_sync.get_current_coordinate(cross_pos_mm=cross_web_pos_mm)
        timings["stage_1_web_sync_ms"] = (time.perf_counter() - t0) * 1000.0

        # =========================================================================
        # STAGE 2: Multi-Modal Physical Acquisition
        # =========================================================================
        t0 = time.perf_counter()
        if optical_rgb is None:
            logger.warning("No optical image supplied; running simulated-input sandbox path")
            mock_opt, mock_th, mock_h = self.fusion.generate_mock_frame(defect_type="random")
            optical_rgb = mock_opt
            thermal_raw = mock_th if thermal_raw is None else thermal_raw
            height_map = mock_h if height_map is None else height_map

        thermal_available = thermal_raw is not None
        height_available = height_map is not None
        frame_h, frame_w = optical_rgb.shape[:2]
        if thermal_raw is None:
            thermal_raw = np.zeros((frame_h, frame_w), dtype=np.uint8)
        if height_map is None:
            height_map = np.zeros((frame_h, frame_w), dtype=np.uint8)

        self.failsafe.health.rgb_sensor_ok = True
        self.failsafe.health.thermal_sensor_ok = thermal_available
        self.failsafe.health.profiler_sensor_ok = height_available
        self.failsafe._update_system_state()

        brightfield_img = optical_rgb.copy()
        darkfield_img = self._simulate_darkfield_scatter(optical_rgb)
        thermal_phase = self._simulate_traveling_wave_thermography(thermal_raw)
        timings["stage_2_acquisition_ms"] = (time.perf_counter() - t0) * 1000.0

        # =========================================================================
        # STAGE 3: Sub-Pixel Homography Registration & 5-Channel Fusion
        # =========================================================================
        t0 = time.perf_counter()
        fusion_res = self.fusion.fuse(
            rgb_image=optical_rgb,
            thermal_override=thermal_raw,
            height_override=height_map
        )
        fused_tensor = fusion_res.fused_tensor
        timings["stage_3_fusion_ms"] = (time.perf_counter() - t0) * 1000.0

        # =========================================================================
        # STAGE 4: High-Throughput Edge AI Instance Segmentation
        # =========================================================================
        t0 = time.perf_counter()
        ai_result = self.failsafe.safe_predict(
            self.predictor,
            optical=optical_rgb,
            thermal=thermal_raw,
            height=height_map,
        )
        inspection_valid = (
            ai_result.get("system_state") == "OPTIMAL"
            and not ai_result.get("untrained_fallback", False)
            and self.failsafe.decision_permitted()
            and self.calibration_verified
        )
        seg_mask = ai_result.get(
            "segmentation_mask",
            np.zeros(optical_rgb.shape[:2], dtype=np.int32),
        )
        raw_detections = ai_result.get("detections", [])
        timings["stage_4_ai_inference_ms"] = (time.perf_counter() - t0) * 1000.0

        # =========================================================================
        # STAGE 5: Physics-Informed Battery Electrode Metrology & Standards Audit
        # =========================================================================
        t0 = time.perf_counter()
        metrology_objects = (
            self.metrology.analyze_segmentation(seg_mask, height_map)
            if height_available and inspection_valid
            else []
        )
        metrology_dicts = [m.to_dict() for m in metrology_objects]
        
        # Temporal Frame Context Anchoring: map defect coordinates back to capture instant
        for m_obj in metrology_objects:
            defect_dict = {
                "defect_id": m_obj.defect_id,
                "class_name": m_obj.class_name,
                "severity": "CRITICAL" if m_obj.quality_tier == "GRADE_C_REJECT" else "WARNING",
                "area_mm2": m_obj.area_mm2,
                "peak_height_um": m_obj.peak_protrusion_um,
            }
            px_x = min(frame_w - 1, max(0, m_obj.bbox_xywh[0] + m_obj.bbox_xywh[2] // 2))
            px_y = min(frame_h - 1, max(0, m_obj.bbox_xywh[1] + m_obj.bbox_xywh[3] // 2))
            defect_coord = self.web_sync.map_defect_to_physical_coordinate(
                frame_ctx=frame_ctx,
                pixel_x_td=float(px_x),
                pixel_y_md=float(px_y),
                frame_width_px=frame_w,
                frame_height_px=frame_h,
                fov_width_mm=(self.fov_width_mm or frame_w * self.pixel_to_mm),
                fov_length_m=(self.fov_length_m or frame_h * self.pixel_to_mm / 1000.0),
                fov_center_td_mm=cross_web_pos_mm,
            )
            self.web_sync.record_defect_on_roll(defect_dict, defect_coord)
            
        timings["stage_5_metrology_ms"] = (time.perf_counter() - t0) * 1000.0

        # =========================================================================
        # STAGE 6: Zero-Escape Decision Policy & Hardware PLC Rejection Gate
        # =========================================================================
        t0 = time.perf_counter()
        all_rejection_reasons: List[str] = []
        std_compliant = True
        overall_tier = "GRADE_A"

        if not inspection_valid:
            overall_tier = "GRADE_B_QUARANTINE"
            std_compliant = False
            if not thermal_available or not height_available:
                all_rejection_reasons.append(
                    "RGB-only or incomplete secondary evidence: calibrated thermal/"
                    "profilometer data unavailable -> HOLD for QA; no release decision "
                    "or dimensional metrology"
                )
            else:
                all_rejection_reasons.append(
                    "Inspection evidence is not safety-ready -> HOLD for QA; "
                    "no release decision or dimensional metrology"
                )

        for m_obj in metrology_objects:
            if not m_obj.guard_band_pass:
                std_compliant = False
                all_rejection_reasons.extend(m_obj.rejection_clauses)
            if m_obj.quality_tier == "GRADE_C_REJECT":
                overall_tier = "GRADE_C_REJECT"
            elif m_obj.quality_tier == "GRADE_B" and overall_tier != "GRADE_C_REJECT":
                overall_tier = "GRADE_B"

        overall_verdict = (
            "HOLD" if overall_tier == "GRADE_B_QUARANTINE" else
            "REJECT" if overall_tier == "GRADE_C_REJECT" else "PASS"
        )

        # Signal PLC Hardware Gate
        # This is the intended gate action. Transport/ACK state is recorded by
        # IndustrialProtocolManager when signaling is enabled; a disabled signal
        # path must never turn a REJECT decision into a displayed PASS action.
        plc_action = overall_verdict
        if enable_plc_signal:
            grade_dict = {
                "passed": (overall_verdict == "PASS"),
                "reject_reasons": all_rejection_reasons,
                "action": overall_verdict
            }
            plc_res = self.industrial.process_inspection_result(
                part_id=sample_id,
                batch_id=self.web_sync.roll.batch_id,
                defects=metrology_dicts,
                grade_result=grade_dict,
                safety_permitted=inspection_valid,
                safety_reasons=all_rejection_reasons,
            )
            plc_action = plc_res.get("gate_action", overall_verdict)

        timings["stage_6_decision_plc_ms"] = (time.perf_counter() - t0) * 1000.0
        measurement_policy = (
            "No dimensional metrology or release decision when inspection evidence is not safety-ready."
            if not inspection_valid else
            "Prototype metrology policy only; not a standards certification."
        )

        # =========================================================================
        # STAGE 7: AI Closed-Loop Diagnostics & Equipment Tuning Feedback
        # =========================================================================
        t0 = time.perf_counter()
        root_cause_report = self.root_cause.diagnose_batch(
            metrology_dicts,
            inspection_valid=inspection_valid,
            gate_action=plc_action,
            raw_detections=raw_detections,
            hold_reasons=all_rejection_reasons,
        )
        timings["stage_7_root_cause_ms"] = (time.perf_counter() - t0) * 1000.0

        total_latency = (time.perf_counter() - t_start_total) * 1000.0

        return MultiStageInspectionResult(
            sample_id=sample_id,
            roll_coordinate=coord.to_dict(),
            stage_latencies_ms=timings,
            total_pipeline_latency_ms=total_latency,
            optical_brightfield=brightfield_img,
            optical_darkfield=darkfield_img,
            thermal_diffusivity_phase=thermal_phase,
            height_topography_map=height_map,
            segmentation_mask=seg_mask,
            raw_detections=raw_detections,
            defect_metrology=metrology_dicts,
            overall_verdict=overall_verdict,
            quality_tier=overall_tier,
            rejection_reasons=all_rejection_reasons,
            standards_compliant=std_compliant,
            plc_gate_action=plc_action,
            root_cause_report=root_cause_report.to_dict(),
            system_health=self.failsafe.system_state.value,
            measurement_policy=measurement_policy,
        )

    def get_gigafactory_spc_summary(self) -> Dict[str, Any]:
        """Compute provisional lane summaries, a slitting scenario, and traceability schema."""
        roll_summary = self.web_sync.get_roll_defect_summary()
        defects_list = list(self.web_sync.recent_defect_cache)
        
        spc_res = self.spc_engine.compute_lane_spc(
            defects_by_lane=roll_summary["defects_by_lane"],
            inspected_length_m=roll_summary["inspected_length_m"]
        )
        
        yield_plan = self.spc_engine.optimize_slitting_yield(
            roll_length_m=roll_summary["inspected_length_m"],
            web_width_mm=self.web_sync.roll.web_width_mm,
            defects_list=defects_list
        )

        md_positions = [d.get("linear_pos_m", 0.0) for d in defects_list]
        anomalies = self.spatial_diagnostics.analyze_spatial_periodicity(
            defect_md_positions_m=md_positions,
            total_scanned_length_m=roll_summary["inspected_length_m"]
        )

        passport = DigitalBatteryPassportGenerator.generate_passport(
            roll_id=self.web_sync.roll.roll_id,
            batch_id=self.web_sync.roll.batch_id,
            spc_results=spc_res,
            yield_plan=yield_plan,
            anomalies=anomalies
        )

        return passport
