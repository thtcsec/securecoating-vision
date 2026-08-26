"""
SecureCoating-Vision: 5-Layer Industrial Electrode Metrology & Standards Audit Engine
====================================================================================
Production metrology architecture separating physical measurement, baseline leveling,
uncertainty propagation, electrochemical risk scoring, and guard-banded compliance.

Architecture:
- Level 1: Calibrated Physical Geometry (minAreaRect, Web-Relative Orientation, Expanded Uncertainty U)
- Level 2: 3D Topography & Surface Plane Baseline Leveling (V_protrusion, V_depression, V_net, Ra)
- Level 3: Micro-Short Hazard Risk Scoring relative to separator safety margin (Cell Stack Model)
- Level 4: Plant Engineering Specification with Guard-Banding (Tolerance - U, ISO 14253-1)
- Level 5: Industrial Quality Standards Traceability (Plant QA Specification & GB/T 38031 Safety Baseline)
"""

import math
import logging
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from industrial.optical_budget import OpticalCalibrationModel

logger = logging.getLogger("SecureCoatingVision.Metrology")


@dataclass
class CellStackSpecification:
    """Configurable battery cell design parameters for electrochemical risk evaluation."""
    separator_nominal_thickness_um: float = 14.0  # Microporous PE/PP separator thickness
    calendering_compaction_ratio: float = 0.75    # 25% thickness reduction in roller press
    nominal_dry_thickness_um: float = 120.0       # Nominal active material dry thickness
    nominal_dry_density_g_cm3: float = 1.60       # Compacted active material density
    nominal_areal_loading_mg_cm2: float = 19.2    # Nominal mass loading = dry_thickness * density * 0.1


@dataclass
class MeasurementUncertaintyBudget:
    """Modeled expanded uncertainty budget (k=2); not an accredited calibration result."""
    u_calibration_um: float = 0.45       # Pixel pitch calibration uncertainty
    u_edge_mtf_um: float = 1.20          # Edge boundary ambiguity from lens MTF
    u_lens_distortion_um: float = 0.30   # Residual optical distortion
    u_sensor_noise_um: float = 0.80      # Laser profilometer speckle noise
    coverage_factor_k: float = 2.0       # k=2 for 95.45% coverage

    def compute_expanded_length_uncertainty_mm(self) -> float:
        u_c = math.sqrt(self.u_calibration_um**2 + self.u_edge_mtf_um**2 + self.u_lens_distortion_um**2)
        return (self.coverage_factor_k * u_c) / 1000.0

    def compute_expanded_area_uncertainty_mm2(self, perimeter_mm: float) -> float:
        u_edge_mm = self.u_edge_mtf_um / 1000.0
        return self.coverage_factor_k * (perimeter_mm * u_edge_mm * 0.5)


@dataclass
class DefectMetrologyResult:
    """Comprehensive physical metrology of an extracted defect region."""
    defect_id: str
    class_id: int
    class_name: str
    bbox_xywh: List[int]
    
    # Level 1: Calibrated Geometry
    length_mm: float
    length_expanded_uncertainty_mm: float
    width_mm: float
    width_expanded_uncertainty_mm: float
    area_mm2: float
    area_expanded_uncertainty_mm2: float
    perimeter_mm: float
    aspect_ratio: float
    web_relative_angle_deg: float        # Angle relative to web travel axis (0° = longitudinal)
    equivalent_diameter_mm: float
    
    # Level 2: 3D Topography (Residual Baseline Leveled)
    peak_protrusion_um: float            # Positive height above leveled baseline
    valley_depression_um: float          # Negative depth below leveled baseline
    volume_protrusion_mm3: float         # Integrated protrusion volume
    volume_depression_mm3: float         # Integrated depression volume
    volume_net_displacement_mm3: float   # Net volumetric displacement
    surface_roughness_ra_um: float       # Arithmetic average roughness inside contour
    
    # Level 3: Electrochemical Risk Scoring
    empirical_micro_short_risk_score: float  # 0.0 to 1.0 heuristic score
    post_calendering_est_height_um: float
    estimated_mass_loading_deviation_pct: float
    attributed_cell_hazard: str
    
    # Level 4 & 5: Guard-Banded Decision & Standards Compliance
    guard_band_pass: bool
    quality_tier: str                    # 'GRADE_A', 'GRADE_B', 'GRADE_C_REJECT'
    rejection_clauses: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "defect_id": self.defect_id,
            "class_name": self.class_name,
            "length_mm": round(self.length_mm, 2),
            "length_uncertainty_mm": round(self.length_expanded_uncertainty_mm, 3),
            "width_mm": round(self.width_mm, 2),
            "width_uncertainty_mm": round(self.width_expanded_uncertainty_mm, 3),
            "area_mm2": round(self.area_mm2, 3),
            "area_uncertainty_mm2": round(self.area_expanded_uncertainty_mm2, 4),
            "perimeter_mm": round(self.perimeter_mm, 2),
            "aspect_ratio": round(self.aspect_ratio, 2),
            "web_relative_angle_deg": round(self.web_relative_angle_deg, 1),
            "peak_height_um": round(self.peak_protrusion_um, 1),
            "valley_depth_um": round(self.valley_depression_um, 1),
            "volume_displacement_mm3": round(self.volume_net_displacement_mm3, 5),
            "surface_roughness_ra_um": round(self.surface_roughness_ra_um, 2),
            "micro_short_hazard_index": round(self.empirical_micro_short_risk_score, 3),
            "post_calendering_height_um": round(self.post_calendering_est_height_um, 1),
            "areal_loading_deviation_pct": round(self.estimated_mass_loading_deviation_pct, 1),
            "battery_failure_mode": self.attributed_cell_hazard,
            "standards_compliant": self.guard_band_pass,
            "quality_tier": self.quality_tier,
            "rejection_clauses": self.rejection_clauses,
        }


# Backwards compatibility alias
DefectMetrology = DefectMetrologyResult


class ElectrodeMetrologyEngine:
    """
    Industrial metrology engine implementing 5-layer separation:
    Geometry -> Topography -> Risk Model -> Guard-Band Decision -> Standards Traceability.
    """

    def __init__(
        self,
        calibration: Optional[OpticalCalibrationModel] = None,
        cell_spec: Optional[CellStackSpecification] = None,
        uncertainty_budget: Optional[MeasurementUncertaintyBudget] = None,
        pixel_to_mm_ratio: Optional[float] = None
    ):
        if calibration is not None:
            self.calib = calibration
        elif pixel_to_mm_ratio is not None:
            self.calib = OpticalCalibrationModel(
                x_um_per_px=pixel_to_mm_ratio * 1000.0,
                y_um_per_px=pixel_to_mm_ratio * 1000.0
            )
        else:
            self.calib = OpticalCalibrationModel()

        self.cell_spec = cell_spec or CellStackSpecification()
        self.uncertainty = uncertainty_budget or MeasurementUncertaintyBudget()
        
        # Plant Engineering Guard-Banded Limits (Electrode Manufacturing QA Baseline)
        self.plant_limits = {
            "scratch": {"max_length_mm": 5.0, "guard_band_mm": 0.15},
            "void": {"max_area_mm2": 1.5, "guard_band_mm2": 0.08},
            "blister": {"max_post_cal_height_um": 11.0, "guard_band_um": 1.0},
            "delamination": {"max_area_mm2": 0.0},  # Strict zero-escape
        }

    def compute_rotated_geometry(
        self, contour: np.ndarray, web_travel_axis_deg: float = 0.0
    ) -> Tuple[float, float, float, float, float, float, float, float]:
        """
        Compute true oriented length and width using minAreaRect and web-relative angle.
        """
        area_px = cv2.contourArea(contour)
        perimeter_px = cv2.arcLength(contour, closed=True)
        
        px_to_mm = self.calib.pixel_to_mm_ratio
        area_mm2 = area_px * (px_to_mm ** 2)
        perimeter_mm = perimeter_px * px_to_mm
        
        # Rotated Minimum Area Bounding Box
        rect = cv2.minAreaRect(contour)
        (center_x, center_y), (rect_w, rect_h), angle_deg = rect
        
        dim1_mm = rect_w * px_to_mm
        dim2_mm = rect_h * px_to_mm
        length_mm = max(dim1_mm, dim2_mm)
        width_mm = max(0.01, min(dim1_mm, dim2_mm))
        aspect_ratio = length_mm / width_mm
        
        # Web-relative orientation (0° = aligned with travel axis, 90° = transverse)
        raw_angle = angle_deg if rect_w < rect_h else angle_deg + 90.0
        web_relative_angle = abs((raw_angle - web_travel_axis_deg) % 180.0)
        
        eq_diameter_mm = 2.0 * math.sqrt(area_mm2 / math.pi) if area_mm2 > 0 else 0.0
        
        return length_mm, width_mm, area_mm2, perimeter_mm, aspect_ratio, web_relative_angle, eq_diameter_mm

    def level_and_integrate_topography(
        self,
        contour: np.ndarray,
        height_map: Optional[np.ndarray],
        img_shape: Tuple[int, int]
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Perform local planar tilt removal and integrate positive/negative volumetric displacements.
        """
        if height_map is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        mask = np.zeros(img_shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, thickness=-1)
        
        h_vals = height_map[mask == 1]
        if len(h_vals) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Local baseline planar estimation: compute median background in bounding area
        x, y, w, h = cv2.boundingRect(contour)
        roi_h = height_map[max(0, y-10):min(img_shape[0], y+h+10), max(0, x-10):min(img_shape[1], x+w+10)]
        baseline_plane_val = float(np.median(roi_h))
        
        # Residual leveled heights: h_res = h_meas - baseline
        h_residual = h_vals - baseline_plane_val
        
        protrusions = h_residual[h_residual > 0]
        depressions = h_residual[h_residual < 0]
        
        peak_protrusion_um = float(np.max(protrusions)) if len(protrusions) > 0 else 0.0
        valley_depression_um = float(np.abs(np.min(depressions))) if len(depressions) > 0 else 0.0
        
        pixel_area_mm2 = self.calib.pixel_to_mm_ratio ** 2
        vol_protrusion_mm3 = float(np.sum(protrusions) / 1000.0) * pixel_area_mm2 if len(protrusions) > 0 else 0.0
        vol_depression_mm3 = float(np.sum(np.abs(depressions)) / 1000.0) * pixel_area_mm2 if len(depressions) > 0 else 0.0
        vol_net_displacement_mm3 = vol_protrusion_mm3 - vol_depression_mm3
        
        mean_res = float(np.mean(h_residual))
        ra_um = float(np.mean(np.abs(h_residual - mean_res)))
        
        return peak_protrusion_um, valley_depression_um, vol_protrusion_mm3, vol_depression_mm3, vol_net_displacement_mm3, ra_um

    def evaluate_risk_and_guardband(
        self,
        class_name: str,
        length_mm: float,
        width_mm: float,
        area_mm2: float,
        peak_h_um: float,
        valley_d_um: float,
        u_length_mm: float,
        u_area_mm2: float
    ) -> Tuple[float, float, float, str, bool, str, List[str]]:
        """
        Evaluate cell stack risk scoring and guard-banded tolerance compliance.
        """
        rejection_clauses = []
        
        # 1. Post-calendering mechanical compaction estimate
        compaction = self.cell_spec.calendering_compaction_ratio
        post_cal_h = peak_h_um * (compaction * 1.1 if class_name == "blister" else compaction * 0.6)
        
        # 2. Empirical Micro-Short Circuit Risk Score (0.0 to 1.0)
        sep_t = self.cell_spec.separator_nominal_thickness_um
        ratio = post_cal_h / sep_t
        risk_score = min(1.0, max(0.0, float(ratio ** 1.6))) if post_cal_h > 0 else 0.0
        
        # 3. Estimated Areal Mass Loading Deviation (%)
        nom_t = self.cell_spec.nominal_dry_thickness_um
        if valley_d_um > 0:
            loading_dev = -1.0 * min(100.0, (valley_d_um / nom_t) * 100.0)
        elif peak_h_um > 0:
            loading_dev = min(100.0, (peak_h_um / nom_t) * 100.0)
        else:
            loading_dev = 0.0

        # 4. Attributed Cell Hazard
        if class_name == "delamination":
            cell_hazard = "Electrode-Substrate De-adhesion / Local High Impedance"
        elif post_cal_h >= 10.0 or class_name == "blister":
            cell_hazard = "Separator Puncture & Internal Micro-Short Risk"
        elif valley_d_um >= 20.0 or class_name == "void":
            cell_hazard = "Localized High Current Density & Lithium Plating"
        else:
            cell_hazard = "Surface Coating Cosmetic Anomaly"

        # 5. Guard-Banded Decision (Plant QA Specification & GB/T 38031 Traceability)
        guard_pass = True
        
        if class_name == "delamination":
            guard_pass = False
            rejection_clauses.append("Plant QA Spec (GB/T 38031 Safety Baseline): Delamination strictly prohibited (Zero Tolerance)")
            
        elif class_name == "scratch":
            limit = self.plant_limits["scratch"]["max_length_mm"]
            gb = self.plant_limits["scratch"]["guard_band_mm"]
            if (length_mm + u_length_mm) > (limit - gb):
                guard_pass = False
                rejection_clauses.append(
                    f"Plant QA Spec: Scratch length ({length_mm:.2f} ± {u_length_mm:.2f}mm) exceeds guard-banded limit ({limit - gb:.2f}mm)"
                )
                
        elif class_name == "void":
            limit = self.plant_limits["void"]["max_area_mm2"]
            gb = self.plant_limits["void"]["guard_band_mm2"]
            if (area_mm2 + u_area_mm2) > (limit - gb):
                guard_pass = False
                rejection_clauses.append(
                    f"Plant QA Spec: Void area ({area_mm2:.2f} ± {u_area_mm2:.3f}mm²) exceeds guard-banded limit ({limit - gb:.2f}mm²)"
                )

        if post_cal_h >= self.plant_limits["blister"]["max_post_cal_height_um"]:
            guard_pass = False
            rejection_clauses.append(
                f"Plant QA Spec (GB/T 38031 Micro-Short Prevention): Post-calendering protrusion ({post_cal_h:.1f}um) exceeds separator safety limit (11.0um)"
            )

        # Quality Tier
        if guard_pass and risk_score < 0.25 and abs(loading_dev) < 8.0:
            tier = "GRADE_A"
        elif guard_pass and risk_score < 0.60:
            tier = "GRADE_B"
        else:
            tier = "GRADE_C_REJECT"

        return risk_score, post_cal_h, loading_dev, cell_hazard, guard_pass, tier, rejection_clauses

    def analyze_segmentation(
        self,
        seg_mask: np.ndarray,
        height_map: Optional[np.ndarray] = None
    ) -> List[DefectMetrologyResult]:
        """
        Complete 5-layer metrology execution with shape validation.
        """
        results: List[DefectMetrologyResult] = []
        if seg_mask is None or np.max(seg_mask) == 0:
            return results

        # Shape validation
        if height_map is not None:
            assert height_map.shape[:2] == seg_mask.shape[:2], (
                f"Shape mismatch: height_map {height_map.shape[:2]} vs seg_mask {seg_mask.shape[:2]}"
            )

        class_map = {
            1: "scratch",
            2: "void",
            3: "blister",
            4: "delamination"
        }
        
        num_classes = int(np.max(seg_mask) + 1)
        h_shape, w_shape = seg_mask.shape[:2]

        for class_id in range(1, num_classes):
            if class_id not in class_map:
                continue
            class_name = class_map[class_id]
            binary = (seg_mask == class_id).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for idx, cnt in enumerate(contours):
                if cv2.contourArea(cnt) < 4:
                    continue

                # Level 1: Calibrated Rotated Geometry
                l_mm, w_mm, a_mm2, p_mm, aspect, web_angle, eq_d = self.compute_rotated_geometry(cnt)
                x, y, bw, bh = cv2.boundingRect(cnt)
                
                # Uncertainty Budget Propagation
                u_len_mm = self.uncertainty.compute_expanded_length_uncertainty_mm()
                u_wid_mm = u_len_mm
                u_area_mm2 = self.uncertainty.compute_expanded_area_uncertainty_mm2(p_mm)

                # Level 2: Topography & Baseline Removal
                (pk_h, vl_d, v_pro, v_dep, v_net, ra_um) = self.level_and_integrate_topography(
                    cnt, height_map, (h_shape, w_shape)
                )

                # Level 3-5: Risk Scoring & Guard-Banded Decision
                (risk_score, post_cal_h, load_dev, hazard, guard_pass, tier, clauses) = self.evaluate_risk_and_guardband(
                    class_name, l_mm, w_mm, a_mm2, pk_h, vl_d, u_len_mm, u_area_mm2
                )

                item = DefectMetrologyResult(
                    defect_id=f"{class_name}_{idx}",
                    class_id=class_id,
                    class_name=class_name,
                    bbox_xywh=[int(x), int(y), int(bw), int(bh)],
                    length_mm=l_mm,
                    length_expanded_uncertainty_mm=u_len_mm,
                    width_mm=w_mm,
                    width_expanded_uncertainty_mm=u_wid_mm,
                    area_mm2=a_mm2,
                    area_expanded_uncertainty_mm2=u_area_mm2,
                    perimeter_mm=p_mm,
                    aspect_ratio=aspect,
                    web_relative_angle_deg=web_angle,
                    equivalent_diameter_mm=eq_d,
                    peak_protrusion_um=pk_h,
                    valley_depression_um=vl_d,
                    volume_protrusion_mm3=v_pro,
                    volume_depression_mm3=v_dep,
                    volume_net_displacement_mm3=v_net,
                    surface_roughness_ra_um=ra_um,
                    empirical_micro_short_risk_score=risk_score,
                    post_calendering_est_height_um=post_cal_h,
                    estimated_mass_loading_deviation_pct=load_dev,
                    attributed_cell_hazard=hazard,
                    guard_band_pass=guard_pass,
                    quality_tier=tier,
                    rejection_clauses=clauses
                )
                results.append(item)

        return results
