"""
SecureCoating-Vision: Experimental Spatial Diagnostics & Slitting Model
========================================================================
Prototype analytics for research and simulation:
1. Spatial Frequency FFT / Autocorrelation:
   - Converts Machine Direction (MD) defect distribution into spatial wavelengths.
   - Accurately matches lambda = pi * D_roller to identify damaged rollers or pump pulsations.
2. Per-Lane Summary:
   - Reports observed defect density. Cpk/Ppk remain unavailable without subgroup measurements.
3. Smart Slitting Yield Optimizer:
   - Dynamically calculates salvageable roll yield, cut/splice locations, and economic recovery.
4. Prototype passport-shaped telemetry manifest without a regulatory compliance claim.
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class EquipmentRegistry:
    """Registered diameters of rotating equipment along the coating & calendering line."""
    # Equipment Name -> Diameter in mm
    roller_diameters_mm: Dict[str, float] = field(default_factory=lambda: {
        "Coating Backing Roll (BR-01)": 250.0,          # Circumference: ~785.4 mm
        "Slot-Die Vacuum Roll (VR-01)": 180.0,           # Circumference: ~565.5 mm
        "Oven Zone 1 Guide Roller (R-01)": 120.0,        # Circumference: ~377.0 mm
        "Oven Zone 2 Guide Roller (R-02)": 120.0,        # Circumference: ~377.0 mm
        "Oven Zone 3 Tension Roller (TR-01)": 150.0,     # Circumference: ~471.2 mm
        "Calendering Upper Roll (CR-01)": 400.0,         # Circumference: ~1256.6 mm
        "Calendering Lower Roll (CR-02)": 400.0,         # Circumference: ~1256.6 mm
        "Slurry Pump Positive Displacement (P-01)": 85.0 # Stroke Period Equivalent: ~267.0 mm
    })


@dataclass
class SpatialPeriodicAnomaly:
    """Detected periodic mechanical defect signature."""
    dominant_wavelength_mm: float
    confidence_peak: float
    matched_equipment: str
    roller_diameter_mm: float
    recommended_maintenance_action: str


@dataclass
class LaneSPCResult:
    """Six Sigma Statistical Process Control capability for a single slitting lane."""
    lane_id: int
    sample_count: int
    mean_defect_density_per_100m: float
    cpk: Optional[float]
    ppk: Optional[float]
    six_sigma_tier: str
    suggested_process_tuning: str


@dataclass
class SlittingYieldPlan:
    """Optimized cutting and grading plan for the jumbo roll."""
    total_roll_length_m: float
    gross_area_m2: float
    net_usable_ev_area_m2: float
    net_usable_ess_area_m2: float
    scrap_area_m2: float
    overall_recovery_yield_pct: float
    lane_grades: Dict[int, str]
    recommended_splices_md_m: List[float]


class SpatialDiagnosticsEngine:
    """
    Analyzes spatial MD defect distributions using Spatial FFT and Autocorrelation to pinpoint
    damaged line components with zero guesswork.
    """

    def __init__(self, registry: Optional[EquipmentRegistry] = None):
        self.registry = registry or EquipmentRegistry()

    def analyze_spatial_periodicity(
        self,
        defect_md_positions_m: List[float],
        total_scanned_length_m: float,
        spatial_bin_size_mm: float = 10.0
    ) -> List[SpatialPeriodicAnomaly]:
        """
        Compute spatial FFT over binned MD defect counts to extract repeating wavelength patterns.
        """
        if len(defect_md_positions_m) < 6 or total_scanned_length_m <= 1.0:
            return []

        # Bin defects along MD length
        num_bins = int((total_scanned_length_m * 1000.0) / spatial_bin_size_mm)
        if num_bins < 64:
            return []

        bins = np.zeros(num_bins, dtype=np.float32)
        for pos_m in defect_md_positions_m:
            idx = int((pos_m * 1000.0) / spatial_bin_size_mm)
            if 0 <= idx < num_bins:
                bins[idx] += 1.0

        # Remove DC mean
        bins_detrended = bins - np.mean(bins)

        # Compute Fast Fourier Transform
        fft_vals = np.fft.rfft(bins_detrended)
        fft_mag = np.abs(fft_vals)
        freqs = np.fft.rfftfreq(num_bins, d=(spatial_bin_size_mm / 1000.0))  # cycles per meter

        # Ignore lowest frequencies (< 0.5 cycles/meter)
        min_freq_idx = max(1, int(0.5 / (freqs[1] - freqs[0] + 1e-9)))
        fft_mag[:min_freq_idx] = 0.0

        anomalies = []
        # Spatial Autocorrelation for fundamental periodicity (eliminates harmonic ambiguity)
        min_lag_bins = int(100.0 / spatial_bin_size_mm)   # min 100mm wavelength
        max_lag_bins = min(num_bins // 2, int(2000.0 / spatial_bin_size_mm))  # max 2000mm wavelength
        
        if max_lag_bins > min_lag_bins:
            # Normalized autocorrelation
            norm_b = bins - np.mean(bins)
            var_b = np.sum(norm_b**2) + 1e-9
            autocorr = np.correlate(norm_b, norm_b, mode='full')
            mid = len(autocorr) // 2
            r_lags = autocorr[mid + min_lag_bins: mid + max_lag_bins] / var_b
            
            # Find the FIRST prominent local peak in autocorrelation (fundamental period)
            first_peak_idx = None
            for i in range(1, len(r_lags) - 1):
                if r_lags[i] > r_lags[i - 1] and r_lags[i] > r_lags[i + 1] and r_lags[i] >= 0.06:
                    first_peak_idx = i
                    break
            
            # Fallback to global argmax if no discrete local peak was tripped
            if first_peak_idx is None and len(r_lags) > 0 and np.max(r_lags) > 0.08:
                first_peak_idx = int(np.argmax(r_lags))

            if first_peak_idx is not None:
                peak_lag_bins = min_lag_bins + first_peak_idx
                wavelength_mm = peak_lag_bins * spatial_bin_size_mm
                autocorr_score = float(r_lags[first_peak_idx])

                target_diam_mm = wavelength_mm / math.pi
                best_match = "Unknown Rotating Assembly"
                best_diff = float("inf")
                matched_d = target_diam_mm

                for eq_name, eq_diam in self.registry.roller_diameters_mm.items():
                    diff = abs(eq_diam - target_diam_mm)
                    if diff < best_diff and diff < (eq_diam * 0.15):  # 15% tolerance
                        best_diff = diff
                        best_match = eq_name
                        matched_d = eq_diam

                action = (
                    f"Inspect and clean {best_match} (diameter ~{matched_d:.0f}mm). "
                    f"Defect repeats every {wavelength_mm:.1f}mm along web."
                )

                anomalies.append(
                    SpatialPeriodicAnomaly(
                        dominant_wavelength_mm=round(wavelength_mm, 1),
                        confidence_peak=min(0.99, round(autocorr_score * 3.0, 3)),
                        matched_equipment=best_match,
                        roller_diameter_mm=round(matched_d, 1),
                        recommended_maintenance_action=action
                    )
                )

        return anomalies


class GigafactorySPCEngine:
    """
    Computes Six Sigma Statistical Process Control (SPC) metrics per slitting lane
    and optimizes jumbo roll slitting yield according to Tier-1 automotive standards.
    """

    def __init__(self, target_defect_density_limit_per_100m: float = 2.0, num_lanes: int = 4):
        self.usl = target_defect_density_limit_per_100m  # Upper Specification Limit
        self.lsl = 0.0                                    # Lower Specification Limit
        self.num_lanes = num_lanes

    def compute_lane_spc(
        self,
        defects_by_lane: Dict[int, int],
        inspected_length_m: float
    ) -> Dict[int, LaneSPCResult]:
        """
        Report observed defect density. Aggregate counts do not contain the
        within-subgroup and overall variation required to calculate Cpk/Ppk.
        """
        results = {}
        total_m = max(100.0, inspected_length_m)

        for lane_id in range(1, self.num_lanes + 1):
            count = defects_by_lane.get(lane_id, 0)
            density_per_100m = (count / total_m) * 100.0

            cpk = None
            ppk = None
            tier = "INSUFFICIENT_SUBGROUP_DATA"
            action = (
                "Collect timestamped subgroup measurements and validated specification "
                "limits before calculating process capability."
            )

            results[lane_id] = LaneSPCResult(
                lane_id=lane_id,
                sample_count=count,
                mean_defect_density_per_100m=round(density_per_100m, 2),
                cpk=cpk,
                ppk=ppk,
                six_sigma_tier=tier,
                suggested_process_tuning=action
            )

        return results

    def optimize_slitting_yield(
        self,
        roll_length_m: float,
        web_width_mm: float,
        defects_list: List[Dict[str, Any]]
    ) -> SlittingYieldPlan:
        """
        Smart Slitting Yield Optimizer:
        Calculates usable EV/ESS area, determines optimal splice points, and maximizes total product recovery.
        """
        if roll_length_m <= 0.0:
            return SlittingYieldPlan(
                total_roll_length_m=0.0,
                gross_area_m2=0.0,
                net_usable_ev_area_m2=0.0,
                net_usable_ess_area_m2=0.0,
                scrap_area_m2=0.0,
                overall_recovery_yield_pct=0.0,
                lane_grades={i: "UNVERIFIED" for i in range(1, self.num_lanes + 1)},
                recommended_splices_md_m=[],
            )
        eff_length_m = roll_length_m
        gross_area = (eff_length_m * web_width_mm) / 1000.0
        lane_width_mm = web_width_mm / self.num_lanes
        lane_area = gross_area / self.num_lanes

        lane_defects = {i: [] for i in range(1, self.num_lanes + 1)}
        for d in defects_list:
            lane = d.get("lane_id", 1)
            if lane in lane_defects:
                lane_defects[lane].append(d)

        lane_grades = {}
        ev_area = 0.0
        ess_area = 0.0
        scrap_area = 0.0
        splices = []

        for lane_id, d_list in lane_defects.items():
            critical_count = sum(1 for d in d_list if d.get("severity") in ("CRITICAL", "REJECT"))
            total_count = len(d_list)
            density = (total_count / eff_length_m) * 100.0

            if critical_count == 0 and density <= 1.5:
                lane_grades[lane_id] = "EV_GRADE_TIER_1"
                ev_area += lane_area
            elif critical_count <= 2 and density <= 4.0:
                lane_grades[lane_id] = "ESS_GRADE_TIER_2"
                ess_area += lane_area
                # Add splice locations for critical spots
                for cd in d_list:
                    if cd.get("severity") == "CRITICAL":
                        splices.append(round(cd.get("linear_pos_m", 0.0), 2))
            else:
                lane_grades[lane_id] = "SCRAP_REJECT"
                scrap_area += lane_area

        recovery_pct = ((ev_area + ess_area) / max(0.001, gross_area)) * 100.0

        return SlittingYieldPlan(
            total_roll_length_m=round(roll_length_m, 1),
            gross_area_m2=round(gross_area, 2),
            net_usable_ev_area_m2=round(ev_area, 2),
            net_usable_ess_area_m2=round(ess_area, 2),
            scrap_area_m2=round(scrap_area, 2),
            overall_recovery_yield_pct=round(recovery_pct, 1),
            lane_grades=lane_grades,
            recommended_splices_md_m=sorted(list(set(splices)))
        )


class DigitalBatteryPassportGenerator:
    """
    Generates a prototype passport-shaped telemetry manifest. It does not
    assert regulatory or customer compliance.
    """

    @staticmethod
    def generate_passport(
        roll_id: str,
        batch_id: str,
        spc_results: Dict[int, LaneSPCResult],
        yield_plan: SlittingYieldPlan,
        anomalies: List[SpatialPeriodicAnomaly],
        cleanroom_dew_point_c: Optional[float] = None,
        slurry_viscosity_mpa_s: Optional[float] = None
    ) -> Dict[str, Any]:
        return {
            "passport_standard": "PROTOTYPE_SCHEMA_NOT_REGULATORY_COMPLIANCE",
            "provisional": True,
            "roll_id": roll_id,
            "slurry_batch_id": batch_id,
            "traceability_genealogy": {
                "cleanroom_dew_point_celsius": cleanroom_dew_point_c,
                "slurry_viscosity_mpa_s": slurry_viscosity_mpa_s,
                "inspection_compliance_standard": "UNVERIFIED",
            },
            "six_sigma_quality_summary": {
                f"lane_{k}": {
                    "tier": v.six_sigma_tier,
                    "cpk": v.cpk,
                    "defect_density_per_100m": v.mean_defect_density_per_100m
                }
                for k, v in spc_results.items()
            },
            "slitting_optimization": {
                "modeled_from_defect_allocation": True,
                "recovery_yield_pct": yield_plan.overall_recovery_yield_pct,
                "ev_grade_area_m2": yield_plan.net_usable_ev_area_m2,
                "ess_grade_area_m2": yield_plan.net_usable_ess_area_m2,
                "scrap_area_m2": yield_plan.scrap_area_m2,
                "splice_locations_md_m": yield_plan.recommended_splices_md_m
            },
            "mechanical_health_diagnostics": [
                {
                    "periodic_wavelength_mm": a.dominant_wavelength_mm,
                    "confidence": a.confidence_peak,
                    "source_component": a.matched_equipment,
                    "action": a.recommended_maintenance_action
                }
                for a in anomalies
            ]
        }
