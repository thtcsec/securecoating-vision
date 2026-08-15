"""
SecureCoating-Vision: Optical & Throughput Engineering Budget
============================================================
Physical optical chain derivation from lens specifications, working distance,
sensor geometry, and line-scan throughput requirements.

Physics Formulations:
1. Optical Magnification: M = f / (WD - f)
2. Object-Space Pixel Pitch: p_x = p_sensor / M,  p_y = p_x (square pixel aspect)
3. Single Camera FOV: FOV_1 = N_pixels * p_x
4. Two-Camera Overlap: Web Coverage = 2*FOV_1 - Overlap_cam
5. Required Line Rate: f_line = v_web / p_y
6. Exposure Time: t_exp <= (blur_fraction * p_y) / v_web
7. Sustained DMA Bandwidth: B = N_total * 3 bytes * f_line
"""

import math
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class OpticalSpecifications:
    """Lens, sensor, and industrial mechanical setup specifications."""
    sensor_pixel_size_um: float = 7.04    # 7.04 um physical pixel size (e.g. Teledyne DALSA Linea 8K)
    pixels_per_camera: int = 8192         # 8K Line-Scan Sensor
    num_cameras: int = 2                  # 2 cameras covering cross-web
    lens_focal_length_mm: float = 50.0    # 50 mm standard industrial line-scan lens
    working_distance_mm: float = 350.0    # Distance from front lens vertex to web surface
    web_width_mm: float = 650.0           # Nominal electrode foil width
    inter_camera_overlap_fraction: float = 0.05  # 5% overlap between adjacent camera FOVs
    target_motion_blur_fraction: float = 0.40   # Max allowable motion blur (< 0.50 pixel)
    illumination_lux: float = 350_000.0   # Strobed high-power line illuminator


@dataclass
class OpticalCalibrationModel:
    """Unified calibration object shared across optical, metrology, and pipeline engines."""
    x_um_per_px: float = 42.24
    y_um_per_px: float = 42.24
    distortion_coefficient_k1: float = 0.0001
    magnification: float = 0.1667

    @property
    def pixel_to_mm_ratio(self) -> float:
        """Convert pixel dimensions to millimeters."""
        return self.x_um_per_px / 1000.0


class OpticalThroughputBudgetEngine:
    """
    Derives physical optical parameters and validates feasibility across operating envelopes.
    """

    def __init__(self, specs: OpticalSpecifications = None):
        self.specs = specs or OpticalSpecifications()

    def derive_calibration(self) -> OpticalCalibrationModel:
        """Derive optical calibration from physical lens focal length and working distance."""
        s = self.specs
        # Thin-lens optical magnification: M = f / (WD - f)
        effective_wd = max(s.lens_focal_length_mm * 1.5, s.working_distance_mm)
        magnification = s.lens_focal_length_mm / (effective_wd - s.lens_focal_length_mm)
        
        # Object-space pixel pitch in micrometers
        obj_pixel_pitch_um = s.sensor_pixel_size_um / max(0.01, magnification)
        
        return OpticalCalibrationModel(
            x_um_per_px=round(obj_pixel_pitch_um, 2),
            y_um_per_px=round(obj_pixel_pitch_um, 2),
            magnification=round(magnification, 4)
        )

    def compute_budget(self, line_speed_m_s: float = 1.8) -> Dict[str, Any]:
        """Compute full optical, exposure, line rate, and bandwidth throughput budget."""
        s = self.specs
        calib = self.derive_calibration()
        p_x_um = calib.x_um_per_px
        p_y_um = calib.y_um_per_px
        
        # Single camera FOV in object space
        fov_single_camera_mm = (s.pixels_per_camera * p_x_um) / 1000.0
        
        # Total two-camera coverage with 5% inter-camera overlap
        # Total Coverage = 2 * FOV - (0.05 * FOV) = 1.95 * FOV
        overlap_mm = fov_single_camera_mm * s.inter_camera_overlap_fraction
        total_optical_coverage_mm = (s.num_cameras * fov_single_camera_mm) - overlap_mm
        
        # Required Line Rate for isotropic 1:1 square pixel acquisition
        p_y_m = p_y_um * 1e-6
        required_line_rate_hz = line_speed_m_s / p_y_m
        line_period_us = (1.0 / required_line_rate_hz) * 1e6
        
        # Maximum exposure time to maintain motion blur <= target fraction
        max_exposure_us = (s.target_motion_blur_fraction * p_y_m / line_speed_m_s) * 1e6
        recommended_exposure_us = round(min(max_exposure_us, line_period_us * 0.70), 2)
        actual_motion_blur_px = round((line_speed_m_s * (recommended_exposure_us * 1e-6)) / p_y_m, 3)
        
        # Raw Data Bandwidth across PCIe bus (RGB 24-bit)
        total_pixels = s.num_cameras * s.pixels_per_camera
        raw_bandwidth_mb_s = round((total_pixels * 3 * required_line_rate_hz) / (1024 * 1024), 1)
        raw_bandwidth_gb_s = round(raw_bandwidth_mb_s / 1024.0, 2)
        
        # D95 Detectability Threshold (Empirical contrast-to-noise ratio + MTF50 model)
        d95_detectable_size_um = round(2.8 * p_x_um, 1)

        is_coverage_sufficient = total_optical_coverage_mm >= s.web_width_mm
        is_bandwidth_feasible = raw_bandwidth_mb_s <= 3500.0  # PCIe Gen3 x4 DMA capacity (~3.5 GB/s)
        is_line_rate_feasible = required_line_rate_hz <= 100_000.0  # 100 kHz sensor limit

        return {
            "line_speed_m_s": line_speed_m_s,
            "magnification_ratio": calib.magnification,
            "spatial_resolution_x_um_per_px": p_x_um,
            "spatial_resolution_y_um_per_px": p_y_um,
            "single_camera_fov_mm": round(fov_single_camera_mm, 1),
            "inter_camera_overlap_mm": round(overlap_mm, 1),
            "total_optical_coverage_mm": round(total_optical_coverage_mm, 1),
            "web_width_mm": s.web_width_mm,
            "is_coverage_sufficient": is_coverage_sufficient,
            "required_line_rate_khz": round(required_line_rate_hz / 1000.0, 2),
            "line_period_us": round(line_period_us, 2),
            "recommended_exposure_time_us": recommended_exposure_us,
            "motion_blur_pixels": actual_motion_blur_px,
            "d95_detectability_threshold_um": d95_detectable_size_um,
            "raw_bandwidth_mb_s": raw_bandwidth_mb_s,
            "raw_bandwidth_gb_s": raw_bandwidth_gb_s,
            "pcie_dma_interface": "PCIe Gen3 x4 (Sustained DMA 3.5 GB/s)",
            "is_theoretically_feasible": (is_coverage_sufficient and is_bandwidth_feasible and is_line_rate_feasible),
            "feasibility_notes": f"Optical coverage {total_optical_coverage_mm:.1f}mm covers {s.web_width_mm}mm web with {actual_motion_blur_px}px motion blur."
        }
