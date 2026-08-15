"""
SecureCoating-Vision: Sensor Fusion Manager
=============================================
Manages multi-source data fusion from three sensing modalities:
1. Optical RGB (High-resolution surface camera)
2. LWIR Thermal (Long-Wave Infrared sub-surface imaging)
3. 3D Laser Profilometer (Height/topology measurement)

Since physical sensors are not available in this prototype, this module:
- Generates realistic mock thermal and 3D height data from RGB input
- Performs spatial alignment via homography transformation
- Implements pixel-level and feature-level fusion strategies
- Provides graceful degradation when sensors are unavailable

The mock data generation uses signal processing techniques to produce
physically plausible sensor readings that correlate with visible defects.
"""

import time
import logging
import numpy as np
import cv2
from typing import Optional, Tuple, Dict
from dataclasses import dataclass, field

logger = logging.getLogger("SecureCoatingVision.SensorFusion")


@dataclass
class SensorStatus:
    """Tracks the health state of each sensor in the system."""
    rgb_online: bool = True
    thermal_online: bool = True
    profiler_online: bool = True
    rgb_last_frame_ms: float = 0.0
    thermal_last_frame_ms: float = 0.0
    profiler_last_frame_ms: float = 0.0
    degradation_level: str = "NONE"  # NONE, PARTIAL, CRITICAL

    def update_degradation(self):
        """Compute overall system degradation level."""
        online_count = sum([self.rgb_online, self.thermal_online, self.profiler_online])
        if online_count == 3:
            self.degradation_level = "NONE"
        elif online_count == 2:
            self.degradation_level = "PARTIAL"
        elif online_count == 1:
            self.degradation_level = "CRITICAL"
        else:
            self.degradation_level = "OFFLINE"


@dataclass
class FusionResult:
    """Container for multi-source fusion output."""
    fused_tensor: np.ndarray  # (H, W, 5) float32 normalized [0,1]
    rgb_frame: np.ndarray  # Original RGB (H, W, 3) uint8
    thermal_frame: Optional[np.ndarray] = None  # (H, W) float32 in Celsius
    height_frame: Optional[np.ndarray] = None  # (H, W) float32 in micrometers
    sensor_status: SensorStatus = field(default_factory=SensorStatus)
    fusion_latency_ms: float = 0.0
    alignment_applied: bool = False


class SensorFusionManager:
    """
    Multi-source sensor fusion manager for coating inspection.
    
    Handles:
    - Mock sensor data generation (thermal, 3D height)
    - Spatial alignment via homography matrices
    - Multi-strategy fusion (concatenation, weighted, attention-based)
    - Graceful degradation when sensors disconnect
    
    Usage:
        fusion_mgr = SensorFusionManager()
        result = fusion_mgr.fuse(rgb_image)
        # result.fused_tensor -> (H, W, 5) normalized for model input
        # result.thermal_frame -> simulated thermal data
        # result.height_frame -> simulated 3D height data
    """

    # Physical simulation parameters
    AMBIENT_TEMP_C = 42.0       # Typical coating surface temp during curing
    TEMP_RANGE_C = (15.0, 75.0) # LWIR camera detection range
    HEIGHT_RANGE_UM = (0.0, 500.0)  # Profilometer measurement range
    COATING_THICKNESS_NOM_UM = 120.0  # Nominal coating thickness

    def __init__(
        self,
        target_size: Tuple[int, int] = (1024, 1024),
        thermal_noise_std: float = 0.8,
        height_noise_std: float = 2.0,
        enable_mock: bool = True
    ):
        """
        Initialize the sensor fusion manager.
        
        Args:
            target_size: Output frame dimensions (H, W) after alignment.
            thermal_noise_std: Standard deviation of thermal sensor noise (Celsius).
            height_noise_std: Standard deviation of profilometer noise (micrometers).
            enable_mock: If True, generate simulated sensor data from RGB.
        """
        self.target_size = target_size
        self.thermal_noise_std = thermal_noise_std
        self.height_noise_std = height_noise_std
        self.enable_mock = enable_mock

        # Homography matrices for sensor-to-RGB alignment
        # In production these would be calibrated; here we use near-identity with minor offsets
        self.H_thermal_to_rgb = self._generate_calibration_matrix(
            rotation_deg=0.3, tx=2.0, ty=-1.5
        )
        self.H_profiler_to_rgb = self._generate_calibration_matrix(
            rotation_deg=-0.2, tx=-1.0, ty=1.0
        )

        # Sensor health tracking
        self.sensor_status = SensorStatus()

        logger.info(f"SensorFusionManager initialized (target_size={target_size}, mock={enable_mock})")

    def _generate_calibration_matrix(
        self, rotation_deg: float = 0.0, tx: float = 0.0, ty: float = 0.0
    ) -> np.ndarray:
        """
        Generate a homography matrix simulating minor camera misalignment.
        In a real system, this would be computed from checkerboard calibration.
        """
        angle_rad = np.radians(rotation_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        H = np.array([
            [cos_a, -sin_a, tx],
            [sin_a,  cos_a, ty],
            [0.0,    0.0,   1.0]
        ], dtype=np.float32)

        return H

    def update_calibration(
        self,
        h_thermal: Optional[np.ndarray] = None,
        h_profiler: Optional[np.ndarray] = None
    ):
        """
        Update homography calibration matrices.
        Called when operator performs recalibration via dashboard controls.
        
        Args:
            h_thermal: 3x3 homography matrix for thermal-to-RGB alignment.
            h_profiler: 3x3 homography matrix for profiler-to-RGB alignment.
        """
        if h_thermal is not None:
            self.H_thermal_to_rgb = h_thermal.astype(np.float32)
            logger.info("Thermal-to-RGB homography updated.")
        if h_profiler is not None:
            self.H_profiler_to_rgb = h_profiler.astype(np.float32)
            logger.info("Profiler-to-RGB homography updated.")

    def generate_mock_thermal(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Generate simulated LWIR thermal data from RGB input.
        
        Simulation logic:
        - Base temperature derived from surface luminance
        - Dark regions (potential voids/delamination) show lower temperature
          due to reduced thermal conductivity
        - Bright/reflective regions may indicate thinner coating (higher temp)
        - Gaussian noise simulates sensor measurement uncertainty
        
        Args:
            rgb_image: Input BGR image (H, W, 3) uint8.
            
        Returns:
            Thermal map (H, W) float32 in Celsius.
        """
        h, w = rgb_image.shape[:2]

        # Convert to grayscale for luminance-based temperature estimation
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Map luminance [0-255] to temperature deviation from ambient
        # Darker pixels -> lower thermal conductivity -> cooler spots (voids underneath)
        # Formula: T = ambient + (luminance_norm - 0.5) * thermal_sensitivity
        luminance_norm = gray / 255.0
        thermal_sensitivity = 8.0  # ±8°C variation from defects

        thermal_map = self.AMBIENT_TEMP_C + (luminance_norm - 0.7) * thermal_sensitivity

        # Add edge-based thermal signature (cracks/scratches create thermal bridges)
        edges = cv2.Canny(rgb_image, 50, 150).astype(np.float32) / 255.0
        edges_blurred = cv2.GaussianBlur(edges, (15, 15), 3.0)
        thermal_map -= edges_blurred * 3.0  # Scratches show as cooler lines

        # Simulate sub-surface void detection (low-frequency thermal anomalies)
        # Apply large Gaussian blur to simulate heat diffusion patterns
        low_freq = cv2.GaussianBlur(gray, (51, 51), 15.0)
        void_signal = (low_freq / 255.0 - 0.5) * 4.0
        thermal_map += void_signal

        # Add sensor noise
        noise = np.random.normal(0, self.thermal_noise_std, (h, w)).astype(np.float32)
        thermal_map += noise

        # Clamp to sensor range
        thermal_map = np.clip(thermal_map, self.TEMP_RANGE_C[0], self.TEMP_RANGE_C[1])

        return thermal_map

    def generate_mock_height(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Generate simulated 3D profilometer height data from RGB input.
        
        Simulation logic:
        - Base height is the nominal coating thickness
        - Texture variations create subtle height differences
        - Bright spots could indicate blisters (height bumps)
        - Very dark areas could indicate delamination (height dips)
        - High-frequency noise simulates laser speckle measurement noise
        
        Args:
            rgb_image: Input BGR image (H, W, 3) uint8.
            
        Returns:
            Height map (H, W) float32 in micrometers.
        """
        h, w = rgb_image.shape[:2]

        # Base nominal coating height
        height_map = np.full((h, w), self.COATING_THICKNESS_NOM_UM, dtype=np.float32)

        # Texture-based height variation (subtle surface roughness)
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        texture = cv2.GaussianBlur(gray, (5, 5), 1.0) - gray
        height_map += texture * 0.3  # ±30um from texture

        # Detect potential blisters: bright circular regions = raised bumps
        # Use local brightness peaks as blister indicators
        bright_mask = (gray > 200).astype(np.float32)
        bright_blurred = cv2.GaussianBlur(bright_mask, (31, 31), 8.0)
        height_map += bright_blurred * 80.0  # Blisters can be up to 80um raised

        # Detect potential delamination: large dark patches = coating separation
        dark_mask = (gray < 80).astype(np.float32)
        dark_blurred = cv2.GaussianBlur(dark_mask, (41, 41), 12.0)
        height_map -= dark_blurred * 40.0  # Delamination dips

        # Edge-based scratch detection: sharp edges = surface grooves
        edges = cv2.Canny(rgb_image, 80, 200).astype(np.float32) / 255.0
        edges_dilated = cv2.dilate(edges, np.ones((3, 3)))
        height_map -= edges_dilated * 15.0  # Scratches are 5-15um deep

        # Add measurement noise (laser speckle)
        noise = np.random.normal(0, self.height_noise_std, (h, w)).astype(np.float32)
        height_map += noise

        # Clamp to physical range
        height_map = np.clip(height_map, self.HEIGHT_RANGE_UM[0], self.HEIGHT_RANGE_UM[1])

        return height_map

    def generate_mock_frame(
        self,
        defect_type: str = "random"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic high-resolution multi-modal sensor frames (RGB, Thermal, 3D Height).
        
        Args:
            defect_type: 'scratch', 'void', 'blister', 'delamination', 'none', or 'random'
            
        Returns:
            Tuple of (optical_bgr, thermal_celsius, height_um)
        """
        h, w = self.target_size
        optical = np.ones((h, w, 3), dtype=np.uint8) * 160
        noise = np.random.randint(-10, 10, (h, w, 3), dtype=np.int16)
        optical = np.clip(optical.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        if defect_type == "random":
            defect_type = np.random.choice(["scratch", "void", "blister", "delamination"])

        if defect_type == "scratch":
            cv2.line(optical, (200, 300), (800, 320), (40, 40, 40), 4)
        elif defect_type == "void":
            cv2.circle(optical, (512, 512), 45, (80, 80, 80), -1)
        elif defect_type == "blister":
            cv2.circle(optical, (750, 700), 60, (230, 230, 230), -1)
        elif defect_type == "delamination":
            cv2.ellipse(optical, (350, 650), (90, 50), 30, 0, 360, (50, 50, 50), -1)

        thermal = self.generate_mock_thermal(optical)
        height = self.generate_mock_height(optical)
        return optical, thermal, height

    def align_frame(
        self,
        frame: np.ndarray,
        homography: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Apply spatial alignment (homography warp) to register a sensor frame
        to the RGB camera coordinate system.
        
        Args:
            frame: Input sensor frame (H, W) or (H, W, C).
            homography: 3x3 transformation matrix.
            target_size: Output size (W, H). Defaults to self.target_size.
            
        Returns:
            Aligned frame with same dtype as input.
        """
        if target_size is None:
            target_size = (self.target_size[1], self.target_size[0])  # (W, H) for cv2

        aligned = cv2.warpPerspective(
            frame, homography, target_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        return aligned

    def normalize_thermal(self, thermal: np.ndarray) -> np.ndarray:
        """Normalize thermal data from Celsius to [0, 1] range."""
        t_min, t_max = self.TEMP_RANGE_C
        normalized = (thermal - t_min) / (t_max - t_min)
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    def normalize_height(self, height: np.ndarray) -> np.ndarray:
        """Normalize height data from micrometers to [0, 1] range."""
        h_min, h_max = self.HEIGHT_RANGE_UM
        normalized = (height - h_min) / (h_max - h_min)
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    def fuse(
        self,
        rgb_image: np.ndarray,
        thermal_override: Optional[np.ndarray] = None,
        height_override: Optional[np.ndarray] = None,
        thermal_online: bool = True,
        profiler_online: bool = True
    ) -> FusionResult:
        """
        Perform multi-source data fusion.
        
        Acquires data from all available sensors (real or mocked),
        aligns them spatially, normalizes, and concatenates into a
        5-channel tensor ready for the fusion model.
        
        Args:
            rgb_image: Input BGR image (H, W, 3) uint8.
            thermal_override: Optional real thermal data (bypasses mock).
            height_override: Optional real height data (bypasses mock).
            thermal_online: Whether thermal sensor is connected.
            profiler_online: Whether 3D profiler is connected.
            
        Returns:
            FusionResult containing the fused 5-channel tensor and metadata.
        """
        start_time = time.time()
        h, w = rgb_image.shape[:2]

        # Update sensor status
        self.sensor_status.rgb_online = True
        self.sensor_status.thermal_online = thermal_online
        self.sensor_status.profiler_online = profiler_online
        self.sensor_status.update_degradation()

        # --- Thermal acquisition ---
        thermal_frame = None
        if thermal_online:
            if thermal_override is not None:
                thermal_frame = thermal_override
            elif self.enable_mock:
                thermal_frame = self.generate_mock_thermal(rgb_image)

            # Spatial alignment to RGB frame size (W, H for cv2)
            if thermal_frame is not None:
                thermal_frame = self.align_frame(
                    thermal_frame, self.H_thermal_to_rgb, target_size=(w, h)
                )
        else:
            logger.warning("[SENSOR] LWIR Thermal camera OFFLINE - using zero-fill fallback")

        # --- 3D Profiler acquisition ---
        height_frame = None
        if profiler_online:
            if height_override is not None:
                height_frame = height_override
            elif self.enable_mock:
                height_frame = self.generate_mock_height(rgb_image)

            # Spatial alignment to RGB frame size (W, H for cv2)
            if height_frame is not None:
                height_frame = self.align_frame(
                    height_frame, self.H_profiler_to_rgb, target_size=(w, h)
                )
        else:
            logger.warning("[SENSOR] 3D Laser Profiler OFFLINE - using zero-fill fallback")

        # --- Normalization ---
        # RGB: [0, 255] -> [0, 1]
        rgb_norm = rgb_image.astype(np.float32) / 255.0

        # Thermal: Celsius -> [0, 1] or zero-fill
        if thermal_frame is not None:
            thermal_norm = self.normalize_thermal(thermal_frame)
        else:
            thermal_norm = np.zeros((h, w), dtype=np.float32)

        # Height: micrometers -> [0, 1] or zero-fill
        if height_frame is not None:
            height_norm = self.normalize_height(height_frame)
        else:
            height_norm = np.zeros((h, w), dtype=np.float32)

        # --- Channel Concatenation Fusion ---
        # Stack into 5-channel tensor: [R, G, B, Thermal, Height]
        thermal_expanded = np.expand_dims(thermal_norm, axis=-1)
        height_expanded = np.expand_dims(height_norm, axis=-1)

        fused_tensor = np.concatenate([rgb_norm, thermal_expanded, height_expanded], axis=-1)

        fusion_latency = (time.time() - start_time) * 1000.0
        self.sensor_status.rgb_last_frame_ms = fusion_latency

        return FusionResult(
            fused_tensor=fused_tensor,
            rgb_frame=rgb_image,
            thermal_frame=thermal_frame,
            height_frame=height_frame,
            sensor_status=self.sensor_status,
            fusion_latency_ms=fusion_latency,
            alignment_applied=True
        )

    def compute_fusion_confidence(self, fusion_result: FusionResult) -> Dict[str, float]:
        """
        Compute confidence metrics for the fused data quality.
        
        Evaluates signal-to-noise ratio and cross-modal consistency
        to give operators insight into data reliability.
        
        Returns:
            Dict with per-channel quality scores [0-1].
        """
        scores = {"rgb": 1.0, "thermal": 0.0, "height": 0.0, "overall": 0.0}

        # RGB quality: based on contrast and sharpness
        if fusion_result.rgb_frame is not None:
            gray = cv2.cvtColor(fusion_result.rgb_frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            scores["rgb"] = min(1.0, laplacian_var / 500.0)

        # Thermal quality: based on signal variance (dead sensor = no variance)
        if fusion_result.thermal_frame is not None:
            thermal_std = np.std(fusion_result.thermal_frame)
            scores["thermal"] = min(1.0, thermal_std / 5.0)
        
        # Height quality: based on measurement range utilization
        if fusion_result.height_frame is not None:
            height_range = np.ptp(fusion_result.height_frame)
            scores["height"] = min(1.0, height_range / 50.0)

        # Overall fusion confidence
        weights = [0.4, 0.3, 0.3]  # RGB weighted higher as primary sensor
        available = [scores["rgb"]]
        w_sum = weights[0]
        if fusion_result.sensor_status.thermal_online:
            available.append(scores["thermal"])
            w_sum += weights[1]
        if fusion_result.sensor_status.profiler_online:
            available.append(scores["height"])
            w_sum += weights[2]

        scores["overall"] = sum(available) / max(len(available), 1)

        return scores

    def get_colorized_thermal(self, thermal_frame: np.ndarray) -> np.ndarray:
        """Convert thermal data to JET colormap for visualization."""
        if thermal_frame is None:
            return np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
        normalized = self.normalize_thermal(thermal_frame)
        colored = cv2.applyColorMap(
            (normalized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        return colored

    def get_colorized_height(self, height_frame: np.ndarray) -> np.ndarray:
        """Convert height data to VIRIDIS colormap for visualization."""
        if height_frame is None:
            return np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.uint8)
        normalized = self.normalize_height(height_frame)
        colored = cv2.applyColorMap(
            (normalized * 255).astype(np.uint8),
            cv2.COLORMAP_VIRIDIS
        )
        return colored
