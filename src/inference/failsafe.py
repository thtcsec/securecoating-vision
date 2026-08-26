"""
SecureCoating-Vision: Fail-Safe & Graceful Degradation Module
==============================================================
Implements production-grade fault tolerance for the inspection system:

1. Sensor Disconnection Handling: Auto-detects when thermal/3D sensors go offline
   and seamlessly falls back to RGB-only detection with reduced capabilities.
   
2. Data Validation: Rejects corrupt or anomalous frames before they reach the model.

3. Model Inference Guard: Catches OOM, timeout, and runtime errors during inference
   and triggers emergency fallback modes.

4. Health Monitoring: Tracks system health metrics and triggers alerts when
   degradation thresholds are exceeded.
"""

import time
import logging
import traceback
import threading
import queue
import numpy as np
import cv2
from typing import Optional, Dict, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("SecureCoatingVision.FailSafe")


class SystemState(Enum):
    """System operational states."""
    OPTIMAL = "OPTIMAL"           # All sensors + model running normally
    DEGRADED = "DEGRADED"         # Some sensors offline, fallback active
    EMERGENCY = "EMERGENCY"       # Critical failure, minimal operation
    OFFLINE = "OFFLINE"           # System cannot process


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


@dataclass
class HealthMetrics:
    """Real-time health metrics for the inspection system."""
    state: SystemState = SystemState.OPTIMAL
    rgb_sensor_ok: bool = True
    thermal_sensor_ok: bool = True
    profiler_sensor_ok: bool = True
    model_inference_ok: bool = True
    last_successful_inference_ms: float = 0.0
    consecutive_failures: int = 0
    total_fallback_activations: int = 0
    uptime_seconds: float = 0.0
    alerts: list = field(default_factory=list)

    def add_alert(self, level: AlertLevel, message: str):
        """Record a system alert."""
        alert = {
            "level": level.value,
            "message": message,
            "timestamp": time.time()
        }
        self.alerts.append(alert)
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        logger.log(
            logging.CRITICAL if level == AlertLevel.EMERGENCY else
            logging.WARNING if level in (AlertLevel.WARNING, AlertLevel.CRITICAL) else
            logging.INFO,
            f"[ALERT-{level.value}] {message}"
        )


class FailSafeManager:
    """
    Manages fail-safe logic for the coating inspection pipeline.
    
    Wraps the inference pipeline with:
    - Automatic sensor failure detection and fallback
    - Frame quality validation gates
    - Inference timeout and error recovery
    - Health monitoring and alerting
    
    Usage:
        failsafe = FailSafeManager(max_inference_timeout_ms=5000)
        result = failsafe.safe_predict(predictor, rgb, thermal, height)
    """

    # Thresholds for frame quality validation
    MIN_FRAME_STD = 2.0          # Minimum pixel stddev (reject blank/saturated frames)
    MAX_FRAME_STD = 120.0        # Maximum pixel stddev (reject extreme noise)
    MIN_FRAME_MEAN = 10.0        # Reject near-black frames
    MAX_FRAME_MEAN = 245.0       # Reject near-white (saturated) frames
    MAX_CONSECUTIVE_FAILURES = 5  # Failures before EMERGENCY state

    def __init__(
        self,
        max_inference_timeout_ms: float = 5000.0,
        enable_frame_validation: bool = True,
        enable_auto_recovery: bool = True
    ):
        """
        Initialize the fail-safe manager.
        
        Args:
            max_inference_timeout_ms: Maximum allowed inference time before timeout.
            enable_frame_validation: Whether to validate frames before inference.
            enable_auto_recovery: Whether to automatically attempt recovery after failures.
        """
        self.max_inference_timeout_ms = max_inference_timeout_ms
        self.enable_frame_validation = enable_frame_validation
        self.enable_auto_recovery = enable_auto_recovery
        self.health = HealthMetrics()
        self._start_time = time.time()
        self._state_lock = threading.RLock()
        self._active_inference_worker: Optional[threading.Thread] = None

        logger.info(f"FailSafeManager initialized (timeout={max_inference_timeout_ms}ms)")

    @property
    def system_state(self) -> SystemState:
        """Get current system state."""
        return self.health.state

    def validate_frame(self, frame: np.ndarray, sensor_name: str = "RGB") -> Dict[str, Any]:
        """
        Validate a sensor frame for quality issues.
        
        Checks for:
        - Blank/dead frames (all zeros or all same value)
        - Saturated frames (all max values)
        - Excessive noise or corruption
        - Wrong dimensions
        
        Args:
            frame: Input frame to validate.
            sensor_name: Name of the sensor for logging.
            
        Returns:
            Dict with 'valid' (bool), 'reason' (str), and 'metrics' (dict).
        """
        if frame is None:
            return {"valid": False, "reason": f"{sensor_name}: Frame is None", "metrics": {}}

        if frame.size == 0:
            return {"valid": False, "reason": f"{sensor_name}: Empty frame", "metrics": {}}

        # Compute statistics
        frame_float = frame.astype(np.float32)
        mean_val = float(np.mean(frame_float))
        std_val = float(np.std(frame_float))
        min_val = float(np.min(frame_float))
        max_val = float(np.max(frame_float))

        metrics = {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "shape": frame.shape
        }

        # Check for dead frame (no variation)
        if std_val < self.MIN_FRAME_STD:
            return {
                "valid": False,
                "reason": f"{sensor_name}: Dead frame (std={std_val:.2f} < {self.MIN_FRAME_STD})",
                "metrics": metrics
            }

        # Check for excessive noise
        if std_val > self.MAX_FRAME_STD and sensor_name == "RGB":
            return {
                "valid": False,
                "reason": f"{sensor_name}: Excessive noise (std={std_val:.2f} > {self.MAX_FRAME_STD})",
                "metrics": metrics
            }

        # Check for saturation
        if sensor_name == "RGB":
            if mean_val < self.MIN_FRAME_MEAN:
                return {
                    "valid": False,
                    "reason": f"{sensor_name}: Under-exposed (mean={mean_val:.1f})",
                    "metrics": metrics
                }
            if mean_val > self.MAX_FRAME_MEAN:
                return {
                    "valid": False,
                    "reason": f"{sensor_name}: Over-exposed/saturated (mean={mean_val:.1f})",
                    "metrics": metrics
                }

        return {"valid": True, "reason": "OK", "metrics": metrics}

    def simulate_sensor_disconnect(self, sensor: str) -> None:
        """
        Simulate a sensor disconnection for testing fail-safe behavior.
        
        Args:
            sensor: One of 'thermal', 'profiler', 'rgb'.
        """
        if sensor == "thermal":
            self.health.thermal_sensor_ok = False
            self.health.add_alert(
                AlertLevel.WARNING,
                "LWIR Thermal Camera disconnected - switching to RGB-only detection"
            )
        elif sensor == "profiler":
            self.health.profiler_sensor_ok = False
            self.health.add_alert(
                AlertLevel.WARNING,
                "3D Laser Profiler disconnected - height measurements unavailable"
            )
        elif sensor == "rgb":
            self.health.rgb_sensor_ok = False
            self.health.add_alert(
                AlertLevel.EMERGENCY,
                "Primary RGB camera disconnected - INSPECTION HALTED"
            )

        self._update_system_state()

    def simulate_sensor_reconnect(self, sensor: str) -> None:
        """Simulate a sensor coming back online."""
        if sensor == "thermal":
            self.health.thermal_sensor_ok = True
            self.health.add_alert(AlertLevel.INFO, "LWIR Thermal Camera reconnected")
        elif sensor == "profiler":
            self.health.profiler_sensor_ok = True
            self.health.add_alert(AlertLevel.INFO, "3D Laser Profiler reconnected")
        elif sensor == "rgb":
            self.health.rgb_sensor_ok = True
            self.health.add_alert(AlertLevel.INFO, "Primary RGB camera reconnected")

        self._update_system_state()

    def _update_system_state(self):
        """Update system state based on current health metrics."""
        if not self.health.rgb_sensor_ok:
            self.health.state = SystemState.OFFLINE
        elif self.health.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            self.health.state = SystemState.EMERGENCY
        elif not self.health.thermal_sensor_ok or not self.health.profiler_sensor_ok:
            self.health.state = SystemState.DEGRADED
        elif not self.health.model_inference_ok:
            self.health.state = SystemState.DEGRADED
        else:
            self.health.state = SystemState.OPTIMAL

    def decision_permitted(self) -> bool:
        """Return True only when inference and every required sensor are healthy."""
        with self._state_lock:
            self._update_system_state()
            return self.health.state == SystemState.OPTIMAL

    def safe_predict(
        self,
        predictor,
        optical: np.ndarray,
        thermal: Optional[np.ndarray] = None,
        height: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Execute prediction with full fail-safe wrapping.
        
        This method:
        1. Validates input frames
        2. Handles sensor disconnection gracefully
        3. Catches inference errors and activates fallback
        4. Tracks health metrics
        
        Args:
            predictor: CoatingPredictor instance with .predict() method.
            optical: BGR image (H, W, 3) uint8.
            thermal: Optional thermal frame (H, W) float32.
            height: Optional height frame (H, W) float32.
            
        Returns:
            Prediction result dict (same format as predictor.predict()) with
            additional fail-safe metadata.
        """
        self.health.uptime_seconds = time.time() - self._start_time

        # A timed-out Python/GPU call cannot be safely killed. Keep the circuit
        # open until that worker has actually exited so inference jobs never
        # overlap on a shared model/session after a deadline breach.
        with self._state_lock:
            active_worker = self._active_inference_worker
            if active_worker is not None and active_worker.is_alive():
                self.health.model_inference_ok = False
                self.health.add_alert(
                    AlertLevel.EMERGENCY,
                    "Previous timed-out inference is still running; circuit remains open",
                )
                self._update_system_state()
                return self._emergency_result(
                    optical, reason="Inference circuit open after timeout"
                )
            self._active_inference_worker = None

        # --- Gate 1: Primary sensor validation ---
        if self.enable_frame_validation:
            rgb_check = self.validate_frame(optical, "RGB")
            if not rgb_check["valid"]:
                self.health.add_alert(AlertLevel.CRITICAL, rgb_check["reason"])
                self.health.consecutive_failures += 1
                self._update_system_state()
                return self._emergency_result(
                    optical, reason=f"Frame rejected: {rgb_check['reason']}"
                )

        # --- Gate 2: Sensor availability check ---
        # If thermal sensor is offline, nullify thermal input
        if not self.health.thermal_sensor_ok:
            thermal = None
            
        # If profiler is offline, nullify height input  
        if not self.health.profiler_sensor_ok:
            height = None

        # --- Gate 3: Protected inference execution ---
        try:
            start_time = time.time()
            result_queue = queue.Queue(maxsize=1)

            def run_prediction():
                try:
                    result_queue.put((True, predictor.predict(optical, thermal, height)))
                except BaseException as error:
                    result_queue.put((False, error))

            worker = threading.Thread(target=run_prediction, daemon=True)
            with self._state_lock:
                self._active_inference_worker = worker
            worker.start()
            worker.join(timeout=self.max_inference_timeout_ms / 1000.0)
            if worker.is_alive():
                elapsed_ms = (time.time() - start_time) * 1000.0
                self.health.model_inference_ok = False
                self.health.consecutive_failures += 1
                self.health.add_alert(
                    AlertLevel.EMERGENCY,
                    f"Inference deadline exceeded: {elapsed_ms:.0f}ms "
                    f"(limit: {self.max_inference_timeout_ms:.0f}ms)"
                )
                self._update_system_state()
                return self._emergency_result(optical, reason="Inference timeout")

            succeeded, value = result_queue.get_nowait()
            with self._state_lock:
                self._active_inference_worker = None
            if not succeeded:
                raise value
            result = value
            elapsed_ms = (time.time() - start_time) * 1000.0

            # Check for timeout
            if elapsed_ms > self.max_inference_timeout_ms:
                self.health.add_alert(
                    AlertLevel.WARNING,
                    f"Inference slow: {elapsed_ms:.0f}ms (limit: {self.max_inference_timeout_ms}ms)"
                )

            # Success: reset failure counter
            is_trained_model = not result.get("untrained_fallback", False)
            self.health.consecutive_failures = 0 if is_trained_model else 1
            self.health.model_inference_ok = is_trained_model
            self.health.last_successful_inference_ms = elapsed_ms

            # Track fallback activations
            if result.get("fallback_active", False):
                self.health.total_fallback_activations += 1

            self._update_system_state()

            # Enrich result with fail-safe metadata
            result["system_state"] = self.health.state.value
            result["degradation_level"] = self.health.state.value
            result["sensors_online"] = {
                "rgb": self.health.rgb_sensor_ok,
                "thermal": self.health.thermal_sensor_ok,
                "profiler": self.health.profiler_sensor_ok
            }
            return result

        except MemoryError as e:
            # GPU/RAM out of memory
            self.health.model_inference_ok = False
            self.health.consecutive_failures += 1
            self.health.add_alert(
                AlertLevel.EMERGENCY,
                f"Out of Memory during inference: {str(e)}"
            )
            self._update_system_state()
            return self._emergency_result(optical, reason="OOM Error - reduce batch/resolution")

        except RuntimeError as e:
            # CUDA/model runtime error
            self.health.model_inference_ok = False
            self.health.consecutive_failures += 1
            self.health.add_alert(
                AlertLevel.CRITICAL,
                f"Runtime error during inference: {str(e)[:200]}"
            )
            self._update_system_state()

            if self.enable_auto_recovery:
                logger.info("Attempting auto-recovery: RGB-only minimal inference...")
                return self._fallback_rgb_only(predictor, optical)

            return self._emergency_result(optical, reason=str(e)[:200])

        except Exception as e:
            # Catch-all for unexpected errors
            self.health.model_inference_ok = False
            self.health.consecutive_failures += 1
            self.health.add_alert(
                AlertLevel.CRITICAL,
                f"Unexpected error: {type(e).__name__}: {str(e)[:150]}"
            )
            logger.error(f"Unexpected inference error:\n{traceback.format_exc()}")
            self._update_system_state()
            return self._emergency_result(optical, reason=f"{type(e).__name__}: {str(e)[:100]}")

    def _fallback_rgb_only(self, predictor, optical: np.ndarray) -> Dict[str, Any]:
        """Attempt RGB-only inference as recovery mechanism."""
        try:
            result = predictor.predict(optical, thermal=None, height=None)
            result["fallback_active"] = True
            result["system_state"] = SystemState.DEGRADED.value
            result["recovery_mode"] = "RGB-only fallback"
            self.health.total_fallback_activations += 1
            self.health.model_inference_ok = not result.get("untrained_fallback", False)
            self._update_system_state()
            logger.info("Auto-recovery successful: running in RGB-only mode")
            return result
        except Exception as e:
            logger.error(f"RGB-only fallback also failed: {e}")
            return self._emergency_result(optical, reason="All inference paths failed")

    def _emergency_result(self, optical: np.ndarray, reason: str) -> Dict[str, Any]:
        """Generate an emergency passthrough result when all inference fails."""
        h, w = optical.shape[:2]
        return {
            "class_probabilities": [1.0, 0.0, 0.0, 0.0, 0.0],
            "predicted_class_id": 0,
            "segmentation_mask": np.zeros((h, w), dtype=np.uint8),
            "detections": [],
            "latency_ms": 0.0,
            "fallback_active": True,
            "status": "EMERGENCY - Inspection Suspended",
            "system_state": SystemState.EMERGENCY.value,
            "engine": "EMERGENCY",
            "model_version": "n/a",
            "error_reason": reason,
            "sensors_online": {
                "rgb": self.health.rgb_sensor_ok,
                "thermal": self.health.thermal_sensor_ok,
                "profiler": self.health.profiler_sensor_ok
            }
        }

    def get_health_report(self) -> Dict[str, Any]:
        """Generate a comprehensive health report for dashboard/API."""
        self.health.uptime_seconds = time.time() - self._start_time
        return {
            "system_state": self.health.state.value,
            "sensors": {
                "rgb_camera": "ONLINE" if self.health.rgb_sensor_ok else "OFFLINE",
                "thermal_camera": "ONLINE" if self.health.thermal_sensor_ok else "OFFLINE",
                "laser_profiler": "ONLINE" if self.health.profiler_sensor_ok else "OFFLINE",
            },
            "inference": {
                "model_ok": self.health.model_inference_ok,
                "last_latency_ms": round(self.health.last_successful_inference_ms, 2),
                "consecutive_failures": self.health.consecutive_failures,
                "total_fallback_activations": self.health.total_fallback_activations,
            },
            "uptime_seconds": round(self.health.uptime_seconds, 1),
            "recent_alerts": self.health.alerts[-10:]  # Last 10 alerts
        }
