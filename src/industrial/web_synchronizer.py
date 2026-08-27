"""
SecureCoating-Vision: Industrial Web Synchronizer & Line-Scan Temporal Frame Anchoring
========================================================================================
Prototype continuous roll-to-roll motion simulation with:
1. Simulated 4-state Gray-code quadrature A/B encoder
2. Monotonic frame-context anchoring for software coordinate tests
3. In-process state machine with a strict transition table
4. Separation of Bounded Event Cache from Lifetime Roll Defect Counters
5. Strict Coordinate Validation & Explicit Slitting Lane Boundaries
"""

import time
import math
import logging
from collections import deque, Counter
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
import copy

logger = logging.getLogger("SecureCoatingVision.WebSync")


class RollState(str, Enum):
    """Formal lifecycle state machine for jumbo roll inspection."""
    IDLE = "IDLE"                  # Standby, awaiting roll mount
    LOADED = "LOADED"              # New roll clamped and verified
    RUNNING = "RUNNING"            # Continuous web motion & online inspection
    PAUSED = "PAUSED"              # Temporary line hold (e.g. splice or recipe check)
    ROLL_CHANGE = "ROLL_CHANGE"    # Roll unwind complete, turret indexing
    COMPLETED = "COMPLETED"        # Inspection finished, digital certificate signed
    FAULT = "FAULT"                # Safety E-stop or web break tripped


# Formal State Machine Allowed Transition Graph
VALID_STATE_TRANSITIONS: Dict[RollState, Set[RollState]] = {
    RollState.IDLE: {RollState.LOADED, RollState.FAULT},
    RollState.LOADED: {RollState.RUNNING, RollState.ROLL_CHANGE, RollState.FAULT, RollState.IDLE},
    RollState.RUNNING: {RollState.PAUSED, RollState.COMPLETED, RollState.FAULT, RollState.ROLL_CHANGE},
    RollState.PAUSED: {RollState.RUNNING, RollState.ROLL_CHANGE, RollState.FAULT},
    RollState.ROLL_CHANGE: {RollState.LOADED, RollState.IDLE, RollState.FAULT},
    RollState.COMPLETED: {RollState.ROLL_CHANGE, RollState.IDLE, RollState.LOADED},
    RollState.FAULT: {RollState.IDLE, RollState.LOADED},
}


@dataclass(frozen=True)
class FrameContext:
    """
    Simulated frame acquisition metadata.
    A hardware adapter must supply authoritative encoder/timestamp values in production.
    """
    frame_id: int
    roll_id: str
    encoder_pulses_at_capture: int
    capture_md_pos_m: float
    capture_monotonic_s: float
    capture_epoch_s: float = field(default_factory=time.time)


@dataclass
class WebCoordinate:
    """Physical position of an inspection event on the continuous web."""
    roll_id: str
    frame_id: int
    linear_pos_m: float           # Machine Direction (MD) along web length
    cross_pos_mm: float           # Transverse Direction (TD) across web width
    lane_id: int                  # Slitting lane (1 to num_lanes)
    encoder_pulse_count: int      # Exact encoder count at capture
    capture_monotonic_s: float    # Monotonic hardware timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "roll_id": self.roll_id,
            "frame_id": self.frame_id,
            "linear_pos_m": round(self.linear_pos_m, 4),
            "cross_pos_mm": round(self.cross_pos_mm, 2),
            "lane_id": self.lane_id,
            "encoder_pulse_count": self.encoder_pulse_count,
            "capture_monotonic_s": round(self.capture_monotonic_s, 6),
        }


@dataclass
class RollMetadata:
    """Parameters of the active electrode jumbo roll."""
    roll_id: str = "ROLL_2026_CATL_001"
    batch_id: str = "BATCH_2026_MSE_01"
    total_length_m: float = 1200.0        # Standard 1,200m jumbo roll
    web_width_mm: float = 650.0           # Nominal coated web width
    num_lanes: int = 4                    # 4 slitting lanes (~162.5 mm each)
    coating_type: str = "Cathode_LFP"


class QuadratureEncoderSimulator:
    """
    Simulates an optical incremental quadrature A/B rotary encoder.
    Generates exact 4-state Gray Code transitions:
    Forward:  00 -> 01 -> 11 -> 10 -> 00
    Reverse:  00 -> 10 -> 11 -> 01 -> 00
    """

    QUADRATURE_FORWARD = {0b00: 0b01, 0b01: 0b11, 0b11: 0b10, 0b10: 0b00}
    QUADRATURE_REVERSE = {0b00: 0b10, 0b10: 0b11, 0b11: 0b01, 0b01: 0b00}

    def __init__(self, resolution_um_per_pulse: float = 10.0):
        self.resolution_um = max(0.1, resolution_um_per_pulse)
        self.resolution_nm = int(self.resolution_um * 1000)
        self.total_pulses = 0
        self.raw_nm_accumulator = 0
        self.quadrature_state = 0b00

    def step_distance_m(self, distance_m: float, direction_forward: bool = True) -> int:
        """Advance encoder by physical distance and return new pulses generated."""
        distance_nm = int(abs(distance_m) * 1e9)
        self.raw_nm_accumulator += distance_nm
        
        new_pulses = self.raw_nm_accumulator // self.resolution_nm
        self.raw_nm_accumulator %= self.resolution_nm
        
        if direction_forward:
            self.total_pulses += new_pulses
            for _ in range(new_pulses % 4):
                self.quadrature_state = self.QUADRATURE_FORWARD[self.quadrature_state]
        else:
            self.total_pulses -= new_pulses
            for _ in range(new_pulses % 4):
                self.quadrature_state = self.QUADRATURE_REVERSE[self.quadrature_state]

        return new_pulses

    @property
    def channel_a(self) -> bool:
        return bool(self.quadrature_state & 0b10)

    @property
    def channel_b(self) -> bool:
        return bool(self.quadrature_state & 0b01)

    @property
    def physical_distance_m(self) -> float:
        return (self.total_pulses * self.resolution_um) / 1e6


class WebSynchronizer:
    """
    Thread-safe industrial motion synchronizer tracking jumbo roll inspection.
    """

    MAX_LINE_SPEED_M_S = 3.5

    def __init__(
        self,
        roll_metadata: Optional[RollMetadata] = None,
        encoder_resolution_um: float = 10.0,
        initial_line_speed_m_s: float = 1.8,
        max_defect_history: int = 2000
    ):
        self.roll = roll_metadata or RollMetadata()
        self.encoder = QuadratureEncoderSimulator(resolution_um_per_pulse=encoder_resolution_um)
        
        self.line_speed_m_s = max(0.0, min(initial_line_speed_m_s, self.MAX_LINE_SPEED_M_S))
        self.state = RollState.LOADED
        self._lock = threading.RLock()
        
        self._frame_counter = 0
        self._last_tick_monotonic = time.monotonic()
        
        # Dual-Layer Defect Tracking:
        # Layer A: Bounded FIFO deque for fast real-time UI rendering
        self.recent_defect_cache = deque(maxlen=max_defect_history)
        # Complete in-process ledger used for certificate snapshots. Production
        # deployments should additionally persist these records through the
        # traceability store; the UI cache must never be used as the certificate source.
        self._defect_ledger: List[Dict[str, Any]] = []
        
        # Layer B: Unbounded lifetime statistical counters
        self.lifetime_total_defects = 0
        self.lifetime_defects_by_lane: Counter = Counter({i: 0 for i in range(1, self.roll.num_lanes + 1)})
        self.lifetime_defects_by_class: Counter = Counter()

    def set_state(self, new_state: RollState) -> bool:
        """Attempt state transition against formal state machine graph."""
        with self._lock:
            allowed = VALID_STATE_TRANSITIONS.get(self.state, set())
            if new_state in allowed:
                old_st = self.state
                self.state = new_state
                logger.info(f"Roll State Transition: {old_st.value} -> {new_state.value}")
                return True
            else:
                logger.warning(f"Rejected invalid state transition: {self.state.value} -> {new_state.value}")
                return False

    def set_line_speed(self, speed_m_s: float):
        """Set web travel speed with explicit boundary validation."""
        if not (0.0 <= speed_m_s <= self.MAX_LINE_SPEED_M_S):
            raise ValueError(f"Line speed {speed_m_s} m/s out of range [0.0, {self.MAX_LINE_SPEED_M_S}].")
        with self._lock:
            self.line_speed_m_s = speed_m_s

    def cross_position_to_lane(self, td_pos_mm: float) -> int:
        """Map cross-web Transverse Direction (TD) coordinate to slitting lane [1..N]."""
        if not (0.0 <= td_pos_mm <= self.roll.web_width_mm):
            raise ValueError(f"Cross-web position {td_pos_mm}mm outside web width [0, {self.roll.web_width_mm}mm].")
        
        lane_w = self.roll.web_width_mm / self.roll.num_lanes
        # Handle exact top edge boundary (lane N)
        lane_idx = min(self.roll.num_lanes, int(td_pos_mm / lane_w) + 1)
        return lane_idx

    def advance_motion(self, dt_seconds: Optional[float] = None) -> FrameContext:
        """
        Advance simulated web motion and create a temporal FrameContext anchor.
        """
        with self._lock:
            now = time.monotonic()
            dt = dt_seconds if dt_seconds is not None else max(0.001, now - self._last_tick_monotonic)
            self._last_tick_monotonic = now

            if self.state == RollState.RUNNING or self.state == RollState.LOADED:
                # Bound distance to end of roll
                dist_remaining = max(0.0, self.roll.total_length_m - self.encoder.physical_distance_m)
                step_m = min(self.line_speed_m_s * dt, dist_remaining)
                self.encoder.step_distance_m(step_m, direction_forward=True)

                if self.encoder.physical_distance_m >= self.roll.total_length_m and self.state == RollState.RUNNING:
                    self.state = RollState.COMPLETED
                    logger.info(f"Roll {self.roll.roll_id} completed. Total: {self.encoder.physical_distance_m:.1f} m")

            self._frame_counter += 1
            ctx = FrameContext(
                frame_id=self._frame_counter,
                roll_id=self.roll.roll_id,
                encoder_pulses_at_capture=self.encoder.total_pulses,
                capture_md_pos_m=self.encoder.physical_distance_m,
                capture_monotonic_s=now
            )
            return ctx

    def map_defect_to_physical_coordinate(
        self,
        frame_ctx: FrameContext,
        pixel_x_td: float,
        pixel_y_md: float,
        frame_width_px: int = 1024,
        frame_height_px: int = 1024,
        fov_width_mm: float = 650.0,
        fov_length_m: float = 0.05,
        fov_center_td_mm: Optional[float] = None,
    ) -> WebCoordinate:
        """
        Map 2D tile pixel coordinates to exact physical Machine & Transverse Direction coordinates.
        Rewinds MD position to the exact frame capture anchor!
        """
        if not (0 <= pixel_x_td < frame_width_px):
            raise ValueError(f"Pixel X {pixel_x_td} out of bounds [0, {frame_width_px}).")
        if not (0 <= pixel_y_md < frame_height_px):
            raise ValueError(f"Pixel Y {pixel_y_md} out of bounds [0, {frame_height_px}).")

        # TD Position relative to the calibrated camera footprint.
        center_td = self.roll.web_width_mm / 2.0 if fov_center_td_mm is None else fov_center_td_mm
        fov_left = center_td - fov_width_mm / 2.0
        fov_right = center_td + fov_width_mm / 2.0
        if fov_left < 0.0 or fov_right > self.roll.web_width_mm:
            raise ValueError(
                f"Calibrated FOV [{fov_left}, {fov_right}]mm exceeds web width "
                f"[0, {self.roll.web_width_mm}]mm"
            )
        td_mm = fov_left + (pixel_x_td / frame_width_px) * fov_width_mm
        lane_id = self.cross_position_to_lane(td_mm)

        # MD Position (Machine Direction: Frame Capture MD + Tile Offset)
        tile_offset_m = (pixel_y_md / frame_height_px) * fov_length_m
        exact_md_m = frame_ctx.capture_md_pos_m + tile_offset_m

        return WebCoordinate(
            roll_id=frame_ctx.roll_id,
            frame_id=frame_ctx.frame_id,
            linear_pos_m=exact_md_m,
            cross_pos_mm=td_mm,
            lane_id=lane_id,
            encoder_pulse_count=frame_ctx.encoder_pulses_at_capture,
            capture_monotonic_s=frame_ctx.capture_monotonic_s
        )

    def record_defect_on_roll(self, defect_info: Dict[str, Any], coord: WebCoordinate):
        """
        Atomic ledger transaction updating both lifetime statistical counters and recent cache.
        """
        with self._lock:
            if coord.roll_id != self.roll.roll_id:
                raise ValueError(
                    f"Coordinate roll {coord.roll_id} does not match active roll {self.roll.roll_id}"
                )
            next_id = self.lifetime_total_defects + 1
            record = {
                "defect_id": defect_info.get("defect_id", f"DEF_{next_id}"),
                "frame_id": coord.frame_id,
                "roll_id": coord.roll_id,
                "batch_id": self.roll.batch_id,
                "linear_pos_m": round(coord.linear_pos_m, 4),
                "cross_pos_mm": round(coord.cross_pos_mm, 2),
                "lane_id": coord.lane_id,
                "class_name": defect_info.get("class_name", "unknown"),
                "severity": defect_info.get("severity", "WARNING"),
                "area_mm2": defect_info.get("area_mm2", 0.0),
                "peak_height_um": defect_info.get("peak_height_um", 0.0),
                "timestamp": time.time(),
            }
            # Layer A: Recent UI cache
            self.recent_defect_cache.append(copy.deepcopy(record))
            self._defect_ledger.append(copy.deepcopy(record))
            
            # Layer B: True lifetime counters
            self.lifetime_total_defects += 1
            self.lifetime_defects_by_lane[coord.lane_id] += 1
            self.lifetime_defects_by_class[record["class_name"]] += 1

    def get_roll_defect_summary(self) -> Dict[str, Any]:
        """Compute exact roll-level defect density based on lifetime counters."""
        with self._lock:
            scanned_m = max(0.001, self.encoder.physical_distance_m)
            density_per_100m = (self.lifetime_total_defects / scanned_m) * 100.0
            
            return {
                "roll_id": self.roll.roll_id,
                "batch_id": self.roll.batch_id,
                "state": self.state.value,
                "line_speed_m_s": round(self.line_speed_m_s, 4),
                "inspected_length_m": round(scanned_m, 2),
                "total_roll_length_m": self.roll.total_length_m,
                "total_defects_lifetime": self.lifetime_total_defects,
                "defect_density_per_100m": round(density_per_100m, 2),
                "defects_by_lane": dict(self.lifetime_defects_by_lane),
                "defects_by_class": dict(self.lifetime_defects_by_class),
                "recent_cached_defects_count": len(self.recent_defect_cache),
                "roll_completed": (self.state == RollState.COMPLETED)
            }

    @property
    def current_pos_m(self) -> float:
        return self.encoder.physical_distance_m

    @property
    def roll_defect_map(self) -> List[Dict[str, Any]]:
        """Return a defensive snapshot of the complete active-roll ledger."""
        with self._lock:
            return copy.deepcopy(self._defect_ledger)

    @property
    def recent_roll_defects(self) -> List[Dict[str, Any]]:
        """Return the bounded UI cache without exposing mutable internal state."""
        with self._lock:
            return copy.deepcopy(list(self.recent_defect_cache))

    def get_roll_snapshot(self, roll_id: str, batch_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Atomically snapshot the active roll, rejecting mismatched identities."""
        with self._lock:
            if roll_id != self.roll.roll_id:
                return None
            if batch_id is not None and batch_id != self.roll.batch_id:
                return None
            return {
                "summary": self.get_roll_defect_summary(),
                "defect_records": copy.deepcopy(self._defect_ledger),
            }

    def get_current_coordinate(self, cross_pos_mm: float = 325.0) -> WebCoordinate:
        """Helper for single-point coordinate snapshot."""
        with self._lock:
            validated_cross = max(0.0, min(self.roll.web_width_mm, cross_pos_mm))
            lane = self.cross_position_to_lane(validated_cross)
            return WebCoordinate(
                roll_id=self.roll.roll_id,
                frame_id=self._frame_counter,
                linear_pos_m=self.encoder.physical_distance_m,
                cross_pos_mm=validated_cross,
                lane_id=lane,
                encoder_pulse_count=self.encoder.total_pulses,
                capture_monotonic_s=self._last_tick_monotonic
            )
