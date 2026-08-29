"""
Unit Tests for Refactored WebSynchronizer & QuadratureEncoderSimulator
"""

import unittest
import time
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from industrial.web_synchronizer import (
    WebSynchronizer, WebCoordinate, RollMetadata, RollState,
    QuadratureEncoderSimulator, FrameContext
)


class TestWebSynchronizerRefactored(unittest.TestCase):

    def setUp(self):
        self.metadata = RollMetadata(
            roll_id="TEST_ROLL_001",
            total_length_m=100.0,
            web_width_mm=650.0,
            num_lanes=4
        )
        self.sync = WebSynchronizer(
            roll_metadata=self.metadata,
            encoder_resolution_um=10.0,
            initial_line_speed_m_s=2.0
        )

    def test_quadrature_gray_code_transitions(self):
        encoder = QuadratureEncoderSimulator(resolution_um_per_pulse=10.0)
        # Advance 15 um -> 1.5 pulses -> should emit 1 pulse, retain 5um in nanometer accumulator
        p1 = encoder.step_distance_m(0.000015, direction_forward=True)
        self.assertEqual(p1, 1)
        self.assertEqual(encoder.total_pulses, 1)
        self.assertEqual(encoder.raw_nm_accumulator, 5000)

        # Advance another 15 um -> total 30 um = 3.0 pulses -> should emit 2 pulses
        p2 = encoder.step_distance_m(0.000015, direction_forward=True)
        self.assertEqual(p2, 2)
        self.assertEqual(encoder.total_pulses, 3)
        self.assertEqual(encoder.raw_nm_accumulator, 0)

    def test_roll_lifecycle_state_transitions(self):
        self.assertEqual(self.sync.state, RollState.LOADED)
        # Transition LOADED -> RUNNING
        self.assertTrue(self.sync.set_state(RollState.RUNNING))
        self.assertEqual(self.sync.state, RollState.RUNNING)

        # Transition RUNNING -> PAUSED
        self.assertTrue(self.sync.set_state(RollState.PAUSED))
        self.assertEqual(self.sync.state, RollState.PAUSED)

        # Invalid transition PAUSED -> COMPLETED should be rejected
        self.assertFalse(self.sync.set_state(RollState.COMPLETED))
        self.assertEqual(self.sync.state, RollState.PAUSED)

    def test_temporal_frame_context_defect_mapping(self):
        self.sync.set_state(RollState.RUNNING)
        frame_ctx = self.sync.advance_motion(dt_seconds=1.0)
        
        # Map defect pixel coordinates (pixel_x = 200 out of 1024 -> td_mm = 126.95 mm -> Lane 1)
        coord = self.sync.map_defect_to_physical_coordinate(
            frame_ctx=frame_ctx,
            pixel_x_td=200.0,
            pixel_y_md=512.0,
            frame_width_px=1024,
            frame_height_px=1024
        )
        self.assertEqual(coord.frame_id, frame_ctx.frame_id)
        self.assertAlmostEqual(coord.linear_pos_m, frame_ctx.capture_md_pos_m + 0.025, places=3)
        self.assertEqual(coord.lane_id, 1)

    def test_coordinate_validation_bounds(self):
        frame_ctx = FrameContext(
            frame_id=1,
            roll_id="ROLL_001",
            encoder_pulses_at_capture=100,
            capture_md_pos_m=1.0,
            capture_monotonic_s=time.monotonic()
        )
        # Invalid pixel bounds
        with self.assertRaises(ValueError):
            self.sync.map_defect_to_physical_coordinate(frame_ctx, -5.0, 500.0)
        with self.assertRaises(ValueError):
            self.sync.map_defect_to_physical_coordinate(frame_ctx, 500.0, 1024.0)

    def test_lifetime_counters_vs_bounded_cache(self):
        coord = self.sync.get_current_coordinate(cross_pos_mm=100.0)
        defect = {"defect_id": "D1", "class_name": "scratch", "area_mm2": 2.0}
        self.sync.record_defect_on_roll(defect, coord)
        
        summary = self.sync.get_roll_defect_summary()
        self.assertEqual(summary["total_defects_lifetime"], 1)
        self.assertEqual(summary["defects_by_lane"][1], 1)
        self.assertEqual(summary["defects_by_class"]["scratch"], 1)

    def test_unverified_coordinates_retain_detection_without_fabricating_position(self):
        frame_ctx = self.sync.advance_motion(dt_seconds=0.0)
        self.sync.record_unlocalized_defect(
            {
                "defect_id": "D_UNLOCALIZED",
                "class_name": "void",
                "bbox": [10, 20, 30, 40],
                "confidence": 0.91,
            },
            frame_ctx,
        )

        snapshot = self.sync.get_roll_snapshot(self.sync.roll.roll_id, self.sync.roll.batch_id)
        record = snapshot["defect_records"][0]
        self.assertIsNone(record["linear_pos_m"])
        self.assertIsNone(record["cross_pos_mm"])
        self.assertIsNone(record["lane_id"])
        self.assertFalse(record["coordinate_verified"])
        self.assertEqual(record["bbox"], [10, 20, 30, 40])
        self.assertEqual(record["confidence"], 0.91)


if __name__ == "__main__":
    unittest.main()
