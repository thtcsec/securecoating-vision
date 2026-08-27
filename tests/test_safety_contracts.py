"""Regression tests for fail-closed safety and certificate integrity contracts."""

import os
import sys
import time
import unittest
import numpy as np
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from inference.failsafe import FailSafeManager, SystemState
from industrial.protocol_manager import IndustrialProtocolManager
from traceability.roll_certificate import RollCertificateGenerator
from industrial.web_synchronizer import WebSynchronizer


class SlowPredictor:
    def predict(self, optical, thermal, height):
        time.sleep(0.05)
        return {"segmentation_mask": optical[:, :, 0] * 0}


class FailingPredictor:
    def predict(self, optical, thermal, height):
        raise RuntimeError("primary inference failed")


class HealthyPredictor:
    def predict(self, optical, thermal, height):
        return {
            "segmentation_mask": np.zeros(optical.shape[:2], dtype=np.uint8),
            "untrained_fallback": False,
        }


class ErrorResponse:
    def isError(self):
        return True


class ErrorModbusClient:
    def __init__(self, *args, **kwargs):
        pass

    def connect(self):
        return True

    def write_register(self, *args, **kwargs):
        return ErrorResponse()

    def close(self):
        pass


class RegisterResponse:
    def __init__(self, registers=None):
        self.registers = registers or []

    def isError(self):
        return False


class AckingModbusClient:
    registers = {}

    def __init__(self, *args, **kwargs):
        self.__class__.registers = {}

    def connect(self):
        return True

    def write_register(self, address, value, **kwargs):
        self.registers[address] = value
        if address == 1010:
            self.registers[1011] = value
        return RegisterResponse()

    def read_holding_registers(self, address, **kwargs):
        return RegisterResponse([self.registers.get(address, 0)])

    def close(self):
        pass


class TestSafetyContracts(unittest.TestCase):
    def test_inference_deadline_returns_emergency(self):
        manager = FailSafeManager(max_inference_timeout_ms=5.0)
        numpy = __import__("numpy")
        optical = numpy.zeros((8, 8, 3), dtype="uint8")
        optical[::2] = 100
        started = time.monotonic()
        result = manager.safe_predict(SlowPredictor(), optical=optical)
        elapsed = time.monotonic() - started
        self.assertEqual(result["system_state"], SystemState.EMERGENCY.value)
        self.assertEqual(result["error_reason"], "Inference timeout")
        self.assertLess(elapsed, 0.05)
        second = manager.safe_predict(SlowPredictor(), optical=optical)
        self.assertEqual(second["error_reason"], "Inference circuit open after timeout")

    def test_runtime_model_failure_cannot_recover_to_decision_state(self):
        manager = FailSafeManager(max_inference_timeout_ms=100.0)
        optical = np.zeros((8, 8, 3), dtype="uint8")
        optical[::2] = 100
        result = manager.safe_predict(FailingPredictor(), optical=optical)
        self.assertEqual(result["system_state"], SystemState.EMERGENCY.value)
        self.assertNotEqual(result.get("system_state"), SystemState.OPTIMAL.value)
        blocked = manager.safe_predict(HealthyPredictor(), optical=optical)
        self.assertIn("operator reset required", blocked["error_reason"])
        self.assertTrue(manager.inference_interlock_latched)
        self.assertTrue(manager.request_inference_reset_probe())
        recovered = manager.safe_predict(HealthyPredictor(), optical=optical)
        self.assertEqual(recovered["system_state"], SystemState.OPTIMAL.value)
        self.assertFalse(manager.inference_interlock_latched)

    def test_safety_failure_latches_hold_even_in_mock_mode(self):
        manager = IndustrialProtocolManager({"enabled": True, "mock_mode": True})
        result = manager.process_inspection_result(
            part_id="PART_UNINSPECTED",
            batch_id="BATCH_UNINSPECTED",
            defects=[],
            grade_result={"passed": True, "reject_reasons": []},
            safety_permitted=False,
            safety_reasons=["Inference timeout"],
        )
        self.assertEqual(result["gate_action"], "HOLD")
        self.assertTrue(manager.interlock_latched)
        self.assertFalse(manager.signal_history[-1].acknowledged)

    def test_estop_blocks_later_pass_until_confirmed_reset(self):
        manager = IndustrialProtocolManager({"enabled": True, "mock_mode": True})
        estop = manager.emergency_stop("test")
        self.assertFalse(estop.acknowledged)
        result = manager.process_inspection_result(
            part_id="PART_AFTER_ESTOP",
            batch_id="BATCH",
            defects=[],
            grade_result={"passed": True, "reject_reasons": []},
        )
        self.assertEqual(result["gate_action"], "HOLD")

    def test_production_plc_failure_holds_part_without_ack(self):
        manager = IndustrialProtocolManager({"enabled": True, "mock_mode": False})
        self.assertTrue(manager.interlock_latched)
        self.assertIn("Startup PLC state is unverified", manager.get_plc_state()["interlock_reason"])
        result = manager.process_inspection_result(
            part_id="PART_FAIL",
            batch_id="BATCH_FAIL",
            defects=[{"class_name": "delamination", "area_mm2": 2.0}],
            grade_result={"passed": False, "reject_reasons": ["defect"]},
        )
        self.assertEqual(result["gate_action"], "HOLD")
        self.assertEqual(manager.signal_history[-1].metadata["delivery_status"], "FAILED")
        self.assertFalse(manager.signal_history[-1].acknowledged)

    def test_modbus_exception_response_is_not_acknowledged(self):
        manager = IndustrialProtocolManager({
            "enabled": True,
            "mock_mode": False,
            "modbus": {"trusted_gateway": True},
        })
        with patch("industrial.protocol_manager.ModbusTcpClient", ErrorModbusClient):
            confirmed = manager._write_modbus("PART_ERROR", manager.should_reject([])["action"])
        self.assertFalse(confirmed)

    def test_modbus_ack_requires_matching_command_sequence(self):
        manager = IndustrialProtocolManager({
            "enabled": True,
            "mock_mode": False,
            "command_channel": "modbus",
            "modbus": {
                "trusted_gateway": True,
                "register_command_sequence": 1010,
                "register_ack_sequence": 1011,
            },
        })
        with patch("industrial.protocol_manager.ModbusTcpClient", AckingModbusClient):
            confirmed = manager._write_modbus(
                "PART_ACK", manager.should_reject([])["action"], command_id=73
            )
        self.assertTrue(confirmed)

    def test_certificate_detects_nested_payload_tampering(self):
        cert = RollCertificateGenerator.build_certificate(
            roll_id="ROLL_SAFE",
            batch_id="BATCH_SAFE",
            inspected_length_m=100.0,
            total_length_m=1000.0,
            defect_records=[{"class_name": "scratch", "lane_id": 1}],
        )
        self.assertTrue(cert.verify_signature())
        cert.defect_map_entries[0]["lane_id"] = 4
        self.assertFalse(cert.verify_signature())

    def test_certificate_source_is_not_truncated_to_ui_cache(self):
        synchronizer = WebSynchronizer(max_defect_history=2)
        context = synchronizer.advance_motion(dt_seconds=0.001)
        coordinate = synchronizer.map_defect_to_physical_coordinate(
            context, 10, 10, frame_width_px=100, frame_height_px=100
        )
        for index in range(3):
            synchronizer.record_defect_on_roll(
                {"defect_id": f"D{index}", "class_name": "scratch"}, coordinate
            )
        self.assertEqual(len(synchronizer.recent_roll_defects), 2)
        self.assertEqual(len(synchronizer.roll_defect_map), 3)
        self.assertEqual(synchronizer.roll_defect_map[0]["defect_id"], "D0")


if __name__ == "__main__":
    unittest.main()
