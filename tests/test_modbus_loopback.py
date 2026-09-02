"""Live localhost Modbus TCP command/ACK loopback. Not vendor PLC HIL."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial.modbus_loopback import EVIDENCE_CLASS, SoftwareModbusGateway
from industrial.protocol_manager import IndustrialProtocolManager


class TestModbusLoopback(unittest.TestCase):
    def test_reset_and_reject_complete_over_real_tcp_socket(self):
        with SoftwareModbusGateway() as gateway:
            self.assertEqual(gateway.evidence_class, EVIDENCE_CLASS)
            manager = IndustrialProtocolManager(
                {
                    "enabled": True,
                    "mock_mode": False,
                    "plc_ip": gateway.host,
                    "command_channel": "modbus",
                    "ack_timeout_seconds": 1.0,
                    "modbus": {
                        "port": gateway.port,
                        "trusted_gateway": True,
                        "register_reject": 1001,
                        "register_estop": 1005,
                        "register_hold": 1006,
                        "register_command_sequence": gateway.command_register,
                        "register_ack_sequence": gateway.ack_register,
                    },
                }
            )
            self.assertTrue(manager.transport_ready)
            self.assertTrue(manager.interlock_latched)
            reset = manager.reset_line()
            self.assertTrue(reset.acknowledged)
            self.assertEqual(reset.metadata["delivery_status"], "ACKNOWLEDGED")
            self.assertEqual(reset.metadata["command_channel"], "MODBUS_TCP")
            self.assertFalse(manager.interlock_latched)
            result = manager.process_inspection_result(
                part_id="LOOPBACK_PART",
                batch_id="LOOPBACK_BATCH",
                defects=[{"class_name": "delamination_crack", "area_mm2": 2.0}],
                grade_result={"passed": False, "reject_reasons": ["delamination"]},
            )
            self.assertEqual(result["gate_action"], "REJECT")
            self.assertEqual(result["protocol"], "MODBUS_TCP")
            self.assertEqual(manager.plc_state.reject_gate_register, 1)
            self.assertEqual(manager.plc_state.hold_register, 0)
            self.assertEqual(EVIDENCE_CLASS, "software_loopback_not_vendor_hil")


if __name__ == "__main__":
    unittest.main()
