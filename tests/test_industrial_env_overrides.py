"""Industrial env overlays and live-script honesty checks."""

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial.protocol_manager import apply_industrial_io_env_overrides


class TestIndustrialEnvOverrides(unittest.TestCase):
    def test_default_yaml_stays_simulated_without_env(self):
        config = {
            "enabled": True,
            "mock_mode": True,
            "command_channel": "opc_ua",
            "plc_ip": "192.168.1.100",
            "modbus": {"port": 502, "trusted_gateway": False},
        }
        out = apply_industrial_io_env_overrides(config, environ={})
        self.assertTrue(out["mock_mode"])
        self.assertEqual(out["command_channel"], "opc_ua")
        self.assertFalse(out["modbus"]["trusted_gateway"])
        self.assertEqual(out["modbus"]["port"], 502)

    def test_loopback_env_points_at_localhost_modbus(self):
        config = {
            "enabled": True,
            "mock_mode": True,
            "command_channel": "opc_ua",
            "modbus": {"port": 502, "trusted_gateway": False},
        }
        out = apply_industrial_io_env_overrides(
            config,
            environ={
                "SECURECOATING_INDUSTRIAL_MOCK_MODE": "false",
                "SECURECOATING_COMMAND_CHANNEL": "modbus",
                "SECURECOATING_PLC_IP": "127.0.0.1",
                "SECURECOATING_MODBUS_PORT": "1502",
                "SECURECOATING_MODBUS_TRUSTED_GATEWAY": "true",
                "SECURECOATING_INDUSTRIAL_EVIDENCE_CLASS": "software_loopback_not_vendor_hil",
            },
        )
        self.assertFalse(out["mock_mode"])
        self.assertEqual(out["command_channel"], "modbus")
        self.assertEqual(out["plc_ip"], "127.0.0.1")
        self.assertEqual(out["modbus"]["port"], 1502)
        self.assertTrue(out["modbus"]["trusted_gateway"])
        self.assertEqual(out["evidence_class"], "software_loopback_not_vendor_hil")
        self.assertFalse(config["modbus"]["trusted_gateway"])

    def test_live_trigger_script_does_not_claim_vendor_hil(self):
        source = (ROOT / "scripts/run_modbus_loopback_trigger.py").read_text(encoding="utf-8")
        self.assertIn("from industrial.modbus_loopback import EVIDENCE_CLASS", source)
        self.assertIn('"vendor_hil": False', source)
        self.assertIn("not vendor PLC HIL", source)
        self.assertIn("/api/operations/control", source)
        self.assertIn("/api/inspect", source)


if __name__ == "__main__":
    unittest.main()
