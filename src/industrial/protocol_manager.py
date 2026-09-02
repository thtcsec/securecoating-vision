"""
SecureCoating-Vision: Industrial Protocol Manager
===================================================
Simulates OPC UA and Modbus TCP communication for PLC-based sorting gate control.

In a real production line, this module would:
- Connect to a Siemens/Rockwell PLC via OPC UA (port 4840)
- Write to Modbus holding registers (port 502) to trigger reject gates
- Read encoder triggers for frame synchronization

This prototype mocks these interactions with realistic timing, logging,
and JSON signaling to demonstrate the full automation flow.

Supports:
- OPC UA node write simulation (reject gate trigger)
- Modbus TCP register write simulation (sorting gate)
- MES (Manufacturing Execution System) REST integration
- Event-driven signaling with configurable latency
"""

import time
import json
import logging
import threading
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import os
from copy import deepcopy
from functools import wraps

# Try to import pymodbus & asyncua for real industrial communication
try:
    from pymodbus.client import ModbusTcpClient
    pymodbus_available = True
except ImportError:
    pymodbus_available = False

try:
    from asyncua import Client as OpcUaClient
    from asyncua import ua as opc_ua_types
    asyncua_available = True
except ImportError:
    opc_ua_types = None
    asyncua_available = False

logger = logging.getLogger("SecureCoatingVision.Industrial")


def _serialized_plc_command(method):
    """Serialize one complete command/ACK/state/history transaction."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._command_lock:
            return method(self, *args, **kwargs)
    return wrapped


def _event_loop_is_running() -> bool:
    """Return whether the current thread is already inside an asyncio loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class GateAction(Enum):
    """Physical sorting gate actions."""
    PASS = "PASS"
    REJECT = "REJECT"
    HOLD = "HOLD"  # Hold for manual inspection
    EMERGENCY_STOP = "EMERGENCY_STOP"
    RESET = "RESET"


class ProtocolType(Enum):
    """Supported industrial protocols."""
    OPC_UA = "OPC_UA"
    MODBUS_TCP = "MODBUS_TCP"
    REST_MES = "REST_MES"


@dataclass
class IndustrialSignal:
    """Represents a signal sent to the PLC/MES system."""
    signal_id: str
    timestamp: str
    protocol: str
    action: str
    part_id: str
    batch_id: str
    defect_summary: Dict[str, Any]
    latency_ms: float
    acknowledged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PLCState:
    """Simulated PLC register state."""
    # Modbus holding registers (simulated)
    reject_gate_register: int = 0       # Register 1001: 0=Pass, 1=Reject
    line_speed_register: int = 200      # Register 1002: Line speed in cm/min
    batch_counter_register: int = 0     # Register 1003: Parts inspected
    defect_counter_register: int = 0    # Register 1004: Total defects
    emergency_stop_register: int = 0    # Register 1005: E-Stop state
    hold_register: int = 0              # Register 1006: Latched manual/safety hold
    
    # OPC UA nodes (simulated)
    opc_reject_node: bool = False       # ns=2;s=Device1.RejectGate
    opc_trigger_node: bool = False      # ns=2;s=Device1.LineTrigger
    opc_line_active: bool = True        # ns=2;s=Device1.LineActive
    opc_last_defect_class: str = "none" # ns=2;s=Device1.LastDefectClass


def apply_industrial_io_env_overrides(
    config: Dict[str, Any],
    environ: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Overlay loopback/runtime PLC settings from the process environment.

    Default YAML stays simulated. Localhost Modbus loopback is opt-in via env.
    """
    env = os.environ if environ is None else environ
    out = deepcopy(config) if isinstance(config, dict) else {}
    out["modbus"] = dict(out.get("modbus") or {})
    out["opc_ua"] = dict(out.get("opc_ua") or {})
    mock_raw = env.get("SECURECOATING_INDUSTRIAL_MOCK_MODE")
    if mock_raw is not None:
        out["mock_mode"] = str(mock_raw).lower() in {"1", "true", "yes"}
    channel = str(env.get("SECURECOATING_COMMAND_CHANNEL") or "").strip().lower()
    if channel:
        out["command_channel"] = channel
    plc_ip = str(env.get("SECURECOATING_PLC_IP") or "").strip()
    if plc_ip:
        out["plc_ip"] = plc_ip
    port_raw = str(env.get("SECURECOATING_MODBUS_PORT") or "").strip()
    if port_raw:
        port = int(port_raw)
        if port <= 0 or port > 65535:
            raise ValueError(f"Invalid SECURECOATING_MODBUS_PORT: {port_raw}")
        out["modbus"]["port"] = port
    if "SECURECOATING_MODBUS_TRUSTED_GATEWAY" in env:
        out["modbus"]["trusted_gateway"] = str(
            env["SECURECOATING_MODBUS_TRUSTED_GATEWAY"]
        ).lower() in {"1", "true", "yes"}
    timeout_raw = str(env.get("SECURECOATING_ACK_TIMEOUT_SECONDS") or "").strip()
    if timeout_raw:
        out["ack_timeout_seconds"] = float(timeout_raw)
    evidence = str(env.get("SECURECOATING_INDUSTRIAL_EVIDENCE_CLASS") or "").strip()
    if evidence:
        out["evidence_class"] = evidence
    return out


class IndustrialProtocolManager:
    """
    Manages simulated industrial communication protocols.
    
    Provides a unified interface for triggering physical sorting gate actions
    based on inspection results. Supports OPC UA and Modbus TCP protocols
    with realistic timing and acknowledgment simulation.
    
    Usage:
        manager = IndustrialProtocolManager(config)
        signal = manager.trigger_reject("PART_001", "BATCH_A", defect_info)
        # Signal is logged and available via get_signal_history()
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the protocol manager.
        
        Args:
            config: Industrial I/O configuration from app.yaml.
                Expected keys: enabled, mock_mode, plc_ip, opc_ua, modbus
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.mock_mode = config.get("mock_mode", True)
        self.evidence_class = str(
            config.get("evidence_class")
            or ("simulated_plc" if self.mock_mode else "")
        )
        self.plc_ip = config.get("plc_ip", "192.168.1.100")
        self.command_channel = str(config.get("command_channel", "opc_ua")).lower()
        self.ack_timeout_seconds = max(
            0.05, min(float(config.get("ack_timeout_seconds", 0.75)), 5.0)
        )
        
        # OPC UA config
        opc_config = config.get("opc_ua", {})
        self.opc_endpoint = opc_config.get("endpoint", "opc.tcp://192.168.1.100:4840")
        self.opc_node_reject = opc_config.get("node_id_reject", "ns=2;s=Device1.RejectGate")
        self.opc_node_trigger = opc_config.get("node_id_trigger", "ns=2;s=Device1.LineTrigger")
        self.opc_node_hold = opc_config.get("node_id_hold", "ns=2;s=Device1.Hold")
        self.opc_node_estop = opc_config.get("node_id_estop", "ns=2;s=Device1.EStop")
        self.opc_node_command_sequence = opc_config.get("node_id_command_sequence", "")
        self.opc_node_ack_sequence = opc_config.get("node_id_ack_sequence", "")
        self.opc_security_policy = opc_config.get("security_policy", "Basic256Sha256")
        self.opc_security_mode = opc_config.get("security_mode", "SignAndEncrypt")
        self.opc_client_certificate = opc_config.get("client_certificate", "")
        self.opc_private_key = opc_config.get("private_key", "")
        self.opc_server_certificate = opc_config.get("server_certificate", "")
        
        # Modbus config
        modbus_config = config.get("modbus", {})
        self.modbus_port = modbus_config.get("port", 502)
        self.modbus_slave_id = modbus_config.get("slave_id", 1)
        self.modbus_register_reject = modbus_config.get("register_reject", 1001)
        self.modbus_register_estop = modbus_config.get("register_estop", 1005)
        self.modbus_register_hold = modbus_config.get("register_hold", 1006)
        self.modbus_register_command_sequence = modbus_config.get("register_command_sequence")
        self.modbus_register_ack_sequence = modbus_config.get("register_ack_sequence")
        self.modbus_trusted_gateway = bool(modbus_config.get("trusted_gateway", False))
        
        # Internal state
        self.plc_state = PLCState()
        self.signal_history: List[IndustrialSignal] = []
        self._signal_counter = 0
        self._lock = threading.RLock()
        # State reads use _lock; every PLC command transaction additionally uses
        # this owner lock from sequence allocation through ACK and history commit.
        self._command_lock = threading.RLock()
        self._interlock_reason = ""
        if not self.mock_mode:
            # Process memory cannot prove the PLC's state after startup/restart.
            # Require an explicit, sequence-acknowledged reset before any release.
            self.plc_state.hold_register = 1
            self.plc_state.reject_gate_register = 1
            self.plc_state.opc_reject_node = True
            self.plc_state.opc_line_active = False
            self._interlock_reason = "Startup PLC state is unverified; confirmed reset required"
        
        # Defect density threshold for automatic rejection
        self.reject_area_threshold_mm2 = 5.0  # Reject if defect area > 5 mm²
        self.reject_density_threshold = 0.02  # Reject if >2% of surface is defective
        
        logger.info(
            f"IndustrialProtocolManager initialized "
            f"(enabled={self.enabled}, mock={self.mock_mode}, plc={self.plc_ip})"
        )

    @property
    def transport_ready(self) -> bool:
        if not self.enabled:
            return False
        if self.mock_mode:
            return True
        opc_files_ready = all(
            os.path.isfile(path)
            for path in (
                self.opc_client_certificate,
                self.opc_private_key,
                self.opc_server_certificate,
            )
        )
        if self.command_channel == "opc_ua":
            return bool(
                asyncua_available
                and opc_files_ready
                and self.opc_node_command_sequence
                and self.opc_node_ack_sequence
            )
        if self.command_channel == "modbus":
            return bool(
                pymodbus_available
                and self.modbus_trusted_gateway
                and self.modbus_register_command_sequence is not None
                and self.modbus_register_ack_sequence is not None
            )
        return False

    def _dispatch_command(
        self,
        part_id: str,
        action: GateAction,
        defect_info: Dict[str, Any],
        command_id: int,
    ) -> tuple[bool, Dict[str, Any]]:
        """Send through exactly one configured command owner and await its explicit ACK."""
        if not self.transport_ready:
            logger.warning(
                "PLC command not dispatched because the configured transport is not ready "
                "(channel=%s, action=%s)",
                self.command_channel,
                action.value,
            )
            return False, {
                "command_channel": self.command_channel.upper(),
                "configuration_error": True,
                "transport_ready": False,
            }
        if self.command_channel == "opc_ua":
            confirmed = self._write_opc_ua(part_id, action, defect_info, command_id)
            return confirmed, {"command_channel": "OPC_UA", "opc_ua_confirmed": confirmed}
        if self.command_channel == "modbus":
            confirmed = self._write_modbus(part_id, action, command_id)
            return confirmed, {"command_channel": "MODBUS_TCP", "modbus_confirmed": confirmed}
        logger.error("Unsupported PLC command_channel=%s", self.command_channel)
        return False, {"command_channel": self.command_channel, "configuration_error": True}

    @property
    def interlock_latched(self) -> bool:
        """Whether a local safety latch forbids normal gate decisions."""
        with self._lock:
            return bool(
                self.plc_state.emergency_stop_register
                or self.plc_state.hold_register
                or not self.plc_state.opc_line_active
            )

    def should_reject(self, defects: List[Dict], inspection_area_mm2: float = 10000.0) -> Dict[str, Any]:
        """
        Evaluate inspection results against rejection criteria.
        
        Decision logic:
        - Any single defect with area > reject_area_threshold_mm2 -> REJECT
        - Total defect area / inspection area > reject_density_threshold -> REJECT
        - Critical defect classes (delamination) always reject regardless of size
        
        Args:
            defects: List of defect dicts from postprocessing.
            inspection_area_mm2: Total inspection area for density calculation.
            
        Returns:
            Dict with 'action' (GateAction), 'reasons', and 'confidence'.
        """
        if not defects:
            return {
                "action": GateAction.PASS,
                "reasons": [],
                "confidence": 1.0,
                "total_defect_area_mm2": 0.0
            }

        reasons = []
        total_area = 0.0
        max_severity = 0.0

        for defect in defects:
            area = defect.get("area_mm2", 0.0)
            total_area += area
            class_name = defect.get("class_name", "unknown")
            
            # Critical defect: always reject
            if class_name in {"delamination", "delamination_crack"} and area > 1.0:
                reasons.append(f"Critical delamination detected ({area:.1f} mm²)")
                max_severity = max(max_severity, 1.0)
                
            # Large area defect
            elif area > self.reject_area_threshold_mm2:
                reasons.append(f"{class_name} exceeds area limit ({area:.1f} > {self.reject_area_threshold_mm2} mm²)")
                max_severity = max(max_severity, 0.8)

        # Density check
        density = total_area / inspection_area_mm2 if inspection_area_mm2 > 0 else 0
        if density > self.reject_density_threshold:
            reasons.append(f"Defect density {density*100:.2f}% exceeds {self.reject_density_threshold*100:.1f}% limit")
            max_severity = max(max_severity, 0.9)

        if reasons:
            action = GateAction.REJECT
        else:
            action = GateAction.PASS

        return {
            "action": action,
            "reasons": reasons,
            "confidence": max_severity if reasons else 1.0,
            "total_defect_area_mm2": round(total_area, 2),
            "defect_density_pct": round(density * 100, 3)
        }

    @_serialized_plc_command
    def trigger_reject(
        self,
        part_id: str,
        batch_id: str,
        defects: List[Dict],
        reasons: List[str]
    ) -> IndustrialSignal:
        """
        Trigger the physical sorting gate to reject a part.
        
        Sends signals via both OPC UA and Modbus TCP (simulated)
        to activate the reject diverter on the production line.
        
        Args:
            part_id: Unique part identifier.
            batch_id: Current batch identifier.
            defects: List of detected defects.
            reasons: Rejection reasons from should_reject().
            
        Returns:
            IndustrialSignal record of the action taken.
        """
        start_time = time.time()
        
        with self._lock:
            self._signal_counter += 1
            signal_id = f"SIG_{self._signal_counter:06d}"

        # Prepare defect summary for PLC/MES
        defect_summary = {
            "count": len(defects),
            "classes": list(set(d.get("class_name", "unknown") for d in defects)),
            "max_area_mm2": max((d.get("area_mm2", 0) for d in defects), default=0),
            "rejection_reasons": reasons
        }

        # --- Write OPC UA and Modbus channels ---
        command_id = self._signal_counter % 65535 or 1
        channels_confirmed, delivery_metadata = self._dispatch_command(
            part_id, GateAction.REJECT, defect_summary, command_id
        )
        if channels_confirmed:
            with self._lock:
                self._apply_confirmed_state(GateAction.REJECT, defect_summary)
                self.plc_state.batch_counter_register += 1
        delivery_status = (
            "SIMULATED" if self.mock_mode else
            "ACKNOWLEDGED" if channels_confirmed else "FAILED"
        )

        latency_ms = (time.time() - start_time) * 1000.0

        # Create signal record
        signal = IndustrialSignal(
            signal_id=signal_id,
            timestamp=datetime.now().isoformat(),
            protocol=delivery_metadata["command_channel"],
            action=GateAction.REJECT.value,
            part_id=part_id,
            batch_id=batch_id,
            defect_summary=defect_summary,
            latency_ms=round(latency_ms, 2),
            acknowledged=channels_confirmed and not self.mock_mode,
            metadata={
                "plc_ip": self.plc_ip,
                "opc_node": self.opc_node_reject,
                "modbus_register": self.modbus_register_reject,
                "gate_response_time_ms": round(latency_ms + 15.0, 1),
                "delivery_status": delivery_status,
                "command_id": command_id,
                **delivery_metadata,
            }
        )

        with self._lock:
            self.signal_history.append(signal)
            # Keep only last 500 signals
            if len(self.signal_history) > 500:
                self.signal_history = self.signal_history[-500:]

        logger.info(
            f"[REJECT SIGNAL] {signal_id} | Part: {part_id} | "
            f"Defects: {defect_summary['count']} | Latency: {latency_ms:.1f}ms"
        )

        return signal

    @_serialized_plc_command
    def trigger_pass(self, part_id: str, batch_id: str) -> IndustrialSignal:
        """Record a PASS signal (gate remains open, part continues on line)."""
        start_time = time.time()

        with self._lock:
            self._signal_counter += 1
            signal_id = f"SIG_{self._signal_counter:06d}"

        command_id = self._signal_counter % 65535 or 1
        channels_confirmed, delivery_metadata = self._dispatch_command(
            part_id, GateAction.PASS, {}, command_id
        )
        if channels_confirmed:
            with self._lock:
                self._apply_confirmed_state(GateAction.PASS, {})
                self.plc_state.batch_counter_register += 1
        delivery_status = (
            "SIMULATED" if self.mock_mode else
            "ACKNOWLEDGED" if channels_confirmed else "FAILED"
        )

        latency_ms = (time.time() - start_time) * 1000.0

        signal = IndustrialSignal(
            signal_id=signal_id,
            timestamp=datetime.now().isoformat(),
            protocol=delivery_metadata["command_channel"],
            action=GateAction.PASS.value,
            part_id=part_id,
            batch_id=batch_id,
            defect_summary={"count": 0, "classes": [], "max_area_mm2": 0},
            latency_ms=round(latency_ms, 2),
            acknowledged=channels_confirmed and not self.mock_mode,
            metadata={
                "delivery_status": delivery_status,
                "command_id": command_id,
                **delivery_metadata,
            }
        )

        with self._lock:
            self.signal_history.append(signal)
            self.signal_history = self.signal_history[-500:]

        return signal

    @_serialized_plc_command
    def trigger_hold(
        self,
        part_id: str,
        batch_id: str,
        reasons: Optional[List[str]] = None,
    ) -> IndustrialSignal:
        """Latch HOLD and request the fail-safe state through the command owner."""
        start_time = time.time()
        reasons = list(reasons or ["Safety interlock requested HOLD"])
        with self._lock:
            self.plc_state.hold_register = 1
            self.plc_state.reject_gate_register = 1
            self.plc_state.opc_reject_node = True
            self._interlock_reason = "; ".join(reasons)
            self._signal_counter += 1
            signal_id = f"SIG_{self._signal_counter:06d}"

        command_id = self._signal_counter % 65535 or 1
        confirmed, delivery_metadata = self._dispatch_command(
            part_id, GateAction.HOLD, {}, command_id
        )
        if confirmed:
            with self._lock:
                self._apply_confirmed_state(GateAction.HOLD, {})

        signal = IndustrialSignal(
            signal_id=signal_id,
            timestamp=datetime.now().isoformat(),
            protocol=delivery_metadata["command_channel"],
            action=GateAction.HOLD.value,
            part_id=part_id,
            batch_id=batch_id,
            defect_summary={"count": 0, "rejection_reasons": reasons},
            latency_ms=round((time.time() - start_time) * 1000.0, 2),
            acknowledged=confirmed and not self.mock_mode,
            metadata={
                "delivery_status": "SIMULATED" if self.mock_mode else (
                    "ACKNOWLEDGED" if confirmed else "FAILED"
                ),
                "command_id": command_id,
                **delivery_metadata,
            },
        )
        with self._lock:
            self.signal_history.append(signal)
            self.signal_history = self.signal_history[-500:]
        return signal

    def _write_opc_ua(
        self,
        part_id: str,
        action: GateAction,
        defect_info: Dict,
        command_id: Optional[int] = None,
    ) -> bool:
        """Write command data then require a matching PLC execution-ACK sequence."""
        if not self.enabled:
            return False

        command_values = {
            GateAction.PASS: [(self.opc_node_reject, False), (self.opc_node_hold, False)],
            GateAction.REJECT: [(self.opc_node_reject, True), (self.opc_node_hold, False)],
            GateAction.HOLD: [(self.opc_node_reject, True), (self.opc_node_hold, True)],
            GateAction.EMERGENCY_STOP: [
                (self.opc_node_reject, True),
                (self.opc_node_hold, True),
                (self.opc_node_estop, True),
            ],
            GateAction.RESET: [
                (self.opc_node_estop, False),
                (self.opc_node_hold, False),
                (self.opc_node_reject, False),
            ],
        }[action]

        if self.mock_mode:
            time.sleep(0.002)
            confirmed = True
        else:
            if not asyncua_available:
                logger.error("[OPC UA] Real mode requested but asyncua is unavailable")
                return False

            if command_id is None or not self.opc_node_command_sequence or not self.opc_node_ack_sequence:
                logger.error("[OPC UA] Explicit command/ACK sequence nodes are required")
                return False
            outcome = {"confirmed": False, "error": None}

            async def write_and_readback():
                client = OpcUaClient(url=self.opc_endpoint)
                security = ",".join(
                    [
                        self.opc_security_policy,
                        self.opc_security_mode,
                        self.opc_client_certificate,
                        self.opc_private_key,
                        self.opc_server_certificate,
                    ]
                )
                await client.set_security_string(security)
                await client.connect()
                try:
                    sequence_node = client.get_node(self.opc_node_command_sequence)
                    ack_node = client.get_node(self.opc_node_ack_sequence)
                    previous_sequence = int(await sequence_node.read_value())
                    previous_ack = int(await ack_node.read_value())
                    if command_id in (previous_sequence, previous_ack):
                        raise RuntimeError(
                            "OPC UA stale sequence collision; retry with a new command ID"
                        )
                    for node_id, expected in command_values:
                        node = client.get_node(node_id)
                        await node.write_value(expected)
                        actual = await node.read_value()
                        if bool(actual) != expected:
                            raise RuntimeError(
                                f"OPC UA readback mismatch for {node_id}: expected {expected}, got {actual}"
                            )
                    await sequence_node.write_value(
                        opc_ua_types.Variant(command_id, opc_ua_types.VariantType.UInt16)
                    )
                    if int(await sequence_node.read_value()) != command_id:
                        raise RuntimeError("OPC UA command sequence readback mismatch")
                    ack_deadline = time.monotonic() + self.ack_timeout_seconds
                    while time.monotonic() < ack_deadline:
                        if int(await ack_node.read_value()) == command_id:
                            return
                        await asyncio.sleep(0.01)
                    raise TimeoutError(f"OPC UA PLC ACK timeout for command {command_id}")
                finally:
                    await client.disconnect()

            def run_write():
                try:
                    asyncio.run(
                        asyncio.wait_for(
                            write_and_readback(), timeout=self.ack_timeout_seconds + 1.0
                        )
                    )
                    outcome["confirmed"] = True
                except BaseException as exc:
                    outcome["error"] = exc

            if _event_loop_is_running():
                worker = threading.Thread(target=run_write, daemon=True)
                worker.start()
                worker.join(timeout=self.ack_timeout_seconds + 1.2)
                if worker.is_alive():
                    logger.error("[OPC UA] Write/readback exceeded deadline")
                    return False
            else:
                run_write()
            confirmed = bool(outcome["confirmed"])
            if not confirmed:
                logger.error(f"[OPC UA] Write/readback failed: {outcome['error']}")
                return False

        logger.debug(f"[OPC UA] {action.value} confirmed={confirmed} part={part_id}")
        return confirmed

    def _write_modbus(
        self, part_id: str, action: GateAction, command_id: Optional[int] = None
    ) -> bool:
        """Write command registers then require a matching gateway ACK sequence."""
        if not self.enabled:
            return False

        command_values = {
            GateAction.PASS: [(self.modbus_register_reject, 0), (self.modbus_register_hold, 0)],
            GateAction.REJECT: [(self.modbus_register_reject, 1), (self.modbus_register_hold, 0)],
            GateAction.HOLD: [(self.modbus_register_reject, 1), (self.modbus_register_hold, 1)],
            GateAction.EMERGENCY_STOP: [
                (self.modbus_register_reject, 1),
                (self.modbus_register_hold, 1),
                (self.modbus_register_estop, 1),
            ],
            GateAction.RESET: [
                (self.modbus_register_estop, 0),
                (self.modbus_register_hold, 0),
                (self.modbus_register_reject, 0),
            ],
        }[action]

        if self.mock_mode:
            time.sleep(0.003)
            return True
        if not pymodbus_available:
            logger.error("[MODBUS TCP] Real mode requested but pymodbus is unavailable")
            return False
        if not self.modbus_trusted_gateway:
            logger.error("[MODBUS TCP] Refusing plaintext control without an explicitly trusted gateway")
            return False
        if (
            command_id is None
            or self.modbus_register_command_sequence is None
            or self.modbus_register_ack_sequence is None
        ):
            logger.error("[MODBUS TCP] Explicit command/ACK sequence registers are required")
            return False

        client = ModbusTcpClient(self.plc_ip, port=self.modbus_port, timeout=1.0)
        try:
            if not client.connect():
                logger.error(f"[MODBUS TCP] Connection failed to {self.plc_ip}:{self.modbus_port}")
                return False
            previous_values = []
            for register in (
                self.modbus_register_command_sequence,
                self.modbus_register_ack_sequence,
            ):
                response = client.read_holding_registers(
                    register, count=1, slave=self.modbus_slave_id
                )
                if response is None or (
                    hasattr(response, "isError") and response.isError()
                ) or not getattr(response, "registers", None):
                    logger.error(
                        "[MODBUS TCP] Preflight read failed for sequence register %s",
                        register,
                    )
                    return False
                previous_values.append(int(response.registers[0]))
            if command_id in previous_values:
                logger.error(
                    "[MODBUS TCP] Stale sequence collision for command %s; retry required",
                    command_id,
                )
                return False
            for register, expected in command_values:
                response = client.write_register(
                    register, expected, slave=self.modbus_slave_id
                )
                if response is None or (
                    hasattr(response, "isError") and response.isError()
                ):
                    logger.error(f"[MODBUS TCP] Write rejected for register {register}")
                    return False
                readback = client.read_holding_registers(
                    register, count=1, slave=self.modbus_slave_id
                )
                if readback is None or (
                    hasattr(readback, "isError") and readback.isError()
                ) or not getattr(readback, "registers", None):
                    logger.error(f"[MODBUS TCP] Readback failed for register {register}")
                    return False
                if int(readback.registers[0]) != expected:
                    logger.error(
                        f"[MODBUS TCP] Readback mismatch for register {register}: "
                        f"expected {expected}, got {readback.registers[0]}"
                    )
                    return False
            sequence_response = client.write_register(
                self.modbus_register_command_sequence,
                command_id,
                slave=self.modbus_slave_id,
            )
            if sequence_response is None or (
                hasattr(sequence_response, "isError") and sequence_response.isError()
            ):
                return False
            ack_deadline = time.monotonic() + self.ack_timeout_seconds
            while time.monotonic() < ack_deadline:
                ack = client.read_holding_registers(
                    self.modbus_register_ack_sequence,
                    count=1,
                    slave=self.modbus_slave_id,
                )
                if (
                    ack is not None
                    and not (hasattr(ack, "isError") and ack.isError())
                    and getattr(ack, "registers", None)
                    and int(ack.registers[0]) == command_id
                ):
                    return True
                time.sleep(0.01)
            logger.error("[MODBUS TCP] PLC ACK timeout for command %s", command_id)
            return False
        except Exception as exc:
            logger.error(f"[MODBUS TCP] Transaction failed: {exc}")
            return False
        finally:
            client.close()

    def _apply_confirmed_state(self, action: GateAction, defect_info: Dict) -> None:
        """Update the local mirror only after OPC UA confirmation or in explicit mock mode."""
        if action == GateAction.PASS:
            self.plc_state.reject_gate_register = 0
            self.plc_state.hold_register = 0
            self.plc_state.opc_reject_node = False
        elif action == GateAction.REJECT:
            self.plc_state.reject_gate_register = 1
            self.plc_state.hold_register = 0
            self.plc_state.opc_reject_node = True
            classes = defect_info.get("classes") or ["unknown"]
            self.plc_state.opc_last_defect_class = classes[0]
            self.plc_state.defect_counter_register += 1
        elif action == GateAction.HOLD:
            self.plc_state.reject_gate_register = 1
            self.plc_state.hold_register = 1
            self.plc_state.opc_reject_node = True
        elif action == GateAction.EMERGENCY_STOP:
            self.plc_state.reject_gate_register = 1
            self.plc_state.hold_register = 1
            self.plc_state.emergency_stop_register = 1
            self.plc_state.opc_reject_node = True
            self.plc_state.opc_line_active = False
        elif action == GateAction.RESET:
            self.plc_state.reject_gate_register = 0
            self.plc_state.hold_register = 0
            self.plc_state.emergency_stop_register = 0
            self.plc_state.opc_reject_node = False
            self.plc_state.opc_line_active = True

    def process_inspection_result(
        self,
        part_id: str,
        batch_id: str,
        defects: List[Dict],
        grade_result: Dict,
        safety_permitted: bool = True,
        safety_reasons: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline: evaluate defects -> decide action -> trigger PLC signal.
        
        This is the main entry point called by the FastAPI endpoint after
        model inference and postprocessing are complete.
        
        Args:
            part_id: Part identifier.
            batch_id: Batch identifier.
            defects: List of defect dicts from postprocessing.
            grade_result: Grading result from grade_coating().
            
        Returns:
            Dict with action taken, signal details, and PLC state.
        """
        # Serialize the complete command transaction. An E-stop/HOLD cannot be
        # interleaved with a PASS/REJECT from another request.
        with self._command_lock:
            interlock_active = self.interlock_latched
            if not safety_permitted or interlock_active:
                reasons = list(safety_reasons or [])
                if interlock_active:
                    reasons.append(self._interlock_reason or "PLC safety interlock is latched")
                decision = {
                    "action": GateAction.HOLD,
                    "reasons": reasons or ["Safety readiness check failed"],
                    "total_defect_area_mm2": sum(float(d.get("area_mm2", 0.0)) for d in defects),
                    "defect_density_pct": 0.0,
                }
                signal = self.trigger_hold(part_id, batch_id, decision["reasons"])
            else:
                decision = self.should_reject(defects)
                if not grade_result.get("passed", True):
                    decision["action"] = GateAction.REJECT
                    decision["reasons"].extend(grade_result.get("reject_reasons", []))

                if decision["action"] == GateAction.REJECT:
                    signal = self.trigger_reject(part_id, batch_id, defects, decision["reasons"])
                else:
                    signal = self.trigger_pass(part_id, batch_id)

                if not signal.acknowledged and not self.mock_mode:
                    failed_action = decision["action"].value
                    decision["action"] = GateAction.HOLD
                    decision["reasons"].append(
                        f"PLC {failed_action} command was not confirmed; HOLD latched"
                    )
                    signal = self.trigger_hold(part_id, batch_id, decision["reasons"])

        return {
            "gate_action": decision["action"].value,
            "signal_id": signal.signal_id,
            "signal_timestamp": signal.timestamp,
            "rejection_reasons": decision["reasons"],
            "total_defect_area_mm2": decision.get("total_defect_area_mm2", 0),
            "defect_density_pct": decision.get("defect_density_pct", 0),
            "plc_state": {
                "reject_register": self.plc_state.reject_gate_register,
                "opc_reject_node": self.plc_state.opc_reject_node,
                "batch_counter": self.plc_state.batch_counter_register,
                "defect_counter": self.plc_state.defect_counter_register,
                "hold_register": self.plc_state.hold_register,
                "estop_register": self.plc_state.emergency_stop_register,
            },
            "communication_latency_ms": signal.latency_ms,
            "protocol": signal.protocol
        }

    def get_signal_history(self, limit: int = 50) -> List[Dict]:
        """Get recent signal history for dashboard display."""
        with self._lock:
            recent = self.signal_history[-limit:]
        return [
            {
                "signal_id": s.signal_id,
                "timestamp": s.timestamp,
                "action": s.action,
                "part_id": s.part_id,
                "batch_id": s.batch_id,
                "defect_count": s.defect_summary.get("count", 0),
                "latency_ms": s.latency_ms,
                "acknowledged": s.acknowledged
            }
            for s in recent
        ]

    def get_plc_state(self) -> Dict[str, Any]:
        """Get current PLC register state for monitoring."""
        return {
            "modbus_registers": {
                f"HR_{self.modbus_register_reject}": self.plc_state.reject_gate_register,
                "HR_1002_line_speed": self.plc_state.line_speed_register,
                "HR_1003_batch_count": self.plc_state.batch_counter_register,
                "HR_1004_defect_count": self.plc_state.defect_counter_register,
                "HR_1005_estop": self.plc_state.emergency_stop_register,
                f"HR_{self.modbus_register_hold}_hold": self.plc_state.hold_register,
            },
            "opc_ua_nodes": {
                self.opc_node_reject: self.plc_state.opc_reject_node,
                self.opc_node_trigger: self.plc_state.opc_trigger_node,
                "ns=2;s=Device1.LineActive": self.plc_state.opc_line_active,
                "ns=2;s=Device1.LastDefectClass": self.plc_state.opc_last_defect_class,
            },
            "connection": {
                "plc_ip": self.plc_ip,
                "opc_endpoint": self.opc_endpoint,
                "modbus_port": self.modbus_port,
                "status": (
                    "SIMULATED" if self.mock_mode else
                    "SOFTWARE_LOOPBACK" if self.evidence_class == "software_loopback_not_vendor_hil" else
                    "SECURELY CONFIGURED" if self.transport_ready else
                    "NOT READY"
                ),
                "evidence_class": self.evidence_class or None,
                "command_channel": self.command_channel,
                "mock_mode": self.mock_mode,
                "transport_ready": self.transport_ready,
            },
            "interlock_latched": self.interlock_latched,
            "interlock_reason": self._interlock_reason,
        }

    def emergency_stop(self, reason: str = "Manual trigger") -> IndustrialSignal:
        """Latch local E-stop immediately, then request and confirm PLC channels."""
        with self._lock:
            self.plc_state.emergency_stop_register = 1
            self.plc_state.hold_register = 1
            self.plc_state.reject_gate_register = 1
            self.plc_state.opc_line_active = False
            self._interlock_reason = reason
        logger.critical(f"[E-STOP] Emergency stop triggered: {reason}")
        return self._dispatch_emergency_stop(reason)

    @_serialized_plc_command
    def _dispatch_emergency_stop(self, reason: str) -> IndustrialSignal:
        started = time.time()
        with self._lock:
            self._signal_counter += 1
            command_id = self._signal_counter % 65535 or 1
        confirmed, delivery_metadata = self._dispatch_command(
            "SYSTEM", GateAction.EMERGENCY_STOP, {}, command_id
        )
        if confirmed:
            with self._lock:
                self._apply_confirmed_state(GateAction.EMERGENCY_STOP, {})

        with self._lock:
            signal = IndustrialSignal(
                signal_id=f"SIG_{self._signal_counter:06d}",
                timestamp=datetime.now().isoformat(),
                protocol=delivery_metadata["command_channel"],
                action=GateAction.EMERGENCY_STOP.value,
                part_id="SYSTEM",
                batch_id="SYSTEM",
                defect_summary={"reason": reason},
                latency_ms=round((time.time() - started) * 1000.0, 2),
                acknowledged=confirmed and not self.mock_mode,
                metadata={
                    "critical": True,
                    "delivery_status": "SIMULATED" if self.mock_mode else (
                        "ACKNOWLEDGED" if confirmed else "FAILED"
                    ),
                    "command_id": command_id,
                    **delivery_metadata,
                },
            )
            self.signal_history.append(signal)
            self.signal_history = self.signal_history[-500:]
        return signal

    @_serialized_plc_command
    def reset_line(self) -> IndustrialSignal:
        """Clear the safety latch only after the command owner confirms reset."""
        started = time.time()
        with self._lock:
            self._signal_counter += 1
            command_id = self._signal_counter % 65535 or 1
        confirmed, delivery_metadata = self._dispatch_command(
            "SYSTEM", GateAction.RESET, {}, command_id
        )
        with self._lock:
            if confirmed:
                self._apply_confirmed_state(GateAction.RESET, {})
                self._interlock_reason = ""
            signal = IndustrialSignal(
                signal_id=f"SIG_{self._signal_counter:06d}",
                timestamp=datetime.now().isoformat(),
                protocol=delivery_metadata["command_channel"],
                action=GateAction.RESET.value,
                part_id="SYSTEM",
                batch_id="SYSTEM",
                defect_summary={},
                latency_ms=round((time.time() - started) * 1000.0, 2),
                acknowledged=confirmed and not self.mock_mode,
                metadata={
                    "delivery_status": "SIMULATED" if self.mock_mode else (
                        "ACKNOWLEDGED" if confirmed else "FAILED"
                    ),
                    "command_id": command_id,
                    **delivery_metadata,
                },
            )
            self.signal_history.append(signal)
            self.signal_history = self.signal_history[-500:]
        if confirmed:
            logger.info("[LINE RESET] Safety latch cleared")
        else:
            logger.error("[LINE RESET] Reset was not confirmed; safety latch remains active")
        return signal
