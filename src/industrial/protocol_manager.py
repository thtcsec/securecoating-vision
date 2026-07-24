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

# Try to import pymodbus & asyncua for real industrial communication
try:
    from pymodbus.client import ModbusTcpClient
    pymodbus_available = True
except ImportError:
    pymodbus_available = False

try:
    from asyncua import Client as OpcUaClient
    asyncua_available = True
except ImportError:
    asyncua_available = False

logger = logging.getLogger("SecureCoatingVision.Industrial")


class GateAction(Enum):
    """Physical sorting gate actions."""
    PASS = "PASS"
    REJECT = "REJECT"
    HOLD = "HOLD"  # Hold for manual inspection
    EMERGENCY_STOP = "EMERGENCY_STOP"


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
    
    # OPC UA nodes (simulated)
    opc_reject_node: bool = False       # ns=2;s=Device1.RejectGate
    opc_trigger_node: bool = False      # ns=2;s=Device1.LineTrigger
    opc_line_active: bool = True        # ns=2;s=Device1.LineActive
    opc_last_defect_class: str = "none" # ns=2;s=Device1.LastDefectClass


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
        self.plc_ip = config.get("plc_ip", "192.168.1.100")
        
        # OPC UA config
        opc_config = config.get("opc_ua", {})
        self.opc_endpoint = opc_config.get("endpoint", "opc.tcp://192.168.1.100:4840")
        self.opc_node_reject = opc_config.get("node_id_reject", "ns=2;s=Device1.RejectGate")
        self.opc_node_trigger = opc_config.get("node_id_trigger", "ns=2;s=Device1.LineTrigger")
        
        # Modbus config
        modbus_config = config.get("modbus", {})
        self.modbus_port = modbus_config.get("port", 502)
        self.modbus_slave_id = modbus_config.get("slave_id", 1)
        self.modbus_register_reject = modbus_config.get("register_reject", 1001)
        
        # Internal state
        self.plc_state = PLCState()
        self.signal_history: List[IndustrialSignal] = []
        self._signal_counter = 0
        self._lock = threading.Lock()
        
        # Defect density threshold for automatic rejection
        self.reject_area_threshold_mm2 = 5.0  # Reject if defect area > 5 mm²
        self.reject_density_threshold = 0.02  # Reject if >2% of surface is defective
        
        logger.info(
            f"IndustrialProtocolManager initialized "
            f"(enabled={self.enabled}, mock={self.mock_mode}, plc={self.plc_ip})"
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
            if class_name == "delamination" and area > 1.0:
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

        # --- Simulate OPC UA Write ---
        self._write_opc_ua(part_id, GateAction.REJECT, defect_summary)

        # --- Simulate Modbus TCP Write ---
        self._write_modbus(part_id, GateAction.REJECT)

        latency_ms = (time.time() - start_time) * 1000.0

        # Create signal record
        signal = IndustrialSignal(
            signal_id=signal_id,
            timestamp=datetime.now().isoformat(),
            protocol="OPC_UA + MODBUS_TCP",
            action=GateAction.REJECT.value,
            part_id=part_id,
            batch_id=batch_id,
            defect_summary=defect_summary,
            latency_ms=round(latency_ms, 2),
            acknowledged=True,  # Simulated ACK
            metadata={
                "plc_ip": self.plc_ip,
                "opc_node": self.opc_node_reject,
                "modbus_register": self.modbus_register_reject,
                "gate_response_time_ms": round(latency_ms + 15.0, 1)  # +15ms mechanical delay
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

    def trigger_pass(self, part_id: str, batch_id: str) -> IndustrialSignal:
        """Record a PASS signal (gate remains open, part continues on line)."""
        start_time = time.time()

        with self._lock:
            self._signal_counter += 1
            signal_id = f"SIG_{self._signal_counter:06d}"

        # Reset reject state
        self.plc_state.reject_gate_register = 0
        self.plc_state.opc_reject_node = False
        self.plc_state.batch_counter_register += 1

        latency_ms = (time.time() - start_time) * 1000.0

        signal = IndustrialSignal(
            signal_id=signal_id,
            timestamp=datetime.now().isoformat(),
            protocol="INTERNAL",
            action=GateAction.PASS.value,
            part_id=part_id,
            batch_id=batch_id,
            defect_summary={"count": 0, "classes": [], "max_area_mm2": 0},
            latency_ms=round(latency_ms, 2),
            acknowledged=True
        )

        with self._lock:
            self.signal_history.append(signal)

        return signal

    def _write_opc_ua(self, part_id: str, action: GateAction, defect_info: Dict):
        """
        Simulate OPC UA node write to PLC, or perform actual write if enabled and library available.
        """
        if not self.enabled:
            return

        # Real connection attempt if mock mode is disabled and asyncua is available
        if not self.mock_mode and asyncua_available:
            async def write_opc_node():
                client = OpcUaClient(url=self.opc_endpoint)
                await client.connect()
                try:
                    node = client.get_node(self.opc_node_reject)
                    val = True if action == GateAction.REJECT else False
                    await node.write_value(val)
                    logger.info(f"[OPC UA] Real write: Node {self.opc_node_reject} = {val}")
                finally:
                    await client.disconnect()

            try:
                # Run the coroutine synchronously with timeout protection
                asyncio.run(asyncio.wait_for(write_opc_node(), timeout=1.0))
                
                # Update state
                if action == GateAction.REJECT:
                    self.plc_state.opc_reject_node = True
                    self.plc_state.opc_last_defect_class = defect_info.get("classes", ["unknown"])[0]
                    self.plc_state.defect_counter_register += 1
                elif action == GateAction.PASS:
                    self.plc_state.opc_reject_node = False
                self.plc_state.batch_counter_register += 1
                return
            except Exception as e:
                logger.warning(f"[OPC UA] Real connection failed: {e}. Falling back to mock write.")

        # Mock Mode / Fallback write
        # Simulate network latency (1-5ms to local PLC)
        time.sleep(0.002)

        # Update simulated PLC state
        if action == GateAction.REJECT:
            self.plc_state.opc_reject_node = True
            self.plc_state.opc_last_defect_class = defect_info.get("classes", ["unknown"])[0]
            self.plc_state.defect_counter_register += 1
        elif action == GateAction.PASS:
            self.plc_state.opc_reject_node = False

        self.plc_state.batch_counter_register += 1

        logger.debug(
            f"[OPC UA] Mock Write to {self.opc_endpoint} | "
            f"Node: {self.opc_node_reject} = {action.value} | Part: {part_id}"
        )

    def _write_modbus(self, part_id: str, action: GateAction):
        """
        Simulate Modbus TCP register write, or perform actual write if enabled and library available.
        """
        if not self.enabled:
            return

        # Real connection attempt if mock mode is disabled and pymodbus is available
        if not self.mock_mode and pymodbus_available:
            try:
                client = ModbusTcpClient(self.plc_ip, port=self.modbus_port, timeout=1.0)
                if client.connect():
                    val = 1 if action == GateAction.REJECT else 0
                    client.write_register(self.modbus_register_reject, val, slave=self.modbus_slave_id)
                    client.close()
                    logger.info(f"[MODBUS TCP] Real write: Reg[{self.modbus_register_reject}] = {val} on {self.plc_ip}")
                    
                    # Update simulated registers
                    if action == GateAction.REJECT:
                        self.plc_state.reject_gate_register = 1
                    elif action == GateAction.PASS:
                        self.plc_state.reject_gate_register = 0
                    return
                else:
                    logger.warning(f"[MODBUS TCP] Connection failed to {self.plc_ip}:{self.modbus_port}. Falling back to mock write.")
            except Exception as e:
                logger.warning(f"[MODBUS TCP] Real connection failed: {e}. Falling back to mock write.")

        # Mock Mode / Fallback write
        # Simulate Modbus transaction (3-8ms)
        time.sleep(0.003)

        # Update simulated registers
        if action == GateAction.REJECT:
            self.plc_state.reject_gate_register = 1
        elif action == GateAction.PASS:
            self.plc_state.reject_gate_register = 0

        logger.debug(
            f"[MODBUS TCP] Mock Write to {self.plc_ip}:{self.modbus_port} | "
            f"Slave={self.modbus_slave_id} Reg[{self.modbus_register_reject}] = "
            f"{self.plc_state.reject_gate_register} | Part: {part_id}"
        )

    def process_inspection_result(
        self,
        part_id: str,
        batch_id: str,
        defects: List[Dict],
        grade_result: Dict
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
        # Evaluate rejection criteria
        decision = self.should_reject(defects)
        
        # Also consider the grading result
        if not grade_result.get("passed", True):
            decision["action"] = GateAction.REJECT
            decision["reasons"].extend(grade_result.get("reject_reasons", []))

        # Execute the appropriate action
        if decision["action"] == GateAction.REJECT:
            signal = self.trigger_reject(part_id, batch_id, defects, decision["reasons"])
        else:
            signal = self.trigger_pass(part_id, batch_id)

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
                "status": "CONNECTED (Mock)" if self.mock_mode else "CONNECTED"
            }
        }

    def emergency_stop(self, reason: str = "Manual trigger"):
        """Trigger emergency stop on the production line."""
        self.plc_state.emergency_stop_register = 1
        self.plc_state.opc_line_active = False
        logger.critical(f"[E-STOP] Emergency stop triggered: {reason}")

        with self._lock:
            self._signal_counter += 1
            signal = IndustrialSignal(
                signal_id=f"SIG_{self._signal_counter:06d}",
                timestamp=datetime.now().isoformat(),
                protocol="OPC_UA + MODBUS_TCP",
                action=GateAction.EMERGENCY_STOP.value,
                part_id="SYSTEM",
                batch_id="SYSTEM",
                defect_summary={"reason": reason},
                latency_ms=0.0,
                acknowledged=True,
                metadata={"critical": True}
            )
            self.signal_history.append(signal)

    def reset_line(self):
        """Reset production line after emergency stop."""
        self.plc_state.emergency_stop_register = 0
        self.plc_state.opc_line_active = True
        self.plc_state.reject_gate_register = 0
        self.plc_state.opc_reject_node = False
        logger.info("[LINE RESET] Production line resumed")
