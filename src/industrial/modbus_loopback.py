"""Localhost Modbus TCP gateway emulator for software protocol evidence.

This is a software loopback, not vendor PLC hardware-in-the-loop. It copies a
command-sequence holding register into the ACK register so IndustrialProtocolManager
can exercise connect, readback, and matching ACK against a real TCP socket.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from typing import Optional

from pymodbus.datastore import ModbusSequentialDataBlock, ModbusServerContext, ModbusSlaveContext
from pymodbus.server import ServerStop, StartAsyncTcpServer

EVIDENCE_CLASS = "software_loopback_not_vendor_hil"


class AckingHoldingBlock(ModbusSequentialDataBlock):
    """Holding registers that ACK by echoing the command sequence register."""

    def __init__(self, address, values, command_register: int, ack_register: int):
        super().__init__(address, values)
        self.command_register = int(command_register)
        self.ack_register = int(ack_register)

    def setValues(self, address, values):
        super().setValues(address, values)
        if int(address) == self.command_register:
            super().setValues(self.ack_register, values)


def _free_tcp_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def wait_for_listen(host: str, port: int, timeout_seconds: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Optional[OSError] = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.05)
    raise TimeoutError(f"Modbus loopback did not listen on {host}:{port}: {last_error}")


class SoftwareModbusGateway:
    """Threaded pymodbus TCP server with command/ACK echo."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        command_register: int = 1010,
        ack_register: int = 1011,
        register_count: int = 2048,
    ):
        self.host = host
        self.port = int(port or _free_tcp_port(host))
        self.command_register = int(command_register)
        self.ack_register = int(ack_register)
        self.evidence_class = EVIDENCE_CLASS
        holding = AckingHoldingBlock(
            0,
            [0] * register_count,
            self.command_register,
            self.ack_register,
        )
        slave = ModbusSlaveContext(hr=holding, zero_mode=True)
        self.context = ModbusServerContext(slaves=slave, single=True)
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[BaseException] = None
        self._started = threading.Event()

    async def _serve(self) -> None:
        self._started.set()
        await StartAsyncTcpServer(context=self.context, address=(self.host, self.port))

    def start(self) -> "SoftwareModbusGateway":
        if self._thread is not None:
            return self

        def runner() -> None:
            try:
                asyncio.run(self._serve())
            except Exception as exc:  # pragma: no cover - surfaced by wait_for_listen
                self._error = exc
                self._started.set()

        self._thread = threading.Thread(
            target=runner,
            name="securecoating-modbus-loopback",
            daemon=True,
        )
        self._thread.start()
        if not self._started.wait(timeout=5.0):
            raise TimeoutError("Modbus loopback thread did not start")
        if self._error is not None:
            raise RuntimeError(f"Modbus loopback failed to start: {self._error}")
        try:
            wait_for_listen(self.host, self.port)
        except TimeoutError:
            if self._error is not None:
                raise RuntimeError(f"Modbus loopback failed to start: {self._error}") from self._error
            raise
        return self

    def stop(self) -> None:
        try:
            ServerStop()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def __enter__(self) -> "SoftwareModbusGateway":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
