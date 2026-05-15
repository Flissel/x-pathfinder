"""Abstract Machine interface for LPBF hardware bridges.

All concrete machines (MockMachine for offline tests, OpcUaMachine for real
hardware via OPC-UA) implement this protocol. The control plane and CLI
never touch a concrete machine directly — it goes through SafetyChecker
first.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from laser_sim.patterns.base import ScanPattern


@dataclass(frozen=True)
class JobId:
    value: str

    def __str__(self) -> str:
        return self.value


class MachineState(str, Enum):
    IDLE = "idle"
    UPLOADING = "uploading"
    READY = "ready"
    BUILDING = "building"
    ERROR = "error"
    ABORTED = "aborted"
    COMPLETED = "completed"


@dataclass(frozen=True)
class MachineStatus:
    state: MachineState
    job_id: JobId | None
    progress: float  # 0..1
    chamber_o2_ppm: float
    chamber_pressure_mbar: float
    last_error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ThermalFrame:
    """Coaxial / 2-color thermal camera frame in machine frame coordinates."""

    timestamp_s: float
    frame_K: np.ndarray            # (H, W) absolute temperature
    pixel_um: float                # physical size of one pixel in micrometers
    origin_xy_mm: tuple[float, float]
    sensor: str                    # 'coax', '2color', etc.


@dataclass(frozen=True)
class PyrometerSample:
    timestamp_s: float
    temperature_K: float
    integration_window_us: float = 50.0
    sensor: str = "spot_pyrometer"


class Machine(ABC):
    """Abstract base for LPBF machine controllers.

    Concrete implementations live in `hardware/mock.py` (for tests / offline
    closed-loop) and `hardware/opcua_machine.py` (real OPC-UA bridge).
    """

    machine_id: str

    @abstractmethod
    async def upload_pattern(self, pattern: ScanPattern) -> JobId:
        """Validate and stage a pattern; return a stable job identifier."""

    @abstractmethod
    async def start_build(self, job: JobId) -> None:
        """Begin executing an already-uploaded job. Must check pre-conditions."""

    @abstractmethod
    async def abort(self, job: JobId) -> None:
        """Halt build immediately and put machine into a safe state."""

    @abstractmethod
    async def status(self) -> MachineStatus:
        """Current machine state, progress, environment."""

    @abstractmethod
    async def read_thermal(self) -> ThermalFrame | None:
        """Latest coaxial / 2-color thermal frame, or None if unavailable."""

    @abstractmethod
    async def read_pyrometer(self) -> PyrometerSample | None:
        """Latest spot pyrometer sample, or None if unavailable."""
