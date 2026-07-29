"""MockMachine: synthesizes hardware behavior from the fast-sim.

Implements the Machine ABC entirely in-process. Builds "progress" linearly
in pattern_total_time; thermal frames are the field-mode T_max map; pyrometer
samples are sampled from the path's peak T history. Used for closed-loop
tests, regression and `cli closed-loop --machine mock`.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass

import numpy as np

from laser_sim.config.schema import ScenarioConfig
from laser_sim.hardware.base import (
    JobId,
    Machine,
    MachineState,
    MachineStatus,
    PyrometerSample,
    ThermalFrame,
)
from laser_sim.patterns.base import ScanPattern
from laser_sim.patterns.rasterize import rasterize_pattern
from laser_sim.physics.fast_sim.transient import t_max_field


@dataclass
class _Job:
    id: JobId
    pattern: ScanPattern
    started_at: float | None = None
    duration_s: float = 0.0
    aborted: bool = False
    thermal: ThermalFrame | None = None
    peak_T_track: tuple[np.ndarray, np.ndarray] | None = None


class MockMachine(Machine):
    """In-process LPBF emulator backed by the fast-sim proxy."""

    def __init__(self, scenario: ScenarioConfig, time_scale: float = 1.0) -> None:
        """time_scale: wall-clock dilation. 1.0 means real-time emulation; 0.0
        means instant (status jumps to COMPLETED on first poll)."""
        self.machine_id = scenario.machine.machine_id
        self.scenario = scenario
        self.time_scale = max(0.0, time_scale)
        self._jobs: dict[str, _Job] = {}
        self._current: JobId | None = None
        self._state: MachineState = MachineState.IDLE
        self._last_error: str | None = None

    async def upload_pattern(self, pattern: ScanPattern) -> JobId:
        await asyncio.sleep(0)
        self._state = MachineState.UPLOADING
        jid = JobId(value=f"mock-{uuid.uuid4().hex[:10]}")
        rp = rasterize_pattern(pattern, ds_mm=0.05)
        field = t_max_field(
            rp,
            self.scenario.roi,
            material=self.scenario.material,
            machine=self.scenario.machine,
            spot_um=80.0,
            nx=41,
            ny=41,
            stride=4,
        )
        thermal = ThermalFrame(
            timestamp_s=time.time(),
            frame_K=field.t_max_K,
            pixel_um=float(
                (field.grid_x_mm[1] - field.grid_x_mm[0]) * 1000.0
            ),
            origin_xy_mm=(
                float(field.grid_x_mm[0]),
                float(field.grid_y_mm[0]),
            ),
            sensor="coax_synth",
        )
        # pyrometer trace: dense path of peak T proxy (here: power-based proxy
        # at the segment ends, since per-path-point T history needs the full
        # 3D field evaluator)
        if rp.n_samples() > 1:
            # crude proxy: instantaneous T proportional to P/sqrt(v)
            inst_T = (
                self.scenario.machine.preheat_K
                + 0.6
                * self.scenario.material.absorptivity
                * rp.power_W
                / np.sqrt(np.maximum(rp.speed_mm_s, 1.0))
                * 30.0
            )
            peak_T = (rp.t_s, inst_T)
        else:
            peak_T = None
        self._jobs[jid.value] = _Job(
            id=jid,
            pattern=pattern,
            duration_s=pattern.total_time_s(),
            thermal=thermal,
            peak_T_track=peak_T,
        )
        self._state = MachineState.READY
        return jid

    async def start_build(self, job: JobId) -> None:
        await asyncio.sleep(0)
        if job.value not in self._jobs:
            self._last_error = f"unknown job {job.value}"
            self._state = MachineState.ERROR
            raise KeyError(self._last_error)
        if self._state == MachineState.BUILDING and self._current is not None:
            raise RuntimeError(
                f"machine busy with {self._current}; abort before starting another"
            )
        self._jobs[job.value].started_at = time.time()
        self._current = job
        self._state = MachineState.BUILDING

    async def abort(self, job: JobId) -> None:
        await asyncio.sleep(0)
        if job.value in self._jobs:
            self._jobs[job.value].aborted = True
        if self._current is not None and self._current.value == job.value:
            self._current = None
        self._state = MachineState.ABORTED

    async def status(self) -> MachineStatus:
        await asyncio.sleep(0)
        progress = 0.0
        if self._current is not None and self._state == MachineState.BUILDING:
            j = self._jobs[self._current.value]
            if self.time_scale == 0.0 or j.duration_s <= 0:
                progress = 1.0
            else:
                elapsed = time.time() - (j.started_at or time.time())
                progress = min(1.0, elapsed / max(j.duration_s * self.time_scale, 1e-9))
            if progress >= 1.0:
                self._state = MachineState.COMPLETED
        return MachineStatus(
            state=self._state,
            job_id=self._current,
            progress=progress,
            chamber_o2_ppm=self.scenario.machine.chamber_o2_ppm,
            chamber_pressure_mbar=1013.0,
            last_error=self._last_error,
        )

    async def read_thermal(self) -> ThermalFrame | None:
        await asyncio.sleep(0)
        if self._current is None:
            return None
        return self._jobs[self._current.value].thermal

    async def read_pyrometer(self) -> PyrometerSample | None:
        await asyncio.sleep(0)
        if self._current is None:
            return None
        j = self._jobs[self._current.value]
        if j.peak_T_track is None:
            return None
        st = await self.status()
        ts, ts_temp = j.peak_T_track
        if ts.size == 0:
            return None
        idx = int(min(max(st.progress, 0.0), 1.0) * (len(ts) - 1))
        return PyrometerSample(
            timestamp_s=float(ts[idx]),
            temperature_K=float(ts_temp[idx]),
        )
