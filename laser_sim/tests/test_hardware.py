"""Tests for SafetyChecker interlocks and MockMachine async lifecycle."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
import yaml

from laser_sim.config.schema import ScenarioConfig
from laser_sim.hardware import (
    MachineStatus,
    MockMachine,
    SafetyChecker,
    SafetyViolation,
)
from laser_sim.hardware.base import MachineState
from laser_sim.hardware.opcua_machine import OpcUaConfig, opcua_available
from laser_sim.hardware.safety import SafetyConfig
from laser_sim.patterns.base import (
    PrimitiveKind,
    PrimitiveSpec,
    ScanPattern,
    Segment,
    Waypoint,
)
from laser_sim.patterns.primitives import build_primitive


@pytest.fixture
def scenario() -> ScenarioConfig:
    path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return ScenarioConfig.model_validate(data)


def _good_pattern(scenario: ScenarioConfig) -> ScanPattern:
    spec = PrimitiveSpec(
        kind=PrimitiveKind.ZIGZAG,
        power_W=200.0,
        speed_mm_s=800.0,
        hatch_um=100.0,
        spot_um=80.0,
    )
    return build_primitive(spec, scenario.roi)


def test_safety_allows_good_pattern_on_mock(scenario: ScenarioConfig) -> None:
    pat = _good_pattern(scenario)
    safety = SafetyChecker()
    decision = safety.check(
        pat,
        scenario=scenario,
        target_machine_id=scenario.machine.machine_id,
        is_mock=True,
        hatch_mm=0.1,
    )
    assert decision.allowed, decision.violations


def test_safety_blocks_non_mock_without_acknowledge(scenario: ScenarioConfig) -> None:
    pat = _good_pattern(scenario)
    safety = SafetyChecker(SafetyConfig(machine_allowlist=("real_machine",)))
    decision = safety.check(
        pat,
        scenario=scenario,
        target_machine_id="real_machine",
        is_mock=False,
        acknowledge=False,
        hatch_mm=0.1,
    )
    assert not decision.allowed
    joined = " ".join(decision.violations)
    assert "acknowledgement" in joined.lower()


def test_safety_blocks_non_allowlisted_machine(scenario: ScenarioConfig) -> None:
    pat = _good_pattern(scenario)
    safety = SafetyChecker(SafetyConfig(machine_allowlist=("only_this",)))
    decision = safety.check(
        pat,
        scenario=scenario,
        target_machine_id="some_other",
        is_mock=False,
        acknowledge=True,
        hatch_mm=0.1,
    )
    assert not decision.allowed
    assert any("allowlist" in v for v in decision.violations)


def test_safety_rejects_too_high_power(scenario: ScenarioConfig) -> None:
    bad = ScanPattern(
        segments=(
            Segment(
                waypoints=(Waypoint(-1.0, 0.0), Waypoint(1.0, 0.0)),
                power_W=10_000.0,  # well above envelope
                speed_mm_s=800.0,
            ),
        )
    )
    safety = SafetyChecker()
    decision = safety.check(
        bad, scenario=scenario, target_machine_id=scenario.machine.machine_id, is_mock=True
    )
    assert not decision.allowed
    assert any("power" in v for v in decision.violations)


def test_safety_rejects_excessive_energy_density(scenario: ScenarioConfig) -> None:
    # Very low speed + high power -> huge E density, must be rejected
    pat = ScanPattern(
        segments=(
            Segment(
                waypoints=(Waypoint(-1.0, 0.0), Waypoint(1.0, 0.0)),
                power_W=400.0,  # within envelope
                speed_mm_s=100.0,  # within envelope but slow
            ),
        )
    )
    safety = SafetyChecker()
    decision = safety.check(
        pat,
        scenario=scenario,
        target_machine_id=scenario.machine.machine_id,
        is_mock=True,
        hatch_mm=0.05,  # tight hatch -> very high E density
    )
    assert not decision.allowed
    assert any("J/mm" in v for v in decision.violations)


def test_safety_geofence(scenario: ScenarioConfig) -> None:
    pat = ScanPattern(
        segments=(
            Segment(
                waypoints=(
                    Waypoint(0.0, 0.0),
                    Waypoint(scenario.machine.build_x_mm, 0.0),  # outside!
                ),
                power_W=200.0,
                speed_mm_s=800.0,
            ),
        )
    )
    safety = SafetyChecker()
    decision = safety.check(
        pat, scenario=scenario, target_machine_id=scenario.machine.machine_id, is_mock=True
    )
    assert not decision.allowed
    assert any("geofence" in v for v in decision.violations)


def test_safety_rejects_oxygen_over_limit(scenario: ScenarioConfig) -> None:
    pat = _good_pattern(scenario)
    bad_sc = scenario.model_copy(
        update={"machine": scenario.machine.model_copy(update={"chamber_o2_ppm": 5000.0})}
    )
    safety = SafetyChecker()
    decision = safety.check(
        pat, scenario=bad_sc, target_machine_id=bad_sc.machine.machine_id, is_mock=True
    )
    assert not decision.allowed
    assert any("O2" in v for v in decision.violations)


def test_safety_raise_if_blocked() -> None:
    from laser_sim.hardware.safety import SafetyDecision

    d = SafetyDecision(allowed=False, violations=("nope",))
    with pytest.raises(SafetyViolation):
        d.raise_if_blocked()


def test_opcua_stub_documents_config() -> None:
    cfg = OpcUaConfig(endpoint="opc.tcp://localhost:4840", namespace_map={"build_state": "ns=2;i=10"})
    assert cfg.endpoint.startswith("opc.tcp://")
    assert isinstance(opcua_available(), bool)


def _arun(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


def test_mockmachine_lifecycle(scenario: ScenarioConfig) -> None:
    async def _go() -> tuple[MachineStatus, MachineStatus]:
        m = MockMachine(scenario=scenario, time_scale=0.0)
        st0 = await m.status()
        pat = _good_pattern(scenario)
        job = await m.upload_pattern(pat)
        await m.start_build(job)
        st_busy = await m.status()
        return st0, st_busy

    st0, st_busy = asyncio.run(_go())
    assert st0.state == MachineState.IDLE
    # time_scale=0 -> status immediately reports completed once polled while building
    assert st_busy.state in (MachineState.BUILDING, MachineState.COMPLETED)
    assert st_busy.progress >= 0.99


def test_mockmachine_thermal_frame_finite(scenario: ScenarioConfig) -> None:
    async def _go():
        m = MockMachine(scenario=scenario, time_scale=0.0)
        job = await m.upload_pattern(_good_pattern(scenario))
        await m.start_build(job)
        thermal = await m.read_thermal()
        pyro = await m.read_pyrometer()
        return thermal, pyro

    thermal, pyro = asyncio.run(_go())
    assert thermal is not None
    assert thermal.frame_K.ndim == 2
    assert np.isfinite(thermal.frame_K).all()
    assert thermal.frame_K.max() > scenario.machine.preheat_K
    assert pyro is not None
    assert np.isfinite(pyro.temperature_K)


def test_mockmachine_abort(scenario: ScenarioConfig) -> None:
    async def _go():
        m = MockMachine(scenario=scenario, time_scale=0.0)
        job = await m.upload_pattern(_good_pattern(scenario))
        await m.start_build(job)
        await m.abort(job)
        return await m.status()

    st = asyncio.run(_go())
    assert st.state == MachineState.ABORTED
