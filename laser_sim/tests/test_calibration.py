"""Tests for hierarchical calibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from laser_sim.config.schema import ScenarioConfig
from laser_sim.fitness.evaluator import PatternEvaluator
from laser_sim.patterns.base import PrimitiveKind, PrimitiveSpec
from laser_sim.patterns.primitives import build_primitive
from laser_sim.validation import (
    CalibrationProblem,
    DEFAULT_LEVEL1_PARAMETERS,
    Measurement,
    MeasurementSet,
    ParameterBound,
    calibrate,
)
from laser_sim.validation.calibration import _golden_section_search
from laser_sim.validation.ingest import ThermalFrame


@pytest.fixture
def scenario() -> ScenarioConfig:
    path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return ScenarioConfig.model_validate(data)


def _build_synthetic_measurement_from_sim(
    scenario: ScenarioConfig, true_params: dict[str, float]
) -> MeasurementSet:
    """Generate a 'real' measurement by running the sim with known parameters.

    The calibrator should then recover those parameters when started from the
    defaults. This is the cleanest way to unit-test a calibration loop.
    """
    spec = PrimitiveSpec(
        kind=PrimitiveKind.ZIGZAG, power_W=200, speed_mm_s=800, hatch_um=100, spot_um=80
    )
    pat = build_primitive(spec, scenario.roi)
    new_mat = scenario.material.model_copy(
        update={
            "absorptivity": float(true_params.get("absorptivity", scenario.material.absorptivity)),
            "emissivity": float(true_params.get("emissivity", scenario.material.emissivity)),
        }
    )
    sc_true = scenario.model_copy(update={"material": new_mat})
    spot_um = float(true_params.get("spot_um", 80.0))
    ev = PatternEvaluator(
        sc_true.material, sc_true.machine, sc_true.roi, mode="field", grid_n=21, stride=8, spot_um=spot_um
    )
    out = ev.evaluate(pat, hatch_mm=0.1, spot_um=spot_um)
    assert out.field is not None
    pixel_um = float(out.field.grid_x_mm[1] - out.field.grid_x_mm[0]) * 1000.0
    real = ThermalFrame(
        timestamp_s=0.0,
        frame_K=out.field.t_max_K.copy(),
        pixel_um=pixel_um,
        origin_xy_mm=(float(out.field.grid_x_mm[0]), float(out.field.grid_y_mm[0])),
        sensor="synthetic",
        source_uri="synthetic",
    )
    return MeasurementSet(
        experiment_id="synth_calib_test",
        scenario_id=str(scenario.scenario_id),
        items=(Measurement(kind="thermal", payload=real),),
    )


def test_golden_section_quadratic_min() -> None:
    f = lambda x: (x - 0.42) ** 2 + 1.0
    x_star, f_star = _golden_section_search(f, lo=0.0, hi=1.0, tol=1e-4)
    assert x_star == pytest.approx(0.42, abs=1e-3)
    assert f_star == pytest.approx(1.0, abs=1e-6)


def test_calibration_recovers_known_absorptivity(scenario: ScenarioConfig) -> None:
    true_eta = 0.55
    ms = _build_synthetic_measurement_from_sim(scenario, {"absorptivity": true_eta})
    spec = PrimitiveSpec(
        kind=PrimitiveKind.ZIGZAG, power_W=200, speed_mm_s=800, hatch_um=100, spot_um=80
    )
    pat = build_primitive(spec, scenario.roi)
    problem = CalibrationProblem(
        pattern=pat,
        measurements=ms,
        base_scenario=scenario,
        parameters=(ParameterBound("absorptivity", lo=0.10, hi=0.80, default=0.30),),
        hatch_mm=0.1,
        grid_n=21,
        stride=8,
    )
    result = calibrate(problem, rounds=1)
    # the calibrator should pull absorptivity toward the true value (within tol)
    assert abs(result.optimum["absorptivity"] - true_eta) < 0.10
    assert result.final_score.composite < result.initial_score.composite


def test_calibration_full_set_runs(scenario: ScenarioConfig) -> None:
    """End-to-end smoke: full default parameter set + 2 rounds + JSON dump."""
    ms = _build_synthetic_measurement_from_sim(
        scenario, {"absorptivity": 0.5, "spot_um": 110.0, "emissivity": 0.5}
    )
    spec = PrimitiveSpec(
        kind=PrimitiveKind.ZIGZAG, power_W=200, speed_mm_s=800, hatch_um=100, spot_um=80
    )
    pat = build_primitive(spec, scenario.roi)
    problem = CalibrationProblem(
        pattern=pat,
        measurements=ms,
        base_scenario=scenario,
        parameters=DEFAULT_LEVEL1_PARAMETERS,
        hatch_mm=0.1,
        grid_n=21,
        stride=8,
    )
    result = calibrate(problem, rounds=2)
    assert result.final_score.composite <= result.initial_score.composite + 1e-6
    for p in DEFAULT_LEVEL1_PARAMETERS:
        assert p.lo <= result.optimum[p.name] <= p.hi
    # JSON round-trip
    out = scenario.model_copy()  # placeholder for tmp_path
    # use a tmp file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        path = Path(tf.name)
    saved = result.to_json(path)
    payload = json.loads(saved.read_text())
    assert "optimum" in payload
    assert "initial_score" in payload
    assert "final_score" in payload
    assert payload["initial_score"]["composite"] >= payload["final_score"]["composite"]


def test_parameter_bound_validation() -> None:
    with pytest.raises(ValueError):
        ParameterBound("bad", lo=0.0, hi=1.0, default=2.0)
