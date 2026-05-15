"""Tests for validation: ingest, register, score, AM-Bench stub."""

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
    cosine_similarity,
    extract_field_features,
    iou_mask,
    load_measurement_set,
    load_pyrometer_csv,
    load_thermal_npz,
    register_rigid_2d,
    rmse,
    score_thermal_frame,
)
from laser_sim.validation.am_bench import KNOWN_RELEASES, load_am_bench_release
from laser_sim.validation.register import resample_field_to_grid


@pytest.fixture
def scenario() -> ScenarioConfig:
    path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return ScenarioConfig.model_validate(data)


def _write_synthetic_measurement(dir_: Path, scenario: ScenarioConfig) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    H, W = 32, 32
    cx, cy = H // 2, W // 2
    yy, xx = np.meshgrid(np.arange(W), np.arange(H), indexing="ij")
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    frame = scenario.machine.preheat_K + 2500 * np.exp(-r2 / 60.0)
    np.savez(
        dir_ / "thermal_001.npz",
        frame_K=frame,
        pixel_um=125.0,
        origin_xy_mm=np.array([-2.0, -2.0]),
        sensor="synth_coax",
        timestamp_s=0.5,
    )
    (dir_ / "pyrometer_001.csv").write_text(
        "time_s,temperature_K,sensor\n"
        "0.0,1500\n"
        "0.1,2200\n"
        "0.2,1800\n"
    )
    (dir_ / "metadata.json").write_text(
        json.dumps(
            {
                "experiment_id": "synth_exp_42",
                "scenario_id": str(scenario.scenario_id),
                "notes": "synthetic for harness test",
            }
        )
    )


def test_load_thermal_npz_round_trip(tmp_path: Path, scenario: ScenarioConfig) -> None:
    _write_synthetic_measurement(tmp_path, scenario)
    frame = load_thermal_npz(tmp_path / "thermal_001.npz")
    assert frame.frame_K.shape == (32, 32)
    assert frame.pixel_um == 125.0
    assert frame.origin_xy_mm == (-2.0, -2.0)
    assert frame.sensor == "synth_coax"


def test_load_pyrometer_csv(tmp_path: Path, scenario: ScenarioConfig) -> None:
    _write_synthetic_measurement(tmp_path, scenario)
    pyro = load_pyrometer_csv(tmp_path / "pyrometer_001.csv")
    assert pyro.time_s.shape == pyro.temperature_K.shape == (3,)
    assert pyro.temperature_K[1] == pytest.approx(2200.0)


def test_load_measurement_set(tmp_path: Path, scenario: ScenarioConfig) -> None:
    _write_synthetic_measurement(tmp_path, scenario)
    ms = load_measurement_set(tmp_path)
    assert ms.experiment_id == "synth_exp_42"
    assert len(ms.items) == 2
    assert ms.first_thermal() is not None
    assert ms.first_pyrometer() is not None


def test_register_rigid_2d_recovers_known_transform() -> None:
    rng = np.random.default_rng(0)
    src = rng.uniform(-5, 5, size=(8, 2))
    angle = 17.0
    rad = np.deg2rad(angle)
    R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    t = np.array([1.5, -0.7])
    dst = src @ R.T + t
    fit = register_rigid_2d(src, dst)
    assert fit.rotation_deg == pytest.approx(angle, abs=0.01)
    assert fit.translation_mm[0] == pytest.approx(1.5, abs=0.01)
    assert fit.translation_mm[1] == pytest.approx(-0.7, abs=0.01)
    assert fit.rms_residual_mm < 1e-6


def test_rmse_iou_cosine_basics() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert rmse(a, b) == 0.0
    assert iou_mask(a > 0, b > 0) == pytest.approx(1.0)
    assert cosine_similarity(a, b) == pytest.approx(1.0)
    c = np.zeros_like(a)
    assert iou_mask(a > 0, c > 0) == 0.0


def test_extract_field_features_consistency(scenario: ScenarioConfig) -> None:
    rng = np.random.default_rng(1)
    field = scenario.machine.preheat_K + rng.uniform(0, 3000, size=(40, 40))
    feats = extract_field_features(
        field, liquidus_K=scenario.material.liquidus_K, boiling_K=scenario.material.boiling_K
    )
    assert feats.t_max_max_K >= feats.t_max_p95_K >= feats.t_max_p50_K
    assert 0.0 <= feats.coverage_fraction <= 1.0
    assert 0.0 <= feats.keyhole_fraction <= 1.0
    assert feats.histogram_K.sum() == pytest.approx(1.0, abs=1e-6)


def test_resample_field_to_grid_identity() -> None:
    src_x = np.linspace(-1, 1, 11)
    src_y = np.linspace(-1, 1, 11)
    sx, sy = np.meshgrid(src_x, src_y, indexing="ij")
    field = sx + sy
    out = resample_field_to_grid(field, src_x, src_y, src_x, src_y)
    assert np.allclose(out, field, atol=1e-9)


def test_score_thermal_frame_self_distance_zero(scenario: ScenarioConfig) -> None:
    spec = PrimitiveSpec(
        kind=PrimitiveKind.ZIGZAG, power_W=200, speed_mm_s=800, hatch_um=100, spot_um=80
    )
    pat = build_primitive(spec, scenario.roi)
    ev = PatternEvaluator(scenario.material, scenario.machine, scenario.roi, mode="field", grid_n=21, stride=4)
    out = ev.evaluate(pat)
    assert out.field is not None
    score = score_thermal_frame(
        out.field.t_max_K,
        out.field.t_max_K,
        liquidus_K=scenario.material.liquidus_K,
        boiling_K=scenario.material.boiling_K,
    )
    assert score.rmse_K == pytest.approx(0.0, abs=1e-6)
    assert score.iou_melt == pytest.approx(1.0)
    assert score.cosine_hist == pytest.approx(1.0, rel=1e-6)
    assert score.composite < 1e-3


def test_score_thermal_frame_distinguishes_different_fields(scenario: ScenarioConfig) -> None:
    rng = np.random.default_rng(2)
    base = scenario.machine.preheat_K + rng.uniform(0, 3000, size=(30, 30))
    perturbed = base + rng.normal(0, 500, size=base.shape)
    score = score_thermal_frame(
        base,
        perturbed,
        liquidus_K=scenario.material.liquidus_K,
        boiling_K=scenario.material.boiling_K,
    )
    assert score.rmse_K > 100.0
    assert score.composite > 0.0


def test_am_bench_known_releases_metadata() -> None:
    assert len(KNOWN_RELEASES) >= 2
    for rel in KNOWN_RELEASES:
        assert rel.name.startswith("AMB")
        assert "nist.gov" in rel.citation


def test_am_bench_loader_unimplemented_raises() -> None:
    with pytest.raises(NotImplementedError):
        load_am_bench_release("AMB2018-02")
    with pytest.raises(KeyError):
        load_am_bench_release("AMB-NOT-A-THING")
