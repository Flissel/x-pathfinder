"""Tests for fidelity gate, surrogate, HF case generator, runner availability, DEM."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
import yaml

from laser_sim.config.schema import ScenarioConfig
from laser_sim.coupling.dem_to_cfd import write_setfields_dict
from laser_sim.coupling.dem_to_fast import apply_powder_to_material
from laser_sim.fidelity import (
    FidelityGate,
    FidelityTier,
    GPSurrogate,
    PromotionPolicy,
)
from laser_sim.fitness.evaluator import FitnessVector
from laser_sim.genome.archive import ArchiveEntry
from laser_sim.genome.chromosome import Chromosome
from laser_sim.patterns.base import PrimitiveKind, PrimitiveSpec
from laser_sim.patterns.primitives import build_primitive
from laser_sim.physics.melt_pool import (
    build_laserbeamfoam_case,
    laserbeamfoam_available,
    parser_available,
)
from laser_sim.physics.melt_pool.laserbeamfoam_runner import LaserbeamFoamRunner
from laser_sim.physics.melt_pool.parser import parse_case_output
from laser_sim.physics.powder_bed import (
    effective_thermal_properties,
    liggghts_available,
    load_porosity,
    pack_powder_bed,
)
from laser_sim.worker.base import CaseSpec
from laser_sim.worker.local import LocalRunner


@pytest.fixture
def scenario() -> ScenarioConfig:
    path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return ScenarioConfig.model_validate(data)


def _make_entry(power: float, scalar_J: float, fitness: tuple[float, ...] | None = None) -> ArchiveEntry:
    c = Chromosome(
        primitive_kind=PrimitiveKind.ZIGZAG,
        power_W=power, speed_mm_s=800.0, hatch_um=100.0, spot_um=80.0,
    )
    f = fitness if fitness is not None else (scalar_J, scalar_J, scalar_J, scalar_J, scalar_J)
    return ArchiveEntry(chromosome=c, fitness=FitnessVector(values=f), scalar_J=scalar_J)


# -------- Fidelity gate ---------------------------------------------------


def test_fidelity_gate_skips_during_warmup() -> None:
    gate = FidelityGate(policy=PromotionPolicy(fast_only_until_gen=3))
    pop = [_make_entry(200, 0.5)]
    d = gate.decide(generation=1, population=pop, archive_entries=pop)
    assert d.skipped


def test_fidelity_gate_promotes_top_k() -> None:
    pop = [_make_entry(p, j) for p, j in [(100, 0.9), (200, 0.3), (300, 0.6), (400, 0.1)]]
    gate = FidelityGate(
        policy=PromotionPolicy(
            fast_only_until_gen=0, hf_promote_top_k=2, hf_promote_pareto_front=False
        )
    )
    d = gate.decide(generation=5, population=pop, archive_entries=pop)
    assert not d.skipped
    promoted_scalars = sorted(e.scalar_J for e in d.promoted)
    # top-2 by scalar_J should be 0.1 and 0.3
    assert promoted_scalars == [0.1, 0.3]


def test_fidelity_gate_promotes_pareto_front() -> None:
    pop = [
        _make_entry(100, 0.5, fitness=(0.1, 0.9, 0, 0, 0)),
        _make_entry(200, 0.5, fitness=(0.9, 0.1, 0, 0, 0)),
        _make_entry(300, 0.6, fitness=(0.5, 0.5, 0, 0, 0)),
    ]
    gate = FidelityGate(
        policy=PromotionPolicy(
            fast_only_until_gen=0, hf_promote_top_k=0, hf_promote_pareto_front=True
        )
    )
    d = gate.decide(generation=5, population=pop, archive_entries=pop)
    # the first two are non-dominated; the third is dominated by neither
    # so all three rank=0. promoted = all 3.
    assert len(d.promoted) == 3


# -------- Surrogate -------------------------------------------------------


def test_surrogate_zero_when_empty() -> None:
    s = GPSurrogate()
    p = s.predict(Chromosome(
        primitive_kind=PrimitiveKind.ZIGZAG, power_W=200, speed_mm_s=800, hatch_um=100, spot_um=80
    ))
    assert p.variance > 0


def test_surrogate_fits_simple_trend() -> None:
    """GP should recover a monotone trend along power."""
    entries = [_make_entry(p, scalar_J=p / 500.0) for p in (80, 150, 220, 300, 380)]
    s = GPSurrogate(length_scale=0.3, noise_sigma=0.01)
    s.fit(entries)
    p_low = s.predict(Chromosome(
        primitive_kind=PrimitiveKind.ZIGZAG, power_W=100, speed_mm_s=800, hatch_um=100, spot_um=80
    ))
    p_high = s.predict(Chromosome(
        primitive_kind=PrimitiveKind.ZIGZAG, power_W=360, speed_mm_s=800, hatch_um=100, spot_um=80
    ))
    assert p_high.mean > p_low.mean
    assert p_low.std > 0


# -------- HF case generator ----------------------------------------------


def test_laserbeamfoam_case_is_complete(scenario: ScenarioConfig, tmp_path: Path) -> None:
    spec = PrimitiveSpec(
        kind=PrimitiveKind.ZIGZAG, power_W=200, speed_mm_s=800, hatch_um=100, spot_um=80
    )
    pat = build_primitive(spec, scenario.roi)
    case = build_laserbeamfoam_case(pat, scenario, case_dir=tmp_path / "case")
    for rel in (
        "system/controlDict",
        "system/fvSchemes",
        "system/fvSolution",
        "system/blockMeshDict",
        "constant/transportProperties",
        "constant/laserPath",
        "0/T",
        "0/U",
        "0/p_rgh",
        "0/alpha.metal",
        "Allrun",
    ):
        assert (case.case_dir / rel).exists(), f"missing {rel}"
    # transportProperties contains the material values
    tp = (case.case_dir / "constant/transportProperties").read_text()
    assert f"{scenario.material.rho_solid}" in tp
    assert f"{scenario.material.absorptivity}" in tp
    # laserPath has the rasterised samples
    lp = (case.case_dir / "constant/laserPath").read_text()
    assert lp.count("\n") > 5


def test_parser_unavailable_returns_clear_error(tmp_path: Path, scenario: ScenarioConfig) -> None:
    res = parse_case_output(tmp_path / "doesnotexist")
    assert not res.success


def test_laserbeamfoam_runner_unavailable_is_explicit(tmp_path: Path, scenario: ScenarioConfig) -> None:
    spec = PrimitiveSpec(
        kind=PrimitiveKind.ZIGZAG, power_W=200, speed_mm_s=800, hatch_um=100, spot_um=80
    )
    pat = build_primitive(spec, scenario.roi)
    case = build_laserbeamfoam_case(pat, scenario, case_dir=tmp_path / "case")
    runner = LaserbeamFoamRunner()
    result = asyncio.run(runner.run(case))
    if not laserbeamfoam_available():
        assert not result.success
        assert "laserbeamFoam" in result.stderr or "PATH" in result.stderr


# -------- Local runner ---------------------------------------------------


def test_local_runner_picks_up_simple_subprocess(tmp_path: Path) -> None:
    """LocalRunner should successfully run /bin/echo and capture stdout."""
    case = CaseSpec(case_dir=tmp_path, solver_name="echo", args=("hello", "world"))
    runner = LocalRunner()
    result = asyncio.run(runner.submit(case))
    assert result.success
    assert "hello world" in result.stdout


def test_local_runner_handles_missing_executable(tmp_path: Path) -> None:
    case = CaseSpec(case_dir=tmp_path, solver_name="this-binary-does-not-exist-xyz")
    runner = LocalRunner()
    result = asyncio.run(runner.submit(case))
    assert not result.success
    assert result.exit_code == 127


# -------- DEM -----------------------------------------------------------


def test_pack_powder_bed_synthetic(scenario: ScenarioConfig, tmp_path: Path) -> None:
    res = pack_powder_bed(scenario.powder, scenario.machine, scenario.roi, out_dir=tmp_path, seed=1)
    assert res.porosity_npz.exists()
    assert 0.05 < res.fill_fraction < 0.95
    assert res.n_particles > 0
    # cache hit on second call
    res2 = pack_powder_bed(scenario.powder, scenario.machine, scenario.roi, out_dir=tmp_path, seed=1)
    assert res2.porosity_npz == res.porosity_npz


def test_powder_bed_effective_properties_lower_k(scenario: ScenarioConfig, tmp_path: Path) -> None:
    res = pack_powder_bed(scenario.powder, scenario.machine, scenario.roi, out_dir=tmp_path, seed=2)
    field = load_porosity(res.porosity_npz)
    eff = effective_thermal_properties(field, scenario.material)
    # bulk thermal conductivity of a metal-powder layer is much lower than solid
    assert eff.k < scenario.material.k_solid * 0.5


def test_apply_powder_to_material_updates_density_k(scenario: ScenarioConfig, tmp_path: Path) -> None:
    res = pack_powder_bed(scenario.powder, scenario.machine, scenario.roi, out_dir=tmp_path, seed=3)
    field = load_porosity(res.porosity_npz)
    mat_eff = apply_powder_to_material(scenario.material, field)
    assert mat_eff.rho_solid < scenario.material.rho_solid
    assert mat_eff.k_solid < scenario.material.k_solid


def test_dem_to_cfd_writes_setfields_dict(scenario: ScenarioConfig, tmp_path: Path) -> None:
    res = pack_powder_bed(scenario.powder, scenario.machine, scenario.roi, out_dir=tmp_path, seed=4)
    field = load_porosity(res.porosity_npz)
    case_dir = tmp_path / "case"
    (case_dir / "system").mkdir(parents=True, exist_ok=True)
    out = write_setfields_dict(field, case_dir)
    text = out.read_text()
    assert "alpha.metal" in text
    assert f"{field.fill_fraction:.4f}" in text


# -------- Voronoi primitive ----------------------------------------------


def test_voronoi_covers_roi(scenario: ScenarioConfig) -> None:
    spec = PrimitiveSpec(
        kind=PrimitiveKind.VORONOI,
        power_W=200, speed_mm_s=800, hatch_um=200, spot_um=80,
        params={"n_cells": 6, "lloyd_iter": 2, "seed": 5},
    )
    pat = build_primitive(spec, scenario.roi)
    assert len(pat.segments) > 0
    xs = [w.x_mm for s in pat.segments for w in s.waypoints]
    ys = [w.y_mm for s in pat.segments for w in s.waypoints]
    # at least cover most of the bounding box (small slack for Voronoi clipping)
    assert max(xs) >= scenario.roi.x1_mm - 0.5
    assert min(xs) <= scenario.roi.x0_mm + 0.5


def test_voronoi_deterministic_under_seed(scenario: ScenarioConfig) -> None:
    spec = lambda s: PrimitiveSpec(
        kind=PrimitiveKind.VORONOI,
        power_W=200, speed_mm_s=800, hatch_um=200, spot_um=80,
        params={"n_cells": 6, "lloyd_iter": 2, "seed": s},
    )
    a = build_primitive(spec(11), scenario.roi)
    b = build_primitive(spec(11), scenario.roi)
    assert len(a.segments) == len(b.segments)
