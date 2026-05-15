"""Tests for the Eagar-Tsai proxy, fitness evaluator, NSGA-II, and Pareto archive."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from laser_sim.config.schema import ScenarioConfig
from laser_sim.fitness.evaluator import (
    FitnessVector,
    PatternEvaluator,
    scalarize,
)
from laser_sim.genome.archive import (
    ArchiveEntry,
    ParetoArchive,
    crowding_distance,
    fast_non_dominated_sort,
)
from laser_sim.genome.chromosome import Chromosome, random_chromosome
from laser_sim.genome.engine import GeneticEngine
from laser_sim.patterns.base import PrimitiveKind, PrimitiveSpec
from laser_sim.patterns.primitives import build_primitive
from laser_sim.physics.fast_sim.eagar_tsai import (
    eagar_tsai_melt_pool,
    energy_density,
)


@pytest.fixture
def scenario() -> ScenarioConfig:
    path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return ScenarioConfig.model_validate(data)


def test_energy_density_units(scenario: ScenarioConfig) -> None:
    # 200W / (800 mm/s * 0.1 mm * 0.03 mm) = 83.33 J/mm^3, Ti-64 sweet spot
    e = energy_density(power_W=200.0, speed_mm_s=800.0, hatch_mm=0.1, layer_mm=0.03)
    assert math.isclose(e, 83.333, rel_tol=1e-3)


def test_eagar_tsai_higher_power_gives_wider_melt(scenario: ScenarioConfig) -> None:
    mat = scenario.material
    e_low = eagar_tsai_melt_pool(
        power_W=100.0,
        speed_mm_s=800.0,
        mat=mat,
        preheat_K=300.0,
        hatch_mm=0.1,
        layer_mm=0.03,
    )
    e_high = eagar_tsai_melt_pool(
        power_W=300.0,
        speed_mm_s=800.0,
        mat=mat,
        preheat_K=300.0,
        hatch_mm=0.1,
        layer_mm=0.03,
    )
    assert e_high.width_mm >= e_low.width_mm
    assert e_high.peak_T_K >= e_low.peak_T_K


def test_eagar_tsai_higher_speed_gives_narrower_melt(scenario: ScenarioConfig) -> None:
    mat = scenario.material
    e_slow = eagar_tsai_melt_pool(
        power_W=200.0, speed_mm_s=400.0, mat=mat, preheat_K=300.0, hatch_mm=0.1, layer_mm=0.03
    )
    e_fast = eagar_tsai_melt_pool(
        power_W=200.0, speed_mm_s=1600.0, mat=mat, preheat_K=300.0, hatch_mm=0.1, layer_mm=0.03
    )
    assert e_fast.width_mm <= e_slow.width_mm


def test_fitness_vector_domination() -> None:
    a = FitnessVector(values=(0.1, 0.2, 0.3, 0.4, 0.5))
    b = FitnessVector(values=(0.2, 0.3, 0.4, 0.5, 0.6))
    c = FitnessVector(values=(0.1, 0.2, 0.3, 0.4, 0.5))
    assert a.dominates(b)
    assert not b.dominates(a)
    assert not a.dominates(c)  # identical -> no strict dominance


def test_fast_non_dominated_sort_simple_case() -> None:
    # 3 points; (0, 1) and (1, 0) are non-dominated; (1, 1) is dominated by both
    fits = [
        FitnessVector(values=(0.0, 1.0, 0.0, 0.0, 0.0)),
        FitnessVector(values=(1.0, 0.0, 0.0, 0.0, 0.0)),
        FitnessVector(values=(1.0, 1.0, 0.0, 0.0, 0.0)),
    ]
    ranks = fast_non_dominated_sort(fits)
    assert ranks[0] == 0
    assert ranks[1] == 0
    assert ranks[2] == 1


def test_crowding_distance_endpoints_infinite() -> None:
    fits = [
        FitnessVector(values=(0.0, 1.0, 0.0, 0.0, 0.0)),
        FitnessVector(values=(0.5, 0.5, 0.0, 0.0, 0.0)),
        FitnessVector(values=(1.0, 0.0, 0.0, 0.0, 0.0)),
    ]
    cd = crowding_distance(fits)
    assert math.isinf(cd[0]) and math.isinf(cd[2])
    assert not math.isinf(cd[1])


def test_pareto_archive_rejects_dominated() -> None:
    arc = ParetoArchive(max_size=10)
    f1 = FitnessVector(values=(0.1, 0.1, 0.0, 0.0, 0.0))
    f2 = FitnessVector(values=(0.2, 0.2, 0.0, 0.0, 0.0))  # dominated by f1

    def _ch() -> Chromosome:
        return Chromosome(
            primitive_kind=PrimitiveKind.ZIGZAG,
            power_W=200,
            speed_mm_s=800,
            hatch_um=100,
            spot_um=80,
        )

    assert arc.add(ArchiveEntry(chromosome=_ch(), fitness=f1, scalar_J=0.1))
    assert not arc.add(ArchiveEntry(chromosome=_ch(), fitness=f2, scalar_J=0.2))
    assert len(arc) == 1


def test_pareto_archive_evicts_dominated_on_new_winner() -> None:
    arc = ParetoArchive(max_size=10)
    f_old = FitnessVector(values=(0.5, 0.5, 0.5, 0.5, 0.5))
    f_better = FitnessVector(values=(0.1, 0.1, 0.1, 0.1, 0.1))

    def _ch(p: float) -> Chromosome:
        return Chromosome(
            primitive_kind=PrimitiveKind.ZIGZAG,
            power_W=p,
            speed_mm_s=800,
            hatch_um=100,
            spot_um=80,
        )

    arc.add(ArchiveEntry(chromosome=_ch(100), fitness=f_old, scalar_J=0.5))
    arc.add(ArchiveEntry(chromosome=_ch(200), fitness=f_better, scalar_J=0.1))
    assert len(arc) == 1


def test_evaluator_segment_mode_returns_finite_fitness(scenario: ScenarioConfig) -> None:
    spec = PrimitiveSpec(
        kind=PrimitiveKind.ZIGZAG,
        power_W=200,
        speed_mm_s=800,
        hatch_um=100,
        spot_um=80,
    )
    pat = build_primitive(spec, scenario.roi)
    ev = PatternEvaluator(scenario.material, scenario.machine, scenario.roi, mode="segment")
    out = ev.evaluate(pat, hatch_mm=0.1)
    assert all(math.isfinite(v) for v in out.fitness.values)
    assert math.isfinite(out.scalar_J)
    assert "peak_T_K_mean" in out.metrics
    assert out.field is None


def test_evaluator_field_mode_uses_grid(scenario: ScenarioConfig) -> None:
    spec = PrimitiveSpec(
        kind=PrimitiveKind.ZIGZAG,
        power_W=200,
        speed_mm_s=800,
        hatch_um=100,
        spot_um=80,
    )
    pat = build_primitive(spec, scenario.roi)
    ev = PatternEvaluator(
        scenario.material, scenario.machine, scenario.roi, mode="field", grid_n=21, stride=4
    )
    out = ev.evaluate(pat, hatch_mm=0.1)
    assert all(math.isfinite(v) for v in out.fitness.values)
    assert out.field is not None
    assert out.field.t_max_K.shape == (21, 21)
    assert out.field.t_max_mean_K > scenario.machine.preheat_K
    assert "coverage" in out.metrics


def test_evaluator_field_mode_distinguishes_patterns(scenario: ScenarioConfig) -> None:
    """Two patterns with the same (P, v, hatch) but different shape should
    produce different field-mode fitness."""
    spec_a = PrimitiveSpec(
        kind=PrimitiveKind.ZIGZAG, power_W=200, speed_mm_s=800, hatch_um=100, spot_um=80
    )
    spec_b = PrimitiveSpec(
        kind=PrimitiveKind.HILBERT, power_W=200, speed_mm_s=800, hatch_um=100, spot_um=80
    )
    pat_a = build_primitive(spec_a, scenario.roi)
    pat_b = build_primitive(spec_b, scenario.roi)
    ev = PatternEvaluator(
        scenario.material, scenario.machine, scenario.roi, mode="field", grid_n=21, stride=4
    )
    out_a = ev.evaluate(pat_a)
    out_b = ev.evaluate(pat_b)
    # uniformity / coverage / cycle time will differ between zigzag and hilbert
    assert out_a.fitness.values != out_b.fitness.values


def test_random_chromosome_within_envelope(scenario: ScenarioConfig) -> None:
    import random as _rng

    r = _rng.Random(0)
    for _ in range(20):
        c = random_chromosome(scenario.machine, r)
        c = c.clamped(scenario.machine)
        las = scenario.machine.laser
        assert las.power_min_W <= c.power_W <= las.power_max_W
        assert las.speed_min_mm_s <= c.speed_mm_s <= las.speed_max_mm_s
        assert las.spot_min_um <= c.spot_um <= las.spot_max_um


def test_chromosome_hash_id_stable(scenario: ScenarioConfig) -> None:
    c1 = Chromosome(
        primitive_kind=PrimitiveKind.ZIGZAG,
        power_W=200.0,
        speed_mm_s=800.0,
        hatch_um=100.0,
        spot_um=80.0,
    )
    c2 = Chromosome(
        primitive_kind=PrimitiveKind.ZIGZAG,
        power_W=200.0,
        speed_mm_s=800.0,
        hatch_um=100.0,
        spot_um=80.0,
    )
    assert c1.hash_id() == c2.hash_id()
    c3 = Chromosome(
        primitive_kind=PrimitiveKind.STRIPES,
        power_W=200.0,
        speed_mm_s=800.0,
        hatch_um=100.0,
        spot_um=80.0,
    )
    assert c1.hash_id() != c3.hash_id()


def test_engine_runs_and_archive_is_pareto_consistent(scenario: ScenarioConfig) -> None:
    """Run the EA end-to-end and verify the archive is a valid Pareto front
    (no member is dominated by another).

    Note: best_scalar_J is NOT guaranteed monotonic under NSGA-II elitism
    because elites are chosen on (rank, -crowding), not scalar_J.
    """
    ea = scenario.ea.model_copy(
        update={"population": 12, "generations": 4, "elitism": 2, "seed": 7}
    )
    eng = GeneticEngine(
        material=scenario.material,
        machine=scenario.machine,
        roi=scenario.roi,
        ea=ea,
    )
    log = eng.run()
    assert len(log.history) == ea.generations
    assert len(log.archive) >= 1
    entries = log.archive.entries
    # archive is Pareto-consistent: no entry dominates another
    for i, ei in enumerate(entries):
        for j, ej in enumerate(entries):
            if i == j:
                continue
            assert not ei.fitness.dominates(ej.fitness), (
                f"archive entry {i} dominates entry {j}; archive invariant broken"
            )


def test_engine_cache_skips_duplicate_evaluations(scenario: ScenarioConfig) -> None:
    """If we feed the engine the same chromosome twice, the second eval is a cache hit."""
    ea = scenario.ea.model_copy(
        update={"population": 4, "generations": 1, "elitism": 1, "seed": 3}
    )
    eng = GeneticEngine(
        material=scenario.material,
        machine=scenario.machine,
        roi=scenario.roi,
        ea=ea,
    )
    c = Chromosome(
        primitive_kind=PrimitiveKind.ZIGZAG,
        power_W=200.0,
        speed_mm_s=800.0,
        hatch_um=100.0,
        spot_um=80.0,
    )
    e1 = eng._evaluate(c)
    e2 = eng._evaluate(c)
    assert e1 is e2  # same cached object


def test_transient_superposition_accumulates_heat(scenario: ScenarioConfig) -> None:
    """At a point BETWEEN two close tracks, the two-track field should be hotter
    than the single-track field at the same point. Validates that the 3D Green's
    function model captures inter-track accumulation (the per-source-max
    approximation would give nearly the same value)."""
    import numpy as np

    from laser_sim.patterns.base import ScanPattern, Segment, Waypoint
    from laser_sim.patterns.rasterize import rasterize_pattern
    from laser_sim.physics.fast_sim.transient import t_max_field_superposition

    # single track exactly at y=0
    iso = ScanPattern(
        segments=(
            Segment(
                waypoints=(Waypoint(-2.0, 0.05), Waypoint(2.0, 0.05)),
                power_W=200, speed_mm_s=800,
            ),
        )
    )
    # two close tracks straddling y=0 with 100um gap
    two = ScanPattern(
        segments=(
            Segment(
                waypoints=(Waypoint(-2.0, -0.05), Waypoint(2.0, -0.05)),
                power_W=200, speed_mm_s=800,
            ),
            Segment(
                waypoints=(Waypoint(2.0, 0.05), Waypoint(-2.0, 0.05)),
                power_W=200, speed_mm_s=800,
            ),
        )
    )
    iso_rp = rasterize_pattern(iso, ds_mm=0.05)
    two_rp = rasterize_pattern(two, ds_mm=0.05)
    f_iso = t_max_field_superposition(
        iso_rp, scenario.roi, scenario.material, scenario.machine,
        spot_um=80.0, nx=41, ny=41, stride=6,
    )
    f_two = t_max_field_superposition(
        two_rp, scenario.roi, scenario.material, scenario.machine,
        spot_um=80.0, nx=41, ny=41, stride=6,
    )
    # find the grid cell closest to (0, 0) — the gap between the two-track pair
    iy = int(np.argmin(np.abs(f_two.grid_y_mm)))
    # average over a thin band along x at y=0 to smooth out grid noise
    band_two = f_two.t_max_K[:, iy].mean()
    band_iso = f_iso.t_max_K[:, iy].mean()
    assert band_two > band_iso * 1.05  # at least 5% hotter from accumulation


def test_transient_superposition_vs_max_differs(scenario: ScenarioConfig) -> None:
    """Superposition and per-source-max should produce different fields for
    a real scan pattern. Either could be 'larger' depending on geometry but
    they must not coincide."""
    import numpy as np

    from laser_sim.patterns.rasterize import rasterize_pattern
    from laser_sim.physics.fast_sim.transient import (
        t_max_field,
        t_max_field_superposition,
    )

    spec = PrimitiveSpec(
        kind=PrimitiveKind.ZIGZAG, power_W=200, speed_mm_s=800, hatch_um=80, spot_um=80
    )
    pat = build_primitive(spec, scenario.roi)
    rp = rasterize_pattern(pat, ds_mm=0.05)
    f_max = t_max_field(
        rp, scenario.roi, scenario.material, scenario.machine,
        spot_um=80.0, nx=21, ny=21, stride=8,
    )
    f_sup = t_max_field_superposition(
        rp, scenario.roi, scenario.material, scenario.machine,
        spot_um=80.0, nx=21, ny=21, stride=8,
    )
    assert not np.allclose(f_max.t_max_K, f_sup.t_max_K)


def test_evaluator_transient_mode_propagates(scenario: ScenarioConfig) -> None:
    spec = PrimitiveSpec(
        kind=PrimitiveKind.ZIGZAG, power_W=200, speed_mm_s=800, hatch_um=80, spot_um=80
    )
    pat = build_primitive(spec, scenario.roi)
    ev_max = PatternEvaluator(
        scenario.material, scenario.machine, scenario.roi,
        mode="field", grid_n=21, stride=8, transient_mode="max",
    )
    ev_sup = PatternEvaluator(
        scenario.material, scenario.machine, scenario.roi,
        mode="field", grid_n=21, stride=8, transient_mode="superposition",
    )
    out_max = ev_max.evaluate(pat)
    out_sup = ev_sup.evaluate(pat)
    assert out_max.fitness.values != out_sup.fitness.values


def test_scalarize_weights_validate() -> None:
    f = FitnessVector(values=(0.1, 0.2, 0.3, 0.4, 0.5))
    assert math.isclose(
        scalarize(f),
        0.30 * 0.1 + 0.25 * 0.2 + 0.20 * 0.3 + 0.10 * 0.4 + 0.15 * 0.5,
        rel_tol=1e-9,
    )
    with pytest.raises(ValueError):
        scalarize(f, weights=np.array([0.5, 0.5]))
