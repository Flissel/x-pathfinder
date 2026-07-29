"""Pareto-style fitness evaluator backed by Eagar-Tsai proxies.

Two evaluation modes:
- "field"    (default): rasterizes the pattern, evaluates a 2D T_max field on
              the ROI grid (transient.t_max_field), so the fitness is sensitive
              to pattern *geometry* (coverage, hot/cold spots), not just (P, v).
- "segment"  (fast path): per-segment Eagar-Tsai only — geometry-blind but
              ~30x cheaper. Useful for smoke tests and warm-up generations.

Objective vector (minimize all):
- u_temp_loss = std(T_max) / mean(T_max)         uniformity loss across ROI
- p_lof       = 1 - melt-coverage fraction        unmelted area is LoF risk
- p_keyhole   = fraction of ROI above 0.9*T_boil  keyhole/spatter risk
- p_surface   = aspect-ratio penalty              vs. Rayleigh plateau
- t_cycle_s   = pure scan time
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from laser_sim.config.schema import GeometryROI, MachineConfig, MaterialConfig
from laser_sim.patterns.base import ScanPattern
from laser_sim.patterns.rasterize import rasterize_pattern
from laser_sim.physics.fast_sim.eagar_tsai import MeltPoolEstimate, eagar_tsai_melt_pool
from laser_sim.physics.fast_sim.transient import (
    FieldResult,
    t_max_field,
    t_max_field_superposition,
)

OBJECTIVE_NAMES: tuple[str, ...] = (
    "u_temp_loss",
    "p_lof",
    "p_keyhole",
    "p_surface",
    "t_cycle_s",
)
OBJECTIVE_SENSES: tuple[str, ...] = ("min", "min", "min", "min", "min")

_DEFAULT_WEIGHTS = np.array([0.30, 0.25, 0.20, 0.10, 0.15])


@dataclass(frozen=True)
class FitnessVector:
    values: tuple[float, ...]

    def as_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=float)

    def dominates(self, other: "FitnessVector") -> bool:
        a = self.as_array()
        b = other.as_array()
        return bool(np.all(a <= b) and np.any(a < b))


@dataclass(frozen=True)
class PatternEvaluation:
    fitness: FitnessVector
    metrics: dict[str, float]
    scalar_J: float
    field: FieldResult | None = None

    def __getitem__(self, key: str) -> float:
        return self.metrics[key]


def _per_segment_eagar_tsai(
    pattern: ScanPattern, mat: MaterialConfig, machine: MachineConfig, spot_um: float
) -> list[MeltPoolEstimate]:
    layer_mm = machine.layer_thickness_um * 1e-3
    out: list[MeltPoolEstimate] = []
    for seg in pattern.segments:
        if not seg.laser_on or seg.power_W <= 0:
            continue
        out.append(
            eagar_tsai_melt_pool(
                power_W=seg.power_W,
                speed_mm_s=seg.speed_mm_s,
                mat=mat,
                preheat_K=machine.preheat_K,
                hatch_mm=0.1,
                layer_mm=layer_mm,
                spot_um=spot_um,
            )
        )
    return out


def scalarize(
    fitness: FitnessVector, weights: np.ndarray | None = None
) -> float:
    w = weights if weights is not None else _DEFAULT_WEIGHTS
    v = fitness.as_array()
    if v.size != w.size:
        raise ValueError(f"weights ({w.size}) and objectives ({v.size}) length mismatch")
    return float(np.dot(w, v))


def _scalar_J_from_values(values: tuple[float, ...]) -> float:
    """Scalarization with t_cycle squashed via tanh so it doesn't dominate raw."""
    sv = list(values)
    sv[-1] = math.tanh(sv[-1] / 5.0)
    return scalarize(FitnessVector(values=tuple(sv)))


def _evaluate_segment_mode(
    pattern: ScanPattern,
    material: MaterialConfig,
    machine: MachineConfig,
    hatch_mm: float,
    spot_um: float,
) -> PatternEvaluation:
    ests = _per_segment_eagar_tsai(pattern, material, machine, spot_um)
    if not ests:
        f = FitnessVector(values=(1.0, 1.0, 1.0, 1.0, 1.0))
        return PatternEvaluation(
            fitness=f,
            metrics={k: float("nan") for k in OBJECTIVE_NAMES},
            scalar_J=scalarize(f),
        )

    widths = np.array([e.width_mm for e in ests])
    peaks = np.array([e.peak_T_K for e in ests])
    ars = np.array([e.aspect_ratio for e in ests])

    u_temp = 1.0 - float(peaks.std() / max(peaks.mean(), 1e-9))
    u_temp = max(0.0, min(1.0, u_temp))
    u_temp_loss = 1.0 - u_temp

    deficit = np.maximum(0.0, hatch_mm - widths) / max(hatch_mm, 1e-9)
    p_lof = max(0.0, min(1.0, float(deficit.mean())))

    t_boil = material.boiling_K
    margin = np.maximum(0.0, peaks - 0.9 * t_boil) / max(0.1 * t_boil, 1e-9)
    p_keyhole = max(0.0, min(1.0, float(margin.mean())))

    target_lo, target_hi = 1.5, 2.5
    below = np.maximum(0.0, target_lo - ars)
    above = np.maximum(0.0, ars - target_hi)
    p_surface = max(0.0, min(1.0, float((below + above).mean() / target_hi)))

    t_cycle = pattern.total_time_s()
    values = (u_temp_loss, p_lof, p_keyhole, p_surface, t_cycle)
    metrics: dict[str, float] = dict(zip(OBJECTIVE_NAMES, values))
    metrics.update(
        peak_T_K_mean=float(peaks.mean()),
        width_mm_mean=float(widths.mean()),
        aspect_ratio_mean=float(ars.mean()),
        energy_density=float(np.mean([e.energy_density_J_mm3 for e in ests])),
    )
    return PatternEvaluation(
        fitness=FitnessVector(values=values),
        metrics=metrics,
        scalar_J=_scalar_J_from_values(values),
    )


def _evaluate_field_mode(
    pattern: ScanPattern,
    roi: GeometryROI,
    material: MaterialConfig,
    machine: MachineConfig,
    spot_um: float,
    grid_n: int,
    stride: int,
    transient_mode: str = "max",
) -> PatternEvaluation:
    rp = rasterize_pattern(pattern, ds_mm=0.05)
    if rp.n_samples() < 2:
        f = FitnessVector(values=(1.0, 1.0, 1.0, 1.0, 1.0))
        return PatternEvaluation(
            fitness=f,
            metrics={k: float("nan") for k in OBJECTIVE_NAMES},
            scalar_J=scalarize(f),
        )

    if transient_mode == "superposition":
        field = t_max_field_superposition(
            rp,
            roi,
            material=material,
            machine=machine,
            spot_um=spot_um,
            nx=grid_n,
            ny=grid_n,
            stride=stride,
        )
    else:
        field = t_max_field(
            rp,
            roi,
            material=material,
            machine=machine,
            spot_um=spot_um,
            nx=grid_n,
            ny=grid_n,
            stride=stride,
        )

    u_temp_loss = float(field.t_max_std_K / max(field.t_max_mean_K, 1e-9))
    u_temp_loss = max(0.0, min(1.0, u_temp_loss))

    p_lof = max(0.0, min(1.0, 1.0 - field.coverage_fraction))
    p_keyhole = max(0.0, min(1.0, field.keyhole_fraction))

    # surface (aspect-ratio) — keep per-segment proxy because the field doesn't
    # see per-track L/W. In a fully-resolved fast-sim this drops out.
    ests = _per_segment_eagar_tsai(pattern, material, machine, spot_um)
    if ests:
        ars = np.array([e.aspect_ratio for e in ests])
        target_lo, target_hi = 1.5, 2.5
        below = np.maximum(0.0, target_lo - ars)
        above = np.maximum(0.0, ars - target_hi)
        p_surface = max(0.0, min(1.0, float((below + above).mean() / target_hi)))
    else:
        p_surface = 1.0

    t_cycle = pattern.total_time_s()
    values = (u_temp_loss, p_lof, p_keyhole, p_surface, t_cycle)
    metrics: dict[str, float] = dict(zip(OBJECTIVE_NAMES, values))
    metrics.update(
        t_max_mean_K=field.t_max_mean_K,
        t_max_std_K=field.t_max_std_K,
        coverage=field.coverage_fraction,
        keyhole_area=field.keyhole_fraction,
    )
    return PatternEvaluation(
        fitness=FitnessVector(values=values),
        metrics=metrics,
        scalar_J=_scalar_J_from_values(values),
        field=field,
    )


class PatternEvaluator:
    """Stateless evaluator: pattern + scenario -> fitness vector + metrics.

    Mode "field" is the default and what the EA should use; "segment" is
    cheaper and used in unit tests + smoke runs.
    """

    def __init__(
        self,
        material: MaterialConfig,
        machine: MachineConfig,
        roi: GeometryROI,
        hatch_mm: float = 0.1,
        spot_um: float = 80.0,
        mode: Literal["field", "segment"] = "field",
        grid_n: int = 41,
        stride: int = 4,
        transient_mode: Literal["max", "superposition"] = "max",
    ) -> None:
        self.material = material
        self.machine = machine
        self.roi = roi
        self.hatch_mm = hatch_mm
        self.spot_um = spot_um
        self.mode = mode
        self.grid_n = grid_n
        self.stride = stride
        self.transient_mode = transient_mode

    def evaluate(
        self,
        pattern: ScanPattern,
        hatch_mm: float | None = None,
        spot_um: float | None = None,
    ) -> PatternEvaluation:
        h = hatch_mm if hatch_mm is not None else self.hatch_mm
        s = spot_um if spot_um is not None else self.spot_um
        if self.mode == "segment":
            return _evaluate_segment_mode(pattern, self.material, self.machine, h, s)
        return _evaluate_field_mode(
            pattern,
            self.roi,
            self.material,
            self.machine,
            s,
            self.grid_n,
            self.stride,
            transient_mode=self.transient_mode,
        )
