"""Pareto-style fitness evaluator backed by the Eagar-Tsai proxy.

Objective vector (minimize all):
- u_temp_loss = 1 - U_temp     (uniformity of T_max across segments)
- p_lof       = LoF risk        (hatch > melt-pool width)
- p_keyhole   = keyhole risk    (T_max approaching boiling)
- p_surface   = aspect-ratio penalty vs Rayleigh plateau (target L/W in [1.5, 2.5])
- t_cycle_s   = pure scan time
Plus a scalarized fallback J (lower is better) for reporting/sorting.

All proxies are deliberately analytic so the EA loop runs without GPU; the
JAX/CuPy solver will swap in behind the same interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from laser_sim.config.schema import GeometryROI, MachineConfig, MaterialConfig
from laser_sim.patterns.base import ScanPattern
from laser_sim.physics.fast_sim.eagar_tsai import MeltPoolEstimate, eagar_tsai_melt_pool

OBJECTIVE_NAMES: tuple[str, ...] = (
    "u_temp_loss",
    "p_lof",
    "p_keyhole",
    "p_surface",
    "t_cycle_s",
)
# all minimized; senses kept explicit for the day someone wants to flip one
OBJECTIVE_SENSES: tuple[str, ...] = ("min", "min", "min", "min", "min")

# default scalarization weights (must sum to ~1; t_cycle gets less mass because
# it's measured in s and dominates raw magnitude otherwise)
_DEFAULT_WEIGHTS = np.array([0.30, 0.25, 0.20, 0.10, 0.15])


@dataclass(frozen=True)
class FitnessVector:
    """Multi-objective fitness; lower is better on every axis."""

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
        est = eagar_tsai_melt_pool(
            power_W=seg.power_W,
            speed_mm_s=seg.speed_mm_s,
            mat=mat,
            preheat_K=machine.preheat_K,
            hatch_mm=0.1,  # placeholder; pattern-level fitness overrides via hatch_mm arg
            layer_mm=layer_mm,
            spot_um=spot_um,
        )
        out.append(est)
    return out


def scalarize(
    fitness: FitnessVector, weights: np.ndarray | None = None
) -> float:
    """Weighted sum of objectives; lower is better. Used for tournament
    selection fallback and reporting, NOT for Pareto selection."""
    w = weights if weights is not None else _DEFAULT_WEIGHTS
    v = fitness.as_array()
    if v.size != w.size:
        raise ValueError(f"weights ({w.size}) and objectives ({v.size}) length mismatch")
    return float(np.dot(w, v))


class PatternEvaluator:
    """Stateless evaluator: pattern + scenario -> fitness vector + metrics.

    Construct once per scenario (so material/machine are bound), call eval()
    for each candidate.
    """

    def __init__(
        self,
        material: MaterialConfig,
        machine: MachineConfig,
        roi: GeometryROI,
        hatch_mm: float = 0.1,
        spot_um: float = 80.0,
    ) -> None:
        self.material = material
        self.machine = machine
        self.roi = roi
        self.hatch_mm = hatch_mm
        self.spot_um = spot_um

    def evaluate(
        self,
        pattern: ScanPattern,
        hatch_mm: float | None = None,
        spot_um: float | None = None,
    ) -> PatternEvaluation:
        s_um = spot_um if spot_um is not None else self.spot_um
        ests = _per_segment_eagar_tsai(pattern, self.material, self.machine, s_um)
        if not ests:
            # degenerate: no active segments -> worst-case fitness
            f = FitnessVector(values=(1.0, 1.0, 1.0, 1.0, 1.0))
            return PatternEvaluation(
                fitness=f,
                metrics={k: float("nan") for k in OBJECTIVE_NAMES},
                scalar_J=scalarize(f),
            )

        h_mm = hatch_mm if hatch_mm is not None else self.hatch_mm
        widths = np.array([e.width_mm for e in ests])
        peaks = np.array([e.peak_T_K for e in ests])
        ars = np.array([e.aspect_ratio for e in ests])

        # uniformity (higher is better) -> loss = 1 - U
        if peaks.mean() > 1e-9:
            u_temp = 1.0 - float(peaks.std() / peaks.mean())
        else:
            u_temp = 0.0
        u_temp = max(0.0, min(1.0, u_temp))
        u_temp_loss = 1.0 - u_temp

        # LoF risk: how often the hatch exceeds the melt pool width
        deficit = np.maximum(0.0, h_mm - widths) / max(h_mm, 1e-9)
        p_lof = float(deficit.mean())
        p_lof = max(0.0, min(1.0, p_lof))

        # keyhole risk: peak T approaches boiling
        t_boil = self.material.boiling_K
        margin = np.maximum(0.0, peaks - 0.9 * t_boil) / max(0.1 * t_boil, 1e-9)
        p_keyhole = float(margin.mean())
        p_keyhole = max(0.0, min(1.0, p_keyhole))

        # surface penalty: aspect ratio outside [1.5, 2.5] is bad
        # (1 is too round, > 3 starts balling per Rayleigh)
        target_lo, target_hi = 1.5, 2.5
        below = np.maximum(0.0, target_lo - ars)
        above = np.maximum(0.0, ars - target_hi)
        p_surface = float((below + above).mean() / target_hi)
        p_surface = max(0.0, min(1.0, p_surface))

        # productivity in pure cycle time (s)
        t_cycle = pattern.total_time_s()

        values = (u_temp_loss, p_lof, p_keyhole, p_surface, t_cycle)
        f = FitnessVector(values=values)
        metrics = dict(zip(OBJECTIVE_NAMES, values))
        metrics["peak_T_K_mean"] = float(peaks.mean())
        metrics["width_mm_mean"] = float(widths.mean())
        metrics["aspect_ratio_mean"] = float(ars.mean())
        metrics["energy_density"] = float(
            np.mean([e.energy_density_J_mm3 for e in ests])
        )
        # rescale t_cycle into 0..1-ish for scalarization without hurting Pareto axis
        # (a 5 mm x 5 mm patch at 800 mm/s and 0.1 mm hatch ~ 0.3 s, so 5 s upper)
        scalar_values = list(values)
        scalar_values[-1] = math.tanh(t_cycle / 5.0)
        scalar_J = scalarize(FitnessVector(values=tuple(scalar_values)))
        return PatternEvaluation(fitness=f, metrics=metrics, scalar_J=scalar_J)
