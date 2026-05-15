"""Hierarchical calibration of fast-sim parameters against measurements.

The default field-mode evaluator is parameterised by three quantities that
the user typically does NOT know precisely from machine spec sheets:

  - absorptivity (eta)             0..1
  - effective laser spot radius    micrometers (true 1/e radius after optics)
  - emissivity                     0..1 (radiative loss, indirect effect)

Calibration tunes these on real measurements so the optimizer does not learn
sim artefacts. Per the plan, calibration runs hierarchically:

  level 1: single-track on substrate (no powder)        -> spot, absorptivity
  level 2: single-track in powder bed                    -> + powder coupling
  level 3: multi-track / hatch patches                   -> + overlap rules
  level 4: 3D coupons (NIST AM-Bench-near)               -> generalisation

This module covers the optimization mechanics shared across levels. The user
hands in a MeasurementSet and a base scenario; the calibrator searches over
the chosen parameters to minimise SimRealScore.composite.

Optimizer is a 1D golden-section line search applied per parameter, with
multiple outer rounds to handle weak coupling (no scipy dependency).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from laser_sim.config.schema import MaterialConfig, ScenarioConfig
from laser_sim.fitness.evaluator import PatternEvaluator
from laser_sim.patterns.base import ScanPattern
from laser_sim.validation.ingest import MeasurementSet
from laser_sim.validation.register import resample_field_to_grid
from laser_sim.validation.score import SimRealScore, score_thermal_frame


@dataclass(frozen=True)
class ParameterBound:
    name: str
    lo: float
    hi: float
    default: float

    def __post_init__(self) -> None:
        if not (self.lo <= self.default <= self.hi):
            raise ValueError(
                f"{self.name}: default {self.default} not in [{self.lo}, {self.hi}]"
            )


@dataclass(frozen=True)
class CalibrationProblem:
    pattern: ScanPattern
    measurements: MeasurementSet
    base_scenario: ScenarioConfig
    parameters: tuple[ParameterBound, ...]
    hatch_mm: float = 0.1
    grid_n: int = 31
    stride: int = 4


@dataclass(frozen=True)
class CalibrationResult:
    optimum: dict[str, float]
    initial_score: SimRealScore
    final_score: SimRealScore
    history: tuple[dict[str, float], ...]   # one entry per outer round
    rounds: int

    def to_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "optimum": self.optimum,
                    "initial_score": _score_to_dict(self.initial_score),
                    "final_score": _score_to_dict(self.final_score),
                    "history": list(self.history),
                    "rounds": self.rounds,
                },
                indent=2,
            )
        )
        return path


def _score_to_dict(s: SimRealScore) -> dict[str, float]:
    return {
        "rmse_K": s.rmse_K,
        "iou_melt": s.iou_melt,
        "iou_keyhole": s.iou_keyhole,
        "cosine_hist": s.cosine_hist,
        "kl_hist": s.kl_hist,
        "composite": s.composite,
    }


def _apply_parameters(
    base: ScenarioConfig, params: dict[str, float]
) -> tuple[ScenarioConfig, float]:
    """Apply tunable params to a copy of the scenario.

    Returns (mutated_scenario, effective_spot_um).

    Recognised parameter names:
      - absorptivity   -> material.absorptivity
      - emissivity     -> material.emissivity
      - spot_um        -> not stored on scenario; returned as effective spot
    """
    mat_overrides: dict[str, float] = {}
    if "absorptivity" in params:
        mat_overrides["absorptivity"] = float(np.clip(params["absorptivity"], 0.0, 1.0))
    if "emissivity" in params:
        mat_overrides["emissivity"] = float(np.clip(params["emissivity"], 0.0, 1.0))
    new_mat = (
        base.material.model_copy(update=mat_overrides) if mat_overrides else base.material
    )
    new_scenario = base.model_copy(update={"material": new_mat})
    spot_um = float(params.get("spot_um", 80.0))
    return new_scenario, spot_um


def _composite_loss(
    problem: CalibrationProblem, params: dict[str, float]
) -> SimRealScore:
    scenario, spot_um = _apply_parameters(problem.base_scenario, params)
    real = problem.measurements.first_thermal()
    if real is None:
        raise ValueError("measurement set has no thermal frame for calibration")
    ev = PatternEvaluator(
        scenario.material,
        scenario.machine,
        scenario.roi,
        mode="field",
        grid_n=problem.grid_n,
        stride=problem.stride,
        spot_um=spot_um,
    )
    sim_eval = ev.evaluate(problem.pattern, hatch_mm=problem.hatch_mm, spot_um=spot_um)
    if sim_eval.field is None:
        raise RuntimeError("sim evaluation returned no field")
    H, W = real.frame_K.shape
    real_x = real.origin_xy_mm[0] + np.arange(H) * real.pixel_um * 1e-3
    real_y = real.origin_xy_mm[1] + np.arange(W) * real.pixel_um * 1e-3
    real_on_sim = resample_field_to_grid(
        real.frame_K, real_x, real_y, sim_eval.field.grid_x_mm, sim_eval.field.grid_y_mm
    )
    valid = np.isfinite(real_on_sim)
    if not valid.any():
        return SimRealScore(
            rmse_K=float("inf"),
            iou_melt=0.0,
            iou_keyhole=0.0,
            cosine_hist=0.0,
            kl_hist=float("inf"),
            composite=float("inf"),
        )
    sim_masked = np.where(valid, sim_eval.field.t_max_K, np.nan)
    return score_thermal_frame(
        sim_field_K=sim_masked,
        real_field_K=real_on_sim,
        liquidus_K=scenario.material.liquidus_K,
        boiling_K=scenario.material.boiling_K,
    )


def _golden_section_search(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    tol: float,
    max_iter: int = 30,
) -> tuple[float, float]:
    """Minimise univariate `f` on [lo, hi]. Returns (x*, f*). No deps."""
    phi = (math.sqrt(5.0) - 1.0) / 2.0  # ~0.618
    a, b = float(lo), float(hi)
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
    if fc < fd:
        return c, fc
    return d, fd


def calibrate(
    problem: CalibrationProblem,
    rounds: int = 2,
    tol_per_param: dict[str, float] | None = None,
) -> CalibrationResult:
    """Hierarchical 1D-line-search calibration.

    For each outer round, sweep through every parameter in the order they
    appear in `problem.parameters`, holding the others at their current best
    value. Convergence is typically reached in 1-3 rounds.
    """
    tol_default = {"absorptivity": 0.005, "emissivity": 0.005, "spot_um": 0.5}
    tol_table = {**tol_default, **(tol_per_param or {})}
    current = {p.name: p.default for p in problem.parameters}
    initial = _composite_loss(problem, current)
    history: list[dict[str, float]] = [{**current, "score": initial.composite}]

    best_score = initial
    for r in range(rounds):
        improved = False
        for p in problem.parameters:
            def _f(x: float, _p=p) -> float:
                trial = {**current, _p.name: x}
                return _composite_loss(problem, trial).composite

            x_star, f_star = _golden_section_search(
                _f, p.lo, p.hi, tol=tol_table.get(p.name, 1e-3)
            )
            if f_star + 1e-9 < best_score.composite:
                current[p.name] = x_star
                best_score = _composite_loss(problem, current)
                improved = True
        history.append({**current, "score": best_score.composite, "round": float(r)})
        if not improved:
            break

    return CalibrationResult(
        optimum={**current},
        initial_score=initial,
        final_score=best_score,
        history=tuple(history),
        rounds=rounds,
    )


DEFAULT_LEVEL1_PARAMETERS: tuple[ParameterBound, ...] = (
    ParameterBound("absorptivity", lo=0.10, hi=0.80, default=0.35),
    ParameterBound("spot_um", lo=40.0, hi=200.0, default=80.0),
    ParameterBound("emissivity", lo=0.10, hi=0.80, default=0.40),
)
