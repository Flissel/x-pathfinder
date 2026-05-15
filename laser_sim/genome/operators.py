"""Mutation, crossover, and selection operators.

Five mutation operators (matching the parent x-pathfinder pattern):
  - jitter_continuous: Gaussian perturbation of (P, v, hatch, spot)
  - kind_swap: replace primitive_kind with another eligible kind
  - rotation_flip: pick a new rotation from {0, 67, 90}
  - stripe_width_jitter: perturb extras['stripe_width_mm'] when applicable
  - spiral_resolution_swap: shift extras['samples_per_turn'] when applicable

Crisis mode (analog to x-pathfinder/genome.py:114-152) multiplies the
mutation sigma when the EA stagnates.

Selection: tournament-k on Pareto rank (primary) then crowding distance.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

from laser_sim.config.schema import EAConfig, MachineConfig
from laser_sim.genome.chromosome import (
    Chromosome,
    allowed_rotations,
    eligible_kinds,
)
from laser_sim.patterns.base import PrimitiveKind


def _jitter_continuous(c: Chromosome, machine: MachineConfig, sigma_mult: float, rng: random.Random) -> Chromosome:
    las = machine.laser
    p_span = las.power_max_W - las.power_min_W
    v_span = las.speed_max_mm_s - las.speed_min_mm_s
    s_span = las.spot_max_um - las.spot_min_um
    h_span = 200.0 - 40.0
    base_sigma = 0.05 * sigma_mult
    return replace(
        c,
        power_W=c.power_W + rng.gauss(0.0, base_sigma * p_span),
        speed_mm_s=c.speed_mm_s + rng.gauss(0.0, base_sigma * v_span),
        hatch_um=c.hatch_um + rng.gauss(0.0, base_sigma * h_span),
        spot_um=c.spot_um + rng.gauss(0.0, base_sigma * s_span),
    ).clamped(machine)


def _kind_swap(c: Chromosome, machine: MachineConfig, sigma_mult: float, rng: random.Random) -> Chromosome:
    kinds = [k for k in eligible_kinds() if k is not c.primitive_kind]
    if not kinds:
        return c
    new = rng.choice(kinds)
    extras: dict[str, Any] = {}
    if new is PrimitiveKind.STRIPES:
        extras["stripe_width_mm"] = rng.uniform(1.0, 5.0)
    elif new is PrimitiveKind.SPIRAL:
        extras["samples_per_turn"] = rng.choice([32, 48, 64, 96])
    elif new is PrimitiveKind.HILBERT:
        extras["order"] = rng.choice([3, 4, 5])
    return replace(c, primitive_kind=new, extras=extras).clamped(machine)


def _rotation_flip(c: Chromosome, machine: MachineConfig, sigma_mult: float, rng: random.Random) -> Chromosome:
    opts = [r for r in allowed_rotations() if r != c.layer_rotation_deg]
    if not opts:
        return c
    return replace(c, layer_rotation_deg=rng.choice(opts)).clamped(machine)


def _stripe_width_jitter(c: Chromosome, machine: MachineConfig, sigma_mult: float, rng: random.Random) -> Chromosome:
    if c.primitive_kind is not PrimitiveKind.STRIPES:
        return c
    w = float(c.extras.get("stripe_width_mm", 5.0))
    w_new = max(0.5, min(10.0, w + rng.gauss(0.0, 0.5 * sigma_mult)))
    return replace(c, extras={**c.extras, "stripe_width_mm": w_new}).clamped(machine)


def _spiral_res_swap(c: Chromosome, machine: MachineConfig, sigma_mult: float, rng: random.Random) -> Chromosome:
    if c.primitive_kind is not PrimitiveKind.SPIRAL:
        return c
    current = int(c.extras.get("samples_per_turn", 64))
    choices = [n for n in (32, 48, 64, 96, 128) if n != current]
    new = rng.choice(choices)
    return replace(c, extras={**c.extras, "samples_per_turn": new}).clamped(machine)


_MUTATIONS = (
    ("jitter_continuous", _jitter_continuous, 0.45),
    ("kind_swap", _kind_swap, 0.15),
    ("rotation_flip", _rotation_flip, 0.15),
    ("stripe_width_jitter", _stripe_width_jitter, 0.15),
    ("spiral_resolution_swap", _spiral_res_swap, 0.10),
)


def mutate(c: Chromosome, machine: MachineConfig, ea: EAConfig, crisis: bool, rng: random.Random) -> Chromosome:
    """Apply one randomly-chosen mutation operator weighted by importance."""
    sigma_mult = ea.crisis_multiplier if crisis else 1.0
    names, fns, weights = zip(*[(n, f, w) for n, f, w in _MUTATIONS])
    op = rng.choices(list(zip(names, fns)), weights=list(weights), k=1)[0]
    return op[1](c, machine, sigma_mult, rng)


def crossover_blx_alpha(
    a: Chromosome, b: Chromosome, machine: MachineConfig, rng: random.Random, alpha: float = 0.5
) -> Chromosome:
    """BLX-alpha on continuous genes; uniform on discrete genes.

    Continuous: child gene in [min - alpha*d, max + alpha*d]. Discrete: pick
    either parent's value uniformly. extras come from whichever parent
    contributed primitive_kind (so type stays consistent).
    """

    def blx(x: float, y: float) -> float:
        lo, hi = min(x, y), max(x, y)
        d = hi - lo
        return rng.uniform(lo - alpha * d, hi + alpha * d)

    use_a_kind = rng.random() < 0.5
    kind_parent = a if use_a_kind else b
    child = Chromosome(
        primitive_kind=kind_parent.primitive_kind,
        power_W=blx(a.power_W, b.power_W),
        speed_mm_s=blx(a.speed_mm_s, b.speed_mm_s),
        hatch_um=blx(a.hatch_um, b.hatch_um),
        spot_um=blx(a.spot_um, b.spot_um),
        layer_rotation_deg=rng.choice([a.layer_rotation_deg, b.layer_rotation_deg]),
        extras=dict(kind_parent.extras),
    )
    return child.clamped(machine)


def tournament_select(
    indices: list[int],
    ranks: list[int],
    crowding: list[float],
    k: int,
    rng: random.Random,
) -> int:
    """NSGA-II tournament: lower rank wins; tie -> larger crowding distance wins."""
    if not indices:
        raise ValueError("empty selection pool")
    picks = [rng.choice(indices) for _ in range(k)]
    picks.sort(key=lambda i: (ranks[i], -crowding[i]))
    return picks[0]
