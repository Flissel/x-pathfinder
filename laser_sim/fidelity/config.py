"""Fidelity tiers + promotion policy.

The EA loop runs everything through the Fast Sim (Eagar-Tsai proxy /
JAX/CuPy 2.5D when available). A small Top-K subset is promoted to
High-Fidelity (DEM + OpenFOAM/laserbeamFoam) per generation. Real
experiments form the third tier and are scheduled separately via
hardware/.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FidelityTier(str, Enum):
    FAST = "fast"
    HF = "hf"
    EXPERIMENT = "experiment"


@dataclass(frozen=True)
class PromotionPolicy:
    """When to promote candidates from Fast -> HF.

    fast_only_until_gen: skip HF entirely for the first N generations
        (warm-up; Pareto front needs to settle first)
    hf_promote_top_k: promote this many top-J candidates per generation
    hf_promote_pareto_front: also promote every non-dominated member
    surrogate_min_runs: only consult the surrogate when at least this many
        evaluations are in the cache
    veto_threshold_sigma: if HF disagrees with the Fast-Sim's prediction
        beyond this many surrogate-sigma, flag for recalibration
    """

    fast_only_until_gen: int = 3
    hf_promote_top_k: int = 5
    hf_promote_pareto_front: bool = True
    surrogate_min_runs: int = 200
    veto_threshold_sigma: float = 2.0
