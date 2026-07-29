"""Chromosome model for LPBF pattern optimization.

A Chromosome is a tuple of genes; for the first slice we pin the structure to
(PrimitiveGene, ContinuousGene(power, speed, hatch, spot), LayerRotationGene)
so the EA can search the full pattern family + continuous parameters without
also juggling waypoint lists. WaypointGene and SegmentGene come later.

Mutation/crossover happen on the gene values, not on rasterized geometry.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field, replace
from typing import Any

from laser_sim.config.schema import MachineConfig
from laser_sim.patterns.base import PrimitiveKind, PrimitiveSpec
from laser_sim.patterns.primitives import _BUILDERS  # only built-in kinds

# Primitives the EA may use. WAYPOINT needs an explicit polyline, which the
# current Chromosome doesn't carry — that lands when WaypointGene is added.
_EA_EXCLUDED: frozenset[PrimitiveKind] = frozenset({PrimitiveKind.WAYPOINT})
_ELIGIBLE_KINDS: tuple[PrimitiveKind, ...] = tuple(
    k for k in _BUILDERS.keys() if k not in _EA_EXCLUDED
)
_ALLOWED_ROTATIONS: tuple[float, ...] = (0.0, 67.0, 90.0)


@dataclass(frozen=True)
class Chromosome:
    """Pattern genome.

    primitive_kind: which fill primitive (zigzag/stripes/spiral so far)
    power_W, speed_mm_s, hatch_um, spot_um: continuous parameters bounded
        by the machine envelope.
    layer_rotation_deg: discrete in {0, 67, 90} per the plan.
    extras: kind-specific params, e.g. {"stripe_width_mm": ...} for stripes.
    """

    primitive_kind: PrimitiveKind
    power_W: float
    speed_mm_s: float
    hatch_um: float
    spot_um: float
    layer_rotation_deg: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    def hash_id(self) -> str:
        """Stable, deterministic identifier for caching."""
        payload = {
            "k": self.primitive_kind.value,
            "P": round(self.power_W, 4),
            "v": round(self.speed_mm_s, 4),
            "h": round(self.hatch_um, 4),
            "s": round(self.spot_um, 4),
            "r": round(self.layer_rotation_deg, 4),
            "x": {k: round(v, 6) if isinstance(v, float) else v for k, v in sorted(self.extras.items())},
        }
        return hashlib.blake2b(
            json.dumps(payload, sort_keys=True).encode(), digest_size=10
        ).hexdigest()

    def to_primitive_spec(self) -> PrimitiveSpec:
        return PrimitiveSpec(
            kind=self.primitive_kind,
            power_W=self.power_W,
            speed_mm_s=self.speed_mm_s,
            hatch_um=self.hatch_um,
            spot_um=self.spot_um,
            rotation_deg=self.layer_rotation_deg,
            params=dict(self.extras),
        )

    def clamped(self, machine: MachineConfig) -> "Chromosome":
        las = machine.laser
        return replace(
            self,
            power_W=float(min(max(self.power_W, las.power_min_W), las.power_max_W)),
            speed_mm_s=float(
                min(max(self.speed_mm_s, las.speed_min_mm_s), las.speed_max_mm_s)
            ),
            hatch_um=float(max(20.0, min(self.hatch_um, 300.0))),
            spot_um=float(
                min(max(self.spot_um, las.spot_min_um), las.spot_max_um)
            ),
            layer_rotation_deg=float(
                min(_ALLOWED_ROTATIONS, key=lambda r: abs(r - self.layer_rotation_deg))
            ),
        )


def random_chromosome(machine: MachineConfig, rng: random.Random) -> Chromosome:
    """Uniform random in the machine envelope."""
    las = machine.laser
    kind = rng.choice(_ELIGIBLE_KINDS)
    extras: dict[str, Any] = {}
    if kind is PrimitiveKind.STRIPES:
        extras["stripe_width_mm"] = rng.uniform(1.0, 5.0)
    elif kind is PrimitiveKind.SPIRAL:
        extras["samples_per_turn"] = rng.choice([32, 48, 64, 96])
    elif kind is PrimitiveKind.HILBERT:
        extras["order"] = rng.choice([3, 4, 5])
    elif kind is PrimitiveKind.ISLAND:
        extras["tile_size_mm"] = rng.uniform(1.5, 4.0)
        extras["shuffle"] = True
        extras["shuffle_seed"] = rng.randrange(0, 1_000_000)
    elif kind is PrimitiveKind.VORONOI:
        extras["n_cells"] = rng.choice([4, 6, 8, 12, 16])
        extras["lloyd_iter"] = rng.choice([1, 2, 3])
        extras["seed"] = rng.randrange(0, 1_000_000)
    elif kind is PrimitiveKind.ADAPTIVE_PATCH:
        extras["grid_nx"] = rng.choice([2, 3])
        extras["grid_ny"] = rng.choice([2, 3])
    return Chromosome(
        primitive_kind=kind,
        power_W=rng.uniform(las.power_min_W, las.power_max_W),
        speed_mm_s=rng.uniform(las.speed_min_mm_s, las.speed_max_mm_s),
        hatch_um=rng.uniform(40.0, 200.0),
        spot_um=rng.uniform(las.spot_min_um, las.spot_max_um),
        layer_rotation_deg=rng.choice(_ALLOWED_ROTATIONS),
        extras=extras,
    )


def eligible_kinds() -> tuple[PrimitiveKind, ...]:
    return _ELIGIBLE_KINDS


def allowed_rotations() -> tuple[float, ...]:
    return _ALLOWED_ROTATIONS
