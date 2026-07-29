"""Scan pattern core types.

ScanPattern is a list of Segments (each a polyline at constant power/speed).
A PrimitiveSpec is a high-level description that primitives.py expands into
Segments inside a GeometryROI. The rasterizer then samples each Segment into
a dense (x, y, t, P, v) trajectory consumed by the fast-sim and HF case
generators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class PrimitiveKind(str, Enum):
    ZIGZAG = "zigzag"
    STRIPES = "stripes"
    ISLAND = "island"
    SPIRAL = "spiral"
    HILBERT = "hilbert"
    VORONOI = "voronoi"
    ADAPTIVE_PATCH = "adaptive_patch"
    WAYPOINT = "waypoint"


@dataclass(frozen=True)
class Waypoint:
    x_mm: float
    y_mm: float


@dataclass(frozen=True)
class Segment:
    """A constant-(P, v) polyline."""

    waypoints: tuple[Waypoint, ...]
    power_W: float
    speed_mm_s: float
    laser_on: bool = True

    def length_mm(self) -> float:
        if len(self.waypoints) < 2:
            return 0.0
        pts = np.array([[w.x_mm, w.y_mm] for w in self.waypoints])
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


@dataclass(frozen=True)
class PrimitiveSpec:
    """High-level primitive description; expanded by primitives.build_primitive."""

    kind: PrimitiveKind
    params: dict[str, Any] = field(default_factory=dict)
    power_W: float = 200.0
    speed_mm_s: float = 800.0
    hatch_um: float = 100.0
    spot_um: float = 80.0
    rotation_deg: float = 0.0


@dataclass(frozen=True)
class ScanPattern:
    """Composition of segments forming a complete layer scan."""

    segments: tuple[Segment, ...]
    rotation_deg: float = 0.0
    layer_index: int = 0

    def total_length_mm(self) -> float:
        return sum(s.length_mm() for s in self.segments if s.laser_on)

    def total_time_s(self) -> float:
        """Pure scan time at commanded speed; ignores jump moves with laser off."""
        t = 0.0
        for s in self.segments:
            if s.speed_mm_s <= 0:
                continue
            t += s.length_mm() / s.speed_mm_s  # mm / (mm/s) = s
        return t


@dataclass(frozen=True)
class RasterizedPath:
    """Dense (x, y, t, P, v) trajectory ready for solvers.

    Arrays are 1-D and aligned: t[k] is the time at (x[k], y[k]).
    """

    x_mm: np.ndarray
    y_mm: np.ndarray
    t_s: np.ndarray
    power_W: np.ndarray
    speed_mm_s: np.ndarray
    laser_on: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.x_mm)
        for arr, name in [
            (self.y_mm, "y_mm"),
            (self.t_s, "t_s"),
            (self.power_W, "power_W"),
            (self.speed_mm_s, "speed_mm_s"),
            (self.laser_on, "laser_on"),
        ]:
            if len(arr) != n:
                raise ValueError(f"length mismatch: {name} has {len(arr)}, expected {n}")

    def duration_s(self) -> float:
        return float(self.t_s[-1] - self.t_s[0]) if len(self.t_s) else 0.0

    def n_samples(self) -> int:
        return int(len(self.x_mm))
