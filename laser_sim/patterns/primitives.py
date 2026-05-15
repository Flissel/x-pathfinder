"""Pattern primitive builders.

Each primitive maps a PrimitiveSpec + GeometryROI to a ScanPattern (a tuple
of Segments). Coordinates are in mm; rotation_deg rotates about the ROI centre.
Only zigzag, stripes, and spiral are implemented in this first slice; the
remaining primitives raise NotImplementedError but are registered so the EA
can already mutate `kind`.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from laser_sim.config.schema import GeometryROI
from laser_sim.patterns.base import (
    PrimitiveKind,
    PrimitiveSpec,
    ScanPattern,
    Segment,
    Waypoint,
)


def _rotate(points: np.ndarray, deg: float, centre: tuple[float, float]) -> np.ndarray:
    if deg == 0.0:
        return points
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    rot = np.array([[c, -s], [s, c]])
    cx, cy = centre
    return (points - np.array([cx, cy])) @ rot.T + np.array([cx, cy])


def _build_zigzag(spec: PrimitiveSpec, roi: GeometryROI) -> tuple[Segment, ...]:
    hatch_mm = spec.hatch_um * 1e-3
    if hatch_mm <= 0:
        raise ValueError("hatch_um must be positive")
    n_lines = max(2, int(roi.height_mm / hatch_mm) + 1)
    ys = np.linspace(roi.y0_mm, roi.y1_mm, n_lines)
    segments: list[Segment] = []
    for i, y in enumerate(ys):
        if i % 2 == 0:
            p0 = np.array([roi.x0_mm, y])
            p1 = np.array([roi.x1_mm, y])
        else:
            p0 = np.array([roi.x1_mm, y])
            p1 = np.array([roi.x0_mm, y])
        cx = 0.5 * (roi.x0_mm + roi.x1_mm)
        cy = 0.5 * (roi.y0_mm + roi.y1_mm)
        pts = _rotate(np.stack([p0, p1]), spec.rotation_deg, (cx, cy))
        segments.append(
            Segment(
                waypoints=(Waypoint(*pts[0]), Waypoint(*pts[1])),
                power_W=spec.power_W,
                speed_mm_s=spec.speed_mm_s,
            )
        )
    return tuple(segments)


def _build_stripes(spec: PrimitiveSpec, roi: GeometryROI) -> tuple[Segment, ...]:
    """Stripes: fill ROI with parallel stripes of width `stripe_width_mm`, each
    stripe internally rastered with `hatch_um`. Common LPBF strategy (EOS).
    """
    stripe_width_mm = float(spec.params.get("stripe_width_mm", 5.0))
    if stripe_width_mm <= 0:
        raise ValueError("stripe_width_mm must be positive")
    hatch_mm = spec.hatch_um * 1e-3
    segments: list[Segment] = []
    cx = 0.5 * (roi.x0_mm + roi.x1_mm)
    cy = 0.5 * (roi.y0_mm + roi.y1_mm)
    x = roi.x0_mm
    stripe_idx = 0
    while x < roi.x1_mm - 1e-9:
        x_next = min(x + stripe_width_mm, roi.x1_mm)
        n_lines = max(2, int((x_next - x) / hatch_mm) + 1)
        xs = np.linspace(x, x_next, n_lines)
        for i, xc in enumerate(xs):
            if (stripe_idx + i) % 2 == 0:
                p0 = np.array([xc, roi.y0_mm])
                p1 = np.array([xc, roi.y1_mm])
            else:
                p0 = np.array([xc, roi.y1_mm])
                p1 = np.array([xc, roi.y0_mm])
            pts = _rotate(np.stack([p0, p1]), spec.rotation_deg, (cx, cy))
            segments.append(
                Segment(
                    waypoints=(Waypoint(*pts[0]), Waypoint(*pts[1])),
                    power_W=spec.power_W,
                    speed_mm_s=spec.speed_mm_s,
                )
            )
        stripe_idx += 1
        x = x_next
    return tuple(segments)


def _hilbert_d2xy(n: int, d: int) -> tuple[int, int]:
    """Convert 1D Hilbert distance d in [0, n*n) to 2D (x, y) coords in [0, n).

    Standard Karney/Wikipedia algorithm.
    """
    rx = ry = 0
    x = y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def _build_hilbert(spec: PrimitiveSpec, roi: GeometryROI) -> tuple[Segment, ...]:
    """Hilbert space-filling curve covering the ROI.

    Order is inferred from hatch: pick the smallest order N such that
    cell_size = ROI_extent / (2^N - 1) <= hatch.
    """
    hatch_mm = spec.hatch_um * 1e-3
    extent = max(roi.width_mm, roi.height_mm)
    if hatch_mm <= 0 or extent <= 0:
        raise ValueError("ROI/hatch invalid")
    order = max(int(spec.params.get("order", 0)), 0)
    if order == 0:
        order = max(1, math.ceil(math.log2(extent / hatch_mm + 1)))
    order = min(order, 6)  # cap so we don't explode segment count
    n = 1 << order
    cx = 0.5 * (roi.x0_mm + roi.x1_mm)
    cy = 0.5 * (roi.y0_mm + roi.y1_mm)
    pts = np.array(
        [_hilbert_d2xy(n, d) for d in range(n * n)], dtype=float
    )
    pts[:, 0] = roi.x0_mm + pts[:, 0] / max(n - 1, 1) * roi.width_mm
    pts[:, 1] = roi.y0_mm + pts[:, 1] / max(n - 1, 1) * roi.height_mm
    pts = _rotate(pts, spec.rotation_deg, (cx, cy))
    wps = tuple(Waypoint(float(p[0]), float(p[1])) for p in pts)
    return (Segment(waypoints=wps, power_W=spec.power_W, speed_mm_s=spec.speed_mm_s),)


def _build_spiral(spec: PrimitiveSpec, roi: GeometryROI) -> tuple[Segment, ...]:
    """Inward Archimedean spiral covering ROI."""
    hatch_mm = spec.hatch_um * 1e-3
    cx = 0.5 * (roi.x0_mm + roi.x1_mm)
    cy = 0.5 * (roi.y0_mm + roi.y1_mm)
    r_max = 0.5 * min(roi.width_mm, roi.height_mm)
    if r_max <= 0 or hatch_mm <= 0:
        raise ValueError("ROI too small or hatch invalid")
    n_turns = max(1, int(r_max / hatch_mm))
    samples_per_turn = int(spec.params.get("samples_per_turn", 64))
    n_total = n_turns * samples_per_turn
    theta = np.linspace(0.0, 2.0 * math.pi * n_turns, n_total)
    r = np.linspace(r_max, 0.0, n_total)
    pts = np.stack([cx + r * np.cos(theta), cy + r * np.sin(theta)], axis=1)
    pts = _rotate(pts, spec.rotation_deg, (cx, cy))
    wps = tuple(Waypoint(float(p[0]), float(p[1])) for p in pts)
    return (
        Segment(waypoints=wps, power_W=spec.power_W, speed_mm_s=spec.speed_mm_s),
    )


_BUILDERS: dict[PrimitiveKind, Callable[[PrimitiveSpec, GeometryROI], tuple[Segment, ...]]] = {
    PrimitiveKind.ZIGZAG: _build_zigzag,
    PrimitiveKind.STRIPES: _build_stripes,
    PrimitiveKind.SPIRAL: _build_spiral,
    PrimitiveKind.HILBERT: _build_hilbert,
}


def list_primitives() -> list[str]:
    return [k.value for k in PrimitiveKind]


def build_primitive(spec: PrimitiveSpec, roi: GeometryROI) -> ScanPattern:
    builder = _BUILDERS.get(spec.kind)
    if builder is None:
        raise NotImplementedError(f"primitive {spec.kind.value} not implemented yet")
    return ScanPattern(segments=builder(spec, roi), rotation_deg=spec.rotation_deg)
