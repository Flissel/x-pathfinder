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


def _build_voronoi(spec: PrimitiveSpec, roi: GeometryROI) -> tuple[Segment, ...]:
    """Voronoi-patch scanning: ROI partitioned into N random seeds (Lloyd-
    relaxed for low variance), each Voronoi cell filled with a small
    parallel raster aligned to a random angle.

    params:
      n_cells: target seed count (default 8)
      lloyd_iter: Lloyd relaxation iterations (default 3; 0 disables)
      seed: rng seed
    """
    n_cells = max(2, int(spec.params.get("n_cells", 8)))
    lloyd_iter = max(0, int(spec.params.get("lloyd_iter", 3)))
    seed = int(spec.params.get("seed", 0))
    hatch_mm = spec.hatch_um * 1e-3
    import random as _random

    rng = _random.Random(seed)
    cx = 0.5 * (roi.x0_mm + roi.x1_mm)
    cy = 0.5 * (roi.y0_mm + roi.y1_mm)
    pts = np.array(
        [
            [rng.uniform(roi.x0_mm, roi.x1_mm), rng.uniform(roi.y0_mm, roi.y1_mm)]
            for _ in range(n_cells)
        ]
    )

    # Lloyd's algorithm via Monte-Carlo centroidal estimation (no scipy):
    # sample many points inside ROI, assign each to its nearest seed, and
    # move each seed to the centroid of its assigned samples.
    n_mc = 2000
    for _ in range(lloyd_iter):
        samples = np.column_stack(
            [
                np.array([rng.uniform(roi.x0_mm, roi.x1_mm) for _ in range(n_mc)]),
                np.array([rng.uniform(roi.y0_mm, roi.y1_mm) for _ in range(n_mc)]),
            ]
        )
        d2 = ((samples[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
        owner = np.argmin(d2, axis=1)
        new_pts = pts.copy()
        for k in range(n_cells):
            mask = owner == k
            if mask.any():
                new_pts[k] = samples[mask].mean(axis=0)
        pts = new_pts

    # For each Voronoi cell: sample grid points inside the cell, fit an
    # axis-aligned bounding box rotated by a per-cell random angle, then
    # raster-fill that box with hatch lines. Lines outside the cell are
    # clipped at the cell boundary (defined by nearest-seed rule).
    grid_n = 80
    gx = np.linspace(roi.x0_mm, roi.x1_mm, grid_n)
    gy = np.linspace(roi.y0_mm, roi.y1_mm, grid_n)
    Xg, Yg = np.meshgrid(gx, gy, indexing="ij")
    cells = np.column_stack([Xg.ravel(), Yg.ravel()])
    d2 = ((cells[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    owner = np.argmin(d2, axis=1)

    segments: list[Segment] = []
    for k in range(n_cells):
        mask = owner == k
        if mask.sum() < 4:
            continue
        cell_pts = cells[mask]
        ang = rng.uniform(0.0, 90.0)
        rad = math.radians(ang)
        c_, s_ = math.cos(rad), math.sin(rad)
        # rotate cell to align with hatch axis
        local = cell_pts - cell_pts.mean(axis=0)
        rot = np.array([[c_, s_], [-s_, c_]])
        rotated = local @ rot.T
        xmin, ymin = rotated.min(axis=0)
        xmax, ymax = rotated.max(axis=0)
        n_lines = max(2, int((ymax - ymin) / hatch_mm) + 1)
        ys = np.linspace(ymin, ymax, n_lines)
        for li, y in enumerate(ys):
            if li % 2 == 0:
                p0 = np.array([xmin, y])
                p1 = np.array([xmax, y])
            else:
                p0 = np.array([xmax, y])
                p1 = np.array([xmin, y])
            # rotate back into global frame
            world = np.stack([p0, p1]) @ rot + cell_pts.mean(axis=0)
            world = _rotate(world, spec.rotation_deg, (cx, cy))
            segments.append(
                Segment(
                    waypoints=(Waypoint(*world[0]), Waypoint(*world[1])),
                    power_W=spec.power_W,
                    speed_mm_s=spec.speed_mm_s,
                )
            )
    return tuple(segments)


def _build_island(spec: PrimitiveSpec, roi: GeometryROI) -> tuple[Segment, ...]:
    """Chess-board / island scanning: ROI tiled into N x N tiles, each filled
    with a small zigzag, neighbouring tiles rotated 90 deg to disrupt
    long-range residual stress and heat accumulation. Common LPBF strategy.

    params:
      tile_size_mm: side of one square tile (default 2.5 mm)
      shuffle: if True, traverse tiles in pseudo-random order; else row-major
      shuffle_seed: rng seed when shuffle is True
    """
    tile_size_mm = float(spec.params.get("tile_size_mm", 2.5))
    if tile_size_mm <= 0:
        raise ValueError("tile_size_mm must be positive")
    shuffle = bool(spec.params.get("shuffle", True))
    seed = int(spec.params.get("shuffle_seed", 0))
    hatch_mm = spec.hatch_um * 1e-3
    cx = 0.5 * (roi.x0_mm + roi.x1_mm)
    cy = 0.5 * (roi.y0_mm + roi.y1_mm)

    nx = max(1, math.ceil(roi.width_mm / tile_size_mm))
    ny = max(1, math.ceil(roi.height_mm / tile_size_mm))
    tile_indices = [(ix, iy) for ix in range(nx) for iy in range(ny)]
    if shuffle:
        import random as _random

        rng = _random.Random(seed)
        rng.shuffle(tile_indices)

    segments: list[Segment] = []
    for ix, iy in tile_indices:
        x0 = roi.x0_mm + ix * tile_size_mm
        y0 = roi.y0_mm + iy * tile_size_mm
        x1 = min(roi.x0_mm + (ix + 1) * tile_size_mm, roi.x1_mm)
        y1 = min(roi.y0_mm + (iy + 1) * tile_size_mm, roi.y1_mm)
        if x1 - x0 <= 1e-9 or y1 - y0 <= 1e-9:
            continue
        # checkerboard parity decides scan direction (alternate by 90 deg)
        rotate_90 = ((ix + iy) % 2) == 1
        n_lines = max(2, int((y1 - y0) / hatch_mm) + 1)
        ys = np.linspace(y0, y1, n_lines)
        for li, y in enumerate(ys):
            if not rotate_90:
                if li % 2 == 0:
                    p0 = np.array([x0, y])
                    p1 = np.array([x1, y])
                else:
                    p0 = np.array([x1, y])
                    p1 = np.array([x0, y])
            else:
                # transposed: scan vertically within tile
                xc = x0 + (y - y0)  # parametrise by line index instead
                xc = x0 + li * (x1 - x0) / max(n_lines - 1, 1)
                if li % 2 == 0:
                    p0 = np.array([xc, y0])
                    p1 = np.array([xc, y1])
                else:
                    p0 = np.array([xc, y1])
                    p1 = np.array([xc, y0])
            pts = _rotate(np.stack([p0, p1]), spec.rotation_deg, (cx, cy))
            segments.append(
                Segment(
                    waypoints=(Waypoint(*pts[0]), Waypoint(*pts[1])),
                    power_W=spec.power_W,
                    speed_mm_s=spec.speed_mm_s,
                )
            )
    return tuple(segments)


def _build_waypoint(spec: PrimitiveSpec, roi: GeometryROI) -> tuple[Segment, ...]:
    """Explicit polyline. params['waypoints'] = [(x_mm, y_mm), ...].

    Optional params:
      power_per_segment: list of P (W), one per polyline edge (overrides spec.power_W)
      speed_per_segment: list of v (mm/s), one per edge (overrides spec.speed_mm_s)
    """
    raw = spec.params.get("waypoints")
    if not raw:
        raise ValueError("waypoint primitive requires params['waypoints'] = [(x, y), ...]")
    pts = np.array(raw, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
        raise ValueError(f"waypoints must be (>=2, 2); got shape {pts.shape}")
    cx = 0.5 * (roi.x0_mm + roi.x1_mm)
    cy = 0.5 * (roi.y0_mm + roi.y1_mm)
    pts = _rotate(pts, spec.rotation_deg, (cx, cy))
    powers = spec.params.get("power_per_segment")
    speeds = spec.params.get("speed_per_segment")
    if powers is not None or speeds is not None:
        n_edges = pts.shape[0] - 1
        powers = list(powers) if powers is not None else [spec.power_W] * n_edges
        speeds = list(speeds) if speeds is not None else [spec.speed_mm_s] * n_edges
        if len(powers) != n_edges or len(speeds) != n_edges:
            raise ValueError(
                f"power/speed per segment must have length {n_edges}; "
                f"got {len(powers)}, {len(speeds)}"
            )
        segments = tuple(
            Segment(
                waypoints=(Waypoint(*pts[k]), Waypoint(*pts[k + 1])),
                power_W=float(powers[k]),
                speed_mm_s=float(speeds[k]),
            )
            for k in range(n_edges)
        )
        return segments
    wps = tuple(Waypoint(float(p[0]), float(p[1])) for p in pts)
    return (Segment(waypoints=wps, power_W=spec.power_W, speed_mm_s=spec.speed_mm_s),)


_BUILDERS: dict[PrimitiveKind, Callable[[PrimitiveSpec, GeometryROI], tuple[Segment, ...]]] = {
    PrimitiveKind.ZIGZAG: _build_zigzag,
    PrimitiveKind.STRIPES: _build_stripes,
    PrimitiveKind.SPIRAL: _build_spiral,
    PrimitiveKind.HILBERT: _build_hilbert,
    PrimitiveKind.ISLAND: _build_island,
    PrimitiveKind.VORONOI: _build_voronoi,
    PrimitiveKind.WAYPOINT: _build_waypoint,
}


def list_primitives() -> list[str]:
    return [k.value for k in PrimitiveKind]


def build_primitive(spec: PrimitiveSpec, roi: GeometryROI) -> ScanPattern:
    builder = _BUILDERS.get(spec.kind)
    if builder is None:
        raise NotImplementedError(f"primitive {spec.kind.value} not implemented yet")
    return ScanPattern(segments=builder(spec, roi), rotation_deg=spec.rotation_deg)
