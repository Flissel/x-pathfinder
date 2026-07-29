"""Rasterizer: ScanPattern -> dense (x, y, t, P, v) trajectory.

Samples each Segment at a configurable spacing in either time or arc length;
arc-length sampling is the default because it keeps the spatial resolution
uniform irrespective of speed and matches what the heat solver needs.
"""

from __future__ import annotations

import numpy as np

from laser_sim.patterns.base import RasterizedPath, ScanPattern, Segment


def _sample_segment(seg: Segment, ds_mm: float, t0: float) -> tuple[np.ndarray, ...]:
    pts = np.array([[w.x_mm, w.y_mm] for w in seg.waypoints], dtype=float)
    if len(pts) < 2 or seg.speed_mm_s <= 0:
        return (
            np.empty(0),
            np.empty(0),
            np.empty(0),
            np.empty(0),
            np.empty(0),
            np.empty(0, dtype=bool),
        )
    deltas = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(deltas, axis=1)
    total = float(seg_lens.sum())
    if total <= 0:
        return (
            np.empty(0),
            np.empty(0),
            np.empty(0),
            np.empty(0),
            np.empty(0),
            np.empty(0, dtype=bool),
        )
    n = max(2, int(np.ceil(total / max(ds_mm, 1e-6))) + 1)
    s = np.linspace(0.0, total, n)
    # locate each s in cumulative arc length
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    idx = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg_lens) - 1)
    frac = (s - cum[idx]) / np.maximum(seg_lens[idx], 1e-12)
    xy = pts[idx] + deltas[idx] * frac[:, None]
    t = t0 + s / seg.speed_mm_s  # mm / (mm/s) = s
    P = np.full(n, seg.power_W if seg.laser_on else 0.0)
    v = np.full(n, seg.speed_mm_s)
    on = np.full(n, seg.laser_on, dtype=bool)
    return xy[:, 0], xy[:, 1], t, P, v, on


def rasterize_pattern(pattern: ScanPattern, ds_mm: float = 0.02) -> RasterizedPath:
    """Resample pattern at uniform arc length `ds_mm` (default 20 um)."""
    if ds_mm <= 0:
        raise ValueError("ds_mm must be positive")
    xs, ys, ts, Ps, vs, ons = [], [], [], [], [], []
    t_cursor = 0.0
    for seg in pattern.segments:
        x, y, t, P, v, on = _sample_segment(seg, ds_mm, t_cursor)
        if x.size == 0:
            continue
        xs.append(x)
        ys.append(y)
        ts.append(t)
        Ps.append(P)
        vs.append(v)
        ons.append(on)
        t_cursor = float(t[-1])
    if not xs:
        return RasterizedPath(
            x_mm=np.empty(0),
            y_mm=np.empty(0),
            t_s=np.empty(0),
            power_W=np.empty(0),
            speed_mm_s=np.empty(0),
            laser_on=np.empty(0, dtype=bool),
        )
    return RasterizedPath(
        x_mm=np.concatenate(xs),
        y_mm=np.concatenate(ys),
        t_s=np.concatenate(ts),
        power_W=np.concatenate(Ps),
        speed_mm_s=np.concatenate(vs),
        laser_on=np.concatenate(ons),
    )
