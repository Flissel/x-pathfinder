"""Matplotlib-based pattern preview.

pyvista/plotly are optional; we use matplotlib here so the first CLI command
runs without extra dependencies.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from laser_sim.patterns.base import RasterizedPath, ScanPattern
from laser_sim.patterns.rasterize import rasterize_pattern


def plot_pattern(
    pattern: ScanPattern,
    out: Path | None = None,
    ds_mm: float = 0.05,
    title: str | None = None,
    show: bool = False,
) -> Path | None:
    """Render scan pattern as a colored polyline (color = speed). Returns out path."""
    rp: RasterizedPath = rasterize_pattern(pattern, ds_mm=ds_mm)
    if rp.n_samples() < 2:
        raise ValueError("pattern rasterized to fewer than 2 points; nothing to plot")
    pts = np.column_stack([rp.x_mm, rp.y_mm])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    fig, ax = plt.subplots(figsize=(7, 7))
    lc = LineCollection(segs, cmap="viridis", linewidth=1.2)
    lc.set_array(rp.power_W[:-1])
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(title or "Scan pattern (color = laser power [W])")
    fig.colorbar(lc, ax=ax, label="Power [W]")
    n_seg = len(pattern.segments)
    t_total = pattern.total_time_s()
    l_total = pattern.total_length_mm()
    ax.text(
        0.02,
        0.98,
        f"segments: {n_seg}\nlength: {l_total:.2f} mm\nscan time: {t_total*1e3:.1f} ms",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="0.7"),
    )
    fig.tight_layout()
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return out
