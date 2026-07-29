"""T_max field heatmap (overlaying the scan path)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from laser_sim.patterns.base import RasterizedPath
from laser_sim.physics.fast_sim.transient import FieldResult


def plot_field(
    field: FieldResult,
    path: RasterizedPath | None = None,
    out: Path | None = None,
    title: str | None = None,
    show: bool = False,
) -> Path | None:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    extent = (
        float(field.grid_x_mm[0]),
        float(field.grid_x_mm[-1]),
        float(field.grid_y_mm[0]),
        float(field.grid_y_mm[-1]),
    )
    im = ax.imshow(
        field.t_max_K.T,
        origin="lower",
        extent=extent,
        cmap="inferno",
        aspect="equal",
    )
    fig.colorbar(im, ax=ax, label="T_max [K]")
    if path is not None and path.n_samples() > 1:
        ax.plot(path.x_mm, path.y_mm, "-", color="cyan", linewidth=0.6, alpha=0.55)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    cov = field.coverage_fraction
    kh = field.keyhole_fraction
    txt = (
        f"coverage: {cov*100:.1f}%\n"
        f"keyhole area: {kh*100:.1f}%\n"
        f"<T_max>: {field.t_max_mean_K:.0f} K\n"
        f"std(T_max): {field.t_max_std_K:.0f} K"
    )
    ax.text(
        0.02, 0.98, txt, transform=ax.transAxes, va="top", ha="left",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.78, edgecolor="0.7"),
    )
    ax.set_title(title or "T_max field (Eagar-Tsai per-source max)")
    fig.tight_layout()
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return out
