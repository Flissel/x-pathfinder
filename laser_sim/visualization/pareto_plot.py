"""Pareto-front diagnostic plots.

Renders a 2D scatter of any two objective axes plus optional projections of
the full archive, colored by scalar_J.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def plot_pareto_2d(
    fitnesses: Iterable[tuple[float, ...]],
    scalars: Iterable[float],
    axes: tuple[int, int] = (2, 4),
    objective_names: tuple[str, ...] | None = None,
    out: Path | None = None,
    title: str | None = None,
    show: bool = False,
) -> Path | None:
    fits = np.array(list(fitnesses), dtype=float)
    sca = np.array(list(scalars), dtype=float)
    if fits.ndim != 2 or fits.shape[0] == 0:
        raise ValueError("no fitness vectors to plot")
    ax_x, ax_y = axes
    name_x = objective_names[ax_x] if objective_names else f"obj[{ax_x}]"
    name_y = objective_names[ax_y] if objective_names else f"obj[{ax_y}]"
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(fits[:, ax_x], fits[:, ax_y], c=sca, cmap="viridis", s=50, edgecolor="0.2")
    fig.colorbar(sc, ax=ax, label="scalar J")
    ax.set_xlabel(name_x)
    ax.set_ylabel(name_y)
    ax.set_title(title or f"Pareto archive: {name_x} vs {name_y}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return out


def plot_pareto_matrix(
    fitnesses: Iterable[tuple[float, ...]],
    scalars: Iterable[float],
    objective_names: tuple[str, ...],
    out: Path | None = None,
    title: str | None = None,
) -> Path | None:
    """Scatter matrix of all objective pairs."""
    fits = np.array(list(fitnesses), dtype=float)
    sca = np.array(list(scalars), dtype=float)
    if fits.ndim != 2 or fits.shape[0] == 0:
        raise ValueError("no fitness vectors to plot")
    n = fits.shape[1]
    fig, axes = plt.subplots(n, n, figsize=(2.4 * n, 2.4 * n))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j] if n > 1 else axes
            if i == j:
                ax.hist(fits[:, i], bins=12, color="steelblue", alpha=0.85)
                ax.set_xlabel(objective_names[i], fontsize=8)
            else:
                sc = ax.scatter(fits[:, j], fits[:, i], c=sca, cmap="viridis", s=14, edgecolor="0.4", linewidth=0.3)
                if i == n - 1:
                    ax.set_xlabel(objective_names[j], fontsize=8)
                if j == 0:
                    ax.set_ylabel(objective_names[i], fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle(title or "Pareto archive: pairwise objectives")
    fig.tight_layout()
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
