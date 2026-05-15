"""Field-based melt-pool evaluation over the actual ROI grid.

Per-segment Eagar-Tsai (eagar_tsai.py) ranks (P, v, spot) but is blind to
pattern geometry — two patterns with the same parameters but different scan
order get the same fitness. This module evaluates the T_max field on a 2D
ROI grid by considering every path sample as a moving Rosenthal source and
taking the per-cell maximum.

Caveats:
- Steady-state assumption per source: each path point contributes its own
  Rosenthal field; we take max over sources, NOT a true time-dependent
  superposition. So this captures *pattern coverage* and *peak-T per cell*,
  but does NOT model accumulation across consecutive passes.
- Real time-domain superposition (with the 3D instantaneous Green's function)
  lands with the JAX/CuPy fast-sim in a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from laser_sim.config.schema import GeometryROI, MachineConfig, MaterialConfig
from laser_sim.patterns.base import RasterizedPath
from laser_sim.physics.fast_sim.eagar_tsai import rosenthal_field


@dataclass(frozen=True)
class FieldResult:
    grid_x_mm: np.ndarray         # (Nx,)
    grid_y_mm: np.ndarray         # (Ny,)
    t_max_K: np.ndarray           # (Nx, Ny)
    melt_mask: np.ndarray         # (Nx, Ny) bool, T_max >= liquidus
    keyhole_mask: np.ndarray      # (Nx, Ny) bool, T_max >= 0.9 * T_boil
    coverage_fraction: float
    keyhole_fraction: float
    t_max_mean_K: float
    t_max_std_K: float


def t_max_field(
    path: RasterizedPath,
    roi: GeometryROI,
    material: MaterialConfig,
    machine: MachineConfig,
    spot_um: float,
    nx: int = 41,
    ny: int = 41,
    stride: int = 4,
) -> FieldResult:
    """Compute the per-cell maximum temperature over the path.

    `stride` thins the path (every Nth sample contributes); the rasterizer
    typically over-samples for solver consumption, so a stride of 4 keeps the
    cost down without missing any peaks. With Np ~ 5000 and (nx, ny) = (41, 41),
    stride 4 -> 1250 path samples * 1681 grid cells = ~2M ops in vectorized numpy.
    """
    if path.n_samples() < 2:
        Xg = np.linspace(roi.x0_mm, roi.x1_mm, nx)
        Yg = np.linspace(roi.y0_mm, roi.y1_mm, ny)
        T = np.full((nx, ny), float(machine.preheat_K))
        return FieldResult(
            grid_x_mm=Xg,
            grid_y_mm=Yg,
            t_max_K=T,
            melt_mask=np.zeros_like(T, dtype=bool),
            keyhole_mask=np.zeros_like(T, dtype=bool),
            coverage_fraction=0.0,
            keyhole_fraction=0.0,
            t_max_mean_K=float(machine.preheat_K),
            t_max_std_K=0.0,
        )

    grid_x = np.linspace(roi.x0_mm, roi.x1_mm, nx)
    grid_y = np.linspace(roi.y0_mm, roi.y1_mm, ny)
    Xg, Yg = np.meshgrid(grid_x, grid_y, indexing="ij")  # (nx, ny)

    px = np.asarray(path.x_mm)
    py = np.asarray(path.y_mm)
    pP = np.asarray(path.power_W)
    pv = np.asarray(path.speed_mm_s)
    pon = np.asarray(path.laser_on, dtype=bool)

    # Heading direction at each path sample (centered finite differences)
    dx = np.gradient(px)
    dy = np.gradient(py)
    norm = np.hypot(dx, dy)
    norm = np.where(norm > 1e-12, norm, 1.0)
    hx = dx / norm
    hy = dy / norm

    keep = pon & (pP > 0)
    indices = np.where(keep)[0][::max(stride, 1)]

    T_max = np.full((nx, ny), float(machine.preheat_K))
    z = np.zeros_like(Xg)
    for k in indices:
        rx = Xg - px[k]
        ry = Yg - py[k]
        xi = rx * hx[k] + ry * hy[k]
        eta = -rx * hy[k] + ry * hx[k]
        T_k = rosenthal_field(
            float(pP[k]),
            float(pv[k]),
            material,
            machine.preheat_K,
            xi,
            eta,
            z,
            spot_um=spot_um,
        )
        np.maximum(T_max, T_k, out=T_max)

    melt_mask = T_max >= material.liquidus_K
    kh_mask = T_max >= 0.9 * material.boiling_K
    return FieldResult(
        grid_x_mm=grid_x,
        grid_y_mm=grid_y,
        t_max_K=T_max,
        melt_mask=melt_mask,
        keyhole_mask=kh_mask,
        coverage_fraction=float(melt_mask.mean()),
        keyhole_fraction=float(kh_mask.mean()),
        t_max_mean_K=float(T_max.mean()),
        t_max_std_K=float(T_max.std()),
    )
