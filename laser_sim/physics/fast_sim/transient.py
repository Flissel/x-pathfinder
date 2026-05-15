"""Field-based melt-pool evaluation over the actual ROI grid.

Two evaluators:

1) t_max_field         — fast per-source-max approximation. Each path
   sample contributes its own steady-state Rosenthal field; per cell we
   take the max. Captures pattern coverage and peak-T per (P, v, spot)
   but is BLIND to heat accumulation across passes.

2) t_max_field_superposition — physically meaningful upgrade. Treats
   every path sample as an instantaneous 3D heat source pulse of energy
   eta * P_k * dt_k. Temperature at (x, y) and time t is the linear
   superposition of past pulse contributions via the 3D Green's
   function for the heat equation:
       G(r, tau) = 1 / (4 pi alpha tau)^(3/2) * exp(-r^2 / (4 alpha tau))
   T_max per cell is then the max over a coarse time grid. This makes
   close hatches and consecutive tracks visibly hotter, so the EA
   sees genuine accumulation effects when choosing scan geometry.

Both are CPU + numpy only; no JAX. The JAX/CuPy 2.5D enthalpy solver
will swap in behind the same interface in a later phase.
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


def t_max_field_superposition(
    path: RasterizedPath,
    roi: GeometryROI,
    material: MaterialConfig,
    machine: MachineConfig,
    spot_um: float,
    nx: int = 41,
    ny: int = 41,
    stride: int = 6,
    n_time_checkpoints: int = 8,
) -> FieldResult:
    """Compute T_max via time-domain superposition of 3D Green's function.

    For each path sample k we model an instantaneous heat-source pulse
    of energy E_k = eta * P_k * dt_k deposited at (x_k, y_k, 0) at time
    t_k. The temperature at any later time t > t_k and any field point
    (x, y, 0) is:

        delta_T_k(x, y, t) = (E_k / (rho * c_p)) * G(r_k, t - t_k)

    where G is the 3D instantaneous heat-source Green's function. We
    evaluate T(x, y, t_m) at `n_time_checkpoints` linearly-spaced t_m
    over the scan duration plus a short tail, sum contributions from all
    k with t_k < t_m, and take the per-cell max across t_m.

    Computational cost: O(n_time_checkpoints * nx*ny * n_path/stride).
    Defaults (8 * 41^2 * ~1000) ≈ 14M vectorized ops, runs in ~1-2s.
    The singularity at tau→0 is regularized by adding (r_spot)^2 to r^2
    in the kernel exponent and floor on tau.
    """
    grid_x = np.linspace(roi.x0_mm, roi.x1_mm, nx)
    grid_y = np.linspace(roi.y0_mm, roi.y1_mm, ny)
    Xg, Yg = np.meshgrid(grid_x, grid_y, indexing="ij")
    preheat = float(machine.preheat_K)

    if path.n_samples() < 2:
        T = np.full((nx, ny), preheat)
        return FieldResult(
            grid_x_mm=grid_x,
            grid_y_mm=grid_y,
            t_max_K=T,
            melt_mask=np.zeros_like(T, dtype=bool),
            keyhole_mask=np.zeros_like(T, dtype=bool),
            coverage_fraction=0.0,
            keyhole_fraction=0.0,
            t_max_mean_K=preheat,
            t_max_std_K=0.0,
        )

    keep = path.laser_on & (path.power_W > 0)
    px = np.asarray(path.x_mm)[keep][::max(stride, 1)]
    py = np.asarray(path.y_mm)[keep][::max(stride, 1)]
    pP = np.asarray(path.power_W)[keep][::max(stride, 1)]
    pt = np.asarray(path.t_s)[keep][::max(stride, 1)]
    if pt.size < 2:
        T = np.full((nx, ny), preheat)
        return FieldResult(
            grid_x_mm=grid_x,
            grid_y_mm=grid_y,
            t_max_K=T,
            melt_mask=np.zeros_like(T, dtype=bool),
            keyhole_mask=np.zeros_like(T, dtype=bool),
            coverage_fraction=0.0,
            keyhole_fraction=0.0,
            t_max_mean_K=preheat,
            t_max_std_K=0.0,
        )

    # dwell time per (subsampled) sample (energy = P * dt)
    dt = np.diff(pt, prepend=pt[0])
    pos_dt = dt[dt > 0]
    fill = float(np.median(pos_dt)) if pos_dt.size else 1e-5
    dt = np.where(dt > 0, dt, fill)

    eta = material.absorptivity
    rho = material.rho_solid
    cp = material.cp_solid
    alpha = material.k_solid / (rho * cp)
    r_min = max(spot_um * 1e-6 * 0.5, 5e-6)
    r_min2 = r_min ** 2

    # precompute (Nx, Ny, Np) distance² in m²; chunk over time checkpoints
    dx2 = (Xg[:, :, None] - px[None, None, :]) ** 2 * 1e-6
    dy2 = (Yg[:, :, None] - py[None, None, :]) ** 2 * 1e-6
    r2 = dx2 + dy2 + r_min2  # (nx, ny, Np)

    # weighting (energy / (rho cp)) for each path sample → units of K * m³
    weights = (eta * pP * dt) / (rho * cp)  # (Np,)

    # time grid for the max
    t0, t1 = float(pt[0]), float(pt[-1])
    # extend slightly past end so post-scan diffusion is captured
    tail = max((t1 - t0) * 0.2, 1e-3)
    t_eval = np.linspace(t0 + 1e-5, t1 + tail, max(n_time_checkpoints, 2))

    T_max = np.full((nx, ny), preheat)
    for tj in t_eval:
        tau = tj - pt  # (Np,)
        valid = tau > 1e-9
        tau_safe = np.where(valid, tau, 1.0)
        # 3D Green's function 1/(4 pi alpha tau)^(3/2) * exp(-r² / (4 alpha tau))
        denom = (4.0 * np.pi * alpha * tau_safe) ** 1.5  # (Np,)
        # kernel shape (Nx, Ny, Np)
        kernel = np.where(
            valid,
            np.exp(-r2 / (4.0 * alpha * tau_safe)) / denom,
            0.0,
        )
        T_tj = preheat + (weights * kernel).sum(axis=2)
        np.maximum(T_max, T_tj, out=T_max)

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
