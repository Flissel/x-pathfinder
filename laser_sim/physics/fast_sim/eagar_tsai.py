"""Eagar-Tsai / Rosenthal analytical proxy for the melt pool.

Stand-in for the JAX/CuPy 2.5D enthalpy solver until that phase lands. Gives
per-(P, v) melt-pool width, length, depth, peak temperature from the
moving-point-source solution on a semi-infinite plate. Cheap, vectorizable,
no GPU, no compile.

Reference: T. Eagar & N. Tsai, "Temperature fields produced by traveling
distributed heat sources", Weld. J. 62, 1983; D. Rosenthal, ASME 1946.

The solution superposes the moving point source on temperature rise above
preheat. For an absorptivity-corrected source eta*P on material with
thermal conductivity k, diffusivity alpha, scanning at speed v in +x:

    T - T0 = (eta P) / (2 pi k r) * exp(-v (r + xi) / (2 alpha))

where xi = x - v t (moving frame), r = sqrt(xi^2 + y^2 + z^2). The melt
pool boundary is the T = T_liquidus isotherm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from laser_sim.config.schema import MaterialConfig


@dataclass(frozen=True)
class MeltPoolEstimate:
    """Per-segment melt pool prediction (Eagar-Tsai)."""

    width_mm: float
    length_mm: float
    depth_mm: float
    peak_T_K: float
    aspect_ratio: float
    energy_density_J_mm3: float


def _alpha_solid(mat: MaterialConfig) -> float:
    """Thermal diffusivity at room temperature, m^2/s."""
    return mat.k_solid / (mat.rho_solid * mat.cp_solid)


def rosenthal_field(
    power_W: float,
    speed_mm_s: float,
    mat: MaterialConfig,
    preheat_K: float,
    xi_mm: np.ndarray,
    y_mm: np.ndarray,
    z_mm: np.ndarray,
    spot_um: float = 80.0,
) -> np.ndarray:
    """Steady-state temperature field (K) in the moving frame for a point source.

    All arrays are broadcast against each other; output has the broadcast shape.
    Returns absolute temperature; the point-source singularity at r=0 is
    regularized with the laser spot radius (Gaussian-source convolution
    approximation), so peak T scales physically with P / spot.
    """
    v_m_s = speed_mm_s * 1e-3
    xi_m = xi_mm * 1e-3
    y_m = y_mm * 1e-3
    z_m = z_mm * 1e-3
    r = np.sqrt(xi_m**2 + y_m**2 + z_m**2)
    # regularize with the 1/e beam radius (spot diameter -> half + sqrt2 factor)
    r_min = max(spot_um * 1e-6 * 0.5, 5e-6)
    r_safe = np.sqrt(r**2 + r_min**2)
    alpha = _alpha_solid(mat)
    eta = mat.absorptivity
    k = mat.k_solid
    exponent = -v_m_s * (r_safe + xi_m) / (2.0 * alpha)
    exponent = np.clip(exponent, -700.0, 0.0)
    dT = (eta * power_W) / (2.0 * np.pi * k * r_safe) * np.exp(exponent)
    return preheat_K + dT


def energy_density(
    power_W: float, speed_mm_s: float, hatch_mm: float, layer_mm: float
) -> float:
    """Volumetric energy density J/mm^3.

    E = P / (v * h * t) with v in mm/s, h hatch in mm, t layer thickness in mm.
    """
    denom = max(speed_mm_s * hatch_mm * layer_mm, 1e-12)
    return power_W / denom


def eagar_tsai_melt_pool(
    power_W: float,
    speed_mm_s: float,
    mat: MaterialConfig,
    preheat_K: float,
    hatch_mm: float,
    layer_mm: float,
    spot_um: float = 80.0,
    grid_n: int = 41,
    grid_extent_mm: float | None = None,
) -> MeltPoolEstimate:
    """Sample the Rosenthal field on a small grid and extract melt-pool features.

    grid_extent_mm spans the moving frame in the +-x / +-y / 0..depth directions.
    grid_n controls resolution per axis (default 41).

    If grid_extent_mm is None (default) the extent is auto-scaled to the
    characteristic thermal length alpha/v so the melt pool resolves cleanly
    at any (P, v): high-speed scans (narrow pool) get a tight grid, slow
    scans (wide pool) get a coarse grid.
    """
    if grid_extent_mm is None:
        alpha = _alpha_solid(mat)  # m^2/s
        v_m_s = max(speed_mm_s * 1e-3, 1e-6)
        # 100 thermal-diffusion lengths gives ~3 melt-pool widths of headroom
        L_char_m = 100.0 * alpha / v_m_s
        grid_extent_mm = float(np.clip(L_char_m * 1e3, 0.15, 2.0))
    half = grid_extent_mm
    xi = np.linspace(-half, half, grid_n)
    y = np.linspace(-half, half, grid_n)
    z = np.linspace(0.0, half, grid_n)
    XI, Y, Z = np.meshgrid(xi, y, z, indexing="ij")
    T = rosenthal_field(power_W, speed_mm_s, mat, preheat_K, XI, Y, Z, spot_um=spot_um)

    T_liq = mat.liquidus_K
    melt = T >= T_liq
    if not melt.any():
        return MeltPoolEstimate(
            width_mm=0.0,
            length_mm=0.0,
            depth_mm=0.0,
            peak_T_K=float(T.max()),
            aspect_ratio=0.0,
            energy_density_J_mm3=energy_density(power_W, speed_mm_s, hatch_mm, layer_mm),
        )

    # surface-slice (z = 0 -> first z index) footprint:
    surf = melt[:, :, 0]
    if surf.any():
        ii, jj = np.where(surf)
        length_mm = float(xi[ii.max()] - xi[ii.min()])
        width_mm = float(2.0 * max(abs(y[jj].min()), abs(y[jj].max())))
    else:
        length_mm = 0.0
        width_mm = 0.0

    # depth: deepest z index that contains any molten cell
    kk = np.where(melt.any(axis=(0, 1)))[0]
    depth_mm = float(z[kk.max()]) if kk.size else 0.0
    peak_T_K = float(T.max())
    ar = length_mm / width_mm if width_mm > 0 else 0.0
    return MeltPoolEstimate(
        width_mm=width_mm,
        length_mm=length_mm,
        depth_mm=depth_mm,
        peak_T_K=peak_T_K,
        aspect_ratio=ar,
        energy_density_J_mm3=energy_density(power_W, speed_mm_s, hatch_mm, layer_mm),
    )
