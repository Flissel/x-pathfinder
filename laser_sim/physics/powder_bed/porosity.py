"""Load packed powder bed and compute effective thermal/density properties.

The voxel field from liggghts_packer is binary (1 = solid particle,
0 = void). Effective properties are simple Maxwell-Garnett-style mixing
for the powder layer; the fully-dense substrate below the layer uses
solid properties unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from laser_sim.config.schema import MaterialConfig


@dataclass(frozen=True)
class PowderBedField:
    voxel: np.ndarray            # (Nx, Ny, Nz) bool
    bed_thickness_um: float
    fill_fraction: float
    roi_mm: tuple[float, float, float, float]


@dataclass(frozen=True)
class EffectiveProperties:
    rho: float
    cp: float
    k: float
    fill_fraction: float


def load_porosity(path: Path) -> PowderBedField:
    data = np.load(Path(path))
    voxel = np.asarray(data["voxel"], dtype=bool)
    roi = tuple(float(x) for x in np.asarray(data["roi"]).tolist())
    return PowderBedField(
        voxel=voxel,
        bed_thickness_um=float(data["bed_thickness_um"]),
        fill_fraction=float(data["fill_fraction"]),
        roi_mm=roi,
    )


def effective_thermal_properties(
    field: PowderBedField, material: MaterialConfig, gas_k: float = 0.026
) -> EffectiveProperties:
    """Maxwell mixing for a two-phase metal+gas powder.

    k_eff = k_gas + (k_solid - k_gas) * 3 * phi / (k_solid/k_gas + 2 - phi)
    where phi is the solid volume fraction.

    rho_eff = phi * rho_solid + (1 - phi) * rho_gas (gas neglected, ~1 kg/m^3)
    cp_eff  = mass-weighted average
    """
    phi = float(field.fill_fraction)
    rho_solid = material.rho_solid
    cp_solid = material.cp_solid
    k_solid = material.k_solid
    rho_eff = phi * rho_solid
    cp_eff = cp_solid  # gas contribution to heat capacity per unit volume is negligible
    ratio = k_solid / gas_k
    k_eff = gas_k * (1 + 3 * phi * (ratio - 1) / max(ratio + 2 - phi * (ratio - 1), 1e-6))
    return EffectiveProperties(
        rho=float(rho_eff),
        cp=float(cp_eff),
        k=float(k_eff),
        fill_fraction=phi,
    )
