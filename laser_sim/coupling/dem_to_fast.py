"""One-way DEM -> fast-sim coupling.

Takes a packed powder bed and replaces solid-material thermal properties
with the Maxwell-mixed effective values in a MaterialConfig copy. The
EA loop can then evaluate fitness on the "powder-aware" material without
modifying the original scenario.
"""

from __future__ import annotations

from laser_sim.config.schema import MaterialConfig
from laser_sim.physics.powder_bed.porosity import (
    PowderBedField,
    effective_thermal_properties,
)


def apply_powder_to_material(
    material: MaterialConfig, bed: PowderBedField
) -> MaterialConfig:
    eff = effective_thermal_properties(bed, material)
    return material.model_copy(
        update={
            "rho_solid": eff.rho,
            "k_solid": eff.k,
            "cp_solid": eff.cp,
        }
    )
