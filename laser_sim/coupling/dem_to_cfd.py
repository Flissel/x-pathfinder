"""One-way DEM -> OpenFOAM coupling.

Writes a system/setFieldsDict that initialises alpha.metal from the
packed powder voxel field. The case's Allrun (Allrun.j2) is expected
to invoke `setFields` after blockMesh (commented in the template — flip
once a packed bed is in hand).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from laser_sim.physics.powder_bed.porosity import PowderBedField


def write_setfields_dict(bed: PowderBedField, case_dir: Path) -> Path:
    """Emit a coarse box-based setFieldsDict approximating the bed.

    Real DEM -> CFD interpolation would mesh the spheres onto the OF grid
    via cellSet of every cell whose centre lies inside a sphere. That's a
    laser_sim/coupling stretch goal (preCICE adapter). For now the
    setFieldsDict prescribes a uniform layer of alpha.metal = fill_fraction
    up to bed_thickness — good enough for level-1 HF verification.
    """
    case_dir = Path(case_dir)
    sys_dir = case_dir / "system"
    sys_dir.mkdir(parents=True, exist_ok=True)
    out = sys_dir / "setFieldsDict"
    fill = bed.fill_fraction
    bed_m = bed.bed_thickness_um * 1e-6
    out.write_text(
        f"""FoamFile {{ version 2.0; format ascii; class dictionary; object setFieldsDict; }}

defaultFieldValues ( volScalarFieldValue alpha.metal 0 );

regions
(
    boxToCell
    {{
        box (-1 -1 0) (1 1 {bed_m});
        fieldValues ( volScalarFieldValue alpha.metal {fill:.4f} );
    }}
);
"""
    )
    return out
