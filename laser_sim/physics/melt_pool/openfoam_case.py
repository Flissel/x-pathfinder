"""OpenFOAM case-folder generator for laserbeamFoam HF runs.

Emits a self-contained case directory:

  case/
    constant/
      transportProperties        (rho, cp, k, latentHeat, etc.)
      thermoPath                 (T-dependent table; optional)
      laserPath                  (x, y, z, t, P, v) trajectory
    system/
      controlDict
      fvSchemes
      fvSolution
      blockMeshDict
    0/
      T, U, p_rgh, alpha.metal   (initial conditions)
    Allrun                       (executable shell wrapper)

Templates live in templates/ as Jinja2 sources. If Jinja2 is not
installed, falls back to str.format substitution for the small subset
we need; this lets the unit tests run without the optional `hf` extra.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from laser_sim.config.schema import ScenarioConfig
from laser_sim.patterns.base import RasterizedPath, ScanPattern
from laser_sim.patterns.rasterize import rasterize_pattern

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _render(template_name: str, ctx: dict[str, Any]) -> str:
    path = _TEMPLATES_DIR / template_name
    text = path.read_text()
    try:
        from jinja2 import Environment

        env = Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)
        return env.from_string(text).render(**ctx)
    except ImportError:
        # very limited fallback: just str.format with {{ name }} -> {name}
        text2 = text.replace("{{ ", "{").replace(" }}", "}").replace("{{", "{").replace("}}", "}")
        return text2.format(**ctx)


@dataclass(frozen=True)
class OpenFoamCase:
    case_dir: Path
    pattern_path_file: Path
    n_path_samples: int
    end_time_s: float
    solver: str = "laserbeamFoam"
    files_written: tuple[Path, ...] = field(default_factory=tuple)


def _write_laser_path(rp: RasterizedPath, path_file: Path) -> None:
    """Write (t, x, y, P, v) trajectory in a simple OF-readable table."""
    path_file.parent.mkdir(parents=True, exist_ok=True)
    with open(path_file, "w") as f:
        f.write("// laser_sim trajectory: t_s x_mm y_mm power_W speed_mm_s\n")
        f.write(f"// n_samples = {rp.n_samples()}\n")
        for i in range(rp.n_samples()):
            f.write(
                f"{rp.t_s[i]:.6e} {rp.x_mm[i]*1e-3:.6e} {rp.y_mm[i]*1e-3:.6e} "
                f"{rp.power_W[i]:.4e} {rp.speed_mm_s[i]:.4e}\n"
            )


def build_laserbeamfoam_case(
    pattern: ScanPattern,
    scenario: ScenarioConfig,
    case_dir: Path,
    *,
    spot_um: float = 80.0,
    cell_size_um: float = 25.0,
    end_time_pad_s: float = 0.001,
    write_interval_s: float = 0.001,
) -> OpenFoamCase:
    """Materialise a laserbeamFoam case directory from a ScanPattern + scenario.

    The case is structurally correct (controlDict, fvSchemes, blockMeshDict,
    transportProperties, 0/ fields, Allrun). It can be executed by the
    laserbeamFoam binary if one is installed; otherwise the runner short-
    circuits with a clear error.
    """
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("0", "constant", "system"):
        (case_dir / sub).mkdir(exist_ok=True)

    rp = rasterize_pattern(pattern, ds_mm=0.05)
    pattern_path_file = case_dir / "constant" / "laserPath"
    _write_laser_path(rp, pattern_path_file)
    end_time = float(pattern.total_time_s() + end_time_pad_s)

    mat = scenario.material
    roi = scenario.roi
    layer_m = scenario.machine.layer_thickness_um * 1e-3
    bbox_pad_mm = 1.0
    bbox_x_min = (roi.x0_mm - bbox_pad_mm) * 1e-3
    bbox_x_max = (roi.x1_mm + bbox_pad_mm) * 1e-3
    bbox_y_min = (roi.y0_mm - bbox_pad_mm) * 1e-3
    bbox_y_max = (roi.y1_mm + bbox_pad_mm) * 1e-3
    bbox_z_min = -2.0e-3
    bbox_z_max = 2.0 * layer_m
    cell_m = cell_size_um * 1e-6
    nx = max(int(round((bbox_x_max - bbox_x_min) / cell_m)), 4)
    ny = max(int(round((bbox_y_max - bbox_y_min) / cell_m)), 4)
    nz = max(int(round((bbox_z_max - bbox_z_min) / cell_m)), 4)

    ctx_common = {
        "end_time": end_time,
        "write_interval": write_interval_s,
        "preheat_K": scenario.machine.preheat_K,
        "absorptivity": mat.absorptivity,
        "rho_solid": mat.rho_solid,
        "rho_liquid": mat.rho_liquid,
        "cp_solid": mat.cp_solid,
        "cp_liquid": mat.cp_liquid,
        "k_solid": mat.k_solid,
        "k_liquid": mat.k_liquid,
        "solidus_K": mat.solidus_K,
        "liquidus_K": mat.liquidus_K,
        "latent_heat_fusion": mat.latent_heat_fusion,
        "boiling_K": mat.boiling_K,
        "emissivity": mat.emissivity,
        "spot_radius_m": spot_um * 1e-6 * 0.5,
        "bbox_x_min": bbox_x_min,
        "bbox_x_max": bbox_x_max,
        "bbox_y_min": bbox_y_min,
        "bbox_y_max": bbox_y_max,
        "bbox_z_min": bbox_z_min,
        "bbox_z_max": bbox_z_max,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "layer_m": layer_m,
        "scenario_id": str(scenario.scenario_id),
    }

    files = [
        ("system/controlDict", "controlDict.j2"),
        ("system/fvSchemes", "fvSchemes.j2"),
        ("system/fvSolution", "fvSolution.j2"),
        ("system/blockMeshDict", "blockMeshDict.j2"),
        ("constant/transportProperties", "transportProperties.j2"),
        ("0/T", "T_init.j2"),
        ("0/U", "U_init.j2"),
        ("0/p_rgh", "p_rgh_init.j2"),
        ("0/alpha.metal", "alpha_metal_init.j2"),
        ("Allrun", "Allrun.j2"),
    ]
    written: list[Path] = []
    for rel, tpl in files:
        out = case_dir / rel
        out.write_text(_render(tpl, ctx_common))
        written.append(out)
    (case_dir / "Allrun").chmod(0o755)
    return OpenFoamCase(
        case_dir=case_dir,
        pattern_path_file=pattern_path_file,
        n_path_samples=rp.n_samples(),
        end_time_s=end_time,
        files_written=tuple(written),
    )
