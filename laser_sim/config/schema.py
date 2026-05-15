"""Versioned scenario schema for LPBF optimization campaigns.

The scenario is the immutable problem statement: material, powder, laser envelope,
machine atmosphere, geometry/ROI, and objective version. Every run is bound to a
scenario_id so results stay reproducible across solver/code revisions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MaterialConfig(_Frozen):
    material_id: str
    name: str
    rho_solid: float = Field(gt=0, description="kg/m^3 at room T")
    rho_liquid: float = Field(gt=0, description="kg/m^3 at liquidus")
    cp_solid: float = Field(gt=0, description="J/(kg·K)")
    cp_liquid: float = Field(gt=0, description="J/(kg·K)")
    k_solid: float = Field(gt=0, description="W/(m·K)")
    k_liquid: float = Field(gt=0, description="W/(m·K)")
    solidus_K: float = Field(gt=0)
    liquidus_K: float = Field(gt=0)
    latent_heat_fusion: float = Field(gt=0, description="J/kg")
    latent_heat_vap: float = Field(gt=0, description="J/kg")
    boiling_K: float = Field(gt=0)
    emissivity: float = Field(ge=0, le=1)
    absorptivity: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _check_phase_order(self) -> "MaterialConfig":
        if not (self.solidus_K < self.liquidus_K < self.boiling_K):
            raise ValueError("require solidus < liquidus < boiling")
        return self


class PowderProfile(_Frozen):
    """Particle size distribution + bulk characterization for DEM."""

    powder_id: str
    d10_um: float = Field(gt=0)
    d50_um: float = Field(gt=0)
    d90_um: float = Field(gt=0)
    sphericity: float = Field(ge=0, le=1)
    bulk_density_kg_m3: float = Field(gt=0)
    moisture_pct: float = Field(ge=0, le=100, default=0.0)
    oxidation_pct: float = Field(ge=0, le=100, default=0.0)
    coh_energy_density: float = Field(ge=0, default=0.0, description="J/m^3, DEM cohesion")

    @model_validator(mode="after")
    def _check_psd(self) -> "PowderProfile":
        if not (self.d10_um <= self.d50_um <= self.d90_um):
            raise ValueError("require d10 <= d50 <= d90")
        return self


class LaserEnvelope(_Frozen):
    power_min_W: float = Field(ge=0)
    power_max_W: float = Field(gt=0)
    speed_min_mm_s: float = Field(ge=0)
    speed_max_mm_s: float = Field(gt=0)
    spot_min_um: float = Field(gt=0)
    spot_max_um: float = Field(gt=0)
    beam_profile: Literal["gaussian", "top_hat", "donut"] = "gaussian"
    wavelength_nm: float = Field(gt=0, default=1064.0)

    @model_validator(mode="after")
    def _check_ranges(self) -> "LaserEnvelope":
        if self.power_min_W > self.power_max_W:
            raise ValueError("power_min > power_max")
        if self.speed_min_mm_s > self.speed_max_mm_s:
            raise ValueError("speed_min > speed_max")
        if self.spot_min_um > self.spot_max_um:
            raise ValueError("spot_min > spot_max")
        return self


class MachineConfig(_Frozen):
    machine_id: str
    vendor: Literal["slm", "eos", "trumpf", "renishaw", "generic"] = "generic"
    build_x_mm: float = Field(gt=0)
    build_y_mm: float = Field(gt=0)
    build_z_mm: float = Field(gt=0)
    laser: LaserEnvelope
    gas: Literal["argon", "nitrogen", "helium", "vacuum"] = "argon"
    gas_flow_m_s: float = Field(ge=0, default=2.0)
    chamber_o2_ppm: float = Field(ge=0, default=100.0)
    preheat_K: float = Field(ge=0, default=298.0)
    layer_thickness_um: float = Field(gt=0, default=30.0)


class GeometryROI(_Frozen):
    """Region of interest within the build volume.

    Coordinates follow ISO/ASTM 52921 build-volume convention (NIST AM-Bench).
    Origin at substrate centre, +Z up.
    """

    roi_id: str
    x0_mm: float
    y0_mm: float
    x1_mm: float
    y1_mm: float
    z_mm: float = 0.0
    layers: int = Field(ge=1, default=1)

    @model_validator(mode="after")
    def _check_bbox(self) -> "GeometryROI":
        if self.x1_mm <= self.x0_mm or self.y1_mm <= self.y0_mm:
            raise ValueError("ROI must have positive extent")
        return self

    @property
    def width_mm(self) -> float:
        return self.x1_mm - self.x0_mm

    @property
    def height_mm(self) -> float:
        return self.y1_mm - self.y0_mm


class EAConfig(_Frozen):
    population: int = Field(ge=2, default=64)
    generations: int = Field(ge=1, default=20)
    elitism: int = Field(ge=0, default=4)
    tournament_k: int = Field(ge=2, default=3)
    crossover_rate: float = Field(ge=0, le=1, default=0.7)
    mutation_rate: float = Field(ge=0, le=1, default=0.3)
    crisis_multiplier: float = Field(ge=1, default=3.0)
    crisis_after_stagnation: int = Field(ge=1, default=5)
    seed: int = Field(default=42)


class FidelityConfig(_Frozen):
    """Multi-fidelity gating: who runs which tier."""

    fast_only_until_gen: int = Field(ge=0, default=3, description="warm-up; no HF before this gen")
    hf_promote_top_k: int = Field(ge=0, default=5, description="per generation")
    hf_promote_pareto_front: bool = True
    surrogate_min_runs: int = Field(ge=0, default=200, description="train surrogate after N runs")
    veto_threshold_sigma: float = Field(
        ge=0, default=2.0, description="HF veto if fast-vs-HF diff > N sigma"
    )


class ScenarioConfig(BaseModel):
    """Versioned, immutable problem statement.

    All evaluations and measurements are bound to scenario_id; changing material,
    powder, or machine forces a new scenario.
    """

    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(default_factory=lambda: str(uuid4()))
    objective_version: str = Field(default="v1", description="bump when fitness metrics change")
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))

    material: MaterialConfig
    powder: PowderProfile
    machine: MachineConfig
    roi: GeometryROI
    ea: EAConfig = Field(default_factory=EAConfig)
    fidelity: FidelityConfig = Field(default_factory=FidelityConfig)
    notes: str = ""
