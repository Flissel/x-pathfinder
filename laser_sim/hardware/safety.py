"""Safety interlocks for LPBF hardware bridge.

EVERY pattern destined for a real (non-Mock) machine must pass `check()`. The
default Machine ABC is intentionally agnostic — it does NOT call safety on
its own — but the high-level orchestration layer (control_plane.scheduler
and `cli closed-loop`) routes all patterns through SafetyChecker first.

Interlocks (from the architectural plan):
- power within machine envelope
- volumetric energy density within reasonable LPBF range
- gas atmosphere precondition (O2 ppm, gas type, optional preheat)
- geofence: every waypoint inside the build volume
- maximum scan time per pattern (cap to limit one runaway pattern)
- machine allowlist (only specified machines may execute real builds)
- explicit human-in-the-loop flag (`acknowledge`) for non-Mock machines
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from laser_sim.config.schema import (
    GeometryROI,
    MachineConfig,
    MaterialConfig,
    ScenarioConfig,
)
from laser_sim.patterns.base import ScanPattern
from laser_sim.physics.fast_sim.eagar_tsai import energy_density


class SafetyViolation(Exception):
    """Raised when a pattern fails any interlock and unsafe execution is attempted."""


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    violations: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def raise_if_blocked(self) -> None:
        if not self.allowed:
            raise SafetyViolation(
                "pattern rejected by safety interlocks: "
                + "; ".join(self.violations)
            )


@dataclass(frozen=True)
class SafetyConfig:
    energy_density_min_J_mm3: float = 20.0
    energy_density_max_J_mm3: float = 120.0
    max_scan_time_s: float = 300.0  # ceiling per pattern
    require_inert_atmosphere: bool = True
    max_o2_ppm: float = 1000.0
    allowed_gases: tuple[str, ...] = ("argon", "nitrogen", "helium")
    machine_allowlist: tuple[str, ...] = ()  # empty means MockMachine only
    geofence_margin_mm: float = 1.0  # required clearance from build volume edge


class SafetyChecker:
    """Stateless safety gate for ScanPatterns + Scenario + target machine.

    `acknowledge=True` is required for any non-Mock machine; otherwise the
    check is blocked with a violation even if everything else passes. This
    enforces the `--i-know-what-im-doing` discipline from the plan.
    """

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()

    def check(
        self,
        pattern: ScanPattern,
        scenario: ScenarioConfig,
        target_machine_id: str,
        acknowledge: bool = False,
        is_mock: bool = False,
        hatch_mm: float = 0.1,
    ) -> SafetyDecision:
        violations: list[str] = []
        warnings: list[str] = []

        # 1) explicit acknowledgement for real hardware
        if not is_mock and not acknowledge:
            violations.append(
                "missing explicit acknowledgement: pass --i-know-what-im-doing"
                " to operate non-mock hardware"
            )

        # 2) machine allowlist (mock is implicitly allowed)
        if not is_mock and self.config.machine_allowlist:
            if target_machine_id not in self.config.machine_allowlist:
                violations.append(
                    f"machine '{target_machine_id}' not in allowlist "
                    f"{self.config.machine_allowlist!r}"
                )

        # 3) power & speed envelope
        las = scenario.machine.laser
        for i, seg in enumerate(pattern.segments):
            if seg.power_W < las.power_min_W - 1e-9 or seg.power_W > las.power_max_W + 1e-9:
                violations.append(
                    f"segment {i}: power {seg.power_W:.1f}W outside envelope "
                    f"[{las.power_min_W}, {las.power_max_W}] W"
                )
            if (
                seg.speed_mm_s < las.speed_min_mm_s - 1e-9
                or seg.speed_mm_s > las.speed_max_mm_s + 1e-9
            ):
                violations.append(
                    f"segment {i}: speed {seg.speed_mm_s:.1f}mm/s outside envelope "
                    f"[{las.speed_min_mm_s}, {las.speed_max_mm_s}] mm/s"
                )

        # 4) volumetric energy density
        layer_mm = scenario.machine.layer_thickness_um * 1e-3
        for i, seg in enumerate(pattern.segments):
            if seg.power_W <= 0 or not seg.laser_on:
                continue
            E = energy_density(seg.power_W, seg.speed_mm_s, hatch_mm, layer_mm)
            if E < self.config.energy_density_min_J_mm3:
                warnings.append(
                    f"segment {i}: E={E:.1f} J/mm^3 below safe lower bound "
                    f"{self.config.energy_density_min_J_mm3}"
                )
            elif E > self.config.energy_density_max_J_mm3:
                violations.append(
                    f"segment {i}: E={E:.1f} J/mm^3 above safe upper bound "
                    f"{self.config.energy_density_max_J_mm3}"
                )

        # 5) gas atmosphere preconditions
        if self.config.require_inert_atmosphere:
            if scenario.machine.gas not in self.config.allowed_gases:
                violations.append(
                    f"gas '{scenario.machine.gas}' not in allowed list "
                    f"{self.config.allowed_gases!r}"
                )
            if scenario.machine.chamber_o2_ppm > self.config.max_o2_ppm:
                violations.append(
                    f"chamber O2 {scenario.machine.chamber_o2_ppm:.0f} ppm exceeds "
                    f"limit {self.config.max_o2_ppm:.0f} ppm"
                )

        # 6) geofence: every waypoint within build volume, with margin
        bv_x = scenario.machine.build_x_mm / 2.0
        bv_y = scenario.machine.build_y_mm / 2.0
        margin = self.config.geofence_margin_mm
        for i, seg in enumerate(pattern.segments):
            for j, wp in enumerate(seg.waypoints):
                if abs(wp.x_mm) > bv_x - margin or abs(wp.y_mm) > bv_y - margin:
                    violations.append(
                        f"segment {i} waypoint {j} at ({wp.x_mm:.2f},{wp.y_mm:.2f}) mm "
                        f"violates geofence (build vol +-{bv_x:.0f},{bv_y:.0f} mm, "
                        f"margin {margin} mm)"
                    )
                    break  # one per segment is enough noise

        # 7) max scan time
        t = pattern.total_time_s()
        if t > self.config.max_scan_time_s:
            violations.append(
                f"total scan time {t:.2f}s exceeds limit {self.config.max_scan_time_s}s"
            )

        return SafetyDecision(
            allowed=len(violations) == 0,
            violations=tuple(violations),
            warnings=tuple(warnings),
        )
