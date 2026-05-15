"""Wraps a built laserbeamFoam case in the worker.Runner protocol.

Checks for the laserbeamFoam binary on PATH; if absent, returns a
descriptive failure rather than crashing. Real HF execution proceeds
via Allrun -> decomposePar -> mpirun -> reconstructPar, sandboxed in
a tmp dir.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from laser_sim.physics.melt_pool.openfoam_case import OpenFoamCase
from laser_sim.worker.base import CaseSpec, EvaluationResult, Runner
from laser_sim.worker.local import LocalRunner


def laserbeamfoam_available() -> bool:
    if shutil.which("laserbeamFoam") is not None:
        return True
    # also accept being inside an OpenFOAM env without the binary on PATH
    return bool(os.environ.get("WM_PROJECT_DIR"))


class LaserbeamFoamRunner:
    """Convenience wrapper that turns OpenFoamCase + Runner into a single call."""

    def __init__(self, runner: Runner | None = None, timeout_s: float = 3600.0) -> None:
        self.runner = runner or LocalRunner()
        self.timeout_s = timeout_s

    async def run(self, case: OpenFoamCase) -> EvaluationResult:
        if not laserbeamfoam_available():
            return EvaluationResult(
                success=False,
                walltime_s=0.0,
                stdout="",
                stderr=(
                    "laserbeamFoam binary not found on PATH and WM_PROJECT_DIR is unset. "
                    "Build the FoamLab/laserbeamFoam solver and source the OpenFOAM env, "
                    "or use a different runner."
                ),
                exit_code=127,
            )
        spec = CaseSpec(
            case_dir=Path(case.case_dir),
            solver_name="./Allrun",
            args=(),
            timeout_s=self.timeout_s,
        )
        return await self.runner.submit(spec)
