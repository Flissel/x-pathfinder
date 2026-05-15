"""LocalRunner: spawn the solver as a subprocess on this host.

Async wrapper around asyncio.create_subprocess_exec. Suitable for
single-machine fast-sim batches and laserbeamFoam runs on a fat
workstation. For real HPC HF runs prefer SlurmRunner / K8sRunner.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from pathlib import Path
from typing import Any

from laser_sim.worker.base import CaseSpec, EvaluationResult, Runner


class LocalRunner(Runner):
    name = "local"

    def __init__(self, max_concurrent: int = 1) -> None:
        self._sema = asyncio.Semaphore(max_concurrent)

    async def submit(self, case: CaseSpec) -> EvaluationResult:
        async with self._sema:
            return await self._run_once(case)

    async def _run_once(self, case: CaseSpec) -> EvaluationResult:
        case_dir = Path(case.case_dir)
        if not case_dir.is_dir():
            return EvaluationResult(
                success=False,
                walltime_s=0.0,
                stdout="",
                stderr=f"case_dir {case_dir} not found",
                exit_code=2,
            )
        env = {**os.environ, **case.env}
        cmd = [case.solver_name, *case.args]
        # check if executable is reachable; if not, fail fast
        if shutil.which(case.solver_name) is None:
            return EvaluationResult(
                success=False,
                walltime_s=0.0,
                stdout="",
                stderr=(
                    f"solver '{case.solver_name}' not found on PATH; "
                    "install the relevant extra or use a different runner"
                ),
                exit_code=127,
            )
        t0 = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(case_dir),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=case.timeout_s
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return EvaluationResult(
                    success=False,
                    walltime_s=time.monotonic() - t0,
                    stdout="",
                    stderr=f"timeout after {case.timeout_s}s",
                    exit_code=124,
                )
            walltime = time.monotonic() - t0
            return EvaluationResult(
                success=(proc.returncode == 0),
                walltime_s=walltime,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
                exit_code=int(proc.returncode if proc.returncode is not None else -1),
            )
        except FileNotFoundError as e:
            return EvaluationResult(
                success=False,
                walltime_s=time.monotonic() - t0,
                stdout="",
                stderr=str(e),
                exit_code=127,
            )

    async def health(self) -> dict[str, Any]:
        return {
            "runner": self.name,
            "cpu_count": os.cpu_count(),
            "openfoam_dir": os.environ.get("FOAM_INST_DIR") or os.environ.get("WM_PROJECT_DIR"),
            "liggghts_bin": shutil.which("liggghts"),
        }
