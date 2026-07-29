"""Worker abstraction: ship a CaseSpec to compute, get back an EvaluationResult.

Concrete implementations:
  - LocalRunner    multiprocessing on the current host
  - SSHRunner      scp + ssh tarball execution (mirrors x-pathfinder/worker_client)
  - K8sRunner      Kubernetes Jobs (HF runs at scale)
  - SlurmRunner    sbatch arrays on a HPC cluster

All runners must be timeout-aware, sandboxed (mkdtemp), and report
health so the scheduler can shed unhealthy nodes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CaseSpec:
    """Self-contained description of a run.

    case_dir: existing on-disk directory containing the case (e.g. an
              OpenFOAM case folder). The runner is allowed to mutate it.
    solver_name: 'laserbeamFoam', 'lasermeltFoam', 'liggghts', 'fast_jax', ...
    args: solver-specific CLI args (positional and flags)
    env: environment variable overrides for the subprocess
    timeout_s: wall-clock timeout per run; runner must kill on overshoot
    """

    case_dir: Path
    solver_name: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 3600.0


@dataclass(frozen=True)
class EvaluationResult:
    success: bool
    walltime_s: float
    stdout: str
    stderr: str
    exit_code: int
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)


class Runner(ABC):
    name: str

    @abstractmethod
    async def submit(self, case: CaseSpec) -> EvaluationResult: ...

    @abstractmethod
    async def health(self) -> dict[str, Any]: ...
