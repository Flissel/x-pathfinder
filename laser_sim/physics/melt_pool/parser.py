"""Read OpenFOAM case output back into numpy fields.

Requires ofpp (and optionally pyvista) — those live behind the `hf`
extra. parser_available() returns True only if at least ofpp is
importable; otherwise parse_case_output returns a synthetic dummy
result and the EvaluationResult marks itself unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ParsedHFResult:
    success: bool
    t_max_K: np.ndarray | None
    alpha_metal: np.ndarray | None
    times_s: tuple[float, ...]
    metrics: dict[str, float]
    error: str = ""


def parser_available() -> bool:
    try:
        import ofpp  # noqa: F401
    except ImportError:
        return False
    return True


def parse_case_output(case_dir: Path) -> ParsedHFResult:
    case_dir = Path(case_dir)
    if not case_dir.is_dir():
        return ParsedHFResult(
            success=False,
            t_max_K=None,
            alpha_metal=None,
            times_s=(),
            metrics={},
            error=f"case_dir {case_dir} not found",
        )
    if not parser_available():
        return ParsedHFResult(
            success=False,
            t_max_K=None,
            alpha_metal=None,
            times_s=(),
            metrics={},
            error=(
                "ofpp not installed; install with `pip install -e \".[hf]\"` "
                "to parse OpenFOAM time directories"
            ),
        )
    try:
        import ofpp

        time_dirs = sorted(
            [
                float(d.name)
                for d in case_dir.iterdir()
                if d.is_dir() and d.name.replace(".", "", 1).replace("e-", "", 1).isdigit()
            ]
        )
        if not time_dirs:
            return ParsedHFResult(
                success=False, t_max_K=None, alpha_metal=None, times_s=(),
                metrics={}, error="no time directories under case_dir",
            )
        last = time_dirs[-1]
        T = ofpp.parse_internal_field(str(case_dir / f"{last}/T"))
        alpha = ofpp.parse_internal_field(str(case_dir / f"{last}/alpha.metal"))
        return ParsedHFResult(
            success=True,
            t_max_K=np.asarray(T, dtype=float),
            alpha_metal=np.asarray(alpha, dtype=float),
            times_s=tuple(time_dirs),
            metrics={
                "t_max_K_max": float(np.asarray(T).max()),
                "t_max_K_mean": float(np.asarray(T).mean()),
                "alpha_metal_max": float(np.asarray(alpha).max()),
            },
        )
    except Exception as e:  # noqa: BLE001
        return ParsedHFResult(
            success=False, t_max_K=None, alpha_metal=None, times_s=(),
            metrics={}, error=f"parse failed: {type(e).__name__}: {e}",
        )
