"""NIST AM-Bench loader stub.

The Additive Manufacturing Benchmark Test Series (AM-Bench) is the closest
thing to a community gold-standard for LPBF model validation:
  https://www.nist.gov/ambench

The 2018, 2022 and 2025 rounds publish single-track and 3D-coupon datasets
covering melt-pool geometry, in-situ thermal histories, residual stress,
microstructure and as-built distortion for IN625, IN718 and SS316L. We use
those as an *external* hold-out: the EA optimizer must never be tuned on
AM-Bench data, so it stays a clean validation anchor.

This module is intentionally a stub: the full loaders depend on which AM-Bench
release the user downloads (the formats vary across rounds). The schema
documented here is what the rest of the harness expects an AM-Bench loader
to return. When you obtain a release, drop a concrete loader at
am_bench/<release>.py and it will plug into validate / calibrate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from laser_sim.validation.ingest import MeasurementSet


@dataclass(frozen=True)
class AmBenchRelease:
    name: str                           # e.g. 'AMB2022-01-LPBF-Bridge'
    material: str                       # e.g. 'in718'
    description: str
    path: Path
    citation: str


KNOWN_RELEASES: tuple[AmBenchRelease, ...] = (
    AmBenchRelease(
        name="AMB2018-02",
        material="in625",
        description="Single-track + 3D builds, IN625, in-situ thermal + ex-situ XCT",
        path=Path("./data/am_bench/AMB2018-02"),
        citation="NIST AM Bench 2018, https://www.nist.gov/ambench/amb2018-02",
    ),
    AmBenchRelease(
        name="AMB2022-01",
        material="in718",
        description="LPBF Bridge geometry, IN718, melt-pool monitoring + distortion",
        path=Path("./data/am_bench/AMB2022-01"),
        citation="NIST AM Bench 2022, https://www.nist.gov/ambench/amb2022-01",
    ),
)


def load_am_bench_release(name: str, root: Path | None = None) -> MeasurementSet:
    """Load a known AM-Bench release into a MeasurementSet.

    NOT IMPLEMENTED in this slice: the per-release loaders need the actual
    AM-Bench data layout (each release is structured differently). Once a
    release is obtained, write a per-release loader and dispatch here.
    """
    matches = [r for r in KNOWN_RELEASES if r.name == name]
    if not matches:
        known = ", ".join(r.name for r in KNOWN_RELEASES)
        raise KeyError(f"unknown AM-Bench release {name!r}; known: {known}")
    raise NotImplementedError(
        f"AM-Bench loader for {name!r} not implemented yet. "
        f"See {matches[0].citation} for data access."
    )
