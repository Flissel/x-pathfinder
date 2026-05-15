# laser_sim

Multi-fidelity evolutionary optimizer for LPBF (Laser Powder-Bed Fusion) scan patterns.

This package lives parallel to the existing `x-pathfinder` Twitter-discovery code in this repo. The architecture (DEM powder bed, OpenFOAM/laserbeamFoam melt pool, custom EA, FastAPI/Redis control plane, NIST AM-Bench validation anchor, closed-loop hardware bridge) is detailed in `/root/.claude/plans/also-es-ist-so-rustling-hoare.md`.

## Status

Phase 0 + 2 (partial) scaffolding:

- versioned `ScenarioConfig` (Pydantic v2) covering material, powder, machine envelope, ROI, EA, fidelity gate
- pattern DSL: `PrimitiveSpec`, `ScanPattern`, `Segment`, `Waypoint`, `RasterizedPath`
- three implemented primitives: `zigzag`, `stripes`, `spiral` (others registered, raise `NotImplementedError`)
- arc-length rasterizer producing dense `(x, y, t, P, v)` trajectories
- matplotlib-based pattern preview
- typer CLI with `info`, `scenario {show,validate}`, `viz-pattern`
- 11 pytest cases covering schema validation, primitives, rasterizer, time/length invariants

## Install (dev)

```bash
pip install -e ".[dev]"
```

## Use

```bash
python -m laser_sim info
python -m laser_sim scenario show
python -m laser_sim scenario validate --path laser_sim/config/default.yaml
python -m laser_sim viz-pattern --primitive zigzag --power 200 --speed 800 --hatch 100 \
    --out pattern.png --json-out pattern.json
python -m laser_sim viz-pattern --primitive stripes --hatch 80 --stripe-width 1.5 --rotation 67
python -m laser_sim viz-pattern --primitive spiral --hatch 120
```

## Next phases

Per the approved plan:

- **P1** FastAPI + Redis Streams control plane
- **P3** 2.5D JAX/CuPy fast-sim solver, vectorized batch
- **P4** custom `GeneticEngine` with NSGA-II Pareto mechanics
- **P5** LIGGGHTS DEM powder bed
- **P6** OpenFOAM/laserbeamFoam HF integration (K8s/Slurm runner)
- **P7** NIST AM-Bench validation harness
- **P8** hardware bridge with safety interlocks
- **P9** GP/PINN surrogate, multi-fidelity gating
