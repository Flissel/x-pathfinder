"""Export a scan-pattern + transient thermal field as a JSON scene for the
3D viewer (laser_sim/visualization/three_js_app/index.html).

The viewer renders three layers:
- the rasterized scan path as a colored polyline (color = power)
- the time-stepped T_max field on a heightfield/heatmap plane
- an animated 'laser head' sphere that scrubs along the path with time

Scene JSON schema:
{
  "scenario_id": str,
  "roi_mm": [x0, y0, x1, y1],
  "path": {
    "x_mm": [..],
    "y_mm": [..],
    "t_s": [..],
    "power_W": [..],
    "speed_mm_s": [..]
  },
  "field": {
    "grid_x_mm": [..],
    "grid_y_mm": [..],
    "frames": [
      {"t_s": float, "t_max_K": [[..], ..]}, ...
    ],
    "liquidus_K": float,
    "boiling_K": float
  },
  "meta": {...}
}
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from laser_sim.config.schema import ScenarioConfig
from laser_sim.patterns.base import RasterizedPath, ScanPattern
from laser_sim.patterns.rasterize import rasterize_pattern
from laser_sim.physics.fast_sim.eagar_tsai import rosenthal_field


def _time_stepped_field(
    path: RasterizedPath,
    scenario: ScenarioConfig,
    spot_um: float,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    n_frames: int,
    stride: int,
) -> list[dict[str, Any]]:
    """Compute T_max field at each of n_frames time checkpoints.

    Each checkpoint includes contributions only from path samples with
    t_k <= t_checkpoint, so the field grows realistically as the laser scans.
    """
    if path.n_samples() < 2:
        return []
    t0, t1 = float(path.t_s.min()), float(path.t_s.max())
    checkpoints = np.linspace(t0, t1, n_frames)
    Xg, Yg = np.meshgrid(grid_x, grid_y, indexing="ij")
    px = path.x_mm
    py = path.y_mm
    pP = path.power_W
    pv = path.speed_mm_s
    pt = path.t_s
    pon = path.laser_on
    dx = np.gradient(px)
    dy = np.gradient(py)
    norm = np.hypot(dx, dy)
    norm = np.where(norm > 1e-12, norm, 1.0)
    hx = dx / norm
    hy = dy / norm
    frames: list[dict[str, Any]] = []
    preheat = float(scenario.machine.preheat_K)
    for tj in checkpoints:
        T_max = np.full(Xg.shape, preheat)
        idx = np.where((pt <= tj) & pon & (pP > 0))[0][::max(stride, 1)]
        z = np.zeros_like(Xg)
        for k in idx:
            rx = Xg - px[k]
            ry = Yg - py[k]
            xi = rx * hx[k] + ry * hy[k]
            eta = -rx * hy[k] + ry * hx[k]
            T_k = rosenthal_field(
                float(pP[k]),
                float(pv[k]),
                scenario.material,
                preheat,
                xi,
                eta,
                z,
                spot_um=spot_um,
            )
            np.maximum(T_max, T_k, out=T_max)
        frames.append(
            {
                "t_s": float(tj),
                "t_max_K": T_max.round(1).tolist(),
            }
        )
    return frames


def export_scene(
    pattern: ScanPattern,
    scenario: ScenarioConfig,
    out_path: Path,
    *,
    spot_um: float = 80.0,
    grid_n: int = 32,
    n_frames: int = 24,
    stride: int = 6,
    rasterize_ds_mm: float = 0.05,
) -> Path:
    """Build and dump the scene JSON.

    Defaults aim for a smooth animation (24 frames) on a moderate grid (32^2)
    with reasonable runtime: ~24 frames * 32^2 cells * 500 path samples ~ 12M ops.
    """
    rp = rasterize_pattern(pattern, ds_mm=rasterize_ds_mm)
    grid_x = np.linspace(scenario.roi.x0_mm, scenario.roi.x1_mm, grid_n)
    grid_y = np.linspace(scenario.roi.y0_mm, scenario.roi.y1_mm, grid_n)
    frames = _time_stepped_field(
        rp, scenario, spot_um, grid_x, grid_y, n_frames=n_frames, stride=stride
    )
    payload = {
        "scenario_id": str(scenario.scenario_id),
        "objective_version": scenario.objective_version,
        "roi_mm": [
            scenario.roi.x0_mm,
            scenario.roi.y0_mm,
            scenario.roi.x1_mm,
            scenario.roi.y1_mm,
        ],
        "path": {
            "x_mm": rp.x_mm.round(4).tolist(),
            "y_mm": rp.y_mm.round(4).tolist(),
            "t_s": rp.t_s.round(6).tolist(),
            "power_W": rp.power_W.round(2).tolist(),
            "speed_mm_s": rp.speed_mm_s.round(2).tolist(),
        },
        "field": {
            "grid_x_mm": grid_x.round(4).tolist(),
            "grid_y_mm": grid_y.round(4).tolist(),
            "frames": frames,
            "liquidus_K": scenario.material.liquidus_K,
            "boiling_K": scenario.material.boiling_K,
        },
        "meta": {
            "spot_um": spot_um,
            "n_frames": n_frames,
            "grid_n": grid_n,
            "stride": stride,
            "n_path_samples": rp.n_samples(),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    return out_path
