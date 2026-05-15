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


def _time_stepped_field_max(
    path: RasterizedPath,
    scenario: ScenarioConfig,
    spot_um: float,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    n_frames: int,
    stride: int,
) -> list[dict[str, Any]]:
    """Cumulative max of steady-state Rosenthal contributions per checkpoint."""
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
        frames.append({"t_s": float(tj), "t_max_K": T_max.round(1).tolist()})
    return frames


def _time_stepped_field_superposition(
    path: RasterizedPath,
    scenario: ScenarioConfig,
    spot_um: float,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    n_frames: int,
    stride: int,
) -> list[dict[str, Any]]:
    """3D Green's function summation per checkpoint — true heat accumulation."""
    if path.n_samples() < 2:
        return []
    keep = path.laser_on & (path.power_W > 0)
    px = np.asarray(path.x_mm)[keep][::max(stride, 1)]
    py = np.asarray(path.y_mm)[keep][::max(stride, 1)]
    pP = np.asarray(path.power_W)[keep][::max(stride, 1)]
    pt = np.asarray(path.t_s)[keep][::max(stride, 1)]
    if pt.size < 2:
        return []
    dt = np.diff(pt, prepend=pt[0])
    pos_dt = dt[dt > 0]
    fill = float(np.median(pos_dt)) if pos_dt.size else 1e-5
    dt = np.where(dt > 0, dt, fill)

    mat = scenario.material
    eta = mat.absorptivity
    rho = mat.rho_solid
    cp = mat.cp_solid
    alpha = mat.k_solid / (rho * cp)
    r_min = max(spot_um * 1e-6 * 0.5, 5e-6)
    r_min2 = r_min ** 2

    Xg, Yg = np.meshgrid(grid_x, grid_y, indexing="ij")
    dx2 = (Xg[:, :, None] - px[None, None, :]) ** 2 * 1e-6
    dy2 = (Yg[:, :, None] - py[None, None, :]) ** 2 * 1e-6
    r2 = dx2 + dy2 + r_min2  # (nx, ny, Np)
    weights = (eta * pP * dt) / (rho * cp)  # (Np,)

    t0, t1 = float(pt[0]), float(pt[-1])
    tail = max((t1 - t0) * 0.15, 1e-3)
    checkpoints = np.linspace(t0 + 1e-5, t1 + tail, n_frames)
    preheat = float(scenario.machine.preheat_K)
    frames: list[dict[str, Any]] = []
    for tj in checkpoints:
        tau = tj - pt
        valid = tau > 1e-9
        tau_safe = np.where(valid, tau, 1.0)
        denom = (4.0 * np.pi * alpha * tau_safe) ** 1.5
        kernel = np.where(valid, np.exp(-r2 / (4.0 * alpha * tau_safe)) / denom, 0.0)
        T = preheat + (weights * kernel).sum(axis=2)
        frames.append({"t_s": float(tj), "t_max_K": T.round(1).tolist()})
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
    transient_mode: str = "superposition",
) -> Path:
    """Build and dump the scene JSON.

    transient_mode:
      "max"           cumulative per-source-max of steady-state Rosenthal
      "superposition" true time-domain 3D Green's function (physical heat
                      accumulation; recommended for visualization)
    """
    rp = rasterize_pattern(pattern, ds_mm=rasterize_ds_mm)
    grid_x = np.linspace(scenario.roi.x0_mm, scenario.roi.x1_mm, grid_n)
    grid_y = np.linspace(scenario.roi.y0_mm, scenario.roi.y1_mm, grid_n)
    if transient_mode == "superposition":
        frames = _time_stepped_field_superposition(
            rp, scenario, spot_um, grid_x, grid_y, n_frames=n_frames, stride=stride
        )
    else:
        frames = _time_stepped_field_max(
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
            "transient_mode": transient_mode,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    return out_path
