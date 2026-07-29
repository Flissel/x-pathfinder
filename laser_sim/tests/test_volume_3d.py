"""Tests for the 3D volumetric pipeline:
  - t_max_volume_superposition shape + heat-decays-with-depth
  - scene.json v2 schema round-trip
  - VOLUME_FRAME event packing + base64 round-trip
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from laser_sim.config.schema import ScenarioConfig
from laser_sim.control_plane.events import EventType, pack_volume_frame
from laser_sim.patterns.base import PrimitiveKind, PrimitiveSpec
from laser_sim.patterns.primitives import build_primitive
from laser_sim.patterns.rasterize import rasterize_pattern
from laser_sim.physics.fast_sim import (
    FieldVolumeResult,
    t_max_volume_superposition,
)
from laser_sim.visualization.scene_export import export_scene


@pytest.fixture
def scenario() -> ScenarioConfig:
    path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    with open(path) as f:
        return ScenarioConfig.model_validate(yaml.safe_load(f))


def _pat(scenario):
    spec = PrimitiveSpec(
        kind=PrimitiveKind.ZIGZAG, power_W=200, speed_mm_s=800, hatch_um=100, spot_um=80
    )
    return build_primitive(spec, scenario.roi)


def test_t_max_volume_superposition_shape(scenario):
    rp = rasterize_pattern(_pat(scenario), ds_mm=0.05)
    vol = t_max_volume_superposition(
        rp, scenario.roi, scenario.material, scenario.machine,
        spot_um=80, nx=16, ny=16, nz=10, depth_mm=0.3, stride=12,
    )
    assert isinstance(vol, FieldVolumeResult)
    assert vol.t_max_K.shape == (16, 16, 10)
    assert vol.grid_x_mm.shape == (16,)
    assert vol.grid_z_mm.shape == (10,)
    assert vol.grid_z_mm[0] == pytest.approx(0.0)
    assert vol.grid_z_mm[-1] == pytest.approx(0.3)


def test_t_max_volume_heat_decays_with_depth(scenario):
    rp = rasterize_pattern(_pat(scenario), ds_mm=0.05)
    vol = t_max_volume_superposition(
        rp, scenario.roi, scenario.material, scenario.machine,
        spot_um=80, nx=16, ny=16, nz=10, depth_mm=0.4, stride=12,
    )
    surface_max = vol.t_max_K[:, :, 0].max()
    deep_max = vol.t_max_K[:, :, -1].max()
    assert surface_max > deep_max
    surface_mean = vol.t_max_K[:, :, 0].mean()
    deep_mean = vol.t_max_K[:, :, -1].mean()
    assert surface_mean >= deep_mean


def test_t_max_volume_baseline_above_preheat(scenario):
    rp = rasterize_pattern(_pat(scenario), ds_mm=0.05)
    vol = t_max_volume_superposition(
        rp, scenario.roi, scenario.material, scenario.machine,
        spot_um=80, nx=12, ny=12, nz=8, depth_mm=0.4, stride=16,
    )
    # linear superposition above preheat — nothing can drop below it
    assert vol.t_max_K.min() >= scenario.machine.preheat_K - 1e-6


def test_t_max_volume_return_frames(scenario):
    rp = rasterize_pattern(_pat(scenario), ds_mm=0.05)
    vol = t_max_volume_superposition(
        rp, scenario.roi, scenario.material, scenario.machine,
        spot_um=80, nx=12, ny=12, nz=8, depth_mm=0.4, stride=16,
        n_time_checkpoints=4, return_frames=True,
    )
    assert len(vol.frames) == 4
    for f in vol.frames:
        assert "t_s" in f and "t_max_K" in f
        assert f["t_max_K"].shape == (12, 12, 8)


def test_scene_export_v2_round_trip(scenario, tmp_path):
    out = tmp_path / "scene_v2.json"
    export_scene(
        _pat(scenario), scenario, out_path=out,
        spot_um=80, grid_n=12, n_frames=4, stride=16,
        transient_mode="superposition_3d", nz=8, depth_mm=0.3,
    )
    data = json.loads(out.read_text())
    assert data["schema_version"] == 2
    assert "grid_z_mm" in data["field"]
    assert len(data["field"]["grid_z_mm"]) == 8
    assert data["field"]["depth_mm"] == pytest.approx(0.3)
    assert "preheat_K" in data["field"]
    assert len(data["field"]["frames"]) == 4
    f0 = data["field"]["frames"][0]
    # nested 3D: t_max_K[i][j][k]
    arr = np.asarray(f0["t_max_K"])
    assert arr.shape == (12, 12, 8)


def test_scene_export_v1_still_works(scenario, tmp_path):
    """The 2D pathway must stay backward-compatible — schema v1."""
    out = tmp_path / "scene_v1.json"
    export_scene(
        _pat(scenario), scenario, out_path=out,
        spot_um=80, grid_n=12, n_frames=4, stride=16,
        transient_mode="superposition",
    )
    data = json.loads(out.read_text())
    assert data.get("schema_version", 1) == 1
    assert "grid_z_mm" not in data["field"]
    arr = np.asarray(data["field"]["frames"][0]["t_max_K"])
    assert arr.ndim == 2  # 2D heightfield


def test_pack_volume_frame_round_trip():
    Nx, Ny, Nz = 8, 8, 4
    rng = np.random.default_rng(0)
    T = rng.uniform(298, 3500, size=(Nx, Ny, Nz))
    p = pack_volume_frame(
        T, vmin_K=298, vmax_K=3500, t_s=0.05,
        laser_x_mm=1.2, laser_y_mm=-0.5, frame_index=7, n_frames=24,
    )
    assert p["shape"] == [Nx, Ny, Nz]
    assert p["dtype"] == "uint8"
    assert p["frame_index"] == 7
    # decode and check it round-trips within quantization tolerance
    raw = base64.b64decode(p["data_b64"])
    assert len(raw) == Nx * Ny * Nz
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((Nx, Ny, Nz))
    span = p["vmax_K"] - p["vmin_K"]
    T_back = p["vmin_K"] + (arr.astype(float) / 255.0) * span
    # at most 1 quant step = span/255 = ~12.5 K
    assert np.abs(T_back - T).max() < span / 255.0 + 1e-6


def test_pack_volume_frame_rejects_non_3d():
    T2 = np.zeros((8, 8))
    with pytest.raises(ValueError):
        pack_volume_frame(
            T2, vmin_K=298, vmax_K=3500, t_s=0, laser_x_mm=0, laser_y_mm=0,
            frame_index=0, n_frames=1,
        )


def test_event_type_volume_frame_exists():
    assert EventType.VOLUME_FRAME.value == "volume_frame"


def test_pack_volume_frame_optional_grid_metadata():
    """VOLUME_FRAME should carry roi_mm + depth_mm as fallback when the
    caller provides them (for late SSE subscribers who missed campaign_start)."""
    T = np.full((4, 4, 3), 500.0)
    p = pack_volume_frame(
        T, vmin_K=298, vmax_K=3500, t_s=0.01,
        laser_x_mm=0.0, laser_y_mm=0.0, frame_index=0, n_frames=1,
        roi_mm=(-2.5, -2.5, 2.5, 2.5), depth_mm=0.4,
    )
    assert p["roi_mm"] == [-2.5, -2.5, 2.5, 2.5]
    assert p["depth_mm"] == pytest.approx(0.4)
    # metadata is optional — omitting the kwargs must not add the keys
    p2 = pack_volume_frame(
        T, vmin_K=298, vmax_K=3500, t_s=0.01,
        laser_x_mm=0.0, laser_y_mm=0.0, frame_index=0, n_frames=1,
    )
    assert "roi_mm" not in p2
    assert "depth_mm" not in p2


def test_sim_live_campaign_start_carries_grid_metadata(scenario, tmp_path, monkeypatch):
    """Invoke the sim-live handler in-process (skip Flask/threads) and confirm
    the CAMPAIGN_START event payload contains grid_x_mm/grid_y_mm/grid_z_mm +
    roi_mm + preheat/liquidus/boiling so the browser viewer can place voxels
    in real mm coordinates without falling back to a synthetic unit cube."""
    import threading
    import laser_sim.cli as cli_mod
    from laser_sim.control_plane.events import EventBus, EventType

    captured: list = []

    class _FakeBus(EventBus):
        def publish(self, ev):
            captured.append(ev)

    # patch typer's helpers we don't need + intercept EventBus + the
    # serve function so sim-live doesn't bind a port or loop forever.
    monkeypatch.setattr(cli_mod, "_console", type("C", (), {"print": lambda *a, **kw: None})())
    stop = threading.Event()

    class _DoneLoop(Exception):
        pass

    call_state = {"first": True}

    def _fake_sleep(seconds):
        # let the very first sleep (server bind wait) succeed, then abort on the
        # first per-frame sleep so we only produce one frame before exiting
        if call_state["first"]:
            call_state["first"] = False
            return
        stop.set()
        raise _DoneLoop()

    monkeypatch.setattr("time.sleep", _fake_sleep)
    monkeypatch.setattr("laser_sim.control_plane.events.EventBus", _FakeBus)

    # neuter Flask server
    def _fake_serve(**kw):
        stop.wait()  # would normally block forever

    monkeypatch.setattr("laser_sim.visualization.web_app.serve", _fake_serve)

    try:
        cli_mod.sim_live(
            primitive="zigzag", power=220.0, speed=1000.0, hatch=100.0, spot=80.0,
            rotation=0.0, scenario=None, grid_n=16, nz=8, depth_mm=0.3,
            frames=3, stride=8, fps=100.0, host="127.0.0.1", port=0,
            loop=False,
        )
    except _DoneLoop:
        pass
    except SystemExit:
        pass  # typer.Exit for the "server keeps running" while True

    starts = [e for e in captured if e.type is EventType.CAMPAIGN_START]
    assert starts, "no CAMPAIGN_START event was published"
    p = starts[0].payload
    for key in ("grid_x_mm", "grid_y_mm", "grid_z_mm",
                "roi_mm", "preheat_K", "liquidus_K", "boiling_K"):
        assert key in p, f"CAMPAIGN_START missing {key}"
    assert len(p["grid_x_mm"]) == 16
    assert len(p["grid_z_mm"]) == 8
    assert p["grid_z_mm"][0] == pytest.approx(0.0)
    assert p["grid_z_mm"][-1] == pytest.approx(0.3)
    # roi matches the scenario ROI
    assert p["roi_mm"][0] == scenario.roi.x0_mm
    assert p["roi_mm"][2] == scenario.roi.x1_mm

    # at least one VOLUME_FRAME was published with the same shape and
    # fallback metadata
    volumes = [e for e in captured if e.type is EventType.VOLUME_FRAME]
    assert volumes, "no VOLUME_FRAME published"
    vp = volumes[0].payload
    assert vp["shape"] == [16, 16, 8]
    assert "roi_mm" in vp and "depth_mm" in vp
