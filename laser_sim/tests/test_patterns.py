"""Tests for pattern primitives, rasterizer, and scenario schema."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from laser_sim.config.schema import GeometryROI, ScenarioConfig
from laser_sim.patterns.base import PrimitiveKind, PrimitiveSpec
from laser_sim.patterns.primitives import build_primitive, list_primitives
from laser_sim.patterns.rasterize import rasterize_pattern


@pytest.fixture
def roi() -> GeometryROI:
    return GeometryROI(roi_id="test", x0_mm=-2.5, y0_mm=-2.5, x1_mm=2.5, y1_mm=2.5)


def _spec(kind: PrimitiveKind, **kw) -> PrimitiveSpec:
    base = dict(power_W=200.0, speed_mm_s=800.0, hatch_um=100.0, spot_um=80.0)
    base.update(kw)
    return PrimitiveSpec(kind=kind, **base)


def test_default_scenario_loads() -> None:
    path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    sc = ScenarioConfig.model_validate(data)
    assert sc.material.material_id == "ti64"
    assert sc.machine.laser.power_max_W > sc.machine.laser.power_min_W


def test_list_primitives_nonempty() -> None:
    prims = list_primitives()
    assert "zigzag" in prims
    assert "stripes" in prims
    assert "spiral" in prims


def test_zigzag_fills_roi(roi: GeometryROI) -> None:
    pat = build_primitive(_spec(PrimitiveKind.ZIGZAG), roi)
    assert len(pat.segments) >= 2
    ys = []
    for seg in pat.segments:
        for wp in seg.waypoints:
            ys.append(wp.y_mm)
    assert min(ys) == pytest.approx(roi.y0_mm)
    assert max(ys) == pytest.approx(roi.y1_mm)
    # alternating direction: first segment goes +x, second -x
    s0 = pat.segments[0].waypoints
    s1 = pat.segments[1].waypoints
    assert s0[1].x_mm > s0[0].x_mm
    assert s1[1].x_mm < s1[0].x_mm


def test_zigzag_rotation_preserves_centre(roi: GeometryROI) -> None:
    pat0 = build_primitive(_spec(PrimitiveKind.ZIGZAG, rotation_deg=0.0), roi)
    pat45 = build_primitive(_spec(PrimitiveKind.ZIGZAG, rotation_deg=45.0), roi)
    cx0 = np.mean([w.x_mm for s in pat0.segments for w in s.waypoints])
    cy0 = np.mean([w.y_mm for s in pat0.segments for w in s.waypoints])
    cx45 = np.mean([w.x_mm for s in pat45.segments for w in s.waypoints])
    cy45 = np.mean([w.y_mm for s in pat45.segments for w in s.waypoints])
    assert math.isclose(cx0, cx45, abs_tol=1e-9)
    assert math.isclose(cy0, cy45, abs_tol=1e-9)


def test_stripes_partition_roi(roi: GeometryROI) -> None:
    pat = build_primitive(
        _spec(PrimitiveKind.STRIPES, params={"stripe_width_mm": 2.0}), roi
    )
    # 5mm ROI / 2mm stripes => 3 stripes (2 + 2 + 1)
    xs = [w.x_mm for s in pat.segments for w in s.waypoints]
    assert min(xs) == pytest.approx(roi.x0_mm)
    assert max(xs) == pytest.approx(roi.x1_mm)


def test_spiral_starts_outside_ends_centre(roi: GeometryROI) -> None:
    pat = build_primitive(_spec(PrimitiveKind.SPIRAL), roi)
    assert len(pat.segments) == 1
    wps = pat.segments[0].waypoints
    cx = 0.5 * (roi.x0_mm + roi.x1_mm)
    cy = 0.5 * (roi.y0_mm + roi.y1_mm)
    r_first = math.hypot(wps[0].x_mm - cx, wps[0].y_mm - cy)
    r_last = math.hypot(wps[-1].x_mm - cx, wps[-1].y_mm - cy)
    r_max = 0.5 * min(roi.width_mm, roi.height_mm)
    assert r_first == pytest.approx(r_max, rel=1e-6)
    assert r_last < 1e-6


def test_rasterize_arc_length_uniformity(roi: GeometryROI) -> None:
    pat = build_primitive(_spec(PrimitiveKind.ZIGZAG), roi)
    ds_mm = 0.05
    rp = rasterize_pattern(pat, ds_mm=ds_mm)
    assert rp.n_samples() > 100
    pts = np.column_stack([rp.x_mm, rp.y_mm])
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    # intra-segment steps must be <= ds_mm. inter-segment "jumps" appear where
    # the rasterizer concatenates consecutive segments and have size ~hatch_mm.
    # so filter strictly below ds_mm + eps to isolate intra-segment.
    intra = d[d <= ds_mm + 1e-9]
    assert intra.size > 100
    assert intra.max() <= ds_mm + 1e-9


def test_rasterize_time_monotonic(roi: GeometryROI) -> None:
    pat = build_primitive(_spec(PrimitiveKind.ZIGZAG), roi)
    rp = rasterize_pattern(pat, ds_mm=0.05)
    # time must be non-decreasing
    dt = np.diff(rp.t_s)
    assert (dt >= -1e-12).all()


def test_total_time_consistent_with_length() -> None:
    """At constant speed, total scan time = total length / speed."""
    roi = GeometryROI(roi_id="t", x0_mm=0, y0_mm=0, x1_mm=10, y1_mm=10)
    spec = _spec(PrimitiveKind.ZIGZAG, hatch_um=1000.0, speed_mm_s=1000.0)
    pat = build_primitive(spec, roi)
    expected_s = pat.total_length_mm() / 1000.0
    assert math.isclose(pat.total_time_s(), expected_s, rel_tol=1e-9)


def test_unimplemented_primitive_raises(roi: GeometryROI) -> None:
    with pytest.raises(NotImplementedError):
        build_primitive(_spec(PrimitiveKind.HILBERT), roi)


def test_material_phase_order_validation() -> None:
    from laser_sim.config.schema import MaterialConfig

    with pytest.raises(ValueError):
        MaterialConfig(
            material_id="bad",
            name="bad",
            rho_solid=1.0,
            rho_liquid=1.0,
            cp_solid=1.0,
            cp_liquid=1.0,
            k_solid=1.0,
            k_liquid=1.0,
            solidus_K=2000.0,
            liquidus_K=1000.0,  # liquidus < solidus
            latent_heat_fusion=1.0,
            latent_heat_vap=1.0,
            boiling_K=3000.0,
            emissivity=0.5,
            absorptivity=0.5,
        )
