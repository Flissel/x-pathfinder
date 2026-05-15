"""Tests for the 3D scene exporter and the Flask web dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from laser_sim.config.schema import ScenarioConfig
from laser_sim.patterns.base import PrimitiveKind, PrimitiveSpec
from laser_sim.patterns.primitives import build_primitive
from laser_sim.storage import open_database
from laser_sim.visualization.scene_export import export_scene
from laser_sim.visualization.web_app import create_app


@pytest.fixture
def scenario() -> ScenarioConfig:
    path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return ScenarioConfig.model_validate(data)


def test_scene_export_round_trip(tmp_path: Path, scenario: ScenarioConfig) -> None:
    spec = PrimitiveSpec(
        kind=PrimitiveKind.ZIGZAG, power_W=200, speed_mm_s=800, hatch_um=100, spot_um=80
    )
    pat = build_primitive(spec, scenario.roi)
    out = tmp_path / "scene.json"
    export_scene(pat, scenario, out, grid_n=16, n_frames=4, stride=8)
    data = json.loads(out.read_text())
    assert "path" in data
    assert "field" in data
    assert len(data["field"]["frames"]) == 4
    assert len(data["field"]["grid_x_mm"]) == 16
    # all frame grids match
    for f in data["field"]["frames"]:
        assert len(f["t_max_K"]) == 16
        assert len(f["t_max_K"][0]) == 16
    # times monotonic
    times = [f["t_s"] for f in data["field"]["frames"]]
    assert times == sorted(times)


def test_scene_export_field_grows_over_time(tmp_path: Path, scenario: ScenarioConfig) -> None:
    """The cumulative T_max field should never lose heat: each frame's mean
    should be at least the previous frame's mean (heat only accumulates)."""
    spec = PrimitiveSpec(
        kind=PrimitiveKind.ZIGZAG, power_W=200, speed_mm_s=800, hatch_um=100, spot_um=80
    )
    pat = build_primitive(spec, scenario.roi)
    out = tmp_path / "scene.json"
    export_scene(pat, scenario, out, grid_n=16, n_frames=4, stride=4)
    data = json.loads(out.read_text())
    means = []
    for f in data["field"]["frames"]:
        flat = [v for row in f["t_max_K"] for v in row]
        means.append(sum(flat) / len(flat))
    # monotone non-decreasing (allow rounding noise)
    for a, b in zip(means, means[1:]):
        assert b >= a - 1.0


def test_webapp_index_lists_campaigns(tmp_path: Path, scenario: ScenarioConfig) -> None:
    db_path = tmp_path / "web.db"
    db = open_database(db_path)
    try:
        sid = db.upsert_scenario(json.loads(scenario.model_dump_json()))
        db.create_campaign("web_smoke", sid)
    finally:
        db.close()
    app = create_app(db_path=db_path)
    client = app.test_client()
    res = client.get("/")
    assert res.status_code == 200
    assert b"web_smoke" in res.data


def test_webapp_api_endpoints(tmp_path: Path, scenario: ScenarioConfig) -> None:
    db_path = tmp_path / "api.db"
    db = open_database(db_path)
    try:
        sid = db.upsert_scenario(json.loads(scenario.model_dump_json()))
        cid = db.create_campaign("api_smoke", sid)
        # insert real candidates first (archive_entries has an FK on them)
        c1 = db.upsert_candidate(cid, "hashA", {"kind": "zigzag", "power_W": 100}, generation=0)
        c2 = db.upsert_candidate(cid, "hashB", {"kind": "zigzag", "power_W": 200}, generation=1)
        db.replace_archive(
            cid,
            [
                (c1, (0.1, 0.2, 0.3, 0.4, 0.5), 0.25, 0),
                (c2, (0.2, 0.1, 0.3, 0.4, 0.6), 0.30, 1),
            ],
        )
    finally:
        db.close()
    app = create_app(db_path=db_path)
    client = app.test_client()
    res = client.get("/api/campaigns")
    assert res.status_code == 200
    payload = res.get_json()
    assert any(c["name"] == "api_smoke" for c in payload)
    res = client.get("/api/campaign/api_smoke")
    assert res.status_code == 200
    body = res.get_json()
    assert len(body["archive"]) == 2
    # ordering by scalar_J ASC
    assert body["archive"][0]["scalar_J"] <= body["archive"][1]["scalar_J"]


def test_webapp_serves_viewer_html(tmp_path: Path, scenario: ScenarioConfig) -> None:
    db_path = tmp_path / "viewer.db"
    db = open_database(db_path)
    db.close()
    app = create_app(db_path=db_path)
    client = app.test_client()
    res = client.get("/viewer")
    assert res.status_code == 200
    assert b"three" in res.data.lower() or b"three.js" in res.data.lower()


def test_webapp_serves_scene_json(tmp_path: Path, scenario: ScenarioConfig) -> None:
    spec = PrimitiveSpec(
        kind=PrimitiveKind.ZIGZAG, power_W=200, speed_mm_s=800, hatch_um=100, spot_um=80
    )
    pat = build_primitive(spec, scenario.roi)
    scene_path = tmp_path / "s.json"
    export_scene(pat, scenario, scene_path, grid_n=12, n_frames=3, stride=8)
    db_path = tmp_path / "s.db"
    db = open_database(db_path)
    db.close()
    app = create_app(db_path=db_path, scene_json=scene_path)
    client = app.test_client()
    res = client.get("/scene.json")
    assert res.status_code == 200
    payload = res.get_json()
    assert payload["meta"]["n_frames"] == 3
