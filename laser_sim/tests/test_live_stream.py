"""Tests for the live SSE event-bus + engine event emission."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from laser_sim.config.schema import ScenarioConfig
from laser_sim.control_plane.events import EventBus, EventType, EvolutionEvent
from laser_sim.fitness.evaluator import PatternEvaluator
from laser_sim.genome.engine import GeneticEngine
from laser_sim.storage import open_database
from laser_sim.visualization.web_app import create_app


@pytest.fixture
def scenario() -> ScenarioConfig:
    path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return ScenarioConfig.model_validate(data)


def test_event_bus_publishes_to_subscribers() -> None:
    bus = EventBus()
    q1, backlog1 = bus.subscribe()
    q2, _ = bus.subscribe()
    assert backlog1 == []
    bus.publish(EvolutionEvent(type=EventType.GENERATION, payload={"g": 0}))
    bus.publish(EvolutionEvent(type=EventType.GENERATION, payload={"g": 1}))
    assert q1.get(timeout=0.5).payload["g"] == 0
    assert q1.get(timeout=0.5).payload["g"] == 1
    # second subscriber gets the same stream
    assert q2.get(timeout=0.5).payload["g"] == 0
    assert q2.get(timeout=0.5).payload["g"] == 1


def test_event_bus_backlog_for_late_subscriber() -> None:
    bus = EventBus()
    bus.publish(EvolutionEvent(type=EventType.GENERATION, payload={"g": 7}))
    q, backlog = bus.subscribe()
    assert len(backlog) == 1
    assert backlog[0].payload["g"] == 7


def test_event_bus_unsubscribe_releases_queue() -> None:
    bus = EventBus()
    q, _ = bus.subscribe()
    assert bus.subscriber_count == 1
    bus.unsubscribe(q)
    assert bus.subscriber_count == 0


def test_engine_emits_campaign_and_generation_events(scenario: ScenarioConfig) -> None:
    bus = EventBus()
    ea = scenario.ea.model_copy(update={"population": 6, "generations": 2, "elitism": 1, "seed": 11})
    eng = GeneticEngine(
        material=scenario.material,
        machine=scenario.machine,
        roi=scenario.roi,
        ea=ea,
        event_bus=bus,
    )
    eng.run()
    # collect all published events from the history
    q, backlog = bus.subscribe()
    types = [ev.type for ev in backlog]
    assert EventType.CAMPAIGN_START in types
    assert EventType.GENERATION in types
    assert EventType.CAMPAIGN_END in types
    # at least one GENERATION event has a 'best' field
    gen_events = [ev for ev in backlog if ev.type is EventType.GENERATION]
    assert any("best" in ev.payload and "scalar_J" in ev.payload["best"] for ev in gen_events)


def test_sse_stream_endpoint_returns_text_event_stream(tmp_path: Path, scenario: ScenarioConfig) -> None:
    bus = EventBus()
    bus.publish(EvolutionEvent(type=EventType.GENERATION, payload={"g": 42}))
    db_path = tmp_path / "live.db"
    open_database(db_path).close()
    app = create_app(db_path=db_path, event_bus=bus)
    client = app.test_client()
    res = client.get("/api/stream", buffered=False)
    assert res.status_code == 200
    assert res.headers["Content-Type"].startswith("text/event-stream")
    # Read the first chunk; it should contain our published event from backlog.
    iterator = res.iter_encoded()
    first = next(iterator)
    payload = first.decode()
    assert "data:" in payload
    data = json.loads(payload.removeprefix("data: ").splitlines()[0])
    assert data["type"] == "generation"
    assert data["payload"]["g"] == 42
    res.close()


def test_api_stream_404_when_bus_missing(tmp_path: Path, scenario: ScenarioConfig) -> None:
    db_path = tmp_path / "x.db"
    open_database(db_path).close()
    app = create_app(db_path=db_path)
    res = app.test_client().get("/api/stream")
    assert res.status_code == 404


def test_evolution_event_to_sse_format() -> None:
    ev = EvolutionEvent(type=EventType.CAMPAIGN_END, payload={"archive_size": 41})
    sse = ev.to_sse()
    assert sse.startswith("data: ")
    assert sse.endswith("\n\n")
    body = json.loads(sse[len("data: ") :].rstrip())
    assert body["type"] == "campaign_end"
    assert body["payload"]["archive_size"] == 41
