"""Tests for SQLite-backed persistence and KnowledgeAccumulator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from laser_sim.config.schema import ScenarioConfig
from laser_sim.fitness.evaluator import FitnessVector
from laser_sim.genome.archive import ArchiveEntry
from laser_sim.genome.chromosome import Chromosome
from laser_sim.genome.engine import GeneticEngine
from laser_sim.patterns.base import PrimitiveKind
from laser_sim.storage import KnowledgeAccumulator, open_database


@pytest.fixture
def scenario() -> ScenarioConfig:
    path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return ScenarioConfig.model_validate(data)


def _make_entry(power: float = 200.0) -> ArchiveEntry:
    c = Chromosome(
        primitive_kind=PrimitiveKind.ZIGZAG,
        power_W=power,
        speed_mm_s=800.0,
        hatch_um=100.0,
        spot_um=80.0,
    )
    f = FitnessVector(values=(0.1, 0.2, 0.3, 0.4, 0.5))
    return ArchiveEntry(chromosome=c, fitness=f, scalar_J=0.25)


def test_database_schema_creates_idempotently(tmp_path: Path) -> None:
    db_path = tmp_path / "a.db"
    db1 = open_database(db_path)
    db1.close()
    db2 = open_database(db_path)  # second open must succeed without error
    assert db2.path == str(db_path)
    db2.close()


def test_upsert_scenario_and_create_campaign(tmp_path: Path, scenario: ScenarioConfig) -> None:
    db = open_database(tmp_path / "c.db")
    try:
        sid = db.upsert_scenario(json.loads(scenario.model_dump_json()))
        cid = db.create_campaign("smoke_test", sid)
        row = db.get_campaign_by_name("smoke_test")
        assert row is not None
        assert row.campaign_id == cid
        assert row.last_generation == 0
        # duplicate campaign name -> error
        with pytest.raises(ValueError):
            db.create_campaign("smoke_test", sid)
    finally:
        db.close()


def test_accumulator_writes_runs_and_archive(tmp_path: Path, scenario: ScenarioConfig) -> None:
    db = open_database(tmp_path / "k.db")
    try:
        sid = db.upsert_scenario(json.loads(scenario.model_dump_json()))
        cid = db.create_campaign("k_test", sid)
        accum = KnowledgeAccumulator(db=db, campaign_id=cid)
        e1 = _make_entry(power=150.0)
        e2 = _make_entry(power=250.0)
        accum.put(e1.chromosome.hash_id(), e1, generation=0, metrics={"a": 1.0})
        accum.put(e2.chromosome.hash_id(), e2, generation=0, metrics={"a": 2.0})
        accum.commit_archive([e1, e2], generation=1)
        # archive snapshot persisted
        rows = db.list_archive(cid)
        assert len(rows) == 2
        # campaign last_generation updated
        row = db.get_campaign_by_name("k_test")
        assert row is not None
        assert row.last_generation == 1
        # runs were inserted
        n_runs = db.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        assert n_runs == 2
        # metrics were inserted
        n_metrics = db.conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
        assert n_metrics == 2
    finally:
        db.close()


def test_accumulator_hydrates_from_disk(tmp_path: Path, scenario: ScenarioConfig) -> None:
    db = open_database(tmp_path / "h.db")
    try:
        sid = db.upsert_scenario(json.loads(scenario.model_dump_json()))
        cid = db.create_campaign("hydrate_test", sid)
        accum1 = KnowledgeAccumulator(db=db, campaign_id=cid)
        e = _make_entry(power=200.0)
        accum1.put(e.chromosome.hash_id(), e, generation=0)
    finally:
        db.close()
    db2 = open_database(tmp_path / "h.db")
    try:
        row = db2.get_campaign_by_name("hydrate_test")
        assert row is not None
        accum2 = KnowledgeAccumulator(db=db2, campaign_id=row.campaign_id)
        assert len(accum2) == 1
        hit = accum2.get(e.chromosome.hash_id())
        assert hit is not None
        assert hit.scalar_J == pytest.approx(0.25)
        assert hit.chromosome.power_W == pytest.approx(200.0)
    finally:
        db2.close()


def test_calibration_persistence(tmp_path: Path, scenario: ScenarioConfig) -> None:
    db = open_database(tmp_path / "cal.db")
    try:
        sid = db.upsert_scenario(json.loads(scenario.model_dump_json()))
        posterior = {"absorptivity": 0.55, "spot_um": 110.0, "emissivity": 0.45}
        cid = db.insert_calibration(scenario_id=sid, posterior=posterior)
        assert cid
        latest = db.latest_calibration(sid)
        assert latest == posterior
    finally:
        db.close()


def test_engine_with_persistent_accumulator(tmp_path: Path, scenario: ScenarioConfig) -> None:
    db = open_database(tmp_path / "engine.db")
    try:
        sid = db.upsert_scenario(json.loads(scenario.model_dump_json()))
        cid = db.create_campaign("engine_test", sid)
        accum = KnowledgeAccumulator(db=db, campaign_id=cid)
        ea = scenario.ea.model_copy(
            update={"population": 6, "generations": 2, "elitism": 1, "seed": 5}
        )
        committed_generations: list[int] = []
        eng = GeneticEngine(
            material=scenario.material,
            machine=scenario.machine,
            roi=scenario.roi,
            ea=ea,
            cache=accum,
            on_commit=lambda g, _entries: (
                accum.commit_archive(_entries, generation=g),
                committed_generations.append(g),
            )[0],
        )
        log = eng.run()
        assert len(log.history) == 2
        assert committed_generations == [0, 1]
        # archive was written to disk
        rows = db.list_archive(cid)
        assert len(rows) >= 1
        # candidates and runs were written
        n_cand = db.conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        n_runs = db.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        assert n_cand > 0
        assert n_runs > 0
    finally:
        db.close()
