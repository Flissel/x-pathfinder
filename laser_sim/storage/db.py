"""SQLite-backed persistence for campaigns, candidates, runs, metrics, archive.

Uses stdlib sqlite3 — zero new dependencies. The schema mirrors the
Postgres/Parquet layout from the plan so a later migration to the
production store is mechanical (rename, swap connection, keep queries).

Schema (versioned via the `schema_meta` table):

  scenarios          versioned ScenarioConfig snapshots
  campaigns          one optimization run with a fixed scenario + objectives
  candidates         chromosomes evaluated in any campaign
  runs               one evaluation of a candidate (fast / hf / experiment)
  metrics            per-run scalar metrics (long format)
  archive_entries    Pareto archive snapshot, keyed by campaign + candidate
  calibrations       calibration posterior per scenario, with valid_from
  measurements       ingested real-world artefacts (thermal/pyrometer/...)
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA_VERSION = 1


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scenarios (
    scenario_id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL,
    machine_id TEXT NOT NULL,
    scenario_json TEXT NOT NULL,
    objective_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    scenario_id TEXT NOT NULL REFERENCES scenarios(scenario_id),
    last_generation INTEGER DEFAULT 0,
    notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(name)
);

CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
    chromosome_hash TEXT NOT NULL,
    chromosome_json TEXT NOT NULL,
    generation INTEGER,
    parent_ids TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(campaign_id, chromosome_hash)
);
CREATE INDEX IF NOT EXISTS idx_candidates_campaign ON candidates(campaign_id);
CREATE INDEX IF NOT EXISTS idx_candidates_hash ON candidates(chromosome_hash);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id),
    fidelity_tier TEXT NOT NULL,
    solver_name TEXT NOT NULL,
    solver_version TEXT,
    status TEXT NOT NULL,
    fitness_json TEXT NOT NULL,
    scalar_J REAL,
    walltime_s REAL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_candidate ON runs(candidate_id);

CREATE TABLE IF NOT EXISTS metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    metric_name TEXT NOT NULL,
    metric_value REAL,
    metric_unit TEXT
);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id);

CREATE TABLE IF NOT EXISTS archive_entries (
    campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
    candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id),
    fitness_json TEXT NOT NULL,
    scalar_J REAL,
    generation INTEGER,
    PRIMARY KEY (campaign_id, candidate_id)
);

CREATE TABLE IF NOT EXISTS calibrations (
    calibration_id TEXT PRIMARY KEY,
    scenario_id TEXT NOT NULL REFERENCES scenarios(scenario_id),
    posterior_json TEXT NOT NULL,
    fit_report_json TEXT,
    valid_from TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS measurements (
    measurement_id TEXT PRIMARY KEY,
    scenario_id TEXT,
    experiment_id TEXT NOT NULL,
    sensor_type TEXT NOT NULL,
    file_uri TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL
);
"""


def _utcnow() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass(frozen=True)
class CampaignRow:
    campaign_id: str
    name: str
    scenario_id: str
    last_generation: int
    created_at: str
    updated_at: str
    notes: str | None = None


@dataclass(frozen=True)
class CandidateRow:
    candidate_id: str
    campaign_id: str
    chromosome_hash: str
    chromosome_json: str
    generation: int | None
    parent_ids: str | None
    created_at: str


@dataclass(frozen=True)
class RunRow:
    run_id: str
    candidate_id: str
    fidelity_tier: str
    solver_name: str
    status: str
    fitness_json: str
    scalar_J: float | None
    walltime_s: float | None
    created_at: str
    solver_version: str | None = None


@dataclass(frozen=True)
class ArchiveRow:
    campaign_id: str
    candidate_id: str
    fitness_json: str
    scalar_J: float | None
    generation: int | None


class Database:
    """Thin wrapper around a sqlite3 connection. Auto-creates schema."""

    def __init__(self, path: str | Path | None = ":memory:") -> None:
        self.path = str(path) if path is not None else ":memory:"
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._ensure_schema()

    def close(self) -> None:
        try:
            self.conn.close()
        except sqlite3.ProgrammingError:
            pass

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _ensure_schema(self) -> None:
        with contextlib.closing(self.conn.cursor()) as cur:
            cur.executescript(_SCHEMA_SQL)
        version_row = self.conn.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()
        if version_row is None:
            self.conn.execute(
                "INSERT INTO schema_meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
        elif int(version_row["value"]) != SCHEMA_VERSION:
            raise RuntimeError(
                f"db schema version {version_row['value']} != code {SCHEMA_VERSION}; "
                "migration not implemented"
            )
        self.conn.commit()

    # --- scenarios -----------------------------------------------------

    def upsert_scenario(self, scenario_json: dict[str, Any]) -> str:
        """Insert the scenario JSON if absent; return its scenario_id."""
        scenario_id = scenario_json.get("scenario_id") or str(uuid.uuid4())
        self.conn.execute(
            """
            INSERT OR REPLACE INTO scenarios
            (scenario_id, material_id, machine_id, scenario_json, objective_version, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                scenario_id,
                scenario_json.get("material", {}).get("material_id", "unknown"),
                scenario_json.get("machine", {}).get("machine_id", "unknown"),
                json.dumps(scenario_json, sort_keys=False),
                scenario_json.get("objective_version", "v1"),
                _utcnow(),
            ),
        )
        self.conn.commit()
        return scenario_id

    # --- campaigns -----------------------------------------------------

    def create_campaign(
        self, name: str, scenario_id: str, notes: str | None = None
    ) -> str:
        campaign_id = str(uuid.uuid4())
        now = _utcnow()
        try:
            self.conn.execute(
                """
                INSERT INTO campaigns
                (campaign_id, name, scenario_id, last_generation, notes, created_at, updated_at)
                VALUES (?, ?, ?, 0, ?, ?, ?)
                """,
                (campaign_id, name, scenario_id, notes, now, now),
            )
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            raise ValueError(f"campaign name {name!r} already exists") from e
        return campaign_id

    def get_campaign_by_name(self, name: str) -> CampaignRow | None:
        row = self.conn.execute(
            "SELECT * FROM campaigns WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            return None
        return CampaignRow(
            campaign_id=row["campaign_id"],
            name=row["name"],
            scenario_id=row["scenario_id"],
            last_generation=row["last_generation"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            notes=row["notes"],
        )

    def update_last_generation(self, campaign_id: str, generation: int) -> None:
        self.conn.execute(
            "UPDATE campaigns SET last_generation = ?, updated_at = ? WHERE campaign_id = ?",
            (generation, _utcnow(), campaign_id),
        )
        self.conn.commit()

    def list_campaigns(self) -> list[CampaignRow]:
        return [
            CampaignRow(
                campaign_id=r["campaign_id"],
                name=r["name"],
                scenario_id=r["scenario_id"],
                last_generation=r["last_generation"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                notes=r["notes"],
            )
            for r in self.conn.execute(
                "SELECT * FROM campaigns ORDER BY updated_at DESC"
            ).fetchall()
        ]

    # --- candidates ----------------------------------------------------

    def upsert_candidate(
        self,
        campaign_id: str,
        chromosome_hash: str,
        chromosome_json: dict[str, Any],
        generation: int | None = None,
    ) -> str:
        existing = self.conn.execute(
            "SELECT candidate_id FROM candidates WHERE campaign_id = ? AND chromosome_hash = ?",
            (campaign_id, chromosome_hash),
        ).fetchone()
        if existing is not None:
            return existing["candidate_id"]
        candidate_id = str(uuid.uuid4())
        self.conn.execute(
            """
            INSERT INTO candidates
            (candidate_id, campaign_id, chromosome_hash, chromosome_json, generation, parent_ids, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate_id,
                campaign_id,
                chromosome_hash,
                json.dumps(chromosome_json),
                generation,
                None,
                _utcnow(),
            ),
        )
        self.conn.commit()
        return candidate_id

    def list_candidates(self, campaign_id: str) -> list[CandidateRow]:
        return [
            CandidateRow(
                candidate_id=r["candidate_id"],
                campaign_id=r["campaign_id"],
                chromosome_hash=r["chromosome_hash"],
                chromosome_json=r["chromosome_json"],
                generation=r["generation"],
                parent_ids=r["parent_ids"],
                created_at=r["created_at"],
            )
            for r in self.conn.execute(
                "SELECT * FROM candidates WHERE campaign_id = ?", (campaign_id,)
            ).fetchall()
        ]

    # --- runs / metrics -----------------------------------------------

    def insert_run(
        self,
        candidate_id: str,
        fidelity_tier: str,
        solver_name: str,
        fitness: tuple[float, ...],
        scalar_J: float,
        status: str = "ok",
        walltime_s: float | None = None,
        solver_version: str | None = None,
        metrics: dict[str, float] | None = None,
    ) -> str:
        run_id = str(uuid.uuid4())
        self.conn.execute(
            """
            INSERT INTO runs
            (run_id, candidate_id, fidelity_tier, solver_name, solver_version, status,
             fitness_json, scalar_J, walltime_s, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                candidate_id,
                fidelity_tier,
                solver_name,
                solver_version,
                status,
                json.dumps(list(fitness)),
                scalar_J,
                walltime_s,
                _utcnow(),
            ),
        )
        if metrics:
            self.conn.executemany(
                "INSERT INTO metrics(run_id, metric_name, metric_value, metric_unit) VALUES (?, ?, ?, ?)",
                [(run_id, k, float(v), None) for k, v in metrics.items()],
            )
        self.conn.commit()
        return run_id

    # --- archive ------------------------------------------------------

    def replace_archive(
        self, campaign_id: str, entries: Iterable[tuple[str, tuple[float, ...], float, int | None]]
    ) -> None:
        """Atomically replace the Pareto archive for a campaign."""
        self.conn.execute("DELETE FROM archive_entries WHERE campaign_id = ?", (campaign_id,))
        self.conn.executemany(
            """
            INSERT INTO archive_entries
            (campaign_id, candidate_id, fitness_json, scalar_J, generation)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (campaign_id, cid, json.dumps(list(fit)), sj, gen)
                for cid, fit, sj, gen in entries
            ],
        )
        self.conn.commit()

    def list_archive(self, campaign_id: str) -> list[ArchiveRow]:
        return [
            ArchiveRow(
                campaign_id=r["campaign_id"],
                candidate_id=r["candidate_id"],
                fitness_json=r["fitness_json"],
                scalar_J=r["scalar_J"],
                generation=r["generation"],
            )
            for r in self.conn.execute(
                "SELECT * FROM archive_entries WHERE campaign_id = ? ORDER BY scalar_J ASC",
                (campaign_id,),
            ).fetchall()
        ]

    # --- calibrations -------------------------------------------------

    def insert_calibration(
        self,
        scenario_id: str,
        posterior: dict[str, Any],
        fit_report: dict[str, Any] | None = None,
        valid_from: str | None = None,
    ) -> str:
        cid = str(uuid.uuid4())
        self.conn.execute(
            """
            INSERT INTO calibrations
            (calibration_id, scenario_id, posterior_json, fit_report_json, valid_from, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                cid,
                scenario_id,
                json.dumps(posterior),
                json.dumps(fit_report) if fit_report else None,
                valid_from,
                _utcnow(),
            ),
        )
        self.conn.commit()
        return cid

    def latest_calibration(self, scenario_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT posterior_json FROM calibrations
            WHERE scenario_id = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (scenario_id,),
        ).fetchone()
        return json.loads(row["posterior_json"]) if row else None


def open_database(path: str | Path | None = None) -> Database:
    return Database(path=path)
