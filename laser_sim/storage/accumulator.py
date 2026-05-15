"""KnowledgeAccumulator: persistent eval cache + archive snapshot.

Mirrors the architecture pattern from x-pathfinder/knowledge_accumulator.py:
elite-retention + per-chromosome hash cache, so re-running the EA on the
same scenario reuses prior evaluations instead of recomputing them. Backed
by Database when one is provided; otherwise pure in-memory.
"""

from __future__ import annotations

import json
from typing import Iterable

from laser_sim.genome.archive import ArchiveEntry
from laser_sim.genome.chromosome import Chromosome
from laser_sim.patterns.base import PrimitiveKind
from laser_sim.storage.db import Database


def _chromosome_to_json(c: Chromosome) -> dict[str, object]:
    return {
        "kind": c.primitive_kind.value,
        "power_W": float(c.power_W),
        "speed_mm_s": float(c.speed_mm_s),
        "hatch_um": float(c.hatch_um),
        "spot_um": float(c.spot_um),
        "layer_rotation_deg": float(c.layer_rotation_deg),
        "extras": dict(c.extras),
    }


def _chromosome_from_json(d: dict[str, object]) -> Chromosome:
    return Chromosome(
        primitive_kind=PrimitiveKind(d["kind"]),
        power_W=float(d["power_W"]),
        speed_mm_s=float(d["speed_mm_s"]),
        hatch_um=float(d["hatch_um"]),
        spot_um=float(d["spot_um"]),
        layer_rotation_deg=float(d.get("layer_rotation_deg", 0.0)),
        extras=dict(d.get("extras", {})),
    )


class KnowledgeAccumulator:
    """Eval cache + archive snapshot, optionally backed by a Database.

    Behaves as a dict-like for the engine's `_cache`: get(hash), put(hash, entry).
    Records every evaluation as a Run and (when committed) refreshes the
    Pareto archive table in atomic replace mode.
    """

    def __init__(
        self,
        db: Database | None = None,
        campaign_id: str | None = None,
        solver_name: str = "eagar_tsai_proxy",
        solver_version: str = "0.0.1",
    ) -> None:
        self.db = db
        self.campaign_id = campaign_id
        self.solver_name = solver_name
        self.solver_version = solver_version
        self._memory: dict[str, ArchiveEntry] = {}
        if db is not None and campaign_id is not None:
            self._hydrate_from_db()

    # --- dict-like API used by GeneticEngine -------------------------

    def __contains__(self, key: str) -> bool:
        return key in self._memory

    def __len__(self) -> int:
        return len(self._memory)

    def get(self, key: str) -> ArchiveEntry | None:
        return self._memory.get(key)

    def put(
        self,
        key: str,
        entry: ArchiveEntry,
        *,
        generation: int | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        self._memory[key] = entry
        if self.db is None or self.campaign_id is None:
            return
        candidate_id = self.db.upsert_candidate(
            campaign_id=self.campaign_id,
            chromosome_hash=key,
            chromosome_json=_chromosome_to_json(entry.chromosome),
            generation=generation,
        )
        self.db.insert_run(
            candidate_id=candidate_id,
            fidelity_tier="fast",
            solver_name=self.solver_name,
            solver_version=self.solver_version,
            fitness=entry.fitness.values,
            scalar_J=entry.scalar_J,
            metrics=metrics,
        )

    # --- archive snapshot --------------------------------------------

    def commit_archive(
        self, archive_entries: Iterable[ArchiveEntry], generation: int | None = None
    ) -> None:
        """Replace the archive snapshot for the current campaign."""
        if self.db is None or self.campaign_id is None:
            return
        entries = list(archive_entries)
        rows: list[tuple[str, tuple[float, ...], float, int | None]] = []
        for e in entries:
            cid = self.db.upsert_candidate(
                campaign_id=self.campaign_id,
                chromosome_hash=e.chromosome.hash_id(),
                chromosome_json=_chromosome_to_json(e.chromosome),
                generation=generation,
            )
            rows.append((cid, e.fitness.values, e.scalar_J, generation))
        self.db.replace_archive(self.campaign_id, rows)
        if generation is not None:
            self.db.update_last_generation(self.campaign_id, generation)

    # --- hydration ----------------------------------------------------

    def _hydrate_from_db(self) -> None:
        assert self.db is not None and self.campaign_id is not None
        from laser_sim.fitness.evaluator import FitnessVector

        candidates = self.db.list_candidates(self.campaign_id)
        for cand in candidates:
            row = self.db.conn.execute(
                """
                SELECT fitness_json, scalar_J FROM runs
                WHERE candidate_id = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (cand.candidate_id,),
            ).fetchone()
            if row is None:
                continue
            fit_values = tuple(float(v) for v in json.loads(row["fitness_json"]))
            chrom = _chromosome_from_json(json.loads(cand.chromosome_json))
            self._memory[cand.chromosome_hash] = ArchiveEntry(
                chromosome=chrom,
                fitness=FitnessVector(values=fit_values),
                scalar_J=float(row["scalar_J"] if row["scalar_J"] is not None else 0.0),
            )

    def memory_snapshot(self) -> list[ArchiveEntry]:
        return list(self._memory.values())
