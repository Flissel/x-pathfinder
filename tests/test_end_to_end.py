# tests/test_end_to_end.py
"""Proves the pipeline end to end with stubbed network, against the real stage DB."""
import os

import psycopg2
import pytest

from x_pathfinder.composite_scorer import CompositeScorer
from x_pathfinder.database import EmailDatabase
from x_pathfinder.fitness_providers import DeterministicScorer
from x_pathfinder.models import XAccount
from x_pathfinder.promotion import PromotionGate
from x_pathfinder.research_scorer import ResearchScorer
from x_pathfinder.validator import EvidenceValidator

# tests/conftest.py guarantees DATABASE_URL points at a throwaway database.
# There is deliberately NO fallback: the fixture below drops tables, and a
# default that resolved to the production stage store would wipe it.
DSN = os.environ["DATABASE_URL"]


class _CollectingWriter:
    def __init__(self):
        self.rows = []

    def insert(self, table, row):
        self.rows.append(row)


@pytest.fixture()
def db():
    conn = psycopg2.connect(DSN)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS emails, accounts, strategies, runs CASCADE;")
    conn.close()
    database = EmailDatabase(dsn=DSN)
    yield database
    database.close()


def test_validate_before_score_does_not_permanently_kill_a_candidate(db, monkeypatch):
    """The wrong tool ordering must be recoverable, not fatal.

    An email-daemon row carries no evidence_urls. Running xpf_validate on
    it first used to write validated=FALSE, and get_unvalidated() selects
    validated IS NULL, so the row was excluded from the validator forever —
    xpf_score afterwards could never redeem it. Two changes make the
    ordering survivable: _tool_validate skips evidence-less rows, and
    save_scored_accounts resets the verdict on re-score.
    """
    from x_pathfinder import mcp_server
    from x_pathfinder import validator as validator_module

    monkeypatch.setattr(mcp_server, "_database", lambda: db)
    monkeypatch.setattr(db, "close", lambda: None)  # fixture owns the handle
    real_validator = validator_module.EvidenceValidator
    monkeypatch.setattr(
        validator_module,
        "EvidenceValidator",
        lambda *a, **k: real_validator(
            fetcher=lambda url: (200, "daemonco raised a seed round")
        ),
    )

    # The email daemon leaves a bare row behind: no scoring, no evidence.
    db.add_account("daemonco", niche="ai")

    # WRONG ORDER: validate first.
    assert mcp_server._tool_validate() == {
        "checked": 0, "confirmed": 0, "skipped_no_evidence": 1,
    }
    assert [row["handle"] for row in db.get_unvalidated()] == ["daemonco"]

    # Now score it for real, then validate again.
    db.save_scored_accounts([
        XAccount(handle="daemonco", niche="ai", fitness_score=80.0,
                 fitness_source="composite",
                 evidence_urls=["https://news.example/daemonco"]),
    ])
    assert mcp_server._tool_validate() == {
        "checked": 1, "confirmed": 1, "skipped_no_evidence": 0,
    }
    assert [row["handle"] for row in db.get_validated_unpromoted()] == ["daemonco"]


def test_only_the_candidate_with_real_evidence_reaches_supabase(db):
    candidates = [XAccount(handle="realco", niche="ai"),
                  XAccount(handle="fakeco", niche="ai")]

    research_payload = (
        '{"results": ['
        '{"handle": "realco", "score": 90, "evidence_urls": ["https://real.example"]},'
        '{"handle": "fakeco", "score": 95, "evidence_urls": ["https://dead.example"]}'
        "]}"
    )
    scorer = CompositeScorer(
        DeterministicScorer(resolver=lambda url: 200),
        ResearchScorer(transport=lambda message: research_payload),
    )
    results = scorer.score_batch(candidates)
    for candidate in candidates:
        result = results[candidate.handle.lower()]
        candidate.fitness_score = 0.0 if result.score is None else result.score
        candidate.fitness_source = result.source
        candidate.evidence_urls = list(result.evidence_urls)

    assert db.save_scored_accounts(candidates) == 2

    def _fetch(url):
        if url == "https://real.example":
            return 200, "realco raised a round"
        return 404, ""

    validator = EvidenceValidator(fetcher=_fetch)
    for row in db.get_unvalidated():
        verdict = validator.validate(
            row["handle"], row.get("evidence_urls") or [], [row["handle"]]
        )
        db.record_verdict(row["handle"], verdict.validated, verdict.reason)

    writer = _CollectingWriter()
    rows = []
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT handle, validated, promoted_at FROM accounts ORDER BY handle;"
        )
        rows = [
            {"handle": r[0], "validated": r[1], "promoted_at": r[2]}
            for r in cur.fetchall()
        ]

    summary = PromotionGate(db, writer).promote(rows)

    assert summary == {"promoted": 1, "skipped": 1}
    assert [row["handle"] for row in writer.rows] == ["realco"]
