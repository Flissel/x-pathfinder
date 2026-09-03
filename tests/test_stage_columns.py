import os

import psycopg2
import pytest

from x_pathfinder.database import EmailDatabase
from x_pathfinder.models import XAccount

# tests/conftest.py guarantees DATABASE_URL points at a throwaway database.
# There is deliberately NO fallback: the fixture below drops tables, and a
# default that resolved to the production stage store would wipe it.
DSN = os.environ["DATABASE_URL"]
STAGE_COLUMNS = {
    "fitness_score", "fitness_source", "signals", "evidence_urls",
    "validated", "verdict_reason", "validated_at", "promoted_at",
}


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


def _columns(table: str) -> set:
    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = %s;",
            (table,),
        )
        return {row[0] for row in cur.fetchall()}


def test_accounts_table_has_stage_columns(db):
    assert STAGE_COLUMNS <= _columns("accounts")


def test_emails_table_has_stage_columns(db):
    assert STAGE_COLUMNS <= _columns("emails")


def test_save_scored_accounts_persists_everything_unvalidated(db):
    """The stage DB is unfiltered: even a zero-fitness account is stored."""
    accounts = [
        XAccount(handle="strong", niche="ai", fitness_score=90.0,
                 fitness_source="composite", evidence_urls=["https://a.example"]),
        XAccount(handle="weak", niche="ai", fitness_score=0.0,
                 fitness_source="unscored"),
    ]
    written = db.save_scored_accounts(accounts)
    assert written == 2

    rows = {row["handle"]: row for row in db.get_unvalidated()}
    assert set(rows) == {"strong", "weak"}
    assert rows["strong"]["fitness_score"] == 90.0
    assert rows["strong"]["evidence_urls"] == ["https://a.example"]
    assert rows["strong"]["validated"] is None      # not yet checked
    assert rows["weak"]["fitness_source"] == "unscored"


def test_signals_are_persisted_not_written_as_an_empty_object(db):
    """save_scored_accounts used to hardcode json.dumps({}), so the
    evidence behind a score was thrown away at the stage boundary and a
    promotion decision could never be audited after the fact."""
    db.save_scored_accounts([
        XAccount(handle="acme", niche="ai", fitness_score=82.0,
                 fitness_source="composite",
                 signals={"handle_wellformed": True,
                          "profile_resolves": True,
                          "reason": "series A announced"},
                 evidence_urls=["https://news.example/acme"]),
    ])

    row = db.get_unvalidated()[0]
    assert row["signals"] == {
        "handle_wellformed": True,
        "profile_resolves": True,
        "reason": "series A announced",
    }


def test_stage_row_carries_its_niche_back_out(db):
    """ResearchScorer reads the niche off the batch and falls back to
    'general'; get_unvalidated() did not select the column at all, so every
    xpf_score researched against a generic niche, silently."""
    db.save_scored_accounts([XAccount(handle="acme", niche="security")])

    assert db.get_unvalidated()[0]["niche"] == "security"

    db.record_verdict("acme", True, "confirmed")
    assert db.get_validated_unpromoted()[0]["niche"] == "security"


def test_rediscovery_preserves_the_audit_bundle(db):
    """A discovery run must not overwrite what a scoring run established.

    AccountDiscoverer.run() returns every account seen in the session, not
    only the new ones, so re-running xpf_discover restages already-validated
    rows. Routed through save_scored_accounts that reset the verdict and
    replaced evidence_urls/signals with deterministic placeholders -- fail-
    closed, but it destroys the audit trail and strands a confirmed row.
    The discovery path refreshes only what discovery actually observed.
    """
    db.save_scored_accounts([
        XAccount(handle="acme", niche="ai", display_name="Acme Inc",
                 followers=500, fitness_score=88.0, fitness_source="composite",
                 signals={"reason": "series A confirmed"},
                 evidence_urls=["https://news.example/acme-series-a"]),
    ])
    db.record_verdict("acme", True, "claim confirmed at https://news.example/acme-series-a")

    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT validated_at FROM accounts WHERE handle = 'acme';")
        validated_at_before = cur.fetchone()[0]
    assert validated_at_before is not None

    # Rediscovery: same handle, fresh discovery metadata, no evidence and
    # only the cheap deterministic score the GA could produce without one.
    written = db.save_discovered_accounts([
        XAccount(handle="acme", niche="ai", display_name="Acme Incorporated",
                 followers=512, fitness_score=40.0,
                 fitness_source="deterministic",
                 signals={"profile_resolves": True}, evidence_urls=[]),
    ])
    assert written == 1

    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT validated, verdict_reason, validated_at, evidence_urls, "
            "signals, fitness_score, fitness_source, display_name, followers "
            "FROM accounts WHERE handle = 'acme';"
        )
        (validated, reason, validated_at, evidence, signals, score, source,
         display_name, followers) = cur.fetchone()

    # The audit bundle survives, untouched.
    assert validated is True
    assert reason == "claim confirmed at https://news.example/acme-series-a"
    assert validated_at == validated_at_before
    assert evidence == ["https://news.example/acme-series-a"]
    assert signals == {"reason": "series A confirmed"}
    assert score == 88.0
    assert source == "composite"
    # The row is still promotable -- rediscovery did not strand it.
    assert [row["handle"] for row in db.get_validated_unpromoted()] == ["acme"]

    # But what discovery genuinely re-observed DID update.
    assert display_name == "Acme Incorporated"
    assert followers == 512


def test_discovery_still_inserts_genuinely_new_accounts(db):
    """A newly discovered account is legitimately unscored and unvalidated."""
    written = db.save_discovered_accounts([
        XAccount(handle="newcomer", niche="ai", display_name="Newcomer",
                 followers=7),
    ])
    assert written == 1

    rows = {row["handle"]: row for row in db.get_unvalidated()}
    assert "newcomer" in rows
    assert rows["newcomer"]["validated"] is None
    assert rows["newcomer"]["evidence_urls"] == []
    assert rows["newcomer"]["fitness_source"] == "unscored"
    assert rows["newcomer"]["niche"] == "ai"


def test_scoring_path_still_resets_verdicts_after_the_discovery_split(db):
    """Regression guard: the discovery/scoring split must not soften FIX 1.

    save_scored_accounts is the path where new evidence arrives, and a
    verdict reached against the OLD evidence must not survive it. If someone
    ever makes the scoring path non-resetting too -- e.g. by pointing
    _tool_score at save_discovered_accounts, or by copying its ON CONFLICT
    list over -- this fails.
    """
    db.save_scored_accounts([
        XAccount(handle="acme", niche="ai", fitness_score=88.0,
                 fitness_source="composite",
                 signals={"reason": "series A confirmed"},
                 evidence_urls=["https://good.example"]),
    ])
    db.record_verdict("acme", True, "claim confirmed at https://good.example")

    db.save_scored_accounts([
        XAccount(handle="acme", niche="ai", fitness_score=91.0,
                 fitness_source="composite",
                 signals={"reason": "re-researched"},
                 evidence_urls=["https://404.example"]),
    ])

    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT validated, verdict_reason, validated_at, evidence_urls "
            "FROM accounts WHERE handle = 'acme';"
        )
        validated, reason, validated_at, evidence = cur.fetchone()

    assert validated is None, "the scoring path must still invalidate verdicts"
    assert reason is None
    assert validated_at is None
    assert evidence == ["https://404.example"]
    assert db.get_validated_unpromoted() == []


def test_rescoring_invalidates_a_previous_verdict(db):
    """A verdict must not survive the evidence it was based on.

    validate -> re-score -> promote used to ship a row still marked
    validated=TRUE whose evidence_urls had been overwritten (possibly with
    a URL that 404s) and whose verdict_reason cited a URL no longer in the
    row. The validator never saw the new evidence, so the row must fall
    back to "not yet checked" (NULL), not stay confirmed.
    """
    db.save_scored_accounts([
        XAccount(handle="acme", niche="ai", fitness_score=90.0,
                 fitness_source="composite",
                 evidence_urls=["https://good.example"]),
    ])
    db.record_verdict("acme", True, "claim confirmed at https://good.example")
    assert [row["handle"] for row in db.get_validated_unpromoted()] == ["acme"]

    # Re-score: same handle, different (unverified) evidence.
    db.save_scored_accounts([
        XAccount(handle="acme", niche="ai", fitness_score=91.0,
                 fitness_source="composite",
                 evidence_urls=["https://404.example"]),
    ])

    with psycopg2.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT validated, verdict_reason, validated_at, evidence_urls "
            "FROM accounts WHERE handle = 'acme';"
        )
        validated, reason, validated_at, evidence = cur.fetchone()

    assert validated is None, "re-scored row must read as not-yet-checked"
    assert reason is None
    assert validated_at is None
    assert evidence == ["https://404.example"]
    # And therefore it is no longer promotable until re-validated.
    assert db.get_validated_unpromoted() == []
    assert [row["handle"] for row in db.get_unvalidated()] == ["acme"]


def test_rescoring_also_clears_a_refutation(db):
    """The reset is symmetric: FALSE is a verdict too, and it was reached
    against evidence this re-score has just replaced."""
    db.save_scored_accounts([XAccount(handle="ghost", niche="ai")])
    db.record_verdict("ghost", False, "no evidence urls to check")

    db.save_scored_accounts([
        XAccount(handle="ghost", niche="ai", fitness_score=70.0,
                 fitness_source="composite",
                 evidence_urls=["https://found-later.example"]),
    ])

    rows = {row["handle"]: row for row in db.get_unvalidated()}
    assert "ghost" in rows, "a re-scored row must be re-examinable"
    assert rows["ghost"]["validated"] is None
    assert rows["ghost"]["evidence_urls"] == ["https://found-later.example"]


def test_get_validated_unpromoted_returns_only_validated_not_promoted(db):
    """The promotion gate needs rows the validator confirmed and that have
    not been promoted yet — the opposite selection of get_unvalidated().
    Reusing get_unvalidated() here would silently promote nothing forever.
    """
    accounts = [
        XAccount(handle="confirmed", niche="ai", fitness_score=80.0,
                 fitness_source="composite", evidence_urls=["https://b.example"]),
        XAccount(handle="still_pending", niche="ai", fitness_score=50.0,
                 fitness_source="composite"),
    ]
    written = db.save_scored_accounts(accounts)
    assert written == 2

    conn = db._get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE accounts SET validated = TRUE, verdict_reason = %s "
            "WHERE handle = %s",
            ("looks legit", "confirmed"),
        )
    conn.commit()

    rows = db.get_validated_unpromoted()
    assert [row["handle"] for row in rows] == ["confirmed"]

    row = rows[0]
    assert row["fitness_score"] == 80.0
    assert row["fitness_source"] == "composite"
    assert row["evidence_urls"] == ["https://b.example"]
    assert row["validated"] is True
    assert row["verdict_reason"] == "looks legit"
    assert row["promoted_at"] is None
