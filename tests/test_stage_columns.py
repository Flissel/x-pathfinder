import os

import psycopg2
import pytest

from x_pathfinder.database import EmailDatabase
from x_pathfinder.models import XAccount

DSN = os.environ.get(
    "DATABASE_URL", "postgresql://pathfinder:pathfinder@127.0.0.1:5434/emails"
)
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
