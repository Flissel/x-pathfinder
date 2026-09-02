"""Schema init must complete on an empty database and create every table."""
import os

import psycopg2
import pytest

DSN = os.environ.get(
    "DATABASE_URL", "postgresql://pathfinder:pathfinder@127.0.0.1:5434/emails"
)
EXPECTED_TABLES = {"accounts", "emails", "strategies", "runs"}


@pytest.fixture()
def empty_schema():
    """Drop the four tables so _init_db runs against a truly empty database."""
    conn = psycopg2.connect(DSN)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS emails, accounts, strategies, runs CASCADE;")
    yield conn
    conn.close()


def test_init_db_creates_all_tables_on_empty_database(empty_schema):
    from x_pathfinder.database import EmailDatabase

    db = EmailDatabase(dsn=DSN)
    try:
        with psycopg2.connect(DSN) as check, check.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public';"
            )
            tables = {row[0] for row in cur.fetchall()}
        assert EXPECTED_TABLES <= tables
    finally:
        db.close()


def test_emails_table_has_country_column(empty_schema):
    """The country index needs this column; if it is missing init rolls back."""
    from x_pathfinder.database import EmailDatabase

    db = EmailDatabase(dsn=DSN)
    try:
        with psycopg2.connect(DSN) as check, check.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'emails';"
            )
            columns = {row[0] for row in cur.fetchall()}
        assert "country" in columns
    finally:
        db.close()
