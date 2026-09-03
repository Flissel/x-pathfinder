"""Session-wide database isolation for the test suite.

Three test modules drop and recreate the stage tables
(``DROP TABLE IF EXISTS emails, accounts, strategies, runs CASCADE``).
Without this file they resolve their DSN to the same default that
``x_pathfinder/database.py`` and ``x_pathfinder/mcp_server.py`` use, i.e.
the real stage store — so simply running ``pytest`` would wipe every
staged candidate.

This module runs at collection time, before any test module is imported,
and points the whole session at a SEPARATE database, creating it on first
use. An explicitly set ``DATABASE_URL`` is left alone: that is the
developer's deliberate choice. What must never happen is the *default*
silently being production.
"""

import logging
import os

import psycopg2
from psycopg2 import errorcodes, sql

logger = logging.getLogger(__name__)

TEST_DB_NAME = "emails_test"
TEST_DSN = f"postgresql://pathfinder:pathfinder@127.0.0.1:5434/{TEST_DB_NAME}"
# Creating a database cannot happen from inside that database, so the
# CREATE runs against the always-present maintenance database.
ADMIN_DSN = "postgresql://pathfinder:pathfinder@127.0.0.1:5434/postgres"


def _create_test_database_if_missing() -> None:
    try:
        conn = psycopg2.connect(ADMIN_DSN)
    except psycopg2.Error as exc:
        # No Postgres reachable: the DB-touching tests will fail loudly on
        # their own. Do not take the rest of the suite down with them.
        logger.warning("could not reach postgres to create %s: %s", TEST_DB_NAME, exc)
        return

    conn.autocommit = True  # CREATE DATABASE cannot run inside a transaction
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(TEST_DB_NAME))
            )
    except psycopg2.Error as exc:
        if getattr(exc, "pgcode", None) != errorcodes.DUPLICATE_DATABASE:
            raise
    finally:
        conn.close()


if "DATABASE_URL" not in os.environ:
    os.environ["DATABASE_URL"] = TEST_DSN
    _create_test_database_if_missing()
