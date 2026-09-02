"""
PostgreSQL database for email storage.

Tables:
- accounts: discovered X handles with metadata
- emails: generated/verified email candidates
- strategies: successful email patterns
- runs: session history

Connection: postgresql://pathfinder:pathfinder@localhost:5434/emails
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

DEFAULT_DSN = "postgresql://pathfinder:pathfinder@localhost:5434/emails"


class EmailDatabase:
    """PostgreSQL database for mass email storage."""

    def __init__(self, dsn: str = None):
        self.dsn = dsn or os.environ.get("DATABASE_URL", DEFAULT_DSN)
        self._conn = None
        self._init_db()

    def _get_conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.dsn)
            self._conn.autocommit = True
        return self._conn

    def _init_db(self):
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS accounts (
                    handle TEXT PRIMARY KEY,
                    display_name TEXT DEFAULT '',
                    bio TEXT DEFAULT '',
                    followers INTEGER DEFAULT 0,
                    niche TEXT DEFAULT '',
                    source TEXT DEFAULT '',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS emails (
                    email TEXT PRIMARY KEY,
                    handle TEXT REFERENCES accounts(handle) ON DELETE CASCADE,
                    confidence REAL DEFAULT 0.0,
                    mx_valid BOOLEAN DEFAULT FALSE,
                    smtp_valid SMALLINT DEFAULT -1,
                    strategy_id TEXT DEFAULT '',
                    domain TEXT DEFAULT '',
                    country TEXT DEFAULT 'XX',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS strategies (
                    id TEXT PRIMARY KEY,
                    format_pattern TEXT,
                    domain TEXT,
                    fitness REAL DEFAULT 0.0,
                    success_count INTEGER DEFAULT 0,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS runs (
                    id SERIAL PRIMARY KEY,
                    started_at TIMESTAMPTZ,
                    ended_at TIMESTAMPTZ,
                    accounts_processed INTEGER DEFAULT 0,
                    emails_generated INTEGER DEFAULT 0,
                    emails_verified INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'running'
                );

                ALTER TABLE accounts ADD COLUMN IF NOT EXISTS fitness_score REAL DEFAULT 0.0;
                ALTER TABLE accounts ADD COLUMN IF NOT EXISTS fitness_source TEXT DEFAULT 'unscored';
                ALTER TABLE accounts ADD COLUMN IF NOT EXISTS signals JSONB;
                ALTER TABLE accounts ADD COLUMN IF NOT EXISTS evidence_urls JSONB;
                ALTER TABLE accounts ADD COLUMN IF NOT EXISTS validated BOOLEAN DEFAULT NULL;
                ALTER TABLE accounts ADD COLUMN IF NOT EXISTS verdict_reason TEXT;
                ALTER TABLE accounts ADD COLUMN IF NOT EXISTS validated_at TIMESTAMPTZ;
                ALTER TABLE accounts ADD COLUMN IF NOT EXISTS promoted_at TIMESTAMPTZ;

                ALTER TABLE emails ADD COLUMN IF NOT EXISTS fitness_score REAL DEFAULT 0.0;
                ALTER TABLE emails ADD COLUMN IF NOT EXISTS fitness_source TEXT DEFAULT 'unscored';
                ALTER TABLE emails ADD COLUMN IF NOT EXISTS signals JSONB;
                ALTER TABLE emails ADD COLUMN IF NOT EXISTS evidence_urls JSONB;
                ALTER TABLE emails ADD COLUMN IF NOT EXISTS validated BOOLEAN DEFAULT NULL;
                ALTER TABLE emails ADD COLUMN IF NOT EXISTS verdict_reason TEXT;
                ALTER TABLE emails ADD COLUMN IF NOT EXISTS validated_at TIMESTAMPTZ;
                ALTER TABLE emails ADD COLUMN IF NOT EXISTS promoted_at TIMESTAMPTZ;

                CREATE INDEX IF NOT EXISTS idx_emails_handle ON emails(handle);
                CREATE INDEX IF NOT EXISTS idx_emails_mx ON emails(mx_valid);
                CREATE INDEX IF NOT EXISTS idx_emails_smtp ON emails(smtp_valid);
                CREATE INDEX IF NOT EXISTS idx_emails_confidence ON emails(confidence DESC);
                CREATE INDEX IF NOT EXISTS idx_emails_domain ON emails(domain);
                CREATE INDEX IF NOT EXISTS idx_emails_country ON emails(country);
                CREATE INDEX IF NOT EXISTS idx_accounts_niche ON accounts(niche);

                ALTER TABLE emails ADD COLUMN IF NOT EXISTS catch_all BOOLEAN DEFAULT FALSE;
                ALTER TABLE emails ADD COLUMN IF NOT EXISTS country TEXT DEFAULT 'XX';
            """)

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None

    # ── Accounts ──

    def add_account(self, handle: str, display_name: str = "",
                    bio: str = "", followers: int = 0,
                    niche: str = "", source: str = ""):
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO accounts (handle, display_name, bio, followers, niche, source)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (handle) DO NOTHING
            """, (handle.lower(), display_name, bio, followers, niche, source))

    def add_accounts_bulk(self, accounts: List[Dict]):
        if not accounts:
            return
        conn = self._get_conn()
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """INSERT INTO accounts (handle, display_name, bio, followers, niche, source)
                   VALUES %s ON CONFLICT (handle) DO NOTHING""",
                [
                    (a["handle"], a.get("display_name", ""), a.get("bio", ""),
                     a.get("followers", 0), a.get("niche", ""), a.get("source", ""))
                    for a in accounts
                ],
            )

    def save_scored_accounts(self, accounts) -> int:
        """Write every scored account to stage, unfiltered and unvalidated."""
        import json

        conn = self._get_conn()
        written = 0
        with conn.cursor() as cur:
            for account in accounts:
                cur.execute(
                    """INSERT INTO accounts
                       (handle, display_name, bio, followers, niche, source,
                        fitness_score, fitness_source, signals, evidence_urls)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (handle) DO UPDATE SET
                           fitness_score = EXCLUDED.fitness_score,
                           fitness_source = EXCLUDED.fitness_source,
                           signals = EXCLUDED.signals,
                           evidence_urls = EXCLUDED.evidence_urls""",
                    (
                        account.handle,
                        account.display_name,
                        account.bio,
                        account.followers,
                        account.niche,
                        account.discovered_by,
                        account.fitness_score,
                        account.fitness_source,
                        json.dumps({}),
                        json.dumps(list(account.evidence_urls)),
                    ),
                )
                written += 1
        conn.commit()
        return written

    def get_unvalidated(self, limit: int = 100):
        """Stage rows that have not been through the validator yet."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """SELECT handle, fitness_score, fitness_source, evidence_urls,
                          validated, verdict_reason
                   FROM accounts WHERE validated IS NULL
                   ORDER BY fitness_score DESC LIMIT %s""",
                (limit,),
            )
            columns = [d[0] for d in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def get_validated_unpromoted(self, limit: int = 100):
        """Rows the validator confirmed that have not yet been promoted.

        This is deliberately the opposite selection of get_unvalidated():
        validated IS NULL means "not yet checked", while this method needs
        validated IS TRUE (checked and confirmed) AND promoted_at IS NULL
        (the promotion gate has not acted on it yet). Reusing
        get_unvalidated() here would look plausible but would select rows
        the validator hasn't touched, so the promotion gate would silently
        promote nothing forever.
        """
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """SELECT handle, fitness_score, fitness_source, evidence_urls,
                          validated, verdict_reason, promoted_at
                   FROM accounts
                   WHERE validated IS TRUE AND promoted_at IS NULL
                   ORDER BY fitness_score DESC LIMIT %s""",
                (limit,),
            )
            columns = [d[0] for d in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def mark_promoted(self, handle: str):
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE accounts SET promoted_at = NOW() WHERE handle = %s",
                (handle,),
            )
        conn.commit()

    def record_verdict(self, handle: str, validated: bool, reason: str):
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE accounts
                   SET validated = %s, verdict_reason = %s, validated_at = NOW()
                   WHERE handle = %s""",
                (validated, reason, handle),
            )
        conn.commit()

    def get_account_count(self) -> int:
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM accounts")
            return cur.fetchone()[0]

    def get_accounts_without_emails(self, limit: int = 100) -> List[Dict]:
        conn = self._get_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT a.* FROM accounts a
                WHERE a.handle NOT IN (SELECT DISTINCT handle FROM emails)
                LIMIT %s
            """, (limit,))
            return [dict(r) for r in cur.fetchall()]

    # ── Emails ──

    def add_email(self, email: str, handle: str, confidence: float = 0,
                  mx_valid: bool = False, smtp_valid: Optional[bool] = None,
                  strategy_id: str = "", domain: str = ""):
        conn = self._get_conn()
        smtp_int = 1 if smtp_valid is True else (0 if smtp_valid is False else -1)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO emails (email, handle, confidence, mx_valid, smtp_valid, strategy_id, domain)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (email) DO UPDATE SET
                    confidence = GREATEST(emails.confidence, EXCLUDED.confidence),
                    mx_valid = EXCLUDED.mx_valid,
                    smtp_valid = CASE WHEN EXCLUDED.smtp_valid > emails.smtp_valid
                                     THEN EXCLUDED.smtp_valid ELSE emails.smtp_valid END
            """, (email.lower(), handle.lower(), confidence,
                  mx_valid, smtp_int, strategy_id, domain))

    def add_emails_bulk(self, emails: List[Dict]):
        if not emails:
            return
        conn = self._get_conn()
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """INSERT INTO emails (email, handle, confidence, mx_valid, smtp_valid, strategy_id, domain, country)
                   VALUES %s
                   ON CONFLICT (email) DO UPDATE SET
                       confidence = GREATEST(emails.confidence, EXCLUDED.confidence),
                       mx_valid = EXCLUDED.mx_valid,
                       smtp_valid = CASE WHEN EXCLUDED.smtp_valid > emails.smtp_valid
                                        THEN EXCLUDED.smtp_valid ELSE emails.smtp_valid END,
                       country = EXCLUDED.country""",
                [
                    (e["email"], e["handle"], e.get("confidence", 0),
                     e.get("mx_valid", False), e.get("smtp_valid", -1),
                     e.get("strategy_id", ""), e.get("domain", ""),
                     e.get("country", "XX"))
                    for e in emails
                ],
            )

    def get_email_count(self, verified_only: bool = False) -> int:
        conn = self._get_conn()
        with conn.cursor() as cur:
            if verified_only:
                cur.execute("SELECT COUNT(*) FROM emails WHERE mx_valid = TRUE")
            else:
                cur.execute("SELECT COUNT(*) FROM emails")
            return cur.fetchone()[0]

    def get_smtp_verified_count(self) -> int:
        conn = self._get_conn()
        with conn.cursor() as cur:
            try:
                cur.execute("SELECT COUNT(*) FROM emails WHERE smtp_valid = 1 AND catch_all = FALSE")
            except Exception:
                cur.execute("SELECT COUNT(*) FROM emails WHERE smtp_valid = 1")
            return cur.fetchone()[0]

    def get_top_emails(self, limit: int = 100) -> List[Dict]:
        conn = self._get_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM emails
                WHERE mx_valid = TRUE
                ORDER BY confidence DESC, smtp_valid DESC
                LIMIT %s
            """, (limit,))
            return [dict(r) for r in cur.fetchall()]

    def email_exists(self, email: str) -> bool:
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM emails WHERE email = %s", (email.lower(),))
            return cur.fetchone() is not None

    # ── Strategies ──

    def record_strategy(self, strategy_id: str, format_pattern: str,
                        domain: str, fitness: float):
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO strategies (id, format_pattern, domain, fitness, success_count)
                VALUES (%s, %s, %s, %s, 1)
                ON CONFLICT (id) DO UPDATE SET
                    fitness = GREATEST(strategies.fitness, EXCLUDED.fitness),
                    success_count = strategies.success_count + 1
            """, (strategy_id, format_pattern, domain, fitness))

    # ── Runs ──

    def start_run(self) -> int:
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO runs (started_at, status)
                VALUES (NOW(), 'running')
                RETURNING id
            """)
            return cur.fetchone()[0]

    def end_run(self, run_id: int, accounts: int, emails: int, verified: int):
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE runs SET
                    ended_at = NOW(),
                    accounts_processed = %s,
                    emails_generated = %s,
                    emails_verified = %s,
                    status = 'completed'
                WHERE id = %s
            """, (accounts, emails, verified, run_id))

    # ── Stats ──

    def get_stats(self) -> Dict[str, Any]:
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM accounts")
            accounts = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM emails")
            total = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM emails WHERE mx_valid = TRUE")
            mx = cur.fetchone()[0]

            try:
                cur.execute("SELECT COUNT(*) FROM emails WHERE smtp_valid = 1 AND catch_all = FALSE")
                smtp = cur.fetchone()[0]
            except Exception:
                cur.execute("SELECT COUNT(*) FROM emails WHERE smtp_valid = 1")
                smtp = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM runs")
            runs = cur.fetchone()[0]

        return {
            "accounts": accounts,
            "emails_total": total,
            "emails_mx_verified": mx,
            "emails_smtp_verified": smtp,
            "runs": runs,
            "top_domains": self._top_domains(),
            "countries": self._safe_countries(),
            "generation": 0,
        }

    def _safe_countries(self) -> List[Dict]:
        try:
            return self._countries_breakdown()
        except Exception as e:
            logger.debug(f"Countries query failed: {e}")
            # Fallback: simple country breakdown without catch_all
            conn = self._get_conn()
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("""
                        SELECT country,
                               COUNT(*) as total,
                               SUM(CASE WHEN mx_valid = TRUE THEN 1 ELSE 0 END) as mx,
                               SUM(CASE WHEN smtp_valid = 1 THEN 1 ELSE 0 END) as verified
                        FROM emails
                        WHERE country IS NOT NULL AND country != 'XX'
                        GROUP BY country
                        ORDER BY verified DESC, total DESC
                    """)
                    return [dict(r) for r in cur.fetchall()]
            except Exception:
                return []

    def get_emails_by_country(self, country: str, verified_only: bool = True,
                               limit: int = 500) -> List[Dict]:
        conn = self._get_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            query = "SELECT * FROM emails WHERE country = %s"
            if verified_only:
                query += " AND smtp_valid = 1 AND catch_all = FALSE"
            query += " ORDER BY confidence DESC LIMIT %s"
            cur.execute(query, (country.upper(), limit))
            return [dict(r) for r in cur.fetchall()]

    def _countries_breakdown(self) -> List[Dict]:
        conn = self._get_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT country,
                       COUNT(*) as total,
                       SUM(CASE WHEN mx_valid = TRUE THEN 1 ELSE 0 END) as mx,
                       SUM(CASE WHEN smtp_valid = 1 AND catch_all = FALSE THEN 1 ELSE 0 END) as verified
                FROM emails
                WHERE country != 'XX'
                GROUP BY country
                ORDER BY verified DESC, total DESC
            """)
            return [dict(r) for r in cur.fetchall()]

    def _top_domains(self, limit: int = 10) -> List[Dict]:
        conn = self._get_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT domain,
                       COUNT(*) as cnt,
                       SUM(CASE WHEN mx_valid = TRUE THEN 1 ELSE 0 END) as mx_cnt,
                       SUM(CASE WHEN smtp_valid = 1 THEN 1 ELSE 0 END) as smtp_cnt
                FROM emails
                GROUP BY domain
                ORDER BY smtp_cnt DESC, mx_cnt DESC
                LIMIT %s
            """, (limit,))
            return [dict(r) for r in cur.fetchall()]

    # ── Export ──

    def export_csv(self, filepath: str, verified_only: bool = True, country: str = None):
        conn = self._get_conn()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            conditions = []
            params = []
            if verified_only:
                conditions.append("mx_valid = TRUE")
            if country:
                conditions.append("country = %s")
                params.append(country.upper())
            query = "SELECT * FROM emails"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY confidence DESC"
            cur.execute(query, params)
            rows = cur.fetchall()

        import csv
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "email", "handle", "confidence", "mx_valid",
                "smtp_valid", "domain", "created_at"
            ])
            for row in rows:
                writer.writerow([
                    row["email"], row["handle"], row["confidence"],
                    row["mx_valid"], row["smtp_valid"],
                    row["domain"], row["created_at"],
                ])

        logger.info(f"Exported {len(rows)} emails to {filepath}")
