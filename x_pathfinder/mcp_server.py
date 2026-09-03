"""MCP surface for X-Pathfinder.

Lets the Brain and OpenFang hands drive discovery, scoring, validation and
promotion. Every handler returns a plain dict; nothing raises across the
tool boundary, because an MCP client cannot act on a traceback.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

DSN = os.environ.get(
    "DATABASE_URL", "postgresql://pathfinder:pathfinder@localhost:5434/emails"
)


def _database():
    from .database import EmailDatabase

    return EmailDatabase(dsn=DSN)


def _tool_discover(niche: str = "ai", generations: int = 3, top: int = 20,
                   concurrent: int = 3, **_):
    import asyncio

    from .account_discoverer import AccountDiscoverer
    from .composite_scorer import CompositeScorer
    from .fitness_providers import DeterministicScorer
    from .research_scorer import ResearchScorer

    # Without a provider the GA falls back to scoring on Twitter profile
    # fields that syndication.twitter.com no longer serves, so every
    # candidate scores 0.0 and there is no selection pressure at all. This
    # is the same pair _tool_score uses.
    provider = CompositeScorer(DeterministicScorer(), ResearchScorer())
    discoverer = AccountDiscoverer(
        niche=niche, max_concurrent=concurrent, provider=provider
    )
    accounts = asyncio.run(discoverer.run(generations=generations))

    db = _database()
    try:
        # Spec §6: discovery -> stage DB. AccountDiscoverer only writes
        # KnowledgeAccumulator JSON, so without this xpf_score could only
        # ever see rows the email daemon happened to leave behind. Stage is
        # the unfiltered tier by design -- everything lands here, unscored
        # candidates included -- because nothing here is trusted yet. Only
        # the validator's verdict lets a row leave for Supabase.
        staged = db.save_scored_accounts(accounts)
    finally:
        db.close()

    return {"discovered": len(accounts),
            "staged": staged,
            "handles": [a.handle for a in accounts[:top]]}


def _tool_score(limit: int = 50, **_):
    from .composite_scorer import CompositeScorer
    from .fitness_providers import DeterministicScorer
    from .models import XAccount
    from .research_scorer import ResearchScorer

    db = _database()
    try:
        rows = db.get_unvalidated(limit=limit)
        candidates = [XAccount(handle=row["handle"]) for row in rows]
        scorer = CompositeScorer(DeterministicScorer(), ResearchScorer())
        results = scorer.score_batch(candidates)
        for candidate in candidates:
            result = results.get(candidate.handle.lower())
            if result is None:
                continue
            candidate.fitness_score = 0.0 if result.score is None else result.score
            candidate.fitness_source = result.source
            candidate.signals = dict(result.signals or {})
            candidate.evidence_urls = list(result.evidence_urls)
        return {"scored": db.save_scored_accounts(candidates)}
    finally:
        db.close()


_SELF_CITATION_HOSTS = {
    "x.com", "twitter.com", "mobile.x.com", "mobile.twitter.com",
}


def _is_self_citation(url, handle) -> bool:
    """True when this URL is the candidate's own X/Twitter page.

    The research agent picks its own evidence URLs, so left unchecked it
    can cite https://x.com/<handle> and certify the candidate with the
    candidate's own profile. That is a self-report, and the whole point of
    the validator is that verdicts come from independent re-query.

    Sub-paths count too (`x.com/<handle>/status/1` is content the candidate
    authored), otherwise the filter is evaded by appending a segment.
    """
    from urllib.parse import urlparse

    try:
        parsed = urlparse(str(url))
    except Exception:
        return False  # unparseable: leave it to the validator, which fails closed

    host = (parsed.netloc or "").rsplit("@", 1)[-1].split(":")[0].lower()
    if host.startswith("www."):
        host = host[4:]
    if host not in _SELF_CITATION_HOSTS:
        return False

    own = str(handle or "").strip().lstrip("@").lower()
    if not own:
        return False
    path = (parsed.path or "").strip("/").lower()
    return path == own or path.startswith(f"{own}/")


def _tool_validate(limit: int = 50, **_):
    from .validator import EvidenceValidator

    db = _database()
    try:
        validator = EvidenceValidator()
        checked = confirmed = skipped = 0
        for row in db.get_unvalidated(limit=limit):
            evidence = [
                url for url in (row.get("evidence_urls") or [])
                if not _is_self_citation(url, row["handle"])
            ]
            if not evidence:
                # A row with nothing to verify has not been refuted — it has
                # not been examined. Writing validated=FALSE here used to be
                # permanent: get_unvalidated() selects validated IS NULL, so
                # the row could never re-enter the validator even after
                # xpf_score gave it real evidence. One wrong tool ordering
                # (xpf_validate before xpf_score) therefore killed every
                # candidate the email daemon had inserted.
                skipped += 1
                continue
            verdict = validator.validate(row["handle"], evidence, [row["handle"]])
            db.record_verdict(row["handle"], verdict.validated, verdict.reason)
            checked += 1
            confirmed += 1 if verdict.validated else 0
        return {"checked": checked, "confirmed": confirmed,
                "skipped_no_evidence": skipped}
    finally:
        db.close()


def _tool_promote(limit: int = 50, **_):
    from .promotion import PromotionGate, SupabaseWriter

    db = _database()
    try:
        writer = SupabaseWriter(
            service_key=os.environ.get("SUPABASE_SERVICE_KEY", "")
        )
        # Only validated=True, not-yet-promoted rows are eligible for the
        # gate. get_unvalidated() selects the opposite (validated IS NULL,
        # i.e. rows the validator hasn't touched) and would leave
        # PromotionGate.promote() skipping everything, forever, silently.
        rows = [row for row in db.get_validated_unpromoted(limit=limit)]
        return PromotionGate(db, writer).promote(rows)
    finally:
        db.close()


def _tool_status(**_):
    db = _database()
    try:
        return db.get_stats()
    finally:
        db.close()


def _tool_export(output: str = "emails_export.csv", country: str = None, **_):
    db = _database()
    try:
        db.export_csv(output, verified_only=True, country=country)
        return {"exported_to": output}
    finally:
        db.close()


TOOLS = {
    "xpf_discover": _tool_discover,
    "xpf_score": _tool_score,
    "xpf_validate": _tool_validate,
    "xpf_promote": _tool_promote,
    "xpf_status": _tool_status,
    "xpf_export": _tool_export,
}


def handle_call(name: str, arguments: dict) -> dict:
    handler = TOOLS.get(name)
    if handler is None:
        return {"ok": False, "error": f"unknown tool: {name}"}
    try:
        return {"ok": True, "result": handler(**(arguments or {}))}
    except Exception as exc:
        logger.exception("tool %s failed", name)
        return {"ok": False, "error": str(exc)}
