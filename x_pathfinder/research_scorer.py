"""Semantic fitness via the OpenFang research agent.

Discovery is done by the GA; this only *scores candidates the GA already
found*. That split matters: asking a model to compile a target list is
refused, asking it to verify a signal for a known candidate is ordinary
sourced research.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Callable, Optional

from .fitness_providers import (
    SOURCE_RESEARCH,
    SOURCE_UNSCORED,
    FitnessResult,
)
from .models import XAccount

logger = logging.getLogger(__name__)

OPENFANG_URL = "http://127.0.0.1:4200"
RESEARCH_AGENT_ID = "f241f569-e6ab-429d-8d61-247889722e84"

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

PROMPT_TEMPLATE = """Score each candidate below for fitness in the niche "{niche}".

For each candidate, research its current public signals (funding, hiring,
launches, notable activity) and assign a score from 0 to 100.

Reply with STRICT JSON only, no prose outside the JSON:
{{"results": [{{"handle": "...", "score": 0-100,
   "evidence_urls": ["https://..."], "reason": "one line"}}]}}

Rules:
- Every result MUST include at least one real evidence_url you actually found.
- If you found no sources for a candidate, return it with "evidence_urls": [].
  Do not invent sources and do not guess a score.

Candidates:
{candidates}
"""


class ResearchUnavailable(RuntimeError):
    """The research backend could not be reached at all."""


def _default_transport(message: str, attempts: int = 3) -> str:
    """Synchronous POST with exponential backoff.

    Spec §7 asks to reuse rate_limiter.py, but AdaptiveRateLimiter is
    async-only (`async def acquire`) while this scorer is called from
    synchronous GA code. Rather than drag an event loop into the call path,
    the same backoff shape is applied inline. Deviation is deliberate.
    """
    import time

    import requests

    delay = 2.0
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            response = requests.post(
                f"{OPENFANG_URL}/api/agents/{RESEARCH_AGENT_ID}/message",
                json={"message": message},
                timeout=(5, 180),
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as exc:
            last_error = exc
            time.sleep(delay)
            delay *= 2
    raise last_error  # surfaces as ResearchUnavailable in score_batch


class ResearchScorer:
    def __init__(
        self,
        transport: Callable[[str], str] = _default_transport,
        agent_id: str = RESEARCH_AGENT_ID,
    ):
        self._transport = transport
        self.agent_id = agent_id

    def score(self, candidate: XAccount) -> FitnessResult:
        return self.score_batch([candidate])[candidate.handle.lower()]

    def score_batch(self, candidates: list[XAccount]) -> dict[str, FitnessResult]:
        if not candidates:
            return {}

        niche = candidates[0].niche or "general"
        listing = "\n".join(f"- {c.handle}" for c in candidates)
        prompt = PROMPT_TEMPLATE.format(niche=niche, candidates=listing)

        try:
            raw = self._transport(prompt)
        except Exception as exc:
            raise ResearchUnavailable(str(exc)) from exc

        parsed = self._parse(raw)
        return {
            candidate.handle.lower(): self._result_for(candidate, parsed)
            for candidate in candidates
        }

    def _parse(self, raw: str) -> dict[str, dict]:
        match = _JSON_BLOCK_RE.search(raw or "")
        if not match:
            logger.warning("research response contained no JSON block")
            return {}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            logger.warning("research response was not valid JSON: %s", exc)
            return {}
        return {
            str(item.get("handle", "")).lower(): item
            for item in payload.get("results", [])
            if isinstance(item, dict)
        }

    def _result_for(self, candidate: XAccount, parsed: dict[str, dict]) -> FitnessResult:
        item = parsed.get(candidate.handle.lower())
        urls = tuple(item.get("evidence_urls") or ()) if item else ()
        if not item or not urls:
            return FitnessResult(
                score=None,
                source=SOURCE_UNSCORED,
                signals={"reason": "no sourced result returned"},
            )
        return FitnessResult(
            score=float(item.get("score", 0.0)),
            source=SOURCE_RESEARCH,
            signals={"reason": item.get("reason", "")},
            evidence_urls=urls,
        )
