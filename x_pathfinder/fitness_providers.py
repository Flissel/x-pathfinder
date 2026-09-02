"""Pluggable fitness scoring for the genetic algorithm.

The original fitness function required Twitter profile data from
syndication.twitter.com, which now returns 429 and blocks. Without a
fitness signal the GA has no selection pressure. These providers replace
that single hard-wired source with swappable scorers.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional, Protocol

from .models import XAccount

logger = logging.getLogger(__name__)

SOURCE_DETERMINISTIC = "deterministic"
SOURCE_RESEARCH = "research"
SOURCE_COMPOSITE = "composite"
SOURCE_UNSCORED = "unscored"

_HANDLE_RE = re.compile(r"^[A-Za-z0-9_]{1,15}$")


@dataclass(frozen=True)
class FitnessResult:
    """The outcome of scoring one candidate.

    `score` is None when the candidate could not be scored at all. That is
    deliberately different from a score of 0.0, which means "scored, and bad".
    """

    score: Optional[float]
    source: str
    signals: dict = field(default_factory=dict)
    evidence_urls: tuple[str, ...] = ()
    scored_at: datetime = field(default_factory=datetime.now)


class FitnessProvider(Protocol):
    def score(self, candidate: XAccount) -> FitnessResult: ...


def _default_resolver(url: str) -> Optional[int]:
    import requests

    return requests.get(url, timeout=(3, 5), allow_redirects=True).status_code


class DeterministicScorer:
    """Cheap, in-process signals. No LLM, no cost, no external spend."""

    def __init__(self, resolver: Callable[[str], Optional[int]] = _default_resolver):
        self._resolver = resolver

    def score(self, candidate: XAccount) -> FitnessResult:
        handle = (candidate.handle or "").strip()
        if not _HANDLE_RE.match(handle):
            return FitnessResult(
                score=None,
                source=SOURCE_UNSCORED,
                signals={"handle_wellformed": False},
            )

        profile_resolves = self._resolve(candidate.profile_url or f"https://x.com/{handle}")

        score = 20.0
        if profile_resolves is True:
            score += 20.0

        return FitnessResult(
            score=score,
            source=SOURCE_DETERMINISTIC,
            signals={
                "handle_wellformed": True,
                "profile_resolves": profile_resolves,
            },
        )

    def _resolve(self, url: str) -> Optional[bool]:
        """True/False when known, None when the check itself failed."""
        try:
            status = self._resolver(url)
        except Exception as exc:
            logger.debug("resolver failed for %s: %s", url, exc)
            return None
        if status is None:
            return None
        return 200 <= int(status) < 400
