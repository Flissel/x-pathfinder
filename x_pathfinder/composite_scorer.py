"""Blends cheap deterministic signals with expensive semantic research.

Deterministic scoring is free, so it runs first and acts as a gate: only
candidates that clear the threshold are worth a research call. If research
is unavailable the GA degrades to deterministic fitness rather than dying.
"""

from __future__ import annotations

import logging

from .fitness_providers import (
    SOURCE_COMPOSITE,
    SOURCE_UNSCORED,
    FitnessResult,
)
from .models import XAccount
from .research_scorer import ResearchUnavailable

logger = logging.getLogger(__name__)


class CompositeScorer:
    def __init__(
        self,
        deterministic,
        research,
        prefilter_threshold: float = 25.0,
        research_weight: float = 0.7,
    ):
        self._deterministic = deterministic
        self._research = research
        self._prefilter_threshold = prefilter_threshold
        self._research_weight = research_weight

    def score_batch(self, candidates: list[XAccount]) -> dict[str, FitnessResult]:
        base: dict[str, FitnessResult] = {}
        eligible: list[XAccount] = []

        for candidate in candidates:
            result = self._deterministic.score(candidate)
            base[candidate.handle.lower()] = result
            if result.score is not None and result.score >= self._prefilter_threshold:
                eligible.append(candidate)

        if not eligible:
            return base

        try:
            researched = self._research.score_batch(eligible)
        except ResearchUnavailable as exc:
            logger.warning("research unavailable, degrading to deterministic: %s", exc)
            return base

        merged = dict(base)
        for candidate in eligible:
            key = candidate.handle.lower()
            research_result = researched.get(key)
            if research_result is None or research_result.score is None:
                continue
            deterministic_result = base[key]
            blended = (
                deterministic_result.score * (1.0 - self._research_weight)
                + research_result.score * self._research_weight
            )
            merged[key] = FitnessResult(
                score=blended,
                source=SOURCE_COMPOSITE,
                signals={
                    **deterministic_result.signals,
                    **research_result.signals,
                },
                evidence_urls=research_result.evidence_urls,
            )
        return merged
