import json

import pytest

from x_pathfinder.fitness_providers import SOURCE_RESEARCH, SOURCE_UNSCORED
from x_pathfinder.models import XAccount
from x_pathfinder.research_scorer import ResearchScorer, ResearchUnavailable


def _transport_returning(payload: str):
    def _transport(message: str) -> str:
        return payload

    return _transport


def test_parses_scores_and_evidence_urls():
    payload = json.dumps(
        {
            "results": [
                {
                    "handle": "acme",
                    "score": 82.5,
                    "evidence_urls": ["https://acme.example/news/funding"],
                    "reason": "raised Series A",
                }
            ]
        }
    )
    scorer = ResearchScorer(transport=_transport_returning(payload))
    results = scorer.score_batch([XAccount(handle="acme")])

    result = results["acme"]
    assert result.source == SOURCE_RESEARCH
    assert result.score == 82.5
    assert result.evidence_urls == ("https://acme.example/news/funding",)


def test_result_without_evidence_urls_is_unscored_not_zero():
    """No sources means unknown, not bad. This is the anti-hallucination gate."""
    payload = json.dumps(
        {"results": [{"handle": "acme", "score": 90, "evidence_urls": []}]}
    )
    scorer = ResearchScorer(transport=_transport_returning(payload))
    result = scorer.score_batch([XAccount(handle="acme")])["acme"]

    assert result.source == SOURCE_UNSCORED
    assert result.score is None


def test_candidate_missing_from_response_is_unscored():
    payload = json.dumps({"results": []})
    scorer = ResearchScorer(transport=_transport_returning(payload))
    result = scorer.score_batch([XAccount(handle="ghost")])["ghost"]

    assert result.source == SOURCE_UNSCORED
    assert result.score is None


def test_json_embedded_in_prose_is_still_parsed():
    payload = "Here are my findings:\n```json\n" + json.dumps(
        {"results": [{"handle": "acme", "score": 50, "evidence_urls": ["https://a.example"]}]}
    ) + "\n```\nHope that helps."
    scorer = ResearchScorer(transport=_transport_returning(payload))
    assert scorer.score_batch([XAccount(handle="acme")])["acme"].score == 50


def test_unparseable_response_yields_unscored_for_all():
    scorer = ResearchScorer(transport=_transport_returning("I could not help with that."))
    result = scorer.score_batch([XAccount(handle="acme")])["acme"]
    assert result.source == SOURCE_UNSCORED


def test_transport_failure_raises_research_unavailable():
    def _boom(message: str) -> str:
        raise OSError("connection refused")

    scorer = ResearchScorer(transport=_boom)
    with pytest.raises(ResearchUnavailable):
        scorer.score_batch([XAccount(handle="acme")])


def test_valid_json_followed_by_trailing_braces_still_parses():
    """Greedy regex would match from first { to last }, breaking on trailing prose with braces."""
    payload = json.dumps(
        {"results": [{"handle": "acme", "score": 75, "evidence_urls": ["https://real.example"]}]}
    ) + "\n\nThreshold is {50}. All done."
    scorer = ResearchScorer(transport=_transport_returning(payload))
    result = scorer.score_batch([XAccount(handle="acme")])["acme"]
    assert result.score == 75
    assert result.source == SOURCE_RESEARCH


def test_null_score_with_evidence_urls_yields_unscored():
    """null score should not crash; candidate should be unscored."""
    payload = json.dumps(
        {"results": [{"handle": "acme", "score": None, "evidence_urls": ["https://real.example"]}]}
    )
    scorer = ResearchScorer(transport=_transport_returning(payload))
    result = scorer.score_batch([XAccount(handle="acme")])["acme"]
    assert result.source == SOURCE_UNSCORED
    assert result.score is None


def test_non_numeric_score_with_evidence_urls_yields_unscored():
    """Non-numeric score like 'high' should not crash; candidate should be unscored."""
    payload = json.dumps(
        {"results": [{"handle": "acme", "score": "high", "evidence_urls": ["https://real.example"]}]}
    )
    scorer = ResearchScorer(transport=_transport_returning(payload))
    result = scorer.score_batch([XAccount(handle="acme")])["acme"]
    assert result.source == SOURCE_UNSCORED
    assert result.score is None


def test_one_malformed_score_does_not_poison_batch():
    """Batch of two candidates where one has malformed score; the other should still get real score."""
    payload = json.dumps(
        {
            "results": [
                {"handle": "good", "score": 80, "evidence_urls": ["https://good.example"]},
                {"handle": "bad", "score": "not_a_number", "evidence_urls": ["https://bad.example"]},
            ]
        }
    )
    scorer = ResearchScorer(transport=_transport_returning(payload))
    results = scorer.score_batch([XAccount(handle="good"), XAccount(handle="bad")])

    # Good candidate should have real score despite bad candidate
    assert results["good"].score == 80
    assert results["good"].source == SOURCE_RESEARCH

    # Bad candidate should be unscored
    assert results["bad"].score is None
    assert results["bad"].source == SOURCE_UNSCORED
