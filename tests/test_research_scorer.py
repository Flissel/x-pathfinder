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
