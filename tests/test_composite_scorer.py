from x_pathfinder.composite_scorer import CompositeScorer
from x_pathfinder.fitness_providers import (
    SOURCE_COMPOSITE,
    SOURCE_DETERMINISTIC,
    SOURCE_UNSCORED,
    FitnessResult,
)
from x_pathfinder.models import XAccount
from x_pathfinder.research_scorer import ResearchUnavailable


class _StubDeterministic:
    def __init__(self, score):
        self._score = score
        self.calls = []

    def score(self, candidate):
        self.calls.append(candidate.handle)
        if self._score is None:
            return FitnessResult(score=None, source=SOURCE_UNSCORED)
        return FitnessResult(score=self._score, source=SOURCE_DETERMINISTIC,
                             signals={"handle_wellformed": True})


class _StubResearch:
    def __init__(self, result=None, raises=False):
        self._result = result
        self._raises = raises
        self.batches = []

    def score_batch(self, candidates):
        if self._raises:
            raise ResearchUnavailable("down")
        self.batches.append([c.handle for c in candidates])
        return {c.handle.lower(): self._result for c in candidates}


def test_candidate_below_prefilter_never_reaches_research():
    research = _StubResearch()
    scorer = CompositeScorer(_StubDeterministic(5.0), research, prefilter_threshold=25.0)
    results = scorer.score_batch([XAccount(handle="weak")])

    assert research.batches == []
    assert results["weak"].source == SOURCE_DETERMINISTIC


def test_candidate_above_prefilter_is_blended_with_research():
    research = _StubResearch(
        FitnessResult(score=100.0, source="research",
                      evidence_urls=("https://a.example",))
    )
    scorer = CompositeScorer(_StubDeterministic(40.0), research,
                             prefilter_threshold=25.0, research_weight=0.7)
    result = scorer.score_batch([XAccount(handle="strong")])["strong"]

    assert result.source == SOURCE_COMPOSITE
    assert result.score == 82.0  # 40*0.3 + 100*0.7
    assert result.evidence_urls == ("https://a.example",)


def test_research_outage_degrades_to_deterministic_without_raising():
    scorer = CompositeScorer(_StubDeterministic(40.0), _StubResearch(raises=True))
    result = scorer.score_batch([XAccount(handle="strong")])["strong"]

    assert result.source == SOURCE_DETERMINISTIC
    assert result.score == 40.0


def test_unscorable_candidate_stays_unscored_and_skips_research():
    research = _StubResearch()
    scorer = CompositeScorer(_StubDeterministic(None), research)
    result = scorer.score_batch([XAccount(handle="  ")])["  "]

    assert result.source == SOURCE_UNSCORED
    assert result.score is None
    assert research.batches == []


def test_unscored_research_result_falls_back_to_deterministic_score():
    research = _StubResearch(FitnessResult(score=None, source=SOURCE_UNSCORED))
    scorer = CompositeScorer(_StubDeterministic(40.0), research)
    result = scorer.score_batch([XAccount(handle="strong")])["strong"]

    assert result.source == SOURCE_DETERMINISTIC
    assert result.score == 40.0
