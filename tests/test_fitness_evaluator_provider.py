# tests/test_fitness_evaluator_provider.py
from x_pathfinder.fitness import AccountFitnessEvaluator
from x_pathfinder.fitness_providers import (
    SOURCE_COMPOSITE,
    SOURCE_UNSCORED,
    FitnessResult,
)
from x_pathfinder.models import XAccount


class _StubProvider:
    def __init__(self, mapping):
        self._mapping = mapping

    def score_batch(self, candidates):
        return self._mapping


def test_provider_scores_are_written_onto_accounts():
    accounts = [XAccount(handle="alpha"), XAccount(handle="beta")]
    provider = _StubProvider(
        {
            "alpha": FitnessResult(score=90.0, source=SOURCE_COMPOSITE,
                                   evidence_urls=("https://a.example",)),
            "beta": FitnessResult(score=10.0, source=SOURCE_COMPOSITE),
        }
    )
    evaluator = AccountFitnessEvaluator(niche="ai", provider=provider)
    evaluated = evaluator.batch_evaluate(accounts)

    assert evaluated[0].handle == "alpha"        # sorted by fitness, best first
    assert evaluated[0].fitness_score == 90.0
    assert evaluated[0].fitness_source == SOURCE_COMPOSITE
    assert evaluated[0].evidence_urls == ["https://a.example"]


def test_unscored_candidate_keeps_zero_score_but_unscored_source():
    """The GA needs a float to sort on; the source field preserves the truth."""
    accounts = [XAccount(handle="ghost")]
    provider = _StubProvider({"ghost": FitnessResult(score=None, source=SOURCE_UNSCORED)})
    evaluator = AccountFitnessEvaluator(niche="ai", provider=provider)
    evaluated = evaluator.batch_evaluate(accounts)

    assert evaluated[0].fitness_score == 0.0
    assert evaluated[0].fitness_source == SOURCE_UNSCORED


def test_without_provider_legacy_scoring_still_runs():
    evaluator = AccountFitnessEvaluator(niche="ai")
    accounts = [XAccount(handle="alpha", bio="machine learning researcher")]
    evaluated = evaluator.batch_evaluate(accounts)
    assert evaluated[0].fitness_score >= 0.0
