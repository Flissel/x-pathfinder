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


def test_dropped_candidate_clears_stale_evidence_alongside_score_and_source():
    """A candidate that drops out of the provider's mapping must not keep
    evidence_urls from a previous, now-superseded scoring pass. Score,
    source, and evidence must reset together so they can never disagree —
    otherwise an 'unscored' account could still carry evidence that looks
    substantiated to downstream validation."""
    accounts = [
        XAccount(
            handle="alpha",
            fitness_score=90.0,
            fitness_source=SOURCE_COMPOSITE,
            evidence_urls=["https://old.example"],
        )
    ]
    provider = _StubProvider({})  # alpha is absent from the mapping
    evaluator = AccountFitnessEvaluator(niche="ai", provider=provider)
    evaluated = evaluator.batch_evaluate(accounts)

    assert evaluated[0].fitness_score == 0.0
    assert evaluated[0].fitness_source == SOURCE_UNSCORED
    assert evaluated[0].evidence_urls == []


def test_xaccount_to_dict_from_dict_roundtrip_preserves_fitness_fields():
    account = XAccount(
        handle="alpha",
        fitness_score=42.0,
        fitness_source=SOURCE_COMPOSITE,
        evidence_urls=["https://a.example", "https://b.example"],
    )
    restored = XAccount.from_dict(account.to_dict())

    assert restored.fitness_source == SOURCE_COMPOSITE
    assert restored.evidence_urls == ["https://a.example", "https://b.example"]


def test_xaccount_from_dict_defaults_fitness_fields_when_absent():
    """A dict written before fitness_source/evidence_urls existed carries
    neither key; from_dict must default them rather than raising."""
    restored = XAccount.from_dict({"handle": "x"})

    assert restored.fitness_source == "unscored"
    assert restored.evidence_urls == []
