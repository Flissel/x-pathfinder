from x_pathfinder.fitness_providers import (
    SOURCE_DETERMINISTIC,
    SOURCE_UNSCORED,
    DeterministicScorer,
    FitnessResult,
)
from x_pathfinder.models import XAccount


def _resolver_ok(url: str) -> int:
    return 200


def _resolver_missing(url: str) -> int:
    return 404


def test_wellformed_handle_with_live_profile_scores_above_zero():
    scorer = DeterministicScorer(resolver=_resolver_ok)
    result = scorer.score(XAccount(handle="JeffDean", niche="ai"))
    assert isinstance(result, FitnessResult)
    assert result.source == SOURCE_DETERMINISTIC
    assert result.score > 0.0
    assert result.signals["handle_wellformed"] is True
    assert result.signals["profile_resolves"] is True


def test_dead_profile_scores_lower_than_live_profile():
    live = DeterministicScorer(resolver=_resolver_ok).score(XAccount(handle="alice"))
    dead = DeterministicScorer(resolver=_resolver_missing).score(XAccount(handle="alice"))
    assert dead.score < live.score


def test_malformed_handle_is_unscored_not_zero():
    """Unknown and bad are different states and must not collapse."""
    scorer = DeterministicScorer(resolver=_resolver_ok)
    result = scorer.score(XAccount(handle="  "))
    assert result.source == SOURCE_UNSCORED
    assert result.score is None


def test_resolver_failure_does_not_raise_and_marks_signal_unknown():
    def _boom(url: str):
        raise OSError("network down")

    result = DeterministicScorer(resolver=_boom).score(XAccount(handle="alice"))
    assert result.signals["profile_resolves"] is None
    assert result.source == SOURCE_DETERMINISTIC
