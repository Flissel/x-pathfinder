"""Promotion gate: which candidates get the expensive HF treatment.

Decision rule (per generation):
  1) Skip HF entirely while generation < policy.fast_only_until_gen.
  2) Always promote the current Pareto-front members (if enabled).
  3) Always promote the top-K by scalar_J that aren't already in (2).
  4) If a surrogate is available with enough training data, also promote
     candidates whose predictive variance exceeds `veto_threshold_sigma`
     of the surrogate's overall variance — these are the "uncertain"
     candidates the HF run will resolve.

The gate does NOT run the HF solver itself; it returns a FidelityDecision
listing which entries to promote. The engine (or scheduler) calls the
HF runner for each.
"""

from __future__ import annotations

from dataclasses import dataclass

from laser_sim.fidelity.config import FidelityTier, PromotionPolicy
from laser_sim.fidelity.surrogate import GPSurrogate
from laser_sim.genome.archive import ArchiveEntry, fast_non_dominated_sort


@dataclass(frozen=True)
class FidelityDecision:
    promoted: tuple[ArchiveEntry, ...]
    reasons: tuple[str, ...]
    skipped: bool = False
    skipped_reason: str = ""


class FidelityGate:
    """Stateless decision maker."""

    def __init__(
        self,
        policy: PromotionPolicy | None = None,
        surrogate: GPSurrogate | None = None,
    ) -> None:
        self.policy = policy or PromotionPolicy()
        self.surrogate = surrogate

    def decide(
        self,
        generation: int,
        population: list[ArchiveEntry],
        archive_entries: list[ArchiveEntry],
    ) -> FidelityDecision:
        pol = self.policy
        if generation < pol.fast_only_until_gen:
            return FidelityDecision(
                promoted=(),
                reasons=(),
                skipped=True,
                skipped_reason=(
                    f"gen {generation} < fast_only_until_gen {pol.fast_only_until_gen}"
                ),
            )

        promoted: list[ArchiveEntry] = []
        reasons: list[str] = []
        seen: set[str] = set()

        def _add(e: ArchiveEntry, reason: str) -> None:
            key = e.chromosome.hash_id()
            if key in seen:
                return
            seen.add(key)
            promoted.append(e)
            reasons.append(reason)

        if pol.hf_promote_pareto_front and archive_entries:
            fits = [e.fitness for e in archive_entries]
            ranks = fast_non_dominated_sort(fits)
            for i, r in enumerate(ranks):
                if r == 0:
                    _add(archive_entries[i], "pareto_front")

        top_k = sorted(population, key=lambda e: e.scalar_J)[: pol.hf_promote_top_k]
        for e in top_k:
            _add(e, "top_k")

        if (
            self.surrogate is not None
            and self.surrogate.n_samples >= pol.surrogate_min_runs
        ):
            ref_var = float(self.surrogate.mean_variance() or 0.0)
            for e in population:
                pred = self.surrogate.predict(e.chromosome)
                if (
                    ref_var > 0
                    and pred.variance > pol.veto_threshold_sigma**2 * ref_var
                ):
                    _add(e, "high_uncertainty")

        return FidelityDecision(promoted=tuple(promoted), reasons=tuple(reasons))
