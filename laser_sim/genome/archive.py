"""Pareto archive + NSGA-II non-dominated sort and crowding distance.

Kept dependency-free (numpy only) so the EA can run on a laptop and so the
mechanics are auditable. Both routines follow Deb et al. 2002, NSGA-II.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from laser_sim.fitness.evaluator import FitnessVector
from laser_sim.genome.chromosome import Chromosome


@dataclass(frozen=True)
class ArchiveEntry:
    chromosome: Chromosome
    fitness: FitnessVector
    scalar_J: float


def fast_non_dominated_sort(fitnesses: list[FitnessVector]) -> list[int]:
    """Return rank per individual (0 = first front). O(M N^2)."""
    n = len(fitnesses)
    if n == 0:
        return []
    F = np.array([f.as_array() for f in fitnesses])  # (n, M)
    dominates_count = np.zeros(n, dtype=int)
    dominated_by: list[list[int]] = [[] for _ in range(n)]
    # i dominates j iff i <= j on all, i < j on at least one
    for i in range(n):
        di = F[i] <= F  # (n, M)
        sj = F[i] < F  # (n, M)
        i_dom_j = di.all(axis=1) & sj.any(axis=1)
        j_dom_i = (F <= F[i]).all(axis=1) & (F < F[i]).any(axis=1)
        dominated_by[i] = [j for j, v in enumerate(i_dom_j) if v and j != i]
        dominates_count[i] = int(j_dom_i.sum()) - int(j_dom_i[i])
    ranks = [-1] * n
    current: list[int] = [i for i in range(n) if dominates_count[i] == 0]
    r = 0
    while current:
        for i in current:
            ranks[i] = r
        next_front: list[int] = []
        for i in current:
            for j in dominated_by[i]:
                dominates_count[j] -= 1
                if dominates_count[j] == 0:
                    next_front.append(j)
        r += 1
        current = next_front
    # any leftover (shouldn't happen but be safe)
    for i in range(n):
        if ranks[i] == -1:
            ranks[i] = r
    return ranks


def crowding_distance(fitnesses: list[FitnessVector]) -> list[float]:
    """NSGA-II crowding distance per individual, on the WHOLE set (callers
    typically pass one front at a time)."""
    n = len(fitnesses)
    if n == 0:
        return []
    if n <= 2:
        return [math.inf] * n
    F = np.array([f.as_array() for f in fitnesses])  # (n, M)
    m = F.shape[1]
    dist = np.zeros(n, dtype=float)
    for k in range(m):
        order = np.argsort(F[:, k])
        dist[order[0]] = math.inf
        dist[order[-1]] = math.inf
        f_min, f_max = F[order[0], k], F[order[-1], k]
        span = f_max - f_min
        if span <= 0:
            continue
        for idx in range(1, n - 1):
            i = order[idx]
            dist[i] += (F[order[idx + 1], k] - F[order[idx - 1], k]) / span
    return dist.tolist()


class ParetoArchive:
    """Bounded non-dominated archive across generations.

    On add(entry): drop entries dominated by `entry`; if `entry` is dominated
    by any current member, ignore it. Cap at max_size by removing the entry
    with the smallest crowding distance.
    """

    def __init__(self, max_size: int = 256) -> None:
        self.max_size = max_size
        self._entries: list[ArchiveEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[ArchiveEntry]:
        return list(self._entries)

    def add(self, entry: ArchiveEntry) -> bool:
        for e in self._entries:
            if e.fitness.dominates(entry.fitness):
                return False
            if e.fitness.values == entry.fitness.values:
                return False
        self._entries = [e for e in self._entries if not entry.fitness.dominates(e.fitness)]
        self._entries.append(entry)
        if len(self._entries) > self.max_size:
            self._prune()
        return True

    def _prune(self) -> None:
        fits = [e.fitness for e in self._entries]
        cds = crowding_distance(fits)
        order = sorted(range(len(self._entries)), key=lambda i: cds[i], reverse=True)
        keep = sorted(order[: self.max_size])
        self._entries = [self._entries[i] for i in keep]

    def best_by_scalar(self) -> ArchiveEntry | None:
        if not self._entries:
            return None
        return min(self._entries, key=lambda e: e.scalar_J)

    def hypervolume_2d(
        self, axes: tuple[int, int], ref: tuple[float, float]
    ) -> float:
        """Indicative 2D hypervolume on two chosen axes (cheap proxy)."""
        if not self._entries:
            return 0.0
        pts = sorted(
            [
                (e.fitness.values[axes[0]], e.fitness.values[axes[1]])
                for e in self._entries
            ]
        )
        # filter to non-dominated on chosen 2 axes
        nd: list[tuple[float, float]] = []
        best_y = math.inf
        for x, y in pts:
            if y < best_y:
                nd.append((x, y))
                best_y = y
        rx, ry = ref
        hv = 0.0
        prev_x = rx
        for x, y in nd:
            if x >= rx or y >= ry:
                continue
            hv += (prev_x - x) * (ry - y)
            prev_x = x
        return hv
