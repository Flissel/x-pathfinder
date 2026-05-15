"""Custom GeneticEngine (NSGA-II Pareto-aware).

Mirrors the structure of x-pathfinder/genome.py:114-152 (tournament-3,
elitism, multi-operator mutate, crisis mode) and extends it with Deb-style
fast non-dominated sort + crowding distance for multi-objective selection,
plus a non-dominated archive that persists best Pareto-front members across
generations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable

from laser_sim.config.schema import EAConfig, GeometryROI, MachineConfig, MaterialConfig
from laser_sim.fitness.evaluator import FitnessVector, PatternEvaluator
from laser_sim.genome.archive import (
    ArchiveEntry,
    ParetoArchive,
    crowding_distance,
    fast_non_dominated_sort,
)
from laser_sim.genome.chromosome import Chromosome, random_chromosome
from laser_sim.genome.operators import (
    crossover_blx_alpha,
    mutate,
    tournament_select,
)
from laser_sim.patterns.primitives import build_primitive


@dataclass(frozen=True)
class GenerationStats:
    generation: int
    population_size: int
    front0_size: int
    archive_size: int
    best_scalar_J: float
    median_scalar_J: float
    hypervolume_2d: float
    crisis: bool


@dataclass(frozen=True)
class EvolutionLog:
    history: tuple[GenerationStats, ...]
    archive: ParetoArchive

    def best(self) -> ArchiveEntry | None:
        return self.archive.best_by_scalar()


class GeneticEngine:
    """NSGA-II + elitism + crisis-mode evolutionary loop.

    Iteration:
      1) evaluate population
      2) merge with elite_pool, rank+crowd, take top N
      3) sample parents via tournament-k on (rank, -crowding)
      4) crossover -> mutate -> clamp; cache-by-hash to skip re-evals
      5) update non-dominated archive
      6) crisis: if best_scalar stagnates for `crisis_after_stagnation`
         generations, multiply mutation sigma until improvement
    """

    def __init__(
        self,
        material: MaterialConfig,
        machine: MachineConfig,
        roi: GeometryROI,
        ea: EAConfig,
        evaluator: PatternEvaluator | None = None,
        on_generation: Callable[[GenerationStats], None] | None = None,
    ) -> None:
        self.material = material
        self.machine = machine
        self.roi = roi
        self.ea = ea
        self.evaluator = evaluator or PatternEvaluator(material, machine, roi)
        self.on_generation = on_generation
        self.rng = random.Random(ea.seed)
        self.archive = ParetoArchive(max_size=256)
        self._cache: dict[str, ArchiveEntry] = {}

    def _evaluate(self, c: Chromosome) -> ArchiveEntry:
        key = c.hash_id()
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        pattern = build_primitive(c.to_primitive_spec(), self.roi)
        ev = self.evaluator.evaluate(
            pattern, hatch_mm=c.hatch_um * 1e-3, spot_um=c.spot_um
        )
        entry = ArchiveEntry(chromosome=c, fitness=ev.fitness, scalar_J=ev.scalar_J)
        self._cache[key] = entry
        return entry

    def _initial_population(self) -> list[ArchiveEntry]:
        out: list[ArchiveEntry] = []
        for _ in range(self.ea.population):
            out.append(self._evaluate(random_chromosome(self.machine, self.rng)))
        return out

    def _make_child(
        self,
        population: list[ArchiveEntry],
        ranks: list[int],
        crowding: list[float],
        crisis: bool,
    ) -> Chromosome:
        idx_pool = list(range(len(population)))
        a_idx = tournament_select(idx_pool, ranks, crowding, self.ea.tournament_k, self.rng)
        b_idx = tournament_select(idx_pool, ranks, crowding, self.ea.tournament_k, self.rng)
        a = population[a_idx].chromosome
        b = population[b_idx].chromosome
        if self.rng.random() < self.ea.crossover_rate:
            child = crossover_blx_alpha(a, b, self.machine, self.rng)
        else:
            child = a if self.rng.random() < 0.5 else b
        if self.rng.random() < self.ea.mutation_rate:
            child = mutate(child, self.machine, self.ea, crisis, self.rng)
        return child.clamped(self.machine)

    def _rank_and_crowd(
        self, population: Iterable[ArchiveEntry]
    ) -> tuple[list[int], list[float]]:
        pop = list(population)
        fits = [e.fitness for e in pop]
        ranks = fast_non_dominated_sort(fits)
        # crowding distance computed per front
        crowding = [0.0] * len(pop)
        from collections import defaultdict

        by_front: dict[int, list[int]] = defaultdict(list)
        for i, r in enumerate(ranks):
            by_front[r].append(i)
        for front_idx, front_members in by_front.items():
            sub = [fits[i] for i in front_members]
            cds = crowding_distance(sub)
            for k, i in enumerate(front_members):
                crowding[i] = cds[k]
        return ranks, crowding

    def run(self, generations: int | None = None) -> EvolutionLog:
        gens = generations if generations is not None else self.ea.generations
        population = self._initial_population()
        for e in population:
            self.archive.add(e)

        history: list[GenerationStats] = []
        last_best = float("inf")
        stagnation = 0

        for g in range(gens):
            ranks, crowding = self._rank_and_crowd(population)
            crisis = stagnation >= self.ea.crisis_after_stagnation

            # elitism: carry first front (or up to `elitism`) into next gen
            elite_indices = sorted(
                range(len(population)),
                key=lambda i: (ranks[i], -crowding[i]),
            )[: self.ea.elitism]
            elites = [population[i] for i in elite_indices]

            children: list[ArchiveEntry] = []
            while len(children) + len(elites) < self.ea.population:
                child = self._make_child(population, ranks, crowding, crisis)
                children.append(self._evaluate(child))

            population = elites + children
            for e in population:
                self.archive.add(e)

            ranks_p, crowd_p = self._rank_and_crowd(population)
            scalars = [e.scalar_J for e in population]
            best_j = min(scalars)
            median_j = sorted(scalars)[len(scalars) // 2]
            front0 = sum(1 for r in ranks_p if r == 0)
            # 2D HV proxy on (p_keyhole, t_cycle_s) with ref (1.0, 5.0).
            # Earlier we used (u_temp_loss, p_lof) but those saturate to zero
            # on the field-mode evaluator within a few generations, so HV stuck
            # at the maximum. (keyhole, t_cycle) keeps differentiating.
            hv = self.archive.hypervolume_2d(axes=(2, 4), ref=(1.0, 5.0))

            stats = GenerationStats(
                generation=g,
                population_size=len(population),
                front0_size=front0,
                archive_size=len(self.archive),
                best_scalar_J=best_j,
                median_scalar_J=median_j,
                hypervolume_2d=hv,
                crisis=crisis,
            )
            history.append(stats)
            if self.on_generation is not None:
                self.on_generation(stats)

            if best_j + 1e-9 < last_best:
                last_best = best_j
                stagnation = 0
            else:
                stagnation += 1
        return EvolutionLog(history=tuple(history), archive=self.archive)
