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
from typing import Any, Callable, Iterable, Protocol

from laser_sim.config.schema import EAConfig, GeometryROI, MachineConfig, MaterialConfig
from laser_sim.control_plane.events import EventBus, EventType, EvolutionEvent
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


class EvalCache(Protocol):
    def __contains__(self, key: str) -> bool: ...
    def get(self, key: str) -> ArchiveEntry | None: ...
    def put(self, key: str, entry: ArchiveEntry, **kw: Any) -> None: ...


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
        cache: EvalCache | dict[str, ArchiveEntry] | None = None,
        on_commit: Callable[[int, list[ArchiveEntry]], None] | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.material = material
        self.machine = machine
        self.roi = roi
        self.ea = ea
        self.evaluator = evaluator or PatternEvaluator(material, machine, roi)
        self.on_generation = on_generation
        self.rng = random.Random(ea.seed)
        self.archive = ParetoArchive(max_size=256)
        self._cache = cache if cache is not None else {}
        # if the cache is pre-populated (e.g. KnowledgeAccumulator hydrated
        # from disk), seed the archive with its entries
        self._seed_archive_from_cache()
        self.on_commit = on_commit
        self.event_bus = event_bus
        self._current_generation = 0

    def _seed_archive_from_cache(self) -> None:
        snapshot = getattr(self._cache, "memory_snapshot", None)
        if snapshot is None:
            return
        for e in snapshot():
            self.archive.add(e)

    def _evaluate(self, c: Chromosome) -> ArchiveEntry:
        key = c.hash_id()
        if isinstance(self._cache, dict):
            hit = self._cache.get(key)
        else:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        pattern = build_primitive(c.to_primitive_spec(), self.roi)
        ev = self.evaluator.evaluate(
            pattern, hatch_mm=c.hatch_um * 1e-3, spot_um=c.spot_um
        )
        entry = ArchiveEntry(chromosome=c, fitness=ev.fitness, scalar_J=ev.scalar_J)
        if isinstance(self._cache, dict):
            self._cache[key] = entry
        else:
            self._cache.put(key, entry, generation=self._current_generation, metrics=dict(ev.metrics))
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

    def _publish(self, event_type: EventType, payload: dict[str, Any]) -> None:
        if self.event_bus is not None:
            self.event_bus.publish(EvolutionEvent(type=event_type, payload=payload))

    def run(self, generations: int | None = None) -> EvolutionLog:
        gens = generations if generations is not None else self.ea.generations
        self._publish(
            EventType.CAMPAIGN_START,
            {
                "population": self.ea.population,
                "generations": gens,
                "seed": self.ea.seed,
                "scenario_material": self.material.material_id,
                "scenario_machine": self.machine.machine_id,
            },
        )
        population = self._initial_population()
        for e in population:
            self.archive.add(e)

        history: list[GenerationStats] = []
        last_best = float("inf")
        stagnation = 0

        for g in range(gens):
            self._current_generation = g
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
            if self.on_commit is not None:
                self.on_commit(g, self.archive.entries)

            best_entry = self.archive.best_by_scalar()
            best_chrom: dict[str, Any] = {}
            if best_entry is not None:
                bc = best_entry.chromosome
                best_chrom = {
                    "kind": bc.primitive_kind.value,
                    "power_W": bc.power_W,
                    "speed_mm_s": bc.speed_mm_s,
                    "hatch_um": bc.hatch_um,
                    "spot_um": bc.spot_um,
                    "rotation_deg": bc.layer_rotation_deg,
                    "scalar_J": best_entry.scalar_J,
                    "fitness": list(best_entry.fitness.values),
                }
            self._publish(
                EventType.GENERATION,
                {
                    "generation": g,
                    "total_generations": gens,
                    "population_size": stats.population_size,
                    "front0_size": stats.front0_size,
                    "archive_size": stats.archive_size,
                    "best_scalar_J": stats.best_scalar_J,
                    "median_scalar_J": stats.median_scalar_J,
                    "hypervolume_2d": stats.hypervolume_2d,
                    "crisis": stats.crisis,
                    "best": best_chrom,
                },
            )

            if best_j + 1e-9 < last_best:
                last_best = best_j
                stagnation = 0
            else:
                stagnation += 1
        self._publish(
            EventType.CAMPAIGN_END,
            {
                "generations_run": len(history),
                "archive_size": len(self.archive),
                "best_scalar_J": history[-1].best_scalar_J if history else None,
            },
        )
        return EvolutionLog(history=tuple(history), archive=self.archive)
