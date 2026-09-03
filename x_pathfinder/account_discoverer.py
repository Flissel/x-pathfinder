"""
Account Discoverer - Main orchestrator for evolutionary X account discovery.

Coordinates the genetic algorithm, scraper, fitness evaluator, and knowledge
accumulator to discover relevant X/Twitter accounts through evolution.

Discovery flow:
1. Bootstrap: GitHub awesome-lists provide initial seed accounts
2. Enrichment: Syndication API gets full profiles for seed accounts
3. Graph traversal: Syndication timelines reveal mentioned/interacted accounts
4. Evolution: Genetic algorithm evolves which accounts to traverse and how
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .models import XAccount, SearchStrategy
from .scraper import XScraper
from .fitness import AccountFitnessEvaluator
from .genome import GeneticEngine
from .knowledge_accumulator import KnowledgeAccumulator
from .rate_limiter import AdaptiveRateLimiter

logger = logging.getLogger(__name__)


class AccountDiscoverer:
    """Discovers X/Twitter accounts through evolutionary search.

    Usage::

        discoverer = AccountDiscoverer(niche="ai")
        results = await discoverer.run(generations=15)
        for account in results:
            print(f"@{account.handle} fitness={account.fitness_score}")

    Pass ``provider=`` (any object with ``score_batch``) to score through the
    pluggable fitness layer instead of the dead Twitter-profile fields::

        AccountDiscoverer(niche="ai",
                          provider=CompositeScorer(DeterministicScorer(),
                                                   ResearchScorer()))
    """

    CRISIS_THRESHOLD = 3

    def __init__(
        self,
        niche: str = "ai",
        custom_keywords: List[str] = None,
        custom_hashtags: List[str] = None,
        max_concurrent: int = 3,
        knowledge_dir: str = None,
        provider=None,
    ):
        self.niche = niche.lower()
        self.max_concurrent = max_concurrent

        self.rate_limiter = AdaptiveRateLimiter()
        self.scraper = XScraper(rate_limiter=self.rate_limiter)
        # provider=None keeps the legacy Twitter-profile scoring path, which
        # is what the CLI and every existing caller expect. Passing a
        # FitnessProvider (e.g. CompositeScorer) is what actually restores
        # selection pressure: the legacy path reads bio/followers/engagement
        # from syndication.twitter.com, which returns 429 permanently, so
        # every candidate scores 0.0 and the GA evolves without optimising.
        self.fitness = AccountFitnessEvaluator(
            niche=self.niche, custom_keywords=custom_keywords, provider=provider
        )
        self.genome = GeneticEngine(
            niche=self.niche,
            custom_keywords=custom_keywords,
            custom_hashtags=custom_hashtags,
        )
        self.knowledge = KnowledgeAccumulator(knowledge_dir=knowledge_dir)

        self.session_start: Optional[datetime] = None
        self.all_discovered: Dict[str, XAccount] = {}
        self.new_accounts_this_session: List[XAccount] = []
        self.generation_stats: List[Dict[str, Any]] = []

    async def run(
        self,
        generations: int = 15,
        on_progress: callable = None,
    ) -> List[XAccount]:
        """Run the evolutionary discovery process."""
        self.session_start = datetime.now()
        known_handles = self.knowledge.get_known_handles()

        logger.info(
            f"Starting X Pathfinder - niche: {self.niche}, "
            f"max generations: {generations}, "
            f"known accounts: {len(known_handles)}"
        )

        # Step 1: Bootstrap - find initial seed accounts via GitHub
        seed_accounts = await self._bootstrap_seeds(known_handles)
        logger.info(f"Bootstrap: {len(seed_accounts)} seed accounts")

        # Step 2: Enrich seed accounts via syndication (get profiles)
        enriched = await self._enrich_accounts(seed_accounts[:20])
        new_from_bootstrap = self._process_new_accounts(enriched, known_handles)
        logger.info(f"Enriched: {len(new_from_bootstrap)} new accounts from bootstrap")

        # Register seeds for graph traversal
        self.genome.register_discovered_accounts(
            [a.handle for a in self.all_discovered.values()]
        )

        # Step 3: Initialize population using discovered seed handles
        known_strategies = self.knowledge.get_known_strategies()
        population = self.genome.create_initial_population(known_strategies)

        # Step 4: Evolution loop - evolve graph traversal strategies
        crisis_mode = False
        generations_without_new = 0
        total_new = len(new_from_bootstrap)

        for gen in range(generations):
            gen_start = datetime.now()

            # Execute strategies (graph traversal + enrichment)
            results = await self._execute_generation(population)

            # Process results
            new_accounts = self._process_generation_results(results, known_handles)
            total_new += len(new_accounts)

            # Update strategy fitness
            for strategy, accounts in results:
                unique_count = len([
                    a for a in accounts
                    if a.handle.lower() not in known_handles
                ])
                avg_fitness = (
                    sum(a.fitness_score for a in accounts) / len(accounts)
                    if accounts else 0
                )
                strategy.fitness = unique_count * 0.6 + avg_fitness * 0.4
                strategy.accounts_found = len(accounts)

            # Feed discovered handles back for mutation
            new_handles = [a.handle for a in new_accounts]
            self.genome.register_discovered_accounts(new_handles)
            known_handles.update(h.lower() for h in new_handles)

            # Crisis detection
            if new_accounts:
                generations_without_new = 0
                crisis_mode = False
            else:
                generations_without_new += 1
                if generations_without_new >= self.CRISIS_THRESHOLD:
                    crisis_mode = True
                    logger.warning(
                        f"CRISIS MODE: No new accounts for "
                        f"{generations_without_new} generations"
                    )

            gen_duration = (datetime.now() - gen_start).total_seconds()
            gen_stats = {
                "generation": gen + 1,
                "new_accounts": len(new_accounts),
                "total_accounts_found": sum(
                    s.accounts_found for s in population
                ),
                "best_fitness": max(
                    (s.fitness for s in population), default=0
                ),
                "crisis_mode": crisis_mode,
                "duration_seconds": gen_duration,
            }
            self.generation_stats.append(gen_stats)

            logger.info(
                f"Gen {gen + 1}/{generations}: "
                f"{len(new_accounts)} new accounts, "
                f"best_strat_fitness={gen_stats['best_fitness']:.1f}, "
                f"crisis={crisis_mode}, "
                f"time={gen_duration:.1f}s"
            )

            if on_progress:
                on_progress(gen + 1, gen_stats)

            # Record successful strategies
            for strategy, accounts in results:
                if accounts:
                    self.knowledge.add_successful_strategy(
                        strategy, len(accounts)
                    )

            # Evolve population
            population = self.genome.evolve(population, crisis_mode=crisis_mode)

        # Finalize
        await self._finalize_session(total_new, generations)

        all_accounts = sorted(
            self.all_discovered.values(),
            key=lambda a: a.fitness_score,
            reverse=True,
        )
        return all_accounts

    async def _bootstrap_seeds(self, known_handles: set) -> List[XAccount]:
        """Bootstrap: find seed accounts via GitHub awesome-lists."""
        # Check APIs
        instances = await self.scraper.discover_nitter_instances()
        if instances:
            self.knowledge.save_nitter_instances(instances)

        # Search GitHub for curated account lists
        niche_queries = {
            "ai": "artificial intelligence machine learning",
            "crypto": "cryptocurrency blockchain",
            "tech": "technology programming developer",
            "security": "cybersecurity infosec",
        }
        query = niche_queries.get(self.niche, self.niche)
        page = await self.scraper.search_accounts(query, max_results=50)

        accounts = page.accounts if page.success else []

        # Also add seed accounts from genome seeds
        for seed_handle in self.genome.seeds.get("seed_accounts", []):
            if seed_handle.lower() not in known_handles:
                accounts.append(XAccount(
                    handle=seed_handle,
                    profile_url=f"https://x.com/{seed_handle}",
                ))

        return accounts

    async def _enrich_accounts(
        self, accounts: List[XAccount], max_enrich: int = 10
    ) -> List[XAccount]:
        """Enrich accounts with full profile data via syndication."""
        enriched = []
        sem = asyncio.Semaphore(1)  # One at a time for syndication

        async def enrich_one(account: XAccount) -> XAccount:
            async with sem:
                if not account.bio and not account.followers:
                    full = await self.scraper.scrape_profile(account.handle)
                    if full:
                        full.niche = self.niche
                        return full
                account.niche = self.niche
                return account

        tasks = [enrich_one(a) for a in accounts[:max_enrich]]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, XAccount):
                enriched.append(result)
            elif isinstance(result, Exception):
                logger.debug(f"Enrichment failed: {result}")

        return enriched

    def _process_new_accounts(
        self, accounts: List[XAccount], known_handles: set
    ) -> List[XAccount]:
        """Process and evaluate new accounts."""
        new_accounts = []

        self.fitness.batch_evaluate(accounts)

        for account in accounts:
            handle_lower = account.handle.lower()
            if handle_lower not in self.all_discovered:
                self.all_discovered[handle_lower] = account
                if handle_lower not in known_handles:
                    new_accounts.append(account)
                    self.new_accounts_this_session.append(account)
                    self.knowledge.add_account(account)
            elif account.fitness_score > self.all_discovered[handle_lower].fitness_score:
                self.all_discovered[handle_lower] = account
                self.knowledge.add_account(account)

        return new_accounts

    async def _execute_generation(
        self, population: List[SearchStrategy]
    ) -> List[tuple]:
        """Execute all strategies in a generation."""
        sem = asyncio.Semaphore(self.max_concurrent)
        results = []

        async def execute_strategy(strategy: SearchStrategy):
            async with sem:
                accounts = await self._execute_single_strategy(strategy)
                return strategy, accounts

        tasks = [asyncio.create_task(execute_strategy(s)) for s in population]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for result in completed:
            if isinstance(result, Exception):
                logger.debug(f"Strategy execution failed: {result}")
                continue
            results.append(result)

        return results

    async def _execute_single_strategy(
        self, strategy: SearchStrategy
    ) -> List[XAccount]:
        """Execute a single search strategy."""
        accounts = []
        traversal = strategy.get_traversal_mode()
        seed_accounts = strategy.get_seed_accounts()

        if traversal in ("followers", "following") and seed_accounts:
            # Graph traversal: get mentioned accounts from seed timelines
            for seed in seed_accounts[:2]:
                mentioned = await self.scraper.scrape_followers(
                    seed, max_results=15
                )
                accounts.extend(mentioned)

        elif traversal == "search" or not seed_accounts:
            # If we have seed accounts, do graph traversal anyway
            if seed_accounts:
                for seed in seed_accounts[:1]:
                    mentioned = await self.scraper.scrape_followers(
                        seed, max_results=10
                    )
                    accounts.extend(mentioned)

        # Enrich a subset with full profiles
        if accounts:
            enriched = await self._enrich_accounts(accounts, max_enrich=3)
            for i, e in enumerate(enriched):
                if i < len(accounts):
                    accounts[i] = e

        for a in accounts:
            a.niche = self.niche
            a.discovered_by = strategy.id

        return accounts

    def _process_generation_results(
        self, results: List[tuple], known_handles: set
    ) -> List[XAccount]:
        """Process all results from a generation."""
        all_accounts = []
        for _, accounts in results:
            all_accounts.extend(accounts)

        return self._process_new_accounts(all_accounts, known_handles)

    async def _finalize_session(self, total_new: int, generations: int):
        """Finalize the discovery session."""
        duration = (datetime.now() - self.session_start).total_seconds()

        self.knowledge.record_session({
            "niche": self.niche,
            "generations": generations,
            "new_accounts": total_new,
            "total_discovered": len(self.all_discovered),
            "duration_seconds": duration,
            "crisis_occurred": any(
                g.get("crisis_mode") for g in self.generation_stats
            ),
        })

        await self.scraper.close()

        logger.info(
            f"Session complete: {total_new} new accounts in {duration:.1f}s "
            f"({generations} generations)"
        )

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current/last session."""
        return {
            "niche": self.niche,
            "total_discovered": len(self.all_discovered),
            "new_this_session": len(self.new_accounts_this_session),
            "generation_stats": self.generation_stats,
            "knowledge_stats": self.knowledge.get_stats(),
        }
