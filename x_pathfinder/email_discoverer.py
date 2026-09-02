"""
Email Discoverer - Main orchestrator for evolutionary email discovery.

Like the sakana-pathfinder evolves CLI commands to find files,
this evolves email patterns to find valid email addresses for
X/Twitter accounts.

Flow:
1. Bootstrap: Find X accounts (GitHub, seeds)
2. Extract: Get bio emails and domains from profiles
3. Evolve: Genetic algorithm evolves email patterns
   a. Generate emails using strategy patterns + account names
   b. Verify via MX records and SMTP
   c. Score strategy fitness based on verification results
   d. Evolve: elitism -> selection -> crossover -> mutation
4. Persist: Store successful patterns and verified emails
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .models import XAccount, EmailStrategy, EmailCandidate
from .scraper import XScraper
from .email_genome import EmailGeneticEngine
from .email_guesser import EmailVerifier
from .knowledge_accumulator import KnowledgeAccumulator
from .rate_limiter import AdaptiveRateLimiter

logger = logging.getLogger(__name__)


class EmailDiscoverer:
    """Discovers email addresses through evolutionary pattern guessing.

    Usage::

        discoverer = EmailDiscoverer(niche="ai")
        results = await discoverer.run(generations=10)
        for email in results:
            print(f"{email.email} (confidence={email.confidence})")
    """

    CRISIS_THRESHOLD = 3

    def __init__(
        self,
        niche: str = "ai",
        max_concurrent: int = 5,
        verify_smtp: bool = True,
        knowledge_dir: str = None,
    ):
        self.niche = niche.lower()
        self.max_concurrent = max_concurrent
        self.verify_smtp = verify_smtp

        self.rate_limiter = AdaptiveRateLimiter()
        self.scraper = XScraper(rate_limiter=self.rate_limiter)
        self.genome = EmailGeneticEngine(niche=self.niche)
        self.verifier = EmailVerifier()
        self.knowledge = KnowledgeAccumulator(knowledge_dir=knowledge_dir)

        self.session_start: Optional[datetime] = None
        self.all_emails: Dict[str, EmailCandidate] = {}
        self.target_accounts: List[XAccount] = []
        self.generation_stats: List[Dict[str, Any]] = []

    async def run(
        self,
        generations: int = 10,
        accounts: List[XAccount] = None,
        on_progress: callable = None,
    ) -> List[EmailCandidate]:
        """Run evolutionary email discovery.

        Args:
            generations: Max evolution generations.
            accounts: Pre-supplied accounts (skip bootstrap if provided).
            on_progress: Callback(generation, stats).

        Returns:
            List of EmailCandidate sorted by confidence.
        """
        self.session_start = datetime.now()

        # Step 1: Get target accounts
        if accounts:
            self.target_accounts = accounts
        else:
            self.target_accounts = await self._bootstrap_accounts()

        logger.info(f"Targets: {len(self.target_accounts)} accounts")

        # Step 2: Extract direct emails and domains from bios
        direct_count = self._extract_direct_info()
        logger.info(
            f"Direct extraction: {direct_count} emails, "
            f"{len(self.genome.discovered_domains)} domains"
        )

        # Step 3: Initialize population
        population = self.genome.create_initial_population()
        logger.info(f"Initial population: {len(population)} strategies")

        # Step 4: Evolution loop
        crisis_mode = False
        generations_without_new = 0

        for gen in range(generations):
            gen_start = datetime.now()

            # Execute: generate emails and verify
            results = await self._execute_generation(population)

            # Count new verified emails
            new_verified = 0
            for strategy, candidates in results:
                for c in candidates:
                    key = c.email.lower()
                    if key not in self.all_emails and c.mx_valid:
                        self.all_emails[key] = c
                        new_verified += 1

                # Update strategy fitness
                strategy.fitness = self.verifier.calculate_strategy_fitness(
                    candidates
                )
                strategy.emails_verified = sum(
                    1 for c in candidates if c.mx_valid
                )

            # Crisis detection
            if new_verified > 0:
                generations_without_new = 0
                crisis_mode = False
            else:
                generations_without_new += 1
                if generations_without_new >= self.CRISIS_THRESHOLD:
                    crisis_mode = True

            gen_duration = (datetime.now() - gen_start).total_seconds()
            gen_stats = {
                "generation": gen + 1,
                "new_verified": new_verified,
                "total_verified": len(self.all_emails),
                "best_fitness": max(
                    (s.fitness for s in population), default=0
                ),
                "crisis_mode": crisis_mode,
                "duration_seconds": gen_duration,
            }
            self.generation_stats.append(gen_stats)

            logger.info(
                f"Gen {gen + 1}/{generations}: "
                f"+{new_verified} verified, "
                f"total={len(self.all_emails)}, "
                f"best_fitness={gen_stats['best_fitness']:.1f}"
                f"{' [CRISIS]' if crisis_mode else ''}"
            )

            if on_progress:
                on_progress(gen + 1, gen_stats)

            # Evolve
            population = self.genome.evolve(population, crisis_mode=crisis_mode)

        # Finalize
        await self._finalize(generations)

        return sorted(
            self.all_emails.values(),
            key=lambda e: e.confidence,
            reverse=True,
        )

    async def _bootstrap_accounts(self) -> List[XAccount]:
        """Bootstrap: get accounts from GitHub and knowledge base."""
        # Check APIs
        instances = await self.scraper.discover_nitter_instances()
        if instances:
            self.knowledge.save_nitter_instances(instances)

        # Get accounts from knowledge base
        known = self.knowledge.get_top_accounts(niche=self.niche, limit=100)
        accounts = [XAccount.from_dict(a) for a in known]

        # If no known accounts, discover via GitHub
        if len(accounts) < 10:
            niche_queries = {
                "ai": "artificial intelligence machine learning",
                "crypto": "cryptocurrency blockchain",
                "tech": "technology programming developer",
                "security": "cybersecurity infosec",
            }
            query = niche_queries.get(self.niche, self.niche)
            page = await self.scraper.search_accounts(query, max_results=50)
            if page.success:
                accounts.extend(page.accounts)

        # Add genome seed accounts
        from .genome import NICHE_SEEDS
        seeds = NICHE_SEEDS.get(self.niche, {})
        for handle in seeds.get("seed_accounts", []):
            accounts.append(XAccount(
                handle=handle,
                profile_url=f"https://x.com/{handle}",
            ))

        # Deduplicate
        seen = set()
        unique = []
        for a in accounts:
            if a.handle.lower() not in seen:
                seen.add(a.handle.lower())
                unique.append(a)

        await self.scraper.close()
        return unique

    def _extract_direct_info(self) -> int:
        """Extract emails and domains directly from bios."""
        direct_count = 0
        all_domains = []

        for account in self.target_accounts:
            # Direct emails
            emails = self.verifier.extract_emails_from_bio(account.bio)
            for email in emails:
                self.all_emails[email.lower()] = EmailCandidate(
                    email=email,
                    handle=account.handle,
                    confidence=0.95,
                    mx_valid=True,  # Assume valid if in bio
                    strategy_id="bio_direct",
                )
                direct_count += 1

            # Domains for genome
            domains = self.verifier.extract_domains(account)
            all_domains.extend(domains)

        self.genome.register_domains(all_domains)
        return direct_count

    async def _execute_generation(
        self, population: List[EmailStrategy]
    ) -> List[tuple]:
        """Execute all strategies against all target accounts."""
        results = []
        sem = asyncio.Semaphore(self.max_concurrent)

        async def execute_one(strategy: EmailStrategy):
            async with sem:
                all_candidates = []
                for account in self.target_accounts:
                    candidates = self.verifier.execute_strategy(
                        strategy, account
                    )
                    all_candidates.extend(candidates)

                # Verify candidates (MX + optional SMTP)
                if all_candidates:
                    # Only verify unique domains first
                    unique_domains = set(
                        c.email.split("@")[1] for c in all_candidates
                    )
                    mx_results = {}
                    for domain in unique_domains:
                        mx_results[domain] = await self.verifier.verify_mx(domain)

                    for c in all_candidates:
                        domain = c.email.split("@")[1]
                        c.mx_valid = mx_results.get(domain, False)

                        if c.mx_valid:
                            c.confidence = 0.5
                            if self.verify_smtp:
                                c.smtp_valid = await self.verifier.verify_smtp(
                                    c.email
                                )
                                if c.smtp_valid is True:
                                    c.confidence = 0.9
                                elif c.smtp_valid is False:
                                    c.confidence = 0.1

                return strategy, all_candidates

        tasks = [asyncio.create_task(execute_one(s)) for s in population]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for r in completed:
            if isinstance(r, tuple):
                results.append(r)

        return results

    async def _finalize(self, generations: int):
        """Finalize session."""
        duration = (datetime.now() - self.session_start).total_seconds()

        # Save verified emails to knowledge base
        for email_key, candidate in self.all_emails.items():
            if candidate.mx_valid:
                self.knowledge.record_keyword_result(
                    candidate.email, "email", 1
                )

        self.knowledge.record_session({
            "niche": self.niche,
            "type": "email_discovery",
            "generations": generations,
            "accounts_targeted": len(self.target_accounts),
            "emails_found": len(self.all_emails),
            "verified_emails": sum(
                1 for e in self.all_emails.values() if e.mx_valid
            ),
            "duration_seconds": duration,
        })

        logger.info(
            f"Email discovery complete: {len(self.all_emails)} emails "
            f"in {duration:.1f}s ({generations} generations)"
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get session summary."""
        mx_valid = sum(1 for e in self.all_emails.values() if e.mx_valid)
        smtp_valid = sum(
            1 for e in self.all_emails.values() if e.smtp_valid is True
        )
        return {
            "niche": self.niche,
            "accounts_targeted": len(self.target_accounts),
            "total_emails": len(self.all_emails),
            "mx_verified": mx_valid,
            "smtp_verified": smtp_valid,
            "generation_stats": self.generation_stats,
        }
