"""
Background daemon for continuous email discovery.

Runs indefinitely:
1. Harvests accounts from GitHub (replenishes targets)
2. Evolves email patterns against all accounts
3. Verifies via MX + SMTP
4. Writes everything to SQLite DB
5. Loops until stopped

Usage:
    python -m x_pathfinder run
    python -m x_pathfinder run --no-smtp --workers 20
"""

import asyncio
import signal
import logging
import time
import re
from typing import List, Optional
from datetime import datetime

from .database import EmailDatabase
from .harvester import MassHarvester
from .email_genome import EmailGeneticEngine, EMAIL_FORMATS, DOMAINS_COMMON
from .email_guesser import EmailVerifier
from .models import XAccount, EmailStrategy, EmailPattern, EmailCandidate
from .rate_limiter import AdaptiveRateLimiter
from .names import NameGenerator, COUNTRY_DOMAINS
from .dashboard import push_event
from .geo import detect_country
from .worker_client import WorkerClient

logger = logging.getLogger(__name__)


class EmailDaemon:
    """Continuous background email discovery process.

    Runs in a loop:
    1. Harvest new accounts (if pool is low)
    2. Take batch of unprocessed accounts
    3. Generate emails with evolved strategies
    4. Verify MX + SMTP in parallel
    5. Write to DB
    6. Evolve strategies based on results
    7. Repeat
    """

    BATCH_SIZE = 10        # Names per batch - keep small for fast SMTP cycles
    NAMES_PER_BATCH = 10

    def __init__(
        self,
        workers: int = 20,
        db_path: str = None,
        countries: List[str] = None,
    ):
        self.workers = workers
        self.verify_smtp = True  # Always verify - no point storing unverified
        self.db = EmailDatabase(db_path)
        self.rate_limiter = AdaptiveRateLimiter()
        self.genome = EmailGeneticEngine()
        self.verifier = EmailVerifier()
        self.countries = countries or ["US", "DE", "GB", "FR", "CH", "RU", "IN"]
        self.name_gen = NameGenerator(countries=self.countries)
        self.worker_client = WorkerClient()

        self._running = False
        self._stats = {
            "started_at": None,
            "accounts_processed": 0,
            "emails_generated": 0,
            "emails_mx_verified": 0,
            "emails_smtp_verified": 0,
            "generations": 0,
        }

    async def run(self, on_progress: callable = None):
        """Run the daemon until stopped."""
        self._running = True
        self._stats["started_at"] = datetime.now().isoformat()
        run_id = self.db.start_run()

        # Check remote workers
        if self.worker_client.available:
            health = await self.worker_client.health_check()
            healthy = sum(1 for v in health.values() if v)
            logger.info(
                f"Remote workers: {healthy}/{len(health)} healthy"
            )
        else:
            logger.info("No remote workers configured - using local SMTP only")

        logger.info(
            f"Daemon started (local_workers={self.workers}, "
            f"remote_workers={len(self.worker_client.workers)})"
        )

        # Handle graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        population = self.genome.create_initial_population()

        try:
            while self._running:
                # Generate names batch directly - no GitHub needed
                names_batch = self.name_gen.generate_batch(self.NAMES_PER_BATCH)

                # Create XAccount objects from names
                accounts = []
                for first, last, country in names_batch:
                    handle = f"{first}{last}"
                    accounts.append(XAccount(
                        handle=handle,
                        display_name=f"{first.title()} {last.title()}",
                        niche=country,
                    ))
                    # Also add to DB
                    self.db.add_account(
                        handle=handle,
                        display_name=f"{first.title()} {last.title()}",
                        niche=country,
                        source="namegen",
                    )

                # Generate + verify emails using evolved strategies
                results = await self._process_batch_names(
                    names_batch, population
                )

                # Write to DB
                self._save_results(results)

                # Evolve strategies
                verified_any = any(
                    any(c.confidence >= 0.9 for c in cands)
                    for _, cands in results
                )
                crisis = not verified_any
                population = self.genome.evolve(
                    population, crisis_mode=crisis
                )
                self._stats["generations"] += 1

                # Progress
                db_stats = self.db.get_stats()
                db_stats["generation"] = self._stats["generations"]
                if on_progress:
                    on_progress(self._stats, db_stats)

                self._print_status(db_stats)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt - shutting down...")
        finally:
            await self.worker_client.close()
            self.db.end_run(
                run_id,
                self._stats["accounts_processed"],
                self._stats["emails_generated"],
                self._stats["emails_mx_verified"],
            )
            self.db.close()
            logger.info("Daemon stopped.")

    def _stop(self):
        logger.info("Stop signal received...")
        self._running = False

    async def _harvest_accounts(self, on_progress: callable = None):
        """Harvest new accounts from GitHub."""
        logger.info("Harvesting new accounts...")

        harvester = MassHarvester(rate_limiter=self.rate_limiter)

        def harvest_progress(msg):
            logger.info(f"Harvest: {msg}")
            if on_progress:
                on_progress(self._stats, {"harvest": msg})

        accounts = await harvester.harvest_all(on_progress=harvest_progress)

        # Bulk insert
        bulk = [
            {
                "handle": a.handle.lower(),
                "display_name": a.display_name,
                "bio": a.bio,
                "followers": a.followers,
                "niche": a.niche or "mixed",
                "source": "github",
            }
            for a in accounts
        ]
        self.db.add_accounts_bulk(bulk)
        logger.info(f"Harvested {len(accounts)} accounts into DB")

    async def _process_batch_names(
        self,
        names_batch: list,
        population: List[EmailStrategy],
    ) -> List[tuple]:
        """Process a batch of names with all strategies."""
        results = []
        sem = asyncio.Semaphore(self.workers)

        async def process_strategy(strategy: EmailStrategy):
            async with sem:
                candidates = []
                fmt = strategy.get_format()
                domain = strategy.get_domain()

                for first, last, country in names_batch:
                    try:
                        email_local = fmt.format(
                            first=first, last=last,
                            f=first[0] if first else "",
                            l=last[0] if last else "",
                            handle=f"{first}{last}",
                            first_last=f"{first}{last}",
                        )
                        email = f"{email_local}@{domain}"

                        if self.db.email_exists(email):
                            continue

                        candidates.append(EmailCandidate(
                            email=email,
                            handle=f"{first}{last}",
                            strategy_id=strategy.id,
                        ))
                    except (KeyError, IndexError):
                        continue

                if not candidates:
                    return strategy, []

                # Verify MX + catchall (once per domain, skip known catchalls)
                mx_valid = await self.verifier.verify_mx(domain)
                if not mx_valid:
                    return strategy, []

                is_catchall = await self.verifier.check_catchall(domain)
                if is_catchall:
                    return strategy, []

                # Adaptive verification: remote workers first, local SMTP fallback
                for c in candidates:
                    c.mx_valid = True
                    verified = False

                    # Try remote worker first (can verify blocked providers)
                    if self.worker_client.available:
                        result = await self.worker_client.verify_email(c.email)
                        if result and result.get("code") != -1:
                            if result.get("catch_all"):
                                return strategy, []
                            c.smtp_valid = result.get("exists")
                            verified = True

                    # Fallback: local SMTP
                    if not verified:
                        c.smtp_valid = await self.verifier.verify_smtp(c.email)

                    if c.smtp_valid is True:
                        c.confidence = 0.95
                    elif c.smtp_valid is False:
                        c.confidence = 0.05
                    else:
                        c.confidence = 0.3

                strategy.fitness = self.verifier.calculate_strategy_fitness(candidates)
                strategy.emails_verified = sum(1 for c in candidates if c.mx_valid)
                return strategy, candidates

        tasks = [asyncio.create_task(process_strategy(s)) for s in population]
        completed = await asyncio.gather(*tasks, return_exceptions=True)
        for r in completed:
            if isinstance(r, tuple):
                results.append(r)
        return results

    async def _process_batch(
        self,
        accounts: List[XAccount],
        population: List[EmailStrategy],
    ) -> List[tuple]:
        """Process a batch: generate emails, verify, score strategies."""
        results = []
        sem = asyncio.Semaphore(self.workers)

        async def process_strategy(strategy: EmailStrategy):
            async with sem:
                candidates = []
                for account in accounts:
                    cands = self.verifier.execute_strategy(strategy, account)
                    candidates.extend(cands)

                # Skip already known emails
                new_candidates = [
                    c for c in candidates
                    if not self.db.email_exists(c.email)
                ]

                if not new_candidates:
                    return strategy, []

                # Verify MX + catch-all per domain, then SMTP per email
                domains = set(c.email.split("@")[1] for c in new_candidates)
                mx_cache = {}
                catchall_cache = {}
                for domain in domains:
                    mx_cache[domain] = await self.verifier.verify_mx(domain)
                    if mx_cache[domain] and self.verify_smtp:
                        catchall_cache[domain] = await self.verifier.check_catchall(domain)

                for c in new_candidates:
                    d = c.email.split("@")[1]
                    c.mx_valid = mx_cache.get(d, False)

                    if not c.mx_valid:
                        c.confidence = 0.0
                        continue

                    if catchall_cache.get(d, False):
                        # Catch-all: SMTP means nothing
                        c.smtp_valid = None
                        c.confidence = 0.15
                        continue

                    if self.verify_smtp:
                        c.smtp_valid = await self.verifier.verify_smtp(c.email)
                        if c.smtp_valid is True:
                            c.confidence = 0.95
                        elif c.smtp_valid is None:
                            c.confidence = 0.3
                        else:
                            c.confidence = 0.05
                    else:
                        c.confidence = 0.4

                # Score strategy
                strategy.fitness = self.verifier.calculate_strategy_fitness(
                    new_candidates
                )
                strategy.emails_verified = sum(
                    1 for c in new_candidates if c.mx_valid
                )

                return strategy, new_candidates

        tasks = [
            asyncio.create_task(process_strategy(s))
            for s in population
        ]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for r in completed:
            if isinstance(r, tuple):
                results.append(r)

        self._stats["accounts_processed"] += len(accounts)
        return results

    def _save_results(self, results: List[tuple]):
        """Save results to database."""
        bulk_emails = []

        for strategy, candidates in results:
            for c in candidates:
                domain = c.email.split("@")[1]
                smtp_int = (
                    1 if c.smtp_valid is True
                    else (0 if c.smtp_valid is False else -1)
                )
                country = detect_country(domain) if domain else "XX"
                bulk_emails.append({
                    "email": c.email.lower(),
                    "handle": c.handle.lower(),
                    "confidence": c.confidence,
                    "mx_valid": bool(c.mx_valid),
                    "smtp_valid": smtp_int,
                    "strategy_id": c.strategy_id,
                    "domain": domain,
                    "country": country,
                })

                if c.mx_valid:
                    self._stats["emails_mx_verified"] += 1
                if c.smtp_valid is True:
                    self._stats["emails_smtp_verified"] += 1

                # Push to dashboard
                push_event("email", {
                    "email": c.email,
                    "handle": c.handle,
                    "mx_valid": c.mx_valid,
                    "smtp_valid": c.smtp_valid is True,
                    "confidence": c.confidence,
                    "country": country,
                })

            # Record successful strategy
            if strategy.fitness > 0:
                self.db.record_strategy(
                    strategy.id,
                    strategy.get_format(),
                    strategy.get_domain(),
                    strategy.fitness,
                )

        if bulk_emails:
            # Only keep verified emails (confidence >= 0.9 = SMTP confirmed, non-catchall)
            seen = set()
            verified_only = []
            for e in bulk_emails:
                if e["email"] not in seen and e["confidence"] >= 0.9:
                    seen.add(e["email"])
                    verified_only.append(e)

            if verified_only:
                self.db.add_emails_bulk(verified_only)
                self._stats["emails_generated"] += len(verified_only)
                logger.info(f"Saved {len(verified_only)} verified emails to DB")

    def _print_status(self, db_stats: dict):
        """Print current status and push to dashboard."""
        elapsed = ""
        if self._stats["started_at"]:
            start = datetime.fromisoformat(self._stats["started_at"])
            mins = (datetime.now() - start).total_seconds() / 60
            elapsed = f" ({mins:.0f}min)"

        db_stats["generation"] = self._stats["generations"]

        logger.info(
            f"STATUS{elapsed}: "
            f"accounts={db_stats['accounts']}, "
            f"emails={db_stats['emails_total']}, "
            f"MX={db_stats['emails_mx_verified']}, "
            f"SMTP={db_stats['emails_smtp_verified']}, "
            f"gen={self._stats['generations']}"
        )

        push_event("stats", db_stats)
