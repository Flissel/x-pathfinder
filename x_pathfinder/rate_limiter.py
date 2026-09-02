"""
Adaptive rate limiter for web scraping.

Per-domain rate limiting with exponential backoff, jitter,
and domain rotation to avoid detection and bans.
"""

import asyncio
import random
import time
import logging
from typing import Dict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DomainState:
    """Tracks rate limiting state for a single domain."""

    last_request: float = 0.0
    consecutive_errors: int = 0
    backoff_seconds: float = 2.0
    total_requests: int = 0
    total_errors: int = 0
    blocked: bool = False
    blocked_until: float = 0.0


class AdaptiveRateLimiter:
    """Per-domain adaptive rate limiter with exponential backoff.

    Features:
    - Base delay between requests (default 2s)
    - Exponential backoff on errors (2x, max 60s)
    - Random jitter to avoid pattern detection
    - Domain blocking after repeated failures
    - Automatic recovery after cooldown
    """

    BASE_DELAY = 1.5
    MAX_BACKOFF = 30.0
    JITTER_RANGE = 0.5
    MAX_CONSECUTIVE_ERRORS = 8
    BLOCK_DURATION = 120.0  # 2 minutes

    def __init__(self):
        self._domains: Dict[str, DomainState] = {}
        self._lock = asyncio.Lock()

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc or parsed.hostname or url

    def _get_state(self, domain: str) -> DomainState:
        """Get or create domain state."""
        if domain not in self._domains:
            self._domains[domain] = DomainState()
        return self._domains[domain]

    async def acquire(self, url: str) -> bool:
        """Wait for rate limit clearance. Returns False if domain is blocked."""
        domain = self._get_domain(url)

        async with self._lock:
            state = self._get_state(domain)
            now = time.monotonic()

            # Check if domain is blocked
            if state.blocked:
                if now < state.blocked_until:
                    remaining = state.blocked_until - now
                    logger.warning(
                        f"Domain {domain} blocked for {remaining:.0f}s more"
                    )
                    return False
                # Unblock after cooldown
                state.blocked = False
                state.consecutive_errors = 0
                state.backoff_seconds = self.BASE_DELAY
                logger.info(f"Domain {domain} unblocked after cooldown")

            # Calculate delay with backoff and jitter
            delay = state.backoff_seconds + random.uniform(
                -self.JITTER_RANGE, self.JITTER_RANGE
            )
            delay = max(0, delay)

            elapsed = now - state.last_request
            if elapsed < delay:
                wait_time = delay - elapsed
                logger.debug(f"Rate limiting {domain}: waiting {wait_time:.1f}s")
                # Release lock while waiting
                self._lock.release()
                try:
                    await asyncio.sleep(wait_time)
                finally:
                    await self._lock.acquire()

            state.last_request = time.monotonic()
            state.total_requests += 1
            return True

    async def report_success(self, url: str):
        """Report successful request - reduce backoff."""
        domain = self._get_domain(url)
        async with self._lock:
            state = self._get_state(domain)
            state.consecutive_errors = 0
            # Gradually reduce backoff back to base
            state.backoff_seconds = max(
                self.BASE_DELAY,
                state.backoff_seconds * 0.8,
            )

    async def report_error(self, url: str, status_code: int = 0):
        """Report failed request - increase backoff."""
        domain = self._get_domain(url)
        async with self._lock:
            state = self._get_state(domain)
            state.consecutive_errors += 1
            state.total_errors += 1

            # Exponential backoff
            state.backoff_seconds = min(
                self.MAX_BACKOFF,
                state.backoff_seconds * 2,
            )

            logger.warning(
                f"Error on {domain} (#{state.consecutive_errors}): "
                f"backoff now {state.backoff_seconds:.1f}s"
            )

            # Block domain after too many consecutive errors
            if state.consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                state.blocked = True
                state.blocked_until = time.monotonic() + self.BLOCK_DURATION
                logger.warning(
                    f"Domain {domain} BLOCKED for {self.BLOCK_DURATION}s "
                    f"after {state.consecutive_errors} consecutive errors"
                )

    def get_healthy_domains(self, domains: list) -> list:
        """Filter to domains that are not blocked."""
        healthy = []
        now = time.monotonic()
        for domain in domains:
            state = self._get_state(domain)
            if not state.blocked or now >= state.blocked_until:
                healthy.append(domain)
        return healthy

    def get_stats(self) -> Dict[str, Dict]:
        """Get rate limiter statistics."""
        stats = {}
        for domain, state in self._domains.items():
            stats[domain] = {
                "total_requests": state.total_requests,
                "total_errors": state.total_errors,
                "consecutive_errors": state.consecutive_errors,
                "backoff_seconds": state.backoff_seconds,
                "blocked": state.blocked,
            }
        return stats
