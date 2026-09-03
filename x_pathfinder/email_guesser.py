"""
Email verification engine for X Pathfinder.

Tests email candidates through:
1. Email extraction from bios (direct finds)
2. MX record verification (does domain accept email?)
3. SMTP RCPT TO verification (does mailbox exist?)

This is the "execution layer" - equivalent to _test_search_strategy
in the sakana-pathfinder. The genetic algorithm generates patterns,
this module tests them.
"""

import re
import asyncio
import socket
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from .models import XAccount, EmailStrategy, EmailCandidate

logger = logging.getLogger(__name__)


class EmailVerifier:
    """Tests and verifies email candidates.

    Verification levels:
    - Level 0: Pattern generated (no verification)
    - Level 1: MX record exists (domain accepts email)
    - Level 2: SMTP accepts RCPT TO (mailbox likely exists)
    """

    SMTP_TIMEOUT = 10

    def __init__(self):
        self._mx_cache: Dict[str, bool] = {}
        self._smtp_cache: Dict[str, Optional[bool]] = {}
        self._catchall_cache: Dict[str, bool] = {}  # domain -> is_catchall

    # ── Name Parsing ──

    @staticmethod
    def parse_name(display_name: str) -> Optional[Tuple[str, str]]:
        """Parse display name into (first, last)."""
        if not display_name:
            return None

        clean = re.sub(r"[^\w\s.-]", "", display_name).strip()
        clean = re.sub(
            r"\b(Dr|Prof|PhD|Mr|Mrs|Ms|CEO|CTO|VP|Sr|Jr|Eng)\b\.?",
            "", clean, flags=re.IGNORECASE,
        ).strip()

        parts = clean.split()
        if len(parts) >= 2:
            first = parts[0].lower()
            last = parts[-1].lower()
            if len(first) >= 2 and len(last) >= 2:
                return first, last

        return None

    # ── Email Extraction ──

    @staticmethod
    def extract_emails_from_bio(bio: str) -> List[str]:
        """Extract email addresses directly from bio text."""
        if not bio:
            return []

        emails = re.findall(
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", bio
        )

        # Obfuscated: "user [at] domain [dot] com"
        obfuscated = re.findall(
            r"([a-zA-Z0-9._%+-]+)\s*[\[\(]?\s*(?:at|AT)\s*[\]\)]?\s*"
            r"([a-zA-Z0-9.-]+)\s*[\[\(]?\s*(?:dot|DOT)\s*[\]\)]?\s*"
            r"([a-zA-Z]{2,})",
            bio,
        )
        for user, domain, tld in obfuscated:
            emails.append(f"{user}@{domain}.{tld}")

        return list(set(emails))

    @staticmethod
    def extract_domains(account: XAccount) -> List[str]:
        """Extract professional domains from profile data."""
        domains = []
        text = f"{account.bio} " + " ".join(account.recent_tweets)

        urls = re.findall(r"https?://([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", text)
        skip = {
            "x.com", "twitter.com", "t.co", "youtube.com", "youtu.be",
            "linkedin.com", "facebook.com", "instagram.com", "github.com",
            "bit.ly", "linktr.ee", "ko-fi.com", "patreon.com", "medium.com",
        }
        for domain in urls:
            if domain.lower() not in skip:
                domains.append(domain.lower())

        return list(set(domains))

    # ── Strategy Execution ──

    def execute_strategy(
        self,
        strategy: EmailStrategy,
        account: XAccount,
    ) -> List[EmailCandidate]:
        """Execute an email strategy against an account.

        Generates email candidates using the strategy's patterns.
        """
        candidates = []
        name = self.parse_name(account.display_name)

        if name:
            first, last = name
            try:
                email = strategy.compile_email(first, last, account.handle)
                candidates.append(EmailCandidate(
                    email=email,
                    handle=account.handle,
                    strategy_id=strategy.id,
                ))
            except (KeyError, IndexError):
                pass

        # Also try with handle if format uses it
        if "{handle}" in strategy.get_format():
            try:
                email = strategy.compile_email(
                    account.handle, "", account.handle
                )
                candidates.append(EmailCandidate(
                    email=email,
                    handle=account.handle,
                    strategy_id=strategy.id,
                ))
            except (KeyError, IndexError):
                pass

        return candidates

    # ── Verification ──

    async def verify_mx(self, domain: str) -> bool:
        """Check if domain has MX records."""
        if domain in self._mx_cache:
            return self._mx_cache[domain]

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._mx_lookup, domain)
            self._mx_cache[domain] = result
            return result
        except Exception:
            self._mx_cache[domain] = False
            return False

    @staticmethod
    def _mx_lookup(domain: str) -> bool:
        """DNS MX lookup."""
        try:
            # Try dnspython first
            import dns.resolver
            answers = dns.resolver.resolve(domain, "MX")
            return len(answers) > 0
        except ImportError:
            # Fallback: check if port 25 is reachable
            try:
                socket.setdefaulttimeout(5)
                socket.getaddrinfo(domain, 25)
                return True
            except socket.gaierror:
                return False
        except Exception:
            return False

    async def verify_smtp(self, email: str) -> Optional[bool]:
        """Verify email via SMTP RCPT TO.

        Returns:
            True = mailbox exists
            False = mailbox rejected
            None = inconclusive (server didn't respond clearly)
        """
        if email in self._smtp_cache:
            return self._smtp_cache[email]

        domain = email.split("@")[1]

        try:
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self._smtp_check, email, domain),
                timeout=self.SMTP_TIMEOUT,
            )
            self._smtp_cache[email] = result
            return result
        except asyncio.TimeoutError:
            self._smtp_cache[email] = None
            return None
        except Exception:
            self._smtp_cache[email] = None
            return None

    @staticmethod
    def _smtp_check(email: str, domain: str) -> Optional[bool]:
        """SMTP RCPT TO verification."""
        import smtplib

        try:
            # Get MX host
            mx_host = domain
            try:
                import dns.resolver
                answers = dns.resolver.resolve(domain, "MX")
                if answers:
                    mx_host = str(answers[0].exchange).rstrip(".")
            except (ImportError, Exception):
                pass

            # Connect and test
            smtp = smtplib.SMTP(timeout=8)
            smtp.connect(mx_host, 25)
            smtp.helo("verify.local")
            smtp.mail("verify@verify.local")
            code, _ = smtp.rcpt(email)
            smtp.quit()

            if code == 250:
                return True
            elif code == 550:
                return False
            else:
                return None

        except Exception:
            return None

    async def check_catchall(self, domain: str) -> bool:
        """Check if domain is catch-all (accepts ANY address).

        Sends RCPT TO with an obviously fake address.
        If server accepts it (250), domain is catch-all.
        """
        if domain in self._catchall_cache:
            return self._catchall_cache[domain]

        import random, string
        rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=20))
        fake_email = f"xpf_catchall_test_{rand}@{domain}"

        try:
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self._smtp_check, fake_email, domain),
                timeout=self.SMTP_TIMEOUT,
            )
            is_catchall = result is True  # Server accepted fake address
            self._catchall_cache[domain] = is_catchall

            if is_catchall:
                logger.info(f"CATCH-ALL detected: {domain} (accepts any address)")
            else:
                logger.debug(f"NOT catch-all: {domain}")

            return is_catchall

        except Exception:
            self._catchall_cache[domain] = False
            return False

    async def verify_candidate(self, candidate: EmailCandidate) -> EmailCandidate:
        """Full verification pipeline for a candidate."""
        domain = candidate.email.split("@")[1]

        # Step 1: MX check
        candidate.mx_valid = await self.verify_mx(domain)

        if candidate.mx_valid:
            # Step 2: Catch-all check (once per domain)
            is_catchall = await self.check_catchall(domain)

            if is_catchall:
                # Catch-all domain - SMTP 250 means nothing
                candidate.smtp_valid = None
                candidate.confidence = 0.15  # Low confidence
                return candidate

            # Step 3: SMTP check (only if NOT catch-all)
            candidate.smtp_valid = await self.verify_smtp(candidate.email)

            if candidate.smtp_valid is True:
                candidate.confidence = 0.95  # Real verified!
            elif candidate.smtp_valid is None:
                candidate.confidence = 0.3   # Inconclusive
            else:
                candidate.confidence = 0.05  # Rejected = doesn't exist
        else:
            candidate.confidence = 0.0

        return candidate

    async def verify_batch(
        self, candidates: List[EmailCandidate], max_concurrent: int = 5
    ) -> List[EmailCandidate]:
        """Verify a batch of candidates concurrently."""
        sem = asyncio.Semaphore(max_concurrent)

        async def verify_one(c: EmailCandidate) -> EmailCandidate:
            async with sem:
                return await self.verify_candidate(c)

        results = await asyncio.gather(
            *[verify_one(c) for c in candidates],
            return_exceptions=True,
        )

        verified = []
        for r in results:
            if isinstance(r, EmailCandidate):
                verified.append(r)

        return verified

    # ── Fitness Calculation ──

    @staticmethod
    def calculate_strategy_fitness(
        candidates: List[EmailCandidate],
    ) -> float:
        """Calculate fitness score for a strategy based on its candidates.

        Only REAL verified emails (non-catchall SMTP 250) get high scores.

        Scoring:
        - SMTP verified (non-catchall): 150 points (the real deal)
        - MX valid, catch-all domain: 10 points (worthless)
        - SMTP inconclusive: 30 points
        - SMTP rejected: 5 points
        - No MX: 0 points
        """
        if not candidates:
            return 0.0

        total = 0.0
        for c in candidates:
            if c.confidence >= 0.9:
                total += 150  # Real SMTP verified, non-catchall
            elif c.confidence >= 0.3:
                total += 30   # Inconclusive
            elif c.confidence >= 0.1:
                total += 10   # Catch-all or low confidence
            elif c.mx_valid and c.smtp_valid is False:
                total += 5    # At least we know it doesn't exist

        return total / len(candidates)
