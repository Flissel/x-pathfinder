"""Independent verification of GA results.

The validator deliberately does not reuse anything the scorer concluded.
It re-fetches the cited evidence and checks the claim is actually present.
Anything it cannot confirm is refuted, never quietly passed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Verdict:
    validated: bool
    reason: str
    checked_at: datetime = field(default_factory=datetime.now)


def _default_fetcher(url: str) -> tuple[int, str]:
    import requests

    response = requests.get(url, timeout=(3, 10), allow_redirects=True)
    return response.status_code, response.text


class EvidenceValidator:
    def __init__(self, fetcher: Callable[[str], tuple[int, str]] = _default_fetcher):
        self._fetcher = fetcher

    def validate(
        self,
        handle: str,
        evidence_urls: Sequence[str],
        claim_tokens: Sequence[str],
    ) -> Verdict:
        if not evidence_urls:
            return Verdict(False, "no evidence urls to check")

        # CRITICAL 1: empty claim_tokens vacuously passes anything reachable
        # Fail closed: there is nothing to verify, so it cannot be confirmed
        if not claim_tokens or all(not t or not t.strip() for t in claim_tokens):
            return Verdict(False, "no claim tokens to verify")

        unreachable = 0
        for url in evidence_urls:
            try:
                status, body = self._fetcher(url)
                # CRITICAL 2: widen guard to catch malformed status/body
                # Coerce status and lowercase body inside the try block
                if not (200 <= int(status) < 300):
                    unreachable += 1
                    continue

                haystack = (body or "").lower()
                if all(token.lower() in haystack for token in claim_tokens):
                    return Verdict(True, f"claim confirmed at {url}")
            except Exception as exc:
                logger.debug("evidence fetch failed for %s: %s", url, exc)
                unreachable += 1
                continue

        if unreachable == len(evidence_urls):
            return Verdict(False, "all evidence urls unreachable")
        return Verdict(False, "claim not found in any reachable evidence url")


class EmailValidator:
    """Re-checks an email independently of whatever the scorer concluded.

    A missing SMTP verifier means *unknown*, which fails closed here: we
    only confirm what we could actually observe.
    """

    def __init__(self, mx_check: Callable[[str], bool], smtp_check: Callable[[str], object]):
        self._mx_check = mx_check
        self._smtp_check = smtp_check

    def validate(self, email: str) -> Verdict:
        if "@" not in email:
            return Verdict(False, "malformed email address")
        # IMPORTANT 3: require BOTH non-empty local-part and non-empty domain
        local_part, _, domain = email.rpartition("@")
        if not local_part or not domain:
            return Verdict(False, "malformed email address")
        try:
            if not self._mx_check(domain):
                return Verdict(False, f"no MX record for {domain}")
            smtp = self._smtp_check(email)
        except Exception as exc:
            return Verdict(False, f"verification failed: {exc}")

        if smtp is True:
            return Verdict(True, "MX present and SMTP accepted")
        if smtp is None:
            return Verdict(False, "SMTP result unknown, failing closed")
        return Verdict(False, "SMTP rejected the address")
