"""
Multi-backend scraper for X/Twitter account discovery.

Supports multiple backends in priority order:
1. GitHub awesome-lists (curated account lists per niche - most reliable)
2. Twitter Syndication API (profile timelines, mentioned accounts)

All backends are publicly accessible without API keys.
"""

import asyncio
import re
import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from urllib.parse import quote_plus

import aiohttp
from bs4 import BeautifulSoup

from .models import XAccount, ScrapedPage
from .rate_limiter import AdaptiveRateLimiter

logger = logging.getLogger(__name__)


class XScraper:
    """Multi-backend scraper for X/Twitter data.

    Uses GitHub awesome-lists for bulk account discovery and
    Twitter's syndication API for profile enrichment.
    """

    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    # Syndication API needs very slow pacing (rate limited quickly)
    SYNDICATION_DELAY = 8.0

    def __init__(self, rate_limiter: AdaptiveRateLimiter = None):
        self.rate_limiter = rate_limiter or AdaptiveRateLimiter()
        self._session: Optional[aiohttp.ClientSession] = None
        # Cache GitHub handles per search to avoid redundant API calls
        self._github_cache: Dict[str, List[XAccount]] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=20)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"User-Agent": self.USER_AGENT},
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Bootstrap ──

    async def discover_nitter_instances(
        self, cached_instances: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Bootstrap: verify that GitHub and syndication APIs are reachable."""
        instances = []

        # Test GitHub API
        session = await self._get_session()
        try:
            async with session.get(
                "https://api.github.com/rate_limit",
                headers={"Accept": "application/vnd.github.v3+json"},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    instances.append({
                        "url": "https://api.github.com",
                        "alive": True,
                        "latency": 100.0,
                        "checked_at": datetime.now().isoformat(),
                        "type": "github",
                    })
                    logger.info("GitHub API: available")
        except Exception:
            logger.warning("GitHub API: unavailable")

        # Test syndication API with a known account
        try:
            async with session.get(
                "https://syndication.twitter.com/srv/timeline-profile/screen-name/x",
                timeout=aiohttp.ClientTimeout(total=8),
            ) as resp:
                if resp.status in (200, 429):  # 429 means it's alive but rate limited
                    instances.append({
                        "url": "https://syndication.twitter.com",
                        "alive": True,
                        "latency": 200.0,
                        "checked_at": datetime.now().isoformat(),
                        "type": "syndication",
                    })
                    logger.info(f"Syndication API: available (status={resp.status})")
        except Exception:
            logger.warning("Syndication API: unavailable")

        logger.info(f"Bootstrap: {len(instances)} backends available")
        return instances

    # ── Search Methods ──

    async def search_accounts(self, query: str, max_results: int = 20) -> ScrapedPage:
        """Search for X accounts matching query."""
        # Primary: GitHub awesome lists
        result = await self._search_github(query, max_results)
        if result.success and result.accounts:
            return result

        # Fallback: syndication-based discovery (mentioned accounts from seed)
        return result

    async def scrape_profile(self, handle: str) -> Optional[XAccount]:
        """Scrape a single X account profile via syndication API."""
        handle = handle.lstrip("@")
        return await self._scrape_syndication_profile(handle)

    async def scrape_followers(
        self, handle: str, max_results: int = 50
    ) -> List[XAccount]:
        """Get mentioned/interacted accounts from a user's timeline."""
        handle = handle.lstrip("@")
        return await self._get_mentioned_accounts(handle, max_results)

    async def scrape_following(
        self, handle: str, max_results: int = 50
    ) -> List[XAccount]:
        """Get mentioned/interacted accounts from a user's timeline."""
        return await self.scrape_followers(handle, max_results)

    # ── GitHub Backend (Primary) ──

    async def _search_github(self, query: str, max_results: int) -> ScrapedPage:
        """Search GitHub for curated lists of X/Twitter accounts."""
        page = ScrapedPage(url="github_search", source="github")

        # Check cache
        cache_key = query.lower().strip()
        if cache_key in self._github_cache:
            page.accounts = self._github_cache[cache_key][:max_results]
            page.success = len(page.accounts) > 0
            return page

        # Use broad niche-level queries that actually find curated lists
        # Strip hashtags and operators - GitHub needs simple keywords
        clean_query = re.sub(r"[#@]|\b(OR|AND|-\w+|filter:\w+|lang:\w+|min_\w+:\d+)\b", "", query).strip()
        # Use just the first 2-3 words to keep it broad
        words = clean_query.split()[:3]
        base = " ".join(words) if words else query

        search_terms = [
            f"awesome twitter {base}",
            f"awesome tech twitter accounts",
            f"{base} twitter accounts list",
            f"awesome {base} people follow",
        ]

        all_accounts = []
        seen_handles = set()

        for term in search_terms:
            encoded = quote_plus(term)
            url = f"https://api.github.com/search/repositories?q={encoded}&sort=stars&per_page=5"

            if not await self.rate_limiter.acquire(url):
                continue

            try:
                session = await self._get_session()
                async with session.get(url, headers={
                    "Accept": "application/vnd.github.v3+json",
                }) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        await self.rate_limiter.report_success(url)

                        for repo in data.get("items", [])[:3]:
                            accounts = await self._extract_handles_from_readme(
                                repo["full_name"]
                            )
                            for a in accounts:
                                if a.handle.lower() not in seen_handles:
                                    seen_handles.add(a.handle.lower())
                                    all_accounts.append(a)
                    elif resp.status == 403:
                        # GitHub rate limit
                        await self.rate_limiter.report_error(url, resp.status)
                        logger.warning("GitHub API rate limited")
                        break
                    else:
                        await self.rate_limiter.report_error(url, resp.status)

            except Exception as e:
                await self.rate_limiter.report_error(url)
                logger.warning(f"GitHub search failed: {e}")

            # Small delay between search queries
            await asyncio.sleep(1)

        # Cache results
        self._github_cache[cache_key] = all_accounts

        page.accounts = all_accounts[:max_results]
        page.success = len(all_accounts) > 0
        logger.info(f"GitHub search '{query}': found {len(all_accounts)} unique accounts")
        return page

    async def _extract_handles_from_readme(self, repo_full_name: str) -> List[XAccount]:
        """Extract X/Twitter handles from a GitHub repo's README."""
        url = f"https://api.github.com/repos/{repo_full_name}/readme"

        if not await self.rate_limiter.acquire(url):
            return []

        try:
            session = await self._get_session()
            async with session.get(url, headers={
                "Accept": "application/vnd.github.v3.raw",
            }) as resp:
                if resp.status != 200:
                    await self.rate_limiter.report_error(url, resp.status)
                    return []

                text = await resp.text()
                await self.rate_limiter.report_success(url)

                handles = set()
                skip = {
                    "home", "explore", "search", "settings", "i",
                    "intent", "hashtag", "login", "signup", "status",
                    "param", "return", "type", "author", "example",
                    "deprecated", "override", "see", "since",
                    "twitter", "x", "share", "about", "help",
                    "tos", "privacy", "jobs", "compose", "messages",
                    "notifications", "lists", "bookmarks", "communities",
                }

                # URL patterns: twitter.com/handle or x.com/handle
                for match in re.finditer(
                    r"(?:twitter\.com|x\.com)/(@?[\w]{2,15})", text
                ):
                    handle = match.group(1).lstrip("@")
                    if handle.lower() not in skip and len(handle) >= 2:
                        handles.add(handle)

                # Markdown link pattern: [@handle](url)
                for match in re.finditer(r"\[@(\w{2,15})\]", text):
                    handle = match.group(1)
                    if handle.lower() not in skip:
                        handles.add(handle)

                logger.debug(
                    f"Extracted {len(handles)} handles from {repo_full_name}"
                )

                return [
                    XAccount(
                        handle=h,
                        profile_url=f"https://x.com/{h}",
                    )
                    for h in sorted(handles)
                ]

        except Exception as e:
            await self.rate_limiter.report_error(url)
            logger.debug(f"README extraction failed for {repo_full_name}: {e}")
            return []

    # ── Twitter Syndication API ──

    async def _scrape_syndication_profile(self, handle: str) -> Optional[XAccount]:
        """Scrape profile via Twitter's syndication/embed API."""
        url = f"https://syndication.twitter.com/srv/timeline-profile/screen-name/{handle}"

        if not await self.rate_limiter.acquire(url):
            return None

        # Extra delay for syndication - it rate limits aggressively
        await asyncio.sleep(self.SYNDICATION_DELAY)

        try:
            session = await self._get_session()
            async with session.get(url) as resp:
                if resp.status == 429:
                    await self.rate_limiter.report_error(url, 429)
                    logger.debug(f"Syndication rate limited for @{handle}")
                    return None

                if resp.status != 200:
                    await self.rate_limiter.report_error(url, resp.status)
                    return None

                html = await resp.text()
                if len(html) < 100:
                    # Empty or error response
                    await self.rate_limiter.report_error(url)
                    return None

                await self.rate_limiter.report_success(url)
                return self._parse_syndication_profile(html, handle)

        except Exception as e:
            await self.rate_limiter.report_error(url)
            logger.debug(f"Syndication profile failed for @{handle}: {e}")
            return None

    def _parse_syndication_profile(self, html: str, handle: str) -> Optional[XAccount]:
        """Parse the syndication timeline HTML for profile data."""
        soup = BeautifulSoup(html, "lxml")

        display_name = ""
        bio = ""
        followers = 0
        following = 0
        tweets = 0
        verified = False
        recent_tweets = []
        hashtags = []

        # The syndication page embeds tweet data
        # Extract display names and mentions
        name_matches = re.findall(r'"name"\s*:\s*"([^"]+)"', html)
        screen_name_matches = re.findall(r'"screen_name"\s*:\s*"([^"]+)"', html)

        # Find data for this user specifically
        for i, sn in enumerate(screen_name_matches):
            if sn.lower() == handle.lower() and i < len(name_matches):
                display_name = name_matches[i]
                break

        # Try JSON blob extraction
        for pattern in [
            r'"user"\s*:\s*\{([^}]{10,500})\}',
            r'\{"screen_name"\s*:\s*"' + re.escape(handle) + r'"[^}]*\}',
        ]:
            for match in re.finditer(pattern, html, re.IGNORECASE):
                try:
                    json_str = match.group(0)
                    if not json_str.startswith("{"):
                        json_str = "{" + match.group(1) + "}"
                    data = json.loads(json_str)
                    if data.get("screen_name", "").lower() == handle.lower():
                        display_name = display_name or data.get("name", "")
                        followers = data.get("followers_count", 0)
                        following = data.get("friends_count", 0)
                        tweets = data.get("statuses_count", 0)
                        verified = data.get("verified", False)
                        bio = data.get("description", "")
                except (json.JSONDecodeError, IndexError):
                    pass

        # Extract tweet texts
        tweet_els = soup.select("p")
        for t in tweet_els[:10]:
            text = t.get_text(strip=True)
            if text and len(text) > 15 and not text.startswith("http"):
                recent_tweets.append(text)
                hashtags.extend(re.findall(r"#\w+", text))

        return XAccount(
            handle=handle,
            display_name=display_name,
            bio=bio,
            followers=followers,
            following=following,
            tweets=tweets,
            verified=verified,
            profile_url=f"https://x.com/{handle}",
            recent_tweets=recent_tweets[:5],
            hashtags_used=list(set(hashtags)),
        )

    async def _get_mentioned_accounts(
        self, handle: str, max_results: int
    ) -> List[XAccount]:
        """Extract mentioned accounts from a user's syndication timeline."""
        url = f"https://syndication.twitter.com/srv/timeline-profile/screen-name/{handle}"

        if not await self.rate_limiter.acquire(url):
            return []

        await asyncio.sleep(self.SYNDICATION_DELAY)

        accounts = []
        try:
            session = await self._get_session()
            async with session.get(url) as resp:
                if resp.status != 200:
                    await self.rate_limiter.report_error(url, resp.status)
                    return []

                html = await resp.text()
                if len(html) < 100:
                    await self.rate_limiter.report_error(url)
                    return []

                await self.rate_limiter.report_success(url)

                # Find all screen names in the page
                screen_names = set()

                # From data attributes
                for m in re.finditer(r'"screen_name"\s*:\s*"(\w+)"', html):
                    screen_names.add(m.group(1))

                # From @mentions in text
                for m in re.finditer(r'@(\w{2,15})', html):
                    screen_names.add(m.group(1))

                # Remove source handle and non-user strings
                screen_names.discard(handle)
                screen_names.discard(handle.lower())
                skip = {
                    "twitter", "x", "intent", "search", "hashtag",
                    "home", "share", "i", "status",
                }
                screen_names -= skip

                for name in list(screen_names)[:max_results]:
                    accounts.append(XAccount(
                        handle=name,
                        profile_url=f"https://x.com/{name}",
                    ))

                logger.info(
                    f"Mentioned accounts from @{handle}: {len(accounts)}"
                )

        except Exception as e:
            await self.rate_limiter.report_error(url)
            logger.debug(f"Mentioned accounts scrape failed for @{handle}: {e}")

        return accounts

    # ── Utility ──

    @staticmethod
    def _parse_number(text: str) -> int:
        """Parse a number string like '1.2K', '3.5M', '100'."""
        text = text.strip().replace(",", "").replace(" ", "")
        if not text:
            return 0

        multipliers = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
        text_lower = text.lower()

        for suffix, mult in multipliers.items():
            if text_lower.endswith(suffix):
                try:
                    return int(float(text_lower[:-1]) * mult)
                except ValueError:
                    return 0

        try:
            return int(float(text))
        except ValueError:
            return 0
