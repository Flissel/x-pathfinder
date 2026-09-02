"""
Mass account harvester - pulls thousands of X handles from multiple sources.

Sources:
1. GitHub awesome-lists (all niches, tech, AI, crypto, security, etc.)
2. GitHub user profiles (contributors to popular repos)
3. Graph traversal of known handles via syndication
"""

import re
import asyncio
import logging
from typing import List, Set, Dict
from urllib.parse import quote_plus

import aiohttp

from .models import XAccount
from .rate_limiter import AdaptiveRateLimiter

logger = logging.getLogger(__name__)

# Broad GitHub search queries to find repos with X/Twitter handles
GITHUB_QUERIES = [
    "awesome twitter accounts",
    "awesome tech twitter",
    "awesome ai twitter",
    "awesome machine learning people",
    "awesome crypto twitter",
    "awesome infosec twitter",
    "awesome cybersecurity people follow",
    "awesome developers twitter",
    "awesome startup founders twitter",
    "awesome data science twitter",
    "twitter accounts follow developers",
    "twitter accounts follow researchers",
    "people to follow twitter tech",
    "must follow twitter ai",
    "best twitter accounts programming",
    "twitter influencers technology",
    "awesome blockchain twitter",
    "awesome fintech twitter",
    "awesome devops twitter",
    "awesome web3 twitter",
    "awesome python twitter",
    "awesome javascript developers",
    "awesome rust community",
    "awesome golang people",
    "twitter list science",
    "twitter list academics",
    "twitter list entrepreneurs",
    "awesome vc twitter",
    "awesome design twitter",
    "awesome product twitter",
]

# Additional repos known to contain large handle lists
KNOWN_REPOS = [
    "shanmukh05/awesome-tech-twitter-accounts",
    "hridaydutta123/awesome-twitter-tools",
]


class MassHarvester:
    """Harvests X/Twitter handles at scale from GitHub."""

    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    def __init__(self, rate_limiter: AdaptiveRateLimiter = None):
        self.rate_limiter = rate_limiter or AdaptiveRateLimiter()
        self._session = None
        self.all_handles: Set[str] = set()
        self.accounts: List[XAccount] = []
        self._repos_searched: Set[str] = set()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=20),
                headers={"User-Agent": self.USER_AGENT},
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def harvest_all(
        self, on_progress: callable = None
    ) -> List[XAccount]:
        """Run full harvest across all sources."""
        logger.info("Starting mass harvest...")

        # 1. Search known repos first
        for repo in KNOWN_REPOS:
            handles = await self._extract_from_repo(repo)
            self._add_handles(handles)
            if on_progress:
                on_progress(f"Known repo {repo}: +{len(handles)}")

        # 2. Search GitHub for awesome lists
        for i, query in enumerate(GITHUB_QUERIES):
            repos = await self._search_github_repos(query)
            for repo_name in repos:
                if repo_name not in self._repos_searched:
                    handles = await self._extract_from_repo(repo_name)
                    self._add_handles(handles)

            if on_progress:
                on_progress(
                    f"Query {i+1}/{len(GITHUB_QUERIES)}: "
                    f"total={len(self.all_handles)} handles"
                )

            # Small delay between searches
            await asyncio.sleep(1)

        # 3. Try GitHub user search for tech profiles with twitter links
        gh_users = await self._search_github_users()
        self._add_handles(gh_users)
        if on_progress:
            on_progress(f"GitHub users: +{len(gh_users)}")

        logger.info(f"Harvest complete: {len(self.all_handles)} unique handles")
        await self.close()
        return self.accounts

    async def _search_github_repos(self, query: str) -> List[str]:
        """Search GitHub for repos matching query."""
        encoded = quote_plus(query)
        url = f"https://api.github.com/search/repositories?q={encoded}&sort=stars&per_page=10"

        if not await self.rate_limiter.acquire(url):
            return []

        try:
            session = await self._get_session()
            async with session.get(url, headers={
                "Accept": "application/vnd.github.v3+json",
            }) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    await self.rate_limiter.report_success(url)
                    return [
                        item["full_name"]
                        for item in data.get("items", [])
                    ]
                elif resp.status == 403:
                    await self.rate_limiter.report_error(url, 403)
                    logger.warning("GitHub rate limited - waiting...")
                    await asyncio.sleep(30)
                else:
                    await self.rate_limiter.report_error(url, resp.status)
        except Exception as e:
            await self.rate_limiter.report_error(url)
            logger.debug(f"GitHub search failed: {e}")

        return []

    async def _extract_from_repo(self, repo_name: str) -> Set[str]:
        """Extract X handles from a repo's README."""
        self._repos_searched.add(repo_name)
        url = f"https://api.github.com/repos/{repo_name}/readme"

        if not await self.rate_limiter.acquire(url):
            return set()

        try:
            session = await self._get_session()
            async with session.get(url, headers={
                "Accept": "application/vnd.github.v3.raw",
            }) as resp:
                if resp.status != 200:
                    await self.rate_limiter.report_error(url, resp.status)
                    return set()

                text = await resp.text()
                await self.rate_limiter.report_success(url)
                return self._extract_handles_from_text(text)

        except Exception as e:
            await self.rate_limiter.report_error(url)
            return set()

    async def _search_github_users(self) -> Set[str]:
        """Search GitHub users who have twitter usernames in their profiles."""
        handles = set()
        queries = [
            "type:user followers:>1000 language:python",
            "type:user followers:>500 language:javascript",
            "type:user followers:>500 language:rust",
        ]

        for query in queries:
            encoded = quote_plus(query)
            url = f"https://api.github.com/search/users?q={encoded}&per_page=30"

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

                        for user in data.get("items", []):
                            # Get user profile for twitter handle
                            twitter = await self._get_user_twitter(
                                user["login"]
                            )
                            if twitter:
                                handles.add(twitter)
                    else:
                        await self.rate_limiter.report_error(url, resp.status)

            except Exception:
                await self.rate_limiter.report_error(url)

            await asyncio.sleep(2)

        return handles

    async def _get_user_twitter(self, username: str) -> str:
        """Get a GitHub user's Twitter/X handle from their profile."""
        url = f"https://api.github.com/users/{username}"

        if not await self.rate_limiter.acquire(url):
            return ""

        try:
            session = await self._get_session()
            async with session.get(url, headers={
                "Accept": "application/vnd.github.v3+json",
            }) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    await self.rate_limiter.report_success(url)
                    return data.get("twitter_username", "") or ""
                else:
                    await self.rate_limiter.report_error(url, resp.status)
        except Exception:
            await self.rate_limiter.report_error(url)

        return ""

    def _extract_handles_from_text(self, text: str) -> Set[str]:
        """Extract X/Twitter handles from text."""
        handles = set()
        skip = {
            "home", "explore", "search", "settings", "i", "intent",
            "hashtag", "login", "signup", "status", "twitter", "x",
            "share", "about", "help", "tos", "privacy", "jobs",
            "compose", "messages", "notifications", "lists",
            "bookmarks", "communities", "param", "return", "type",
            "author", "example", "deprecated", "override", "see",
            "since", "readme", "license", "contributing", "changelog",
            "issues", "pulls", "actions", "projects", "wiki",
            "security", "insights", "discussions", "code",
        }

        # twitter.com/handle or x.com/handle
        for m in re.finditer(r"(?:twitter\.com|x\.com)/(@?[\w]{2,15})", text):
            handle = m.group(1).lstrip("@")
            if handle.lower() not in skip:
                handles.add(handle)

        # [@handle](url) markdown pattern
        for m in re.finditer(r"\[@(\w{2,15})\]", text):
            handle = m.group(1)
            if handle.lower() not in skip:
                handles.add(handle)

        return handles

    def _add_handles(self, handles: Set[str]):
        """Add new handles to the collection."""
        for handle in handles:
            if handle.lower() not in {h.lower() for h in self.all_handles}:
                self.all_handles.add(handle)
                self.accounts.append(XAccount(
                    handle=handle,
                    profile_url=f"https://x.com/{handle}",
                ))
