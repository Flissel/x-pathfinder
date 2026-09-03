"""
Backer Score Evaluator for X Pathfinder.

Scores discovered accounts on a 0-100 scale for crowdfunding backer likelihood.
Designed for VibeMind OS campaign — identifies people most likely to back
an open-source AI project for $1.

Five scoring dimensions:
- Open Source Affinity (25 pts): contributes, stars, maintains OSS projects
- AI/Agent Interest (25 pts): tweets about AI agents, LLMs, automation tools
- Builder/Maker Identity (20 pts): ships products, indie hacker, founder
- Engagement Propensity (15 pts): replies, backs projects, shares others' work
- Reach Multiplier (15 pts): can amplify the campaign if they back it

Stacks on TOP of the existing fitness_score — does not replace it.
Use both: fitness_score for niche relevance, backer_score for conversion likelihood.
"""

import math
import re
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from .models import XAccount

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Keyword sets for backer scoring
# ──────────────────────────────────────────────────────────────

OPEN_SOURCE_SIGNALS: Dict[str, List[str]] = {
    "strong": [
        "open source", "open-source", "oss maintainer", "foss",
        "github.com/", "gitlab.com/", "contributor", "maintainer",
        "mit license", "apache license", "gpl", "i maintain",
        "my repo", "star my", "check out my github",
    ],
    "moderate": [
        "github", "gitlab", "open source", "pull request", "pr merged",
        "commit", "fork", "starred", "repository", "npm package",
        "pypi", "crates.io", "cargo", "pip install",
    ],
    "hashtags": [
        "#opensource", "#oss", "#foss", "#github", "#openSourceFriday",
        "#hacktoberfest", "#100daysofcode", "#buildinpublic",
    ],
}

AI_AGENT_SIGNALS: Dict[str, List[str]] = {
    "strong": [
        "ai agent", "multi-agent", "autogen", "crewai", "langchain",
        "llamaindex", "rag pipeline", "vector database", "embeddings",
        "autonomous agent", "agent framework", "mcp server",
        "model context protocol", "swarm", "ai orchestration",
        "openfang", "openclaw", "n8n ai", "ai workflow",
    ],
    "moderate": [
        "llm", "gpt", "claude", "gemini", "chatgpt", "openai",
        "anthropic", "hugging face", "transformers", "fine-tuning",
        "prompt engineering", "ai tool", "ai app", "ai startup",
        "generative ai", "machine learning", "deep learning",
    ],
    "hashtags": [
        "#aiagents", "#llm", "#rag", "#langchain", "#autogen",
        "#generativeai", "#ai", "#machinelearning", "#openai",
        "#buildwithAI", "#agentic",
    ],
}

BUILDER_SIGNALS: Dict[str, List[str]] = {
    "strong": [
        "building", "shipping", "launched", "just shipped",
        "founder", "co-founder", "indie hacker", "solo founder",
        "bootstrapped", "i built", "i made", "my project",
        "side project", "saas", "micro-saas", "maker",
        "indiehackers.com", "producthunt", "show hn",
    ],
    "moderate": [
        "startup", "entrepreneur", "ceo", "cto", "product",
        "mvp", "beta", "launch", "pre-seed", "seed round",
        "yc", "y combinator", "techstars", "500 startups",
        "build in public", "wip", "working on",
    ],
    "hashtags": [
        "#buildinpublic", "#indiehackers", "#maker", "#startup",
        "#shipped", "#saas", "#solofounder", "#bootstrap",
        "#sideproject", "#producthunt", "#showyourwork",
    ],
}

ENGAGEMENT_SIGNALS: Dict[str, List[str]] = {
    "backer_behavior": [
        "backed", "supported", "donated", "sponsor", "github sponsors",
        "buy me a coffee", "ko-fi", "patreon", "kickstarter",
        "crowdfunding", "just backed", "happy to support",
        "take my money", "shut up and take",
    ],
    "sharing_behavior": [
        "check this out", "thread", "must read", "bookmark this",
        "sharing", "retweet", "signal boost", "please share",
        "spread the word", "give this a star",
    ],
    "community": [
        "discord", "community", "meetup", "hackathon",
        "conference", "devcon", "contributor", "volunteer",
    ],
}

REACH_TIERS = [
    (1_000_000, 15.0),  # 1M+ followers = max reach score
    (500_000, 13.0),
    (100_000, 11.0),
    (50_000, 9.0),
    (10_000, 7.0),
    (5_000, 5.0),
    (1_000, 3.0),
    (500, 2.0),
    (100, 1.0),
    (0, 0.0),
]


# ──────────────────────────────────────────────────────────────
# Backer Score Result
# ──────────────────────────────────────────────────────────────

@dataclass
class BackerScoreResult:
    """Detailed breakdown of a backer score evaluation."""

    handle: str
    total: float = 0.0
    open_source: float = 0.0
    ai_interest: float = 0.0
    builder: float = 0.0
    engagement: float = 0.0
    reach: float = 0.0
    tier: str = ""  # "hot", "warm", "cold"
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "handle": self.handle,
            "total": self.total,
            "open_source": self.open_source,
            "ai_interest": self.ai_interest,
            "builder": self.builder,
            "engagement": self.engagement,
            "reach": self.reach,
            "tier": self.tier,
            "reasons": self.reasons,
        }


# ──────────────────────────────────────────────────────────────
# Evaluator
# ──────────────────────────────────────────────────────────────

class BackerScoreEvaluator:
    """Evaluates X accounts for crowdfunding backer potential (0-100).

    Scoring breakdown:
    - Open Source Affinity:   25 points
    - AI/Agent Interest:      25 points
    - Builder/Maker Identity: 20 points
    - Engagement Propensity:  15 points
    - Reach Multiplier:       15 points

    Usage::

        evaluator = BackerScoreEvaluator()
        result = evaluator.evaluate(account)
        print(f"@{result.handle}: {result.total} ({result.tier})")
    """

    def __init__(self, custom_keywords: Optional[List[str]] = None):
        self.custom_keywords = [k.lower() for k in (custom_keywords or [])]

    def evaluate(self, account: XAccount) -> BackerScoreResult:
        """Evaluate backer potential. Returns BackerScoreResult with breakdown."""
        result = BackerScoreResult(handle=account.handle)

        text = self._build_text(account)
        hashtags = " ".join(account.hashtags_used).lower()

        result.open_source = self._score_open_source(text, hashtags, result.reasons)
        result.ai_interest = self._score_ai_interest(text, hashtags, result.reasons)
        result.builder = self._score_builder(text, hashtags, result.reasons)
        result.engagement = self._score_engagement(text, hashtags, account, result.reasons)
        result.reach = self._score_reach(account, result.reasons)

        # Custom keyword bonus (distributed across dimensions)
        if self.custom_keywords:
            bonus = sum(1 for kw in self.custom_keywords if kw in text)
            result.ai_interest = min(25.0, result.ai_interest + bonus * 2)

        result.total = round(
            result.open_source + result.ai_interest + result.builder
            + result.engagement + result.reach,
            1,
        )

        # Assign tier
        if result.total >= 60:
            result.tier = "hot"
        elif result.total >= 35:
            result.tier = "warm"
        else:
            result.tier = "cold"

        logger.debug(
            f"@{account.handle} backer={result.total:.1f} ({result.tier}) "
            f"[oss={result.open_source:.0f} ai={result.ai_interest:.0f} "
            f"bld={result.builder:.0f} eng={result.engagement:.0f} "
            f"reach={result.reach:.0f}]"
        )

        return result

    def _build_text(self, account: XAccount) -> str:
        """Combine all searchable text from account."""
        parts = [
            account.bio,
            account.display_name,
            " ".join(account.recent_tweets),
            " ".join(account.hashtags_used),
        ]
        return " ".join(parts).lower()

    def _count_matches(self, text: str, keywords: List[str]) -> int:
        return sum(1 for kw in keywords if kw.lower() in text)

    def _score_open_source(self, text: str, hashtags: str, reasons: List[str]) -> float:
        """Score open source affinity (0-25 points)."""
        score = 0.0

        strong = self._count_matches(text, OPEN_SOURCE_SIGNALS["strong"])
        if strong:
            score += min(15.0, strong * 5)
            reasons.append(f"OSS strong signals: {strong}")

        moderate = self._count_matches(text, OPEN_SOURCE_SIGNALS["moderate"])
        if moderate:
            score += min(6.0, moderate * 1.5)

        ht = self._count_matches(text + " " + hashtags, OPEN_SOURCE_SIGNALS["hashtags"])
        if ht:
            score += min(4.0, ht * 2)
            reasons.append(f"OSS hashtags: {ht}")

        return min(25.0, score)

    def _score_ai_interest(self, text: str, hashtags: str, reasons: List[str]) -> float:
        """Score AI/Agent interest (0-25 points)."""
        score = 0.0

        strong = self._count_matches(text, AI_AGENT_SIGNALS["strong"])
        if strong:
            score += min(15.0, strong * 4)
            reasons.append(f"AI agent signals: {strong}")

        moderate = self._count_matches(text, AI_AGENT_SIGNALS["moderate"])
        if moderate:
            score += min(6.0, moderate * 1.5)

        ht = self._count_matches(text + " " + hashtags, AI_AGENT_SIGNALS["hashtags"])
        if ht:
            score += min(4.0, ht * 2)

        return min(25.0, score)

    def _score_builder(self, text: str, hashtags: str, reasons: List[str]) -> float:
        """Score builder/maker identity (0-20 points)."""
        score = 0.0

        strong = self._count_matches(text, BUILDER_SIGNALS["strong"])
        if strong:
            score += min(12.0, strong * 4)
            reasons.append(f"Builder signals: {strong}")

        moderate = self._count_matches(text, BUILDER_SIGNALS["moderate"])
        if moderate:
            score += min(5.0, moderate * 1.5)

        ht = self._count_matches(text + " " + hashtags, BUILDER_SIGNALS["hashtags"])
        if ht:
            score += min(3.0, ht * 1.5)

        return min(20.0, score)

    def _score_engagement(
        self, text: str, hashtags: str, account: XAccount, reasons: List[str]
    ) -> float:
        """Score engagement propensity (0-15 points)."""
        score = 0.0

        # Backer behavior (strongest signal)
        backer = self._count_matches(text, ENGAGEMENT_SIGNALS["backer_behavior"])
        if backer:
            score += min(8.0, backer * 4)
            reasons.append(f"Backer behavior: {backer}")

        # Sharing behavior
        sharing = self._count_matches(text, ENGAGEMENT_SIGNALS["sharing_behavior"])
        if sharing:
            score += min(4.0, sharing * 2)

        # Community involvement
        community = self._count_matches(text, ENGAGEMENT_SIGNALS["community"])
        if community:
            score += min(3.0, community * 1.5)

        # Active tweeter (engagement proxy)
        if account.recent_tweets and len(account.recent_tweets) >= 3:
            score += 1.0

        return min(15.0, score)

    def _score_reach(self, account: XAccount, reasons: List[str]) -> float:
        """Score reach/amplification potential (0-15 points)."""
        followers = account.followers

        for threshold, pts in REACH_TIERS:
            if followers >= threshold:
                if pts >= 7.0:
                    reasons.append(f"Reach: {followers:,} followers")
                return pts

        return 0.0

    def batch_evaluate(self, accounts: List[XAccount]) -> List[BackerScoreResult]:
        """Evaluate a batch and sort by backer score descending."""
        results = [self.evaluate(a) for a in accounts]
        results.sort(key=lambda r: r.total, reverse=True)
        return results

    def filter_hot(self, accounts: List[XAccount], min_score: float = 60.0) -> List[BackerScoreResult]:
        """Return only hot leads (score >= min_score)."""
        results = self.batch_evaluate(accounts)
        return [r for r in results if r.total >= min_score]

    def summary(self, results: List[BackerScoreResult]) -> Dict:
        """Generate summary stats from a batch of results."""
        if not results:
            return {"total": 0, "hot": 0, "warm": 0, "cold": 0, "avg_score": 0}
        return {
            "total": len(results),
            "hot": sum(1 for r in results if r.tier == "hot"),
            "warm": sum(1 for r in results if r.tier == "warm"),
            "cold": sum(1 for r in results if r.tier == "cold"),
            "avg_score": round(sum(r.total for r in results) / len(results), 1),
            "top_5": [
                {"handle": r.handle, "score": r.total, "tier": r.tier, "reasons": r.reasons}
                for r in sorted(results, key=lambda x: x.total, reverse=True)[:5]
            ],
        }
