"""
Account fitness evaluation for X Pathfinder.

Scores discovered accounts on a 0-100 scale across four dimensions:
- Content Relevance (40 pts): bio/tweet keyword match to target niche
- Influence (25 pts): follower count (log-scaled), follower/following ratio
- Engagement Quality (20 pts): posting frequency, interaction patterns
- Signal Indicators (15 pts): verified, account age, profile completeness
"""

import math
import re
import logging
from typing import Dict, List, Set

from .models import XAccount

logger = logging.getLogger(__name__)

# Niche keyword sets for relevance scoring
NICHE_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "ai": {
        "primary": [
            "artificial intelligence", "machine learning", "deep learning",
            "neural network", "nlp", "computer vision", "llm", "gpt",
            "transformer", "reinforcement learning", "ai researcher",
            "ai engineer", "ml engineer", "data scientist",
        ],
        "secondary": [
            "python", "pytorch", "tensorflow", "hugging face", "openai",
            "anthropic", "google deepmind", "meta ai", "stable diffusion",
            "midjourney", "chatgpt", "claude", "gemini", "llama",
            "fine-tuning", "rag", "embeddings", "vector database",
        ],
        "hashtags": [
            "#ai", "#machinelearning", "#deeplearning", "#llm", "#nlp",
            "#artificialintelligence", "#datascience", "#gpt", "#openai",
            "#generativeai", "#agi",
        ],
    },
    "crypto": {
        "primary": [
            "bitcoin", "ethereum", "blockchain", "defi", "web3",
            "cryptocurrency", "crypto trader", "crypto analyst",
            "smart contract", "nft", "dao", "dex",
        ],
        "secondary": [
            "solana", "cardano", "polygon", "avalanche", "cosmos",
            "binance", "coinbase", "metamask", "uniswap", "aave",
            "yield farming", "staking", "tokenomics", "whitepaper",
        ],
        "hashtags": [
            "#bitcoin", "#ethereum", "#crypto", "#defi", "#web3",
            "#blockchain", "#nft", "#btc", "#eth", "#cryptocurrency",
        ],
    },
    "tech": {
        "primary": [
            "software engineer", "developer", "programmer", "startup",
            "tech lead", "cto", "founder", "open source", "devops",
            "cloud computing", "full stack", "backend", "frontend",
        ],
        "secondary": [
            "javascript", "typescript", "rust", "go", "kubernetes",
            "docker", "aws", "azure", "gcp", "react", "nextjs",
            "microservices", "api", "saas", "yc", "venture capital",
        ],
        "hashtags": [
            "#tech", "#programming", "#coding", "#developer", "#startup",
            "#opensource", "#devops", "#webdev", "#javascript", "#rust",
        ],
    },
    "security": {
        "primary": [
            "cybersecurity", "infosec", "hacker", "penetration testing",
            "bug bounty", "security researcher", "red team", "blue team",
            "malware analyst", "threat intelligence", "soc analyst",
        ],
        "secondary": [
            "cve", "exploit", "vulnerability", "zero day", "osint",
            "reverse engineering", "forensics", "incident response",
            "ctf", "owasp", "burp suite", "metasploit",
        ],
        "hashtags": [
            "#cybersecurity", "#infosec", "#hacking", "#bugbounty",
            "#pentesting", "#redteam", "#blueteam", "#threatintel",
        ],
    },
}


class AccountFitnessEvaluator:
    """Evaluates X accounts on a 0-100 fitness scale.

    Scoring breakdown:
    - Content Relevance: 40 points
    - Influence: 25 points
    - Engagement Quality: 20 points
    - Signal Indicators: 15 points
    """

    def __init__(self, niche: str = "ai", custom_keywords: List[str] = None,
                 provider=None):
        self.niche = niche.lower()
        self.keywords = NICHE_KEYWORDS.get(self.niche, NICHE_KEYWORDS["ai"])
        self.custom_keywords = [k.lower() for k in (custom_keywords or [])]
        self.provider = provider

    def evaluate(self, account: XAccount) -> float:
        """Evaluate account fitness. Returns score 0-100."""
        relevance = self._score_relevance(account)
        influence = self._score_influence(account)
        engagement = self._score_engagement(account)
        signals = self._score_signals(account)

        total = relevance + influence + engagement + signals
        account.fitness_score = round(total, 1)

        logger.debug(
            f"@{account.handle} fitness={total:.1f} "
            f"(rel={relevance:.1f} inf={influence:.1f} "
            f"eng={engagement:.1f} sig={signals:.1f})"
        )
        return total

    def _score_relevance(self, account: XAccount) -> float:
        """Score content relevance (0-40 points)."""
        score = 0.0

        # Combine all text for analysis
        text = f"{account.bio} {account.display_name} {' '.join(account.recent_tweets)}".lower()
        hashtags_text = " ".join(account.hashtags_used).lower()

        # Primary keywords (higher weight)
        primary_matches = sum(
            1 for kw in self.keywords["primary"] if kw in text
        )
        score += min(20, primary_matches * 5)

        # Secondary keywords
        secondary_matches = sum(
            1 for kw in self.keywords["secondary"] if kw in text
        )
        score += min(10, secondary_matches * 2)

        # Hashtag matches
        hashtag_matches = sum(
            1 for ht in self.keywords["hashtags"]
            if ht.lower() in hashtags_text or ht.lower() in text
        )
        score += min(5, hashtag_matches * 1.5)

        # Custom keyword bonus
        if self.custom_keywords:
            custom_matches = sum(1 for kw in self.custom_keywords if kw in text)
            score += min(5, custom_matches * 2.5)

        return min(40.0, score)

    def _score_influence(self, account: XAccount) -> float:
        """Score influence metrics (0-25 points)."""
        score = 0.0

        # Follower count (logarithmic scaling)
        if account.followers > 0:
            # log10(1000) = 3, log10(1M) = 6, log10(10M) = 7
            log_followers = math.log10(max(1, account.followers))
            # Scale: 100 followers = 5pts, 10K = 12pts, 100K = 18pts, 1M = 20pts
            score += min(20.0, log_followers * 4)

        # Follower/Following ratio (higher = more influential)
        if account.following > 0:
            ratio = account.followers / account.following
            # Ratio > 10 is excellent, 1-10 is good, < 1 is low
            if ratio >= 10:
                score += 5.0
            elif ratio >= 2:
                score += 3.0
            elif ratio >= 0.5:
                score += 1.0

        return min(25.0, score)

    def _score_engagement(self, account: XAccount) -> float:
        """Score engagement quality (0-20 points)."""
        score = 0.0

        # Posting activity (based on tweet count as proxy)
        if account.tweets > 0:
            # Active accounts typically have 1000+ tweets
            if account.tweets >= 10000:
                score += 8.0
            elif account.tweets >= 1000:
                score += 6.0
            elif account.tweets >= 100:
                score += 3.0
            elif account.tweets >= 10:
                score += 1.0

        # Recent tweet content quality
        if account.recent_tweets:
            # Has recent tweets = active
            score += 4.0

            # Check for engagement signals in tweets
            engagement_signals = 0
            for tweet in account.recent_tweets:
                if any(c in tweet for c in ["@", "http", "https"]):
                    engagement_signals += 1
                if re.search(r"#\w+", tweet):
                    engagement_signals += 1

            score += min(4.0, engagement_signals)

        # Hashtag diversity
        if account.hashtags_used:
            unique_hashtags = len(set(account.hashtags_used))
            score += min(4.0, unique_hashtags * 0.5)

        return min(20.0, score)

    def _score_signals(self, account: XAccount) -> float:
        """Score signal indicators (0-15 points)."""
        score = 0.0

        # Verified account
        if account.verified:
            score += 4.0

        # Has bio (profile completeness)
        if account.bio and len(account.bio) > 10:
            score += 3.0

        # Has display name
        if account.display_name and account.display_name != account.handle:
            score += 2.0

        # Account age (joined date)
        if account.joined:
            score += 2.0  # Having a joined date means we got profile data

        # Reasonable follower/following counts (not spam-like)
        if account.followers > 0 and account.following > 0:
            ratio = account.followers / account.following
            if 0.01 < ratio < 1000:  # Not a bot pattern
                score += 2.0

        # Has some content
        if account.tweets > 0:
            score += 2.0

        return min(15.0, score)

    def batch_evaluate(self, accounts: List[XAccount]) -> List[XAccount]:
        """Evaluate a batch of accounts and sort by fitness, best first."""
        if self.provider is None:
            for account in accounts:
                self.evaluate(account)
        else:
            results = self.provider.score_batch(accounts)
            for account in accounts:
                result = results.get(account.handle.lower())
                if result is None:
                    account.fitness_score = 0.0
                    account.fitness_source = "unscored"
                    continue
                account.fitness_score = 0.0 if result.score is None else result.score
                account.fitness_source = result.source
                account.evidence_urls = list(result.evidence_urls)
        accounts.sort(key=lambda a: a.fitness_score, reverse=True)
        return accounts
