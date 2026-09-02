"""
Data models for X Pathfinder

Defines the core data structures: accounts, search genes, strategies,
and scraped page results.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class XAccount:
    """Represents a discovered X/Twitter account."""

    handle: str
    display_name: str = ""
    bio: str = ""
    followers: int = 0
    following: int = 0
    tweets: int = 0
    joined: str = ""
    verified: bool = False
    profile_url: str = ""
    discovered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    discovered_by: str = ""  # Strategy ID that found this account
    fitness_score: float = 0.0
    fitness_source: str = "unscored"
    evidence_urls: List[str] = field(default_factory=list)
    niche: str = ""
    recent_tweets: List[str] = field(default_factory=list)
    hashtags_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "handle": self.handle,
            "display_name": self.display_name,
            "bio": self.bio,
            "followers": self.followers,
            "following": self.following,
            "tweets": self.tweets,
            "joined": self.joined,
            "verified": self.verified,
            "profile_url": self.profile_url,
            "discovered_at": self.discovered_at,
            "discovered_by": self.discovered_by,
            "fitness_score": self.fitness_score,
            "fitness_source": self.fitness_source,
            "evidence_urls": self.evidence_urls,
            "niche": self.niche,
            "recent_tweets": self.recent_tweets,
            "hashtags_used": self.hashtags_used,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "XAccount":
        return cls(
            handle=data["handle"],
            display_name=data.get("display_name", ""),
            bio=data.get("bio", ""),
            followers=data.get("followers", 0),
            following=data.get("following", 0),
            tweets=data.get("tweets", 0),
            joined=data.get("joined", ""),
            verified=data.get("verified", False),
            profile_url=data.get("profile_url", ""),
            discovered_at=data.get("discovered_at", datetime.now().isoformat()),
            discovered_by=data.get("discovered_by", ""),
            fitness_score=data.get("fitness_score", 0.0),
            fitness_source=data.get("fitness_source", "unscored"),
            evidence_urls=data.get("evidence_urls", []),
            niche=data.get("niche", ""),
            recent_tweets=data.get("recent_tweets", []),
            hashtags_used=data.get("hashtags_used", []),
        )


@dataclass
class SearchGene:
    """A single gene in a search strategy genome."""

    gene_type: str  # "query_term", "hashtag", "seed_account", "operator", "traversal_mode"
    value: str

    def to_dict(self) -> Dict[str, str]:
        return {"gene_type": self.gene_type, "value": self.value}

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "SearchGene":
        return cls(gene_type=data["gene_type"], value=data["value"])


@dataclass
class SearchStrategy:
    """A search strategy composed of multiple genes, subject to evolution."""

    id: str
    genes: List[SearchGene] = field(default_factory=list)
    fitness: float = 0.0
    accounts_found: int = 0
    generation: int = 0

    def compile_query(self) -> str:
        parts = []
        for gene in self.genes:
            if gene.gene_type == "query_term":
                parts.append(gene.value)
            elif gene.gene_type == "hashtag":
                parts.append(gene.value if gene.value.startswith("#") else f"#{gene.value}")
            elif gene.gene_type == "operator":
                parts.append(gene.value)
        return " ".join(parts)

    def get_seed_accounts(self) -> List[str]:
        return [g.value for g in self.genes if g.gene_type == "seed_account"]

    def get_traversal_mode(self) -> str:
        for gene in self.genes:
            if gene.gene_type == "traversal_mode":
                return gene.value
        return "search"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "genes": [g.to_dict() for g in self.genes],
            "fitness": self.fitness,
            "accounts_found": self.accounts_found,
            "generation": self.generation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchStrategy":
        return cls(
            id=data["id"],
            genes=[SearchGene.from_dict(g) for g in data.get("genes", [])],
            fitness=data.get("fitness", 0.0),
            accounts_found=data.get("accounts_found", 0),
            generation=data.get("generation", 0),
        )


# ── Email Evolution Models ──

@dataclass
class EmailPattern:
    """A single email pattern gene - the core unit of evolution."""

    pattern_type: str  # "format", "domain", "separator", "casing"
    value: str

    def to_dict(self) -> Dict[str, str]:
        return {"pattern_type": self.pattern_type, "value": self.value}

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "EmailPattern":
        return cls(pattern_type=data["pattern_type"], value=data["value"])


@dataclass
class EmailStrategy:
    """An email guessing strategy - a genome of patterns.

    Compiles into an email address when given a name.
    Example genes:
        format="{first}.{last}"  domain="gmail.com"  separator="."
    """

    id: str
    patterns: List[EmailPattern] = field(default_factory=list)
    fitness: float = 0.0
    emails_verified: int = 0
    generation: int = 0

    def get_format(self) -> str:
        for p in self.patterns:
            if p.pattern_type == "format":
                return p.value
        return "{first}.{last}"

    def get_domain(self) -> str:
        for p in self.patterns:
            if p.pattern_type == "domain":
                return p.value
        return "gmail.com"

    def compile_email(self, first: str, last: str, handle: str = "") -> str:
        """Generate an email from name parts using this strategy's patterns."""
        fmt = self.get_format()
        domain = self.get_domain()

        email_local = fmt.format(
            first=first.lower(),
            last=last.lower(),
            f=first[0].lower() if first else "",
            l=last[0].lower() if last else "",
            handle=handle.lower(),
            first_last=f"{first}{last}".lower(),
        )

        return f"{email_local}@{domain}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "patterns": [p.to_dict() for p in self.patterns],
            "fitness": self.fitness,
            "emails_verified": self.emails_verified,
            "generation": self.generation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmailStrategy":
        return cls(
            id=data["id"],
            patterns=[EmailPattern.from_dict(p) for p in data.get("patterns", [])],
            fitness=data.get("fitness", 0.0),
            emails_verified=data.get("emails_verified", 0),
            generation=data.get("generation", 0),
        )


@dataclass
class EmailCandidate:
    """A generated email candidate with verification results."""

    email: str
    handle: str
    confidence: float = 0.0
    mx_valid: bool = False
    smtp_valid: Optional[bool] = None  # None = not checked
    strategy_id: str = ""
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "email": self.email,
            "handle": self.handle,
            "confidence": self.confidence,
            "mx_valid": self.mx_valid,
            "smtp_valid": self.smtp_valid,
            "strategy_id": self.strategy_id,
            "generated_at": self.generated_at,
        }


@dataclass
class ScrapedPage:
    """Result of scraping a single page."""

    url: str
    accounts: List[XAccount] = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = ""  # "nitter", "duckduckgo", "playwright"
