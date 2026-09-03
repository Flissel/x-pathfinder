"""
Knowledge Accumulator - Persistent learning across sessions.

Stores discovered accounts, successful strategies, working Nitter instances,
effective keywords, and session history. Adapted from the sakana-pathfinder
KnowledgeAccumulator for X/Twitter account discovery.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from pathlib import Path

from .models import XAccount, SearchStrategy, SearchGene

logger = logging.getLogger(__name__)


class KnowledgeAccumulator:
    """Accumulates discovered knowledge permanently across sessions.

    Persists to ~/.x_pathfinder/knowledge/ as JSON files:
    - accounts.json: All discovered accounts (deduplicated by handle)
    - strategies.json: Top 50 successful strategies
    - nitter_instances.json: Working Nitter instances with latency
    - niche_keywords.json: Effective keywords per niche with success rate
    - sessions.json: Last 100 discovery sessions
    """

    MAX_ACCOUNTS = 5000
    MAX_STRATEGIES = 50
    MAX_SESSIONS = 100

    def __init__(self, knowledge_dir: str = None):
        self.knowledge_dir = Path(
            knowledge_dir or os.path.expanduser("~/.x_pathfinder/knowledge")
        )
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.accounts_file = self.knowledge_dir / "accounts.json"
        self.strategies_file = self.knowledge_dir / "strategies.json"
        self.nitter_file = self.knowledge_dir / "nitter_instances.json"
        self.keywords_file = self.knowledge_dir / "niche_keywords.json"
        self.sessions_file = self.knowledge_dir / "sessions.json"

        # Load existing knowledge
        self.accounts: Dict[str, Dict[str, Any]] = self._load_json(
            self.accounts_file, {}
        )
        self.strategies: List[Dict[str, Any]] = self._load_json(
            self.strategies_file, []
        )
        self.nitter_instances: List[Dict[str, Any]] = self._load_json(
            self.nitter_file, []
        )
        self.niche_keywords: Dict[str, Dict[str, Any]] = self._load_json(
            self.keywords_file, {}
        )
        self.sessions: List[Dict[str, Any]] = self._load_json(
            self.sessions_file, []
        )

    def _load_json(self, path: Path, default: Any) -> Any:
        """Load JSON file or return default."""
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load {path.name}: {e}")
        return default

    def _save_json(self, path: Path, data: Any):
        """Save data to JSON file."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save {path.name}: {e}")

    # ── Accounts ──

    def add_account(self, account: XAccount):
        """Add or update a discovered account."""
        handle = account.handle.lower()

        if handle in self.accounts:
            # Update with new data if better
            existing = self.accounts[handle]
            if account.fitness_score > existing.get("fitness_score", 0):
                self.accounts[handle] = account.to_dict()
                logger.debug(f"Updated account @{handle} (better fitness)")
        else:
            self.accounts[handle] = account.to_dict()
            logger.info(f"NEW ACCOUNT: @{handle} (fitness={account.fitness_score:.1f})")

        # Trim to max size, keeping highest fitness
        if len(self.accounts) > self.MAX_ACCOUNTS:
            sorted_handles = sorted(
                self.accounts.keys(),
                key=lambda h: self.accounts[h].get("fitness_score", 0),
            )
            for h in sorted_handles[: len(self.accounts) - self.MAX_ACCOUNTS]:
                del self.accounts[h]

        self._save_json(self.accounts_file, self.accounts)

    def add_accounts(self, accounts: List[XAccount]):
        """Add multiple accounts."""
        for account in accounts:
            self.add_account(account)

    def get_known_handles(self) -> Set[str]:
        """Get set of all known account handles."""
        return set(self.accounts.keys())

    def get_top_accounts(
        self, niche: str = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get top accounts by fitness, optionally filtered by niche."""
        accounts = list(self.accounts.values())
        if niche:
            accounts = [a for a in accounts if a.get("niche") == niche]
        accounts.sort(key=lambda a: a.get("fitness_score", 0), reverse=True)
        return accounts[:limit]

    def get_account_count(self) -> int:
        return len(self.accounts)

    # ── Strategies ──

    def add_successful_strategy(self, strategy: SearchStrategy, accounts_found: int):
        """Record a successful strategy."""
        strat_dict = strategy.to_dict()
        strat_dict["accounts_found"] = accounts_found
        strat_dict["recorded_at"] = datetime.now().isoformat()

        # Check for duplicates (same gene composition)
        strat_key = self._strategy_key(strategy)
        for existing in self.strategies:
            if self._strategy_key_from_dict(existing) == strat_key:
                existing["success_count"] = existing.get("success_count", 0) + 1
                existing["total_accounts_found"] = (
                    existing.get("total_accounts_found", 0) + accounts_found
                )
                existing["last_used"] = datetime.now().isoformat()
                self._save_json(self.strategies_file, self.strategies)
                return

        strat_dict["success_count"] = 1
        strat_dict["total_accounts_found"] = accounts_found
        self.strategies.append(strat_dict)

        # Keep top strategies
        self.strategies.sort(
            key=lambda s: s.get("total_accounts_found", 0), reverse=True
        )
        self.strategies = self.strategies[: self.MAX_STRATEGIES]
        self._save_json(self.strategies_file, self.strategies)

    def get_known_strategies(self) -> List[SearchStrategy]:
        """Get previously successful strategies as SearchStrategy objects."""
        result = []
        for s in self.strategies:
            try:
                result.append(SearchStrategy.from_dict(s))
            except Exception:
                pass
        return result

    def _strategy_key(self, strategy: SearchStrategy) -> str:
        """Create a hashable key for deduplication."""
        genes = sorted(
            [(g.gene_type, g.value) for g in strategy.genes]
        )
        return str(genes)

    def _strategy_key_from_dict(self, strat_dict: Dict) -> str:
        """Create key from dict form."""
        genes = sorted(
            [(g["gene_type"], g["value"]) for g in strat_dict.get("genes", [])]
        )
        return str(genes)

    # ── Nitter Instances ──

    def save_nitter_instances(self, instances: List[Dict[str, Any]]):
        """Save working Nitter instances."""
        self.nitter_instances = instances
        self._save_json(self.nitter_file, instances)

    def get_nitter_instances(self) -> List[Dict[str, Any]]:
        """Get cached Nitter instances."""
        return self.nitter_instances

    # ── Keyword Tracking ──

    def record_keyword_result(
        self, keyword: str, niche: str, accounts_found: int
    ):
        """Track keyword effectiveness per niche."""
        if niche not in self.niche_keywords:
            self.niche_keywords[niche] = {}

        kw = keyword.lower()
        if kw not in self.niche_keywords[niche]:
            self.niche_keywords[niche][kw] = {
                "uses": 0,
                "total_accounts_found": 0,
                "first_used": datetime.now().isoformat(),
            }

        entry = self.niche_keywords[niche][kw]
        entry["uses"] += 1
        entry["total_accounts_found"] += accounts_found
        entry["last_used"] = datetime.now().isoformat()
        entry["avg_accounts"] = entry["total_accounts_found"] / entry["uses"]

        self._save_json(self.keywords_file, self.niche_keywords)

    def get_best_keywords(self, niche: str, limit: int = 20) -> List[str]:
        """Get most effective keywords for a niche."""
        if niche not in self.niche_keywords:
            return []

        keywords = self.niche_keywords[niche]
        sorted_kw = sorted(
            keywords.items(),
            key=lambda kv: kv[1].get("avg_accounts", 0),
            reverse=True,
        )
        return [kw for kw, _ in sorted_kw[:limit]]

    # ── Sessions ──

    def record_session(self, session_data: Dict[str, Any]):
        """Record a discovery session."""
        session = {
            "timestamp": datetime.now().isoformat(),
            "session_id": os.urandom(8).hex(),
            **session_data,
        }

        self.sessions.append(session)
        self.sessions = self.sessions[-self.MAX_SESSIONS:]
        self._save_json(self.sessions_file, self.sessions)

        logger.info(
            f"Session recorded: {session_data.get('new_accounts', 0)} new accounts, "
            f"{session_data.get('generations', 0)} generations"
        )

    # ── Stats ──

    def get_stats(self) -> Dict[str, Any]:
        """Get accumulated knowledge statistics."""
        total_accounts = len(self.accounts)
        total_strategies = len(self.strategies)
        total_sessions = len(self.sessions)
        total_niches = len(self.niche_keywords)

        # Average new accounts per session
        new_per_session = []
        for s in self.sessions[-10:]:
            new_per_session.append(s.get("new_accounts", 0))

        avg_new = (
            sum(new_per_session) / len(new_per_session)
            if new_per_session else 0
        )

        # Accounts by niche
        niche_counts: Dict[str, int] = {}
        for acc in self.accounts.values():
            n = acc.get("niche", "unknown")
            niche_counts[n] = niche_counts.get(n, 0) + 1

        return {
            "total_accounts": total_accounts,
            "total_strategies": total_strategies,
            "total_sessions": total_sessions,
            "total_niches": total_niches,
            "avg_new_accounts_per_session": round(avg_new, 1),
            "accounts_by_niche": niche_counts,
            "nitter_instances_cached": len(self.nitter_instances),
            "growth_trend": "saturating" if avg_new < 2 else "growing",
        }

    def export_report(self) -> str:
        """Generate a human-readable knowledge report."""
        stats = self.get_stats()

        report = f"""
X Pathfinder Knowledge Report
==============================

Accumulated Knowledge:
- Total Accounts: {stats['total_accounts']}
- Successful Strategies: {stats['total_strategies']}
- Discovery Sessions: {stats['total_sessions']}
- Nitter Instances: {stats['nitter_instances_cached']}

Growth Trend: {stats['growth_trend']}
Avg New Accounts/Session: {stats['avg_new_accounts_per_session']}

Accounts by Niche:
"""
        for niche, count in sorted(
            stats["accounts_by_niche"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            report += f"  {niche}: {count}\n"

        # Top accounts
        top = self.get_top_accounts(limit=10)
        if top:
            report += "\nTop 10 Accounts by Fitness:\n"
            for acc in top:
                report += (
                    f"  @{acc['handle']} - "
                    f"fitness={acc.get('fitness_score', 0):.1f}, "
                    f"followers={acc.get('followers', 0)}\n"
                )

        return report
