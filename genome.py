"""
Genetic algorithm engine for X Pathfinder.

Evolves search strategies through mutation, crossover, and selection.
Each strategy is a genome of SearchGenes that compile into executable
queries for the scraper.

Inspired by the sakana-pathfinder's evolutionary approach to file discovery.
"""

import random
import os
import logging
from typing import List, Tuple, Dict, Any, Optional

from .models import SearchGene, SearchStrategy

logger = logging.getLogger(__name__)

# Seed data per niche for initial population bootstrap
NICHE_SEEDS: Dict[str, Dict[str, List[str]]] = {
    "ai": {
        "queries": [
            "artificial intelligence researcher",
            "machine learning engineer",
            "deep learning",
            "LLM developer",
            "neural network",
            "AI startup founder",
            "NLP researcher",
            "computer vision",
            "reinforcement learning",
            "generative AI",
        ],
        "hashtags": [
            "#AI", "#MachineLearning", "#DeepLearning", "#LLM",
            "#NLP", "#GenerativeAI", "#GPT", "#OpenAI", "#AGI",
            "#DataScience", "#PyTorch", "#TensorFlow",
        ],
        "seed_accounts": [
            "ylecun", "AndrewYNg", "kaborhugo", "GaryMarcus",
            "hardmaru", "sama", "demaboreal", "fcaboreal",
            "goodaboreal", "abor", "jimaboreal",
        ],
    },
    "crypto": {
        "queries": [
            "bitcoin analyst",
            "ethereum developer",
            "DeFi researcher",
            "blockchain engineer",
            "crypto trader",
            "web3 builder",
            "smart contract auditor",
        ],
        "hashtags": [
            "#Bitcoin", "#Ethereum", "#Crypto", "#DeFi", "#Web3",
            "#Blockchain", "#NFT", "#BTC", "#ETH", "#Solana",
        ],
        "seed_accounts": [
            "VitalikButerin", "saboreal", "caboreal",
            "aaboreal", "coinaboreal",
        ],
    },
    "tech": {
        "queries": [
            "software engineer",
            "full stack developer",
            "startup founder",
            "open source contributor",
            "DevOps engineer",
            "tech lead",
            "indie hacker",
        ],
        "hashtags": [
            "#Tech", "#Programming", "#Coding", "#OpenSource",
            "#Developer", "#Startup", "#DevOps", "#WebDev",
        ],
        "seed_accounts": [
            "levelsio", "taboreal", "paboreal",
        ],
    },
    "security": {
        "queries": [
            "cybersecurity researcher",
            "penetration tester",
            "bug bounty hunter",
            "malware analyst",
            "threat intelligence",
            "red team operator",
            "security engineer",
        ],
        "hashtags": [
            "#CyberSecurity", "#InfoSec", "#BugBounty", "#Hacking",
            "#PenTesting", "#RedTeam", "#ThreatIntel", "#CTF",
        ],
        "seed_accounts": [
            "staboreal", "malaboreal",
        ],
    },
}

# Traversal modes for graph exploration
TRAVERSAL_MODES = ["search", "followers", "following"]

# Operators for query refinement
QUERY_OPERATORS = [
    "OR", "AND", "-bot", "-spam", "filter:verified",
    "lang:en", "lang:de", "lang:es", "lang:fr",
    "min_faves:100", "min_retweets:10",
]


class GeneticEngine:
    """Genetic algorithm engine for evolving search strategies.

    Manages population evolution through tournament selection,
    uniform crossover, and five mutation operators.
    """

    POPULATION_SIZE = 20
    MAX_GENERATIONS = 15
    CRISIS_MAX_GENERATIONS = 40
    ELITE_COUNT = 4
    TOURNAMENT_K = 3
    CROSSOVER_RATE = 0.7
    BASE_MUTATION_RATE = 0.25
    CRISIS_MUTATION_MULTIPLIER = 3
    IMMIGRATION_RATE = 0.15

    def __init__(
        self,
        niche: str = "ai",
        custom_keywords: List[str] = None,
        custom_hashtags: List[str] = None,
    ):
        self.niche = niche.lower()
        self.seeds = NICHE_SEEDS.get(self.niche, NICHE_SEEDS["ai"])
        self._strategy_counter = 0

        # Add custom keywords/hashtags to seed data
        if custom_keywords:
            self.seeds["queries"] = self.seeds["queries"] + custom_keywords
        if custom_hashtags:
            self.seeds["hashtags"] = self.seeds["hashtags"] + custom_hashtags

        # Track discovered accounts for seed mutation
        self.discovered_handles: List[str] = []

    def _next_id(self) -> str:
        self._strategy_counter += 1
        return f"strat_{self._strategy_counter:04d}"

    # ── Population Initialization ──

    def create_initial_population(
        self, known_strategies: List[SearchStrategy] = None
    ) -> List[SearchStrategy]:
        """Create initial population from seeds and known strategies."""
        population = []

        # Include top known strategies from knowledge base
        if known_strategies:
            for strat in known_strategies[: self.ELITE_COUNT]:
                strat.id = self._next_id()
                strat.generation = 0
                population.append(strat)

        # Generate strategies from seed data
        while len(population) < self.POPULATION_SIZE:
            population.append(self._create_seed_strategy())

        return population[: self.POPULATION_SIZE]

    def _create_seed_strategy(self) -> SearchStrategy:
        """Create a strategy from niche seed data."""
        genes = []

        # 1-2 query terms
        num_queries = random.randint(1, 2)
        for _ in range(num_queries):
            query = random.choice(self.seeds["queries"])
            genes.append(SearchGene(gene_type="query_term", value=query))

        # 0-2 hashtags
        num_hashtags = random.randint(0, 2)
        for _ in range(num_hashtags):
            hashtag = random.choice(self.seeds["hashtags"])
            genes.append(SearchGene(gene_type="hashtag", value=hashtag))

        # Maybe a seed account for graph traversal
        if random.random() < 0.3 and self.seeds.get("seed_accounts"):
            account = random.choice(self.seeds["seed_accounts"])
            genes.append(SearchGene(gene_type="seed_account", value=account))

        # Maybe an operator
        if random.random() < 0.2:
            operator = random.choice(QUERY_OPERATORS)
            genes.append(SearchGene(gene_type="operator", value=operator))

        # Traversal mode
        mode = random.choices(
            TRAVERSAL_MODES, weights=[0.6, 0.2, 0.2], k=1
        )[0]
        genes.append(SearchGene(gene_type="traversal_mode", value=mode))

        return SearchStrategy(id=self._next_id(), genes=genes)

    # ── Evolution ──

    def evolve(
        self,
        population: List[SearchStrategy],
        crisis_mode: bool = False,
    ) -> List[SearchStrategy]:
        """Evolve population to next generation.

        Uses elitism + tournament selection + crossover + mutation + immigration.
        """
        # Sort by fitness
        population.sort(key=lambda s: s.fitness, reverse=True)

        new_population = []
        generation = population[0].generation + 1 if population else 1
        mutation_rate = self.BASE_MUTATION_RATE * (
            self.CRISIS_MUTATION_MULTIPLIER if crisis_mode else 1
        )

        # Elitism: preserve top strategies
        elite_count = min(self.ELITE_COUNT, len(population))
        for i in range(elite_count):
            elite = self._clone_strategy(population[i])
            elite.generation = generation
            new_population.append(elite)

        # Fill rest through selection + crossover + mutation
        while len(new_population) < self.POPULATION_SIZE:
            # Immigration: inject random strategy
            if random.random() < (self.IMMIGRATION_RATE * (3 if crisis_mode else 1)):
                immigrant = self._create_seed_strategy()
                immigrant.generation = generation
                new_population.append(immigrant)
                continue

            # Tournament selection
            parent1 = self._tournament_select(population)
            parent2 = self._tournament_select(population)

            # Crossover
            if random.random() < self.CROSSOVER_RATE:
                child = self._crossover(parent1, parent2)
            else:
                child = self._clone_strategy(parent1)

            # Mutation
            child = self._mutate(child, mutation_rate)
            child.generation = generation
            child.fitness = 0.0
            child.accounts_found = 0
            new_population.append(child)

        return new_population[: self.POPULATION_SIZE]

    def _tournament_select(
        self, population: List[SearchStrategy]
    ) -> SearchStrategy:
        """Select a strategy via tournament selection (k=3)."""
        k = min(self.TOURNAMENT_K, len(population))
        contestants = random.sample(population, k)
        return max(contestants, key=lambda s: s.fitness)

    def _crossover(
        self, parent1: SearchStrategy, parent2: SearchStrategy
    ) -> SearchStrategy:
        """Uniform crossover on gene lists."""
        child_genes = []
        max_len = max(len(parent1.genes), len(parent2.genes))

        for i in range(max_len):
            if i < len(parent1.genes) and i < len(parent2.genes):
                # Pick from either parent
                gene = random.choice([parent1.genes[i], parent2.genes[i]])
            elif i < len(parent1.genes):
                gene = parent1.genes[i]
            else:
                gene = parent2.genes[i]

            child_genes.append(SearchGene(
                gene_type=gene.gene_type, value=gene.value
            ))

        return SearchStrategy(id=self._next_id(), genes=child_genes)

    def _clone_strategy(self, strategy: SearchStrategy) -> SearchStrategy:
        """Create a deep copy of a strategy."""
        return SearchStrategy(
            id=self._next_id(),
            genes=[SearchGene(gene_type=g.gene_type, value=g.value) for g in strategy.genes],
            fitness=strategy.fitness,
            accounts_found=strategy.accounts_found,
            generation=strategy.generation,
        )

    # ── Mutation Operators ──

    def _mutate(
        self, strategy: SearchStrategy, mutation_rate: float
    ) -> SearchStrategy:
        """Apply mutation operators to a strategy."""
        # 1. Query mutation
        if random.random() < mutation_rate * 0.8:
            strategy = self._mutate_query(strategy)

        # 2. Hashtag mutation
        if random.random() < mutation_rate:
            strategy = self._mutate_hashtag(strategy)

        # 3. Seed account mutation
        if random.random() < mutation_rate * 0.6:
            strategy = self._mutate_seed_account(strategy)

        # 4. Traversal mode mutation
        if random.random() < mutation_rate * 0.8:
            strategy = self._mutate_traversal(strategy)

        # 5. Structural mutation
        if random.random() < mutation_rate * 0.4:
            strategy = self._mutate_structure(strategy)

        return strategy

    def _mutate_query(self, strategy: SearchStrategy) -> SearchStrategy:
        """Mutate query terms."""
        query_genes = [g for g in strategy.genes if g.gene_type == "query_term"]

        if query_genes:
            # Replace a random query gene
            target = random.choice(query_genes)
            new_query = random.choice(self.seeds["queries"])

            # Sometimes combine/modify queries
            if random.random() < 0.3:
                words = new_query.split()
                if len(words) > 1:
                    new_query = " ".join(random.sample(words, min(2, len(words))))

            target.value = new_query
        else:
            # Add a query gene
            strategy.genes.append(SearchGene(
                gene_type="query_term",
                value=random.choice(self.seeds["queries"]),
            ))

        return strategy

    def _mutate_hashtag(self, strategy: SearchStrategy) -> SearchStrategy:
        """Mutate hashtags."""
        hashtag_genes = [g for g in strategy.genes if g.gene_type == "hashtag"]

        if hashtag_genes and random.random() < 0.5:
            # Replace existing hashtag
            target = random.choice(hashtag_genes)
            target.value = random.choice(self.seeds["hashtags"])
        elif random.random() < 0.5:
            # Add new hashtag
            strategy.genes.append(SearchGene(
                gene_type="hashtag",
                value=random.choice(self.seeds["hashtags"]),
            ))
        elif hashtag_genes:
            # Remove a hashtag
            strategy.genes.remove(random.choice(hashtag_genes))

        return strategy

    def _mutate_seed_account(self, strategy: SearchStrategy) -> SearchStrategy:
        """Mutate seed accounts using discovered accounts."""
        seed_genes = [g for g in strategy.genes if g.gene_type == "seed_account"]

        # Prefer discovered accounts over seed accounts
        account_pool = list(self.seeds.get("seed_accounts", []))
        if self.discovered_handles:
            account_pool.extend(self.discovered_handles[-20:])

        if not account_pool:
            return strategy

        if seed_genes and random.random() < 0.6:
            # Replace existing seed
            target = random.choice(seed_genes)
            target.value = random.choice(account_pool)
        elif random.random() < 0.4:
            # Add new seed account
            strategy.genes.append(SearchGene(
                gene_type="seed_account",
                value=random.choice(account_pool),
            ))

        return strategy

    def _mutate_traversal(self, strategy: SearchStrategy) -> SearchStrategy:
        """Mutate traversal mode."""
        trav_genes = [g for g in strategy.genes if g.gene_type == "traversal_mode"]

        new_mode = random.choice(TRAVERSAL_MODES)

        if trav_genes:
            trav_genes[0].value = new_mode
        else:
            strategy.genes.append(SearchGene(
                gene_type="traversal_mode", value=new_mode
            ))

        return strategy

    def _mutate_structure(self, strategy: SearchStrategy) -> SearchStrategy:
        """Structural mutation: add or remove genes."""
        if len(strategy.genes) > 2 and random.random() < 0.4:
            # Remove a random non-essential gene
            removable = [
                g for g in strategy.genes
                if g.gene_type in ("operator", "hashtag")
            ]
            if removable:
                strategy.genes.remove(random.choice(removable))
        elif random.random() < 0.4:
            # Add an operator gene
            strategy.genes.append(SearchGene(
                gene_type="operator",
                value=random.choice(QUERY_OPERATORS),
            ))

        return strategy

    # ── Feedback ──

    def register_discovered_accounts(self, handles: List[str]):
        """Register newly discovered account handles for seed mutation."""
        self.discovered_handles.extend(handles)
        # Keep last 100
        self.discovered_handles = self.discovered_handles[-100:]
