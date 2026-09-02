"""
Genetic algorithm for email pattern evolution.

Evolves email format patterns and domain selections to find
working email addresses for discovered X accounts.

Like the sakana-pathfinder evolves CLI commands to find files,
this evolves email patterns to find valid addresses.
"""

import random
import logging
from typing import List, Dict, Any

from .models import EmailPattern, EmailStrategy

logger = logging.getLogger(__name__)

# Email format templates - the "commands" we evolve
EMAIL_FORMATS = [
    "{first}.{last}",
    "{first}{last}",
    "{first}",
    "{last}",
    "{f}.{last}",
    "{f}{last}",
    "{first}.{l}",
    "{last}.{first}",
    "{first}_{last}",
    "{first}-{last}",
    "{handle}",
    "{first}{l}",
    "{l}{first}",
    "{f}.{first}",
    "{first}.{last}1",
    "{first}.{last}123",
]

# All 19 domains where SMTP verification works (rejects fake with 550)
# Tested 94 providers - these are the only ones that give reliable responses
DOMAINS_COMMON = [
    # US/International
    "gmail.com",        # US - reliable 550
    "icloud.com",       # US - reliable 550
    "zoho.com",         # IN - reliable 550
    # DE
    "freenet.de",       # DE - reliable 550
    "arcor.de",         # DE - reliable 550
    "vodafone.de",      # DE - reliable 550
    "mailbox.org",      # DE - reliable 550
    "tutanota.com",     # DE - reliable 550
    "tutamail.com",     # DE - reliable 550
    "kabelmail.de",     # DE - reliable 550
    "unitybox.de",      # DE - reliable 550
    "kabelbw.de",       # DE - reliable 550
    # RU
    "yandex.com",       # RU - reliable 550
    "yandex.ru",        # RU - reliable 550
    # PL
    "interia.pl",       # PL - reliable 550
    # Privacy
    "runbox.com",       # NO - reliable 550
    "startmail.com",    # NL - reliable 550
    "cock.li",          # ?? - reliable 550
    "airmail.cc",       # ?? - reliable 550
]

# Professional/niche domains - only those verified to give reliable SMTP responses
# Catch-all domains (google.com, ethz.ch, openai.com, etc.) are excluded
DOMAINS_BY_NICHE: Dict[str, List[str]] = {
    "ai": [
        "stanford.edu", "mit.edu", "berkeley.edu",
    ],
    "crypto": [],
    "tech": [],
    "security": [],
}


class EmailGeneticEngine:
    """Evolves email patterns through genetic algorithms.

    Population = email strategies (format + domain combinations)
    Fitness = MX verification success, SMTP acceptance
    Mutation = swap formats, domains, add variations
    """

    POPULATION_SIZE = 12
    MAX_GENERATIONS = 20
    ELITE_COUNT = 5
    TOURNAMENT_K = 3
    CROSSOVER_RATE = 0.6
    BASE_MUTATION_RATE = 0.3
    CRISIS_MUTATION_MULTIPLIER = 3

    def __init__(self, niche: str = "ai"):
        self.niche = niche.lower()
        self.niche_domains = DOMAINS_BY_NICHE.get(self.niche, [])
        self._counter = 0

        # Track domains extracted from actual profiles
        self.discovered_domains: List[str] = []

    def _next_id(self) -> str:
        self._counter += 1
        return f"email_{self._counter:04d}"

    # ── Population ──

    def create_initial_population(
        self, known_strategies: List[EmailStrategy] = None
    ) -> List[EmailStrategy]:
        """Create initial population of email strategies."""
        population = []

        # Include known successful strategies
        if known_strategies:
            for s in known_strategies[:self.ELITE_COUNT]:
                s.id = self._next_id()
                s.generation = 0
                population.append(s)

        # Generate diverse initial strategies
        while len(population) < self.POPULATION_SIZE:
            population.append(self._create_random_strategy())

        return population[:self.POPULATION_SIZE]

    def _create_random_strategy(self) -> EmailStrategy:
        """Create a random email strategy."""
        fmt = random.choice(EMAIL_FORMATS)

        # Weight domain selection: 50% common, 30% niche, 20% discovered
        r = random.random()
        if r < 0.5 or not self.niche_domains:
            domain = random.choice(DOMAINS_COMMON)
        elif r < 0.8 or not self.discovered_domains:
            domain = random.choice(self.niche_domains)
        else:
            domain = random.choice(self.discovered_domains)

        return EmailStrategy(
            id=self._next_id(),
            patterns=[
                EmailPattern(pattern_type="format", value=fmt),
                EmailPattern(pattern_type="domain", value=domain),
            ],
        )

    # ── Evolution ──

    def evolve(
        self,
        population: List[EmailStrategy],
        crisis_mode: bool = False,
    ) -> List[EmailStrategy]:
        """Evolve to next generation."""
        population.sort(key=lambda s: s.fitness, reverse=True)
        new_pop = []
        generation = population[0].generation + 1 if population else 1
        mutation_rate = self.BASE_MUTATION_RATE * (
            self.CRISIS_MUTATION_MULTIPLIER if crisis_mode else 1
        )

        # Elitism
        for i in range(min(self.ELITE_COUNT, len(population))):
            elite = self._clone(population[i])
            elite.generation = generation
            new_pop.append(elite)

        # Fill with offspring
        while len(new_pop) < self.POPULATION_SIZE:
            # 15% immigration (random new strategy)
            if random.random() < (0.15 * (3 if crisis_mode else 1)):
                immigrant = self._create_random_strategy()
                immigrant.generation = generation
                new_pop.append(immigrant)
                continue

            # Selection + crossover + mutation
            p1 = self._tournament_select(population)
            p2 = self._tournament_select(population)

            if random.random() < self.CROSSOVER_RATE:
                child = self._crossover(p1, p2)
            else:
                child = self._clone(p1)

            child = self._mutate(child, mutation_rate)
            child.generation = generation
            child.fitness = 0.0
            child.emails_verified = 0
            new_pop.append(child)

        return new_pop[:self.POPULATION_SIZE]

    def _tournament_select(self, pop: List[EmailStrategy]) -> EmailStrategy:
        k = min(self.TOURNAMENT_K, len(pop))
        return max(random.sample(pop, k), key=lambda s: s.fitness)

    def _crossover(self, p1: EmailStrategy, p2: EmailStrategy) -> EmailStrategy:
        """Crossover: take format from one parent, domain from other."""
        child_patterns = []

        # Take format from p1 or p2
        fmt1 = p1.get_format()
        fmt2 = p2.get_format()
        child_patterns.append(EmailPattern(
            pattern_type="format",
            value=random.choice([fmt1, fmt2]),
        ))

        # Take domain from the other parent
        dom1 = p1.get_domain()
        dom2 = p2.get_domain()
        child_patterns.append(EmailPattern(
            pattern_type="domain",
            value=random.choice([dom1, dom2]),
        ))

        return EmailStrategy(id=self._next_id(), patterns=child_patterns)

    def _clone(self, strategy: EmailStrategy) -> EmailStrategy:
        return EmailStrategy(
            id=self._next_id(),
            patterns=[EmailPattern(p.pattern_type, p.value) for p in strategy.patterns],
            fitness=strategy.fitness,
            emails_verified=strategy.emails_verified,
            generation=strategy.generation,
        )

    # ── Mutation ──

    def _mutate(self, strategy: EmailStrategy, rate: float) -> EmailStrategy:
        """Apply mutations to strategy."""
        # Mutate format
        if random.random() < rate:
            fmt_gene = next(
                (p for p in strategy.patterns if p.pattern_type == "format"),
                None,
            )
            if fmt_gene:
                # Sometimes tweak existing format, sometimes replace entirely
                if random.random() < 0.5:
                    fmt_gene.value = random.choice(EMAIL_FORMATS)
                else:
                    fmt_gene.value = self._tweak_format(fmt_gene.value)

        # Mutate domain
        if random.random() < rate:
            dom_gene = next(
                (p for p in strategy.patterns if p.pattern_type == "domain"),
                None,
            )
            if dom_gene:
                all_domains = DOMAINS_COMMON + self.niche_domains + self.discovered_domains
                if all_domains:
                    dom_gene.value = random.choice(all_domains)

        return strategy

    def _tweak_format(self, fmt: str) -> str:
        """Make small modifications to a format string."""
        tweaks = [
            # Add/remove separator
            lambda f: f.replace(".", "_") if "." in f else f.replace("_", "."),
            # Add number suffix
            lambda f: f + str(random.randint(1, 99)),
            # Swap first/last
            lambda f: f.replace("{first}", "{LAST}").replace("{last}", "{first}").replace("{LAST}", "{last}"),
            # Use initial
            lambda f: f.replace("{first}", "{f}") if "{first}" in f else f,
            # Remove numbers
            lambda f: "".join(c for c in f if not c.isdigit()),
        ]
        return random.choice(tweaks)(fmt)

    # ── Feedback ──

    def register_domains(self, domains: List[str]):
        """Register domains extracted from real profiles."""
        self.discovered_domains.extend(domains)
        self.discovered_domains = list(set(self.discovered_domains))[-50:]
