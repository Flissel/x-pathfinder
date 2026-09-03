"""
Name generator for email pattern evolution.

Provides first/last name pools by country for direct email generation.
No scraping needed - just names + domains + SMTP verification.
"""

from typing import List, Dict, Tuple
import random

# Top first names by country/region
FIRST_NAMES: Dict[str, List[str]] = {
    "US": [
        "james", "john", "robert", "michael", "david", "william", "richard",
        "joseph", "thomas", "christopher", "charles", "daniel", "matthew",
        "anthony", "mark", "donald", "steven", "paul", "andrew", "joshua",
        "kenneth", "kevin", "brian", "george", "timothy", "ronald", "jason",
        "edward", "jeffrey", "ryan", "jacob", "gary", "nicholas", "eric",
        "jonathan", "stephen", "larry", "justin", "scott", "brandon",
        "benjamin", "samuel", "raymond", "gregory", "frank", "alexander",
        "patrick", "jack", "dennis", "jerry", "tyler", "aaron", "jose",
        "adam", "nathan", "henry", "peter", "zachary", "douglas", "harold",
        "mary", "patricia", "jennifer", "linda", "barbara", "elizabeth",
        "susan", "jessica", "sarah", "karen", "lisa", "nancy", "betty",
        "margaret", "sandra", "ashley", "dorothy", "kimberly", "emily",
        "donna", "michelle", "carol", "amanda", "melissa", "deborah",
        "stephanie", "rebecca", "sharon", "laura", "cynthia", "kathleen",
        "amy", "angela", "shirley", "anna", "brenda", "pamela", "emma",
        "nicole", "helen", "samantha", "katherine", "christine", "debra",
    ],
    "DE": [
        "peter", "michael", "thomas", "andreas", "stefan", "christian",
        "markus", "martin", "daniel", "frank", "matthias", "alexander",
        "jan", "tobias", "florian", "felix", "lukas", "jonas", "leon",
        "maximilian", "paul", "tim", "niklas", "julian", "sebastian",
        "philipp", "moritz", "david", "dominik", "simon", "benjamin",
        "patrick", "marcel", "dennis", "sascha", "dirk", "jens", "ralf",
        "oliver", "karsten", "joerg", "bernd", "klaus", "dieter", "hans",
        "wolfgang", "manfred", "helmut", "gerhard", "werner", "karl",
        "anna", "maria", "sarah", "laura", "julia", "lisa", "katharina",
        "sophie", "lena", "marie", "jana", "johanna", "christina",
        "sandra", "nicole", "sabrina", "nadine", "melanie", "stefanie",
        "claudia", "petra", "monika", "andrea", "birgit", "kerstin",
    ],
    "GB": [
        "james", "john", "robert", "david", "william", "richard", "thomas",
        "christopher", "daniel", "matthew", "andrew", "mark", "paul",
        "michael", "peter", "adam", "simon", "luke", "jack", "oliver",
        "harry", "charlie", "george", "edward", "samuel", "joseph",
        "benjamin", "alexander", "henry", "oscar", "jacob", "joshua",
        "max", "lewis", "ryan", "nathan", "connor", "liam", "callum",
        "emma", "sarah", "laura", "emily", "jessica", "sophie", "hannah",
        "charlotte", "rebecca", "lucy", "olivia", "katie", "rachel",
        "amy", "victoria", "megan", "holly", "abigail", "grace", "chloe",
    ],
    "FR": [
        "jean", "pierre", "michel", "jacques", "philippe", "alain",
        "patrick", "nicolas", "christophe", "laurent", "pascal", "eric",
        "francois", "frederic", "olivier", "david", "thomas", "antoine",
        "julien", "sebastien", "alexandre", "maxime", "romain", "hugo",
        "lucas", "clement", "arthur", "louis", "gabriel", "raphael",
        "marie", "nathalie", "isabelle", "sylvie", "catherine", "sophie",
        "valerie", "christine", "sandrine", "camille", "julie", "emma",
        "lea", "chloe", "manon", "clara", "charlotte", "alice", "louise",
    ],
    "CH": [
        "peter", "thomas", "daniel", "martin", "michael", "andreas",
        "stefan", "christian", "markus", "patrick", "marcel", "beat",
        "reto", "urs", "marc", "lukas", "simon", "david", "fabian",
        "raphael", "jonas", "luca", "noah", "leon", "benjamin",
        "anna", "sandra", "maria", "sarah", "laura", "nicole", "andrea",
        "claudia", "monika", "barbara", "ruth", "elisabeth", "eva",
    ],
    "RU": [
        "alexander", "dmitry", "maxim", "sergey", "andrey", "alexey",
        "artem", "ilya", "kirill", "mikhail", "nikita", "ivan",
        "denis", "evgeny", "roman", "vladimir", "igor", "oleg",
        "pavel", "anton", "nikolay", "vitaly", "stanislav", "timur",
        "anna", "maria", "elena", "olga", "tatiana", "natalia",
        "ekaterina", "irina", "svetlana", "anastasia", "yulia", "daria",
    ],
    "IN": [
        "rahul", "amit", "priya", "sanjay", "vikram", "rajesh", "suresh",
        "anil", "deepak", "arjun", "rohan", "vivek", "kiran", "anand",
        "nikhil", "akash", "gaurav", "harsh", "ravi", "sachin",
        "pooja", "neha", "shreya", "anjali", "divya", "swati", "meera",
    ],
    "JP": [
        "takeshi", "hiroshi", "kenji", "yuki", "taro", "akira", "satoshi",
        "takashi", "masashi", "daisuke", "ryota", "shota", "yuto",
        "haruto", "sota", "riku", "kaito", "hinata", "ren", "minato",
    ],
}

# Top last names by country
LAST_NAMES: Dict[str, List[str]] = {
    "US": [
        "smith", "johnson", "williams", "brown", "jones", "garcia",
        "miller", "davis", "rodriguez", "martinez", "hernandez", "lopez",
        "gonzalez", "wilson", "anderson", "thomas", "taylor", "moore",
        "jackson", "martin", "lee", "perez", "thompson", "white",
        "harris", "sanchez", "clark", "ramirez", "lewis", "robinson",
        "walker", "young", "allen", "king", "wright", "scott", "torres",
        "nguyen", "hill", "flores", "green", "adams", "nelson", "baker",
        "hall", "rivera", "campbell", "mitchell", "carter", "roberts",
        "gomez", "phillips", "evans", "turner", "diaz", "parker",
        "cruz", "edwards", "collins", "reyes", "stewart", "morris",
        "morales", "murphy", "cook", "rogers", "gutierrez", "ortiz",
        "morgan", "cooper", "peterson", "bailey", "reed", "kelly",
        "howard", "ramos", "kim", "cox", "ward", "richardson",
    ],
    "DE": [
        "mueller", "schmidt", "schneider", "fischer", "weber", "meyer",
        "wagner", "becker", "schulz", "hoffmann", "schafer", "koch",
        "bauer", "richter", "klein", "wolf", "schroeder", "neumann",
        "schwarz", "zimmermann", "braun", "krueger", "hofmann", "hartmann",
        "lange", "schmitt", "werner", "schmitz", "krause", "meier",
        "lehmann", "schmid", "schulze", "maier", "koehler", "herrmann",
        "koenig", "walter", "mayer", "huber", "kaiser", "fuchs",
        "peters", "lang", "scholz", "moeller", "weis", "jung", "hahn",
    ],
    "GB": [
        "smith", "jones", "taylor", "brown", "williams", "wilson",
        "johnson", "davies", "robinson", "wright", "thompson", "evans",
        "walker", "white", "roberts", "green", "hall", "thomas", "clarke",
        "jackson", "wood", "harris", "edwards", "turner", "martin",
        "cooper", "hill", "ward", "hughes", "moore", "clark", "king",
        "harrison", "lewis", "baker", "allen", "young", "bennett",
    ],
    "FR": [
        "martin", "bernard", "thomas", "petit", "robert", "richard",
        "durand", "dubois", "moreau", "laurent", "simon", "michel",
        "lefevre", "leroy", "roux", "david", "bertrand", "morel",
        "fournier", "girard", "bonnet", "dupont", "lambert", "fontaine",
        "rousseau", "vincent", "muller", "lefevre", "faure", "andre",
    ],
    "CH": [
        "mueller", "meier", "schmid", "keller", "weber", "huber",
        "schneider", "meyer", "steiner", "fischer", "gerber", "brunner",
        "baumann", "frei", "zimmermann", "moser", "widmer", "wyss",
        "graf", "roth", "suter", "peter", "kunz", "leutwyler",
    ],
    "RU": [
        "ivanov", "smirnov", "kuznetsov", "popov", "vasiliev", "petrov",
        "sokolov", "mikhailov", "novikov", "fedorov", "morozov", "volkov",
        "alekseev", "lebedev", "semenov", "egorov", "pavlov", "kozlov",
        "stepanov", "nikolaev", "orlov", "andreev", "makarov", "nikitin",
    ],
    "IN": [
        "sharma", "kumar", "singh", "gupta", "patel", "shah", "joshi",
        "mehta", "verma", "rao", "reddy", "nair", "iyer", "pillai",
        "das", "mishra", "pandey", "agarwal", "jain", "bhat",
    ],
    "JP": [
        "sato", "suzuki", "takahashi", "tanaka", "watanabe", "ito",
        "yamamoto", "nakamura", "kobayashi", "kato", "yoshida", "yamada",
        "sasaki", "yamaguchi", "matsumoto", "inoue", "kimura", "shimizu",
    ],
}

# All 19 SMTP-verifiable domains, mapped to countries where they're popular
COUNTRY_DOMAINS: Dict[str, List[str]] = {
    "US": ["gmail.com", "icloud.com", "zoho.com", "startmail.com", "runbox.com"],
    "DE": ["gmail.com", "freenet.de", "arcor.de", "vodafone.de", "mailbox.org",
           "tutanota.com", "tutamail.com", "kabelmail.de", "unitybox.de", "kabelbw.de"],
    "GB": ["gmail.com", "icloud.com", "zoho.com", "runbox.com", "startmail.com"],
    "FR": ["gmail.com", "icloud.com"],
    "CH": ["gmail.com", "mailbox.org", "tutanota.com", "startmail.com"],
    "RU": ["gmail.com", "yandex.com", "yandex.ru"],
    "IN": ["gmail.com", "zoho.com"],
    "JP": ["gmail.com", "icloud.com"],
}


class NameGenerator:
    """Generates name combinations for email pattern evolution."""

    def __init__(self, countries: List[str] = None):
        self.countries = [c.upper() for c in (countries or ["US"])]

    def random_name(self, country: str = None) -> Tuple[str, str, str]:
        """Generate a random (first, last, country) tuple."""
        c = country or random.choice(self.countries)
        c = c.upper()

        firsts = FIRST_NAMES.get(c, FIRST_NAMES["US"])
        lasts = LAST_NAMES.get(c, LAST_NAMES["US"])

        return random.choice(firsts), random.choice(lasts), c

    def random_domain(self, country: str = None) -> str:
        """Get a random domain for a country."""
        c = (country or random.choice(self.countries)).upper()
        domains = COUNTRY_DOMAINS.get(c, COUNTRY_DOMAINS["US"])
        return random.choice(domains)

    def generate_batch(self, size: int = 100) -> List[Tuple[str, str, str]]:
        """Generate a batch of (first, last, country) tuples."""
        return [self.random_name() for _ in range(size)]

    def get_all_countries(self) -> List[str]:
        """Get all available countries."""
        return list(FIRST_NAMES.keys())

    def get_name_count(self, country: str) -> int:
        """Get total possible name combinations for a country."""
        c = country.upper()
        firsts = len(FIRST_NAMES.get(c, []))
        lasts = len(LAST_NAMES.get(c, []))
        return firsts * lasts
