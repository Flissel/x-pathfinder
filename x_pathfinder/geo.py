"""
Country detection for email domains.

Determines country from:
1. ccTLD (.de, .uk, .fr, .ch, etc.)
2. Known domain -> country mapping
3. MX server GeoIP lookup (fallback)
"""

import socket
import struct
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Country code TLDs
CCTLD_MAP = {
    "ac": "SH", "ad": "AD", "ae": "AE", "af": "AF", "ag": "AG",
    "al": "AL", "am": "AM", "ao": "AO", "ar": "AR", "at": "AT",
    "au": "AU", "az": "AZ", "ba": "BA", "bb": "BB", "bd": "BD",
    "be": "BE", "bf": "BF", "bg": "BG", "bh": "BH", "bi": "BI",
    "bj": "BJ", "bn": "BN", "bo": "BO", "br": "BR", "bs": "BS",
    "bt": "BT", "bw": "BW", "by": "BY", "bz": "BZ", "ca": "CA",
    "cd": "CD", "cf": "CF", "cg": "CG", "ch": "CH", "ci": "CI",
    "cl": "CL", "cm": "CM", "cn": "CN", "co": "CO", "cr": "CR",
    "cu": "CU", "cv": "CV", "cy": "CY", "cz": "CZ", "de": "DE",
    "dj": "DJ", "dk": "DK", "dm": "DM", "do": "DO", "dz": "DZ",
    "ec": "EC", "ee": "EE", "eg": "EG", "er": "ER", "es": "ES",
    "et": "ET", "fi": "FI", "fj": "FJ", "fm": "FM", "fr": "FR",
    "ga": "GA", "gb": "GB", "gd": "GD", "ge": "GE", "gh": "GH",
    "gm": "GM", "gn": "GN", "gq": "GQ", "gr": "GR", "gt": "GT",
    "gw": "GW", "gy": "GY", "hk": "HK", "hn": "HN", "hr": "HR",
    "ht": "HT", "hu": "HU", "id": "ID", "ie": "IE", "il": "IL",
    "in": "IN", "iq": "IQ", "ir": "IR", "is": "IS", "it": "IT",
    "jm": "JM", "jo": "JO", "jp": "JP", "ke": "KE", "kg": "KG",
    "kh": "KH", "km": "KM", "kn": "KN", "kp": "KP", "kr": "KR",
    "kw": "KW", "kz": "KZ", "la": "LA", "lb": "LB", "lc": "LC",
    "li": "LI", "lk": "LK", "lr": "LR", "ls": "LS", "lt": "LT",
    "lu": "LU", "lv": "LV", "ly": "LY", "ma": "MA", "mc": "MC",
    "md": "MD", "me": "ME", "mg": "MG", "mk": "MK", "ml": "ML",
    "mm": "MM", "mn": "MN", "mo": "MO", "mr": "MR", "mt": "MT",
    "mu": "MU", "mv": "MV", "mw": "MW", "mx": "MX", "my": "MY",
    "mz": "MZ", "na": "NA", "ne": "NE", "ng": "NG", "ni": "NI",
    "nl": "NL", "no": "NO", "np": "NP", "nz": "NZ", "om": "OM",
    "pa": "PA", "pe": "PE", "pg": "PG", "ph": "PH", "pk": "PK",
    "pl": "PL", "pt": "PT", "py": "PY", "qa": "QA", "ro": "RO",
    "rs": "RS", "ru": "RU", "rw": "RW", "sa": "SA", "sb": "SB",
    "sc": "SC", "sd": "SD", "se": "SE", "sg": "SG", "si": "SI",
    "sk": "SK", "sl": "SL", "sm": "SM", "sn": "SN", "so": "SO",
    "sr": "SR", "ss": "SS", "sv": "SV", "sy": "SY", "sz": "SZ",
    "td": "TD", "tg": "TG", "th": "TH", "tj": "TJ", "tl": "TL",
    "tm": "TM", "tn": "TN", "to": "TO", "tr": "TR", "tt": "TT",
    "tw": "TW", "tz": "TZ", "ua": "UA", "ug": "UG", "uk": "GB",
    "us": "US", "uy": "UY", "uz": "UZ", "vc": "VC", "ve": "VE",
    "vn": "VN", "vu": "VU", "ws": "WS", "ye": "YE", "za": "ZA",
    "zm": "ZM", "zw": "ZW",
    # Education TLDs
    "edu": "US", "ac.uk": "GB", "ac.jp": "JP", "edu.au": "AU",
    "edu.cn": "CN", "ac.in": "IN",
}

# Known domains -> country
DOMAIN_COUNTRY = {
    # US companies
    "gmail.com": "US", "google.com": "US", "googlemail.com": "US",
    "outlook.com": "US", "hotmail.com": "US", "live.com": "US",
    "yahoo.com": "US", "aol.com": "US", "icloud.com": "US",
    "me.com": "US", "mac.com": "US",
    "openai.com": "US", "anthropic.com": "US", "meta.com": "US",
    "ai.meta.com": "US", "facebook.com": "US", "instagram.com": "US",
    "apple.com": "US", "microsoft.com": "US", "amazon.com": "US",
    "nvidia.com": "US", "tesla.com": "US", "stripe.com": "US",
    "github.com": "US", "twitter.com": "US", "x.com": "US",
    "stanford.edu": "US", "mit.edu": "US", "berkeley.edu": "US",
    "cmu.edu": "US", "harvard.edu": "US", "princeton.edu": "US",
    "caltech.edu": "US", "columbia.edu": "US", "yale.edu": "US",
    "cornell.edu": "US", "nyu.edu": "US", "ucla.edu": "US",
    # Germany
    "gmx.de": "DE", "web.de": "DE", "t-online.de": "DE",
    "posteo.de": "DE", "mailbox.org": "DE",
    # Switzerland
    "ethz.ch": "CH", "epfl.ch": "CH", "protonmail.com": "CH",
    "proton.me": "CH", "protonmail.ch": "CH",
    # UK
    "ox.ac.uk": "GB", "cam.ac.uk": "GB", "imperial.ac.uk": "GB",
    "ucl.ac.uk": "GB", "ed.ac.uk": "GB",
    # Canada
    "mila.quebec": "CA", "utoronto.ca": "CA", "ubc.ca": "CA",
    # France
    "inria.fr": "FR", "cnrs.fr": "FR",
    # Russia
    "yandex.com": "RU", "yandex.ru": "RU", "mail.ru": "RU",
    # Japan
    "u-tokyo.ac.jp": "JP", "riken.jp": "JP",
    # China
    "tsinghua.edu.cn": "CN", "pku.edu.cn": "CN", "qq.com": "CN",
    "163.com": "CN", "126.com": "CN",
    # India
    "iisc.ac.in": "IN", "iitb.ac.in": "IN",
    # Australia
    "unimelb.edu.au": "AU", "anu.edu.au": "AU",
    # International / generic
    "fastmail.com": "AU", "zoho.com": "IN", "mail.com": "US",
    "tutanota.com": "DE", "gmx.com": "DE",
}

# Country code -> name
COUNTRY_NAMES = {
    "US": "USA", "GB": "UK", "DE": "Deutschland", "FR": "Frankreich",
    "CH": "Schweiz", "AT": "Oesterreich", "NL": "Niederlande",
    "BE": "Belgien", "IT": "Italien", "ES": "Spanien", "PT": "Portugal",
    "SE": "Schweden", "NO": "Norwegen", "DK": "Daenemark", "FI": "Finnland",
    "PL": "Polen", "CZ": "Tschechien", "RO": "Rumaenien", "HU": "Ungarn",
    "RU": "Russland", "UA": "Ukraine", "TR": "Tuerkei", "IL": "Israel",
    "JP": "Japan", "KR": "Suedkorea", "CN": "China", "TW": "Taiwan",
    "HK": "Hongkong", "SG": "Singapur", "IN": "Indien", "AU": "Australien",
    "NZ": "Neuseeland", "CA": "Kanada", "BR": "Brasilien", "MX": "Mexiko",
    "AR": "Argentinien", "ZA": "Suedafrika", "NG": "Nigeria",
    "KE": "Kenia", "EG": "Aegypten", "AE": "VAE", "SA": "Saudi-Arabien",
    "IE": "Irland", "IS": "Island", "EE": "Estland", "LV": "Lettland",
    "LT": "Litauen", "GR": "Griechenland", "BG": "Bulgarien",
    "HR": "Kroatien", "RS": "Serbien", "SI": "Slowenien",
    "SK": "Slowakei",
}


def detect_country(domain: str) -> str:
    """Detect country code from email domain.

    Returns ISO 2-letter country code or 'XX' if unknown.
    """
    domain = domain.lower().strip()

    # 1. Check known domains
    if domain in DOMAIN_COUNTRY:
        return DOMAIN_COUNTRY[domain]

    # 2. Check compound TLDs first (ac.uk, edu.au, etc.)
    parts = domain.split(".")
    if len(parts) >= 3:
        compound = ".".join(parts[-2:])
        if compound in CCTLD_MAP:
            return CCTLD_MAP[compound]

    # 3. Check simple TLD
    tld = parts[-1]
    if tld in CCTLD_MAP:
        return CCTLD_MAP[tld]

    # 4. Unknown
    return "XX"


def country_name(code: str) -> str:
    """Get country name from code."""
    return COUNTRY_NAMES.get(code, code)
