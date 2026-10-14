"""Where a WildChat user's clock stands: the time zone of a country and, where it spans several,
of a state, as GeoIP2 names them.

Countries with one zone are read off tzdata's zone.tab. A country with several zones but
one holding nearly all its people takes that one (PRINCIPAL). The large countries split
across zones go by state (BY_STATE): the state's zone, or None when it is not listed, and
the answer is then left out rather than placed at a guessed hour.
"""

from __future__ import annotations

import collections
import importlib.resources
import unicodedata

ALIASES = {
    "United Kingdom": "GB",
    "South Korea": "KR",
    "DR Congo": "CD",
    "Türkiye": "TR",
    "The Netherlands": "NL",
    "Macao": "MO",
    "Czechia": "CZ",
    "Bosnia and Herzegovina": "BA",
    "Kosovo": "XK",
    "Myanmar": "MM",
    "Trinidad and Tobago": "TT",
    "Ivory Coast": "CI",
    "Eswatini": "SZ",
    "Congo Republic": "CG",
    "Antigua and Barbuda": "AG",
    "Timor-Leste": "TL",
    "Cabo Verde": "CV",
    "St Vincent and Grenadines": "VC",
    "British Virgin Islands": "VG",
    "Saint Lucia": "LC",
    "Wallis and Futuna": "WF",
}
"""GeoIP2 country names that differ from tzdata's iso3166.tab, to ISO 3166 codes."""

EXTRA_ZONES = {"XK": "Europe/Belgrade"}
"""Countries with no line of their own in zone.tab."""

PRINCIPAL = {
    "CN": "Asia/Shanghai",
    "DE": "Europe/Berlin",
    "ES": "Europe/Madrid",
    "UA": "Europe/Kyiv",
    "ID": "Asia/Jakarta",
    "AR": "America/Argentina/Buenos_Aires",
    "NZ": "Pacific/Auckland",
    "MY": "Asia/Kuala_Lumpur",
    "PT": "Europe/Lisbon",
    "KZ": "Asia/Almaty",
    "UZ": "Asia/Tashkent",
    "CL": "America/Santiago",
    "EC": "America/Guayaquil",
    "PS": "Asia/Hebron",
    "CY": "Asia/Nicosia",
    "PG": "Pacific/Port_Moresby",
    "MN": "Asia/Ulaanbaatar",
    "CD": "Africa/Kinshasa",
}
"""Countries spanning several zones where nearly all the people live in one."""

_US = {
    "America/New_York": [
        "Connecticut",
        "Delaware",
        "District of Columbia",
        "Washington, D.C.",
        "Florida",
        "Georgia",
        "Indiana",
        "Kentucky",
        "Maine",
        "Maryland",
        "Massachusetts",
        "Michigan",
        "New Hampshire",
        "New Jersey",
        "New York",
        "North Carolina",
        "Ohio",
        "Pennsylvania",
        "Rhode Island",
        "South Carolina",
        "Vermont",
        "Virginia",
        "West Virginia",
    ],
    "America/Chicago": [
        "Alabama",
        "Arkansas",
        "Illinois",
        "Iowa",
        "Kansas",
        "Louisiana",
        "Minnesota",
        "Mississippi",
        "Missouri",
        "Nebraska",
        "North Dakota",
        "Oklahoma",
        "South Dakota",
        "Tennessee",
        "Texas",
        "Wisconsin",
    ],
    "America/Denver": ["Colorado", "Idaho", "Montana", "New Mexico", "Utah", "Wyoming"],
    "America/Phoenix": ["Arizona"],
    "America/Los_Angeles": ["California", "Nevada", "Oregon", "Washington"],
    "America/Anchorage": ["Alaska"],
    "Pacific/Honolulu": ["Hawaii"],
    "America/Puerto_Rico": ["Puerto Rico"],
}

_RUSSIA = {
    "Europe/Kaliningrad": ["Kaliningrad"],
    "Europe/Moscow": [
        "Moscow",
        "St.-Petersburg",
        "Saint Petersburg",
        "Leningrad",
        "Krasnodar",
        "Tatarstan",
        "Rostov",
        "Nizhny Novgorod",
        "Voronezh",
        "Kirov",
        "Chuvash",
        "Mordovia",
        "Mordoviya",
        "Mari El",
        "Mariy-El",
        "Penza",
        "Tula",
        "Ryazan",
        "Tver",
        "Yaroslavl",
        "Vladimir",
        "Kaluga",
        "Bryansk",
        "Smolensk",
        "Belgorod",
        "Kursk",
        "Lipetsk",
        "Tambov",
        "Orel",
        "Oryol",
        "Ivanovo",
        "Kostroma",
        "Vologda",
        "Arkhangelsk",
        "Murmansk",
        "Karelia",
        "Komi",
        "Novgorod",
        "Pskov",
        "Stavropol",
        "Dagestan",
        "Chechnya",
        "Ingushetia",
        "Kabardino",
        "Karachay",
        "North Ossetia",
        "Adygea",
        "Adygeya",
        "Ingushetiya",
        "Kalmykia",
        "Volgograd",
        "Crimea",
        "Sevastopol",
        "Nenets",
    ],
    "Europe/Samara": ["Samara", "Udmurt", "Ulyanovsk", "Saratov", "Astrakhan"],
    "Asia/Yekaterinburg": [
        "Bashkortostan",
        "Sverdlovsk",
        "Chelyabinsk",
        "Perm",
        "Orenburg",
        "Tyumen",
        "Kurgan",
        "Khanty",
        "Yamal",
    ],
    "Asia/Omsk": ["Omsk"],
    "Asia/Novosibirsk": [
        "Novosibirsk",
        "Altai",
        "Tomsk",
        "Kemerovo",
        "Kuzbass",
        "Krasnoyarsk",
        "Tuva",
        "Tyva",
        "Khakassia",
        "Khakasiya",
    ],
    "Asia/Irkutsk": ["Irkutsk", "Buryatia", "Buryatiya"],
    "Asia/Yakutsk": ["Sakha", "Yakutia", "Zabaykalsky", "Transbaikal", "Amur"],
    "Asia/Vladivostok": ["Primorsky", "Primorye", "Khabarovsk", "Jewish"],
    "Asia/Magadan": ["Magadan"],
    "Asia/Sakhalin": ["Sakhalin"],
    "Asia/Kamchatka": ["Kamchatka", "Chukotka"],
}

_CANADA = {
    "America/Toronto": ["Ontario", "Quebec"],
    "America/Vancouver": ["British Columbia"],
    "America/Edmonton": ["Alberta"],
    "America/Regina": ["Saskatchewan"],
    "America/Winnipeg": ["Manitoba"],
    "America/Halifax": ["Nova Scotia", "New Brunswick", "Prince Edward Island"],
    "America/St_Johns": ["Newfoundland", "Newfoundland and Labrador"],
    "America/Whitehorse": ["Yukon"],
    "America/Yellowknife": ["Northwest Territories"],
    "America/Iqaluit": ["Nunavut"],
}

_AUSTRALIA = {
    "Australia/Sydney": ["New South Wales", "Australian Capital Territory", "Victoria", "Tasmania"],
    "Australia/Brisbane": ["Queensland"],
    "Australia/Adelaide": ["South Australia"],
    "Australia/Darwin": ["Northern Territory"],
    "Australia/Perth": ["Western Australia"],
}

_MEXICO = {
    "America/Tijuana": ["Baja California"],
    "America/Mazatlan": ["Baja California Sur", "Sinaloa", "Nayarit"],
    "America/Hermosillo": ["Sonora"],
    "America/Cancun": ["Quintana Roo"],
    "America/Mexico_City": [
        "Aguascalientes",
        "Campeche",
        "Chiapas",
        "Chihuahua",
        "Coahuila",
        "Colima",
        "Mexico City",
        "Durango",
        "Guanajuato",
        "Guerrero",
        "Hidalgo",
        "Jalisco",
        "México",
        "Michoacán",
        "Morelos",
        "Nuevo León",
        "Oaxaca",
        "Puebla",
        "Querétaro",
        "San Luis Potosí",
        "Tabasco",
        "Tamaulipas",
        "Tlaxcala",
        "Veracruz",
        "Yucatán",
        "Zacatecas",
    ],
}

_BRAZIL = {
    "America/Rio_Branco": ["Acre"],
    "America/Manaus": ["Amazonas"],
    "America/Porto_Velho": ["Rondonia"],
    "America/Boa_Vista": ["Roraima"],
    "America/Campo_Grande": ["Mato Grosso do Sul"],
    "America/Cuiaba": ["Mato Grosso"],
}
"""Brazil's states off Brasília time; every other state is on America/Sao_Paulo."""

BY_STATE = {"US": _US, "RU": _RUSSIA, "CA": _CANADA, "AU": _AUSTRALIA, "MX": _MEXICO}
"""Countries placed by state: zone -> state names (exact, or a word of the name for Russia)."""


def _plain(text: str) -> str:
    """Return text lower-cased without accents, so "Piauí" and "Piaui" match."""
    return "".join(
        char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char)
    ).lower()


def _tab(name: str) -> list[list[str]]:
    """Return the rows of one of tzdata's .tab files."""
    text = importlib.resources.files("tzdata.zoneinfo").joinpath(name).read_text(encoding="utf-8")
    return [line.split("\t") for line in text.splitlines() if line and not line.startswith("#")]


class LocalTime:  # pylint: disable=too-few-public-methods
    """Maps a GeoIP2 (country, state) to an IANA time zone."""

    def __init__(self):
        """Read tzdata's country names and zones."""
        self.codes = {_plain(name): code for code, name in _tab("iso3166.tab")}
        self.codes.update({_plain(name): code for name, code in ALIASES.items()})
        zones = collections.defaultdict(list)
        for row in _tab("zone.tab"):
            zones[row[0]].append(row[2])
        self.single = {code: found[0] for code, found in zones.items() if len(found) == 1}
        self.single.update(EXTRA_ZONES)
        self.single.update(PRINCIPAL)

    def zone(self, country: str | None, state: str | None) -> str | None:
        """Return the zone of a user, or None when it cannot be told.

        Args:
            country: GeoIP2's country name.
            state: GeoIP2's state name, if any.
        """
        country = country if isinstance(country, str) else ""
        state = state if isinstance(state, str) else None  # pandas gives NaN for a missing one
        code = self.codes.get(_plain(country))
        if code == "BR":
            return _state_zone(_BRAZIL, state) or "America/Sao_Paulo"
        if code in BY_STATE:
            return _state_zone(BY_STATE[code], state, partial=code == "RU")
        return self.single.get(code)


def _state_zone(zones: dict, state: str | None, partial: bool = False) -> str | None:
    """Return the zone a state is listed under, or None.

    Args:
        zones: Zone -> state names.
        state: The state, or None.
        partial: Also match a listed name found inside the state's (Russia's "Oblast",
            "Republic" and "Krai" vary).
    """
    if not state:
        return None
    plain = _plain(state)
    for zone, names in zones.items():
        for name in names:
            if plain == _plain(name) or (partial and _plain(name) in plain):
                return zone
    return None
