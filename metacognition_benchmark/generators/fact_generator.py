"""
Generate numerical fact questions from Wikidata, OEIS, and NIST/CODATA.

Uses Wikidata SPARQL to find entities with numeric properties, then estimates
difficulty using Wikipedia pageview counts as a proxy for training data prevalence.

Usage:
    python -m metacognition_benchmark.fact_generator --count 50 --output generated_facts.json
"""

import json
import time
import random
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import requests

from metacognition_benchmark.facts import NumericalFact, Tolerance

USER_AGENT = "MetacognitionBenchmark/1.0 (Rolv-Arild Braaten; rolv_arild@hotmail.com)"


# ---------------------------------------------------------------------------
# Wikidata SPARQL
# ---------------------------------------------------------------------------

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
WIKIPEDIA_PAGEVIEWS_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{title}/monthly/20250101/20250501"

# Queries that return (entity label, numeric value, unit label, property label)
# Each query targets a different kind of numeric fact.

WIKIDATA_QUERIES = {
    "elevation": {
        "sparql": """
SELECT ?itemLabel ?value ?unitLabel WHERE {
  ?item wdt:P31 wd:Q8502 .          # instance of: mountain
  ?item wdt:P2044 ?height .          # elevation above sea level
  ?item p:P2044 ?stmt .
  ?stmt psv:P2044 ?valueNode .
  ?valueNode wikibase:quantityAmount ?value .
  ?valueNode wikibase:quantityUnit ?unit .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY MD5(CONCAT(STR(?item), "salt"))
LIMIT 200
""",
        "question_template": "What is the elevation of {label} above sea level in {unit}?",
        "category": "geography",
        "tolerance": Tolerance.NARROW,
    },
    "population": {
        "sparql": """
SELECT ?itemLabel ?value ?unitLabel WHERE {
  ?item wdt:P31 wd:Q515 .           # instance of: city
  ?item wdt:P1082 ?value .           # population
  BIND("inhabitants" AS ?unitLabel)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY MD5(CONCAT(STR(?item), "salt2"))
LIMIT 200
""",
        "question_template": "What is the population of {label} (as recorded in Wikidata)?",
        "category": "demographics",
        "tolerance": Tolerance.MODERATE,
        "unit_override": "inhabitants",
    },
    "area_country": {
        "sparql": """
SELECT ?itemLabel ?value ?unitLabel WHERE {
  ?item wdt:P31 wd:Q6256 .          # instance of: country
  ?item p:P2046 ?stmt .              # area
  ?stmt psv:P2046 ?valueNode .
  ?valueNode wikibase:quantityAmount ?value .
  ?valueNode wikibase:quantityUnit ?unit .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY MD5(CONCAT(STR(?item), "salt3"))
LIMIT 100
""",
        "question_template": "What is the total area of {label} in {unit}?",
        "category": "geography",
        "tolerance": Tolerance.NARROW,
    },
    "lake_depth": {
        "sparql": """
SELECT ?itemLabel ?value ?unitLabel WHERE {
  ?item wdt:P31 wd:Q23397 .         # instance of: lake
  ?item p:P4511 ?stmt .              # maximum depth
  ?stmt psv:P4511 ?valueNode .
  ?valueNode wikibase:quantityAmount ?value .
  ?valueNode wikibase:quantityUnit ?unit .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY MD5(CONCAT(STR(?item), "salt4"))
LIMIT 100
""",
        "question_template": "What is the maximum depth of {label} in {unit}?",
        "category": "geography",
        "tolerance": Tolerance.NARROW,
    },
    "building_height": {
        "sparql": """
SELECT ?itemLabel ?value ?unitLabel WHERE {
  ?item wdt:P31/wdt:P279* wd:Q18142 .  # instance/subclass of: high-rise building
  ?item p:P2048 ?stmt .                  # height
  ?stmt psv:P2048 ?valueNode .
  ?valueNode wikibase:quantityAmount ?value .
  ?valueNode wikibase:quantityUnit ?unit .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY MD5(CONCAT(STR(?item), "salt5"))
LIMIT 200
""",
        "question_template": "What is the height of {label} in {unit}?",
        "category": "architecture",
        "tolerance": Tolerance.NARROW,
    },
    "river_length": {
        "sparql": """
SELECT ?itemLabel ?value ?unitLabel WHERE {
  ?item wdt:P31 wd:Q4022 .          # instance of: river
  ?item p:P2043 ?stmt .              # length
  ?stmt psv:P2043 ?valueNode .
  ?valueNode wikibase:quantityAmount ?value .
  ?valueNode wikibase:quantityUnit ?unit .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY MD5(CONCAT(STR(?item), "salt6"))
LIMIT 200
""",
        "question_template": "What is the length of the {label} river in {unit}?",
        "category": "geography",
        "tolerance": Tolerance.NARROW,
    },
    "boiling_point": {
        "sparql": """
SELECT ?itemLabel ?value ?unitLabel WHERE {
  ?item wdt:P31 wd:Q11344 .         # instance of: chemical element
  ?item p:P2102 ?stmt .              # boiling point
  ?stmt psv:P2102 ?valueNode .
  ?valueNode wikibase:quantityAmount ?value .
  ?valueNode wikibase:quantityUnit ?unit .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 120
""",
        "question_template": "What is the boiling point of {label} in {unit}?",
        "category": "chemistry",
        "tolerance": Tolerance.NARROW,
    },
    "orbital_period": {
        "sparql": """
SELECT ?itemLabel ?value ?unitLabel WHERE {
  ?item wdt:P31 wd:Q634 .           # instance of: planet (solar system)
  ?item p:P2146 ?stmt .              # orbital period
  ?stmt psv:P2146 ?valueNode .
  ?valueNode wikibase:quantityAmount ?value .
  ?valueNode wikibase:quantityUnit ?unit .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 50
""",
        "question_template": "What is the orbital period of {label} in {unit}?",
        "category": "astronomy",
        "tolerance": Tolerance.NARROW,
    },
}


def query_wikidata(sparql: str, retries: int = 3) -> list[dict]:
    """Execute a SPARQL query against the Wikidata Query Service."""
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": USER_AGENT,
    }
    for attempt in range(retries):
        try:
            resp = requests.get(
                WIKIDATA_SPARQL_URL,
                params={"query": sparql},
                headers=headers,
                timeout=60,
            )
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["results"]["bindings"]
        except Exception as e:
            if attempt == retries - 1:
                print(f"  SPARQL query failed: {e}")
                return []
            time.sleep(2)
    return []


def get_wikipedia_pageviews(title: str, max_retries: int = 3) -> Optional[int]:
    """Get approximate yearly pageviews for a Wikipedia article (difficulty proxy)."""
    url = WIKIPEDIA_PAGEVIEWS_URL.format(title=title.replace(" ", "_"))
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("retry-after", 2 ** (attempt + 1)))
                time.sleep(retry_after)
                continue
            if resp.status_code != 200:
                return None
            data = resp.json()
            total = sum(item["views"] for item in data.get("items", []))
            return total
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
    return None


def pageviews_to_difficulty(pageviews: Optional[int]) -> int:
    """Convert pageview count to difficulty level 1-5."""
    if pageviews is None:
        return 3  # Unknown, assume medium
    if pageviews > 1_000_000:
        return 1
    elif pageviews > 200_000:
        return 2
    elif pageviews > 50_000:
        return 3
    elif pageviews > 10_000:
        return 4
    else:
        return 5


def generate_facts_from_wikidata(
    query_name: str,
    max_facts: int = 20,
    estimate_difficulty: bool = True,
) -> list[NumericalFact]:
    """Generate NumericalFact instances from a Wikidata query."""
    query_config = WIKIDATA_QUERIES[query_name]
    sparql = query_config["sparql"]
    template = query_config["question_template"]
    category = query_config["category"]
    tolerance = query_config["tolerance"]

    print(f"  Querying Wikidata for '{query_name}'...")
    results = query_wikidata(sparql)
    if not results:
        return []

    facts = []
    for row in results:
        try:
            label = row["itemLabel"]["value"]
            # Skip items that didn't resolve to a label (show as Q-id)
            if label.startswith("Q") and label[1:].isdigit():
                continue

            value = float(row["value"]["value"])
            if value <= 0:
                continue

            unit = query_config.get("unit_override") or row.get("unitLabel", {}).get("value", "")
            # Clean up unit URIs
            if unit.startswith("http"):
                unit = unit.split("/")[-1]

            question = template.format(label=label, unit=unit)

            difficulty = 3
            if estimate_difficulty:
                pageviews = get_wikipedia_pageviews(label)
                difficulty = pageviews_to_difficulty(pageviews)
                time.sleep(0.1)  # Be polite to the API

            facts.append(NumericalFact(
                question=question,
                answer=value,
                unit=unit,
                tolerance=tolerance,
                source=f"Wikidata ({query_name})",
                difficulty=difficulty,
                category=category,
            ))

            if len(facts) >= max_facts:
                break

        except (KeyError, ValueError):
            continue

    return facts


# ---------------------------------------------------------------------------
# OEIS (On-Line Encyclopedia of Integer Sequences)
# ---------------------------------------------------------------------------

OEIS_SEQUENCES = [
    # (sequence_id, name, description, indices to query, difficulty)
    ("A000040", "prime numbers", "the {n}th prime number", [100, 500, 1000, 5000], 3),
    ("A000045", "Fibonacci numbers", "the {n}th Fibonacci number", [20, 30, 40, 50], 2),
    ("A000108", "Catalan numbers", "the {n}th Catalan number", [10, 15, 20], 4),
    ("A000041", "partition numbers", "the number of partitions of {n}", [30, 50, 100], 4),
    ("A002808", "composite numbers", "the {n}th composite number", [100, 500, 1000], 4),
    ("A000079", "powers of 2", "2 to the power of {n}", [20, 30, 40, 50], 2),
    ("A005100", "deficient numbers", "the {n}th deficient number", [50, 100, 200], 5),
    ("A000010", "Euler's totient", "Euler's totient function φ({n})", [100, 256, 1000], 4),
]


def fetch_oeis_sequence(seq_id: str, max_terms: int = 200) -> tuple[list[int], int]:
    """Fetch terms of an OEIS sequence. Returns (terms, offset) where offset is the starting index."""
    try:
        url = f"https://oeis.org/search?q=id:{seq_id}&fmt=json"
        resp = requests.get(url, timeout=15, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        data = resp.json()
        if data.get("results"):
            result = data["results"][0]
            # The 'data' field contains comma-separated terms
            terms_str = result.get("data", "")
            terms = [int(x) for x in terms_str.split(",") if x.strip()]
            # The 'offset' field is "a,b" where a is the starting index
            offset_str = result.get("offset", "0,1")
            offset = int(offset_str.split(",")[0])
            return terms[:max_terms], offset
    except Exception as e:
        print(f"  OEIS fetch failed for {seq_id}: {e}")
    return [], 0


def generate_facts_from_oeis(max_facts: int = 15) -> list[NumericalFact]:
    """Generate NumericalFact instances from OEIS sequences."""
    facts = []

    for seq_id, name, desc_template, indices, difficulty in OEIS_SEQUENCES:
        print(f"  Fetching OEIS {seq_id} ({name})...")
        terms, offset = fetch_oeis_sequence(seq_id)
        if not terms:
            continue

        for n in indices:
            # Convert requested index n to list position using the sequence's offset
            list_index = n - offset
            if 0 <= list_index < len(terms):
                value = terms[list_index]
                question = f"What is {desc_template.format(n=n)}?"
                facts.append(NumericalFact(
                    question=question,
                    answer=float(value),
                    unit="(integer)",
                    tolerance=Tolerance.EXACT,
                    source=f"OEIS {seq_id} ({name}), offset={offset}",
                    difficulty=difficulty,
                    category="mathematics",
                ))

            if len(facts) >= max_facts:
                return facts

        time.sleep(1)  # Be polite to OEIS

    return facts


# ---------------------------------------------------------------------------
# NIST/CODATA Physical Constants (parsed from allascii.txt)
# ---------------------------------------------------------------------------

# Source: https://physics.nist.gov/constants (2022 CODATA adjustment)
# The file "allascii.txt" is included in the package and contains the complete listing.

CODATA_FILE = Path(__file__).parent.parent / "data" / "allascii.txt"

# Constants we consider "well-known" get lower difficulty; everything else is higher.
# Maps substrings of constant names to difficulty overrides.
_DIFFICULTY_OVERRIDES = {
    "Avogadro": 1,
    "speed of light": 1,
    "Planck constant": 2,
    "elementary charge": 2,
    "Boltzmann constant": 2,
    "molar gas constant": 2,
    "gravitational constant": 2,
    "Newtonian constant of gravitation": 2,
    "standard acceleration of gravity": 1,
    "standard atmosphere": 1,
    "electron mass": 3,
    "proton mass": 3,
    "neutron mass": 3,
    "fine-structure constant": 3,
    "inverse fine-structure constant": 3,
    "Rydberg constant": 4,
    "Stefan-Boltzmann constant": 3,
    "Faraday constant": 2,
    "Bohr magneton": 4,
    "Bohr radius": 3,
    "nuclear magneton": 5,
    "von Klitzing constant": 5,
    "mag. flux quantum": 5,
    "classical electron radius": 5,
    "Compton wavelength": 4,
    "first radiation constant": 5,
    "Wien wavelength displacement": 4,
    "Thomson cross section": 5,
    "Planck length": 4,
    "Planck mass": 4,
    "Planck temperature": 4,
    "Planck time": 4,
}


def _parse_codata_value(value_str: str) -> Optional[float]:
    """Parse a CODATA value string like '6.626 070 15 e-34' into a float."""
    # Remove spaces within the number (NIST format uses spaces as digit grouping)
    # but keep the space before 'e' notation
    cleaned = value_str.strip()
    # Remove trailing '...' (indicates exact/computed digits)
    cleaned = cleaned.replace("...", "")
    # Remove internal spaces (digit grouping)
    cleaned = cleaned.replace(" ", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _estimate_codata_difficulty(name: str) -> int:
    """Estimate difficulty for a CODATA constant based on how well-known it is."""
    name_lower = name.lower()
    for substr, diff in _DIFFICULTY_OVERRIDES.items():
        if substr.lower() in name_lower:
            return diff
    # Default: obscure
    return 5


def parse_codata_file(path: Path = CODATA_FILE) -> list[tuple[str, float, str, int]]:
    """
    Parse the NIST CODATA allascii.txt file.
    Returns list of (name, value, unit, difficulty).
    """
    constants = []
    lines = path.read_text().splitlines()

    for line in lines:
        # Skip header, separator, and empty lines
        if not line.strip() or line.startswith(" " * 2) and "Quantity" in line:
            continue
        if line.startswith("---") or line.startswith("  From:") or "Fundamental Physical" in line:
            continue
        if "Quantity" in line and "Value" in line and "Unit" in line:
            continue

        # The format is fixed-width:
        # Quantity (cols ~0-60), Value (cols ~61-84), Uncertainty (cols ~85-107), Unit (cols ~108+)
        if len(line) < 70:
            continue

        # Parse using fixed-width columns based on the header
        name = line[:60].strip()
        value_str = line[60:85].strip()
        # uncertainty = line[85:110].strip()  # Not used
        unit = line[110:].strip() if len(line) > 110 else ""

        if not name or not value_str:
            continue

        value = _parse_codata_value(value_str)
        if value is None or value == 0:
            continue

        difficulty = _estimate_codata_difficulty(name)
        constants.append((name, value, unit, difficulty))

    return constants


def generate_facts_from_codata(max_facts: int = 15) -> list[NumericalFact]:
    """Generate NumericalFact instances from the NIST CODATA allascii.txt file."""
    all_constants = parse_codata_file()

    # Shuffle to get variety when max_facts is small
    random.shuffle(all_constants)

    # Sort by difficulty to prefer a mix (take some from each level)
    # Group by difficulty and sample proportionally
    by_difficulty: dict[int, list] = {}
    for entry in all_constants:
        d = entry[3]
        by_difficulty.setdefault(d, []).append(entry)

    facts = []
    # Round-robin across difficulty levels
    per_level = max(1, max_facts // max(1, len(by_difficulty)))
    for diff in sorted(by_difficulty):
        for name, value, unit, difficulty in by_difficulty[diff][:per_level]:
            # Skip dimensionless ratios and relationship constants (less interesting as questions)
            if not unit or "relationship" in name.lower():
                continue
            # Avoid redundant unit in question if the name already specifies units
            # e.g., "Boltzmann constant in eV/K" already implies the unit
            if " in " in name:
                question = f"What is the value of the {name}?"
            else:
                question = f"What is the value of the {name} in {unit}?"
            facts.append(NumericalFact(
                question=question,
                answer=value,
                unit=unit,
                tolerance=Tolerance.NARROW,
                source="NIST CODATA 2022 (allascii.txt)",
                difficulty=difficulty,
                category="physics",
                notes="Parsed from NIST Fundamental Physical Constants — Complete Listing (2022 CODATA)",
            ))
            if len(facts) >= max_facts:
                return facts

    return facts


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


def generate_all_facts(
    wikidata_per_query: int = 10,
    oeis_count: int = 10,
    codata_count: int = 10,
    estimate_difficulty: bool = True,
) -> list[NumericalFact]:
    """Generate facts from all sources."""
    all_facts = []

    # Wikidata
    print("Generating facts from Wikidata...")
    for query_name in WIKIDATA_QUERIES:
        facts = generate_facts_from_wikidata(
            query_name,
            max_facts=wikidata_per_query,
            estimate_difficulty=estimate_difficulty,
        )
        all_facts.extend(facts)
        print(f"    → {len(facts)} facts from '{query_name}'")
        time.sleep(2)  # Rate limiting between queries

    # OEIS
    print("Generating facts from OEIS...")
    oeis_facts = generate_facts_from_oeis(max_facts=oeis_count)
    all_facts.extend(oeis_facts)
    print(f"    → {len(oeis_facts)} facts from OEIS")

    # CODATA
    print("Generating facts from CODATA...")
    codata_facts = generate_facts_from_codata(max_facts=codata_count)
    all_facts.extend(codata_facts)
    print(f"    → {len(codata_facts)} facts from CODATA")

    print(f"\nTotal: {len(all_facts)} facts generated")

    # Distribution
    for diff in range(1, 6):
        count = sum(1 for f in all_facts if f.difficulty == diff)
        print(f"  Difficulty {diff}: {count}")

    return all_facts


def save_facts(facts: list[NumericalFact], path: Path):
    """Save generated facts to JSON."""
    data = []
    for f in facts:
        d = {
            "question": f.question,
            "answer": f.answer,
            "unit": f.unit,
            "tolerance": f.tolerance.value,
            "source": f.source,
            "difficulty": f.difficulty,
            "category": f.category,
            "notes": f.notes,
        }
        data.append(d)

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Saved to {path}")


def load_facts(path: Path) -> list[NumericalFact]:
    """Load generated facts from JSON."""
    data = json.loads(path.read_text())
    facts = []
    for d in data:
        facts.append(NumericalFact(
            question=d["question"],
            answer=d["answer"],
            unit=d["unit"],
            tolerance=Tolerance(d["tolerance"]),
            source=d["source"],
            difficulty=d.get("difficulty"),
            category=d.get("category", ""),
            notes=d.get("notes", ""),
        ))
    return facts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate numerical facts for metacognition benchmark")
    parser.add_argument("--wikidata-per-query", type=int, default=10, help="Facts per Wikidata query type")
    parser.add_argument("--oeis", type=int, default=10, help="Facts from OEIS")
    parser.add_argument("--codata", type=int, default=10, help="Facts from CODATA")
    parser.add_argument("--no-difficulty", action="store_true", help="Skip pageview-based difficulty estimation")
    parser.add_argument("--output", type=str, default="metacognition_benchmark/generated_facts.json")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    facts = generate_all_facts(
        wikidata_per_query=args.wikidata_per_query,
        oeis_count=args.oeis,
        codata_count=args.codata,
        estimate_difficulty=not args.no_difficulty,
    )

    save_facts(facts, Path(args.output))

