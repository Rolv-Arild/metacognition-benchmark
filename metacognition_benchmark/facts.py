"""Curated numerical facts for the metacognition benchmark.

These facts are kept ONLY when they satisfy all of:
1. The answer comes from a single, specific authoritative source (not "various")
2. The value is stable over time (not discovery counts, growing trees, etc.)
3. The question tests something Wikidata can't easily provide (exact definitions,
   specific paper citations, multi-step lookups)

For general geographic/demographic/scientific measurements, prefer the Wikidata
and CODATA generators which provide verifiable ground truth automatically.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Tolerance(Enum):
    """How close the answer needs to be."""
    EXACT = "exact"          # Must match exactly (integers, defined constants)
    NARROW = "narrow"        # Within 0.5% (well-established measurements)
    MODERATE = "moderate"    # Within 2% (measurements with some uncertainty)
    WIDE = "wide"            # Within 10% (estimates, contested values)


@dataclass
class NumericalFact:
    """A factual question with a numeric answer."""

    question: str
    answer: float
    unit: str
    tolerance: Tolerance
    source: str
    category: str = ""
    notes: str = ""
    # Optional: only set by generators that have empirical data (e.g., pageviews).
    # None means "unknown difficulty" — will not be used in difficulty-stratified reports.
    difficulty: Optional[int] = None


def within_tolerance(predicted: float, actual: float, tolerance: Tolerance) -> bool:
    """Check if predicted value is within the specified tolerance of actual."""
    if actual == 0:
        return predicted == 0

    ratio = abs(predicted - actual) / abs(actual)

    if tolerance == Tolerance.EXACT:
        if isinstance(actual, int) or actual == int(actual):
            return predicted == actual
        return ratio < 0.0001
    elif tolerance == Tolerance.NARROW:
        return ratio < 0.005
    elif tolerance == Tolerance.MODERATE:
        return ratio < 0.02
    elif tolerance == Tolerance.WIDE:
        return ratio < 0.10
    return False


# ---------------------------------------------------------------------------
# Curated questions with verified answers from authoritative sources.
#
# Inclusion criteria:
# - Exact definitions (SI, IAU) that test precise recall
# - Single-source facts from specific named papers/documents
# - Values that are NOT easily available via Wikidata SPARQL
# - Temporally stable (won't change with new discoveries)
# ---------------------------------------------------------------------------

NUMERICAL_FACTS = [
    # ---- SI / metrological definitions (exact by definition) ----
    NumericalFact(
        question="What is the speed of light in a vacuum in meters per second?",
        answer=299_792_458,
        unit="m/s",
        tolerance=Tolerance.EXACT,
        source="SI definition: BIPM Resolution 1 of the 17th CGPM (1983)",
        category="physics",
    ),
    NumericalFact(
        question="What is the exact integer frequency in Hertz that defines the SI second, based on the caesium-133 atom?",
        answer=9_192_631_770,
        unit="Hz",
        tolerance=Tolerance.EXACT,
        source="SI definition: BIPM Resolution 1 of the 13th CGPM (1967)",
        category="physics",
    ),
    NumericalFact(
        question="What is the distance from the Earth to the Sun in kilometers (1 AU)?",
        answer=149_597_870.7,
        unit="km",
        tolerance=Tolerance.EXACT,
        source="IAU 2012 Resolution B2: 1 AU = 149,597,870,700 m exactly",
        category="astronomy",
    ),

    # ---- Well-established reference values (CRC Handbook, named sources) ----
    NumericalFact(
        question="What is the boiling point of nitrogen at standard pressure (1 atm) in Kelvin?",
        answer=77.36,
        unit="K",
        tolerance=Tolerance.NARROW,
        source="CRC Handbook of Chemistry and Physics, 97th ed., Section 4",
        category="chemistry",
    ),
    NumericalFact(
        question="What is the thermal conductivity of copper at 20°C in watts per meter-kelvin?",
        answer=401,
        unit="W/(m·K)",
        tolerance=Tolerance.NARROW,
        source="CRC Handbook of Chemistry and Physics, 97th ed., Section 12",
        category="physics",
    ),
    NumericalFact(
        question="What is the half-life of Carbon-14 in years (Cambridge half-life)?",
        answer=5730,
        unit="years",
        tolerance=Tolerance.NARROW,
        source="Godwin, H. (1962). Nature, 195(4845), 984. doi:10.1038/195984a0",
        category="physics",
        notes="Libby half-life (5568 years) is older and less accurate but still used in some contexts",
    ),

    # ---- Facts from specific papers / official documents ----
    NumericalFact(
        question="How many individual neurons comprise the entire nervous system of an adult hermaphrodite Caenorhabditis elegans?",
        answer=302,
        unit="neurons",
        tolerance=Tolerance.EXACT,
        source="White, J.G. et al. (1986). Phil. Trans. R. Soc. Lond. B, 314(1165), 1-340. doi:10.1098/rstb.1986.0056",
        category="biology",
    ),
    NumericalFact(
        question="How many operational Martian days (sols) did NASA's Opportunity rover function on the surface of Mars before its final communication?",
        answer=5111,
        unit="sols",
        tolerance=Tolerance.EXACT,
        source="NASA JPL Mars Exploration Rovers mission log: last contact Sol 5111, June 10 2018. https://mars.nasa.gov/mer/",
        category="space",
    ),
    NumericalFact(
        question="What is the total number of tiles in a standard Scrabble set (English edition)?",
        answer=100,
        unit="tiles",
        tolerance=Tolerance.EXACT,
        source="Official Scrabble Players Dictionary; Hasbro Inc. official rules",
        category="games",
    ),
    NumericalFact(
        question="What is the sidereal orbital period of the Moon around Earth in days?",
        answer=27.321661,
        unit="days",
        tolerance=Tolerance.NARROW,
        source="NASA Moon Fact Sheet, https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html",
        category="astronomy",
    ),
]

