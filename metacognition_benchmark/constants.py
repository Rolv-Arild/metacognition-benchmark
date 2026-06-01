"""Mathematical constants for the digit recital task."""

from dataclasses import dataclass
from typing import Callable

import mpmath

mpmath.mp.dps = 500


@dataclass
class MathConstant:
    """A mathematical constant for the digit recital task."""

    name: str
    description: str
    value_func: Callable[[], "mpmath.mpf"]  # Returns mpmath arbitrary-precision value
    difficulty: int  # 1 (trivial/common) to 5 (certainly not in training data)

    def digits(self, n: int = 1000) -> tuple[str, str]:
        """Return (integer_part, first n digits after the decimal point)."""
        with mpmath.workdps(n + 50):
            val = self.value_func()
            s = mpmath.nstr(val, n + 10, strip_zeros=False)
            if "." in s:
                integer_part, decimal_part = s.split(".")
                return integer_part, decimal_part[:n]
            return s, ""


MATH_CONSTANTS = [
    # Difficulty 1: Definitely memorized
    MathConstant("π", "pi", lambda: mpmath.pi, difficulty=1),
    MathConstant("e", "Euler's number", lambda: mpmath.e, difficulty=1),
    # Difficulty 2: Common but less drilled
    MathConstant("√2", "square root of 2", lambda: mpmath.sqrt(2), difficulty=2),
    MathConstant("φ", "golden ratio (1+√5)/2", lambda: (1 + mpmath.sqrt(5)) / 2, difficulty=2),
    # Difficulty 3: Probably in training data but not common
    MathConstant("√3", "square root of 3", lambda: mpmath.sqrt(3), difficulty=3),
    MathConstant("ln(2)", "natural log of 2", lambda: mpmath.log(2), difficulty=3),
    MathConstant("√17", "square root of 17", lambda: mpmath.sqrt(17), difficulty=3),
    # Difficulty 4: Unlikely to be memorized
    MathConstant("∛7", "cube root of 7", lambda: mpmath.cbrt(7), difficulty=4),
    MathConstant("√223", "square root of 223", lambda: mpmath.sqrt(223), difficulty=4),
    MathConstant("e^π", "e to the power of pi", lambda: mpmath.power(mpmath.e, mpmath.pi), difficulty=4),
    # Difficulty 5: Almost certainly not in training data
    MathConstant("∛492", "cube root of 492", lambda: mpmath.cbrt(492), difficulty=5),
    MathConstant("√8291", "square root of 8291", lambda: mpmath.sqrt(8291), difficulty=5),
    MathConstant("ln(347)", "natural log of 347", lambda: mpmath.log(347), difficulty=5),
    MathConstant("π^e", "pi to the power of e", lambda: mpmath.power(mpmath.pi, mpmath.e), difficulty=5),
]

