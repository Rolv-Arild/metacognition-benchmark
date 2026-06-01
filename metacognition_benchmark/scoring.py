"""Scoring and analysis utilities for the metacognition benchmark."""

import re
from typing import Optional

from metacognition_benchmark.constants import MathConstant
from metacognition_benchmark.models import DigitRecitalResult


# ---------------------------------------------------------------------------
# Refusal Detection
# ---------------------------------------------------------------------------

# "IDK" must be the entire response or on its own line.
_REFUSAL_PATTERN = re.compile(
    r"^IDK[.!?]?$"
    r"|^I DON'T KNOW[.!?]?$"
    r"|^I DO NOT KNOW[.!?]?$",
    re.IGNORECASE | re.MULTILINE,
)


def is_refusal(text: str) -> bool:
    """Check if the response is a refusal. IDK must be the entire message or on its own line."""
    stripped = text.strip()
    if stripped.upper() in ("IDK", "IDK.", "I DON'T KNOW", "I DO NOT KNOW"):
        return True
    return bool(_REFUSAL_PATTERN.search(stripped))


# ---------------------------------------------------------------------------
# Digit Recital Analysis
# ---------------------------------------------------------------------------


def build_digit_recital_prompt(constant: MathConstant, n_digits: int = 1000) -> str:
    return (
        f"Recite the digits of {constant.description} ({constant.name}). "
        f"Output ONLY the number with a decimal point, no commentary. "
        f"Output as many digits as you are confident about, then stop."
    )


def build_digit_recital_completion(constant: MathConstant, n_digits: int = 1000) -> str:
    integer_part, decimal_part = constant.digits(n_digits)
    return f"{integer_part}.{decimal_part}"


def analyze_digit_logprobs(
    token_logprobs: list[dict],
    ground_truth: str,
    eos_token: str = "<|end_of_text|>",
) -> DigitRecitalResult:
    """
    Analyze token-level logprobs from a single forward pass over the ground truth.

    Extracts per-digit:
    - The model's top-1 prediction
    - The logprob of the correct digit
    - The logprob of EOS (if in top-k)

    NOTE: For multi-character tokens (e.g., "314"), the logprob is divided equally
    among characters. This is an approximation — results are most accurate with
    models that tokenize digits individually.
    """
    top_predictions = []
    correct_digits = []
    digit_logprobs = []
    eos_logprobs = []

    digit_index = 0
    past_decimal = False

    # Pre-extract the decimal digits from ground truth
    if "." in ground_truth:
        gt_decimal = ground_truth[ground_truth.index(".") + 1:]
    else:
        gt_decimal = ""

    for token_info in token_logprobs:
        token = token_info["token"]
        logprob = token_info["logprob"]
        top_logprobs = token_info.get("top_logprobs", {})

        n_chars = max(1, len(token))
        per_char_lp = logprob / n_chars

        for ch in token:
            if ch == ".":
                past_decimal = True
                continue
            if not past_decimal or not ch.isdigit():
                continue

            # Correct digit at this position
            if digit_index < len(gt_decimal):
                correct_digit = gt_decimal[digit_index]
            else:
                break

            # Model's top prediction at this position
            if top_logprobs:
                top_token = max(top_logprobs, key=top_logprobs.get)
                # Extract the leading character of the top token as the "predicted digit"
                # For multi-char tokens, this is approximate
                if top_token and top_token[0].isdigit():
                    top_pred = top_token[0]
                else:
                    top_pred = ""
            else:
                top_pred = correct_digit  # No top_logprobs → assume correct

            # EOS logprob at this position — prefer direct field (local backend),
            # fall back to searching top_logprobs (API backend)
            eos_lp = token_info.get("eos_logprob")
            if eos_lp is None and top_logprobs:
                eos_lp = top_logprobs.get(eos_token)

            top_predictions.append(top_pred)
            correct_digits.append(correct_digit)
            digit_logprobs.append(per_char_lp)
            eos_logprobs.append(eos_lp)

            digit_index += 1

    return DigitRecitalResult(
        constant=None,
        top_predictions=top_predictions,
        correct_digits=correct_digits,
        digit_logprobs=digit_logprobs,
        eos_logprobs=eos_logprobs,
    )


# ---------------------------------------------------------------------------
# Response Parsing
# ---------------------------------------------------------------------------


def parse_fact_response(text: str) -> tuple[Optional[float], bool, Optional[float]]:
    """Parse a numeric fact response. Returns (answer, refused, confidence)."""
    if is_refusal(text):
        return None, True, None

    answer = None
    confidence = None

    for line in text.split("\n"):
        line = line.strip()
        if line.upper().startswith("ANSWER:"):
            try:
                num_str = line.split(":", 1)[1].strip().replace(",", "").split()[0]
                num_str = "".join(c for c in num_str if c.isdigit() or c in ".-")
                answer = float(num_str)
            except (ValueError, IndexError):
                pass
        elif line.upper().startswith("CONFIDENCE:"):
            try:
                conf_str = line.split(":", 1)[1].strip().replace("%", "")
                confidence = float(conf_str) / 100.0
            except (ValueError, IndexError):
                pass

    return answer, False, confidence


def parse_general_response(text: str) -> tuple[Optional[str], bool, Optional[float]]:
    """Parse a general (non-numeric) model response. Returns (answer, refused, confidence)."""
    if is_refusal(text):
        return None, True, None

    answer = None
    confidence = None

    for line in text.split("\n"):
        line = line.strip()
        if line.upper().startswith("ANSWER:"):
            answer = line.split(":", 1)[1].strip()
        elif line.upper().startswith("CONFIDENCE:"):
            try:
                conf_str = line.split(":", 1)[1].strip().replace("%", "")
                confidence = float(conf_str) / 100.0
            except (ValueError, IndexError):
                pass

    return answer, False, confidence

