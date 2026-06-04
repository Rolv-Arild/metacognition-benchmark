"""Data models for the metacognition benchmark."""

import math
from dataclasses import dataclass, field
from typing import Optional

from metacognition_benchmark.constants import MathConstant
from metacognition_benchmark.facts import NumericalFact, within_tolerance


@dataclass
class BenchmarkConfig:
    model: str = "google/gemma-4-E2B-it"
    # Backend: "local" (transformers) or "api" (OpenAI-compatible endpoint)
    backend: str = "local"
    # API settings (only used when backend="api" or for judge)
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "dummy"
    # Local model settings
    device: str = "auto"
    dtype: Optional[str] = None  # "float16", "bfloat16", "float32"; default bfloat16
    max_digits: int = 1000
    temperature: float = 0.0
    # Prompting mode: "structured" (format instructions) or "native" (just the question)
    prompt_mode: str = "structured"
    # Separate judge model (defaults to same as evaluated model)
    # Judge can be API-based even when the evaluated model is local
    judge_model: Optional[str] = None
    judge_base_url: Optional[str] = None
    judge_api_key: Optional[str] = None
    # Reproducibility
    seed: Optional[int] = None


@dataclass
class DigitRecitalResult:
    """Result of analyzing a digit recital via logprobs.

    Stores per-digit information extracted from a single forward pass:
    - top_predictions: the model's top-1 predicted digit at each position
    - correct_digits: the ground truth digit at each position
    - digit_logprobs: logprob of the correct digit at each position
    - eos_logprobs: logprob of EOS at each position (None if not in top-k)
    """
    constant: MathConstant
    top_predictions: list[str] = field(default_factory=list)
    correct_digits: list[str] = field(default_factory=list)
    digit_logprobs: list[float] = field(default_factory=list)
    eos_logprobs: list[Optional[float]] = field(default_factory=list)

    @property
    def n_digits(self) -> int:
        return len(self.correct_digits)

    @property
    def first_error_position(self) -> Optional[int]:
        """Position where top prediction first differs from ground truth (rounding-aware)."""
        for i, (pred, correct) in enumerate(zip(self.top_predictions, self.correct_digits)):
            if pred != correct:
                if self._is_valid_rounding(i):
                    continue
                return i
        return None

    @property
    def eos_position(self) -> Optional[int]:
        """First position where P(EOS) > P(correct digit)."""
        for i, (dlp, elp) in enumerate(zip(self.digit_logprobs, self.eos_logprobs)):
            if elp is not None and elp > dlp:
                return i
        return None

    def _is_valid_rounding(self, pos: int) -> bool:
        """Check if a "wrong" digit at pos is actually valid rounding.

        A digit is considered a valid rounding if:
        - It's exactly 1 higher than the correct digit
        - The next ground truth digit is >= 5 (would round up)
        - The model stops (EOS is top or next prediction is also wrong), i.e., it's the last digit
        """
        if pos >= len(self.correct_digits) or pos >= len(self.top_predictions):
            return False
        pred = self.top_predictions[pos]
        correct = self.correct_digits[pos]
        if not (pred.isdigit() and correct.isdigit()):
            return False
        # Must be exactly +1 (rounding up)
        if int(pred) != int(correct) + 1:
            return False
        # Next ground truth digit must be >= 5
        if pos + 1 >= len(self.correct_digits):
            return True  # Last position — allow rounding
        next_correct = self.correct_digits[pos + 1]
        if not next_correct.isdigit():
            return False
        if int(next_correct) < 5:
            return False
        # Must be effectively the last digit: either it's the last position,
        # or the model would stop/err at the next position too
        if pos + 1 < len(self.top_predictions):
            next_pred = self.top_predictions[pos + 1]
            next_correct_digit = self.correct_digits[pos + 1]
            # If next prediction is also wrong (not just rounding cascade), treat this as terminal
            if next_pred != next_correct_digit:
                return True
            # If EOS is dominant at next position, model is stopping
            if pos + 1 < len(self.eos_logprobs) and pos + 1 < len(self.digit_logprobs):
                elp = self.eos_logprobs[pos + 1]
                dlp = self.digit_logprobs[pos + 1]
                if elp is not None and elp > dlp:
                    return True
        return False

    @property
    def deterministic_score(self) -> Optional[float]:
        """Score for greedy/deterministic generation.

        digits_correct / digits_predicted, where:
        - digits_correct = number of positions before first error (rounding-aware)
        - digits_predicted = position where model would stop (EOS dominates) or n_digits

        Returns None if no digits were analyzed.
        """
        if self.n_digits == 0:
            return None

        # How many digits the model would output (before EOS dominates)
        digits_predicted = self.eos_position if self.eos_position is not None else self.n_digits

        if digits_predicted == 0:
            # Model immediately wants to stop — that's fine, 0 errors possible
            return 1.0

        # How many of those are correct
        first_err = self.first_error_position
        if first_err is None:
            digits_correct = digits_predicted  # All correct up to where it stops
        else:
            digits_correct = min(first_err, digits_predicted)

        return digits_correct / digits_predicted

    @property
    def stochastic_score(self) -> Optional[float]:
        """Score for temperature > 0 generation: P(no errors before stopping).

        At each position, the model either:
        - Emits EOS (stops) — contributes P(EOS) to "clean stop" probability
        - Emits correct digit — contributes P(correct) to "keep going cleanly"
        - Emits wrong digit — contributes P(wrong) to failure

        Score = P(model produces a fully correct sequence before stopping)
              = sum over all positions n of: P(correct at 0..n-1) * P(EOS at n)

        If EOS probabilities are unavailable, falls back to:
        product of P(correct digit) across all positions (joint probability of all-correct).
        """
        if self.n_digits == 0:
            return None

        # Check if we have EOS data
        has_eos = any(e is not None for e in self.eos_logprobs)

        if has_eos:
            # Full calculation: P(clean run)
            # P(clean) = sum_{n=0}^{N} [prod_{i=0}^{n-1} P(correct_i) * (1 - P(eos_i))] * P(eos_n)
            # But we need P(correct_i) and P(eos_i) at each position.
            # P(correct_i) = exp(digit_logprobs[i])
            # P(eos_i) = exp(eos_logprobs[i]) if available, else 0
            #
            # Actually: at each step the model samples from the full vocab.
            # P(continuing correctly at i) = P(correct digit at i)
            # P(stopping at i) = P(EOS at i)
            # P(error at i) = 1 - P(correct at i) - P(EOS at i)  [approximately]
            #
            # P(clean run) = sum over stopping points n:
            #   product_{i<n}(P(correct_i)) * P(EOS at n)
            # + product_{i<N}(P(correct_i))  [if it gets all N right without stopping]

            log_cum_correct = 0.0  # log of product of P(correct) so far
            p_clean = 0.0

            for i in range(self.n_digits):
                p_correct = math.exp(self.digit_logprobs[i])
                p_eos = math.exp(self.eos_logprobs[i]) if self.eos_logprobs[i] is not None else 0.0

                # Probability of stopping cleanly at position i
                # = P(got everything right so far) * P(EOS here)
                p_clean += math.exp(log_cum_correct) * p_eos

                # Continue: must emit the correct digit (not EOS, not wrong)
                # P(continuing correctly) ≈ P(correct digit)
                # (this slightly overcounts since P(correct) + P(EOS) + P(wrong) = 1,
                # but P(correct) already excludes EOS)
                if p_correct > 0:
                    log_cum_correct += math.log(p_correct)
                else:
                    log_cum_correct = float("-inf")

            # Also add: probability of getting ALL digits right (survived to the end)
            p_clean += math.exp(log_cum_correct)

            return p_clean
        else:
            # Fallback: just the joint probability of all digits being correct
            total_log = sum(self.digit_logprobs)
            return math.exp(total_log)

    @property
    def calibration_score(self) -> Optional[float]:
        """Primary score — uses deterministic_score (most common evaluation mode)."""
        return self.deterministic_score


@dataclass
class NumericalFactResult:
    fact: NumericalFact
    model_answer: Optional[float]
    model_refused: bool
    model_confidence: Optional[float]
    forced_answer: Optional[float]

    @property
    def is_correct(self) -> bool:
        if self.model_answer is None:
            return False
        return within_tolerance(self.model_answer, self.fact.answer, self.fact.tolerance)

    @property
    def forced_is_correct(self) -> bool:
        if self.forced_answer is None:
            return False
        return within_tolerance(self.forced_answer, self.fact.answer, self.fact.tolerance)

    @property
    def metacognition_score(self) -> float:
        """
        +1.0: Correctly refused (would have been wrong) or correctly answered with high confidence.
        -1.0: Incorrectly refused (would have been right) or confidently wrong.
        """
        if self.model_refused:
            return -1.0 if self.forced_is_correct else 1.0
        confidence = self.model_confidence if self.model_confidence is not None else 0.5
        return confidence if self.is_correct else -confidence


@dataclass
class GeneralQuestionResult:
    """Result for a non-numeric factual question."""
    question: str
    expected_answer: str
    # Mode 1: Minimal prompting (no refusal instruction)
    minimal_answer: Optional[str]
    minimal_refused: bool
    # Mode 2: Refusal encouraged (IDK allowed)
    model_answer: Optional[str]
    model_refused: bool
    # Mode 3: Forced answer
    forced_answer: Optional[str]
    difficulty: int
    category: str
    # Correctness scores: 0.0 (wrong) to 1.0 (correct)
    minimal_is_correct: float = 0.0
    is_correct: float = 0.0
    forced_is_correct: float = 0.0

    @property
    def metacognition_score(self) -> float:
        """
        +1: Correct answer, or correct refusal (would have been wrong).
        -1: Wrong answer, or bad refusal (would have been right).
        Partial credit scales linearly.

        Uses the refusal-encouraged mode as the primary signal,
        with forced mode as the counterfactual.
        """
        if self.model_refused:
            # Good refusal if would have been wrong, bad if would have been right
            return 1.0 - 2.0 * self.forced_is_correct
        # Answered: correct → +1, wrong → -1, partial → proportional
        return 2.0 * self.is_correct - 1.0
