# AGENTS.md

Context for AI coding agents working on this project.

## Project Overview

This is a **metacognition benchmark for LLMs** — it measures how well a model knows what it knows. The core insight: we care about calibration (does the model refuse/stop at the right moments?) more than raw accuracy.

## Architecture

```
benchmark.py          — The orchestrator. MetacognitionBenchmark class runs all tasks,
                        calls the model (local or API), and aggregates scores.
inference.py          — LocalModel class: loads HuggingFace transformers models,
                        provides get_logprobs_for_completion() and generate().
models.py             — Data classes: BenchmarkConfig, DigitRecitalResult,
                        NumericalFactResult, GeneralQuestionResult.
scoring.py            — Scoring logic: refusal detection, response parsing,
                        digit logprob analysis. No API calls — pure functions.
constants.py          — MathConstant dataclass + list of 14 constants (difficulty 1-5).
                        Ground truth computed via mpmath at arbitrary precision.
facts.py              — NumericalFact dataclass, tolerance checking, and 10 curated
                        numeric questions with verified single-source citations.
generators/
  fact_generator.py   — Generates numeric facts from Wikidata SPARQL, OEIS, and CODATA.
                        Uses Wikipedia pageviews as a difficulty proxy.
  wikidata_questions.py — Generates non-numeric questions by fetching random Wikidata
                        entities and using an LLM to compose questions.
data/
  allascii.txt        — NIST CODATA 2022 Fundamental Physical Constants (355 entries).
```

## Key Design Constraints

1. **The LLM paradox**: We cannot use an LLM to generate questions it would then know the answers to. All ground truth must come from external sources (mpmath, Wikidata, OEIS, NIST).

2. **Difficulty estimation**: Uses Wikipedia pageview counts as a proxy for "how likely is this in training data." Low pageviews → high difficulty.

3. **Two-mode evaluation**: Every factual question is asked twice — once allowing refusal ("IDK"), once forcing an answer. This gives us the counterfactual needed for scoring.

4. **Digit recital uses logprobs**: Primary backend uses HuggingFace transformers
   for a single forward pass with full logit access. API backend (optional) requires
   a vLLM-style completions endpoint with `echo=True` and `logprobs`.

5. **LLM-as-judge for non-numeric answers**: `benchmark.py` uses the same model (or could use a separate judge model) to determine if string answers are semantically correct.

## Key Types

- `BenchmarkConfig` — model name, backend (local/api), device, dtype, API URL, temperature, max_digits, judge_model, judge_base_url, seed
- `MathConstant` — name, description, value_func (mpmath callable), difficulty
- `NumericalFact` — question, answer (float), unit, tolerance, source, difficulty
- `GeneratedQuestion` — question, answer (str), category, source_entity, difficulty
- `DigitRecitalResult` — logprob analysis results with `calibration_score` property
- `NumericalFactResult` — model_answer, model_refused, confidence, `metacognition_score` property
- `GeneralQuestionResult` — same pattern for non-numeric questions

## Scoring

- **Digit recital**: `calibration_score` ∈ [-1, 1]. Positive = model stops before errors (good).
- **Facts/Questions**: `metacognition_score` ∈ [-1, 1]. Correct refusal or confident-and-right = positive. Confident-and-wrong or refused-but-knew = negative.
- **Overall**: Average of digit calibration score and fact/question metacognition scores.

## Dependencies

- `mpmath` — arbitrary precision math
- `transformers` — HuggingFace model loading and tokenization
- `torch` — tensor computation and model inference
- `accelerate` — multi-GPU and device placement
- `requests` — HTTP calls to Wikidata/OEIS/Wikipedia
- `openai` (optional, `[api]` extra) — for API backend or API-based judge model

## Common Tasks

### Adding new math constants
Edit `constants.py`. Provide a lambda that returns an mpmath value. Assign difficulty 1-5.

### Adding new numeric facts
Edit `facts.py`. Provide question, answer, unit, tolerance level, source citation, and difficulty.

### Adding new Wikidata query types
Edit `generators/fact_generator.py` → `WIKIDATA_QUERIES` dict. Each entry needs a SPARQL query, question template, category, and tolerance.

### Adding new Wikidata property types for general questions
Edit `generators/wikidata_questions.py` → `INTERESTING_PROPERTIES` dict. Map Wikidata property IDs to human-readable labels and value types.

### Changing the scoring formula
Edit `metacognition_score` property in `NumericalFactResult` or `GeneralQuestionResult` in `models.py`.

### Supporting a different LLM API
The benchmark supports two backends: `local` (HuggingFace transformers, default) and `api` (OpenAI-compatible). The local backend gives full logprob access for digit recital. The API backend requires `pip install metacognition-benchmark[api]` and needs a vLLM-style endpoint with `echo=True` and `logprobs` for digit recital.

## Testing

No test suite yet. To validate locally:
```bash
# Check imports
python -c "from metacognition_benchmark import MetacognitionBenchmark"

# Test fact generation (no LLM needed)
python -m metacognition_benchmark.generators.fact_generator --no-difficulty --codata 5 --oeis 0 --wikidata-per-query 0 --output /tmp/test.json

# Test Wikidata entity fetching (no LLM needed)
python -m metacognition_benchmark.generators.wikidata_questions --entities-only --count 3 --no-difficulty --output /tmp/entities.json
```

## Future Directions

- Add a proper test suite with mocked API responses
- Support batched inference for efficiency
- Add visualization of calibration curves (logprob vs digit position)
- Separate judge model from evaluated model
- Add temporal knowledge task (questions about events after various dates)
- Export results in a standardized format for cross-model comparison

