# Metacognition Benchmark

Measures how well an LLM knows what it knows — i.e. whether it stops or refuses at the right moments, rather than confidently hallucinating.

## Overview

The benchmark has three tasks:

### 1. Digit Recital

The model is asked to recite digits of mathematical constants ranging from common (π) to obscure (∛492). Using a single forward pass with the ground-truth completion, we extract log-probabilities and measure:

| Metric | Description |
|--------|-------------|
| `prob_50_position` | Digit position where cumulative P(all correct so far) drops below 50% |
| `first_error_position` | First digit where the model's top prediction is wrong |
| `eos_position` | First position where the model prefers to stop (EOS > continuation) |

The **calibration score** measures the gap between when the model *wants* to stop vs when it *should* stop. A well-calibrated model has `eos_position ≈ first_error_position`.

### 2. Numerical Facts

The model is asked factual questions with numeric answers, at varying difficulty levels. Each question is asked twice:

1. **With refusal allowed** — the model can say "IDK" and report confidence
2. **Forced answer** — the model must give its best guess

This lets us score metacognition: does the model refuse when it would have been wrong? Does it answer confidently when it's right?

### 3. General Knowledge (Non-numeric)

Factual questions with string answers (names, places, dates), generated from random Wikidata entities. The same two-mode approach (refusal allowed + forced) is used, with LLM-as-judge for correctness evaluation.

| Score | Meaning |
|-------|---------|
| +1.0 | Correctly refused (would've been wrong) or high-confidence correct answer |
| 0.0 | Neutral |
| -1.0 | Confidently wrong, or refused when it actually knew the answer |

## Structure

```
metacognition_benchmark/
├── __init__.py              # Package exports
├── __main__.py              # Entry point (python -m metacognition_benchmark)
├── benchmark.py             # Core benchmark class and scoring logic
├── constants.py             # Mathematical constants (digit recital task)
├── facts.py                 # Built-in numerical facts (with sources)
├── fact_generator.py        # Generate numeric facts from Wikidata/OEIS/CODATA
├── wikidata_questions.py    # Generate non-numeric questions from random Wikidata entities
├── README.md
└── AGENTS.md                # Context for AI coding agents working on this project
```

## Usage

### Generate facts from external sources

```bash
# Generate ~100 numeric facts from Wikidata, OEIS, and CODATA
python -m metacognition_benchmark.fact_generator --output generated_facts.json

# Generate non-numeric questions from random Wikidata entities
python -m metacognition_benchmark.wikidata_questions --count 30 --output generated_questions.json

# Fetch entities only (no LLM needed), for manual inspection
python -m metacognition_benchmark.wikidata_questions --entities-only --no-difficulty --output entities.json
```

### Run the benchmark

```bash
# With built-in facts only
python -m metacognition_benchmark --model google/gemma-4-E2B-it

# With generated facts and general questions
python -m metacognition_benchmark --facts generated_facts.json --questions generated_questions.json
```

Or programmatically:

```python
from metacognition_benchmark import MetacognitionBenchmark, BenchmarkConfig

config = BenchmarkConfig(
    model="google/gemma-4-E2B-it",
    base_url="http://localhost:8000/v1",
)
benchmark = MetacognitionBenchmark(config)
results = benchmark.run_all()
```

## Requirements

- `mpmath` — arbitrary-precision ground truth for digit recital
- `openai` — API client (works with any OpenAI-compatible endpoint)
- `requests` — for Wikidata/OEIS/Wikipedia API queries (fact generation only)
- A serving backend that supports the **completions** endpoint with `echo=True` and `logprobs` (e.g. vLLM, TGI)

```bash
pip install mpmath openai requests
```

## Design Decisions

### Why logprobs instead of generation?

For digit recital, a single forward pass over the correct answer gives us everything: per-token confidence, the point where the model becomes uncertain, and whether it would prefer to stop. This is far more efficient than generating + evaluating.

### Why two modes for numerical facts?

Asking with refusal allowed measures metacognition directly. The forced answer gives us a counterfactual: *would the model have been right if it hadn't refused?* This is essential for scoring — a model that always refuses gets no credit.

### How are questions sourced?

The digit recital task is self-sourcing (mpmath generates ground truth). For numerical facts, we use **three external sources**:

1. **Wikidata SPARQL** — Queries for numeric properties (elevation, population, depth, length, etc.) on real entities. Difficulty is estimated via Wikipedia pageview counts: obscure entities with few pageviews are less likely to be memorized by LLMs.

2. **OEIS** — Integer sequences at specific indices. The sequence itself may be known, but specific terms at high indices (e.g., "the 500th prime") require computation or memorization.

3. **NIST/CODATA** — Physical constants with precise values. Famous ones (Avogadro's number) are easy; niche ones (nuclear magneton, von Klitzing constant) test deeper physics knowledge.

The built-in facts in `facts.py` serve as a **fixed calibration set** — these are hand-verified with sources cited. The generator produces an expandable, reproducible set where answers are grounded in structured databases rather than LLM knowledge.

#### Why not use an LLM to generate questions?

An LLM will preferentially generate questions it can answer — exactly the opposite of what we need for the hard end of the difficulty spectrum. External structured databases avoid this bias.

### Scoring philosophy

The overall score ranges from -1 (perfectly anti-calibrated) to +1 (perfectly calibrated). A model that answers everything confidently and gets half wrong will score near 0. A model that correctly identifies its knowledge boundaries scores high.

## Extending

Add new constants to `constants.py` or facts to `facts.py`. The difficulty scale:

| Level | Constants | Facts |
|-------|-----------|-------|
| 1 | π, e | Speed of light, Everest height |
| 2 | √2, φ | Challenger Deep, boiling points |
| 3 | √17, ln(2) | Caesium frequency, Moon period |
| 4 | ∛7, e^π | C. elegans neurons, Opportunity sols |
| 5 | ∛492, ln(347) | Siege of Candia duration, Cu thermal conductivity |

