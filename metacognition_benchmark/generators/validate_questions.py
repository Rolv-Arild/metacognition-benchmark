"""
Validate generated questions using an LLM as a quality filter.

Usage:
    python -m metacognition_benchmark.generators.validate_questions --input generated_questions.json --output validated_questions.json
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from metacognition_benchmark.generators.wikipedia_questions import (
    GeneratedQuestion,
    load_questions,
    save_questions,
)


# ---------------------------------------------------------------------------
# Validation prompts and logic
# ---------------------------------------------------------------------------

VALIDATION_SYSTEM_PROMPT = """\
You are a quality checker for trivia questions. You validate that questions follow strict rules. \
You always respond with valid JSON arrays. Be strict — reject anything borderline."""

VALIDATION_PROMPT = """\
Review the following trivia questions and reject any that violate these rules:

1. The answer must NOT be obvious from the question itself (e.g. "What time are ghosts active in the film 3AM?" → answer is in the title)
2. All answers/aliases must be in English (no Italian, Swedish, etc. translations as aliases)
3. The question must be self-contained and answerable without knowing the source article
4. The question must ask about a concrete fact, not a tautology or circular reference
5. The question should not be trivially answerable from common sense alone
6. The answer must fit one of these types ONLY:
   - Person name (full name, not surname alone)
   - Place name (city, country, landmark, etc.)
   - Year or date
   - Bare number (with the unit specified in the question, not the answer)
   - Organization/company name
   - Creative work title (film, book, song, etc.)
   - Language name
   - Chemical/biological term (element, compound, species, disease)
   - Religion or philosophy
   - Currency
   - Sport or game
   - Musical instrument
   - Animal or plant name
   - Color (only when not guessable from context)
   - Material/substance (only when specific)
   - Named concept (≤4 words, specific, not guessable)
7. REJECT if the answer is: a generic word that could fit dozens of questions (e.g. "water", "iron", "fast"), a full sentence/phrase/description, a position/rank/role (e.g. "midfielder", "fourth place", "engineer"), or a yes/no.

For each question, respond with "keep" or "reject" and a brief reason if rejected.

Output format (JSON array with one entry per question):
[
  {{"index": 0, "verdict": "keep"}},
  {{"index": 1, "verdict": "reject", "reason": "answer is a generic common word"}},
  ...
]

Questions to validate:
{questions_json}
"""


def validate_questions(
        questions: list[GeneratedQuestion],
        chat_fn,
        batch_size: int = 10,
        max_workers: int = 1,
) -> list[GeneratedQuestion]:
    """Use an LLM to validate generated questions, filtering out low-quality ones."""
    if not questions:
        return []

    batches = []
    for i in range(0, len(questions), batch_size):
        batches.append((i, questions[i:i + batch_size]))

    def _validate_batch(batch_idx_and_questions):
        batch_i, batch = batch_idx_and_questions
        questions_for_prompt = [
            {"index": j, "question": q.question, "answers": q.answers}
            for j, q in enumerate(batch)
        ]

        prompt = VALIDATION_PROMPT.format(
            questions_json=json.dumps(questions_for_prompt, ensure_ascii=False)
        )

        try:
            text = chat_fn(
                [
                    {"role": "system", "content": VALIDATION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperature=0.0,
            )

            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            verdicts = json.loads(text)
            keep_indices = {v["index"] for v in verdicts if v.get("verdict") == "keep"}

            return batch_i, [q for j, q in enumerate(batch) if j in keep_indices]

        except (json.JSONDecodeError, KeyError, Exception):
            return batch_i, list(batch)

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_validate_batch, b): b[0] for b in batches}
        for future in as_completed(futures):
            batch_i, kept = future.result()
            results[batch_i] = kept

    # Reassemble in original order
    valid = []
    for batch_i, _ in batches:
        valid.extend(results[batch_i])

    return valid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate generated trivia questions")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file with questions")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file (default: input with .validated suffix)")
    parser.add_argument("--llm-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--llm-model", type=str, default="google/gemma-4-E4B-it")
    parser.add_argument("--llm-api-key", type=str, default="dummy")
    parser.add_argument("--api", action="store_true", help="Use OpenAI-compatible API")
    parser.add_argument("--quantization", type=str, default=None, choices=["4bit", "8bit"])
    parser.add_argument("--batch-size", type=int, default=10, help="Questions per validation batch")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent validation workers")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".validated.json")

    questions = load_questions(input_path)
    print(f"Loaded {len(questions)} questions from {input_path}")

    # Set up chat function
    if not args.api:
        from metacognition_benchmark.inference import LocalModel
        print(f"Loading local model: {args.llm_model}")
        model = LocalModel(args.llm_model, quantization=args.quantization)

        def chat_fn(messages, max_tokens=1000, temperature=0.3):
            return model.generate(messages, max_tokens=max_tokens, temperature=temperature)
    else:
        from openai import OpenAI
        client = OpenAI(base_url=args.llm_url, api_key=args.llm_api_key)

        def chat_fn(messages, max_tokens=1000, temperature=0.3):
            response = client.chat.completions.create(
                model=args.llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()

    validated = validate_questions(questions, chat_fn, batch_size=args.batch_size, max_workers=args.workers)
    print(f"Kept {len(validated)}/{len(questions)} questions after validation")

    save_questions(validated, output_path)
