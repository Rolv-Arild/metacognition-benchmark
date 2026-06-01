"""Run the metacognition benchmark."""

import argparse
from pathlib import Path

from metacognition_benchmark.benchmark import MetacognitionBenchmark
from metacognition_benchmark.models import BenchmarkConfig
from metacognition_benchmark.generators import load_facts


def main():
    parser = argparse.ArgumentParser(description="Run metacognition benchmark")
    parser.add_argument("--model", type=str, default="google/gemma-4-4b-it")
    parser.add_argument("--backend", type=str, choices=["local", "api"], default="local",
                        help="Inference backend: 'local' (transformers) or 'api' (OpenAI-compatible)")
    parser.add_argument("--device", type=str, default="auto", help="Device for local backend (auto, cuda, cpu)")
    parser.add_argument("--dtype", type=str, default=None, choices=["float16", "bfloat16", "float32"],
                        help="Model dtype for local backend")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1",
                        help="API base URL (only used with --backend api)")
    parser.add_argument("--api-key", type=str, default="dummy")
    parser.add_argument("--judge-model", type=str, default=None, help="Separate judge model (defaults to same as --model)")
    parser.add_argument("--judge-base-url", type=str, default=None, help="Judge model API base URL")
    parser.add_argument("--judge-api-key", type=str, default=None, help="Judge model API key")
    parser.add_argument("--facts", type=str, default=None, help="Path to generated_facts.json")
    parser.add_argument("--questions", type=str, default=None, help="Path to generated_questions.json (non-numeric)")
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--max-digits", type=int, default=1000, help="Max digits for digit recital")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    config = BenchmarkConfig(
        model=args.model,
        backend=args.backend,
        base_url=args.base_url,
        api_key=args.api_key,
        device=args.device,
        dtype=args.dtype,
        max_digits=args.max_digits,
        judge_model=args.judge_model,
        judge_base_url=args.judge_base_url,
        judge_api_key=args.judge_api_key,
        seed=args.seed,
    )
    benchmark = MetacognitionBenchmark(config)

    extra_facts = None
    if args.facts:
        extra_facts = load_facts(Path(args.facts))
        print(f"Loaded {len(extra_facts)} additional numeric facts from {args.facts}")

    general_questions = None
    if args.questions:
        from metacognition_benchmark.generators import load_questions
        general_questions = load_questions(Path(args.questions))
        print(f"Loaded {len(general_questions)} general questions from {args.questions}")

    results = benchmark.run_all(extra_facts=extra_facts, general_questions=general_questions)

    if args.output:
        MetacognitionBenchmark.save_results(results, Path(args.output))


if __name__ == "__main__":
    main()
