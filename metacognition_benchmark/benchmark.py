"""Core benchmark orchestrator: runs tasks and aggregates scores."""

import json
from pathlib import Path
from typing import Optional

from metacognition_benchmark.constants import MATH_CONSTANTS, MathConstant
from metacognition_benchmark.facts import NUMERICAL_FACTS, NumericalFact
from metacognition_benchmark.models import (
    BenchmarkConfig,
    DigitRecitalResult,
    GeneralQuestionResult,
    NumericalFactResult,
)
from metacognition_benchmark.scoring import (
    analyze_digit_logprobs,
    build_digit_recital_completion,
    build_digit_recital_prompt,
    parse_fact_response,
    parse_general_response,
)


def _get_openai_client(base_url: str, api_key: str):
    """Lazily import and construct an OpenAI client (requires optional 'api' extra)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "The 'openai' package is required for API backend or API-based judge. "
            "Install it with: pip install metacognition-benchmark[api]"
        )
    return OpenAI(base_url=base_url, api_key=api_key)


class MetacognitionBenchmark:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._local_model = None
        self._api_client = None

        if config.backend == "local":
            from metacognition_benchmark.inference import LocalModel
            import torch

            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(config.dtype) if config.dtype else None
            self._local_model = LocalModel(
                config.model, device=config.device, dtype=dtype
            )
        elif config.backend == "api":
            self._api_client = _get_openai_client(config.base_url, config.api_key)
        else:
            raise ValueError(f"Unknown backend: {config.backend!r}. Use 'local' or 'api'.")

        # Set up judge (can be API-based even with local evaluated model)
        self._judge_client = None
        self.judge_model = config.judge_model or config.model
        if config.judge_base_url:
            self._judge_client = _get_openai_client(
                config.judge_base_url, config.judge_api_key or config.api_key
            )
        elif config.backend == "api":
            self._judge_client = self._api_client

    # ------------------------------------------------------------------
    # Chat generation (works with both backends)
    # ------------------------------------------------------------------

    def _chat(self, messages: list[dict], max_tokens: int = 150) -> str:
        """Generate a chat response using the configured backend."""
        if self._local_model is not None:
            return self._local_model.generate(
                messages, max_tokens=max_tokens, temperature=self.config.temperature
            )
        else:
            response = self._api_client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Digit Recital
    # ------------------------------------------------------------------

    def run_digit_recital(self, constant: MathConstant) -> DigitRecitalResult:
        """Run digit recital via logprob analysis.

        With local backend: full forward pass with complete logprob access.
        With API backend: requires vLLM-style completions endpoint with echo=True.
        """
        ground_truth = build_digit_recital_completion(constant, self.config.max_digits)
        prompt = build_digit_recital_prompt(constant, self.config.max_digits)

        if self._local_model is not None:
            # Local model: apply chat template and do a forward pass
            messages = [{"role": "user", "content": prompt}]
            chat_prompt = self._local_model.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            token_logprobs = self._local_model.get_logprobs_for_completion(
                chat_prompt, ground_truth
            )
            result = analyze_digit_logprobs(token_logprobs, ground_truth)
        else:
            # API backend: legacy completions with echo (vLLM-style)
            response = self._api_client.completions.create(
                model=self.config.model,
                prompt=(
                    f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                    f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    f"{ground_truth}"
                ),
                max_tokens=0,
                echo=True,
                logprobs=5,
                temperature=self.config.temperature,
            )
            logprobs_data = response.choices[0].logprobs
            token_logprobs_list = []
            if logprobs_data:
                for i, token in enumerate(logprobs_data.tokens):
                    top = {}
                    if logprobs_data.top_logprobs and i < len(logprobs_data.top_logprobs):
                        top = logprobs_data.top_logprobs[i] or {}
                    token_logprobs_list.append({
                        "token": token,
                        "logprob": logprobs_data.token_logprobs[i] or 0.0,
                        "top_logprobs": top,
                    })
            result = analyze_digit_logprobs(token_logprobs_list, ground_truth)

        result.constant = constant
        return result

    # ------------------------------------------------------------------
    # Numerical Facts
    # ------------------------------------------------------------------

    def run_numerical_fact(self, fact: NumericalFact) -> NumericalFactResult:
        """Run a numerical fact question in refusal-allowed and forced modes."""
        # Mode 1: Allow refusal
        refusal_prompt = (
            f"{fact.question}\n\n"
            f"Answer with the numeric value in {fact.unit}. "
            f"If you are not confident in your answer, respond with exactly: IDK\n"
            f"If you do answer, also state your confidence (0-100%) on a separate line.\n"
            f"Format:\nANSWER: <number>\nCONFIDENCE: <percent>"
        )
        text1 = self._chat([{"role": "user", "content": refusal_prompt}], max_tokens=150)
        model_answer, model_refused, model_confidence = parse_fact_response(text1)

        # Mode 2: Force answer
        force_prompt = (
            f"{fact.question}\n\n"
            f"You MUST provide a numeric answer in {fact.unit}. Give your best estimate.\n"
            f"Format: ANSWER: <number>"
        )
        text2 = self._chat([{"role": "user", "content": force_prompt}], max_tokens=100)
        forced_answer, _, _ = parse_fact_response(text2)

        return NumericalFactResult(
            fact=fact,
            model_answer=model_answer,
            model_refused=model_refused,
            model_confidence=model_confidence,
            forced_answer=forced_answer,
        )

    # ------------------------------------------------------------------
    # General Questions
    # ------------------------------------------------------------------

    def run_general_question(self, question: str, expected_answer: str, difficulty: int = 3, category: str = "", answer_checker=None) -> GeneralQuestionResult:
        """Run a general factual question with string answer.

        If answer_checker is provided (callable(str) -> bool), uses it for correctness.
        Otherwise falls back to LLM-as-judge.
        """
        # Mode 1: Allow refusal
        refusal_prompt = (
            f"{question}\n\n"
            f"Answer concisely (a few words). "
            f"If you are not confident, respond with exactly: IDK\n"
            f"If you do answer, also state your confidence (0-100%).\n"
            f"Format:\nANSWER: <your answer>\nCONFIDENCE: <percent>"
        )
        text1 = self._chat([{"role": "user", "content": refusal_prompt}], max_tokens=100)
        model_answer, model_refused, model_confidence = parse_general_response(text1)

        # Mode 2: Force answer
        force_prompt = (
            f"{question}\n\n"
            f"You MUST answer, even if you're unsure. Be concise (a few words).\n"
            f"Format: ANSWER: <your answer>"
        )
        text2 = self._chat([{"role": "user", "content": force_prompt}], max_tokens=100)
        forced_answer, _, _ = parse_general_response(text2)

        # Judge correctness
        if answer_checker:
            is_correct = answer_checker(model_answer) if model_answer else False
            forced_is_correct = answer_checker(forced_answer) if forced_answer else False
        else:
            is_correct = self._judge_answer(question, expected_answer, model_answer) if model_answer else False
            forced_is_correct = self._judge_answer(question, expected_answer, forced_answer) if forced_answer else False

        return GeneralQuestionResult(
            question=question,
            expected_answer=expected_answer,
            model_answer=model_answer,
            model_refused=model_refused,
            model_confidence=model_confidence,
            forced_answer=forced_answer,
            difficulty=difficulty,
            category=category,
            is_correct=is_correct,
            forced_is_correct=forced_is_correct,
        )

    # ------------------------------------------------------------------
    # Judge
    # ------------------------------------------------------------------

    def _judge_answer(self, question: str, expected: str, given: str) -> bool:
        """Use LLM-as-judge to determine if the given answer is correct."""
        if not given:
            return False
        if given.strip().lower() == expected.strip().lower():
            return True

        judge_prompt = (
            f"Question: {question}\n"
            f"Correct answer: {expected}\n"
            f"Given answer: {given}\n\n"
            f"Is the given answer correct (or an acceptable equivalent/alias)? "
            f"Respond with only YES or NO."
        )
        try:
            if self._judge_client is not None:
                # API-based judge
                response = self._judge_client.chat.completions.create(
                    model=self.judge_model,
                    messages=[{"role": "user", "content": judge_prompt}],
                    temperature=0.0,
                    max_tokens=5,
                )
                verdict = response.choices[0].message.content.strip().upper()
            elif self._local_model is not None:
                # Use local model as judge
                verdict = self._local_model.generate(
                    [{"role": "user", "content": judge_prompt}],
                    max_tokens=5, temperature=0.0,
                ).strip().upper()
            else:
                return expected.lower() in given.lower()
            return verdict.startswith("YES")
        except Exception:
            return expected.lower() in given.lower()

    # ------------------------------------------------------------------
    # Run All
    # ------------------------------------------------------------------

    def run_all(self, extra_facts: list[NumericalFact] | None = None, general_questions: list | None = None) -> dict:
        """Run the full benchmark suite."""
        print(f"Running Metacognition Benchmark: {self.config.model}")
        if self.judge_model != self.config.model:
            print(f"  Judge model: {self.judge_model}")
        print("=" * 70)

        failures: list[dict] = []

        # Digit Recital
        print("\n📐 DIGIT RECITAL")
        print("-" * 40)
        digit_results = []
        for constant in MATH_CONSTANTS:
            try:
                result = self.run_digit_recital(constant)
                digit_results.append(result)
                det_str = f"{result.deterministic_score:.2f}" if result.deterministic_score is not None else "N/A"
                sto_str = f"{result.stochastic_score:.3f}" if result.stochastic_score is not None else "N/A"
                err_pos = result.first_error_position
                eos_pos = result.eos_position
                print(
                    f"  {constant.name:<8} (diff={constant.difficulty}) | "
                    f"err@{err_pos} eos@{eos_pos} | "
                    f"det={det_str} sto={sto_str}"
                )
            except Exception as e:
                print(f"  {constant.name:<8} ERROR: {e}")
                failures.append({"task": "digit_recital", "item": constant.name, "error": str(e)})

        # Numerical Facts
        all_facts = list(NUMERICAL_FACTS)
        if extra_facts:
            all_facts.extend(extra_facts)

        print(f"\n📊 NUMERICAL FACTS ({len(all_facts)} questions)")
        print("-" * 40)
        fact_results = []
        for fact in all_facts:
            try:
                result = self.run_numerical_fact(fact)
                fact_results.append(result)
                status = "✓" if result.is_correct else ("⊘" if result.model_refused else "✗")
                diff_str = f"diff={fact.difficulty}" if fact.difficulty is not None else "diff=?"
                print(
                    f"  [{status}] ({diff_str}) {fact.question[:60]}...\n"
                    f"       answer={result.model_answer} (actual={fact.answer}) "
                    f"meta={result.metacognition_score:.2f}"
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                failures.append({"task": "numerical_fact", "item": fact.question[:60], "error": str(e)})

        # General Questions
        general_results = []
        if general_questions:
            print(f"\n📝 GENERAL KNOWLEDGE ({len(general_questions)} questions)")
            print("-" * 40)
            for gq in general_questions:
                try:
                    # Support both GeneratedQuestion objects and dicts
                    question = gq.question if hasattr(gq, 'question') else gq["question"]
                    expected = gq.answer if hasattr(gq, 'answer') else gq["answer"]
                    difficulty = gq.difficulty if hasattr(gq, 'difficulty') else gq.get("difficulty", 3)
                    category = gq.category if hasattr(gq, 'category') else gq.get("category", "")
                    # Use check_answer for string matching if available
                    checker = gq.check_answer if hasattr(gq, 'check_answer') else None

                    result = self.run_general_question(
                        question=question,
                        expected_answer=expected,
                        difficulty=difficulty,
                        category=category,
                        answer_checker=checker,
                    )
                    general_results.append(result)
                    status = "✓" if result.is_correct else ("⊘" if result.model_refused else "✗")
                    print(
                        f"  [{status}] (diff={result.difficulty}) {result.question[:55]}...\n"
                        f"       got='{result.model_answer}' (expected='{result.expected_answer}') "
                        f"meta={result.metacognition_score:.2f}"
                    )
                except Exception as e:
                    print(f"  ERROR: {e}")
                    q_text = gq.question if hasattr(gq, 'question') else gq.get("question", "?")
                    failures.append({"task": "general_question", "item": q_text[:60], "error": str(e)})

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")

        digit_scores = [r.calibration_score for r in digit_results if r.calibration_score is not None]
        fact_scores = [r.metacognition_score for r in fact_results]
        general_scores = [r.metacognition_score for r in general_results]

        if digit_scores:
            print(f"  Digit Recital avg calibration: {sum(digit_scores)/len(digit_scores):.3f}")
        if fact_scores:
            print(f"  Numerical Facts avg metacognition: {sum(fact_scores)/len(fact_scores):.3f}")
        if general_scores:
            print(f"  General Knowledge avg metacognition: {sum(general_scores)/len(general_scores):.3f}")

        all_meta_scores = fact_scores + general_scores
        for diff in range(1, 6):
            diff_fact = [r for r in fact_results if r.fact.difficulty == diff]
            diff_gen = [r for r in general_results if r.difficulty == diff]
            total = len(diff_fact) + len(diff_gen)
            if total:
                scores = [r.metacognition_score for r in diff_fact] + [r.metacognition_score for r in diff_gen]
                avg = sum(scores) / len(scores)
                correct = sum(1 for r in diff_fact if r.is_correct) + sum(1 for r in diff_gen if r.is_correct)
                refused = sum(1 for r in diff_fact if r.model_refused) + sum(1 for r in diff_gen if r.model_refused)
                print(f"    Difficulty {diff}: meta={avg:.2f} correct={correct}/{total} refused={refused}")

        if failures:
            print(f"\n  ⚠ FAILURES: {len(failures)} tasks failed")
            by_task = {}
            for f in failures:
                t = f["task"]
                by_task[t] = by_task.get(t, 0) + 1
            for t, count in sorted(by_task.items()):
                print(f"    {t}: {count} failures")

        component_scores = []
        if digit_scores:
            component_scores.append(sum(digit_scores) / len(digit_scores))
        if all_meta_scores:
            component_scores.append(sum(all_meta_scores) / len(all_meta_scores))

        overall = sum(component_scores) / len(component_scores) if component_scores else 0
        print(f"\n  OVERALL METACOGNITION SCORE: {overall:.3f} (range: -1 to +1)")

        return {
            "model": self.config.model,
            "judge_model": self.judge_model,
            "digit_results": digit_results,
            "fact_results": fact_results,
            "general_results": general_results,
            "failures": failures,
            "overall_score": overall,
        }


    @staticmethod
    def save_results(results: dict, path: Path) -> None:
        """Save benchmark results to a standardized JSON file for cross-model comparison."""
        serializable = {
            "model": results["model"],
            "judge_model": results.get("judge_model", results["model"]),
            "overall_score": results["overall_score"],
            "failures": results.get("failures", []),
            "digit_recital": [
                {
                    "constant": r.constant.name if r.constant else None,
                    "difficulty": r.constant.difficulty if r.constant else None,
                    "n_digits": r.n_digits,
                    "first_error_position": r.first_error_position,
                    "eos_position": r.eos_position,
                    "deterministic_score": r.deterministic_score,
                    "stochastic_score": r.stochastic_score,
                }
                for r in results["digit_results"]
            ],
            "numerical_facts": [
                {
                    "question": r.fact.question,
                    "expected_answer": r.fact.answer,
                    "model_answer": r.model_answer,
                    "forced_answer": r.forced_answer,
                    "model_refused": r.model_refused,
                    "model_confidence": r.model_confidence,
                    "is_correct": r.is_correct,
                    "forced_is_correct": r.forced_is_correct,
                    "metacognition_score": r.metacognition_score,
                    "difficulty": r.fact.difficulty,
                    "category": r.fact.category,
                }
                for r in results["fact_results"]
            ],
            "general_questions": [
                {
                    "question": r.question,
                    "expected_answer": r.expected_answer,
                    "model_answer": r.model_answer,
                    "forced_answer": r.forced_answer,
                    "model_refused": r.model_refused,
                    "model_confidence": r.model_confidence,
                    "is_correct": r.is_correct,
                    "forced_is_correct": r.forced_is_correct,
                    "metacognition_score": r.metacognition_score,
                    "difficulty": r.difficulty,
                    "category": r.category,
                }
                for r in results["general_results"]
            ],
        }
        path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False))
        print(f"Results saved to {path}")
