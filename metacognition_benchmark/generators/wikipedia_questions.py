"""
Generate factual questions from random Wikipedia articles.

Strategy:
1. Fetch random Wikipedia articles (summaries) across one or more languages
2. Use an LLM to extract factual Q&A pairs from the article text
3. Ground truth comes from Wikipedia's text, not from the LLM's own knowledge

This avoids the LLM paradox: the LLM phrases questions, but the ANSWERS come
from Wikipedia article text provided in-context.

Usage:
    python -m metacognition_benchmark.generators.wikidata_questions --count 50 --output generated_questions.json
"""

import argparse
import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests

USER_AGENT = "MetacognitionBenchmark/1.0 (Rolv-Arild Braaten; rolv_arild@hotmail.com)"


# ---------------------------------------------------------------------------
# Wikipedia Article Fetching
# ---------------------------------------------------------------------------


def _load_wikipedia_languages() -> list[tuple[str, int, int]]:
    """Load Wikipedia language codes, article counts, and user counts from data/wikipedias.tsv.

    Returns list of (language_code, article_count, users).
    Excludes bot-generated Wikipedias (depth < 10).
    """
    tsv_path = Path(__file__).parent.parent / "data" / "wikipedias.tsv"
    languages = []
    for line in tsv_path.read_text(encoding="utf-8").splitlines()[1:]:  # skip header
        parts = line.split("\t")
        if len(parts) < 12:
            continue
        code = parts[3].strip()
        try:
            articles = int(parts[4].replace(",", ""))
            users = int(parts[8].replace(",", ""))
            depth = int(parts[11].replace(",", ""))
        except (ValueError, IndexError):
            continue
        # Filter out bot-generated wikis (Cebuano, Waray, etc.) via depth
        if depth < 10:
            continue
        languages.append((code, articles, users))
    return languages


WIKIPEDIA_LANGUAGES = _load_wikipedia_languages()

# Map language code -> total users for normalized difficulty
_USERS = {code: users for code, _, users in WIKIPEDIA_LANGUAGES}

WIKIPEDIA_API_URL = "https://{lang}.wikipedia.org/w/api.php"


@dataclass
class WikipediaArticle:
    """A Wikipedia article with its summary text."""
    title: str
    language: str
    text: str  # Full article text (plaintext)
    url: str
    pageviews: Optional[int] = None


def get_random_articles(count: int = 50, language: str = "en") -> list[WikipediaArticle]:
    """Fetch random Wikipedia articles with their summaries."""
    articles = []
    attempts = 0
    max_attempts = count * 5

    while len(articles) < count and attempts < max_attempts:
        batch_size = min(20, count - len(articles))
        attempts += 1

        try:
            # Get random titles
            resp = requests.get(
                WIKIPEDIA_API_URL.format(lang=language),
                params={
                    "action": "query",
                    "list": "random",
                    "rnnamespace": 0,
                    "rnlimit": batch_size,
                    "format": "json",
                },
                headers={"User-Agent": USER_AGENT},
                timeout=15,
            )
            if resp.status_code == 429:
                time.sleep(5)
                continue
            resp.raise_for_status()
            data = resp.json()
            titles = [item["title"] for item in data.get("query", {}).get("random", [])]

            if not titles:
                continue

            # Fetch extracts and page properties for these titles
            resp2 = requests.get(
                WIKIPEDIA_API_URL.format(lang=language),
                params={
                    "action": "query",
                    "titles": "|".join(titles[:1]),
                    "prop": "extracts|info|pageprops",
                    "explaintext": True,
                    "exsectionformat": "plain",
                    "inprop": "url",
                    "ppprop": "disambiguation|wikibase_item",
                    "format": "json",
                },
                headers={"User-Agent": USER_AGENT},
                timeout=15,
            )
            if resp2.status_code == 429:
                time.sleep(5)
                continue
            resp2.raise_for_status()
            pages = resp2.json().get("query", {}).get("pages", {})

            for page_id, page in pages.items():
                if page_id == "-1":
                    continue

                # Skip disambiguation pages (universal across all Wikipedia languages)
                if "disambiguation" in page.get("pageprops", {}):
                    continue

                extract = page.get("extract", "").strip()
                title = page.get("title", "")

                # Skip very short articles (stubs)
                if len(extract) < 200:
                    continue

                # Skip list-like articles: low prose density (many short lines)
                lines = extract.split("\n")
                non_empty_lines = [l for l in lines if l.strip()]
                if non_empty_lines:
                    avg_line_len = len(extract) / len(non_empty_lines)
                    # List articles have many short lines (names, items);
                    # prose articles have fewer, longer paragraphs
                    if avg_line_len < 40 and len(non_empty_lines) > 10:
                        continue

                articles.append(WikipediaArticle(
                    title=title,
                    language=language,
                    text=extract,
                    url=page.get("fullurl", f"https://{language}.wikipedia.org/wiki/{title.replace(' ', '_')}"),
                ))

                if len(articles) >= count:
                    break

        except Exception:
            time.sleep(2)

    return articles


def get_pageviews(title: str, language: str = "en", max_retries: int = 3) -> Optional[int]:
    """Get all-time pageviews for a Wikipedia article (since July 2015 when the API started)."""
    encoded_title = quote(title.replace(" ", "_"), safe="")
    url = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"{language}.wikipedia/all-access/all-agents/{encoded_title}/monthly/20150701/20250501"
    )
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
                time.sleep(1)
    return None


def pageviews_to_difficulty(pageviews: int, language: str = "en") -> int:
    """Convert pageview count to difficulty level 1-5.

    Normalizes by the wiki's total registered user count, then uses a log scale:
        difficulty = clamp(round(-log10(views/users)) + 1, 1, 5)

    This means each difficulty level corresponds to roughly an order of magnitude
    difference in relative popularity.
    """
    import math
    users = _USERS.get(language, 53_000_000)  # default to English
    if users <= 0:
        users = 1
    views_per_user = pageviews / users
    if views_per_user <= 0:
        return 5
    raw = round(-math.log10(views_per_user))
    return max(1, min(5, raw))


# ---------------------------------------------------------------------------
# LLM-based question generation
# ---------------------------------------------------------------------------

QUESTION_GENERATION_SYSTEM_PROMPT = """\
You are an expert trivia question writer and data extraction engine for a knowledge benchmark. Your objective is to extract verifiable, factual questions strictly from the provided text.

Constraints:
1. You never use outside knowledge or hallucinate facts.
2. You output valid, raw JSON arrays only. 
3. You never include conversational filler, greetings, or markdown code blocks (do not wrap output in ```json)."""

QUESTION_GENERATION_PROMPT = """\
Given the following Wikipedia article, extract 0 to 5 clear, specific factual questions that can be unambiguously answered using only the provided text.

### EXCLUSION CRITERIA
Return an empty array [] if the article meets any of these criteria:
- Niche/Localized: The topic has no broader relevance (e.g., a minor local road, a tiny administrative division, a single episode of an obscure show).
- List-Based: The text is mostly statistics, sports results, or rosters without narrative facts.
- Lacks Concrete Facts: The text lacks verifiable entities, dates, names, places, or numbers.
- Ephemeral: The text relies entirely on time-sensitive facts that change without historical grounding.

### QUESTION GENERATION RULES
- Single Unambiguous Answer: Each question must have exactly ONE correct answer explicitly stated in the text.
- Natural Tone: Phrase questions like a standard pub quiz or trivia benchmark. Mix question types ("What...", "Who...", "Where...", "In what year...", "Which...").
- Self-Contained Context: The question must contain enough context to uniquely identify the subject. Include disambiguating details (e.g., profession, year, country, or type). Use "the 2019 film Parasite", not just "Parasite".
- Pronoun Resolution: Never use pronouns ("he", "it", "this city") or vague references. Explicitly name the subject in every question.
- Temporal Stability: Never ask about relative time ("currently", "last year", "recently"). Convert these to absolute dates based on the text, or do not ask the question. Don't ask questions whose answer may change in the future (e.g., "Who is the current governor?" or "What is the population?").
- English Translation: Questions and answers must ALWAYS be in English, even if the source text is in another language. Translate facts as needed, but retain universally recognized proper nouns.
- Diversity: If generating multiple questions from one article, vary the salience levels and question types. Don't ask five questions about the same aspect.
- Non-Obvious: Do not reveal the answer in the question. Avoid questions where the answer can be trivially guessed from the phrasing alone (e.g., "In the film 3AM, what time are ghosts active?" — the answer is in the title). A good test: could someone who has never read the article narrow it down to fewer than 5 plausible answers just from the question wording?

### ANSWER GENERATION RULES
- Brevity: Answers must be short (a name, a number, a year, or a few words at most).
- Aliases for String Matching: Provide a robust list of acceptable answers (aliases, abbreviations, alternate spellings) so the answer can be programmatically evaluated via string matching. Think about how someone might naturally phrase the answer.
- Years: Include just the number (e.g., "1969").
- People: Include the full name and common variations. Do NOT include the surname alone. Do NOT include initials unless the person is universally known by them (e.g., ["J.R.R. Tolkien", "J. R. R. Tolkien"] is acceptable, but ["Einstein", "A. Einstein"] is not; use ["Albert Einstein"]).
- Places: Include common alternate names and abbreviations (e.g., ["United States", "USA", "US", "United States of America"]).
- Numbers: Include common representations (e.g., ["330", "330 meters", "330 m"]).

### OUTPUT SCHEMA
Respond strictly with a JSON array using the following structure. If the article is unsuitable, output [].

[
  {{
    "question": "string",
    "answers": ["primary answer", "alias1", "alias2"],
    "category": "string (person, place, science, history, art, music, film, literature, sports, politics, technology, nature, other)",
    "salience": integer (1-5),
    "global_relevance": integer (1-5)
  }}
]

### SCORING RUBRICS
salience (How central is this fact to the article's subject?):
  1 = Defining/essential (e.g., "What country is Paris the capital of?")
  2 = Core fact (e.g., "Who painted the Mona Lisa?")
  3 = Notable detail (e.g., "In what year was the Eiffel Tower completed?")
  4 = Specific detail (e.g., "How tall is the Eiffel Tower in meters?")
  5 = Obscure detail (e.g., "Who was the structural engineer of the Eiffel Tower?")

global_relevance (How globally known is this topic?):
  1 = Known worldwide (e.g., Albert Einstein, the Moon, World War II)
  2 = Known across multiple countries/cultures (e.g., Shirley Chisholm, the Danube)
  3 = Known primarily in one large country or region (e.g., a US state capital)
  4 = Known primarily in a specific field or small country (e.g., a niche scientific concept)
  5 = Known only to specialists or locals (e.g., a minor local landmark)

### INPUT DATA
Article title: {title}
Article text:
{text}"""


@dataclass
class GeneratedQuestion:
    """A question generated from a Wikipedia article."""
    question: str
    answers: list[str]  # Acceptable answers for string matching (first is primary)
    category: str
    source_entity: str  # Wikipedia URL
    source_label: str  # Article title
    difficulty: int  # Topic obscurity (from pageviews), 1-5
    salience: int = 3  # How central the fact is to the topic (from LLM), 1-5
    global_relevance: int = 3  # How globally known the topic is (from LLM), 1-5

    @property
    def answer(self) -> str:
        """Primary answer (first in list)."""
        return self.answers[0] if self.answers else ""

    @property
    def combined_difficulty(self) -> float:
        """Combined difficulty: topic obscurity × fact salience.

        Returns a value from 1-5 where higher = harder to know.
        A salient fact (1) about an obscure topic (5) is easier than
        an obscure fact (5) about an obscure topic (5).
        """
        # Geometric mean keeps the 1-5 range
        import math
        return math.sqrt(self.difficulty * self.salience)

    def check_answer(self, response: str) -> bool:
        """Check if any acceptable answer appears in the response (case-insensitive)."""
        response_lower = response.lower()
        return any(a.lower() in response_lower for a in self.answers)


def generate_questions_from_article(
        article: WikipediaArticle,
        chat_fn,
        difficulty: int = 3,
) -> list[GeneratedQuestion]:
    """Use an LLM to generate questions from a Wikipedia article.

    chat_fn: callable(messages, max_tokens, temperature) -> str
    """
    # Truncate article text to avoid exceeding context window
    article_text = article.text
    if len(article_text) > 8000:
        article_text = article_text[:8000] + "\n[...truncated]"

    prompt = QUESTION_GENERATION_PROMPT.format(
        title=article.title,
        text=article_text,
    )

    try:
        text = chat_fn(
            [
                {"role": "system", "content": QUESTION_GENERATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.3,
        )

        # Parse JSON from response (handle markdown code blocks)
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        questions_data = json.loads(text)

        results = []
        for q in questions_data:
            if "question" not in q:
                continue
            # Support both "answers" (list) and "answer" (string) from LLM
            if "answers" in q and isinstance(q["answers"], list):
                answers = [str(a) for a in q["answers"] if a]
            elif "answer" in q:
                answers = [str(q["answer"])]
            else:
                continue
            if not answers:
                continue
            salience = int(q.get("salience", 3))
            salience = max(1, min(5, salience))
            global_relevance = int(q.get("global_relevance", 3))
            global_relevance = max(1, min(5, global_relevance))
            results.append(GeneratedQuestion(
                question=q["question"],
                answers=answers,
                category=q.get("category", "other"),
                source_entity=article.url,
                source_label=article.title,
                difficulty=difficulty,
                salience=salience,
                global_relevance=global_relevance,
            ))
        return results

    except (json.JSONDecodeError, KeyError, Exception):
        return []


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def fetch_articles(
        count: int = 100,
        language: str = "any",
        estimate_difficulty: bool = True,
        db_path: Optional[str] = None,
) -> list[tuple[WikipediaArticle, int]]:
    """Fetch random Wikipedia articles with difficulty estimates.

    Args:
        count: Target number of articles.
        language: Wikipedia language to sample from. "any" samples from multiple languages.
        estimate_difficulty: Whether to estimate difficulty via pageviews.
        db_path: Path to SQLite database for caching articles. If it exists,
            previously fetched articles are loaded and counted toward the target.
            New articles are saved incrementally.
    """
    if language == "any":
        # Sample languages weighted by article count
        codes = [code for code, _, _ in WIKIPEDIA_LANGUAGES]
        weights = [articles for _, articles, _ in WIKIPEDIA_LANGUAGES]
    else:
        codes = [language]
        weights = [1]

    print(f"Fetching Wikipedia articles (target: {count}, language={'weighted mix' if language == 'any' else language})...")

    # Initialize database if path given
    conn = None
    if db_path:
        from metacognition_benchmark.generators.db import init_db, load_articles, upsert_article
        conn = init_db(db_path)

    # Load cached articles from database
    cached_articles: list[tuple[WikipediaArticle, int]] = []
    if conn:
        for item in load_articles(conn):
            article = WikipediaArticle(
                title=item["title"], language=item["language"],
                text=item["text"], url=item["url"], pageviews=item.get("pageviews"),
            )
            cached_articles.append((article, item["difficulty"]))
        if cached_articles:
            print(f"  Loaded {len(cached_articles)} cached articles from database")

    results = []

    # Pre-fill from cache
    for article, difficulty in cached_articles:
        if len(results) >= count:
            break
        results.append((article, difficulty))

    if len(results) >= count:
        print(f"  Already have {len(results)} articles from cache")
        return results[:count]

    attempts = 0
    max_attempts = count * 5

    while len(results) < count and attempts < max_attempts:
        attempts += 1
        # Pick a random language weighted by size
        lang = random.choices(codes, weights=weights, k=1)[0]

        # Fetch a batch of articles
        batch_size = 1
        batch = get_random_articles(count=batch_size, language=lang)
        if not batch:
            continue

        for article in batch:
            if len(results) >= count:
                break

            if estimate_difficulty:
                article.pageviews = get_pageviews(article.title, language=article.language)
                if article.pageviews is None:
                    continue
                difficulty = pageviews_to_difficulty(article.pageviews, language=article.language)
            else:
                difficulty = 3

            results.append((article, difficulty))
            print(f"  [{len(results)}/{count}] {article.title} "
                  f"(d={difficulty}, views={article.pageviews}, {article.language})")

            # Save to database
            if conn:
                upsert_article(conn, article.title, article.language, article.text,
                               article.url, article.pageviews, difficulty)

    if conn:
        conn.close()

    print(f"  Selected {len(results)} articles")

    return results


def generate_questions_batch(
        articles: list[tuple[WikipediaArticle, int]],
        llm_base_url: str = "http://localhost:8000/v1",
        llm_model: str = "google/gemma-4-E2B-it",
        llm_api_key: str = "dummy",
        local_model=None,
        db_path: Optional[str] = None,
) -> list[GeneratedQuestion]:
    """Generate questions for a batch of articles using an LLM."""
    # Initialize database for saving questions
    conn = None
    if db_path:
        from metacognition_benchmark.generators.db import init_db, insert_question
        conn = init_db(db_path)

    if local_model is not None:
        def chat_fn(messages, max_tokens=500, temperature=0.3):
            return local_model.generate(messages, max_tokens=max_tokens, temperature=temperature)
    else:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for API-based question generation. "
                "Install with: pip install metacognition-benchmark[api]"
            )
        client = OpenAI(base_url=llm_base_url, api_key=llm_api_key)

        def chat_fn(messages, max_tokens=500, temperature=0.3):
            response = client.chat.completions.create(
                model=llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()

    all_questions = []

    print(f"Generating questions from {len(articles)} articles...")
    for i, (article, difficulty) in enumerate(articles):
        questions = generate_questions_from_article(article, chat_fn, difficulty=difficulty)
        all_questions.extend(questions)
        if questions:
            print(f"  [{i + 1}/{len(articles)}] {article.title} → {len(questions)} questions")
            if conn:
                for q in questions:
                    insert_question(conn, q.question, q.answers, q.category,
                                    q.source_entity, q.source_label, q.difficulty,
                                    q.salience, q.global_relevance)
        else:
            print(f"  [{i + 1}/{len(articles)}] {article.title} → (skipped)")
        time.sleep(0.2)

    if conn:
        conn.close()

    print(f"\nTotal: {len(all_questions)} questions generated")
    return all_questions


def save_questions(questions: list[GeneratedQuestion], path: Path):
    """Save generated questions to JSON."""
    data = [asdict(q) for q in questions]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Saved {len(questions)} questions to {path}")


def load_questions(path: Path) -> list[GeneratedQuestion]:
    """Load generated questions from JSON."""
    data = json.loads(path.read_text())
    questions = []
    for d in data:
        # Handle both old format (answer: str) and new format (answers: list)
        if "answers" not in d and "answer" in d:
            d["answers"] = [d.pop("answer")]
        questions.append(GeneratedQuestion(**d))
    return questions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate factual questions from Wikipedia articles")
    parser.add_argument("--count", type=int, default=100, help="Number of articles to fetch")
    parser.add_argument("--language", type=str, default="any",
                        help="Wikipedia language: 'any' for multilingual, or a code like 'en', 'de', 'fr'")
    parser.add_argument("--no-difficulty", action="store_true", help="Skip pageview difficulty estimation")
    parser.add_argument("--llm-url", type=str, default="http://localhost:8000/v1",
                        help="LLM API base URL (ignored if using local model)")
    parser.add_argument("--llm-model", type=str, default="google/gemma-4-E4B-it",
                        help="Model name (HuggingFace ID or API model)")
    parser.add_argument("--llm-api-key", type=str, default="dummy")
    parser.add_argument("--api", action="store_true",
                        help="Use OpenAI-compatible API instead of local model")
    parser.add_argument("--quantization", type=str, default=None, choices=["4bit", "8bit"],
                        help="Quantize local model (requires bitsandbytes)")
    parser.add_argument("--output", type=str, default="generated_questions.json")
    parser.add_argument("--articles-only", action="store_true",
                        help="Only fetch articles, skip LLM question generation")
    parser.add_argument("--db", type=str, default="wikipedia_benchmark.db",
                        help="SQLite database for caching articles and questions")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Step 1: Fetch articles
    articles = fetch_articles(
        count=args.count,
        language=args.language,
        estimate_difficulty=not args.no_difficulty,
        db_path=args.db,
    )

    if args.articles_only:
        out = Path(args.output).with_suffix(".articles.json")
        data = [{"title": a.title, "language": a.language, "text": a.text,
                 "url": a.url, "pageviews": a.pageviews, "difficulty": d}
                for a, d in articles]
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"Saved {len(articles)} articles to {out}")
    else:
        # Step 2: Generate questions
        local = None
        if not args.api:
            from metacognition_benchmark.inference import LocalModel

            print(f"Loading local model: {args.llm_model}")
            local = LocalModel(args.llm_model, quantization=args.quantization)

        questions = generate_questions_batch(
            articles,
            llm_base_url=args.llm_url,
            llm_model=args.llm_model,
            llm_api_key=args.llm_api_key,
            local_model=local,
            db_path=args.db,
        )

        # Save
        save_questions(questions, Path(args.output))
