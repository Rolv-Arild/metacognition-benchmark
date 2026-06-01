"""Question and fact generators from external data sources."""

from metacognition_benchmark.generators.fact_generator import (
    generate_all_facts,
    generate_facts_from_codata,
    generate_facts_from_oeis,
    generate_facts_from_wikidata,
    load_facts,
    save_facts,
    parse_codata_file,
)
from metacognition_benchmark.generators.wikipedia_questions import (
    GeneratedQuestion,
    fetch_articles,
    generate_questions_batch,
    load_questions,
    save_questions,
)

