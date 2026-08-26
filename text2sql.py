"""FLINT — a minimal, standalone text-to-SQL agent.

Four steps: retrieve similar examples -> link schema (implicit in the prompt) ->
generate SQL -> optionally execute and reflect on error. Every step is standard,
published technique; nothing here depends on any proprietary system.

The LLM is called through litellm, so any provider works. Set an API key in the
environment and pick a model:

    export ANTHROPIC_API_KEY=sk-...           # or OPENAI_API_KEY, etc.
    export FLINT_MODEL=anthropic/claude-3-5-sonnet-latest

Quick demo (no database needed, just an API key):

    python text2sql.py --demo

Generate SQL for your own schema + question:

    python text2sql.py \
        --schema "CREATE TABLE film (title text, length int);" \
        --question "Which films are longer than two hours?"

Run in domain-aware (FLINT) mode against one of the benchmark datasets — the
runner automatically pulls that dataset's domain rules from the `datasets/`
package (see `datasets/__init__.py`):

    python text2sql.py --mode flint --dataset cordis \
        --schema "$(cat cordis_schema.sql)" \
        --examples cordis_examples.json \
        --question "How many projects did each institution coordinate?"

See README.md for the full description.
"""

import argparse
import asyncio
import json
import os

import litellm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from datasets import available, get_dataset

litellm.drop_params = True
litellm.num_retries = 3

# Any litellm-supported model string works (anthropic/..., openai/..., etc.).
# Override with the FLINT_MODEL env var. Keys are read from the environment by
# litellm (ANTHROPIC_API_KEY, OPENAI_API_KEY, ...).
# The paper's reported results use Claude Opus 4.6, pinned to the dated snapshot
# below for reproducibility. Set FLINT_MODEL to the exact model id your
# provider/proxy exposes if it differs.
DEFAULT_MODEL = os.getenv("FLINT_MODEL", "anthropic/claude-opus-4-6-20260205")


# --------------------------------------------------------------------------- #
# LLM call (provider-agnostic; no internal gateway)                           #
# --------------------------------------------------------------------------- #
async def call_llm(prompt, model=DEFAULT_MODEL, temperature=0.0):
    """Send a single-turn prompt to the LLM and return its text.

    Honors an optional OpenAI-compatible proxy via LLM_BASE_URL; otherwise
    litellm talks to the provider directly using the API key in the env.
    """
    kwargs = {}
    base_url = os.getenv("LLM_BASE_URL")
    if base_url:
        kwargs["api_base"] = base_url
    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        **kwargs,
    )
    return response.choices[0].message.content


# --------------------------------------------------------------------------- #
# Example retrieval (TF-IDF + cosine similarity)                              #
# --------------------------------------------------------------------------- #
class SimpleRetriever:
    """Retrieve the top-k most similar example questions for few-shot prompting.

    `examples` is a list of {"question": str, "sql": str} dicts.
    """

    def __init__(self, examples):
        self.examples = examples
        self.questions = [e["question"] for e in examples]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.vectors = self.vectorizer.fit_transform(self.questions)

    def retrieve(self, question, k=3):
        q_vec = self.vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self.vectors)[0]
        top_k = np.argsort(sims)[-k:][::-1]
        return [self.examples[i] for i in top_k]


# --------------------------------------------------------------------------- #
# Prompts                                                                     #
# --------------------------------------------------------------------------- #
ZERO_SHOT_PROMPT = """You are a SQL expert. Given the database schema below, write a SQL query to answer the question.

Database Schema:
{schema}

Question: {question}

Write ONLY the SQL query, no explanations.
"""

FEW_SHOT_PROMPT = """You are a SQL expert. Given the database schema below and example question-SQL pairs, write a SQL query to answer the question.

Database Schema:
{schema}

{few_shot_examples}

Question: {question}

Write ONLY the SQL query, no explanations.
"""

FLINT_PROMPT = """You are a SQL expert for a {domain} database. Given the schema and example queries, write SQL to answer the question.

Database Schema:
{schema}

{few_shot_examples}

Domain-Specific Rules:
{domain_rules}

Question: {question}

Write ONLY the SQL query, no explanations.
"""

REFLECTION_PROMPT = """The following SQL query produced an error or unexpected empty results.

Original question: {question}
Generated SQL: {sql}
Problem: {error}

Database Schema:
{schema}

Common issues to check:
- Wrong join path (check foreign key relationships)
- Case-sensitive string values
- Wrong column selected from the wrong table
- Selecting from a joined table when the answer is in a single table

Fix the SQL query. Write ONLY the corrected SQL, no explanations.
"""


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def format_few_shot(examples, k=3):
    if not examples:
        return ""
    lines = ["Here are similar example questions and their SQL queries:"]
    for i, ex in enumerate(examples[:k], 1):
        lines.append(f"\nExample {i}:")
        lines.append(f"Question: {ex['question']}")
        lines.append(f"SQL: {ex['sql']}")
    return "\n".join(lines)


def strip_markdown_fences(text):
    if not text:
        return text
    text = text.strip()
    if text.startswith("```sql"):
        text = text[6:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def build_prompt(question, schema, mode="few-shot", few_shot_text="", domain="", domain_rules=""):
    """Assemble the prompt for the requested mode."""
    if mode == "zero-shot":
        return ZERO_SHOT_PROMPT.format(schema=schema, question=question)
    if mode == "few-shot":
        return FEW_SHOT_PROMPT.format(
            schema=schema, question=question, few_shot_examples=few_shot_text
        )
    if mode == "flint":
        return FLINT_PROMPT.format(
            schema=schema,
            question=question,
            few_shot_examples=few_shot_text,
            domain=domain,
            domain_rules=domain_rules,
        )
    raise ValueError(f"unknown mode: {mode}")


# --------------------------------------------------------------------------- #
# Pipeline                                                                    #
# --------------------------------------------------------------------------- #
async def generate_sql(
    question,
    schema,
    model=DEFAULT_MODEL,
    mode="few-shot",
    retriever=None,
    k=3,
    domain="",
    domain_rules="",
):
    """Retrieve (if a retriever is given) and generate SQL for a question."""
    few_shot_text = ""
    if retriever is not None and mode in ("few-shot", "flint"):
        few_shot_text = format_few_shot(retriever.retrieve(question, k=k), k=k)
    prompt = build_prompt(
        question, schema, mode=mode, few_shot_text=few_shot_text,
        domain=domain, domain_rules=domain_rules,
    )
    return strip_markdown_fences(await call_llm(prompt, model=model))


async def reflect_on_error(question, sql, error, schema, model=DEFAULT_MODEL):
    """Ask the model to fix a query that errored or returned nothing."""
    prompt = REFLECTION_PROMPT.format(question=question, sql=sql, error=error, schema=schema)
    return strip_markdown_fences(await call_llm(prompt, model=model))


async def solve(
    question,
    schema,
    model=DEFAULT_MODEL,
    mode="few-shot",
    retriever=None,
    k=3,
    domain="",
    domain_rules="",
    execute=None,
):
    """Full loop: generate -> (optionally) execute -> reflect once on failure.

    `execute`, if provided, is a callable `sql -> (rows, error)` where `error`
    is None on success. It stays out of this file so the agent has no database
    dependency; wire in psycopg2/sqlite/whatever you use. When omitted, this
    just returns the first generated query.
    """
    sql = await generate_sql(
        question, schema, model=model, mode=mode, retriever=retriever, k=k,
        domain=domain, domain_rules=domain_rules,
    )
    if execute is None:
        return sql

    rows, error = execute(sql)
    if error or not rows:
        problem = error or "query returned no rows"
        sql = await reflect_on_error(question, sql, problem, schema, model=model)
    return sql


# --------------------------------------------------------------------------- #
# Demo                                                                        #
# --------------------------------------------------------------------------- #
DEMO_SCHEMA = """CREATE TABLE film (
    film_id     integer PRIMARY KEY,
    title       text,
    length      integer,          -- minutes
    rating      text,             -- 'G', 'PG', 'PG-13', 'R'
    rental_rate numeric
);
CREATE TABLE category (category_id integer PRIMARY KEY, name text);
CREATE TABLE film_category (film_id integer, category_id integer);"""

DEMO_EXAMPLES = [
    {"question": "How many films are rated R?",
     "sql": "SELECT COUNT(*) FROM film WHERE rating = 'R';"},
    {"question": "List the titles of comedy films.",
     "sql": "SELECT f.title FROM film f JOIN film_category fc ON f.film_id = fc.film_id "
            "JOIN category c ON fc.category_id = c.category_id WHERE c.name = 'Comedy';"},
    {"question": "What is the average rental rate?",
     "sql": "SELECT AVG(rental_rate) FROM film;"},
]


async def _demo():
    retriever = SimpleRetriever(DEMO_EXAMPLES)
    question = "Which films are longer than two hours?"
    print(f"Model:    {DEFAULT_MODEL}")
    print(f"Question: {question}\n")
    sql = await generate_sql(question, DEMO_SCHEMA, mode="few-shot", retriever=retriever)
    print("Generated SQL:\n" + sql)


def _load_examples(path):
    """Load a query bank: JSON list of question/SQL pairs.

    Accepts both this repo's {"question", "sql"} shape and ScienceBenchmark's
    native seed.json {"question", "query"} shape, so you can point --examples
    straight at a downloaded seed.json.
    """
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"{path} must be a JSON list of objects")
    examples = []
    for e in data:
        sql = e.get("sql", e.get("query"))
        if "question" not in e or sql is None:
            raise SystemExit(f"{path} entries need 'question' and 'sql' (or 'query')")
        examples.append({"question": e["question"], "sql": sql})
    return examples


def _resolve_examples(args, ds):
    """Pick the query bank: explicit --examples wins, else the dataset's bundled bank."""
    if args.examples:
        return _load_examples(args.examples)
    if ds is not None:
        bundled = ds.bundled_examples_path()
        if bundled and os.path.exists(bundled):
            return _load_examples(bundled)
        raise SystemExit(
            f"dataset '{ds.name}' has no bundled query bank. Provide --examples.\n"
            f"{ds.examples_source}"
        )
    raise SystemExit(f"--mode {args.mode} requires --examples (a JSON query bank), or --dataset")


def main():
    parser = argparse.ArgumentParser(description="Minimal standalone text-to-SQL agent.")
    parser.add_argument("--demo", action="store_true", help="run the built-in demo")
    parser.add_argument("--schema", help="CREATE TABLE ... schema text")
    parser.add_argument("--question", help="natural-language question")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="litellm model string")
    parser.add_argument("--mode", default="zero-shot", choices=["zero-shot", "few-shot", "flint"])
    parser.add_argument(
        "--dataset", choices=available(),
        help="domain config to load its prompts + bundled query bank (required for --mode flint)",
    )
    parser.add_argument(
        "--examples", help="path to a JSON query bank; overrides the dataset's bundled bank",
    )
    parser.add_argument("-k", type=int, default=3, help="number of examples to retrieve")
    args = parser.parse_args()

    if args.demo:
        asyncio.run(_demo())
        return
    if not (args.schema and args.question):
        parser.error("provide --schema and --question, or use --demo")

    # Toggle the domain: flint mode pulls the selected dataset's rules automatically.
    ds = None
    domain, domain_rules = "", ""
    if args.dataset:
        ds = get_dataset(args.dataset)
    if args.mode == "flint":
        if ds is None:
            parser.error("--mode flint requires --dataset (choices: %s)" % ", ".join(available()))
        domain, domain_rules = ds.domain, ds.rules

    retriever = None
    if args.mode in ("few-shot", "flint"):
        retriever = SimpleRetriever(_resolve_examples(args, ds))

    sql = asyncio.run(
        generate_sql(
            args.question, args.schema, model=args.model, mode=args.mode,
            retriever=retriever, k=args.k, domain=domain, domain_rules=domain_rules,
        )
    )
    print(sql)


if __name__ == "__main__":
    main()
