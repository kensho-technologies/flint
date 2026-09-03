# FLINT — a staged text-to-SQL agent

A self-contained reference implementation of the text-to-SQL agent behind
the paper *From Benchmarks to Production: A Text-to-SQL System for Complex
Financial Data*. It runs on **public data only**.

The agent is a [LangGraph](https://github.com/langchain-ai/langgraph)
`StateGraph` with six stages:

1. **Retrieve** similar examples — TF-IDF + cosine similarity over a small query bank.
2. **Lookup** values — resolve the concepts in the question (a category code, an exact label) to the values actually stored in the database, by probing `SELECT DISTINCT` columns through it. On by default; no-ops without a database.
3. **Link schema** — reason over the schema to a flat list of the columns, joins, and filters the answer needs. Opt-in (`--schema-link`): helps FK-dense schemas, can over-constrain elsewhere.
4. **Generate** SQL — one LLM call with the schema + retrieved examples (+ optional domain rules), with the earlier stages folded in as additive hints.
5. **Execute** the SQL (only when a database is provided).
6. **Reflect** once — on an execution error, or when the query returns empty while the lookup stage verified an exact stored value the query's filter didn't use. Otherwise the result stands.

## Layout

One runner, one prompt module per dataset:

```
text2sql.py          # the LangGraph staged agent: LLM call, retrieval, stages, CLI
datasets/
    base.py          # the Dataset config type
    __init__.py      # registry — maps a name to its config
    cordis.py        # EU research funding (CORDIS) domain rules
    oncomx.py        # cancer biomarker (OncoMX) domain rules
    sdss.py          # astronomy (SDSS) domain rules
    sm3.py           # medical / Synthea (SM3) domain rules
    data/
        sm3_query_bank.json   # SM3 few-shot bank (510 examples)
```

The runner toggles on `--dataset` and automatically pulls that dataset's domain
rules **and** its few-shot query bank from `datasets/`. To add a dataset, drop a
module in `datasets/` that defines `DATASET = Dataset(...)` and register it in
`datasets/__init__.py` — no change to the runner.

## Install

```bash
pip install -r requirements.txt
```

## Configure an LLM

The agent calls the LLM through [litellm](https://github.com/BerriAI/litellm), so
any provider works. Set an API key and (optionally) a model:

```bash
export ANTHROPIC_API_KEY=sk-...                     # or OPENAI_API_KEY, etc.
export FLINT_MODEL=anthropic/claude-opus-4-6-20260205
```

The default model is **Claude Opus 4.6**, pinned to the `20260205` snapshot that
produced the results reported in the paper. Any litellm model string works
(`anthropic/...`, `openai/gpt-4o`, ...); set `FLINT_MODEL` to the exact model id
your provider or proxy exposes. To route through an OpenAI-compatible proxy
instead, set `LLM_BASE_URL`.

## Run

Built-in demo (needs only an API key — no database):

```bash
python text2sql.py --demo
```

Your own schema and question (zero-shot):

```bash
python text2sql.py \
    --schema "CREATE TABLE film (title text, length int);" \
    --question "Which films are longer than two hours?"
```

Domain-aware (FLINT) mode. The runner loads the dataset's domain rules **and**
its few-shot bank automatically. SM3's bank is in this repo, so this runs
with just a schema:

```bash
python text2sql.py --mode flint --dataset sm3 \
    --schema "$(cat sm3_schema.txt)" \
    --question "How many patients are allergic to penicillin?"
```

ScienceBenchmark's data is unlicensed, so its banks are **not** bundled — pass
the `seed.json` you download from the dataset (see below); the runner reads its
native `{question, query}` entries directly:

```bash
python text2sql.py --mode flint --dataset cordis \
    --schema "$(cat cordis_schema.txt)" \
    --examples path/to/sciencebenchmark/cordis/seed.json \
    --question "How many projects did each institution coordinate?"
```

`--dataset` choices come from `datasets/` (`cordis`, `oncomx`, `sdss`, `sm3`).
`--examples` accepts a JSON list of `{"question", "sql"}` (or `{"question",
"query"}`) and overrides the bundled bank. See **Data sources & attribution**
below for where each dataset comes from.

By default the CLI prints the first generated query. To run the **full FLINT
loop** — execute the SQL and reflect once if it raises an execution error,
exactly as the paper's numbers were produced — pass `--db-dsn` with a Postgres
connection string:

```bash
python text2sql.py --mode flint --dataset sm3 \
    --schema "$(cat sm3_schema.txt)" \
    --question "How many patients are allergic to penicillin?" \
    --db-dsn "dbname=sm3 host=localhost user=me password=..."
```

Reflection fires once: on an execution error, or when the query returns empty
and the lookup stage has verified an exact stored value that the query's filter
did not use (a case- or spelling-mismatched literal — Postgres `=` is
case-sensitive). A legitimately empty result with no such evidence is a valid
answer and is left as-is.

Use it as a library:

```python
import asyncio
from text2sql import SimpleRetriever, solve

examples = [{"question": "How many R-rated films?", "sql": "SELECT COUNT(*) FROM film WHERE rating='R';"}]
retriever = SimpleRetriever(examples)
sql = asyncio.run(solve("List the longest films", MY_SCHEMA, mode="few-shot", retriever=retriever))
print(sql)
```

To enable the database-backed stages (lookup + reflect) programmatically, call
`solve(..., execute=fn)` where `fn` is a callable `sql -> (rows, error)` backed
by your own database connection. The CLI's `--db-dsn` wires a psycopg2 callback
for you; pass your own `fn` here to use a different backend (sqlite3, etc.).

## Data sources & attribution

This agent is evaluated in the paper on two **public** benchmarks, both created
by other researchers. **Get the datasets themselves from their official
sources**. Full
credit to the original authors — please cite them if you use these datasets.

- **SM3-Text-to-Query** — medical questions over a Synthea synthetic-patient
  database. **The full SM3 dataset lives in the original repo**
  (https://github.com/jf87/SM3-Text-to-Query) — download it from there. The
  few-shot bank we ship at `datasets/data/sm3_query_bank.json` (510 examples) is
  **created by subsampling the SM3 training data** (~510 of its ~6,000 train
  rows) and reformatting to `{question, sql}`. You still need the
  SM3 database + schema from the official repo to execute queries.
  *How the few-shot bank is formed:* a subsample of the SM3 train split (bundled
  here); retrieval draws the top-k from it.

- **ScienceBenchmark** — CORDIS (EU research funding), OncoMX (cancer
  biomarkers), and SDSS (astronomy). **Get the data from the official repo**
  (https://github.com/ckosten/sciencebenchmark_dataset).
  Download it from the authors; the few-shot bank is each database's
  `seed.json`, which you pass via `--examples`.
  *How the few-shot bank is formed:* the benchmark's own `seed.json` for that
  database, used **in full** (not subsampled or hand-written) — we only
  normalize its `{question, query}` entries to `{question, sql}`; retrieval
  draws the top-k from it.

To add a new benchmark, drop a `datasets/<name>.py` module with its domain rules
(and a bundled bank only if you have the right to redistribute it).


## Prompts

We also include sanitized versions of the prompts used in our own system in the `prompts`  folder.
