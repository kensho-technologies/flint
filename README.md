# FLINT — minimal standalone text-to-SQL agent

A small, self-contained reference implementation of the text-to-SQL agent behind
the paper *From Benchmarks to Production: A Text-to-SQL System for Complex
Financial Data*. It runs on **public data only** and depends on **no proprietary
system** — every component is standard, published technique.

The agent is four steps:

1. **Retrieve** similar examples — TF-IDF + cosine similarity over a small query bank.
2. **Link schema** — implicit in the prompt (the schema is passed in full).
3. **Generate** SQL — one LLM call with the schema + retrieved examples (+ optional domain rules).
4. **Reflect** — on execution error or empty results, feed the problem back and ask for a fix (optional; needs a database).

## Layout

One runner, one prompt module per dataset:

```
text2sql.py          # the runner: LLM call, retrieval, prompts, pipeline, CLI
datasets/
    base.py          # the Dataset config type
    __init__.py      # registry — maps a name to its config
    cordis.py        # EU research funding (CORDIS) domain rules
    oncomx.py        # cancer biomarker (OncoMX) domain rules
    sdss.py          # astronomy (SDSS) domain rules
    sm3.py           # medical / Synthea (SM3) domain rules
    data/
        sm3_query_bank.json   # SM3 few-shot bank (510 examples, Apache-2.0)
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
its few-shot bank automatically. SM3's bank ships with the repo, so this runs
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

Use it as a library:

```python
import asyncio
from text2sql import SimpleRetriever, generate_sql

examples = [{"question": "How many R-rated films?", "sql": "SELECT COUNT(*) FROM film WHERE rating='R';"}]
retriever = SimpleRetriever(examples)
sql = asyncio.run(generate_sql("List the longest films", MY_SCHEMA, mode="few-shot", retriever=retriever))
print(sql)
```

To enable the reflection step, pass `solve(..., execute=fn)` where `fn` is a
callable `sql -> (rows, error)` backed by your own database connection
(psycopg2, sqlite3, etc.). No database code ships here, so the agent has no DB
dependency of its own.

## Data sources & attribution

This agent is evaluated in the paper on two **public** benchmarks, both created
by other researchers. **Get the datasets themselves from their official
sources** (linked below) — we ship only the prompts (domain rules) for every
dataset, plus one small derived query bank where the license permits. Full
credit to the original authors — please cite them if you use these datasets.

- **SM3-Text-to-Query** — medical questions over a Synthea synthetic-patient
  database. **The full SM3 dataset lives in the original repo**
  (https://github.com/jf87/SM3-Text-to-Query) — download it from there. The
  few-shot bank we ship at `datasets/data/sm3_query_bank.json` (510 examples) is
  **created by subsampling the SM3 training data** (~510 of its ~6,000 train
  rows) and reformatting to `{question, sql}`; SM3 is **Apache-2.0**, so
  redistributing this subset with attribution is permitted. You still need the
  SM3 database + schema from the official repo to execute queries.
  *How the few-shot bank is formed:* a subsample of the SM3 train split (bundled
  here); retrieval draws the top-k from it.

- **ScienceBenchmark** — CORDIS (EU research funding), OncoMX (cancer
  biomarkers), and SDSS (astronomy). **Get the data from the official repo**
  (https://github.com/ckosten/sciencebenchmark_dataset). It has **no license
  file**, so we do **not** redistribute any of it: no schema, no query bank.
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
