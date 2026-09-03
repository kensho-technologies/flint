"""FLINT — a staged text-to-SQL agent (LangGraph).

Stages: retrieve examples -> lookup values -> link schema -> write SQL ->
execute -> reflect. `lookup` and `reflect` are grounded in the database; the
schema-link stage is opt-in. Any litellm-supported model works.

    export FLINT_MODEL=anthropic/claude-opus-4-6-20260205   # + the provider's API key
    python text2sql.py --demo                                # generation only, no DB

    python text2sql.py --mode flint --dataset cordis \
        --schema "$(cat schema.sql)" --examples bank.json \
        --db-dsn "dbname=cordis host=... user=... password=..." \
        --question "How many projects did each institution coordinate?"

See README.md for the full description.
"""

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any, Callable, Optional, TypedDict

import litellm
import numpy as np
import psycopg2
from langgraph.graph import END, StateGraph
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from datasets import available, get_dataset

litellm.drop_params = True
litellm.num_retries = 3

# Any litellm-supported model string works (anthropic/..., openai/..., etc.).
# Override with the FLINT_MODEL env var; keys are read from the environment by
# litellm. The paper's numbers use Claude Opus 4.6, pinned to the dated snapshot
# below for reproducibility.
DEFAULT_MODEL = os.getenv("FLINT_MODEL", "anthropic/claude-opus-4-6-20260205")


# --------------------------------------------------------------------------- #
# LLM call (provider-agnostic via litellm; honors an OpenAI-compatible proxy)  #
# --------------------------------------------------------------------------- #
async def call_llm(prompt, model=DEFAULT_MODEL, temperature=0.0):
    """Single-turn prompt -> text. Uses LLM_BASE_URL as an api_base if set."""
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
    return response.choices[0].message.content or ""


# --------------------------------------------------------------------------- #
# Example retrieval (TF-IDF + cosine similarity)                              #
# --------------------------------------------------------------------------- #
class SimpleRetriever:
    """Retrieve the top-k most similar example questions for few-shot prompting.
    `examples` is a list of {"question": str, "sql": str} dicts."""

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

# Token the lookup step emits when a question implies no literal value to resolve.
NOTHING_TOKEN = "NOTHING"

# A genuine code/category column has few distinct values. If a probe returns more
# than this, treat the column as high-cardinality / free-text and skip grounding
# an exact value there (the query likely wants a substring match).
MAX_CATEGORY_CARDINALITY = 60

# Lookup step 1: plan the value probes.
LOOKUP_PLAN_PROMPT = """You are the value-lookup planner of a text-to-SQL agent.

A question often refers to a concept -- a category, status, type, role, class, or
proper label ("partner role", "star-forming galaxies", "cancelled deals") -- that
the database stores as a specific CODE or an exact string in some column. The
mapping from the concept to the stored value lives in the DATA, not the schema, so
before writing a WHERE clause you must inspect the distinct values of the columns
involved.

Given the schema and question, list the probe queries you need -- one
`SELECT DISTINCT <column> FROM <table>` per column whose stored values you must see
to turn a concept in the question into a filter. Use ONE of two probe shapes:
  * SMALL code/category column (a handful of codes -- a status, a type lookup
    table): probe it whole, `SELECT DISTINCT status FROM t` (you may pair a code
    with its label, `SELECT DISTINCT code, label FROM t`), so you see every code.
  * A SPECIFIC named value on a LARGE column (a proper label / subclass / name the
    question quotes, where the column holds many distinct values): narrow it with
    the concept's own word --
    `SELECT DISTINCT subclass FROM t WHERE subclass::text ILIKE '%Carbon%'` -- so a
    high-cardinality column still returns just the few stored values that match,
    letting you pick the exact one instead of guessing a spelling or a LIKE.
Choose only columns holding categorical / coded / label values a WHERE clause would
filter on -- never numbers, dates, ids, or free-text sentences. Separate multiple
queries with a pipe "|". Write ONLY the queries, no prose. If the question needs no
such value lookup, respond with exactly: {nothing}

Database Schema:
{schema}

Question: {question}
Queries:"""

# Lookup step 2: read the rows, write the constraint.
LOOKUP_CONSTRAINT_PROMPT = """You are resolving the exact SQL filter for a question
using the actual values stored in a column.

Below is one probe query and the distinct rows it returned. Decide which value(s)
correspond to the concept in the question and write the SQL constraint(s) for that
column -- e.g. `table.col = 'Value'` or `table.col IN ('A','B')`. Use ONLY values
that appear in the rows below (copy them exactly, including case). If several
apply, use IN(...). Separate multiple constraints with a pipe "|". Write ONLY the
constraint(s), no prose. If none of the rows correspond to the question, respond
with exactly: {nothing}

Question: {question}
Query: {query}
Result rows:
{rows}
Constraint:"""

# Schema linking: reason, then emit a flat list of columns + joins + filters.
SCHEMA_LINK_PROMPT = """You are the schema-linking step of a text-to-SQL agent.

Given the database schema and a question, work out exactly what is needed to
answer it: the tables and columns to read, the JOIN conditions to connect them
(written `t1.a = t2.b`), and the value filters. Reason step by step. Then, on the
final line, concatenate everything into ONE flat, bracketed list.

Rules:
- Use ONLY tables and columns from the schema below.
- Follow the join paths and table-family choices shown in the examples above: when
  the examples reach a fact through a particular table, use that SAME table, not a
  different one with a similar name.
- Include EVERY column the question asks to return, plus the minimal set of tables
  and any bridge/junction tables needed to join them.
- Include the Lookup Constraints below verbatim in the list, and add the joins
  needed to reach the tables they mention.
- End with a line in exactly this format (nothing after it):
Schema_Links: [col_or_join_or_filter, col_or_join_or_filter, ...]

Database Schema:
{schema}

{few_shot_examples}

Question: "{question}"
Lookup Constraints: {lookup_constraints}
Answer: Let's think step by step."""

# Reflect (grounding branch): the written SQL filters on a value the lookup step did
# NOT verify against the DB, while it DID find the exact stored value. Repair in place.
GROUNDED_REFLECT_PROMPT = """You are correcting a SQL query. The lookup step probed the
database and found the EXACT stored value(s) for the concept in the question (listed
below as verified constraints, each in `table.column = 'value'` form). The current
query instead filters on a DIFFERENT, unverified value -- so it matches the wrong rows,
or none.

Rewrite the query so its filter uses a verified constraint:
- If the verified value is in the SAME column the query already filters, just replace
  the literal.
- If the verified value lives in a DIFFERENT table/column, filter on THAT column
  instead, adding the minimal JOIN needed to reach it.
Change nothing else: keep the SELECTed columns, the other joins and filters, grouping,
ordering and limit exactly as they are. Keep a substring (LIKE) match if the question
asks about text inside a longer free-text field. Write ONLY the SQL query, no
explanation. Do not end with ";".

Database Schema:
{schema}

Question: "{question}"
Current SQL: {sql}
Verified constraints (exact, from the database): {lookup_constraints}
Corrected SQL:"""


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
    """Assemble the write-SQL prompt for the requested mode."""
    if mode == "zero-shot":
        return ZERO_SHOT_PROMPT.format(schema=schema, question=question)
    if mode == "few-shot":
        return FEW_SHOT_PROMPT.format(
            schema=schema, question=question, few_shot_examples=few_shot_text
        )
    if mode == "flint":
        return FLINT_PROMPT.format(
            schema=schema, question=question, few_shot_examples=few_shot_text,
            domain=domain, domain_rules=domain_rules,
        )
    raise ValueError(f"unknown mode: {mode}")


def _schema_table_names(schema):
    """Table names mentioned in the schema text (Table: X / CREATE TABLE X)."""
    names = set()
    for m in re.finditer(r"(?:table:|create\s+table)\s+[`\"']?(\w+)", schema, re.I):
        names.add(m.group(1).lower())
    return names


def _parse_pipe_list(text, trigger=None):
    """Text after an optional trigger word, split on '|', NOTHING -> []."""
    if not text:
        return []
    body = text.strip()
    if trigger:
        m = re.search(trigger, body)
        if m:
            body = body[m.end():]
    body = body.strip()
    if not body or body.upper().startswith(NOTHING_TOKEN):
        return []
    parts = [p.strip().rstrip(";").strip() for p in body.split("|")]
    return [p for p in parts if p and p.upper() != NOTHING_TOKEN]


def _safe_probe(sql, table_names):
    """Accept a probe only if it is a single read-only SELECT over known tables."""
    s = sql.strip().rstrip(";").strip()
    if not re.match(r"(?is)^select\b", s):
        return None
    if re.search(r"(?is)\b(insert|update|delete|drop|alter|create|truncate|grant|;)\b", s):
        return None
    used = {t.lower() for t in re.findall(r"(?is)\bfrom\s+([`\"']?\w+)", s)}
    used |= {t.lower() for t in re.findall(r"(?is)\bjoin\s+([`\"']?\w+)", s)}
    used = {t.strip("`\"'") for t in used}
    if table_names and not used.issubset(table_names):
        return None
    if not re.search(r"(?is)\blimit\b", s):
        s += " LIMIT 200"
    return s


def _render_rows(rows, max_rows=60):
    out = []
    for r in rows[:max_rows]:
        if isinstance(r, (list, tuple)):
            out.append("  ".join("" if v is None else str(v) for v in r))
        else:
            out.append("" if r is None else str(r))
    if len(rows) > max_rows:
        out.append(f"(... {len(rows) - max_rows} more rows ...)")
    return "\n".join(out)


def _sql_literals(text):
    """String literals in a SQL / constraint string ('' unescaped). Exact case --
    Postgres '=' is case-sensitive, so grounding must be too."""
    if not text:
        return []
    return [m.replace("''", "'") for m in re.findall(r"'((?:[^']|'')*)'", text)]


def _parse_schema_links(text):
    """The bracketed list after the final 'Schema_Links:' marker (or last [...])."""
    if not text:
        return ""
    m = list(re.finditer(r"schema_links\s*:", text, re.I))
    tail = text[m[-1].end():] if m else text
    b = tail.find("[")
    e = tail.rfind("]")
    if b != -1 and e > b:
        return tail[b:e + 1].strip()
    return tail.strip()


# --------------------------------------------------------------------------- #
# Agent state                                                                 #
# --------------------------------------------------------------------------- #
class AgentState(TypedDict, total=False):
    question: str
    schema: str
    model: str
    mode: str
    k: int
    domain: str
    domain_rules: str
    do_schema_link: bool
    do_lookup: bool
    ground_at_generate: bool  # fold verified values into the write prompt (vs. repair at reflect)
    retriever: Any
    execute: Optional[Callable]
    few_shot_text: str
    lookup_constraints: str
    lookup_details: str
    verified_values: list  # DB-confirmed stored literals (grounding truth for reflect)
    schema_links: str
    sql: str
    error: Optional[str]
    empty: bool  # execute returned zero rows (grounding-reflect precondition)
    reflected: bool


# --------------------------------------------------------------------------- #
# Nodes                                                                       #
# --------------------------------------------------------------------------- #
async def retrieve_node(state: AgentState) -> dict:
    """Retrieve top-k few-shot examples (skipped in zero-shot mode)."""
    retriever = state.get("retriever")
    few = ""
    if retriever is not None and state.get("mode") in ("few-shot", "flint"):
        k = state.get("k", 3)
        few = format_few_shot(retriever.retrieve(state["question"], k=k), k=k)
    return {"few_shot_text": few}


async def lookup_node(state: AgentState) -> dict:
    """Two-call lookup grounded in the real data (pass-through when off / no DB):
    plan SELECT DISTINCT probes, run them, read the rows, emit exact constraints and
    the DB-confirmed stored literals."""
    execute = state.get("execute")
    if not state.get("do_lookup", True) or execute is None:
        return {}
    model, question, schema = state["model"], state["question"], state["schema"]

    plan = await call_llm(LOOKUP_PLAN_PROMPT.format(
        schema=schema, question=question, nothing=NOTHING_TOKEN), model=model)
    probes = _parse_pipe_list(plan, trigger=r"queries\s*:")
    if not probes:
        return {}

    table_names = _schema_table_names(schema)
    constraints, details, verified = [], [], []
    for probe in probes[:4]:
        safe = _safe_probe(probe, table_names)
        if not safe:
            continue
        rows, error = execute(safe)
        if error or not rows:
            continue
        # High-cardinality / free-text column: don't ground an exact value here.
        if len(rows) > MAX_CATEGORY_CARDINALITY:
            continue
        # DB-confirmed stored literals -- the grounding truth (exact case, textual,
        # non-numeric, non-trivial cells; the kind that appear as filter literals).
        for r in rows:
            cells = r if isinstance(r, (list, tuple)) else [r]
            for v in cells:
                if v is None:
                    continue
                s = str(v).strip()
                if len(s) >= 2 and not s.isdigit():
                    verified.append(s)  # EXACT DB case (Postgres '=' is case-sensitive)
        rendered = _render_rows(rows)
        c = await call_llm(LOOKUP_CONSTRAINT_PROMPT.format(
            question=question, query=safe, rows=rendered, nothing=NOTHING_TOKEN), model=model)
        got = _parse_pipe_list(c, trigger=r"constraints?\s*:")
        if got:
            constraints.extend(got)
            details.append(f"{safe}\n{rendered}")

    if not constraints:
        return {}
    constraints = list(dict.fromkeys(constraints))  # de-dupe, preserve order
    return {
        "lookup_constraints": " | ".join(constraints),
        "lookup_details": "\n\n".join(details),
        "verified_values": list(dict.fromkeys(verified)),
    }


async def schema_link_node(state: AgentState) -> dict:
    """Reasoned schema linking -> flat list of columns + joins + filters (opt-in)."""
    if not state.get("do_schema_link", True):
        return {}
    prompt = SCHEMA_LINK_PROMPT.format(
        schema=state["schema"],
        few_shot_examples=state.get("few_shot_text", ""),
        question=state["question"],
        lookup_constraints=state.get("lookup_constraints", "") or "[]",
    )
    links = _parse_schema_links(await call_llm(prompt, model=state["model"]))
    return {"schema_links": links}


async def generate_node(state: AgentState) -> dict:
    """Write SQL on the few-shot prompt (its retrieved examples are the strongest
    guide), with the earlier stages appended as ADDITIVE schema hints:
      - schema links: advisory ("consider these AND any others the examples/question
        need") -- an incomplete link must never narrow generation to one table.
      - lookup constraints: injected only when `ground_at_generate` is set; otherwise
        the DB-verified values are held for reflect (Design B)."""
    schema_links = state.get("schema_links", "")
    lookup_constraints = state.get("lookup_constraints", "")
    eff_schema = state["schema"]
    if schema_links:
        eff_schema += (
            "\n\n-- Likely-relevant tables/columns (ADDITIVE hint from schema linking): "
            "use these to make sure you cover the needed tables, but still join any "
            "other tables the examples or the question require -- do NOT restrict the "
            "query to only these, and do not drop a join or column they imply.\n"
            + schema_links
        )
    if lookup_constraints and state.get("ground_at_generate", False):
        eff_schema += (
            "\n\n-- Verified column values (exact stored values, checked against "
            "the DB): use these instead of guessing spellings, but keep a "
            "substring (LIKE) match for text inside a longer free-text field, and "
            "do NOT add an OR/IN over extra spellings the question did not ask for.\n"
            + lookup_constraints
        )
    prompt = build_prompt(
        state["question"], eff_schema, mode=state.get("mode", "few-shot"),
        few_shot_text=state.get("few_shot_text", ""),
        domain=state.get("domain", ""), domain_rules=state.get("domain_rules", ""),
    )
    sql = strip_markdown_fences(await call_llm(prompt, model=state["model"]))
    return {"sql": sql}


async def execute_node(state: AgentState) -> dict:
    """Run the query. `empty` = ran clean but returned no rows (grounding-reflect
    precondition), so a query that DOES return rows is never second-guessed."""
    execute = state.get("execute")
    if execute is None:
        return {"error": None, "empty": False}
    rows, error = execute(state["sql"])
    empty = (not error) and (rows is not None) and (len(rows) == 0)
    return {"error": error, "empty": empty}


async def reflect_node(state: AgentState) -> dict:
    """Single reflect stage, two triggers (see `_route_after_execute`): an execution
    error -> fix the error; else a phantom filter literal -> substitute the
    DB-verified value the lookup stage already found."""
    if state.get("error"):
        prompt = REFLECTION_PROMPT.format(
            question=state["question"], sql=state["sql"], error=state["error"],
            schema=state["schema"],
        )
    else:
        verified = state.get("verified_values") or []
        lits = _sql_literals(state["sql"])
        lits_lc = [l.lower() for l in lits]
        # Values the query has right except for case (Postgres '=' won't match them).
        miscased = [v for v in verified
                    if any(v.lower() in l for l in lits_lc)
                    and not any(v in l for l in lits)]
        # Keep only constraint segments naming a confirmed value (drop guesses).
        segs = [s.strip() for s in state.get("lookup_constraints", "").split("|")]
        clean = [s for s in segs if s and any(v.lower() in s.lower() for v in verified)]
        parts = []
        if clean:
            parts.append(" | ".join(clean))
        if miscased:
            parts.append("Exact stored values (match case EXACTLY): "
                         + ", ".join("'%s'" % v for v in dict.fromkeys(miscased)))
        hint = "\n".join(parts) if parts else state.get("lookup_constraints", "")
        prompt = GROUNDED_REFLECT_PROMPT.format(
            question=state["question"], sql=state["sql"],
            lookup_constraints=hint, schema=state["schema"],
        )
    fixed = strip_markdown_fences(await call_llm(prompt, model=state["model"]))
    return {"sql": fixed, "reflected": True}


def _route_after_execute(state: AgentState) -> str:
    """Reflect once, on an execution error OR a grounding-verified phantom literal.

    The grounding trigger fires only when the lookup stage confirmed exact stored
    value(s) against the DB AND the query came back EMPTY AND none of its string
    literals case-sensitively match a confirmed value -- i.e. the filter value is
    absent or miscased (WHERE col='Blood' returns nothing when the stored value is
    'blood'). A query that returns rows, or a correctly-empty answer that uses valid
    values, is never touched. Empty results alone never trigger reflection."""
    if state.get("reflected"):
        return "end"
    if state.get("error"):
        return "reflect"
    verified = state.get("verified_values") or []
    if verified and state.get("empty"):
        lits = _sql_literals(state.get("sql", ""))
        if lits and not any(v in lit for lit in lits for v in verified):
            return "reflect"
    return "end"


# --------------------------------------------------------------------------- #
# Graph                                                                       #
# --------------------------------------------------------------------------- #
def build_graph():
    """retrieve -> lookup -> schema_link -> generate -> execute -> reflect?"""
    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("lookup", lookup_node)
    g.add_node("schema_link", schema_link_node)
    g.add_node("generate", generate_node)
    g.add_node("execute", execute_node)
    g.add_node("reflect", reflect_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "lookup")
    g.add_edge("lookup", "schema_link")
    g.add_edge("schema_link", "generate")
    g.add_edge("generate", "execute")
    g.add_conditional_edges("execute", _route_after_execute,
                            {"reflect": "reflect", "end": END})
    g.add_edge("reflect", END)
    return g.compile()


_APP = build_graph()


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
    schema_link=False,
    lookup=True,
    ground_at_generate=False,
):
    """Run the staged agent and return the final SQL string.

    `execute`, if given, is a callable `sql -> (rows, error)` (error None on
    success); it drives the lookup and reflect stages. Without it, generation-only.
    `lookup` (default ON) no-ops without `execute`. `schema_link` (default OFF) is a
    per-schema opt-in: it helps FK-dense schemas but can over-constrain table choice
    elsewhere. `ground_at_generate` (default OFF): hold verified values for the
    reflect stage's grounded self-repair rather than folding them into generation.
    """
    state: AgentState = {
        "question": question, "schema": schema, "model": model, "mode": mode, "k": k,
        "domain": domain, "domain_rules": domain_rules,
        "do_schema_link": schema_link, "do_lookup": lookup,
        "ground_at_generate": ground_at_generate,
        "retriever": retriever, "execute": execute,
        "few_shot_text": "", "lookup_constraints": "", "lookup_details": "",
        "verified_values": [], "empty": False,
        "schema_links": "", "sql": "", "error": None, "reflected": False,
    }
    final = await _APP.ainvoke(state)
    return final.get("sql", "")


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
    sql = await solve(question, DEMO_SCHEMA, mode="few-shot", retriever=retriever)
    print("Generated SQL:\n" + sql)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def _load_examples(path):
    """Load a query bank: JSON list of question/SQL pairs. Accepts both this repo's
    {"question", "sql"} shape and ScienceBenchmark's seed.json {"question", "query"}."""
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
    parser = argparse.ArgumentParser(description="FLINT staged text-to-SQL agent (LangGraph).")
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
    parser.add_argument(
        "--db-dsn", dest="db_dsn", default=None,
        help="Postgres DSN (e.g. 'dbname=sm3 host=... user=... password=...'). When "
             "given, the agent EXECUTES the SQL and runs the lookup + reflect stages "
             "against the DB. Omit for generation-only output (no database dependency).",
    )
    parser.add_argument(
        "--schema-link", dest="schema_link", action="store_true",
        help="enable the schema-linking stage (default OFF): helps FK-dense schemas "
             "with many join paths, but can over-constrain table choice on others -- "
             "opt in per schema.",
    )
    parser.add_argument(
        "--no-lookup", dest="lookup", action="store_false",
        help="ablation: skip the lookup stage (also disables the grounded reflect).",
    )
    parser.set_defaults(schema_link=False, lookup=True)
    args = parser.parse_args()

    if args.demo:
        asyncio.run(_demo())
        return
    if not (args.schema and args.question):
        parser.error("provide --schema and --question, or use --demo")

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

    execute = None
    conn = None
    if args.db_dsn:
        conn = psycopg2.connect(args.db_dsn)

        def execute(sql):
            try:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    rows = cur.fetchall() if cur.description else []
                conn.commit()
                return rows, None
            except Exception as exc:  # noqa: BLE001 - surface any DB error to reflection
                conn.rollback()
                print(f"[runner] query failed: {exc}", file=sys.stderr)
                return [], f"ERROR: {exc}"

    try:
        sql = asyncio.run(solve(
            args.question, args.schema, model=args.model, mode=args.mode,
            retriever=retriever, k=args.k, domain=domain, domain_rules=domain_rules,
            execute=execute, schema_link=args.schema_link, lookup=args.lookup,
        ))
    finally:
        if conn is not None:
            conn.close()
    print(sql)


if __name__ == "__main__":
    main()
