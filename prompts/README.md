# Sanitized prompts

The domain-specific instruction prompts for the **Transactions** dataset, as used in the paper *From Benchmarks to Production: A Text-to-SQL System for Complex Financial Data* — configuration artifact 4 of §3.7.

| File | Paper | Contents |
|---|---|---|
| `lookup_prompt.py` | §3.1 Lookup Agent | Table selection and constraint generation — the two LLM calls that resolve natural-language concepts to reference-table IDs |
| `schema_linking_prompt.py` | §3.4 Schema Linking | Instructions for selecting tables, columns, joins, and filter predicates |
| `write_sql_prompt.py` | §3.5 SQL Generation | Rules for rendering schema links as valid SQL |

At inference these run in pipeline order — lookup → schema linking → SQL generation, each stage consuming the previous stage's output. 

## Sanitization

The production system queries a licensed commercial database, so no real table or column name appears in these prompts. Identifiers are opaque placeholders (e.g., `TableA`, `col6`) 
Reference-table values are anonymized to `ValueN` too, except the ones an example actually resolves to, which have to stay for the example to teach anything. Ids are kept where a constraint references them and renumbered otherwise. Prompt structure and rule wording are unchanged.
