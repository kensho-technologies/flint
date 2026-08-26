"""SQL-generation prompts for an internal Transactions dataset (sanitized).

The model receives the question, the ``schema_links`` produced by
schema linking, the disambiguated entity IDs, and the lookup constraints (plus the
rows that justified them), and writes the SQL.

Assembly
--------
::

    WRITE_QUERY_PROMPT_TEMPLATE.format(
        write_query_instructions=WRITE_SQL_INSTRUCTIONS.format(todays_date=...),
        examples="\\n\\n".join(EXAMPLE_FORMATTING_TEMPLATE.format(...) for each retrieved example),
        question_components=QUESTION_COMPONENTS_TEMPLATE.format(...),
    )

``examples`` is the top-k most similar entries retrieved from a query bank, each
rendered with ``WRITE_QUERY_EXAMPLE_FORMATTING_TEMPLATE`` so that examples and task
have an identical shape.

Placeholders
------------
``{todays_date}``          today's date, so relative dates ("last quarter") resolve
``{question}``             the user's natural-language question
``{schema_links}``         flat list of tables, columns, joins, and filters
``{entity_ids}``           resolved company/entity identifiers, as a mapping
``{lookup_constraints}``   constraints resolved by the lookup agent
``{lookup_details}``       the reference rows those constraints came from
``{sql}``                  the gold SQL, in examples only


Sanitization
------------
Identifiers are opaque placeholders: our production system runs against a
licensed commercial transactions database, so no real table or column name
appears here. 
"""

# Instructions (dataset-specific)
WRITE_SQL_INSTRUCTIONS = """Using the provided schema links, write a SQL query which retrieves data to answer the question.
You need to only write the SQL query; do not include explanations of your approach.
Only write SQL code to retrieve data to answer the provided question. You are not allowed to write another question.

Today is {todays_date}.


# Important rules:
- When asked about specific companies, use the provided company id (when available) as part of the context.
- include the names of the companies involved, transaction status information, and any other data which provides good context (dates, ...)
- when no date is specified, please fetch the latest information available from the database and expose the most recent data first
- do NOT use ";" to end your queries
- answer the provided question only. Under NO CIRCUMSTANCES, NEVER write another question of your own
- for any question asking for largest transactions, or any ordering of transaction based on their size, or value, you MUST include TableA.col6 > 0 as part of your query
- be mindful about now including columns that are not a valid group by expression to the SELECT statement (even if these are part of the schema links)
- only use table names and columns from the schema links provided. Do not invent or assume any additional table names or fields.
- when a question requires several identifiers but only some of them are provided, write your query with available ones. Do not try to use col18 to find the company.
- for questions targeting a specific acquisition, for which both identifiers are provided, refer to the specific rules below.

# Currency conversion Rules:
- always convert the values to USD, unless the question asks for a conversion in a specific currency.
- to convert to USD (or GBP, EUR), join TableE to the rest of the query as follows:
    - TableA.col7 = TableE.col7
    - DATE_FROM_PARTS(TableA.col3, TableA.col4, TableA.col5) = TableE.col1
- IMPORTANT: TableE contains these exact columns: col1, col7, col8, col9, col10, col11, col12
- do not forget to filter to the corresponding currency using appropriate ISO code (TableE.col11 = 'USD' for example). You can then multiply values by the col12 column, to obtain converted values.
- when doing conversions, also return values in reported currency using col8, and name of the reported currency
- when asked for "largest acquisition" or questions requiring comparing transaction sizes, default to converting to USD and compare USD values
- when asked about transactions in a given currency, return all corresponding transactions with col6 converted to the specified currency.

# Specific rules for specific mergers, acquisitions, private placements or offerings:
When a question is about a very specific transaction, where both buyer and seller, or buyer and target are part of the question (questions such as 'tell me about the acquisition of ... by ...'), proceed as follows.
This applies to cases where BOTH identifiers are provided.
If not, disregard the following section and only use the available identifiers as part of your query. Do not write clauses for the missing ones. For example, if you only have the identifier of the buyer, write a query to retrieve all of their acquisitions.
1. Create a companyIdentifiers CTE with buyer and target company ids (similarly for seller, if this is the question you receive), with ids as integers, such as
WITH companyIdentifiers AS (
    SELECT ... AS targetId, ... AS buyerId
)
2. Join the companyIdentifiers to the rest of your query, using simply JOIN companyIdentifiers. Do not use any other column to join this table.
3. Use the following priority list as part of select statement
    CASE
        WHEN targetcompany.col2 = companyIdentifiers.targetId AND buyer.col2 = companyIdentifiers.buyerId  THEN 1
        WHEN targetcompany.col2 = companyIdentifiers.buyerId AND buyer.col2 = companyIdentifiers.targetId THEN 2
        WHEN targetcompany.col2 = companyIdentifiers.targetId THEN 3
        WHEN buyer.col2 = companyIdentifiers.buyerId  THEN 4
        ELSE 0
    END as score
4. When selecting final results, make sure to filter to positive score (WHERE score > 0), and only select the lowest score.
5. Do not use the LIMIT keyword in this specific case. Instead, make sure to select the lowest positive score. You can use CTEs if needed.

Here are some examples for you, to get a better sense of what is required:
"""


# Assembly templates (dataset-agnostic)
WRITE_QUERY_QUESTION_COMPONENTS_TEMPLATE = """Question: "{question}". Do not write another question.
Schema_links: {schema_links}
Entity IDs: {entity_ids}
Lookup Constraints: {lookup_constraints}
Lookup Details: {lookup_details}
SQL:"""

WRITE_QUERY_EXAMPLE_FORMATTING_TEMPLATE = """Question: "{question}"
Schema_links: {schema_links}
Entity IDs: {entity_ids}
Lookup Constraints: {lookup_constraints}
Lookup Details: {lookup_details}
SQL: {sql}"""

WRITE_QUERY_PROMPT_TEMPLATE = """{write_query_instructions}

Examples:
{examples}

Here's your task:
{question_components}"""
