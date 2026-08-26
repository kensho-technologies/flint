"""Schema-linking prompts for internal Transactions dataset (sanitized).

Schema linking is the step between lookup and SQL generation. Given the question,
the full database schema, and the constraints the lookup agent resolved, the model
emits a flat list of the tables, columns, join conditions, and filters needed to
answer the question. It reasons step by step first, then concatenates the result
into a single bracketed list, which is the only part the next step consumes.

Assembly
--------
The prompt is split into a static system message and a dynamic user message:
    system = SCHEMA_LINKING_INSTRUCTIONS.format(todays_date=...) + schema_information
    user   = "Examples\\n" + examples + "\\n\\n\\n" + question_components

``examples`` is the top-k most similar entries retrieved from a query bank, each
rendered as a worked step-by-step derivation ending in ``Schema_Links: [...]``.

Placeholders
------------
``{todays_date}``          today's date, so relative dates ("last quarter") resolve
``{schema_information}``   the full schema: tables, columns, and descriptions
``{question}``             the user's natural-language question
``{lookup_constraints}``   constraints resolved by the lookup agent

Sanitization
------------
Identifiers are opaque placeholders: our production system runs against a
licensed commercial transactions database, so no real table or column name
appears here. 
"""

# Instructions (dataset-specific)
SCHEMA_LINKING_INSTRUCTIONS = """You are tasked with finding the tables and columns from a SQL database which are needed to answer a question, referred to as 'schema_links'.
Include the company name when asked about a company. In addition please provide additional context when it seems relevant, such as a currency name and transaction size when asked about a specific acquisition...
Find the schema_links for generating SQL queries for the question based on the database schema. You need to provide data required to answer the question, not necessarily answer it exactly (for example, if some analysis of the data is required, please just return the data).
When part of a question requires access to data which is not available, ignore that part of the question.

Today is {todays_date}.

Important Rules:
- use company ID to search for companies (the corresponding ids will be provided to you later)
- ONLY use the tables specified below, under Database Schema. DO NOT use a table or column not mentioned there.
- return the Schema links in a single, flat list, following the format in the examples. When having elements coming from several reasoning steps, concatenate them.
- include currency name and company name to the selected columns whenever possible. Include additional columns which provide useful context to a result, such as the date...
- always include transaction status information (col16 from TableD) to provide context about whether transactions are completed, pending, cancelled, etc.
- always include TableB.col18 when referencing companies to provide human-readable company identification alongside internal IDs
- if additional constraints are specified in the database documentation, include them to the schema links
- some additional constraints will be provided in the "Lookup Constraints" section. These constraints are known from running prior queries. Include these constraints to the schema links, as well as required steps to join the relevant tables.
- for any question asking for largest transactions, or any ordering of transaction based on their size, or value, you MUST include TableA.col6 > 0 as part of your query
- do not use TableF when asked about buybacks, share repurchases, or bankruptcies
- carefully distinguish between announced dates and closed dates based on user intent. Use announced dates (col3, col4, col5) for announcement timing and closed dates (col27, col28, col29) when users ask about completion, closing, or finalized transactions
- if the question only targets american companies, do not perform currency conversion
- provide the transaction type whenever possible


Currency conversion Rules:
- always convert the values to USD, unless the question asks for a conversion in a specific currency.
- to convert to USD (or GBP, EUR), join TableE to the rest of the query as follows:
    - TableA.col7 = TableE.col7
    - DATE_FROM_PARTS(TableA.col3, TableA.col4, TableA.col5) = TableE.col1
- IMPORTANT: TableE contains these exact columns: col1, col7, col8, col9, col10, col11, col12
- do not forget to filter to the corresponding currency using appropriate ISO code (TableE.col11 = 'USD' for example). You can then multiply values by the col12 column, to obtain converted values.
- when doing conversions, also return values in reported currency using col8, and name of the reported currency
- when asked for "largest acquisition" or questions requiring comparing transaction sizes, default to converting to USD and compare USD values
- when presenting transaction sizes, use descriptive column aliases that indicate units, such as 'tr.col6 AS "Transaction Size (Millions)"' or 'tr.col6 * exrate.col12 AS "Transaction Size USD (Millions)"'

Database Schema:
"""


# Assembly templates (dataset-agnostic)
SCHEMA_LINKING_QUESTION_COMPONENTS_TEMPLATE = """Question: "{question}"
Lookup Constraints: {lookup_constraints}
Answer: Let's think step by step."""

# Single-message form, for reference. In production the same content is split
# into a cacheable system message (instructions + schema) and a user message
# (examples + question), as described in the module docstring.
SCHEMA_LINKING_PROMPT_TEMPLATE = """{schema_linking_instructions}

Database Schema:
{schema_information}


Examples
{examples}


{question_components}"""


# --------------------------------------------------------------------------- #
# Example shape                                                               #
# --------------------------------------------------------------------------- #
# One worked example, to show the derivation format the instructions refer to.
# At run time the top-k most similar entries are retrieved from a query bank and
# concatenated in this form. Values shown are illustrative.
ILLUSTRATIVE_EXAMPLE = """Question: "List bankruptcies of US companies since October 2023."
Lookup Constraints: []
Answer: Let's think step by step.
To answer the question "List bankruptcies of US companies since October 2023", we need to follow these steps:

1. Start from TableA to collect details about transactions, using the columns col17, col2, col5, col4, col3.
2. We will need to filter to US companies only. To do so, we join TableH through TableB as follows:
 - join TableB using TableB.col2 = TableA.col2
 - join TableH using TableH.col19 = TableB.col19
3. We filter on TableH.col19 for the US (the specific country ID will be resolved by lookup and provided at the query-writing step).
4. To select only bankruptcies, we can first join TableC using TableC.col13 = TableA.col13
5. We filter on TableC.col13 (the specific transaction type ID will be resolved by lookup and provided at the query-writing step).
6. To limit transactions to after October 2023, we use TableA.col3 >= 2023 and TableA.col4 >= 10
7. To provide more details, we will get the transaction status by joining TableD using TableD.col15 = TableA.col15, and collect the column col16
8. For completeness, we also select the columns TableC.col14 and TableH.col20
9. Because bankruptcies do not have transaction sizes, we do not need to perform currency conversion here.
10. Since the question does not require ordering based on transaction size, we do not need to add the constraint TableA.col6 > 0

Hence, the schema links are as follows:
Schema_Links:
[TableA.col17, TableA.col2, TableA.col5, TableA.col4, TableA.col3, TableD.col16, TableC.col14, TableH.col20, TableB.col2 = TableA.col2, TableH.col19 = TableB.col19, TableC.col13 = TableA.col13, TableD.col15 = TableA.col15, TableH.col19 = 213, TableC.col13 = 12, TableA.col3 >= 2023, TableA.col4 >= 10]
"""
