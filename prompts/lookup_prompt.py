"""Lookup agent prompts for an internal Transactions dataset (sanitized).

The lookup agent resolves the *values* that a question implies but does not
state. A question like "list all bankruptcies since 2023" needs
``TableC.col13 = 12``, but nothing in the question or the schema
documentation says that 12 means "Bankruptcy", instead that mapping lives in the data.

Two LLM calls, run in sequence:

1. ``FIND_TABLES_TO_LOOKUP_PROMPT``: table selection. Given the question and the
   reference-table schema, emit the ``SELECT`` statements whose contents are
   needed to write correct ``WHERE`` clauses (pipe-separated), or ``NOTHING``.
   The generated queries are validated against an allowlist of reference tables
   and then executed.
2. ``WRITE_CONSTRAINTS_PROMPT``: constraint generation. Called once per executed
   query, with that query's result set rendered as a text table. Returns the SQL
   constraint(s) to apply (pipe-separated), or ``NOTHING``.

The resulting constraints are passed downstream as the "Lookup Constraints"
section of the schema-linking and SQL-generation prompts, and the rows that
justified them become the "Lookup Details" (and the user-facing citation).

Placeholders
------------
``{reference_schema}``  tables + columns of the reference (dimension) tables
``{question}``          the user's natural-language question
``{query}``             the single lookup query whose results are shown
``{retrieved_data}``    that query's result set, formatted as a text table

Sanitization
------------
Identifiers are opaque placeholders: our production system runs against a
licensed commercial transactions database, so no real table or column name
appears here. Reference-table values are anonymized to ``ValueN``, 
except the ones an example actually resolves
to, which have to stay for the example to teach anything. Ids are unchanged
where a constraint references them, and a table read by several examples shows
the same listing in each.
"""

# Step 1 — table selection
FIND_TABLES_TO_LOOKUP_PROMPT = """
Your task is to determine if you need to look at values in reference tables, in order to provide accurate sql filters. This task will be referred to as "lookup".
This applies to cases where a question suggests that a specific filter (or several filters) should be added but the documentation does not provide enough information to know exactly how to write the "WHERE" constraint.
You can run queries on the reference tables to determine the appropriate value for constraint(s).
If relevant, please provide a query or list of queries (separated by a pipe character "|") that you would need to run to obtain this information.

Additional rules:
- DO NOT write another question under any circumstances
- If no lookup is required, return "NOTHING"
- do not include constraints as part of your queries, select any column which seems relevant and do not filter on rows.
- if a question asks about a company performing an acquisition or a sale of another company, or an investment in a company, you have to include SELECT * FROM TableG AND SELECT * FROM TableC as part of your answer.
- for questions mentioning a country, state or region, include SELECT * FROM TableH as part of your answer
- if a question mentions any specific type of transaction (acquisition, merger, buyback, offering, public offering, round of funding, spin off, split off, bankruptcy, private placement, shelf registration...), include SELECT * FROM TableC as part of your answer
- if a question asks about transaction completion, closure, status, or lifecycle stage, include SELECT * FROM TableD as part of your answer
- it is a safe assumption to return SELECT * FROM TableC when not sure about its relevance

Here are the reference tables and columns available:
{reference_schema}

Here are some examples to give you a sense of the task.

Question: Can you list all bankruptcies since 2023?
Queries: SELECT * FROM TableC

Question: Has Apple been involved in any cancelled transactions as a buyer?
Queries: SELECT * FROM TableD|SELECT * FROM TableG

Question: Fetch all the transactions involving Apple since March 2023
Queries: NOTHING

Question: Give me the 10 largest acquisitions from European companies in 2022, including buyer and seller
Queries: SELECT * FROM TableC|SELECT * FROM TableH|SELECT * FROM TableG

Question: Who were the underwriters in transaction with ID SPTRO4180725
Queries: SELECT * FROM TableJ

Question: What were the highest transactions in the energy sector in 2023?
Queries: SELECT * FROM TableI

Question: When did Disney acquire 21st Century Fox?
Queries: SELECT * FROM TableC|SELECT * FROM TableG

Question: Twitter acquisition by Musk
Queries: SELECT * FROM TableC|SELECT * FROM TableG

Question: How much did Verizon pay to acquire Yahoo?
Queries: SELECT * FROM TableC|SELECT * FROM TableG

Question: {question}
Queries:
"""


# Step 2 — constraint generation
# We shorten the retrieved data for brevity in the examples (lines with `# ...`)
WRITE_CONSTRAINTS_PROMPT = """
Your task is to determine, using the provided values of the reference table, which constraint (or sql filter) needs to be applied in order to accurately answer a given question.
You can ONLY use the provided information in order to write your sql constraint. Keep in mind that you are required to write the constraint for the specified table, and not provide any other constraints.

Additional rules:
- DO NOT write another question under any circumstances
- If several values apply, use IN syntax and provide the list of relevant values.
- If no constraints are required, write NOTHING
- for mergers, acquisitions and private placements, add the constraint TableG.col25 = 1 as a relation between buyer and target
- for status constraints: only add them if the question explicitly wants to FILTER by status (e.g., "in progress", "ongoing", "completed", "cancelled"). If the question asks ABOUT a date field (e.g., "what was the completion date", "when did it close", "announcement date") without filtering, do NOT add a status constraint.
- When a question mentions BOTH buyers and target companies with different attributes, you must generate separate constraints for each entity. Separate multiple constraints with a pipe character "|". See the examples with "technology companies buying healthcare companies" and "REIT companies buying real estate management companies" below.

Here are some examples for you to get a sense of what is required:

Question: Can you list all buybacks from Apple since 2022?
Query: SELECT * FROM TableC
Retrieved Data:
col13     col14
1         Private Placement
2         Merger/Acquisition
3         Fund Raising
14        Buyback
Constraint: TableC.col13 = 14


Question: What were the highest transactions in the healthcare sector in 2023?
Query: SELECT * FROM TableI
Retrieved Data:
col23         col24
1             Value1
2             Value2
3             Value3

(…36 rows…) 

40            Health Care Equipment and Supplies
41            Health Care Providers and Services
42            Health Care Technology
43            Biotechnology
44            Pharmaceuticals
45            Life Sciences Tools and Services
46            Value31
49            Value32
50            Value33
51            Value34
53            Real Estate Management and Development
55            Value35
56            Value36
Constraint: TableI.col23 IN (40, 41, 42, 43, 44, 45)


Question: List companies acquired by blackstone since 2013
Query: SELECT * FROM TableG
Retrieved Data:
col25        col26
1            Investor / Buyer
2            Value1
3            Value2
4            Value3
5            Value4
Constraint: TableG.col25 = 1


Question: Companies bought by S&P Global since 2015?
Query: SELECT * FROM TableC
Retrieved Data:
col13     col14
1         Private Placement
2         Merger/Acquisition
6         Value1
7         Value2
10        Fund Raising
12        Bankruptcy
14        Buyback
15        Value3
Constraint: TableC.col13 = 2


Question: How many rounds of funding has Company X received?
Query: SELECT * FROM TableC
Retrieved Data:
col13     col14
1         Private Placement
2         Merger/Acquisition
6         Value1
7         Value2
10        Fund Raising
12        Bankruptcy
14        Buyback
15        Value3
Constraint: TableC.col13 IN (1, 10)


Question: Are REIT companies of North America buying real estate management companies?
Query: SELECT * FROM TableI
Retrieved Data:
col23         col24
1             Value1
2             Value2
29            Value24
45            Life Sciences Tools and Services
52            Diversified REITs
53            Real Estate Management and Development
55            Value35
71            Health Care REITs
72            Hotel and Resort REITs
73            Industrial REITs
74            Office REITs
75            Residential REITs
76            Retail REITs
77            Specialized REITs
Constraint: TableI.col23 = 53|TableI.col23 IN (52, 71, 72, 73, 74, 75, 76, 77)


Question: Are cable companies of North America buying Australian companies?
Query: SELECT * FROM TableH
Retrieved Data:
col19        col20             col21       col22
1            Value1            10001       Region1
13           Australia         10001       Region1
37           Value2            10359       United States and Canada
43           Value3            10001       Region1
76           Value4            10002       Region2
212          Value5            10002       Region2
213          United States     10359       United States and Canada
Constraint: TableH.col21 = 10359|TableH.col19 = 13


Question: What was the bankruptcy completion date for Pier 1 Imports?
Query: SELECT * FROM TableD
Retrieved Data:
col15       col16
1           In Progress
2           Value1
3           Value2
7           Value3
Constraint: NOTHING


Question: How many M&A deals are still in progress for Apple?
Query: SELECT * FROM TableD
Retrieved Data:
col15       col16
1           In Progress
2           Value1
3           Value2
7           Value3
Constraint: TableD.col15 = 1


Your job is to take care of the following:
Question: {question}
Query: {query}
Retrieved Data:
{retrieved_data}
Constraint:
"""


# Parsing
# The agent reads each generation after its trigger token ("Queries:" /
# "Constraint:"), treats the sentinel below as an empty result, and splits the
# remainder on "|" to recover the individual queries or constraints.
NOTHING_TO_LOOKUP_TOKEN = NO_CONSTRAINT_TOKEN = "NOTHING"
WRITE_QUERIES_TRIGGER = "Queries:"
WRITE_CONSTRAINTS_TRIGGER = r"Constraints?:"

# Reference (dimension) tables the lookup agent is allowed to query. Generated
# queries that touch anything outside this allowlist are dropped.
REFERENCE_TABLES = [
    "TableC",
    "TableD",
    # ...
]
