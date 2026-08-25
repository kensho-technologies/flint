"""CORDIS — EU research funding (ScienceBenchmark)."""
from .base import Dataset

DATASET = Dataset(
    name="cordis",
    domain="EU research funding (CORDIS)",
    rules="""- Table and column names use underscores (e.g., project_members, start_year)
- The projects table has start_year and end_year as integers, plus start_date/end_date as full dates
- project_members links projects to people and institutions via project_id, person_id, institution_id
- Institutions have country_id which links to countries.unics_id
- Use project_member_roles to determine coordinator vs participant (role column)
- funding_schemes and activity_types use a 'code' column as their identifier
- project_programmes links projects to programmes, project_topics links projects to topics
- When counting members/participants, use project_members (not a count column on projects)
- projects has ec_fund_percentage for EU funding share and total_cost/ec_max_contribution for amounts
- For name searches use ILIKE for case-insensitive matching
- Follow the JOIN style shown in the examples (use aliases like T1, T2 or descriptive names)""",
    examples_source=(
        "ScienceBenchmark ships the few-shot bank as cordis/seed.json in its dataset "
        "repo. Download it and pass --examples path/to/cordis/seed.json (its "
        "{question, query} entries are read directly). Not bundled here: the dataset "
        "has no license, so we do not redistribute it."
    ),
    credit=(
        "CORDIS is part of ScienceBenchmark (Kosten, Kuhn, et al.), "
        "https://github.com/ckosten/sciencebenchmark_dataset — data © its authors."
    ),
)
