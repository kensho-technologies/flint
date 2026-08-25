"""SM3-Text-to-Query — medical database on Synthea synthetic patient data.

The few-shot query bank (query_bank.json, 510 examples) ships with this repo
because SM3 is Apache-2.0. See datasets/data/sm3_query_bank.json.
"""
from .base import Dataset

DATASET = Dataset(
    name="sm3",
    domain="medical database built on Synthea synthetic patient data",
    rules="""- Patient names include numeric suffixes (e.g., "Britany225 Deckow585") — use exact match on first/last columns
- The payers table has pre-aggregated columns (covered_encounters, unique_customers, etc.) — use these directly instead of computing aggregates from other tables
- Medical codes: procedures/conditions/allergies/encounters use SNOMED-CT (bigint code column), observations use LOINC (text code like '72166-2'), medications use RxNorm (bigint), immunizations use CVX (bigint)
- Every CODE column has a companion DESCRIPTION column with the human-readable name
- REASONCODE/REASONDESCRIPTION pairs exist in encounters, procedures, medications, careplans
- When joining clinical events to patients: use the 'patient' FK column (not 'patientid' — that's only in claims/claims_transactions)
- When joining to encounters: use the 'encounter' FK column
- The encounters table connects patients to organizations (via organization FK) and payers (via payer FK)
- payer_transitions tracks insurance history; patient_expenses aggregates yearly costs
- When a question asks "which encounter" or "in which encounter", always JOIN to the encounters table and return encounters.description (not the raw UUID)
- When asked about concepts not stored in the database (e.g., "is X a donor?", "how long does effect last?", effectiveness, side effects), return: SELECT 'No answer possible based on given input'
- If the question asks about something not in the database schema, return: SELECT 'No answer possible based on given input'
- Follow the style of the example queries closely — prefer LEFT JOIN and use table aliases matching the examples""",
    examples="sm3_query_bank.json",
    credit=(
        "SM3-Text-to-Query (Sivasubramaniam et al.), Apache-2.0, "
        "https://github.com/jf87/SM3-Text-to-Query — query bank redistributed here "
        "under that license; built on Synthea synthetic patient data."
    ),
)
