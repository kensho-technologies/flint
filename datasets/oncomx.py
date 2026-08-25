"""OncoMX — cancer biomarkers and gene expression (ScienceBenchmark)."""
from .base import Dataset

DATASET = Dataset(
    name="oncomx",
    domain="cancer biomarker (OncoMX)",
    rules="""- disease_mutation contains cancer mutations with columns: gene_symbol, ref_aa, alt_aa, position, disease_name, etc.
- biomarker_fda_test has FDA-approved biomarker tests with test_manufacturer, biomarker_id
- biomarker_edrn has its own columns (biomarker_title, biomarker_type, phase) — query it directly without joining to biomarker table
- species table maps genus/species to speciescommonname
- Gene expression data: healthy_expression and differential_expression link to anatomical_entity via uberon_anatomical_id
- Use exact string matching for gene symbols, amino acids, and disease names
- String values in this database use UPPERCASE for categorical fields (e.g., 'LOW', 'HIGH', 'Protein', 'Two') — preserve case exactly as shown in examples
- Many tables link through biomarker_id or gene_symbol
- cancer_type has type_name and parent_type for the cancer hierarchy
- Only use DISTINCT when the question explicitly asks for unique/distinct/different values or when the example queries use DISTINCT
- Do NOT use BETWEEN — use > and < operators for range queries (this database convention avoids boundary ambiguity)
- Follow the join patterns shown in the examples closely — prefer the simplest join path""",
    examples_source=(
        "ScienceBenchmark ships the few-shot bank as oncomx/seed.json in its dataset "
        "repo. Download it and pass --examples path/to/oncomx/seed.json. Not bundled "
        "here: the dataset has no license, so we do not redistribute it."
    ),
    credit=(
        "OncoMX is part of ScienceBenchmark (Kosten, Kuhn, et al.), "
        "https://github.com/ckosten/sciencebenchmark_dataset — data © its authors."
    ),
)
