"""SDSS — Sloan Digital Sky Survey astronomy catalog (ScienceBenchmark)."""
from .base import Dataset

DATASET = Dataset(
    name="sdss",
    domain="astronomy (SDSS Sloan Digital Sky Survey)",
    rules="""- photoobj is the main photometric catalog with positions (ra, dec), magnitudes, and classification (type)
- specobj is the spectroscopic catalog with redshift (z), spectral class (class), and links to photoobj via bestobjid
- photo_type maps integer type values to names ('STAR', 'GALAXY', 'QSO', etc.)
- spec_class maps integer class values to names
- Use JOINs through photo_type/spec_class to filter by object classification
- Positions are in degrees (ra: 0-360, dec: -90 to +90)
- When asked for 'star'/'galaxy'/'QSO', join to the type/class lookup table
- spplines contains spectral line measurements linked to specobj via specobjid
- Magnitudes are in columns u, g, r, i, z (SDSS filter bands)""",
    examples_source=(
        "ScienceBenchmark ships the few-shot bank as sdss/seed.json in its dataset "
        "repo. Download it and pass --examples path/to/sdss/seed.json. Not bundled "
        "here: the dataset has no license, so we do not redistribute it."
    ),
    credit=(
        "SDSS (Sloan Digital Sky Survey) is part of ScienceBenchmark (Kosten, Kuhn, "
        "et al.), https://github.com/ckosten/sciencebenchmark_dataset — data © its authors."
    ),
)
