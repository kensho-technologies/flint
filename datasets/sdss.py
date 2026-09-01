"""SDSS — Sloan Digital Sky Survey astronomy catalog (ScienceBenchmark)."""
from .base import Dataset

DATASET = Dataset(
    name="sdss",
    domain="astronomy (SDSS Sloan Digital Sky Survey)",
    rules="""- photoobj is the main photometric catalog: positions (ra, dec), galactic coordinates (l, b), magnitudes (u, g, r, i, z), and an INTEGER classification column `type`.
- specobj is the spectroscopic catalog: redshift (z), a TEXT classification column `class` holding 'STAR'/'GALAXY'/'QSO' directly, a `subclass` (e.g. 'STARFORMING', 'AGN', 'K'...), and `bestobjid` linking to photoobj.objid.
- Classification depends on WHICH observation the question names:
    * "photometrically observed/observable STAR/GALAXY/QSO" -> classify on photoobj: JOIN photo_type ON photoobj.type = photo_type.value and filter photo_type.name = 'STAR'|'GALAXY'|'QSO'.
    * "spectroscopic / spectroscopically observed STAR/GALAXY/QSO" -> filter specobj.class = 'STAR'|'GALAXY'|'QSO' DIRECTLY. specobj.class is already text: do NOT join any spec_class lookup, and do NOT add a photo_type join for a spectroscopic-class question.
- When BOTH photoobj (p) and specobj (s) are joined, SELECT the identifier and positional columns from photoobj: use p.objid (never s.bestobjid), p.ra, p.dec, p.l, p.b. specobj has its own ra/dec whose values differ from photoobj's.
- spplines holds spectral-line / "equivalent width" measurements (linked to specobj via specobjid); galspecline holds galaxy "emission line" estimates (linked via specobjid). A question about "equivalent width estimates of galactic-stars" or "emission line estimates of galaxies" refers to THESE TABLES themselves — join them as shown in the examples and do NOT add a class filter (no class='STAR'/'GALAXY') unless the question separately states a classification.
- For "everything about X" / "all about X", use SELECT * over the FULL join (all columns from every joined table), not just one table's columns.
- Do NOT use BETWEEN; write ranges with >= and < (half-open), matching this benchmark's convention.
- Positions are in degrees (ra: 0-360, dec: -90 to +90). Match subclass with LIKE (e.g. subclass LIKE '%K%').""",
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
