"""Per-dataset domain configuration registry.

The runner (`text2sql.py`) toggles on ``--dataset`` and automatically pulls the
matching domain rules from here. To add a dataset: create a module in this
package that defines ``DATASET = Dataset(...)`` and register it below.
"""
from .base import Dataset
from . import cordis, oncomx, sdss, sm3

REGISTRY = {d.name: d for d in (cordis.DATASET, oncomx.DATASET, sdss.DATASET, sm3.DATASET)}


def available():
    """Names of the registered datasets, sorted."""
    return sorted(REGISTRY)


def get_dataset(name):
    """Return the Dataset config for `name`, or exit with a clear message."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise SystemExit(
            f"unknown dataset '{name}'. available: {', '.join(available())}"
        )
