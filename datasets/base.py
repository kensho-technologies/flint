"""Shared type for per-dataset domain configuration.

Each dataset module in this package defines a module-level ``DATASET = Dataset(...)``
and is registered in ``datasets/__init__.py``. The runner loads one by name and
automatically pulls both its domain rules and its few-shot query bank.
"""
import os
from dataclasses import dataclass

_HERE = os.path.dirname(__file__)


@dataclass(frozen=True)
class Dataset:
    name: str                # cli identifier, e.g. "cordis"
    domain: str              # human label injected into the domain-aware prompt
    rules: str               # domain-specific rules block
    examples: str = ""       # bundled query-bank filename under datasets/data/, if we may ship it
    examples_source: str = ""  # how to obtain the query bank when it is not bundled
    credit: str = ""         # data provenance + attribution (documented in the README)

    def bundled_examples_path(self):
        """Absolute path to the shipped query bank, or None if not bundled."""
        if not self.examples:
            return None
        return os.path.join(_HERE, "data", self.examples)
