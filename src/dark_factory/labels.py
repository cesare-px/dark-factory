"""Label identity, decoupled from the strings a downstream repo actually uses.

Agents emit `LabelKey` members; only the pipeline boundary (pipeline.py)
resolves them to the configured label strings via `FactoryConfig.label()`.
This removes every hardcoded "factory-blocked" / "spec-validated" string
from agent code, so renaming a label in config never touches Python.
"""

from __future__ import annotations

from enum import StrEnum


class LabelKey(StrEnum):
    """Identity for a factory label, decoupled from its configured string."""

    SPEC_VALIDATED = "spec_validated"
    NEEDS_SPECIFICATION = "needs_specification"
    BLOCKED = "blocked"
    SHIPPED = "shipped"
