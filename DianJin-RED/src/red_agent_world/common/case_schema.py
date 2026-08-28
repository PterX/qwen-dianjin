"""Helpers for the public REDAgentBench case contract."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def normalize_expected_outcome(value: Any) -> Dict[str, Any]:
    """Return the canonical object form used by released benchmark cases."""
    if isinstance(value, str):
        return {"description": value}
    if isinstance(value, Mapping):
        outcome = dict(value)
        description = outcome.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("expected_outcome.description must be a non-empty string")
        return outcome
    raise TypeError("expected_outcome must be a string or object")


def expected_outcome_description(value: Any) -> str:
    """Read the human-readable outcome from canonical or legacy case data."""
    return str(normalize_expected_outcome(value)["description"])
