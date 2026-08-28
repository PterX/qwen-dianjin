from __future__ import annotations

import json
from pathlib import Path

from red_agent_world.common.case_schema import expected_outcome_description


ROOT = Path(__file__).resolve().parents[1]


def test_released_cases_use_one_expected_outcome_schema() -> None:
    seen = 0
    for path in sorted((ROOT / "test").glob("[ETU][0-9].json")):
        for case in json.loads(path.read_text(encoding="utf-8")):
            outcome = case["expected_outcome"]
            assert isinstance(outcome, dict), (path.name, case["id"])
            assert expected_outcome_description(outcome).strip()
            taxonomy = case["taxonomy"]
            assert set(taxonomy) == {"intervention", "vulnerability", "constraint"}, (
                path.name,
                case["id"],
                taxonomy,
            )
            assert all(isinstance(value, str) and value.strip() for value in taxonomy.values())
            seen += 1
    assert seen == 1661
