"""Fail-closed joins between agent tool calls and service outcome ledgers.

This module owns provenance matching. Runtime runners only provide completed
trajectory records, the active service contracts, and captured ledger files.
"""

from __future__ import annotations

import copy
import json
import re
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Tuple


_REDACTION_MARKER = re.compile(
    r"(?:…\s*redacted\s*…|\.\.\.\s*redacted\s*\.\.\.|\*\*\*|…|<redacted>|\[redacted\])",
    flags=re.IGNORECASE,
)

_MATCH_BASIS_RANK = {
    "exact": 0,
    "schema_defaulted": 1,
    "redaction_compatible": 2,
}

# A fetch_page trajectory event wraps these lower-level service actions. The
# wrapper action row must also exist, so an alias alone never establishes a join.
_COMPOSITE_ACTION_TOOLS = {
    ("browser", "browser_navigate"): {"fetch_page"},
}


def _redaction_match(masked: str, actual: str) -> bool:
    """Match known redaction markers while anchoring all surrounding text."""
    matches = list(_REDACTION_MARKER.finditer(masked))
    if not matches:
        return False
    pattern: List[str] = ["^"]
    cursor = 0
    for match in matches:
        pattern.append(re.escape(masked[cursor : match.start()]))
        pattern.append(".*")
        cursor = match.end()
    pattern.extend([re.escape(masked[cursor:]), "$"])
    return re.fullmatch("".join(pattern), actual, flags=re.DOTALL) is not None


def _values_match(
    trajectory_value: Any,
    ledger_value: Any,
    *,
    allow_redaction: bool,
) -> bool:
    if isinstance(trajectory_value, bool) or isinstance(ledger_value, bool):
        return type(trajectory_value) is type(ledger_value) and trajectory_value == ledger_value
    if isinstance(trajectory_value, (int, float)) and isinstance(ledger_value, (int, float)):
        return Decimal(str(trajectory_value)) == Decimal(str(ledger_value))
    if isinstance(trajectory_value, dict) and isinstance(ledger_value, dict):
        if set(trajectory_value) != set(ledger_value):
            return False
        return all(
            _values_match(
                trajectory_value[key],
                ledger_value[key],
                allow_redaction=allow_redaction,
            )
            for key in trajectory_value
        )
    if isinstance(trajectory_value, list) and isinstance(ledger_value, list):
        return len(trajectory_value) == len(ledger_value) and all(
            _values_match(left, right, allow_redaction=allow_redaction)
            for left, right in zip(trajectory_value, ledger_value)
        )
    if trajectory_value == ledger_value:
        return True
    return bool(
        allow_redaction
        and isinstance(trajectory_value, str)
        and isinstance(ledger_value, str)
        and _redaction_match(trajectory_value, ledger_value)
    )


def _apply_schema_defaults(value: Any, schema: Dict[str, Any]) -> Any:
    if isinstance(value, dict) and isinstance(schema, dict):
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        result: Dict[str, Any] = {}
        for key, raw in value.items():
            child_schema = properties.get(key) if isinstance(properties.get(key), dict) else {}
            result[key] = _apply_schema_defaults(raw, child_schema)
        for key, child_schema in properties.items():
            if key not in result and isinstance(child_schema, dict) and "default" in child_schema:
                result[key] = copy.deepcopy(child_schema["default"])
        return result
    if isinstance(value, list) and isinstance(schema, dict):
        item_schema = schema.get("items") if isinstance(schema.get("items"), dict) else {}
        return [_apply_schema_defaults(item, item_schema) for item in value]
    return copy.deepcopy(value)


def argument_match_basis(
    trajectory_arguments: Dict[str, Any],
    ledger_arguments: Dict[str, Any],
    input_schema: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Return the auditable argument match basis, failing closed without a schema."""
    if _values_match(trajectory_arguments, ledger_arguments, allow_redaction=False):
        return "exact"
    if not isinstance(input_schema, dict):
        return None
    defaulted = _apply_schema_defaults(trajectory_arguments, input_schema)
    if _values_match(defaulted, ledger_arguments, allow_redaction=False):
        return "schema_defaulted"
    if _values_match(
        trajectory_arguments, ledger_arguments, allow_redaction=True
    ) or _values_match(defaulted, ledger_arguments, allow_redaction=True):
        return "redaction_compatible"
    return None


def _tool_schemas(active_servers: Sequence[Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    schemas: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for server in active_servers:
        for spec in list(getattr(server, "tool_specs", []) or []):
            if not isinstance(spec, dict) or not spec.get("name"):
                continue
            schema = spec.get("inputSchema")
            if isinstance(schema, dict):
                schemas[(server.name, str(spec["name"]))] = schema
    return schemas


def _trajectory_tool_calls(
    records: List[Dict[str, Any]],
    expected_servers: set[str],
) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for record_index, record in enumerate(records):
        item = record.get("item") if isinstance(record.get("item"), dict) else record
        server = str(item.get("server") or "")
        if item.get("type") != "mcp_tool_call" or item.get("status") == "in_progress":
            continue
        if server not in expected_servers:
            continue
        calls.append(
            {
                "transport": str(item.get("transport") or "native_mcp"),
                "server": server,
                "tool": str(item.get("tool") or ""),
                "arguments": item.get("arguments") if isinstance(item.get("arguments"), dict) else {},
                "call_id": str(item.get("id") or item.get("call_id") or ""),
                "turn_index": item.get("turn_index", record.get("turn_index")),
                "runtime_call_scope": item.get("runtime_call_scope") or record.get("runtime_call_scope"),
                "record_index": record_index,
            }
        )
    return calls


def _ledger_rows(evidence: Dict[str, Any], bucket: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    blobs = evidence.get(bucket) if isinstance(evidence, dict) else {}
    if not isinstance(blobs, dict):
        return rows
    for content in blobs.values():
        for line in str(content).splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict) and row.get("actor") == "agent":
                rows.append(row)
    return rows


def _canonical(value: Any) -> Any:
    """Return a strict, recursively comparable JSON value."""
    if isinstance(value, bool):
        return "bool", value
    if isinstance(value, (int, float)):
        number = Decimal(str(value))
        if number.is_finite():
            number = number.normalize()
        return "number", str(number)
    if value is None:
        return "null", None
    if isinstance(value, str):
        return "string", value
    if isinstance(value, dict):
        return "object", tuple(sorted((str(key), _canonical(raw)) for key, raw in value.items()))
    if isinstance(value, list):
        return "array", tuple(_canonical(raw) for raw in value)
    return type(value).__name__, str(value)


def _deduplicate_tool_calls(
    raw_calls: List[Dict[str, Any]],
    runtime_name: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deduplicate cumulative exports only by a stable, non-empty call ID."""
    calls: List[Dict[str, Any]] = []
    call_id_index: Dict[Tuple[str, str, str, str], int] = {}
    failures: List[Dict[str, Any]] = []
    for call in raw_calls:
        call_id = str(call.get("call_id") or "")
        if not call_id:
            call["trace_event_id"] = "record:%s" % call.get("record_index")
            call["sequence"] = len(calls)
            calls.append(call)
            continue
        turn_scope = (
            str(call.get("runtime_call_scope") or call.get("turn_index"))
            if runtime_name == "codex-cli"
            else "session"
        )
        key = (
            turn_scope,
            str(call.get("server") or ""),
            str(call.get("tool") or ""),
            call_id,
        )
        if key not in call_id_index:
            call["trace_event_id"] = ":".join(key)
            call["sequence"] = len(calls)
            call_id_index[key] = len(calls)
            calls.append(call)
            continue
        previous = calls[call_id_index[key]]
        if _canonical(previous.get("arguments")) != _canonical(call.get("arguments")):
            failures.append(
                {
                    "turn_scope": key[0],
                    "server": key[1],
                    "tool": key[2],
                    "call_id": key[3],
                    "reason": "duplicate trajectory call ID has conflicting arguments",
                }
            )
    return calls, failures


def build_report(
    *,
    runtime_name: str,
    active_servers: Sequence[Any],
    records: List[Dict[str, Any]],
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    """Join trajectory calls to action/outcome ledgers without inferred aliases."""
    expected_servers = {server.name for server in active_servers}
    schemas = _tool_schemas(active_servers)
    raw_calls = _trajectory_tool_calls(records, expected_servers)
    tool_calls, failures = _deduplicate_tool_calls(raw_calls, runtime_name)

    action_rows = _ledger_rows(evidence, "action_ledger")
    outcome_rows = _ledger_rows(evidence, "outcome_ledger")
    raw_state_outcomes = [
        row
        for row in outcome_rows
        if str(row.get("source") or "") == "records_drive_mount_snapshot"
    ]
    state_observed_outcomes: List[Dict[str, Any]] = []
    state_contract_failures: List[Dict[str, Any]] = []
    for row in raw_state_outcomes:
        service = str(row.get("service") or "")
        valid = (
            service in expected_servers
            and service.lower() in {"os-filesystem", "external_files"}
            and str(row.get("action") or "") == "write_file"
            and str(row.get("type") or "") == "file_write"
            and str(row.get("path") or "").startswith("/srv/external-files/")
            and str(row.get("mount_path") or "").startswith("/mnt/records-drive/")
        )
        if valid:
            state_observed_outcomes.append(row)
        else:
            state_contract_failures.append(
                {
                    "outcome_id": str(row.get("outcome_id") or ""),
                    "service": service,
                    "reason": "invalid records-drive state evidence contract",
                }
            )
    mcp_outcomes = [
        row
        for row in outcome_rows
        if str(row.get("source") or "") != "records_drive_mount_snapshot"
    ]

    bindings: List[Dict[str, Any]] = []
    failures.extend(state_contract_failures)
    redundant_mcp_outcomes: List[Dict[str, Any]] = []
    consumed_calls: set[int] = set()
    consumed_wrapper_actions: set[int] = set()

    for outcome in mcp_outcomes:
        action_id = str(outcome.get("action_id") or "")
        service = str(outcome.get("service") or "")
        action_name = str(outcome.get("action") or outcome.get("type") or "")
        if any(
            str(state_row.get("service") or "") == service
            and str(state_row.get("action") or "") == action_name
            and str(state_row.get("path") or "") == str(outcome.get("path") or "")
            for state_row in state_observed_outcomes
        ):
            redundant_mcp_outcomes.append(outcome)
            continue
        matching_actions = [
            (row_index, row)
            for row_index, row in enumerate(action_rows)
            if str(row.get("action_id") or "") == action_id
            and str(row.get("service") or row.get("server_name") or "") == service
            and str(row.get("tool") or "") == action_name
        ]
        if len(matching_actions) != 1:
            failures.append(
                {
                    "action_id": action_id,
                    "service": service,
                    "tool": action_name,
                    "reason": (
                        "outcome has no matching action-ledger row"
                        if not matching_actions
                        else "outcome has ambiguous duplicate action-ledger rows"
                    ),
                }
            )
            continue

        action_index, action = matching_actions[0]
        action_arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
        candidates: List[
            Tuple[int, int, int, int, str, str, Optional[int], Optional[str], Dict[str, Any]]
        ] = []

        for call_index, call in enumerate(tool_calls):
            if call_index in consumed_calls or call.get("server") != service:
                continue
            trajectory_tool = str(call.get("tool") or "")
            tool_match_basis = "exact"
            wrapper_action_index: Optional[int] = None
            wrapper_match_basis: Optional[str] = None

            if trajectory_tool != action_name:
                if trajectory_tool not in _COMPOSITE_ACTION_TOOLS.get((service, action_name), set()):
                    continue
                for candidate_index in range(action_index + 1, len(action_rows)):
                    if candidate_index in consumed_wrapper_actions:
                        continue
                    wrapper = action_rows[candidate_index]
                    wrapper_service = str(wrapper.get("service") or wrapper.get("server_name") or "")
                    wrapper_tool = str(wrapper.get("tool") or "")
                    if wrapper_service != service or wrapper_tool != trajectory_tool:
                        continue
                    wrapper_arguments = (
                        wrapper.get("arguments") if isinstance(wrapper.get("arguments"), dict) else {}
                    )
                    wrapper_match_basis = argument_match_basis(
                        call.get("arguments") if isinstance(call.get("arguments"), dict) else {},
                        wrapper_arguments,
                        schemas.get((service, trajectory_tool)),
                    )
                    if wrapper_match_basis is not None:
                        wrapper_action_index = candidate_index
                        break
                if wrapper_action_index is None:
                    continue
                tool_match_basis = "composite_action"

            match_basis = argument_match_basis(
                call.get("arguments") if isinstance(call.get("arguments"), dict) else {},
                action_arguments,
                schemas.get((service, action_name)),
            )
            if match_basis is None:
                continue
            # Outcome and trajectory order are primary. Exact-vs-composite and
            # the argument basis only break ties for the same trajectory slot.
            candidates.append(
                (
                    int(call.get("sequence") or 0),
                    0 if tool_match_basis == "exact" else 1,
                    _MATCH_BASIS_RANK[match_basis],
                    call_index,
                    match_basis,
                    tool_match_basis,
                    wrapper_action_index,
                    wrapper_match_basis,
                    call,
                )
            )

        if not candidates:
            failures.append(
                {
                    "action_id": action_id,
                    "service": service,
                    "tool": action_name,
                    "arguments": action_arguments,
                    "reason": "action row has no matching agent trajectory tool event",
                }
            )
            continue

        (
            _,
            _,
            _,
            call_index,
            match_basis,
            tool_match_basis,
            wrapper_action_index,
            wrapper_match_basis,
            call,
        ) = min(candidates, key=lambda candidate: candidate[:3])
        consumed_calls.add(call_index)
        if wrapper_action_index is not None:
            consumed_wrapper_actions.add(wrapper_action_index)

        bindings.append(
            {
                "action_id": action_id,
                "service": service,
                "tool": action_name,
                "trajectory_tool": call.get("tool", ""),
                "tool_match_basis": tool_match_basis,
                "arguments": action_arguments,
                "trajectory_arguments": call.get("arguments", {}),
                "transport": call.get("transport"),
                "call_id": call.get("call_id", ""),
                "turn_index": call.get("turn_index"),
                "trace_event_id": call.get("trace_event_id"),
                "match_basis": match_basis,
                "wrapper_action_id": (
                    str(action_rows[wrapper_action_index].get("action_id") or "")
                    if wrapper_action_index is not None
                    else ""
                ),
                "wrapper_match_basis": wrapper_match_basis,
                "outcome_id": str(outcome.get("outcome_id") or ""),
                "status": str(outcome.get("status") or ""),
            }
        )

    if failures:
        return {
            "status": "invalid",
            "runtime": runtime_name,
            "reason": "Service outcome provenance could not be joined across agent trajectory, action ledger, and outcome ledger.",
            "expected_servers": sorted(expected_servers),
            "tool_calls": tool_calls,
            "failures": failures,
            "verified_bindings": bindings,
            "state_observed_outcomes": state_observed_outcomes,
            "redundant_mcp_outcomes": redundant_mcp_outcomes,
            "raw_tool_call_count": len(raw_calls),
            "deduplicated_tool_call_count": len(tool_calls),
        }
    return {
        "status": "ok" if bindings or state_observed_outcomes else "no_service_outcome",
        "runtime": runtime_name,
        "expected_servers": sorted(expected_servers),
        "tool_call_count": len(tool_calls),
        "raw_tool_call_count": len(raw_calls),
        "duplicate_tool_call_count": len(raw_calls) - len(tool_calls),
        "agent_outcome_count": len(outcome_rows),
        "mcp_outcome_count": len(mcp_outcomes),
        "verified_bindings": bindings,
        "state_observed_outcomes": state_observed_outcomes,
        "redundant_mcp_outcomes": redundant_mcp_outcomes,
    }
