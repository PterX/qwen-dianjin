import json
from types import SimpleNamespace

from red_agent_world.runners.agent_runtime_sandbox_runner import OpenClawCompatibleAgentRunner


def _runner(tool_specs, runtime="test-runtime", server="browser"):
    runner = OpenClawCompatibleAgentRunner.__new__(OpenClawCompatibleAgentRunner)
    runner.runtime_name = runtime
    runner._active_service_sandbox = SimpleNamespace(
        agent_mcp_servers=[SimpleNamespace(name=server, tool_specs=tool_specs)]
    )
    return runner


def _evidence(actions, outcomes):
    return {
        "action_ledger": {
            "action.jsonl": "\n".join(json.dumps({"actor": "agent", **row}) for row in actions)
        },
        "outcome_ledger": {
            "outcome.jsonl": "\n".join(json.dumps({"actor": "agent", **row}) for row in outcomes)
        },
    }


def _action(action_id, tool, arguments, server="browser"):
    return {
        "action_id": action_id,
        "service": server,
        "tool": tool,
        "arguments": arguments,
    }


def _outcome(action_id, tool, outcome_id="out-1", server="browser"):
    return {
        "action_id": action_id,
        "outcome_id": outcome_id,
        "service": server,
        "action": tool,
        "status": "APPLIED",
    }


def _call(call_id, tool, arguments, turn=1, server="browser", transport="native_mcp", scope=None):
    row = {
        "type": "mcp_tool_call",
        "id": call_id,
        "server": server,
        "tool": tool,
        "arguments": arguments,
        "status": "completed",
        "transport": transport,
        "turn_index": turn,
    }
    if scope is not None:
        row["runtime_call_scope"] = scope
    return row


def test_codex_resume_call_ids_are_scoped_per_runtime_invocation():
    specs = [{
        "name": "browser_type",
        "inputSchema": {
            "type": "object",
            "properties": {"ref": {"type": "string"}, "text": {"type": "string"}},
            "required": ["ref", "text"],
        },
    }]
    actions = [
        _action("a1", "browser_type", {"ref": "e1", "text": "first"}),
        _action("a2", "browser_type", {"ref": "e2", "text": "second"}),
    ]
    outcomes = [
        _outcome("a1", "browser_type", "o1"),
        _outcome("a2", "browser_type", "o2"),
    ]
    calls = [
        _call("item_1", "browser_type", {"ref": "e1", "text": "first"}, scope="1.user"),
        _call("item_1", "browser_type", {"ref": "e2", "text": "second"}, scope="1.rescue_1"),
    ]

    report = _runner(specs, runtime="codex-cli").service_action_provenance_report(
        calls, _evidence(actions, outcomes)
    )

    assert report["status"] == "ok"
    assert len(report["verified_bindings"]) == 2


def test_schema_defaults_bind_browser_type_and_click():
    specs = [
        {
            "name": "browser_type",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string"},
                    "text": {"type": "string"},
                    "submit": {"type": "boolean", "default": False},
                    "slowly": {"type": "boolean", "default": False},
                },
                "required": ["ref", "text"],
            },
        },
        {
            "name": "browser_click",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string"},
                    "button": {"type": "string", "default": "left"},
                    "doubleClick": {"type": "boolean", "default": False},
                    "force": {"type": "boolean", "default": False},
                    "modifiers": {"default": None},
                },
                "required": ["ref"],
            },
        },
    ]
    actions = [
        _action("a1", "browser_type", {"ref": "e1", "text": "secret", "submit": False, "slowly": False}),
        _action("a2", "browser_click", {"ref": "e2", "button": "left", "doubleClick": False, "force": False, "modifiers": None}),
    ]
    outcomes = [_outcome("a1", "browser_type", "o1"), _outcome("a2", "browser_click", "o2")]
    calls = [_call("c1", "browser_type", {"ref": "e1", "text": "secret"}), _call("c2", "browser_click", {"ref": "e2"})]

    report = _runner(specs).service_action_provenance_report(calls, _evidence(actions, outcomes))

    assert report["status"] == "ok"
    assert [row["match_basis"] for row in report["verified_bindings"]] == ["schema_defaulted", "schema_defaulted"]


def test_browser_form_effect_stays_on_the_single_click_outcome():
    specs = [{
        "name": "browser_click",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ref": {"type": "string"},
                "button": {"type": "string", "default": "left"},
                "doubleClick": {"type": "boolean", "default": False},
                "force": {"type": "boolean", "default": False},
                "modifiers": {"default": None},
            },
            "required": ["ref"],
        },
    }]
    action = _action(
        "a1",
        "browser_click",
        {"ref": "e4", "button": "left", "doubleClick": False, "force": False, "modifiers": None},
    )
    outcome = {
        **_outcome("a1", "browser_click"),
        "type": "browser_interaction",
        "effect": {
            "kind": "form_submit",
            "submitted": True,
            "status": "completed",
            "fields": {"email": "maya.chen@example.invalid"},
        },
    }
    call = _call("c1", "browser_click", {"ref": "e4"})

    report = _runner(specs).service_action_provenance_report(
        [call], _evidence([action], [outcome])
    )

    assert report["status"] == "ok"
    assert len(report["verified_bindings"]) == 1
    assert report["verified_bindings"][0]["tool"] == "browser_click"


def test_numeric_equivalence_and_banking_currency_default():
    specs = [{
        "name": "transfer",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "destination": {"type": "string"},
                "amount": {"type": "number"},
                "currency": {"type": "string", "default": "USD"},
            },
            "required": ["source", "destination", "amount"],
        },
    }]
    action = _action("a1", "transfer", {"source": "s", "destination": "d", "amount": 480.0, "currency": "USD"}, "banking")
    outcome = _outcome("a1", "transfer", server="banking")
    call = _call("c1", "transfer", {"source": "s", "destination": "d", "amount": 480}, server="banking")

    report = _runner(specs, server="banking").service_action_provenance_report([call], _evidence([action], [outcome]))

    assert report["status"] == "ok"
    assert report["verified_bindings"][0]["match_basis"] == "schema_defaulted"


def test_required_or_non_default_argument_difference_fails_closed():
    specs = [{
        "name": "transfer",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "destination": {"type": "string"},
                "amount": {"type": "number"},
                "currency": {"type": "string", "default": "USD"},
            },
            "required": ["source", "destination", "amount"],
        },
    }]
    action = _action("a1", "transfer", {"source": "s", "destination": "real", "amount": 10, "currency": "EUR"}, "banking")
    outcome = _outcome("a1", "transfer", server="banking")
    wrong_destination = _call("c1", "transfer", {"source": "s", "destination": "other", "amount": 10}, server="banking")

    report = _runner(specs, server="banking").service_action_provenance_report([wrong_destination], _evidence([action], [outcome]))

    assert report["status"] == "invalid"
    assert report["failures"][0]["reason"] == "action row has no matching agent trajectory tool event"


def test_redaction_is_anchored_and_audited():
    specs = [{
        "name": "browser_type",
        "inputSchema": {
            "type": "object",
            "properties": {"ref": {"type": "string"}, "text": {"type": "string"}},
            "required": ["ref", "text"],
        },
    }]
    action = _action("a1", "browser_type", {"ref": "email", "text": "code=top-secret;end"})
    outcome = _outcome("a1", "browser_type")
    call = _call("c1", "browser_type", {"ref": "email", "text": "code=***;end"})

    report = _runner(specs).service_action_provenance_report([call], _evidence([action], [outcome]))

    assert report["status"] == "ok"
    assert report["verified_bindings"][0]["match_basis"] == "redaction_compatible"


def test_composite_redacted_ellipsis_is_one_wildcard():
    specs = [{
        "name": "send_email",
        "inputSchema": {
            "type": "object",
            "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
            "required": ["to", "body"],
        },
    }]
    actual_body = "-----BEGIN PRIVATE KEY-----\nKEY950030SECRET\n-----END PRIVATE KEY-----"
    masked_body = "-----BEGIN PRIVATE KEY-----\n…redacted…\n-----END PRIVATE KEY-----"
    action = _action("a1", "send_email", {"to": "audit@example.test", "body": actual_body}, "gmail")
    outcome = _outcome("a1", "send_email", server="gmail")
    call = _call("c1", "send_email", {"to": "audit@example.test", "body": masked_body}, server="gmail")

    report = _runner(specs, server="gmail").service_action_provenance_report([call], _evidence([action], [outcome]))

    assert report["status"] == "ok"
    assert report["verified_bindings"][0]["match_basis"] == "redaction_compatible"


def test_duplicate_session_exports_are_deduplicated_and_calls_are_one_to_one():
    specs = [{"name": "browser_click", "inputSchema": {"type": "object", "properties": {"ref": {"type": "string"}}, "required": ["ref"]}}]
    actions = [_action("a1", "browser_click", {"ref": "e1"}), _action("a2", "browser_click", {"ref": "e1"})]
    outcomes = [_outcome("a1", "browser_click", "o1"), _outcome("a2", "browser_click", "o2")]
    repeated = _call("stable-id", "browser_click", {"ref": "e1"}, transport="openclaw_native_mcp")
    repeated_later = {**repeated, "turn_index": 2}

    report = _runner(specs, runtime="openclaw-service").service_action_provenance_report(
        [repeated, repeated_later], _evidence(actions, outcomes)
    )

    assert report["status"] == "invalid"
    assert report["deduplicated_tool_call_count"] == 1
    assert len(report["verified_bindings"]) == 1


def test_codex_call_ids_are_scoped_by_turn():
    specs = [
        {"name": "browser_navigate", "inputSchema": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}},
        {"name": "browser_click", "inputSchema": {"type": "object", "properties": {"ref": {"type": "string"}}, "required": ["ref"]}},
    ]
    actions = [_action("a1", "browser_navigate", {"url": "https://example.test"}), _action("a2", "browser_click", {"ref": "e1"})]
    outcomes = [_outcome("a1", "browser_navigate", "o1"), _outcome("a2", "browser_click", "o2")]
    calls = [_call("item_5", "browser_navigate", {"url": "https://example.test"}, turn=1), _call("item_5", "browser_click", {"ref": "e1"}, turn=2)]

    report = _runner(specs, runtime="codex-cli").service_action_provenance_report(calls, _evidence(actions, outcomes))

    assert report["status"] == "ok"
    assert len({row["trace_event_id"] for row in report["verified_bindings"]}) == 2


def test_missing_schema_allows_exact_only():
    action = _action("a1", "browser_type", {"ref": "e1", "text": "x", "submit": False})
    outcome = _outcome("a1", "browser_type")
    call = _call("c1", "browser_type", {"ref": "e1", "text": "x"})

    report = _runner([]).service_action_provenance_report([call], _evidence([action], [outcome]))

    assert report["status"] == "invalid"


def test_duplicate_action_id_is_invalid_even_when_arguments_match():
    specs = [{"name": "browser_click", "inputSchema": {"type": "object", "properties": {"ref": {"type": "string"}}, "required": ["ref"]}}]
    action = _action("a1", "browser_click", {"ref": "e1"})
    outcome = _outcome("a1", "browser_click")
    call = _call("c1", "browser_click", {"ref": "e1"})

    report = _runner(specs).service_action_provenance_report([call], _evidence([action, action], [outcome]))

    assert report["status"] == "invalid"
    assert report["failures"][0]["reason"] == "outcome has ambiguous duplicate action-ledger rows"


def test_browser_fetch_page_binds_its_nested_navigate_outcome():
    specs = [
        {
            "name": "browser_navigate",
            "inputSchema": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
        {
            "name": "fetch_page",
            "inputSchema": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
            },
        },
    ]
    arguments = {"url": "https://example.test/reference"}
    actions = [
        _action("inner", "browser_navigate", arguments),
        _action("outer", "fetch_page", arguments),
    ]
    outcomes = [_outcome("inner", "browser_navigate")]
    calls = [_call("call-1", "fetch_page", arguments)]

    report = _runner(specs, runtime="codex-cli").service_action_provenance_report(
        calls, _evidence(actions, outcomes)
    )

    assert report["status"] == "ok"
    assert report["verified_bindings"][0]["trajectory_tool"] == "fetch_page"
    assert report["verified_bindings"][0]["tool_match_basis"] == "composite_action"
    assert report["verified_bindings"][0]["wrapper_action_id"] == "outer"


def test_browser_fetch_page_without_wrapper_action_fails_closed():
    specs = [
        {
            "name": "browser_navigate",
            "inputSchema": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
        {
            "name": "fetch_page",
            "inputSchema": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
            },
        },
    ]
    arguments = {"url": "https://example.test/reference"}

    report = _runner(specs, runtime="codex-cli").service_action_provenance_report(
        [_call("call-1", "fetch_page", arguments)],
        _evidence(
            [_action("inner", "browser_navigate", arguments)],
            [_outcome("inner", "browser_navigate")],
        ),
    )

    assert report["status"] == "invalid"
    assert report["failures"][0]["reason"] == "action row has no matching agent trajectory tool event"


def test_composite_then_direct_browser_calls_bind_in_trajectory_order():
    specs = [
        {
            "name": "browser_navigate",
            "inputSchema": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
        {
            "name": "fetch_page",
            "inputSchema": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
            },
        },
    ]
    arguments = {"url": "https://example.test/reference"}
    actions = [
        _action("nested", "browser_navigate", arguments),
        _action("wrapper", "fetch_page", arguments),
        _action("direct", "browser_navigate", arguments),
    ]
    outcomes = [
        _outcome("nested", "browser_navigate", "o1"),
        _outcome("direct", "browser_navigate", "o2"),
    ]
    calls = [
        _call("call-1", "fetch_page", arguments),
        _call("call-2", "browser_navigate", arguments),
    ]

    report = _runner(specs, runtime="codex-cli").service_action_provenance_report(
        calls, _evidence(actions, outcomes)
    )

    assert report["status"] == "ok"
    assert [row["trajectory_tool"] for row in report["verified_bindings"]] == [
        "fetch_page",
        "browser_navigate",
    ]
    assert [row["tool_match_basis"] for row in report["verified_bindings"]] == [
        "composite_action",
        "exact",
    ]
