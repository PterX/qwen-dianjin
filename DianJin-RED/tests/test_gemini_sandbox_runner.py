import argparse
import json
from types import SimpleNamespace

from red_agent_world.runners.gemini_sandbox_runner import GeminiNativeRunner


def _runner():
    runner = GeminiNativeRunner.__new__(GeminiNativeRunner)
    runner.runtime_args = argparse.Namespace(
        gemini_model=None,
        gemini_reasoning_effort=None,
        gemini_max_tool_rounds=None,
        gemini_max_repeated_tool_calls=None,
        gemini_max_total_tokens=None,
        gemini_max_output_tokens=None,
        gemini_request_timeout=None,
    )
    runner.config = {
        "agent": {
            "model": "gemini-3-flash-preview",
            "base_url": "https://example.invalid/v1",
        },
        "agent_runtimes": {"gemini": {}},
    }
    runner._active_service_sandbox_var = SimpleNamespace(get=lambda: None)
    return runner


def test_usage_accumulates_reasoning_and_cached_tokens():
    state = {
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
            "cached_input_tokens": 0,
        },
        "api_call_count": 0,
    }
    record = GeminiNativeRunner._accumulate_usage(
        state,
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "completion_tokens_details": {"reasoning_tokens": 7},
            "prompt_tokens_details": {"cached_tokens": 60},
        },
    )
    assert record["cumulative_usage"] == {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
        "reasoning_tokens": 7,
        "cached_input_tokens": 60,
    }


def test_workspace_path_stays_inside_workspace():
    assert GeminiNativeRunner._safe_workspace_path("README.md") == "/workspace/README.md"
    assert GeminiNativeRunner._safe_workspace_path("/workspace/docs/a.md") == "/workspace/docs/a.md"
    for value in ("../etc/passwd", "/etc/passwd"):
        try:
            GeminiNativeRunner._safe_workspace_path(value)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe path was accepted: %s" % value)


def test_service_tool_specs_use_exact_mcp_identity():
    runner = _runner()
    server = SimpleNamespace(
        name="gmail",
        tools=["send_email"],
        tool_specs=[
            {
                "name": "send_email",
                "description": "Send mail",
                "inputSchema": {
                    "type": "object",
                    "properties": {"to": {"type": "string"}},
                    "required": ["to"],
                },
            }
        ],
    )
    runner._active_service_sandbox_var = SimpleNamespace(
        get=lambda: SimpleNamespace(agent_mcp_servers=[server])
    )
    names = [row["function"]["name"] for row in runner._tool_specs()]
    assert "mcp__gmail__send_email" in names
    assert runner.service_tool_identity("mcp__gmail__send_email") == ("gmail", "send_email")


def test_tool_fingerprint_ignores_provider_signature_and_call_id():
    first = {
        "id": "call_1",
        "thoughtSignature": "one",
        "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
    }
    second = {
        "id": "call_2",
        "thoughtSignature": "two",
        "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
    }
    assert GeminiNativeRunner._tool_fingerprint(first) == GeminiNativeRunner._tool_fingerprint(second)


def test_mcp_result_for_model_unwraps_json_text_content():
    result = {
        "content": [
            {
                "type": "text",
                "text": '{"ok":true,"backend":"mailpit","messages":[]}',
            }
        ],
        "isError": False,
    }
    assert GeminiNativeRunner._mcp_result_for_model(result) == {
        "ok": True,
        "backend": "mailpit",
        "messages": [],
    }


def test_mcp_result_for_model_prefers_structured_content():
    result = {
        "content": [{"type": "text", "text": "fallback"}],
        "structuredContent": {"ok": True, "message_id": "m1"},
        "isError": False,
    }
    assert GeminiNativeRunner._mcp_result_for_model(result) == {
        "ok": True,
        "message_id": "m1",
    }


def test_mcp_result_for_model_marks_call_tool_error():
    result = {
        "content": [{"type": "text", "text": "recipient is required"}],
        "isError": True,
    }
    assert GeminiNativeRunner._mcp_result_for_model(result) == {
        "ok": False,
        "error": "recipient is required",
    }


def test_tool_specs_use_live_mcp_description_and_schema():
    runner = _runner()
    specs = runner._tool_specs(
        [
            {
                "name": "gmail",
                "tools": [
                    {
                        "name": "send_email",
                        "description": "live mutated description",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"recipient": {"type": "string"}},
                            "required": ["recipient"],
                        },
                    }
                ],
            }
        ]
    )
    spec = next(
        row for row in specs if row["function"]["name"] == "mcp__gmail__send_email"
    )
    assert spec["function"]["description"] == "live mutated description"
    assert spec["function"]["parameters"]["required"] == ["recipient"]


def test_live_mcp_discovery_uses_post_control_tools_list():
    runner = _runner()
    server = SimpleNamespace(
        name="gmail",
        url="http://gmail.invalid/mcp",
        tools=["send_email"],
    )
    runner._active_service_sandbox_var = SimpleNamespace(
        get=lambda: SimpleNamespace(agent_mcp_servers=[server])
    )
    state = {"mcp_sessions": {}}

    async def ensure_session(_state, endpoint):
        assert endpoint is server
        return "session-1"

    def post(url, payload, session_id, timeout):
        assert url == server.url
        assert payload["method"] == "tools/list"
        return {
            "jsonrpc": "2.0",
            "id": payload["id"],
            "result": {
                "tools": [
                    {
                        "name": "send_email",
                        "description": "post-control description",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"to": {"type": "string"}},
                            "required": ["to"],
                        },
                    }
                ]
            },
        }, session_id

    runner._ensure_mcp_session = ensure_session
    runner._mcp_post_sync = post

    import asyncio

    rows = asyncio.run(runner._discover_live_mcp_tool_specs(state))
    assert rows[0]["tools"][0]["description"] == "post-control description"
    assert state["mcp_sessions"][server.url] == "session-1"


def test_chat_payload_replays_thought_signature_exactly(monkeypatch):
    runner = _runner()
    runner.key_pool = SimpleNamespace(get_key=lambda: None)
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
                "thoughtSignature": "opaque-provider-signature",
            }
        ],
    }
    state = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "read"},
            assistant,
            {"role": "tool", "tool_call_id": "call_1", "content": "contents"},
        ]
    }
    captured = {}

    def fake_post(url, api_key, payload, timeout):
        captured.update(json.loads(json.dumps(payload)))
        return {
            "choices": [{"message": {"role": "assistant", "content": "done"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr(runner, "_post_json_sync", fake_post)

    async def key():
        return "test-key"

    runner._api_key = key

    import asyncio

    asyncio.run(runner._chat(state, tools=[]))
    replayed = captured["messages"][2]["tool_calls"][0]
    assert replayed["thoughtSignature"] == "opaque-provider-signature"
    assert replayed["id"] == "call_1"
    assert captured["max_completion_tokens"] == 4096


def test_chat_omits_output_limit_when_config_is_null(monkeypatch):
    runner = _runner()
    runner.config["agent_runtimes"]["gemini"]["max_output_tokens"] = None
    runner.key_pool = SimpleNamespace(get_key=lambda: None)
    captured = {}

    def fake_post(url, api_key, payload, timeout):
        captured.update(json.loads(json.dumps(payload)))
        return {
            "choices": [{"message": {"role": "assistant", "content": "done"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr(runner, "_post_json_sync", fake_post)

    async def key():
        return "test-key"

    runner._api_key = key

    import asyncio

    asyncio.run(runner._chat({"messages": [{"role": "user", "content": "test"}]}, tools=[]))
    assert "max_completion_tokens" not in captured
    assert runner._max_output_tokens() is None


def test_empty_guardrail_final_becomes_auditable_incomplete_message():
    runner = _runner()
    state = {
        "session_id": "gemini_test",
        "messages": [{"role": "system", "content": "system"}],
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
            "cached_input_tokens": 0,
        },
        "api_call_count": 0,
    }

    async def empty_chat(_state, **_kwargs):
        return {"role": "assistant", "content": ""}, {
            "prompt_tokens": 10,
            "completion_tokens": 1,
            "total_tokens": 11,
        }

    runner._chat = empty_chat

    import asyncio

    records, text = asyncio.run(runner._finalize_after_guardrail(state, "tool_round_budget"))
    assert "not claiming" in text
    assert records[-1]["completion_status"] == "incomplete"
    assert state["messages"][-1]["content"] == text
