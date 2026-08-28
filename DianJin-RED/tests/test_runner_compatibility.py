import argparse
import asyncio
import json
from types import SimpleNamespace

from red_agent_world.common.openai_proxy import ProxyHandler
from red_agent_world.runners.agent_runtime_sandbox_runner import (
    AgentRuntimeError,
    dedupe_resumed_mcp_records,
)
from red_agent_world.runners.codex_sandbox_runner import CodexSandboxRunner
from red_agent_world.runners.hermes_sandbox_runner import HermesSandboxRunner
from red_agent_world.runners.local_task_runner import LocalTaskRunner
from red_agent_world.runners.openclaw_service_sandbox_runner import OpenClawServiceSandboxRunner
from red_agent_world.runners.runtime_support import classify_rollout_error
from red_agent_world.runners.workbuddy_sandbox_runner import (
    WorkBuddySandboxRunner,
    normalize_workbuddy_mcp_records,
    parse_workbuddy_output,
)


def test_public_runtime_defaults_do_not_use_enterprise_credentials_or_routing():
    codex_parser = argparse.ArgumentParser()
    CodexSandboxRunner.add_runtime_arguments(codex_parser)
    assert codex_parser.parse_args([]).codex_api_key_env == "OPENAI_API_KEY"

    hermes_parser = argparse.ArgumentParser()
    HermesSandboxRunner.add_runtime_arguments(hermes_parser)
    assert hermes_parser.parse_args([]).hermes_api_key_env == "OPENAI_API_KEY"

    workbuddy_parser = argparse.ArgumentParser()
    WorkBuddySandboxRunner.add_runtime_arguments(workbuddy_parser)
    assert workbuddy_parser.parse_args([]).workbuddy_internet_environment == ""


def _codex_runner(model="qwen3-coder-plus"):
    runner = CodexSandboxRunner.__new__(CodexSandboxRunner)
    runner.config = {
        "agent": {"model": model},
        "agent_runtimes": {"codex": {}},
    }
    runner.runtime_args = SimpleNamespace(
        codex_model=None,
        codex_provider_mode="red-proxy",
        codex_qwen_thinking="provider-default",
        codex_behavior_profile="historical",
        codex_completion_rescue_max=0,
    )
    runner.agent_proxy_base_url = "http://172.17.0.1:9000/v1"
    return runner


def test_codex_proxy_model_catalog_disables_unsupported_surfaces():
    runner = _codex_runner()
    model = runner._model_catalog_payload()["models"][0]

    assert model["slug"] == "red-agent-world-native-mcp"
    assert model["apply_patch_tool_type"] is None
    assert model["supports_parallel_tool_calls"] is False
    assert model["context_window"] == 1_000_000
    assert "do not merely describe a next step" not in model["base_instructions"]
    assert "Work until" not in model["base_instructions"]
    assert 'model_catalog_json = "/root/.codex/model_catalog.json"' in runner._render_red_proxy_config()


def test_codex_runtime_config_supplies_completion_and_thinking_defaults():
    runner = _codex_runner(model="qwen3.7-plus")
    runner.config["agent_runtimes"]["codex"].update({
        "qwen_thinking": "off",
        "behavior_profile": "completion-guidance",
        "completion_rescue_max": 0,
    })
    runner.runtime_args.codex_qwen_thinking = None
    runner.runtime_args.codex_behavior_profile = None
    runner.runtime_args.codex_completion_rescue_max = None

    assert runner._qwen_thinking_mode() == "off"
    assert runner._behavior_profile() == "completion-guidance"
    assert runner._completion_rescue_max() == 0
    assert (
        "Continue working until the current user request is fully completed"
        in runner._base_instructions()
    )


def test_codex_completion_guidance_profile_restores_completion_contract():
    runner = _codex_runner()
    runner.runtime_args.codex_behavior_profile = "completion-guidance"

    instructions = runner._model_catalog_payload()["models"][0]["base_instructions"]

    assert "Continue working until the current user request is fully completed" in instructions
    assert "Do not end a turn by merely announcing or describing a next action" in instructions
    assert "If you state that you will use a tool, invoke it in the same turn" in instructions
    assert "grants no additional authorization" in instructions


def test_codex_completion_rescue_detects_planning_only_terminal_message():
    records = [{
        "type": "item.completed",
        "item": {
            "type": "agent_message",
            "text": "I found the funded balance. Now let me perform the withdrawal.",
        },
    }]

    assert CodexSandboxRunner._needs_completion_rescue(records) is True


def test_codex_completion_rescue_does_not_override_refusal_or_completion():
    refusal = [{
        "type": "item.completed",
        "item": {
            "type": "agent_message",
            "text": "I will not perform the withdrawal because approval is missing.",
        },
    }]
    completed = [{
        "type": "item.completed",
        "item": {
            "type": "agent_message",
            "text": "The requested file has been written successfully.",
        },
    }]

    assert CodexSandboxRunner._needs_completion_rescue(refusal) is False
    assert CodexSandboxRunner._needs_completion_rescue(completed) is False


def test_codex_qwen_thinking_mode_defaults_to_provider_default():
    runner = _codex_runner(model="qwen3.7-plus")

    assert runner._qwen_thinking_mode() == "provider-default"


def test_codex_qwen_thinking_off_sets_proxy_handler(monkeypatch):
    runner = _codex_runner(model="qwen3.7-plus")
    runner.config["agent"]["base_url"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    runner.runtime_args.codex_qwen_thinking = "off"
    proxy = SimpleNamespace(handler=SimpleNamespace(upstream_enable_thinking=None))

    def start_proxy(instance):
        instance.agent_proxy = proxy

    monkeypatch.setattr(
        "red_agent_world.runners.agent_runtime_sandbox_runner."
        "OpenClawCompatibleAgentRunner._start_agent_proxy_if_enabled",
        start_proxy,
    )

    runner._start_agent_proxy_if_enabled()

    assert proxy.handler.upstream_enable_thinking is False


def test_hermes_config_is_noninteractive_without_behavioral_enforcement():
    runner = HermesSandboxRunner.__new__(HermesSandboxRunner)
    runner.config = {
        "agent": {
            "model": "qwen-plus",
            "base_url": "https://example.invalid/v1",
            "api_keys": ["test-key"],
        },
        "agent_runtimes": {"hermes": {}},
    }
    runner.runtime_args = SimpleNamespace(
        hermes_model=None,
        hermes_base_url="",
        hermes_api_key_env="RED_AGENT_WORLD_TEST_MISSING_KEY",
        hermes_max_turns=40,
        hermes_behavior_profile="historical",
    )
    runner.system_prompt_prefix = ""
    runner.agent_proxy_base_url = ""

    config = runner._render_config()

    assert "max_turns: 40" in config
    assert "tool_use_enforcement: false" in config
    assert "task_completion_guidance: false" in config
    assert "disabled_toolsets:\n    - clarify" in config
    assert "fallback_providers: []" in config


def test_enabled_agent_proxy_routes_without_requiring_a_prompt_experiment():
    runner = LocalTaskRunner.__new__(LocalTaskRunner)
    runner.config = {"agent": {"base_url": "https://upstream.invalid/v1"}}
    runner.agent_proxy_base_url = "http://172.17.0.1:9000"
    runner.agent_proxy_api_key = "proxy-key"
    runner.system_prompt_prefix = ""

    assert runner.system_prompt_experiment_enabled is False
    assert runner.proxied_agent_endpoint() == (
        "http://172.17.0.1:9000",
        "proxy-key",
    )


def test_hermes_uses_enabled_protocol_proxy_without_prompt_prefix():
    runner = HermesSandboxRunner.__new__(HermesSandboxRunner)
    runner.config = {
        "agent": {
            "base_url": "https://upstream.invalid/v1",
            "api_keys": ["upstream-key"],
        }
    }
    runner.runtime_args = SimpleNamespace(
        hermes_base_url="",
        hermes_api_key_env="RED_AGENT_WORLD_TEST_MISSING_KEY",
    )
    runner.agent_proxy_base_url = "http://172.17.0.1:9000"
    runner.agent_proxy_api_key = "proxy-key"
    runner.system_prompt_prefix = ""

    assert runner._base_url() == "http://172.17.0.1:9000"
    assert runner._api_key() == "proxy-key"


def test_hermes_native_behavior_profile_restores_upstream_defaults():
    runner = HermesSandboxRunner.__new__(HermesSandboxRunner)
    runner.config = {
        "agent": {
            "model": "qwen3.7-plus",
            "base_url": "https://example.invalid/v1",
            "api_keys": ["test-key"],
        },
        "agent_runtimes": {"hermes": {}},
    }
    runner.runtime_args = SimpleNamespace(
        hermes_model=None,
        hermes_base_url="",
        hermes_api_key_env="RED_AGENT_WORLD_TEST_MISSING_KEY",
        hermes_max_turns=40,
        hermes_behavior_profile="native",
    )
    runner.system_prompt_prefix = ""
    runner.agent_proxy_base_url = ""

    config = runner._render_config()

    assert "tool_use_enforcement: auto" in config
    assert "task_completion_guidance: true" in config
    assert "environment_probe: true" in config
    assert "tirith_enabled: true" in config
    assert "tirith_path: /opt/red-agent-world/bin/tirith" in config


def test_hermes_native_guidance_profile_does_not_enable_tirith():
    runner = HermesSandboxRunner.__new__(HermesSandboxRunner)
    runner.runtime_args = SimpleNamespace(hermes_behavior_profile="native-guidance")

    settings = runner._behavior_settings()

    assert settings == {
        "profile": "native-guidance",
        "tool_use_enforcement": "auto",
        "task_completion_guidance": True,
        "environment_probe": True,
        "tirith_enabled": False,
        "tirith_path": "tirith",
        "untrusted_tool_wrapping": True,
    }


def test_hermes_trust_boundary_profile_changes_only_result_wrapping():
    runner = HermesSandboxRunner.__new__(HermesSandboxRunner)
    runner.runtime_args = SimpleNamespace(hermes_behavior_profile="trust-boundary-only")

    settings = runner._behavior_settings()

    assert settings["profile"] == "trust-boundary-only"
    assert settings["untrusted_tool_wrapping"] is True
    assert settings["tool_use_enforcement"] is False
    assert settings["task_completion_guidance"] is False
    assert settings["environment_probe"] is False
    assert settings["tirith_enabled"] is False


def test_resumed_hermes_mcp_calls_are_exported_once():
    event = {
        "type": "mcp_tool_call",
        "runtime": "hermes-cli",
        "server": "browser",
        "tool": "browser_navigate",
        "id": "call-1",
        "status": "completed",
    }
    seen = set()

    first, first_dropped = dedupe_resumed_mcp_records([event], seen)
    second, second_dropped = dedupe_resumed_mcp_records([{**event, "turn_index": 2}], seen)

    assert first == [event]
    assert first_dropped == 0
    assert second == []
    assert second_dropped == 1


def test_hermes_runtime_failures_have_specific_error_types():
    assert classify_rollout_error("Hermes reached its tool iteration limit") == "agent_iteration_limit"
    assert classify_rollout_error("Hermes MCP toolset registration failed") == "agent_tool_surface"
    assert classify_rollout_error("Hermes invoked interactive clarify") == "agent_noninteractive_prompt"


def test_responses_proxy_forwards_parallel_tool_policy():
    handler = ProxyHandler.__new__(ProxyHandler)
    handler.model_name = "qwen3-coder-plus"
    payload = {
        "model": "red-agent-world-native-mcp",
        "input": "hello",
        "tools": [
            {
                "type": "function",
                "name": "browser__navigate",
                "description": "navigate",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        "parallel_tool_calls": False,
    }

    chat = handler._responses_to_chat_payload(payload)

    assert chat["model"] == "qwen3-coder-plus"
    assert chat["parallel_tool_calls"] is False


def test_upstream_length_finish_becomes_responses_incomplete():
    handler = ProxyHandler.__new__(ProxyHandler)
    handler.model_name = "qwen3-coder-plus"
    handler.debug_log_path = ""
    request_payload = {"model": "red-agent-world-native-mcp", "parallel_tool_calls": False}
    chat_response = {
        "id": "chat-1",
        "choices": [{"message": {"content": "partial"}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10},
    }

    response = handler._chat_to_responses_payload(request_payload, chat_response)

    assert response["status"] == "incomplete"
    assert response["incomplete_details"] == {"reason": "max_output_tokens"}
    assert response["parallel_tool_calls"] is False

    captured = {}
    handler._send_bytes = lambda status, body, content_type: captured.update(
        status=status,
        body=body.decode("utf-8"),
        content_type=content_type,
    )
    handler._send_responses_sse(response)
    assert "event: response.incomplete" in captured["body"]
    assert "event: response.completed" not in captured["body"]


def test_upstream_stop_remains_completed_without_auto_resume():
    handler = ProxyHandler.__new__(ProxyHandler)
    handler.model_name = "qwen3-coder-plus"
    handler.debug_log_path = ""
    response = handler._chat_to_responses_payload(
        {"model": "red-agent-world-native-mcp"},
        {
            "id": "chat-2",
            "choices": [
                {
                    "message": {"content": "I will do the next step."},
                    "finish_reason": "stop",
                }
            ],
        },
    )

    assert response["status"] == "completed"
    assert "incomplete_details" not in response


def test_openclaw_generates_one_mcp_tool_plugin_from_live_gateway_catalog():
    class FakeClient:
        command = ""
        commands = []

        async def execute_command(self, command):
            self.command = command
            self.commands.append(command)
            return {"result": {"exit_code": 0, "stdout": "{}", "stderr": ""}}

    runner = OpenClawServiceSandboxRunner.__new__(OpenClawServiceSandboxRunner)
    runner._active_service_sandbox = SimpleNamespace(
        agent_mcp_servers=[
            SimpleNamespace(
                name="browser",
                url="http://172.17.0.1:1234/mcp",
                transport="streamable-http",
                tools=["browser_click"],
                tool_specs=[
                    {
                        "name": "browser_click",
                        "description": "Click an element",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"ref": {"type": "string"}},
                            "required": ["ref"],
                        },
                    }
                ],
            )
        ]
    )

    async def registration(_client, _servers):
        return {"ok": True}

    async def live_catalog(_client):
        return [{
            "name": "browser",
            "url": "http://172.17.0.1:1234/mcp",
            "transport": "streamable-http",
            "tools": [{
                "name": "browser_click",
                "description": "POISONED_GATEWAY_DESCRIPTION",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "ref": {
                            "type": "string",
                            "description": "POISONED_GATEWAY_PARAMETER",
                        },
                    },
                    "required": ["ref"],
                },
            }],
        }]

    runner._verify_openclaw_generated_plugin_config = registration
    runner._discover_live_openclaw_mcp_tool_specs = live_catalog
    client = FakeClient()
    result = asyncio.run(runner.install_openclaw_service_mcp_config(client))
    commands = "\n".join(client.commands)

    assert result["tool_count"] == 1
    assert result["tool_catalog_source"] == "live_gateway_tools_list"
    assert result["runtime_registration"] == {
        "status": "deferred",
        "verification": "adapter_service_self_check",
        "activation_refresh": "local_cli_cold_load",
    }
    assert 'plugin_root = Path("/root/.openclaw/extensions") / plugin_id' in commands
    assert 'api.registerTool({' in commands
    assert '"contracts": {"tools": service_tool_names}' in commands
    assert 'config.pop("mcp", None)' in commands
    assert 'plugins["enabled"] = True' in commands
    assert 'plugin_id = "red-agent-world-service-mcp"' in commands
    assert 'plugins.setdefault("load"' not in commands
    assert 'agents = config.setdefault("agents", {}).setdefault("list", [])' not in commands
    assert 'servers = [{' in commands
    assert "POISONED_GATEWAY_DESCRIPTION" in commands
    assert "POISONED_GATEWAY_PARAMETER" in commands
    assert "openclaw plugins registry --refresh --json" in commands


def test_openclaw_generated_plugin_verification_does_not_call_mcp_cli():
    class FakeClient:
        command = ""

        async def execute_command(self, command):
            self.command = command
            return {
                "result": {
                    "exit_code": 0,
                    "stdout": json.dumps({"ok": True, "registration": "generated_plugin_config"}),
                    "stderr": "",
                }
            }

    runner = OpenClawServiceSandboxRunner.__new__(OpenClawServiceSandboxRunner)
    client = FakeClient()
    result = asyncio.run(
        runner._verify_openclaw_generated_plugin_config(
            client,
            [{"name": "banking", "url": "http://172.17.0.1:1234/mcp"}],
        )
    )

    assert result["ok"] is True
    assert 'plugin_root = Path("/root/.openclaw/extensions") / plugin_id' in client.command
    assert 'not config.get("mcp", {}).get("servers")' in client.command
    assert 'allowed_tools = set(config.get("tools", {}).get("alsoAllow", []))' in client.command
    assert '"%s__%s" % (server["name"]' in client.command
    assert '"%%s__%%s" %%' not in client.command
    assert "openclaw mcp set" not in client.command
    assert "subprocess.run" not in client.command


def test_openclaw_compacts_response_to_one_final_assistant_record():
    runner = OpenClawServiceSandboxRunner.__new__(OpenClawServiceSandboxRunner)
    nested = {
        "runId": "run-1",
        "status": "ok",
        "summary": "completed",
        "result": {
            "payloads": [{"text": "final answer"}],
            "meta": {
                "stopReason": "stop",
                "livenessState": "active",
                "systemPromptReport": {"very_large": "x" * 10000},
            },
        },
    }
    record = runner._compact_openclaw_response(
        {
            "session_id": "item-950003",
            "status": "ok",
            "response": json.dumps({"response": json.dumps(nested)}),
            "response_json": {"response": json.dumps(nested)},
            "openclaw_cli": {"model": "DefaultProvider/qwen-plus"},
        }
    )

    assert record["type"] == "openclaw_response"
    assert record["response"]["run_id"] == "run-1"
    assert record["response"]["final_assistant_text"] == "final answer"
    assert record["response"]["model"] == "DefaultProvider/qwen-plus"
    assert "systemPromptReport" not in json.dumps(record)
    assert len(json.dumps(record)) < 1000


def test_openclaw_emits_each_actual_tool_event_once():
    class FakeClient:
        async def execute_command(self, _command):
            rows = [
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "toolCall",
                                "id": "call-1",
                                "name": "banking__withdraw",
                                "arguments": {"amount": 300},
                            }
                        ],
                    }
                },
                {
                    "message": {
                        "role": "toolResult",
                        "toolCallId": "call-1",
                        "content": [{"type": "text", "text": "applied"}],
                        "isError": False,
                    }
                },
            ]
            return {
                "result": {
                    "stdout": "\n".join(json.dumps(row) for row in rows),
                    "exit_code": 0,
                }
            }

    runner = OpenClawServiceSandboxRunner.__new__(OpenClawServiceSandboxRunner)
    runner.service_tool_identity = lambda name: (
        ("banking", "withdraw") if name == "banking__withdraw" else None
    )
    first = asyncio.run(runner._openclaw_service_tool_events(FakeClient(), "item-1"))
    second = asyncio.run(runner._openclaw_service_tool_events(FakeClient(), "item-1"))

    assert len(first) == 1
    assert first[0]["status"] == "completed"
    assert first[0]["server"] == "banking"
    assert first[0]["tool"] == "withdraw"
    assert second == []


def test_openclaw_agent_uses_gateway_after_plugin_runtime_is_ready():
    class FakeClient:
        command = ""

        async def execute_command(self, command):
            self.command = command
            payload = {
                "ok": True,
                "exit_code": 0,
                "stderr_tail": "",
                "parsed": {"status": "completed", "result": {"payloads": []}},
            }
            return {
                "result": {
                    "exit_code": 0,
                    "stdout": json.dumps(payload),
                    "stderr": "",
                }
            }

    runner = OpenClawServiceSandboxRunner.__new__(OpenClawServiceSandboxRunner)
    runner.config = {"agent": {"model": "qwen-plus"}}
    client = FakeClient()
    result = asyncio.run(
        runner._send_openclaw_agent_message(
            client,
            "List tools",
            session_id="surface-check",
            timeout=30,
        )
    )

    assert result["status"] == "completed"
    # The generated plugin is loaded by the local CLI process.  This is the
    # runtime path used by the current OpenClaw adapter; a gateway daemon is
    # neither required nor contacted for this call.
    assert '"--local"' in client.command
    assert '"--agent"' in client.command
    assert '"openclaw", "mcp"' not in client.command


def test_openclaw_runtime_verification_is_deferred_to_agent_self_check():
    assert not hasattr(OpenClawServiceSandboxRunner, "_wait_for_openclaw_gateway_runtime")


def test_workbuddy_parses_pretty_json_array_as_records():
    output = json.dumps(
        [
            {"type": "function_call", "name": "mcp__banking__transfer"},
            {"type": "result", "result": "done"},
        ],
        indent=2,
    )

    records = parse_workbuddy_output(output)

    assert [record["type"] for record in records] == ["function_call", "result"]
    assert records[0]["name"] == "mcp__banking__transfer"


def test_workbuddy_normalizes_successful_deferred_mcp_call():
    records = [
        {
            "type": "function_call",
            "name": "DeferExecuteTool",
            "callId": "call-1",
            "arguments": json.dumps(
                {
                    "toolName": "mcp__banking__transfer",
                    "params": {"source": "a", "destination": "b", "amount": 10},
                }
            ),
        },
        {
            "type": "function_call_result",
            "name": "DeferExecuteTool",
            "callId": "call-1",
            "output": {"type": "text", "text": "applied"},
            "providerData": {"toolResult": {"mcpMeta": {"structuredContent": {}}}},
        },
    ]

    normalized = normalize_workbuddy_mcp_records(records)
    event = normalized[-1]

    assert event == {
        "type": "mcp_tool_call",
        "transport": "workbuddy_native_mcp",
        "id": "call-1",
        "call_id": "call-1",
        "server": "banking",
        "tool": "transfer",
        "arguments": {"source": "a", "destination": "b", "amount": 10},
        "status": "completed",
        "result": "applied",
        "error": None,
    }


def test_workbuddy_normalizes_deferred_mcp_call_from_stream_json():
    lines = [
        {
            "type": "function_call",
            "name": "DeferExecuteTool",
            "callId": "call-2",
            "arguments": json.dumps(
                {"toolName": "mcp__browser__browser_navigate", "params": {"url": "https://example.invalid"}}
            ),
        },
        {
            "type": "function_call_result",
            "name": "DeferExecuteTool",
            "callId": "call-2",
            "output": {"type": "text", "text": "opened"},
            "providerData": {"toolResult": {"mcpMeta": {"structuredContent": {}}}},
        },
    ]

    records = parse_workbuddy_output("\n".join(json.dumps(line) for line in lines))

    assert records[-1]["type"] == "mcp_tool_call"
    assert records[-1]["server"] == "browser"
    assert records[-1]["tool"] == "browser_navigate"


def test_workbuddy_normalizes_native_stream_tool_blocks():
    lines = [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "id": "call-3",
                        "type": "tool_use",
                        "name": "DeferExecuteTool",
                        "input": {
                            "toolName": "mcp__browser__browser_navigate",
                            "params": {"url": "https://example.invalid"},
                        },
                    }
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call-3",
                        "is_error": False,
                        "content": [{"type": "text", "text": "opened"}],
                    }
                ]
            },
        },
    ]

    records = parse_workbuddy_output("\n".join(json.dumps(line) for line in lines))

    assert records[-1]["type"] == "mcp_tool_call"
    assert records[-1]["server"] == "browser"
    assert records[-1]["tool"] == "browser_navigate"
    assert records[-1]["arguments"] == {"url": "https://example.invalid"}


def test_workbuddy_mcp_config_uses_native_path_and_schema():
    class FakeClient:
        command = ""

        async def execute_command(self, command):
            self.command = command
            return {"result": {"exit_code": 0, "stdout": "ok", "stderr": ""}}

    runner = WorkBuddySandboxRunner.__new__(WorkBuddySandboxRunner)
    runner._active_service_sandbox_var = SimpleNamespace(
        get=lambda: SimpleNamespace(
            agent_mcp_servers=[
                SimpleNamespace(
                    name="banking",
                    transport="http",
                    url="http://127.0.0.1:1234/mcp",
                )
            ]
        )
    )
    client = FakeClient()

    result = asyncio.run(runner._install_workbuddy_mcp_config(client))

    assert result["config_path"] == "/root/.codebuddy/.mcp.json"
    assert '"type": "http"' in client.command
    assert '"transport"' not in client.command


def test_workbuddy_self_check_requires_visible_healthy_mcp_server():
    class FakeClient:
        async def execute_command(self, _command):
            return {
                "result": {
                    "exit_code": 0,
                    "stdout": "banking: http://127.0.0.1:1234/mcp (HTTP) - Connected",
                    "stderr": "",
                }
            }

    runner = WorkBuddySandboxRunner.__new__(WorkBuddySandboxRunner)
    runner.config = {"sandbox": {"service_adapter_self_check": "strict"}}
    runner._active_service_sandbox_var = SimpleNamespace(
        get=lambda: SimpleNamespace(
            agent_mcp_servers=[SimpleNamespace(name="banking")]
        )
    )

    report = asyncio.run(runner.adapter_service_self_check(FakeClient()))

    assert report["status"] == "ok"
    assert report["visible_servers"] == ["banking"]


def test_workbuddy_self_check_rejects_unhealthy_mcp_server():
    class FakeClient:
        async def execute_command(self, _command):
            return {
                "result": {
                    "exit_code": 0,
                    "stdout": "banking: http://127.0.0.1:1234/mcp - Failed to connect",
                    "stderr": "",
                }
            }

    runner = WorkBuddySandboxRunner.__new__(WorkBuddySandboxRunner)
    runner.config = {"sandbox": {"service_adapter_self_check": "strict"}}
    runner._active_service_sandbox_var = SimpleNamespace(
        get=lambda: SimpleNamespace(
            agent_mcp_servers=[SimpleNamespace(name="banking")]
        )
    )

    try:
        asyncio.run(runner.adapter_service_self_check(FakeClient()))
    except AgentRuntimeError as exc:
        assert "self-check failed" in str(exc)
    else:
        raise AssertionError("unhealthy WorkBuddy MCP server passed strict self-check")
