#!/usr/bin/env python3
"""Workspace/service runner for WorkBuddy (Tencent CodeBuddy Code CLI)."""

import argparse
import json
import os
import shlex
import uuid
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
from typing import Any, Dict, List, Optional, Tuple

from red_agent_world.sandbox.container_files import SANDBOX_WORKSPACE, put_text_file
from red_agent_world.runners.agent_runtime_sandbox_runner import (
    AgentRuntimeError,
    OpenClawCompatibleAgentRunner,
    ensure_npm_binary,
    parse_jsonl_lines,
    run_cli,
    shell_join,
)
from red_agent_world.sandbox.local_docker import LocalDockerSandbox


def parse_workbuddy_output(text: str) -> List[Dict[str, Any]]:
    """Parse CodeBuddy JSON arrays or JSONL streams without fragmentation."""
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return normalize_workbuddy_mcp_records(
            parse_jsonl_lines(text, fallback_type="workbuddy_stdout")
        )

    values = value if isinstance(value, list) else [value]
    records = [
        item
        if isinstance(item, dict)
        else {"type": "workbuddy_stdout", "value": item}
        for item in values
    ]
    return normalize_workbuddy_mcp_records(records)


def normalize_workbuddy_mcp_records(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Add canonical MCP events for CodeBuddy's deferred-tool wrapper."""
    calls: Dict[str, Dict[str, Any]] = {}
    normalized = list(records)
    for record in records:
        if record.get("type") == "function_call" and record.get("name") == "DeferExecuteTool":
            call_id = str(record.get("callId") or record.get("call_id") or "")
            if call_id:
                calls[call_id] = record
            continue
        if record.get("type") == "assistant":
            message = record.get("message") if isinstance(record.get("message"), dict) else {}
            content = message.get("content") if isinstance(message.get("content"), list) else []
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use" or block.get("name") != "DeferExecuteTool":
                    continue
                call_id = str(block.get("id") or "")
                wrapper_args = block.get("input") if isinstance(block.get("input"), dict) else {}
                if call_id:
                    calls[call_id] = {
                        "stream_wrapper_args": wrapper_args,
                        "stream_record": record,
                    }
            continue
        if record.get("type") != "function_call_result" or record.get("name") != "DeferExecuteTool":
            if record.get("type") != "user":
                continue
            message = record.get("message") if isinstance(record.get("message"), dict) else {}
            content = message.get("content") if isinstance(message.get("content"), list) else []
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result" or block.get("is_error") is True:
                    continue
                call_id = str(block.get("tool_use_id") or "")
                call = calls.get(call_id)
                if call is None or not isinstance(call.get("stream_wrapper_args"), dict):
                    continue
                output_parts = block.get("content") if isinstance(block.get("content"), list) else []
                result = "\n".join(
                    str(part.get("text") or "")
                    for part in output_parts
                    if isinstance(part, dict) and part.get("text") is not None
                )
                event = _workbuddy_mcp_event(
                    call_id,
                    call["stream_wrapper_args"],
                    result,
                    None,
                )
                if event is not None:
                    normalized.append(event)
            continue

        call_id = str(record.get("callId") or record.get("call_id") or "")
        call = calls.get(call_id)
        if call is None:
            continue
        provider_data = record.get("providerData") if isinstance(record.get("providerData"), dict) else {}
        tool_result = provider_data.get("toolResult") if isinstance(provider_data.get("toolResult"), dict) else {}
        if not isinstance(tool_result.get("mcpMeta"), dict):
            continue

        try:
            wrapper_args = json.loads(call.get("arguments") or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        output = record.get("output")
        result = output.get("text") if isinstance(output, dict) else output
        error = tool_result.get("error")
        event = _workbuddy_mcp_event(call_id, wrapper_args, result, error)
        if event is not None:
            normalized.append(event)
    return normalized


def _workbuddy_mcp_event(
    call_id: str,
    wrapper_args: Dict[str, Any],
    result: Any,
    error: Any,
) -> Optional[Dict[str, Any]]:
    tool_name = str(wrapper_args.get("toolName") or "")
    parts = tool_name.split("__", 2)
    if len(parts) != 3 or parts[0] != "mcp" or not parts[1] or not parts[2]:
        return None
    arguments = wrapper_args.get("params", {})
    if not isinstance(arguments, dict):
        arguments = {}
    return {
        "type": "mcp_tool_call",
        "transport": "workbuddy_native_mcp",
        "id": call_id,
        "call_id": call_id,
        "server": parts[1],
        "tool": parts[2],
        "arguments": arguments,
        "status": "failed" if error else "completed",
        "result": result,
        "error": error,
    }


class WorkBuddySandboxRunner(OpenClawCompatibleAgentRunner):
    """Runner for Tencent CodeBuddy Code (WorkBuddy) CLI in sandbox."""

    runtime_name = "workbuddy-cli"
    runtime_config_key = "workbuddy"
    default_runtime_image = "red-agent-world-workbuddy:v1"
    default_output_name = "workbuddy_sandbox_runner"

    @classmethod
    def add_runtime_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--workbuddy-model",
            default=None,
            help="Model for WorkBuddy CLI; defaults to agent_runtimes.workbuddy.model or config.agent.model.",
        )
        parser.add_argument(
            "--workbuddy-package",
            default="@tencent-ai/codebuddy-code",
            help="npm package for WorkBuddy CLI.",
        )
        parser.add_argument(
            "--workbuddy-api-key-env",
            default="CODEBUDDY_API_KEY",
        )
        parser.add_argument(
            "--workbuddy-internet-environment",
            default="",
            choices=["internal", "ioa", ""],
            help="Optional vendor-specific CODEBUDDY_INTERNET_ENVIRONMENT routing mode.",
        )
        parser.add_argument(
            "--workbuddy-max-turns",
            type=int,
            default=100,
            help="Maximum WorkBuddy execution turns (sets CODEBUDDY_CODE_MAX_TURNS).",
        )
        parser.add_argument(
            "--workbuddy-permission-mode",
            default="bypassPermissions",
            choices=["bypassPermissions", "acceptEdits", "default", "plan"],
            help="Permission mode for non-interactive execution.",
        )
        parser.add_argument("--no-install-workbuddy", action="store_true")

    def _model(self) -> str:
        runtime_model = (
            self.config.get("agent_runtimes", {}).get("workbuddy", {}).get("model", "")
        )
        return (
            self.runtime_args.workbuddy_model
            or runtime_model
            or self.config.get("agent", {}).get("model", "")
        )

    def _start_agent_proxy_if_enabled(self) -> None:
        super()._start_agent_proxy_if_enabled()

    def _api_key(self) -> str:
        key = os.environ.get(self.runtime_args.workbuddy_api_key_env, "")
        if key:
            return key
        if self.agent_proxy_base_url:
            return self.agent_proxy_api_key
        keys = self.config.get("agent", {}).get("api_keys", [])
        return str(keys[0]) if keys else ""

    def _base_url(self) -> str:
        if self.agent_proxy_base_url:
            return self.agent_proxy_base_url
        return self.config.get("agent", {}).get("base_url", "")

    async def prepare_runtime(
        self, client: LocalDockerSandbox, item: Dict[str, Any]
    ) -> Dict[str, Any]:
        api_key = self._api_key()
        base_url = self._base_url()
        model = self._model()
        setup = {
            "runtime": self.runtime_name,
            "model": model,
            "permission_mode": self.runtime_args.workbuddy_permission_mode,
            "max_turns": self.runtime_args.workbuddy_max_turns,
            "internet_environment": self.runtime_args.workbuddy_internet_environment,
        }

        env_lines = []
        if api_key:
            env_lines.append(
                "export CODEBUDDY_API_KEY=%s" % shlex.quote(api_key)
            )
        if base_url:
            env_lines.append(
                "export CODEBUDDY_BASE_URL=%s" % shlex.quote(base_url)
            )
        if self.runtime_args.workbuddy_internet_environment:
            env_lines.append(
                "export CODEBUDDY_INTERNET_ENVIRONMENT=%s"
                % shlex.quote(self.runtime_args.workbuddy_internet_environment)
            )
        # Disable telemetry and auto-updates in sandbox
        env_lines.append("export DISABLE_TELEMETRY=1")
        env_lines.append("export DISABLE_AUTOUPDATER=1")
        env_lines.append("export CODEBUDDY_CODE_MAX_TURNS=%s" % self.runtime_args.workbuddy_max_turns)

        env_script = "/root/.codebuddy/workbuddy_runtime_env.sh"
        await client.execute_command("mkdir -p /root/.codebuddy")
        await put_text_file(
            client, env_script, "\n".join(env_lines) + "\n", mode=0o600
        )
        setup["env_script"] = env_script
        setup["api_key_configured"] = bool(api_key)

        # Configure MCP service sandbox servers via CodeBuddy's user MCP file.
        mcp_servers_config = await self._install_workbuddy_mcp_config(client)
        setup["mcp_servers"] = mcp_servers_config

        if self.runtime_args.no_install_workbuddy:
            check = await client.execute_command(
                "command -v codebuddy && codebuddy --version"
            )
            result = check.get("result", {})
            if result.get("exit_code") not in (0, None):
                raise AgentRuntimeError(
                    "codebuddy is not installed in container and --no-install-workbuddy was set"
                )
            setup["codebuddy"] = (result.get("stdout") or "").strip()
        else:
            setup["codebuddy"] = await ensure_npm_binary(
                client, "codebuddy", self.runtime_args.workbuddy_package
            )

        return setup

    async def _install_workbuddy_mcp_config(
        self, client: LocalDockerSandbox
    ) -> Dict[str, Any]:
        """Install MCP server configuration in CodeBuddy's native user file.

        `codebuddy mcp add --scope user` writes `/root/.codebuddy/.mcp.json`.
        Match that path and schema so non-interactive sessions discover the
        benchmark service tools.
        """
        installed_services: List[Dict[str, str]] = []
        if self._active_service_sandbox is None:
            return {"mcp_servers": [], "status": "no_service_sandbox"}

        mcp_servers = {}
        for server in self._active_service_sandbox.agent_mcp_servers:
            mcp_servers[server.name] = {
                "args": [],
                "print": True,
                "type": server.transport,
                "url": server.url,
            }
            installed_services.append({
                "name": server.name,
                "url": server.url,
                "transport": server.transport,
            })

        if not mcp_servers:
            return {"mcp_servers": [], "status": "no_mcp_servers"}

        settings_path = "/root/.codebuddy/.mcp.json"
        script = r"""python3 - <<'PY'
import json
from pathlib import Path

mcp_servers = json.loads('''%s''')
settings_path = Path('%s')
settings_path.parent.mkdir(parents=True, exist_ok=True)

settings = {'mcpServers': {}, 'disabledMcpServers': []}
if settings_path.exists():
    try:
        settings = json.loads(settings_path.read_text(encoding='utf-8'))
    except Exception:
        settings = {}

settings['mcpServers'] = mcp_servers
settings.setdefault('disabledMcpServers', [])
settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': 'ok', 'mcp_servers': list(mcp_servers.keys())}, ensure_ascii=False))
PY""" % (
            json.dumps(mcp_servers, ensure_ascii=False),
            settings_path,
        )

        response = await client.execute_command(script)
        result = response.get("result", {})
        if result.get("exit_code") not in (0, None):
            raise AgentRuntimeError(
                "failed to install WorkBuddy MCP config: %s"
                % ((result.get("stderr") or result.get("stdout") or "")[:1000])
            )

        return {
            "config_path": settings_path,
            "mcp_servers": installed_services,
            "status": "installed",
        }

    async def adapter_service_self_check(
        self, client: LocalDockerSandbox
    ) -> Dict[str, Any]:
        handle = self._active_service_sandbox
        if handle is None or not handle.agent_mcp_servers:
            return {"status": "not_required", "runtime": self.runtime_name}

        expected_servers = [server.name for server in handle.agent_mcp_servers]
        try:
            response = await client.execute_command(
                "source /root/.codebuddy/workbuddy_runtime_env.sh && "
                "HOME=/root timeout 30s codebuddy mcp list"
            )
            result = response.get("result", {})
            stdout = result.get("stdout") or ""
            stderr = result.get("stderr") or ""
            combined = stdout + "\n" + stderr
            visible_servers = [
                name for name in expected_servers if name in combined
            ]
            unhealthy = any(
                marker in combined.lower()
                for marker in ("failed to connect", "disconnected", "connection error")
            )
            ok = (
                result.get("exit_code") in (0, None)
                and len(visible_servers) == len(expected_servers)
                and not unhealthy
            )
        except Exception as exc:
            ok = False
            stdout = ""
            stderr = str(exc)
            visible_servers = []
            result = {"stderr": stderr}

        report = {
            "status": "ok" if ok else "failed",
            "runtime": self.runtime_name,
            "command": "codebuddy mcp list",
            "expected_servers": expected_servers,
            "visible_servers": visible_servers,
            "exit_code": result.get("exit_code"),
            "stdout": stdout[-4000:],
            "stderr": stderr[-4000:],
        }
        mode = str(
            self.config.get("sandbox", {}).get(
                "service_adapter_self_check", "strict"
            )
        ).strip().lower()
        if not ok and mode in {"strict", "true", "1", "required"}:
            raise AgentRuntimeError(
                "WorkBuddy service adapter self-check failed: %s" % report
            )
        return report

    async def run_agent_turn(
        self,
        client: LocalDockerSandbox,
        item: Dict[str, Any],
        turn_idx: int,
        query: str,
        runtime_session_id: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        item_id = int(item["id"])
        session_id = runtime_session_id or str(uuid.uuid4())

        # CodeBuddy Code CLI non-interactive mode:
        #   -p : print mode (non-interactive, outputs final result)
        #   -y : auto-approve all permissions
        #   --output-format stream-json : bounded JSONL events (including resumes)
        #   --model : model selection
        #   --resume : resume session (if session_id provided)
        argv = [
            "codebuddy",
            "-p",
            "-y",
            "--output-format",
            "stream-json",
            "--permission-mode",
            self.runtime_args.workbuddy_permission_mode,
        ]

        model = self._model()
        if model:
            argv.extend(["--model", model])

        if self.runtime_args.workbuddy_max_turns:
            argv.extend(["--max-turns", str(self.runtime_args.workbuddy_max_turns)])

        if runtime_session_id:
            argv.extend(["--resume", session_id])
        else:
            argv.extend(["--session-id", session_id])

        argv.append(query)

        timeout = int(self.config.get("execution", {}).get("timeout", 900)) + 120
        env_script = "/root/.codebuddy/workbuddy_runtime_env.sh"
        inner = (
            "source %s && cd %s && %s"
            % (
                shlex.quote(env_script),
                shlex.quote(SANDBOX_WORKSPACE),
                shell_join(argv),
            )
        )
        response = await client.execute_command(
            "timeout %ss bash -lc %s" % (timeout, shlex.quote(inner))
        )
        result = response.get("result", {})
        stdout = result.get("stdout") or ""
        stderr = result.get("stderr") or ""

        records = parse_workbuddy_output(stdout)
        if stderr.strip():
            records.append({"type": "stderr", "text": stderr})

        if result.get("exit_code") not in (0, None):
            raise AgentRuntimeError(
                "codebuddy failed exit_code=%s stderr=%s stdout=%s"
                % (result.get("exit_code"), stderr[:1000], stdout[:1000]),
                records=records,
                runtime_session_id=session_id,
            )

        return records, session_id


async def main() -> None:
    await run_cli(
        WorkBuddySandboxRunner,
        "WorkBuddy (Tencent CodeBuddy Code) workspace/service runner",
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
