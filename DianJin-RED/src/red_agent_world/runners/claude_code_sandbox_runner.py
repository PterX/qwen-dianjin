#!/usr/bin/env python3
"""Workspace/service runner for Claude Code CLI."""

import argparse
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
from red_agent_world.runners.agent_runtime_sandbox_runner import AgentRuntimeError, OpenClawCompatibleAgentRunner, ensure_npm_binary, parse_jsonl_lines, run_cli, shell_join
from red_agent_world.sandbox.local_docker import LocalDockerSandbox


class ClaudeCodeSandboxRunner(OpenClawCompatibleAgentRunner):
    runtime_name = "claude-code-cli"
    runtime_config_key = "claudecode"
    default_runtime_image = "red-agent-world-claudecode:v1"
    default_output_name = "claude_code_sandbox_runner"

    def _start_agent_proxy_if_enabled(self) -> None:
        return None

    @classmethod
    def add_runtime_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--claude-model", default="claude-sonnet-5")
        parser.add_argument("--claude-package", default="@anthropic-ai/claude-code@latest")
        parser.add_argument("--claude-mode", choices=["cli"], default="cli")
        parser.add_argument("--claude-output-format", choices=["stream-json", "json", "text"], default="stream-json")
        parser.add_argument("--claude-permission-mode", default="acceptEdits", choices=["acceptEdits", "auto", "manual", "dontAsk", "plan"])
        parser.add_argument("--claude-dangerously-skip-permissions", action="store_true")
        parser.add_argument("--anthropic-api-key-env", default="ANTHROPIC_API_KEY")
        parser.add_argument("--no-install-claude", action="store_true")

    async def prepare_runtime(self, client: LocalDockerSandbox, item: Dict[str, Any]) -> Dict[str, Any]:
        env_lines = []
        setup = {
            "runtime": self.runtime_name,
            "mode": self.runtime_args.claude_mode,
            "model": self.runtime_args.claude_model,
            "output_format": self.runtime_args.claude_output_format,
            "auth_mode": "claude-code-native",
            "dangerously_skip_permissions": bool(self.runtime_args.claude_dangerously_skip_permissions),
        }
        api_key = os.environ.get(self.runtime_args.anthropic_api_key_env, "")
        if not api_key:
            raise AgentRuntimeError(
                "%s is not set on host; refusing to run Claude Code without native Anthropic/Claude authentication"
                % self.runtime_args.anthropic_api_key_env
            )
        env_lines.extend([
            "unset ANTHROPIC_BASE_URL",
            "unset ANTHROPIC_AUTH_TOKEN",
            "export ANTHROPIC_API_KEY=%s" % shlex.quote(api_key),
        ])
        env_script = "/tmp/claude_runtime_env.sh" if self.runtime_args.claude_dangerously_skip_permissions else "/root/.claude/claude_runtime_env.sh"
        await client.execute_command("mkdir -p /root/.claude")
        await put_text_file(client, env_script, "\n".join(env_lines) + "\n", mode=0o600)
        setup["env_script"] = env_script
        if self.runtime_args.no_install_claude:
            check = await client.execute_command("command -v claude && claude --version")
            result = check.get("result", {})
            if result.get("exit_code") not in (0, None):
                raise AgentRuntimeError("claude is not installed in container and --no-install-claude was set")
            setup["claude"] = (result.get("stdout") or "").strip()
        else:
            setup["claude"] = await ensure_npm_binary(client, "claude", self.runtime_args.claude_package)
        if self.runtime_args.claude_dangerously_skip_permissions:
            user_setup = await client.execute_command(
                "id -u redagent >/dev/null 2>&1 || useradd -m -s /bin/bash redagent; "
                "chmod 755 /root; "
                "chmod -R a+rwX /workspace /tmp"
            )
            result = user_setup.get("result", {})
            if result.get("exit_code") not in (0, None):
                raise AgentRuntimeError("failed to prepare non-root Claude user: %s" % ((result.get("stderr") or result.get("stdout") or "")[:1000]))
            setup["run_user"] = "redagent"
        return setup

    async def run_agent_turn(self, client: LocalDockerSandbox, item: Dict[str, Any], turn_idx: int, query: str, runtime_session_id: Optional[str]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        item_id = int(item["id"])
        session_id = runtime_session_id or str(uuid.uuid4())
        argv = ["claude", "--bare", "--print", "--output-format", self.runtime_args.claude_output_format, "--permission-mode", self.runtime_args.claude_permission_mode, "--model", self.runtime_args.claude_model]
        if self.runtime_args.claude_output_format == "stream-json":
            argv.append("--verbose")
        if self.runtime_args.claude_dangerously_skip_permissions:
            argv.append("--dangerously-skip-permissions")
        argv.extend(["--resume", session_id] if runtime_session_id else ["--session-id", session_id])
        argv.append(query)
        timeout = int(self.config.get("execution", {}).get("timeout", 900)) + 120
        env_script = "/tmp/claude_runtime_env.sh" if self.runtime_args.claude_dangerously_skip_permissions else "/root/.claude/claude_runtime_env.sh"
        inner = "source %s && cd %s && %s" % (shlex.quote(env_script), shlex.quote(SANDBOX_WORKSPACE), shell_join(argv))
        if self.runtime_args.claude_dangerously_skip_permissions:
            inner = "su -s /bin/bash redagent -c %s" % shlex.quote(inner)
        response = await client.execute_command("timeout %ss bash -lc %s" % (timeout, shlex.quote(inner)))
        result = response.get("result", {})
        stdout = result.get("stdout") or ""
        stderr = result.get("stderr") or ""
        records = parse_jsonl_lines(stdout, fallback_type="claude_stdout")
        if stderr.strip():
            records.append({"type": "stderr", "text": stderr})
        if result.get("exit_code") not in (0, None):
            raise AgentRuntimeError("claude failed exit_code=%s stderr=%s stdout=%s" % (result.get("exit_code"), stderr[:1000], stdout[:1000]))
        return records, session_id


async def main() -> None:
    await run_cli(ClaudeCodeSandboxRunner, "Claude Code workspace/service runner")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
