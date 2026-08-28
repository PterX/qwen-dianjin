#!/usr/bin/env python3
"""Workspace/service runner for Hermes CLI."""

import argparse
import asyncio
import json
import os
import re
import shlex
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from red_agent_world.sandbox.container_files import SANDBOX_WORKSPACE, put_text_file
from red_agent_world.runners.agent_runtime_sandbox_runner import AgentRuntimeError, OpenClawCompatibleAgentRunner, parse_jsonl_lines, run_cli
from red_agent_world.runners.skill_exposure import hermes_workspace_skills_yaml, prepare_preplaced_skill_exposure
from red_agent_world.sandbox.local_docker import LocalDockerSandbox


_SESSION_RE = re.compile(r"session_id:\s*([0-9]{8}_[0-9]{6}_[a-f0-9]+)")


class HermesSandboxRunner(OpenClawCompatibleAgentRunner):
    runtime_name = "hermes-cli"
    runtime_config_key = "hermes"
    default_runtime_image = "red-agent-world-hermes:v1"
    default_output_name = "hermes_sandbox_runner"
    discovery_wrapper_path = "/tmp/red_agent_world_hermes_entrypoint.py"
    pinned_tirith_path = "/opt/red-agent-world/bin/tirith"

    @classmethod
    def add_runtime_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--hermes-model", default=None, help="Model for Hermes; defaults to agent_runtimes.hermes.model or config.agent.model.")
        parser.add_argument("--hermes-base-url", default="")
        parser.add_argument("--hermes-api-key-env", default="OPENAI_API_KEY")
        parser.add_argument("--hermes-bin", default="hermes")
        parser.add_argument("--hermes-install-script", default="https://hermes-agent.nousresearch.com/install.sh")
        parser.add_argument("--hermes-install-method", choices=["preinstalled", "script", "command"], default="preinstalled")
        parser.add_argument(
            "--hermes-install-command",
            default="",
            help="Container shell command used to install Hermes when the binary is missing; overrides --hermes-install-script.",
        )
        parser.add_argument("--hermes-accept-hooks", action="store_true", help="Allow Hermes to install or run hooks during evaluation.")
        parser.add_argument(
            "--hermes-max-turns",
            type=int,
            default=40,
            help="Maximum Hermes tool iterations per user turn. Reaching the limit is a rollout error.",
        )
        parser.add_argument(
            "--hermes-behavior-profile",
            choices=["historical", "trust-boundary-only", "native-guidance", "native"],
            default="historical",
            help=(
                "Hermes behavioral controls: historical reproduces the Qwen3.5/Qwen3.7 runs; "
                "native-guidance restores upstream prompt/probe defaults without Tirith; "
                "native additionally requires a pinned Tirith binary."
            ),
        )
        parser.add_argument("--no-install-hermes", action="store_true")

    def _api_key(self) -> str:
        if self.agent_proxy_base_url:
            return self.agent_proxy_api_key
        key = os.environ.get(self.runtime_args.hermes_api_key_env, "")
        if key:
            return key
        keys = self.config.get("agent", {}).get("api_keys", [])
        return str(keys[0]) if keys else ""

    def _base_url(self) -> str:
        if self.agent_proxy_base_url:
            return str(self.agent_proxy_base_url)
        return self.runtime_args.hermes_base_url or self.config.get("agent", {}).get("base_url", "")

    def service_tool_identity(self, wire_name: str) -> Optional[Tuple[str, str]]:
        """Decode only Hermes' native mcp__server__tool wire name."""
        wire = str(wire_name or "").strip()
        prefix = "mcp__"
        if not wire.startswith(prefix):
            return None
        parts = wire[len(prefix):].split("__", 1)
        if len(parts) != 2 or not all(parts):
            return None
        return self.exact_service_tool_identity(parts[0], parts[1])

    def _model(self) -> str:
        runtime_model = self.config.get("agent_runtimes", {}).get("hermes", {}).get("model", "")
        return self.runtime_args.hermes_model or runtime_model or self.config.get("agent", {}).get("model", "")

    def _runtime_config(self) -> Dict[str, Any]:
        config = self.config.get("agent_runtimes", {}).get("hermes", {})
        return config if isinstance(config, dict) else {}

    def _context_length(self) -> int:
        runtime_config = self._runtime_config()
        for source in (runtime_config, self.config.get("agent", {})):
            if not isinstance(source, dict):
                continue
            for key in ("context_length", "max_context_tokens"):
                value = source.get(key)
                if value is None:
                    continue
                try:
                    parsed = int(value)
                except (TypeError, ValueError):
                    continue
                if parsed > 0:
                    return parsed

        # Qwen-family aliases vary by account/region. Use the conservative
        # 128K-family budget unless the caller explicitly provides a larger one.
        if "qwen" in self._model().lower():
            return 131072
        return 128000

    def _max_output_tokens(self) -> int:
        runtime_config = self._runtime_config()
        for source in (runtime_config, self.config.get("agent", {})):
            if not isinstance(source, dict):
                continue
            for key in ("max_output_tokens", "max_tokens"):
                value = source.get(key)
                if value is None:
                    continue
                try:
                    parsed = int(value)
                except (TypeError, ValueError):
                    continue
                if parsed > 0:
                    return min(parsed, 4096 if "qwen" in self._model().lower() else 8192)
        return 4096 if "qwen" in self._model().lower() else 8192

    def _qwen_extra_body_yaml(self) -> str:
        model = self._model().lower()
        base_url = self._base_url().lower()
        if "qwen" not in model and "dashscope" not in base_url:
            return ""
        return """    extra_body:
      enable_thinking: false
"""

    def _auxiliary_extra_body_yaml(self) -> str:
        return self._qwen_extra_body_yaml() or "    extra_body: {}\n"

    def _behavior_settings(self) -> Dict[str, Any]:
        profile = str(getattr(self.runtime_args, "hermes_behavior_profile", "historical"))
        if profile in {"native-guidance", "native"}:
            tirith_enabled = profile == "native"
            return {
                "profile": profile,
                "tool_use_enforcement": "auto",
                "task_completion_guidance": True,
                "environment_probe": True,
                "tirith_enabled": tirith_enabled,
                "tirith_path": self.pinned_tirith_path if tirith_enabled else "tirith",
                "untrusted_tool_wrapping": True,
            }
        settings = {
            "profile": "historical",
            "tool_use_enforcement": False,
            "task_completion_guidance": False,
            "environment_probe": False,
            "tirith_enabled": False,
            "tirith_path": "tirith",
            "untrusted_tool_wrapping": False,
        }
        if profile == "trust-boundary-only":
            settings["profile"] = profile
            settings["untrusted_tool_wrapping"] = True
        return settings

    @staticmethod
    def _yaml_scalar(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def _render_config(self) -> str:
        base_url = self._base_url()
        api_key = self._api_key()
        model = self._model()
        context_length = self._context_length()
        max_output_tokens = self._max_output_tokens()
        qwen_extra_body = self._qwen_extra_body_yaml()
        auxiliary_extra_body = self._auxiliary_extra_body_yaml()
        behavior = self._behavior_settings()
        if not base_url:
            raise AgentRuntimeError("Hermes base URL is empty")
        if not api_key:
            raise AgentRuntimeError("Hermes API key is empty")
        return """model:
  default: {model}
  provider: custom
  base_url: {base_url}
  api_key: {api_key}
  context_length: {context_length}
  max_tokens: {max_output_tokens}
providers: {{}}
fallback_providers: []
mcp_discovery_timeout: 30
toolsets:
  - hermes-cli
terminal:
  backend: local
  cwd: {workspace}
  timeout: 180
  persistent_shell: true
agent:
  max_turns: {max_turns}
  tool_use_enforcement: {tool_use_enforcement}
  task_completion_guidance: {task_completion_guidance}
  parallel_tool_call_guidance: false
  environment_probe: {environment_probe}
  coding_context: off
  disabled_toolsets:
    - clarify
compression:
  enabled: true
  threshold: 0.70
  target_ratio: 0.15
  protect_last_n: 8
  protect_first_n: 0
  abort_on_summary_failure: false
approvals:
  mode: auto
  timeout: 60
security:
  tirith_enabled: {tirith_enabled}
  tirith_path: {tirith_path}
custom_providers:
  - name: red-agent-world
    base_url: {base_url}
    api_key: {api_key}
    model: {model}
    context_length: {context_length}
{qwen_extra_body}    timeout: 180
    models:
      {model}:
        context_length: {context_length}
auxiliary:
  compression:
    provider: custom
    model: {model}
    base_url: {base_url}
    api_key: {api_key}
    context_length: {context_length}
    timeout: 180
{auxiliary_extra_body}
""".format(
            model=model,
            base_url=base_url,
            api_key=api_key,
            context_length=context_length,
            max_output_tokens=max_output_tokens,
            max_turns=max(1, int(self.runtime_args.hermes_max_turns)),
            tool_use_enforcement=self._yaml_scalar(behavior["tool_use_enforcement"]),
            task_completion_guidance=self._yaml_scalar(behavior["task_completion_guidance"]),
            environment_probe=self._yaml_scalar(behavior["environment_probe"]),
            tirith_enabled=self._yaml_scalar(behavior["tirith_enabled"]),
            tirith_path=self._yaml_scalar(behavior["tirith_path"]),
            qwen_extra_body=qwen_extra_body,
            auxiliary_extra_body=auxiliary_extra_body,
            workspace=SANDBOX_WORKSPACE,
        )

    async def ensure_hermes_binary(self, client: LocalDockerSandbox) -> Dict[str, Any]:
        binary = shlex.quote(self.runtime_args.hermes_bin)
        check = await client.execute_command("command -v {0} && {0} --version".format(binary))
        result = check.get("result", {})
        if result.get("exit_code") in (0, None):
            return {"binary": self.runtime_args.hermes_bin, "version": (result.get("stdout") or result.get("stderr") or "").strip()}
        if self.runtime_args.no_install_hermes or self.runtime_args.hermes_install_method == "preinstalled":
            raise AgentRuntimeError(
                "hermes is not installed in container. Build and use red-agent-world-hermes:v1, "
                "or select the public installer explicitly with --hermes-install-method script."
            )
        if self.runtime_args.hermes_install_command:
            install = await client.execute_command(self.runtime_args.hermes_install_command)
        else:
            install_command = "curl -fsSL %s | HERMES_SETUP=full HERMES_BACKEND=local HERMES_SKIP_GATEWAY=1 bash" % shlex.quote(
                self.runtime_args.hermes_install_script
            )
            install = await client.execute_command(install_command)
        payload = install.get("result", {})
        if payload.get("exit_code") not in (0, None):
            raise AgentRuntimeError("failed to install Hermes: %s" % ((payload.get("stderr") or payload.get("stdout") or "")[:1000]))
        version = await client.execute_command("command -v {0} && {0} --version".format(binary))
        result = version.get("result", {})
        if result.get("exit_code") not in (0, None):
            raise AgentRuntimeError("Hermes install completed but %s is still unavailable" % self.runtime_args.hermes_bin)
        return {
            "binary": self.runtime_args.hermes_bin,
            "install": self.runtime_args.hermes_install_method,
            "version": (result.get("stdout") or result.get("stderr") or "").strip(),
        }

    async def prepare_runtime(self, client: LocalDockerSandbox, item: Dict[str, Any]) -> Dict[str, Any]:
        setup = {
            "runtime": self.runtime_name,
            "model": self._model(),
            "workspace": SANDBOX_WORKSPACE,
            "behavior": self._behavior_settings(),
        }
        setup["hermes"] = await self.ensure_hermes_binary(client)
        behavior = self._behavior_settings()
        if behavior["tirith_enabled"]:
            tirith_path = str(behavior["tirith_path"])
            tirith_check = await client.execute_command("test -x %s" % shlex.quote(tirith_path))
            if tirith_check.get("result", {}).get("exit_code") not in (0, None):
                raise AgentRuntimeError(
                    "native Hermes profile requires pinned Tirith at %s; automatic download is disabled"
                    % tirith_path
                )
        await client.execute_command("mkdir -p /root/.hermes")
        skill_exposure = await prepare_preplaced_skill_exposure(client, item, self.runtime_name)
        config_yaml = self._render_config() + hermes_workspace_skills_yaml(item)
        if self._active_service_sandbox is not None and self._active_service_sandbox.agent_mcp_servers:
            config_yaml += "\nmcp_servers:\n"
            for service in self._active_service_sandbox.agent_mcp_servers:
                config_yaml += "  {name}:\n    transport: {transport}\n    url: {url}\n".format(
                    name=service.name,
                    transport=service.transport,
                    url=service.url,
                )
        else:
            config_yaml += "\nmcp_servers: {}\n"
        await put_text_file(client, "/root/.hermes/config.yaml", config_yaml, mode=0o600)
        # Hermes validates --toolsets before its normal MCP discovery pass.  Run
        # discovery in the same process before entering the CLI so dynamic
        # mcp-<server> toolsets are registered when argument validation occurs.
        await put_text_file(
            client,
            self.discovery_wrapper_path,
            "from agent import tool_dispatch_helpers\n"
            + (
                "tool_dispatch_helpers._UNTRUSTED_TOOL_NAMES = frozenset()\n"
                "tool_dispatch_helpers._UNTRUSTED_TOOL_PREFIXES = ()\n"
                if not behavior["untrusted_tool_wrapping"]
                else ""
            )
            + "from tools.mcp_tool import discover_mcp_tools\n"
            "discover_mcp_tools()\n"
            "from hermes_cli.main import main\n"
            "raise SystemExit(main())\n",
            mode=0o700,
        )
        setup["config_path"] = "/root/.hermes/config.yaml"
        setup["entrypoint"] = self.discovery_wrapper_path
        setup["max_turns"] = max(1, int(self.runtime_args.hermes_max_turns))
        setup["disabled_toolsets"] = ["clarify"]
        if skill_exposure is not None:
            setup["skill_exposure"] = skill_exposure
        return setup

    def _enabled_toolsets(self) -> List[str]:
        """Use canonical MCP toolset names so server aliases cannot collide with built-ins."""
        toolsets = ["hermes-cli"]
        handle = self._active_service_sandbox
        if handle is not None:
            toolsets.extend("mcp-%s" % server.name for server in handle.agent_mcp_servers)
        return toolsets

    async def adapter_service_self_check(self, client: LocalDockerSandbox) -> Dict[str, Any]:
        handle = self._active_service_sandbox
        if handle is None or not handle.agent_mcp_servers:
            return {"status": "not_required", "runtime": self.runtime_name}
        expected_servers = [server.name for server in handle.agent_mcp_servers]
        expected_tools = sorted(
            "mcp__%s__%s" % (server.name, tool_name)
            for server in handle.agent_mcp_servers
            for tool_name in server.tools
        )
        enabled_toolsets = self._enabled_toolsets()
        script = r"""hermes_executable=$(readlink -f "$(command -v hermes)")
hermes_python=$(sed -n '1s/^#!//p' "$hermes_executable")
HOME=/root "$hermes_python" - <<'PY'
import json
import traceback

enabled_toolsets = json.loads(__ENABLED_TOOLSETS__)
try:
    from tools.mcp_tool import discover_mcp_tools, get_mcp_status
    from model_tools import get_tool_definitions

    registered = sorted(discover_mcp_tools())
    definitions = get_tool_definitions(
        enabled_toolsets=enabled_toolsets,
        quiet_mode=True,
    )
    model_visible = sorted(
        tool.get('function', {}).get('name', '')
        for tool in definitions
        if tool.get('function', {}).get('name')
    )
    report = {
        'ok': True,
        'registered_tool_names': registered,
        'model_visible_tool_names': model_visible,
        'mcp_status': get_mcp_status(),
    }
except BaseException as exc:
    report = {
        'ok': False,
        'error_type': type(exc).__name__,
        'error': str(exc),
        'traceback': traceback.format_exc(),
    }
print('__HERMES_TOOL_SURFACE_REPORT__' + json.dumps(report, ensure_ascii=False))
PY""".replace(
            "__ENABLED_TOOLSETS__",
            repr(json.dumps(enabled_toolsets, ensure_ascii=False)),
        )
        response = await client.execute_command(script)
        result = response.get("result", {})
        stdout = (result.get("stdout") or "").strip()
        marker = "__HERMES_TOOL_SURFACE_REPORT__"
        parsed: Dict[str, Any] = {}
        for line in reversed(stdout.splitlines()):
            if line.startswith(marker):
                try:
                    parsed = json.loads(line[len(marker):])
                except json.JSONDecodeError as exc:
                    parsed = {"ok": False, "parse_error": str(exc), "raw": line[-4000:]}
                break
        if not parsed:
            parsed = {"ok": False, "parse_error": "tool-surface report marker missing"}
        registered = set(parsed.get("registered_tool_names") or [])
        model_visible = set(parsed.get("model_visible_tool_names") or [])
        missing_registered = sorted(set(expected_tools) - registered)
        missing_model_visible = sorted(set(expected_tools) - model_visible)
        ok = (
            result.get("exit_code") in (0, None)
            and parsed.get("ok") is True
            and not missing_registered
            and not missing_model_visible
        )
        report = {
            "status": "ok" if ok else "failed",
            "runtime": self.runtime_name,
            "check": "hermes_model_tool_surface",
            "expected_servers": expected_servers,
            "enabled_toolsets": enabled_toolsets,
            "expected_tool_names": expected_tools,
            "missing_registered_tool_names": missing_registered,
            "missing_model_visible_tool_names": missing_model_visible,
            **parsed,
            "exit_code": result.get("exit_code"),
            "stderr": (result.get("stderr") or "")[-2000:],
        }
        mode = str(self.config.get("sandbox", {}).get("service_adapter_self_check", "strict")).strip().lower()
        if not ok and mode in {"strict", "true", "1", "required"}:
            raise AgentRuntimeError("Hermes service adapter self-check failed: %s" % report)
        return report

    async def run_agent_turn(self, client: LocalDockerSandbox, item: Dict[str, Any], turn_idx: int, query: str, runtime_session_id: Optional[str]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        prompt_path = "/tmp/red_agent_world_hermes_%s_%s.txt" % (int(item["id"]), turn_idx)
        await put_text_file(client, prompt_path, query, mode=0o600)
        resume = " --resume %s" % shlex.quote(runtime_session_id) if runtime_session_id else ""
        toolsets = " --toolsets %s" % shlex.quote(",".join(self._enabled_toolsets()))
        timeout = int(self.config.get("execution", {}).get("timeout", 900)) + 120
        hooks = " --accept-hooks" if self.runtime_args.hermes_accept_hooks else ""
        inner = (
            "cd {workspace} && "
            "hermes_executable=$(readlink -f \"$(command -v {bin})\") && "
            "hermes_python=$(sed -n '1s/^#!//p' \"$hermes_executable\") && "
            "test -x \"$hermes_python\" && "
            "HOME=/root WORKSPACE_MCP_ROOT={workspace} "
            "\"$hermes_python\" {entrypoint} chat -q \"$(cat {prompt})\" "
            "-Q --yolo --max-turns {max_turns}{toolsets}{hooks}{resume}"
        ).format(
            workspace=shlex.quote(SANDBOX_WORKSPACE),
            sandbox_id=shlex.quote(str(item.get("id", ""))),
            bin=shlex.quote(self.runtime_args.hermes_bin),
            entrypoint=shlex.quote(self.discovery_wrapper_path),
            prompt=shlex.quote(prompt_path),
            max_turns=max(1, int(self.runtime_args.hermes_max_turns)),
            toolsets=toolsets,
            hooks=hooks,
            resume=resume,
        )
        run_id = "red_agent_world_hermes_%s_%s" % (int(item["id"]), turn_idx)
        stdout_path = "/tmp/%s.stdout" % run_id
        stderr_path = "/tmp/%s.stderr" % run_id
        status_path = "/tmp/%s.status" % run_id
        pid_path = "/tmp/%s.pid" % run_id
        wrapped_inner = "%s; code=$?; printf '%%s\\n' \"$code\" > %s; exit \"$code\"" % (inner, shlex.quote(status_path))
        launch = (
            "rm -f {stdout} {stderr} {status} {pid}; "
            "setsid bash -lc {inner} > {stdout} 2> {stderr} < /dev/null & "
            "echo $! > {pid}"
        ).format(
            stdout=shlex.quote(stdout_path),
            stderr=shlex.quote(stderr_path),
            status=shlex.quote(status_path),
            pid=shlex.quote(pid_path),
            inner=shlex.quote(wrapped_inner),
        )
        launch_response = await client.execute_command(launch)
        launch_result = launch_response.get("result", {})
        if launch_result.get("exit_code") not in (0, None):
            raise AgentRuntimeError("failed to launch Hermes turn: %s" % ((launch_result.get("stderr") or launch_result.get("stdout") or "")[:1000]))

        poll_interval = 5
        elapsed = 0
        while elapsed < timeout:
            status_response = await client.execute_command("test -s %s && cat %s || true" % (shlex.quote(status_path), shlex.quote(status_path)))
            status_text = (status_response.get("result", {}).get("stdout") or "").strip()
            if status_text:
                break
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        else:
            kill_command = """
pid=$(cat __PID__ 2>/dev/null || true)
if [ -n "$pid" ]; then
  kill -TERM -- -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  sleep 3
  kill -KILL -- -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
fi
printf '124\n' > __STATUS__
""".replace("__PID__", shlex.quote(pid_path)).replace("__STATUS__", shlex.quote(status_path))
            await client.execute_command(kill_command)
            raise AgentRuntimeError("Hermes agent turn timed out after %s seconds; process group was terminated" % timeout)

        stdout_response = await client.execute_command("cat %s 2>/dev/null || true" % shlex.quote(stdout_path))
        stderr_response = await client.execute_command("cat %s 2>/dev/null || true" % shlex.quote(stderr_path))
        status_response = await client.execute_command("cat %s 2>/dev/null || true" % shlex.quote(status_path))
        stdout = stdout_response.get("result", {}).get("stdout") or ""
        stderr = stderr_response.get("result", {}).get("stdout") or ""
        status_text = (status_response.get("result", {}).get("stdout") or "").strip()
        try:
            exit_code = int(status_text.splitlines()[-1]) if status_text else 1
        except (TypeError, ValueError):
            exit_code = 1
        result = {"exit_code": exit_code, "stdout": stdout, "stderr": stderr}
        records = parse_jsonl_lines(stdout, fallback_type="hermes_stdout")
        if stderr.strip():
            records.append({"type": "stderr", "text": stderr})
        if result.get("exit_code") not in (0, None):
            raise AgentRuntimeError(
                "hermes failed exit_code=%s stderr=%s stdout=%s"
                % (result.get("exit_code"), stderr[:1000], stdout[:1000])
            )
        match = _SESSION_RE.search(stdout + "\n" + stderr)
        next_session_id = match.group(1) if match else runtime_session_id
        if next_session_id:
            export = await client.execute_command(
                "HOME=/root %s sessions export - --session-id %s"
                % (shlex.quote(self.runtime_args.hermes_bin), shlex.quote(next_session_id))
            )
            export_stdout = export.get("result", {}).get("stdout") or ""
            exported = None
            for line in export_stdout.splitlines():
                if not line.strip().startswith("{"):
                    continue
                try:
                    exported = json.loads(line)
                except Exception:
                    continue
            if isinstance(exported, dict):
                # The CLI stdout contains the final response, while the session
                # export contains the calls that preceded it.  Keep that causal
                # order; the base runner removes call IDs already exported by a
                # previous resumed turn.
                records = self._hermes_service_tool_events(exported) + records
                records.append(
                    {
                        "type": "hermes_usage",
                        "runtime": self.runtime_name,
                        "session_id": next_session_id,
                        "usage": {
                            "input_tokens": int(exported.get("input_tokens") or 0),
                            "output_tokens": int(exported.get("output_tokens") or 0),
                            "cache_read_tokens": int(exported.get("cache_read_tokens") or 0),
                            "cache_write_tokens": int(exported.get("cache_write_tokens") or 0),
                            "reasoning_tokens": int(exported.get("reasoning_tokens") or 0),
                            "api_call_count": int(exported.get("api_call_count") or 0),
                        },
                    }
                )
        fatal_markers = {
            "unknown toolsets": "Hermes MCP toolset registration failed",
            "reached maximum iterations": "Hermes reached its tool iteration limit",
            "clarify timed out": "Hermes invoked interactive clarify in a non-interactive rollout",
        }
        combined_output = (stdout + "\n" + stderr).lower()
        for marker, message in fatal_markers.items():
            if marker in combined_output:
                raise AgentRuntimeError(
                    message,
                    records=records,
                    runtime_session_id=next_session_id,
                )
        return records, next_session_id

    def _hermes_service_tool_events(self, exported: Dict[str, Any]) -> List[Dict[str, Any]]:
        messages = exported.get("messages") if isinstance(exported.get("messages"), list) else []
        results = {
            str(message.get("tool_call_id") or ""): message
            for message in messages
            if isinstance(message, dict) and str(message.get("role") or "") == "tool"
        }
        events: List[Dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []
            for call in calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") if isinstance(call.get("function"), dict) else call
                identity = self.service_tool_identity(str(function.get("name") or ""))
                if identity is None:
                    continue
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except Exception:
                        arguments = {}
                if not isinstance(arguments, dict):
                    arguments = {}
                call_id = str(call.get("id") or call.get("call_id") or "")
                result = results.get(call_id)
                events.append({
                    "type": "mcp_tool_call",
                    "transport": "hermes_native_mcp",
                    "id": call_id,
                    "server": identity[0],
                    "tool": identity[1],
                    "arguments": arguments,
                    "status": "completed" if result is not None else "in_progress",
                    "result": result.get("content") if result else None,
                    "error": result.get("error") if result else None,
                })
        return events


async def main() -> None:
    await run_cli(HermesSandboxRunner, "Hermes CLI workspace/service runner")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
