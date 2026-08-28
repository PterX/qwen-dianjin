#!/usr/bin/env python3
"""Service-backed workspace runner for OpenClaw.

This runner intentionally does not use the legacy filesystem bootstrap path. It
shares the same ServiceSandboxSession flow as Codex/Hermes and talks to the
OpenClaw container through its /prompt API.
"""

import argparse
import asyncio
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from red_agent_world.sandbox.container_files import SANDBOX_WORKSPACE
from red_agent_world.runners.agent_runtime_sandbox_runner import AgentRuntimeError, OpenClawCompatibleAgentRunner, run_cli
from red_agent_world.runners.skill_exposure import prepare_preplaced_skill_exposure
from red_agent_world.sandbox.local_docker import LocalDockerSandbox


OPENCLAW_WORKSPACE_RUNTIME_FILES = {
    "AGENTS.md",
    "BOOTSTRAP.md",
    "HEARTBEAT.md",
    "IDENTITY.md",
    "SOUL.md",
    "TOOLS.md",
    "USER.md",
}


def _openclaw_safe_tool_name(server_name: str, tool_name: str) -> str:
    """Mirror OpenClaw bundle-mcp's normal server__tool naming for simple ids."""
    safe_server = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(server_name or "").strip())
    safe_server = "_".join(part for part in safe_server.split("_") if part)
    return "%s__%s" % (safe_server or "service", str(tool_name or "").strip())


def _canonical_openclaw_tool_name(name: str) -> str:
    """Normalize OpenClaw's display aliases before comparing the MCP surface."""
    server_name, separator, tool_name = str(name or "").strip().partition("__")
    if not separator:
        return str(name or "").strip().lower()
    return _openclaw_safe_tool_name(server_name, tool_name).lower()


class OpenClawServiceSandboxRunner(OpenClawCompatibleAgentRunner):
    runtime_name = "openclaw-service"
    runtime_config_key = "openclaw"
    default_runtime_image = "red-agent-world-openclaw:v1"
    default_runtime_init_mode = "openclaw"
    default_container_api_port = 8000
    default_output_name = "openclaw_service_sandbox_runner"

    @classmethod
    def add_runtime_arguments(cls, parser: argparse.ArgumentParser) -> None:
        return None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        os.environ.setdefault("RED_AGENT_WORLD_AGENT_MCP_HOST", "host.docker.internal")
        super().__init__(*args, **kwargs)

    def service_tool_identity(self, wire_name: str) -> Optional[Tuple[str, str]]:
        """Decode only OpenClaw's documented server__tool wire name."""
        wire = str(wire_name or "").strip()
        handle = self._active_service_sandbox
        if handle is None:
            return None
        for server in handle.agent_mcp_servers:
            for tool in server.tools or []:
                canonical = _openclaw_safe_tool_name(server.name, str(tool))
                if wire == canonical:
                    return self.exact_service_tool_identity(server.name, str(tool))
        return None

    def runtime_docker_config(self) -> Dict[str, Any]:
        config = super().runtime_docker_config()
        if config.get("init_mode") == "openclaw":
            config["network_mode"] = "isolated"
        return config

    async def _discover_live_openclaw_mcp_tool_specs(self, client: LocalDockerSandbox) -> List[Dict[str, Any]]:
        """Read the model-facing tool catalog from the active MCP gateway.

        ``server.tool_specs`` is captured before MCP control rules are installed,
        so it must never be used to build the OpenClaw plugin. E5 mutates
        ``tools/list`` descriptions and input schemas at the gateway; this
        request makes those mutations part of OpenClaw's registered tools.
        """
        handle = self._active_service_sandbox
        if handle is None or not handle.agent_mcp_servers:
            return []
        servers = [
            {
                "name": server.name,
                "url": server.url,
                "transport": server.transport,
                "expected_tools": list(server.tools or []),
            }
            for server in handle.agent_mcp_servers
        ]
        script = r"""python3 - <<'PY'
import json
import sys
import urllib.request

servers = json.loads(__SERVERS_JSON__)
sessions = {}

def post(url, payload, timeout=20):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if sessions.get(url):
        headers["Mcp-Session-Id"] = sessions[url]
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
        session_id = response.headers.get("Mcp-Session-Id")
    if session_id:
        sessions[url] = session_id
    text = raw.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        events = []
        for line in text.splitlines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data and data != "[DONE]":
                events.append(json.loads(data))
        if events:
            return events[-1]
        raise

def list_tools(url):
    initialized = post(url, {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {
                "name": "red-agent-world-openclaw-plugin-registration",
                "version": "1",
            },
        },
    })
    if isinstance(initialized, dict) and initialized.get("error"):
        raise RuntimeError(str(initialized["error"]))
    post(url, {
        "jsonrpc": "2.0",
        "method": "notifications/initialized",
        "params": {},
    })
    listed = post(url, {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {},
    })
    if not isinstance(listed, dict) or listed.get("error"):
        raise RuntimeError(str(listed.get("error") if isinstance(listed, dict) else listed))
    tools = (listed.get("result") or {}).get("tools")
    if not isinstance(tools, list):
        raise RuntimeError("tools/list did not return a tool list")
    return tools

rows = []
for server in servers:
    row = {
        "name": server["name"],
        "url": server["url"],
        "transport": server.get("transport"),
        "ok": False,
        "tools": [],
    }
    try:
        row["tools"] = list_tools(server["url"])
        row["ok"] = True
    except Exception as exc:
        row["error"] = repr(exc)[:500]
    rows.append(row)
print(json.dumps(rows, ensure_ascii=False))
sys.exit(0 if all(row.get("ok") for row in rows) else 2)
PY""".replace("__SERVERS_JSON__", repr(json.dumps(servers, ensure_ascii=False)))

        config = getattr(self, "config", {}) or {}
        sandbox_config = config.get("sandbox", {}) or {}
        attempts = int(sandbox_config.get("service_preflight_attempts", 8) or 8)
        delay = float(sandbox_config.get("service_preflight_delay", 2.0) or 2.0)
        rows: List[Dict[str, Any]] = []
        errors: List[str] = []
        for attempt in range(1, max(1, attempts) + 1):
            response = await client.execute_command(script)
            result = response.get("result", {})
            stdout = (result.get("stdout") or "").strip()
            try:
                rows = json.loads(stdout.splitlines()[-1]) if stdout else []
            except Exception:
                rows = []
            errors = []
            if len(rows) != len(servers):
                errors.append("gateway returned %d server catalogs; expected %d" % (len(rows), len(servers)))
            for expected_server, row in zip(servers, rows):
                if not isinstance(row, dict) or not row.get("ok"):
                    errors.append("%s tools/list failed: %s" % (expected_server["name"], row))
                    continue
                if row.get("name") != expected_server["name"]:
                    errors.append("server order/name mismatch for %s" % expected_server["name"])
                    continue
                tool_specs = row.get("tools") or []
                live_names = [
                    str(tool.get("name") or "")
                    for tool in tool_specs
                    if isinstance(tool, dict) and tool.get("name")
                ]
                expected_names = [str(name) for name in expected_server["expected_tools"] if str(name)]
                if len(live_names) != len(set(live_names)):
                    errors.append("%s tools/list contains duplicate names" % expected_server["name"])
                if set(live_names) != set(expected_names):
                    errors.append(
                        "%s tools/list mismatch: live=%s expected=%s"
                        % (expected_server["name"], sorted(live_names), sorted(expected_names))
                    )
            if rows and not errors:
                return [
                    {
                        "name": row["name"],
                        "url": row["url"],
                        "transport": row.get("transport"),
                        "tools": row["tools"],
                    }
                    for row in rows
                ]
            if attempt < max(1, attempts):
                await asyncio.sleep(delay)
        raise AgentRuntimeError(
            "OpenClaw live MCP tool discovery failed after %d attempt(s): %s"
            % (attempt, json.dumps(errors or ["invalid gateway tools/list output"], ensure_ascii=False))
        )

    async def install_openclaw_service_mcp_config(self, client: LocalDockerSandbox) -> Dict[str, Any]:
        """Register work systems through one generated OpenClaw tool plugin.

        This consumes the live gateway ``tools/list`` inventory without invoking the slow
        ``openclaw mcp set/reload/probe`` CLI sequence. The OpenClaw build in
        the runtime image does not project a bare ``mcp.servers`` config into
        agent tools, so the plugin owns each projected tool exactly once and
        proxies its calls to the service MCP endpoint.
        """
        discovered = await self._discover_live_openclaw_mcp_tool_specs(client)

        total_tools = sum(len(server.get("tools", [])) for server in discovered)
        if discovered and total_tools <= 0:
            raise AgentRuntimeError("OpenClaw service MCP registration found no tools: %s" % json.dumps(discovered, ensure_ascii=False))

        config_patch = r"""python3 - <<'PY'
import json
import shutil
from pathlib import Path

config_path = Path("/root/.openclaw/openclaw.json")
config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
servers = %r
plugin_id = "red-agent-world-service-mcp"
plugin_root = Path("/root/.openclaw/extensions") / plugin_id

service_tool_names = []
plugin_inventory = []
for server in servers:
    for tool in server.get("tools", []):
        name = str(tool.get("name") or "").strip()
        if name:
            exposed_name = f'{server["name"]}__{name}'
            service_tool_names.append(exposed_name)
            plugin_inventory.append({
                "name": exposed_name,
                "mcpToolName": name,
                "url": server["url"],
                "description": str(tool.get("description") or f'{server["name"]} MCP tool {name}'),
                "parameters": tool.get("inputSchema") or {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                },
            })

if len(service_tool_names) != len(set(service_tool_names)):
    raise RuntimeError("duplicate generated OpenClaw service tool names")

shutil.rmtree(plugin_root, ignore_errors=True)
plugin_root.mkdir(parents=True, exist_ok=True)
(plugin_root / "package.json").write_text(json.dumps({
    "name": "@red-agent-world/openclaw-service-mcp",
    "version": "1.0.0",
    "type": "module",
    "openclaw": {"extensions": ["./index.mjs"]},
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(plugin_root / "openclaw.plugin.json").write_text(json.dumps({
    "id": plugin_id,
    "name": "RED Agent World Service MCP",
    "description": "Projects the live RED Agent World MCP catalog into OpenClaw tools.",
    "version": "1.0.0",
    "activation": {"onStartup": True},
    "contracts": {"tools": service_tool_names},
    "configSchema": {"type": "object", "additionalProperties": False, "properties": {}},
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

plugin_source = r'''
const inventory = __INVENTORY__;

function parseMcpResponse(text) {
  const trimmed = String(text || "").trim();
  if (!trimmed) return null;
  const messages = [];
  for (const line of trimmed.split(/\r?\n/)) {
    if (!line.startsWith("data:")) continue;
    const data = line.slice(5).trim();
    if (!data || data === "[DONE]") continue;
    try { messages.push(JSON.parse(data)); } catch (_) {}
  }
  if (messages.length) return messages[messages.length - 1];
  try { return JSON.parse(trimmed); } catch (_) { return null; }
}

async function postMcp(url, payload, sessionId) {
  const headers = {
    "content-type": "application/json",
    "accept": "application/json, text/event-stream",
  };
  if (sessionId) headers["mcp-session-id"] = sessionId;
  const response = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(120000),
  });
  const text = await response.text();
  if (!response.ok && response.status !== 202) {
    throw new Error(`MCP HTTP ${response.status}: ${text.slice(0, 2000)}`);
  }
  return {
    message: parseMcpResponse(text),
    sessionId: response.headers.get("mcp-session-id") || sessionId || "",
  };
}

async function callMcp(tool) {
  const initialized = await postMcp(tool.url, {
    jsonrpc: "2.0",
    id: 1,
    method: "initialize",
    params: {
      protocolVersion: "2025-03-26",
      capabilities: {},
      clientInfo: {name: "red-agent-world-openclaw", version: "1.0.0"},
    },
  }, "");
  await postMcp(tool.url, {
    jsonrpc: "2.0",
    method: "notifications/initialized",
    params: {},
  }, initialized.sessionId);
  const called = await postMcp(tool.url, {
    jsonrpc: "2.0",
    id: 2,
    method: "tools/call",
    params: {name: tool.mcpToolName, arguments: tool.arguments || {}},
  }, initialized.sessionId);
  if (!called.message) throw new Error("MCP tools/call returned no JSON-RPC message");
  if (called.message.error) throw new Error(JSON.stringify(called.message.error));
  return called.message.result || {};
}

export default {
  id: "red-agent-world-service-mcp",
  name: "RED Agent World Service MCP",
  description: "Projects RED Agent World MCP services into OpenClaw tools.",
  register(api) {
    for (const definition of inventory) {
      api.registerTool({
        name: definition.name,
        label: definition.name,
        description: definition.description,
        parameters: definition.parameters,
        async execute(_id, params) {
          const result = await callMcp({...definition, arguments: params || {}});
          const content = Array.isArray(result.content)
            ? result.content
            : [{type: "text", text: JSON.stringify(result)}];
          return {content, details: result.structuredContent || result};
        },
      });
    }
  },
};
'''.replace("__INVENTORY__", json.dumps(plugin_inventory, ensure_ascii=False))
(plugin_root / "index.mjs").write_text(plugin_source, encoding="utf-8")

# A bare mcp.servers block is not a live tool registration in this image and
# would make config-only checks look healthy while the model sees no MCP tools.
config.pop("mcp", None)
plugins = config.setdefault("plugins", {})
plugins["enabled"] = True
plugin_allow = plugins.setdefault("allow", [])
if plugin_id not in plugin_allow:
    plugin_allow.append(plugin_id)
plugins.setdefault("entries", {}).setdefault(plugin_id, {})["enabled"] = True

tools = config.setdefault("tools", {})
tools["profile"] = "full"
deny = tools.setdefault("deny", [])
for tool_name in ("web_search", "web_fetch"):
    if tool_name not in deny:
        deny.append(tool_name)
also_allow = tools.setdefault("alsoAllow", [])
for tool_name in service_tool_names:
    if tool_name not in also_allow:
        also_allow.append(tool_name)
tools.setdefault("web", {}).setdefault("search", {})["enabled"] = False
tools.setdefault("web", {}).setdefault("fetch", {})["enabled"] = False
config.setdefault("browser", {})["enabled"] = False
agent_defaults = config.setdefault("agents", {}).setdefault("defaults", {})
agent_defaults["workspace"] = "/workspace"
agent_defaults["skipBootstrap"] = True
agent_defaults["skipOptionalBootstrapFiles"] = ["SOUL.md", "USER.md", "HEARTBEAT.md", "IDENTITY.md"]
agent_defaults["contextInjection"] = "never"

sandbox_allow = (
    tools.setdefault("sandbox", {})
    .setdefault("tools", {})
    .setdefault("alsoAllow", [])
)
for tool_name in service_tool_names:
    if tool_name not in sandbox_allow:
        sandbox_allow.append(tool_name)

config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({
    "tool_count": sum(len(s.get("tools", [])) for s in servers),
    "service_tool_names": service_tool_names,
    "plugin_root": str(plugin_root),
}, ensure_ascii=False))
PY""" % (discovered,)
        patch_response = await client.execute_command(config_patch)
        patch_result = patch_response.get("result", {})
        if patch_result.get("exit_code") not in (0, None):
            raise AgentRuntimeError("failed to generate OpenClaw service MCP plugin: %s" % ((patch_result.get("stderr") or patch_result.get("stdout") or "")[:2000]))

        if not discovered:
            return {"status": "not_required", "agent_mcp_servers": []}
        plugin_config_verification = await self._verify_openclaw_generated_plugin_config(client, discovered)
        plugin_registry_refresh = await self._refresh_openclaw_plugin_registry(client)
        # One-shot local CLI runs cold-load the refreshed plugin registry, so
        # Gateway restart/watcher timing is not part of tool activation.
        runtime_registration = {
            "status": "deferred",
            "verification": "adapter_service_self_check",
            "activation_refresh": "local_cli_cold_load",
        }
        return {
            "status": "registered",
            "registration": "openclaw_generated_mcp_plugin",
            "tool_catalog_source": "live_gateway_tools_list",
            "tool_count": total_tools,
            "plugin_config_verification": plugin_config_verification,
            "plugin_registry_refresh": plugin_registry_refresh,
            "runtime_registration": runtime_registration,
            "agent_mcp_servers": [
                {
                    "name": server["name"],
                    "url": server["url"],
                    "transport": server["transport"],
                    "tools": [tool.get("name") for tool in server.get("tools", [])],
                }
                for server in discovered
            ],
        }

    async def _refresh_openclaw_plugin_registry(self, client: LocalDockerSandbox) -> Dict[str, Any]:
        """Persist the generated plugin into OpenClaw's registry cache."""
        command = (
            "HOME=/root OPENCLAW_CONFIG_PATH=/root/.openclaw/openclaw.json "
            "openclaw plugins registry --refresh --json"
        )
        response = await client.execute_command(command)
        result = response.get("result", {})
        stdout = (result.get("stdout") or "").strip()
        try:
            parsed = json.loads(stdout.splitlines()[-1]) if stdout else {}
        except Exception:
            parsed = {"raw": stdout[-2000:]}
        if result.get("exit_code") not in (0, None):
            raise AgentRuntimeError(
                "failed to refresh OpenClaw plugin registry after service MCP plugin update: %s"
                % ((result.get("stderr") or result.get("stdout") or json.dumps(parsed, ensure_ascii=False))[-2000:])
            )
        return {
            "status": "refreshed",
            "registry_refresh": "openclaw_plugins_registry_refresh",
            "details": parsed,
        }

    async def _restart_openclaw_gateway_for_plugin_activation(self, client: LocalDockerSandbox) -> Dict[str, Any]:
        """Force OpenClaw to reload the generated plugin before the preflight probe.

        The local OpenClaw image starts the Gateway before we write the generated
        plugin/config, and its `/reload-config` API only validates JSON. We send
        the same SIGUSR1 that OpenClaw's watcher emits, wait for the in-process
        restart markers, and finally verify the Gateway's health endpoint.
        """
        signal_script = r"""python3 - <<'PY'
import json
import glob
import os
import signal
import subprocess
import sys
import time

process = subprocess.run(
    ["pgrep", "-xo", "openclaw"],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
if process.returncode != 0 or not process.stdout.strip():
    print(json.dumps({
        "ok": False,
        "error": "gateway_pid_not_found",
        "stderr": (process.stderr or "")[-500:],
    }, ensure_ascii=False))
    sys.exit(2)

gateway_pid = int(process.stdout.strip().splitlines()[0])
log_offsets = {}
for path in glob.glob("/tmp/openclaw/openclaw-*.log"):
    try:
        log_offsets[path] = os.path.getsize(path)
    except OSError:
        pass
try:
    os.kill(gateway_pid, signal.SIGUSR1)
except Exception as exc:
    print(json.dumps({
        "ok": False,
        "error": "gateway_signal_failed",
        "gateway_pid": gateway_pid,
        "detail": repr(exc)[:500],
    }, ensure_ascii=False))
    sys.exit(2)

restart_log = ""
deadline = time.time() + 30.0
restart_marker = "received SIGUSR1; restarting"
ready_marker = "gateway ready"
restart_observed = False
while time.time() < deadline:
    for path in glob.glob("/tmp/openclaw/openclaw-*.log"):
        offset = log_offsets.get(path, 0)
        try:
            size = os.path.getsize(path)
            if size < offset:
                offset = 0
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                handle.seek(offset)
                restart_log += handle.read()
                log_offsets[path] = handle.tell()
        except OSError:
            continue
    restart_index = restart_log.find(restart_marker)
    ready_index = (
        restart_log.find(ready_marker, restart_index + len(restart_marker))
        if restart_index >= 0
        else -1
    )
    if restart_index >= 0 and ready_index >= 0:
        restart_observed = True
        break
    time.sleep(0.1)

if not restart_observed:
    print(json.dumps({
        "ok": False,
        "error": "gateway_restart_not_observed",
        "gateway_pid": gateway_pid,
        "restart_log_tail": restart_log[-2000:],
    }, ensure_ascii=False))
    sys.exit(2)

print(json.dumps({
    "ok": True,
    "activation_refresh": "explicit_gateway_sigusr1",
    "gateway_pid": gateway_pid,
    "restart_observed": True,
}, ensure_ascii=False))
sys.exit(0)
PY"""
        signal_response = await client.execute_command(signal_script)
        signal_result = signal_response.get("result", {})
        signal_stdout = (signal_result.get("stdout") or "").strip()
        try:
            signal_parsed = json.loads(signal_stdout.splitlines()[-1]) if signal_stdout else {}
        except Exception:
            signal_parsed = {"ok": False, "raw": signal_stdout[-2000:]}
        if signal_result.get("exit_code") not in (0, None) or not signal_parsed.get("ok"):
            raise AgentRuntimeError(
                "failed to trigger OpenClaw gateway restart after service MCP plugin update: %s"
                % (
                    (signal_result.get("stderr") or signal_result.get("stdout") or json.dumps(signal_parsed, ensure_ascii=False))[
                        -2000:
                    ]
                )
            )

        health_script = r"""python3 - <<'PY'
import json
import sys
import time
import urllib.request

gateway_health_url = "http://127.0.0.1:18789/health"

def gateway_health():
    request = urllib.request.Request(gateway_health_url, method="GET")
    with urllib.request.urlopen(request, timeout=5.0) as response:
        raw = response.read().decode("utf-8", errors="replace").strip()
        body = json.loads(raw) if raw else {}
        return {"status": response.status, "body": body}

time.sleep(1.0)
deadline = time.time() + 30.0
last_error = ""
while time.time() < deadline:
    try:
        health = gateway_health()
        if health.get("status") == 200 and (health.get("body") or {}).get("ok") is True:
            print(json.dumps({
                "ok": True,
                "gateway_health_url": gateway_health_url,
            }, ensure_ascii=False))
            sys.exit(0)
    except Exception as exc:
        last_error = repr(exc)[:500]
    time.sleep(0.5)

print(json.dumps({
    "ok": False,
    "error": "gateway_health_timeout",
    "gateway_health_url": gateway_health_url,
    "detail": last_error,
}, ensure_ascii=False))
sys.exit(2)
PY"""
        response = await client.execute_command(health_script)
        result = response.get("result", {})
        stdout = (result.get("stdout") or "").strip()
        try:
            parsed = json.loads(stdout.splitlines()[-1]) if stdout else {}
        except Exception:
            parsed = {"ok": False, "raw": stdout[-2000:]}
        if result.get("exit_code") not in (0, None) or not parsed.get("ok"):
            raise AgentRuntimeError("OpenClaw gateway restart failed after service MCP plugin update: %s" % parsed)
        return {
            "ok": True,
            "activation_refresh": "explicit_gateway_sigusr1",
            "gateway_pid": signal_parsed.get("gateway_pid"),
            "restart_observed": signal_parsed.get("restart_observed") is True,
            "gateway_health_url": parsed.get("gateway_health_url"),
        }

    async def _verify_openclaw_generated_plugin_config(self, client: LocalDockerSandbox, servers: List[Dict[str, Any]]) -> Dict[str, Any]:
        script = r"""python3 - <<'PY'
import json
import sys
from pathlib import Path

servers = json.loads(__SERVERS__)
config_path = Path("/root/.openclaw/openclaw.json")
plugin_id = "red-agent-world-service-mcp"
plugin_root = Path("/root/.openclaw/extensions") / plugin_id

try:
    config = json.loads(config_path.read_text(encoding="utf-8"))
except Exception as exc:
    print(json.dumps({
        "ok": False,
        "registration": "generated_plugin_config",
        "error": "failed_to_read_openclaw_config",
        "detail": str(exc),
    }, ensure_ascii=False))
    sys.exit(2)

allowed_tools = set(config.get("tools", {}).get("alsoAllow", []))
expected_tools = [
        "%s__%s" % (server["name"], str(tool.get("name") or ""))
        for server in servers
        for tool in server.get("tools", [])
        if str(tool.get("name") or "").strip()
]
try:
    manifest = json.loads((plugin_root / "openclaw.plugin.json").read_text(encoding="utf-8"))
except Exception as exc:
    manifest = {"error": str(exc)}
plugins = config.get("plugins", {})
contracts = manifest.get("contracts", {}).get("tools", []) if isinstance(manifest, dict) else []

ok = (
    plugins.get("enabled") is True
    and plugin_id in plugins.get("allow", [])
    and plugins.get("entries", {}).get(plugin_id, {}).get("enabled") is True
    and not config.get("mcp", {}).get("servers")
    and set(contracts) == set(expected_tools)
    and len(contracts) == len(expected_tools) == len(set(expected_tools))
    and all(name in allowed_tools for name in expected_tools)
    and (plugin_root / "package.json").is_file()
    and (plugin_root / "index.mjs").is_file()
)
print(json.dumps({
    "ok": ok,
    "registration": "generated_plugin_config",
    "plugin_id": plugin_id,
    "expected_tools": expected_tools,
    "manifest_tools": contracts,
}, ensure_ascii=False))
sys.exit(0 if ok else 2)
PY""".replace("__SERVERS__", json.dumps(json.dumps(servers, ensure_ascii=False)))
        response = await client.execute_command(script)
        result = response.get("result", {})
        if result.get("exit_code") not in (0, None):
            raise AgentRuntimeError("failed to verify OpenClaw generated MCP plugin: %s" % ((result.get("stderr") or result.get("stdout") or "")[-2000:]))
        try:
            parsed = json.loads((result.get("stdout") or "{}").strip().splitlines()[-1])
        except Exception:
            parsed = {"ok": False, "raw": (result.get("stdout") or "")[-2000:]}
        if not parsed.get("ok"):
            raise AgentRuntimeError("OpenClaw generated MCP plugin verification failed: %s" % parsed)
        return parsed

    async def prepare_runtime(self, client: LocalDockerSandbox, item: Dict[str, Any]) -> Dict[str, Any]:
        setup = {"runtime": self.runtime_name, "workspace": SANDBOX_WORKSPACE}
        setup["service_mcp_registration"] = await self.install_openclaw_service_mcp_config(client)
        setup["openclaw_workspace_runtime_cleanup"] = await self.cleanup_openclaw_workspace_runtime_files(client)
        skill_exposure = await prepare_preplaced_skill_exposure(client, item, self.runtime_name)
        if skill_exposure is not None:
            setup["skill_exposure"] = skill_exposure
        return setup

    async def cleanup_openclaw_workspace_runtime_files(self, client: LocalDockerSandbox) -> Dict[str, Any]:
        """Keep OpenClaw attestation files in place during rollout.

        OpenClaw treats its workspace bootstrap files as part of workspace
        attestation. Moving them during a session triggers WorkspaceVanishedError.
        We disable prompt injection via config and leave evidence handling to the
        baseline/judge layer instead of mutating the runtime workspace.
        """
        return {"moved": [], "status": "preserved_for_openclaw_attestation"}

    def _expected_service_tool_names(self) -> List[str]:
        handle = self._active_service_sandbox
        if handle is None:
            return []
        expected: List[str] = []
        for server in handle.agent_mcp_servers:
            for tool in server.tools or []:
                expected.append(_openclaw_safe_tool_name(server.name, str(tool)))
        return expected

    def _openclaw_model_id(self) -> str:
        model = str(self.config.get("agent", {}).get("model", "")).strip()
        if not model:
            raise ValueError(
                "config.agent.model is required for OpenClaw; implicit model fallback is disabled"
            )
        return model if "/" in model else "DefaultProvider/%s" % model

    @staticmethod
    def _parsed_openclaw_payload(response_json: Any) -> Dict[str, Any]:
        """Return the innermost parsed OpenClaw response object."""
        payload = response_json if isinstance(response_json, dict) else {}
        nested = payload.get("response")
        if isinstance(nested, str):
            try:
                nested = json.loads(nested)
            except Exception:
                nested = None
        return nested if isinstance(nested, dict) else payload

    def _compact_openclaw_response(self, resp: Dict[str, Any]) -> Dict[str, Any]:
        """Keep one judge-facing response record without CLI wrapper duplication."""
        payload = self._parsed_openclaw_payload(resp.get("response_json"))
        result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
        completion = meta.get("completion") if isinstance(meta.get("completion"), dict) else {}

        final_text = str(meta.get("finalAssistantRawText") or "").strip()
        if not final_text:
            texts = []
            for item in result.get("payloads") or []:
                if isinstance(item, dict) and item.get("text"):
                    texts.append(str(item["text"]))
            final_text = "\n".join(texts).strip()

        cli = resp.get("openclaw_cli") if isinstance(resp.get("openclaw_cli"), dict) else {}
        return {
            "type": "openclaw_response",
            "response": {
                "session_id": resp.get("session_id"),
                "run_id": payload.get("runId"),
                "status": payload.get("status") or resp.get("status") or "completed",
                "summary": payload.get("summary"),
                "model": cli.get("model"),
                "stop_reason": meta.get("stopReason") or completion.get("stopReason"),
                "liveness_state": meta.get("livenessState"),
                "final_assistant_text": final_text,
            },
        }

    async def _send_openclaw_agent_message(self, client: LocalDockerSandbox, prompt: str, session_id: str, timeout: int) -> Dict[str, Any]:
        model = self._openclaw_model_id()
        script = r"""python3 - <<'PY'
import json
import os
import subprocess
import sys

prompt = __PROMPT_JSON__
session_id = __SESSION_ID_JSON__
model = __MODEL_JSON__
timeout = int(__TIMEOUT_SECONDS__)
cmd = [
    "openclaw",
    "agent",
    "--local",
    "--agent",
    "main",
    "--model",
    model,
    "--message",
    prompt,
    "--session-id",
    session_id,
    "--json",
    "--timeout",
    str(timeout),
    "--thinking",
    "off",
]
env = os.environ.copy()
env["OPENCLAW_CONFIG_PATH"] = "/root/.openclaw/openclaw.json"
def run_agent_command():
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout + 30,
        env=env,
    )

try:
    proc = run_agent_command()
    if proc.returncode != 0 and "Unknown model" in (proc.stderr or ""):
        subprocess.run(["openclaw", "models", "set", model], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, env=env)
        proc = run_agent_command()
except subprocess.TimeoutExpired as exc:
    print(json.dumps({"ok": False, "error": "timeout", "cmd": cmd, "stdout": (exc.stdout or "")[-4000:], "stderr": (exc.stderr or "")[-4000:]}, ensure_ascii=False))
    sys.exit(124)

parsed = None
try:
    parsed = json.loads(proc.stdout.strip())
except Exception:
    pass
for line in [] if parsed is not None else proc.stdout.splitlines():
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        parsed = json.loads(line)
    except Exception:
        pass
if isinstance(parsed, dict) and isinstance(parsed.get("meta"), dict) and "result" not in parsed:
    parsed = {"status": "completed", "result": parsed}

payload = {
    "ok": proc.returncode == 0,
    "cmd": cmd,
    "exit_code": proc.returncode,
    "stdout_tail": proc.stdout[-8000:],
    "stderr_tail": proc.stderr[-8000:],
    "parsed": parsed if parsed is not None else {"response": proc.stdout, "status": "completed"},
}
print(json.dumps(payload, ensure_ascii=False))
sys.exit(0 if proc.returncode == 0 else proc.returncode)
PY"""
        script = (
            script
            .replace("__PROMPT_JSON__", json.dumps(prompt, ensure_ascii=False))
            .replace("__SESSION_ID_JSON__", json.dumps(session_id, ensure_ascii=False))
            .replace("__MODEL_JSON__", json.dumps(model, ensure_ascii=False))
            .replace("__TIMEOUT_SECONDS__", str(int(timeout)))
        )
        response = await client.execute_command(script)
        result = response.get("result", {})
        stdout = (result.get("stdout") or "").strip()
        try:
            payload = json.loads(stdout.splitlines()[-1]) if stdout else {}
        except Exception:
            payload = {"ok": False, "stdout_tail": stdout[-8000:], "stderr_tail": (result.get("stderr") or "")[-8000:]}
        if result.get("exit_code") not in (0, None) or not payload.get("ok"):
            raise AgentRuntimeError("OpenClaw CLI prompt failed: %s" % json.dumps(payload, ensure_ascii=False)[-4000:])
        parsed = payload.get("parsed") if isinstance(payload.get("parsed"), dict) else {}
        return {
            "session_id": session_id,
            "status": parsed.get("status", "completed"),
            "response": json.dumps(parsed, ensure_ascii=False),
            "response_json": parsed,
            "openclaw_cli": {
                "agent": "main",
                "model": model,
                "exit_code": payload.get("exit_code"),
                "stderr_tail": payload.get("stderr_tail", ""),
            },
        }


    async def adapter_service_self_check(self, client: LocalDockerSandbox) -> Dict[str, Any]:
        handle = self._active_service_sandbox
        if handle is None or not handle.agent_mcp_servers:
            return {"status": "not_required", "runtime": self.runtime_name}
        sandbox_config = (self.config or {}).get("sandbox", {}) or {}
        max_attempts = int(sandbox_config.get("service_adapter_self_check_attempts", 1) or 1)
        retry_delay = float(sandbox_config.get("service_adapter_self_check_delay", 4.0) or 4.0)
        attempts: List[Dict[str, Any]] = []
        surface_report: Dict[str, Any] = {}
        for attempt in range(1, max(1, max_attempts) + 1):
            surface_report = await self.openclaw_agent_tool_surface_check(
                client,
                session_id="service-surface-preflight-%s" % attempt,
            )
            attempts.append({"attempt": attempt, **surface_report})
            if surface_report.get("status") == "ok":
                break
            if attempt < max(1, max_attempts):
                # OpenClaw can report gateway health before the reloaded agent
                # surface has projected generated plugin tools.
                await asyncio.sleep(retry_delay)
        report = {
            "status": "ok" if surface_report.get("status") == "ok" else "failed",
            "runtime": self.runtime_name,
            "endpoint_preflight": "validated_by_base_runner",
            "agent_tool_surface": surface_report,
            "attempts": attempts,
        }
        mode = str(self.config.get("sandbox", {}).get("service_adapter_self_check", "strict")).strip().lower()
        if report["status"] != "ok" and mode in {"strict", "true", "1", "required"}:
            raise AgentRuntimeError("OpenClaw service adapter self-check failed: %s" % report)
        return report

    async def openclaw_agent_tool_surface_check(
        self,
        client: LocalDockerSandbox,
        session_id: str = "service-surface-preflight",
    ) -> Dict[str, Any]:
        """Check the same one-shot OpenClaw CLI path used for real rollout."""
        expected = self._expected_service_tool_names()
        probe = "List available tool names only. Do not perform any business action."
        try:
            response = await self._send_openclaw_agent_message(client, probe, session_id=session_id, timeout=120)
        except Exception as exc:
            return {
                "status": "failed",
                "expected_service_tool_names": expected,
                "missing_service_tool_names": expected,
                "tool_names": [],
                "runtime_workspace_files_in_prompt": [],
                "leaked_runtime_workspace_files": [],
                "exit_code": 1,
                "stderr_tail": str(exc)[-1000:],
                "stdout_tail": "",
            }

        payload = response.get("response_json") if isinstance(response, dict) else None
        if not isinstance(payload, dict):
            raw_response = response.get("response") if isinstance(response, dict) else ""
            if isinstance(raw_response, str):
                try:
                    payload = json.loads(raw_response)
                except Exception:
                    payload = None
        if not isinstance(payload, dict):
            payload = response if isinstance(response, dict) else {}

        candidates: List[Dict[str, Any]] = []
        if isinstance(payload, dict):
            candidates.append(payload)
            nested = payload.get("response")
            if isinstance(nested, str):
                try:
                    parsed_nested = json.loads(nested)
                    if isinstance(parsed_nested, dict):
                        candidates.append(parsed_nested)
                except Exception:
                    pass
            if isinstance(payload.get("response_json"), dict):
                candidates.append(payload["response_json"])

        names: List[str] = []
        runtime_files: List[str] = []
        for candidate in candidates:
            report = (((candidate.get("result") or {}).get("meta") or {}).get("systemPromptReport") or {})
            entries = (((report.get("tools") or {}).get("entries")) or [])
            if entries:
                names = [str(entry.get("name")) for entry in entries if isinstance(entry, dict) and entry.get("name")]
            injected = report.get("injectedWorkspaceFiles", []) or []
            for item in injected:
                if isinstance(item, dict) and item.get("name"):
                    runtime_files.append(str(item["name"]))
            if names:
                break

        visible_canonical = {_canonical_openclaw_tool_name(name) for name in names}
        missing = [
            name
            for name in expected
            if _canonical_openclaw_tool_name(name) not in visible_canonical
        ]
        service_names_visible = bool(expected) and not missing
        runtime_file_set = set(runtime_files)
        leaked_runtime = sorted(runtime_file_set.intersection(OPENCLAW_WORKSPACE_RUNTIME_FILES))
        prompt_ok = isinstance(response, dict) and response.get("status") != "error"
        status = "ok" if prompt_ok and service_names_visible else "failed"
        return {
            "status": status,
            "expected_service_tool_names": expected,
            "missing_service_tool_names": missing,
            "tool_names": names,
            "runtime_workspace_files_in_prompt": runtime_files,
            "leaked_runtime_workspace_files": leaked_runtime,
            "exit_code": 0 if status == "ok" else 1,
            "stderr_tail": "",
            "stdout_tail": json.dumps(response, ensure_ascii=False)[-1000:] if isinstance(response, dict) else str(response)[-1000:],
            "checked_via": "openclaw_cli_explicit_model",
        }

    async def run_agent_turn(self, client: LocalDockerSandbox, item: Dict[str, Any], turn_idx: int, query: str, runtime_session_id: Optional[str]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        session_id = runtime_session_id or "item-%s" % int(item["id"])
        try:
            resp = await self._send_openclaw_agent_message(client, query, session_id=session_id, timeout=self.config["execution"]["timeout"])
        except Exception as exc:
            raise AgentRuntimeError(str(exc)) from exc
        if resp.get("status") == "error":
            raise AgentRuntimeError(str(resp.get("error") or resp))
        response_json = resp.get("response_json")
        payload = self._parsed_openclaw_payload(response_json)
        meta = (((payload.get("result") or {}).get("meta")) or {}) if isinstance(payload, dict) else {}
        stop_reason = str(meta.get("stopReason") or ((meta.get("completion") or {}).get("stopReason")) or "").lower()
        liveness = str(meta.get("livenessState") or "").lower()
        raw_text = str(meta.get("finalAssistantRawText") or "")
        if stop_reason == "error" or liveness == "blocked":
            raise AgentRuntimeError(
                "OpenClaw provider/runtime error: %s"
                % (raw_text[:1000] or json.dumps({"stopReason": stop_reason, "livenessState": liveness}, ensure_ascii=False))
            )
        records = [self._compact_openclaw_response(resp)]
        records.extend(await self._openclaw_service_tool_events(client, session_id))
        cleanup = await self.cleanup_openclaw_workspace_runtime_files(client)
        if cleanup.get("moved"):
            records.append({"type": "openclaw_workspace_runtime_cleanup", "data": cleanup})
        return records, session_id

    async def _openclaw_service_tool_events(
        self,
        client: LocalDockerSandbox,
        session_id: str,
    ) -> List[Dict[str, Any]]:
        path = "/root/.openclaw/agents/main/sessions/%s.jsonl" % session_id
        response = await client.execute_command("cat %s 2>/dev/null || true" % shlex.quote(path))
        text = response.get("result", {}).get("stdout") or ""
        rows: List[Dict[str, Any]] = []
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
        seen_usage_by_session = getattr(self, "_openclaw_seen_usage_responses", None)
        if not isinstance(seen_usage_by_session, dict):
            seen_usage_by_session = {}
            self._openclaw_seen_usage_responses = seen_usage_by_session
        seen_usage = seen_usage_by_session.setdefault(session_id, set())
        usage_events: List[Dict[str, Any]] = []
        for row in rows:
            message = row.get("message") if isinstance(row.get("message"), dict) else {}
            usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
            if message.get("role") != "assistant" or not usage:
                continue
            response_id = str(message.get("responseId") or row.get("id") or "")
            if not response_id or response_id in seen_usage:
                continue
            seen_usage.add(response_id)
            usage_events.append(
                {
                    "type": "openclaw_usage",
                    "runtime": self.runtime_name,
                    "response_id": response_id,
                    "usage": {
                        "input_tokens": int(usage.get("input") or 0),
                        "output_tokens": int(usage.get("output") or 0),
                        "cache_read_tokens": int(usage.get("cacheRead") or 0),
                        "cache_write_tokens": int(usage.get("cacheWrite") or 0),
                        "reasoning_tokens": int(usage.get("reasoningTokens") or 0),
                        "total_tokens": int(usage.get("totalTokens") or usage.get("total") or 0),
                    },
                }
            )
        results: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            message = row.get("message") if isinstance(row.get("message"), dict) else {}
            if message.get("role") == "toolResult" and message.get("toolCallId"):
                results[str(message["toolCallId"])] = message
        seen_by_session = getattr(self, "_openclaw_seen_tool_calls", None)
        if not isinstance(seen_by_session, dict):
            seen_by_session = {}
            self._openclaw_seen_tool_calls = seen_by_session
        seen_statuses = seen_by_session.setdefault(session_id, {})
        events: List[Dict[str, Any]] = usage_events
        for row in rows:
            message = row.get("message") if isinstance(row.get("message"), dict) else {}
            if message.get("role") != "assistant":
                continue
            content = message.get("content") if isinstance(message.get("content"), list) else []
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "toolCall":
                    continue
                identity = self.service_tool_identity(str(part.get("name") or ""))
                if identity is None:
                    continue
                arguments = part.get("arguments") if isinstance(part.get("arguments"), dict) else {}
                call_id = str(part.get("id") or "")
                result = results.get(call_id)
                status = "completed" if result is not None else "in_progress"
                previous_status = seen_statuses.get(call_id)
                if previous_status == "completed" or previous_status == status:
                    continue
                seen_statuses[call_id] = status
                events.append({
                    "type": "mcp_tool_call",
                    "transport": "openclaw_plugin_mcp_bridge",
                    "id": call_id,
                    "server": identity[0],
                    "tool": identity[1],
                    "arguments": arguments,
                    "status": status,
                    "result": result.get("content") if result else None,
                    "error": result.get("content") if result and result.get("isError") else None,
                })
        return events


async def main() -> None:
    await run_cli(OpenClawServiceSandboxRunner, "Service-backed OpenClaw runner")


if __name__ == "__main__":
    asyncio.run(main())
