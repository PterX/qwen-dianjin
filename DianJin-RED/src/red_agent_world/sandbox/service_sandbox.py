"""Service-backed RED sandbox built around native service worlds.

This module intentionally keeps the public name "sandbox" semantics while
changing the implementation model from a single fake workspace container to a
service-backed world:

* Profile-backed MCP services provide deterministic service state.
* Agent-visible MCP servers expose normal user capabilities.
* Red-team injection MCP servers seed or poison the environment.
* Evidence is collected from service state, MCP/action logs, trajectory, then
  final answer as a fallback.

The current implementation is a profile-backed service world. Gmail is backed
by Mailpit, while some other worlds can still use lightweight deterministic
services behind the same reset/seed/MCP/evidence contract.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import socket
import subprocess
import time
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import yaml
except Exception:  # pragma: no cover - dependency is present in rollout env
    yaml = None  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SERVICE_WORLD_ROOT = Path(os.environ.get("RED_AGENT_WORLD_SERVICE_WORLD_ROOT", REPO_ROOT / "sandbox" / "service_world"))
DEFAULT_SERVICE_PROFILE_CONFIG = "config/service_profiles.yaml"
logger = logging.getLogger(__name__)


class ServiceSandboxError(RuntimeError):
    """Raised when a service-backed sandbox cannot be prepared."""


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise ServiceSandboxError("PyYAML is required for service sandbox config")
    if not path.exists():
        raise ServiceSandboxError(f"missing YAML config: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ServiceSandboxError(f"YAML root must be a mapping: {path}")
    return value


def _load_optional_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return _load_yaml(path)


def _find_free_port(start: int, end: int = 65000) -> int:
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("", port))
            except OSError:
                continue
            return port
    raise ServiceSandboxError(f"no free port found from {start} to {end}")




def _find_ephemeral_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])

def _agent_mcp_host() -> str:
    # Docker adds this host alias for isolated runtime networks. Unlike
    # network_mode=host, it exposes only host-published benchmark endpoints.
    return os.environ.get("RED_AGENT_WORLD_AGENT_MCP_HOST", "host.docker.internal")


def _safe_project_component(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value or "")).strip("_").lower()
    return cleaned[:48] or "run"


def _render_template(text: str, values: Dict[str, Any]) -> str:
    out = str(text)
    for key, value in values.items():
        out = out.replace("${%s}" % key, str(value))
    return out


def _request_url(method: str, url: str, timeout: float = 20.0) -> Dict[str, Any]:
    req = urllib.request.Request(url, data=b"{}" if method.upper() != "GET" else None, headers={"Content-Type": "application/json"}, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return {"ok": True, "status": response.status, "body": raw}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "body": exc.read().decode("utf-8", errors="replace")}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _request_json(method: str, url: str, payload: Dict[str, Any], timeout: float = 20.0) -> Dict[str, Any]:
    data = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data if method.upper() != "GET" else None, headers={"Content-Type": "application/json"}, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            body = json.loads(raw) if raw.strip() else {}
            return {"ok": True, "status": response.status, "body": body}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body: Any = json.loads(raw) if raw.strip() else {}
        except Exception:
            body = raw
        return {"ok": False, "status": exc.code, "body": body}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _parse_mcp_http_body(raw: str) -> Dict[str, Any]:
    for line in raw.splitlines():
        if line.startswith("data:"):
            try:
                value = json.loads(line.split(":", 1)[1].strip())
                return value if isinstance(value, dict) else {"value": value}
            except Exception:
                continue
    value = json.loads(raw)
    return value if isinstance(value, dict) else {"value": value}


def _server_map(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(server.get("name")): server for server in config.get("servers", []) if server.get("name")}


def _profile_services(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    services = config.get("services", {})
    return services if isinstance(services, dict) else {}


def _profile_by_environment(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for name, service in _profile_services(config).items():
        if isinstance(service, dict):
            out[str(service.get("environment") or name)] = service
    return out


@dataclass
class ServiceMCPServer:
    """Agent-visible or red-team-only MCP server endpoint."""

    name: str
    url: str
    transport: str = "http"
    role: str = "agent"
    environment: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    tool_specs: List[Dict[str, Any]] = field(default_factory=list)

    def as_agent_config(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "transport": self.transport,
            "role": self.role,
            "environment": self.environment,
            "tools": self.tools,
            "tool_specs": self.tool_specs,
        }


@dataclass
class ServiceSandboxHandle:
    """Materialized service sandbox state shared by all agent adapters."""

    sandbox_id: str
    service_world_root: Path
    environments: List[str]
    ports: Dict[str, int]
    agent_mcp_servers: List[ServiceMCPServer]
    injection_mcp_servers: List[ServiceMCPServer]
    project_names: Dict[str, str] = field(default_factory=dict)
    evidence_dir: Optional[Path] = None
    physical_workspace_dir: Optional[Path] = None
    started_at: float = field(default_factory=time.time)

    def agent_mcp_config(self) -> Dict[str, Any]:
        """Return a transport-neutral MCP config consumed by all adapters."""
        return {
            "mcpServers": {
                server.name: {
                    "type": server.transport,
                    "url": server.url,
                }
                for server in self.agent_mcp_servers
            }
        }

    def env_exports(self) -> Dict[str, str]:
        # Keep env exports compact; full tools/list specs live in manifests and adapter contracts.
        env = {
            "RED_AGENT_WORLD_SERVICE_SANDBOX": "1",
            "RED_AGENT_WORLD_SERVICE_MCP": json.dumps(
                [
                    {
                        "name": server.name,
                        "url": server.url,
                        "transport": server.transport,
                        "role": server.role,
                        "environment": server.environment,
                        "tools": server.tools,
                    }
                    for server in self.agent_mcp_servers
                ],
                ensure_ascii=False,
            ),
        }
        for key, value in self.ports.items():
            env[key] = str(value)
        if self.evidence_dir:
            env["RED_AGENT_WORLD_SERVICE_STATE_DIR"] = str(self.evidence_dir / "service_state")
        if self.physical_workspace_dir:
            env["WORKSPACE_HOST_DIR"] = str(self.physical_workspace_dir)
        for env_name, project in self.project_names.items():
            env[f"{env_name.upper().replace('-', '_')}_PROJECT_NAME"] = project
        return env


class ServiceSandboxSession:
    """Start and collect the services named by a normalized initialization plan."""

    def __init__(
        self,
        runtime_spec: Dict[str, Any],
        sandbox_id: str,
        service_world_root: Path | str = DEFAULT_SERVICE_WORLD_ROOT,
        evidence_root: Path | str | None = None,
        physical_workspace_dir: Path | str | None = None,
        dry_run: bool = False,
        tool_aliases: Optional[Dict[str, Dict[str, str]]] = None,
        tool_descriptions: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        if not isinstance(runtime_spec, dict):
            raise TypeError("runtime_spec must be an object")
        allowed = {"services", "mcp_servers", "injection_mcp_servers", "start_services"}
        extra = sorted(set(runtime_spec) - allowed)
        if extra:
            raise ValueError("unsupported service runtime fields: %s" % ", ".join(extra))
        self.runtime_spec = dict(runtime_spec)
        self.sandbox_id = str(sandbox_id)
        self.service_world_root = Path(service_world_root)
        self.evidence_root = (Path(evidence_root) if evidence_root else REPO_ROOT / "tmp" / "service_sandbox_evidence").resolve()
        self.physical_workspace_dir = Path(physical_workspace_dir).resolve() if physical_workspace_dir else None
        self.dry_run = dry_run
        self.tool_aliases = {
            str(service): {str(canonical): str(alias) for canonical, alias in aliases.items()}
            for service, aliases in (tool_aliases or {}).items()
            if isinstance(aliases, dict)
        }
        self.tool_descriptions = {
            str(service): {str(canonical): str(description) for canonical, description in descriptions.items()}
            for service, descriptions in (tool_descriptions or {}).items()
            if isinstance(descriptions, dict)
        }
        self.env_config = _load_yaml(self.service_world_root / "config" / "env.yaml")
        self.mcp_config = _load_yaml(self.service_world_root / "config" / "mcp.yaml")
        self.injection_config = _load_yaml(self.service_world_root / "config" / "injection_mcp.yaml")
        self.profile_config = _load_optional_yaml(self.service_world_root / DEFAULT_SERVICE_PROFILE_CONFIG)
        self._processes: List[subprocess.Popen] = []
        self.handle: Optional[ServiceSandboxHandle] = None
        self._endpoint_health: Dict[str, Dict[str, Any]] = {}
        self._mcp_session_ids: Dict[str, str] = {}
        self._mcp_initialized: set[str] = set()
        self._seeded_external_paths: set[str] = set()
        self._external_records_baseline: Dict[str, str] = {}
        self._external_records_baseline_captured = False

    def _sandbox_service_names(self) -> List[str]:
        values = self.runtime_spec.get("services", [])
        if not isinstance(values, list):
            raise ValueError("runtime_spec.services must be a list")
        profiles = _profile_services(self.profile_config)
        out: List[str] = []
        for value in values:
            service = str(value)
            if service not in profiles:
                raise ValueError("unknown service profile: %s" % service)
            if service not in out:
                out.append(service)
        return out

    def _dedupe(self, values: Iterable[str]) -> List[str]:
        out: List[str] = []
        for value in values:
            if value and value not in out:
                out.append(value)
        return out

    def _uses_records_drive_mount(self) -> bool:
        services = set(self._sandbox_service_names())
        raw_servers = set(self._canonical_server_names("mcp_servers"))
        return "external_files" in services or "external_files" in raw_servers

    def _canonical_server_names(self, field: str) -> List[str]:
        values = self.runtime_spec.get(field, [])
        if not isinstance(values, list) or not all(isinstance(value, str) and value for value in values):
            raise ValueError("runtime_spec.%s must be a list of non-empty strings" % field)
        return self._dedupe(values)

    def _requested_agent_servers(self) -> List[str]:
        return self._canonical_server_names("mcp_servers")

    def _requested_injection_servers(self) -> List[str]:
        return self._canonical_server_names("injection_mcp_servers")

    def _required_environments(self, agent_servers: Iterable[str], injection_servers: Iterable[str]) -> List[str]:
        envs: set[str] = set()
        mcp_servers = _server_map(self.mcp_config)
        injection_servers_cfg = _server_map(self.injection_config)
        profile_agent = self._profile_server_by_role("agent")
        profile_injection = self._profile_server_by_role("injection")
        for name in agent_servers:
            env_value = mcp_servers.get(name, {}).get("environment") or profile_agent.get(name, {}).get("environment")
            if isinstance(env_value, list):
                envs.update(str(v) for v in env_value)
            elif env_value:
                envs.add(str(env_value))
        for name in injection_servers:
            env_value = injection_servers_cfg.get(name, {}).get("target_environment") or profile_injection.get(name, {}).get("target_environment")
            if env_value:
                envs.add(str(env_value))
        return sorted(envs)

    def _allocate_ports(self, envs: Iterable[str]) -> Dict[str, int]:
        allocated: Dict[str, int] = {}
        used_ports: set[int] = set()
        env_defs = self.env_config.get("environments", {})
        profile_by_env = {
            str(service.get("environment") or name): service
            for name, service in _profile_services(self.profile_config).items()
            if isinstance(service, dict)
        }
        for env_name in envs:
            env_def = env_defs.get(env_name, {})
            ports = dict(env_def.get("ports") or {})
            ports.update(profile_by_env.get(env_name, {}).get("ports") or {})
            for var_name, _meta in ports.items():
                for _ in range(32):
                    port = _find_ephemeral_port()
                    if port not in used_ports:
                        used_ports.add(port)
                        allocated[var_name] = port
                        break
                else:
                    raise ServiceSandboxError(f"could not allocate unique port for {var_name}")
        return allocated

    def _profile_server_by_role(self, role: str) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        key = "agent_mcp" if role == "agent" else "injection_mcp"
        env_key = "environment" if role == "agent" else "target_environment"
        for service_name, service in _profile_services(self.profile_config).items():
            if not isinstance(service, dict) or not isinstance(service.get(key), dict):
                continue
            server = dict(service[key])
            name = str(server.get("name") or service_name)
            server["name"] = name
            server[env_key] = str(service.get("environment") or service_name)
            server["__profile"] = True
            out[name] = server
        return out

    def _server_by_name(self, server_name: str, config: Dict[str, Any], role: str) -> Optional[Dict[str, Any]]:
        profile_server = self._profile_server_by_role(role).get(server_name)
        if profile_server:
            return profile_server
        return _server_map(config).get(server_name)

    def _mcp_endpoint(self, server_name: str, config: Dict[str, Any], ports: Dict[str, int], role: str) -> ServiceMCPServer:
        server = self._server_by_name(server_name, config, role)
        if not server:
            raise ServiceSandboxError(f"MCP server not found in config: {server_name}")
        raw_port = server.get("port") or server.get("env", {}).get("PORT") or _find_free_port(8800)
        try:
            port = int(_render_template(str(raw_port), ports))
        except Exception:
            port = _find_free_port(8800)
        if str(server.get("env", {}).get("PORT", "")).strip():
            port = int(_render_template(server["env"]["PORT"], ports))
        # PORT is the listener for this endpoint. Other *_MCP_PORT variables
        # may identify a backend that the process calls (for example the
        # canonical Browser MCP used by browser-injection), so they must not
        # override an explicit PORT value.
        explicit_port = str((server.get("env") or {}).get("PORT", "")).strip()
        if not explicit_port:
            for key, value in (server.get("env") or {}).items():
                rendered = _render_template(str(value), ports)
                if key.endswith("MCP_PORT") and rendered.isdigit():
                    port = int(rendered)
        if server_name.endswith("-injection") and not server.get("__profile"):
            for key, value in (server.get("env") or {}).items():
                if key.endswith("_PORT") and str(value).isdigit():
                    port = int(value)
                    break
        host = _agent_mcp_host() if role == "agent" else "127.0.0.1"
        url = f"http://{host}:{port}/mcp"
        env_value = server.get("environment") or server.get("target_environment")
        if isinstance(env_value, list):
            env_value = ",".join(str(v) for v in env_value)
        return ServiceMCPServer(
            name=server_name,
            url=url,
            transport=str(server.get("transport", "http")),
            role=role,
            environment=str(env_value) if env_value else None,
        )

    async def start(self) -> ServiceSandboxHandle:
        agent_server_names = self._requested_agent_servers()
        injection_server_names = self._requested_injection_servers()
        envs = self._required_environments(agent_server_names, injection_server_names)
        ports = self._allocate_ports(envs)
        evidence_dir = self.evidence_root / f"item-{self.sandbox_id}"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        if self.physical_workspace_dir:
            self.physical_workspace_dir.mkdir(parents=True, exist_ok=True)
        handle = ServiceSandboxHandle(
            sandbox_id=self.sandbox_id,
            service_world_root=self.service_world_root,
            environments=envs,
            ports=ports,
            agent_mcp_servers=[
                self._mcp_endpoint(name, self.mcp_config, ports, role="agent")
                for name in agent_server_names
            ],
            injection_mcp_servers=[
                self._mcp_endpoint(name, self.injection_config, ports, role="injection")
                for name in injection_server_names
            ],
            project_names={
                env: "red_%s_%s_%s_%s" % (
                    _safe_project_component(env),
                    _safe_project_component(os.environ.get("RED_AGENT_WORLD_RUNTIME_NAME", "runtime")),
                    _safe_project_component(os.environ.get("RED_AGENT_WORLD_RUN_ID", uuid.uuid4().hex[:8])),
                    _safe_project_component(str(self.sandbox_id)),
                )
                for env in envs
            },
            evidence_dir=evidence_dir,
            physical_workspace_dir=self.physical_workspace_dir,
        )
        self.handle = handle
        await self._write_manifest(handle)
        if not self.dry_run and self.runtime_spec.get("start_services", False):
            await self._start_compose_environments(handle)
            await self._start_mcp_servers(handle)
            await self._wait_mcp_endpoints(handle)
            await self._refresh_mcp_tool_lists(handle)
            await self._write_manifest(handle)
        return handle

    async def _write_manifest(self, handle: ServiceSandboxHandle) -> None:
        if not handle.evidence_dir:
            return
        (handle.evidence_dir / "service_state").mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "red-agent-world-service-sandbox-v1",
            "backend_kind": "profile-backed-service-world",
            "backend_note": "Services are started from service_profiles.yaml; Gmail uses local Mailpit, banking uses Blnk/PostgreSQL/Redis, and browser uses the repository-local Playwright service.",
            "sandbox_id": handle.sandbox_id,
            "service_world_root": str(handle.service_world_root),
            "environments": handle.environments,
            "ports": handle.ports,
            "project_names": handle.project_names,
            "physical_workspace": (
                {
                    "host_path": str(handle.physical_workspace_dir),
                    "container_path": "/workspace",
                    "mode": "shared-bind-mount",
                }
                if handle.physical_workspace_dir else {}
            ),
            "agent_mcp_servers": [server.as_agent_config() for server in handle.agent_mcp_servers],
            "injection_mcp_servers": [server.as_agent_config() for server in handle.injection_mcp_servers],
            "runtime_health": [
                {
                    "name": server.name,
                    "role": server.role,
                    "url": server.url,
                    **self._endpoint_health.get(server.name, {"ok": False, "status": "not_checked"}),
                }
                for server in [*handle.agent_mcp_servers, *handle.injection_mcp_servers]
            ],
            "tool_inventory": [
                {
                    "name": server.name,
                    "role": server.role,
                    "url": server.url,
                    "tools": server.tools,
                }
                for server in [*handle.agent_mcp_servers, *handle.injection_mcp_servers]
            ],
            "runtime_spec": self.runtime_spec,
        }
        (handle.evidence_dir / "service_sandbox_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    async def _start_compose_environments(self, handle: ServiceSandboxHandle) -> None:
        env_defs = self.env_config.get("environments", {})
        profile_by_env = _profile_by_environment(self.profile_config)
        for env_name in handle.environments:
            profile = profile_by_env.get(env_name, {})
            env_def = env_defs.get(env_name, {})
            compose_rel = profile.get("compose") or env_def.get("docker_compose")
            if not compose_rel:
                continue
            compose_file = (REPO_ROOT / compose_rel).resolve() if profile.get("compose") else (self.service_world_root / compose_rel).resolve()
            env = os.environ.copy()
            env.update(handle.env_exports())
            cmd = [
                "docker",
                "compose",
                "-p",
                handle.project_names[env_name],
                "-f",
                str(compose_file),
                "up",
                "-d",
                "--wait",
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(compose_file.parent),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise ServiceSandboxError(
                    f"compose up failed for {env_name}: {(stderr or stdout).decode(errors='replace')[:2000]}"
                )
            await self._wait_profile_health(env_name, profile, handle)
            await self._reset_profile_environment(env_name, profile, handle)

    async def _wait_profile_health(self, env_name: str, profile: Dict[str, Any], handle: ServiceSandboxHandle) -> None:
        health = profile.get("health") if isinstance(profile, dict) else None
        if not isinstance(health, dict) or not health.get("url"):
            return
        url = _render_template(str(health["url"]), handle.ports)
        method = str(health.get("method") or "GET")
        deadline = time.time() + float(health.get("timeout_seconds") or 60)
        last = {}
        while time.time() < deadline:
            last = await asyncio.to_thread(_request_url, method, url, 5.0)
            if last.get("ok"):
                return
            await asyncio.sleep(2.0)
        raise ServiceSandboxError(f"health check failed for {env_name} at {url}: {last}")

    async def _reset_profile_environment(self, env_name: str, profile: Dict[str, Any], handle: ServiceSandboxHandle) -> None:
        reset = profile.get("reset") if isinstance(profile, dict) else None
        if not isinstance(reset, dict) or not reset.get("url"):
            return
        url = _render_template(str(reset["url"]), handle.ports)
        method = str(reset.get("method") or "POST")
        result = await asyncio.to_thread(_request_url, method, url, 20.0)
        if not result.get("ok"):
            note = str(reset.get("note") or "")
            if note:
                logger.warning("reset skipped/failed for %s at %s: %s (%s)", env_name, url, result, note)
                return
            raise ServiceSandboxError(f"reset failed for {env_name} at {url}: {result}")

    def _server_config_for_endpoint(self, endpoint: ServiceMCPServer) -> Dict[str, Any]:
        cfg = self.injection_config if endpoint.role == "injection" else self.mcp_config
        server = self._server_by_name(endpoint.name, cfg, endpoint.role)
        if not server:
            raise ServiceSandboxError(f"server config not found for {endpoint.name}")
        return server

    def _endpoint_port(self, endpoint: ServiceMCPServer) -> str:
        return endpoint.url.rsplit(":", 1)[-1].split("/", 1)[0]

    def _server_env(self, endpoint: ServiceMCPServer, handle: ServiceSandboxHandle) -> Dict[str, str]:
        server = self._server_config_for_endpoint(endpoint)
        env = os.environ.copy()
        env.update(handle.env_exports())
        for key, value in (server.get("env") or {}).items():
            if value is None or value == "":
                if key == "PORT" or key.endswith("_PORT"):
                    env[key] = self._endpoint_port(endpoint)
                else:
                    env[key] = ""
            else:
                env[key] = _render_template(str(value), handle.ports)
        if "PORT" not in env:
            env["PORT"] = self._endpoint_port(endpoint)
        aliases = self.tool_aliases.get(endpoint.name, {}) if endpoint.role == "agent" else {}
        if aliases:
            env["RED_AGENT_WORLD_TOOL_ALIASES"] = json.dumps(aliases, ensure_ascii=False, sort_keys=True)
        descriptions = self.tool_descriptions.get(endpoint.name, {}) if endpoint.role == "agent" else {}
        if descriptions:
            env["RED_AGENT_WORLD_TOOL_DESCRIPTIONS"] = json.dumps(descriptions, ensure_ascii=False, sort_keys=True)
        return env

    def _server_command(self, endpoint: ServiceMCPServer, handle: ServiceSandboxHandle) -> tuple[List[str], Path]:
        server = self._server_config_for_endpoint(endpoint)
        if server.get("__profile"):
            base = REPO_ROOT
        else:
            base_dir = self.injection_config.get("global", {}).get("base_dir") if endpoint.role == "injection" else self.mcp_config.get("global", {}).get("base_dir")
            base = (self.service_world_root / str(base_dir or ".")).resolve()
        main_path = (base / str(server["path"])).resolve()
        env = self._server_env(endpoint, handle)
        if "command" in server:
            cmd = []
            for part in server["command"]:
                expanded = str(part)
                for key, value in env.items():
                    expanded = expanded.replace(f"${{{key}}}", str(value)).replace(f"${key}", str(value))
                cmd.append(expanded)
        else:
            python_exe = self.mcp_config.get("global", {}).get("python_executable", "python3")
            cmd = [str(python_exe), str(main_path)]
        return cmd, main_path.parent

    async def _start_mcp_servers(self, handle: ServiceSandboxHandle) -> None:
        if not handle.evidence_dir:
            raise ServiceSandboxError("evidence_dir is required for MCP logs")
        log_dir = handle.evidence_dir / "mcp_process_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        for endpoint in [*handle.agent_mcp_servers, *handle.injection_mcp_servers]:
            server = self._server_config_for_endpoint(endpoint)
            if server.get("managed_by_compose"):
                continue
            cmd, cwd = self._server_command(endpoint, handle)
            env = self._server_env(endpoint, handle)
            stdout = (log_dir / f"{endpoint.name}_stdout.log").open("wb")
            stderr = (log_dir / f"{endpoint.name}_stderr.log").open("wb")
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                env=env,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            self._processes.append(proc)
        await asyncio.sleep(1.0)
        failed = [proc for proc in self._processes if proc.poll() is not None]
        if failed:
            raise ServiceSandboxError(f"{len(failed)} MCP server process(es) exited during startup")

    def _mcp_management_url(self, endpoint: ServiceMCPServer, path: str) -> str:
        base = endpoint.url.rsplit("/", 1)[0]
        return base + path

    def _mcp_health_sync(self, endpoint: ServiceMCPServer) -> Dict[str, Any]:
        return _request_url("GET", self._mcp_management_url(endpoint, "/health"), timeout=5.0)

    def _mcp_endpoint_key(self, endpoint: ServiceMCPServer) -> str:
        return f"{endpoint.role}:{endpoint.name}:{endpoint.url}"

    def _mcp_post_sync(self, endpoint: ServiceMCPServer, payload: Dict[str, Any], timeout: float = 10.0) -> tuple[Dict[str, Any], Dict[str, str]]:
        key = self._mcp_endpoint_key(endpoint)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._mcp_session_ids.get(key):
            headers["Mcp-Session-Id"] = self._mcp_session_ids[key]
        req = urllib.request.Request(
            endpoint.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            response_headers = {str(name).lower(): str(value) for name, value in response.headers.items()}
        return (_parse_mcp_http_body(raw) if raw.strip() else {}), response_headers

    def _ensure_mcp_initialized_sync(self, endpoint: ServiceMCPServer) -> None:
        key = self._mcp_endpoint_key(endpoint)
        if key in self._mcp_initialized:
            return
        initialize = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000) % 100000000,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "red-agent-world-harness", "version": "1"},
            },
        }
        _result, headers = self._mcp_post_sync(endpoint, initialize)
        session_id = headers.get("mcp-session-id")
        if session_id:
            self._mcp_session_ids[key] = session_id
        self._mcp_post_sync(endpoint, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        self._mcp_initialized.add(key)

    def _mcp_tools_list_sync(self, endpoint: ServiceMCPServer) -> Dict[str, Any]:
        self._ensure_mcp_initialized_sync(endpoint)
        payload = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000) % 100000000,
            "method": "tools/list",
            "params": {},
        }
        response, _headers = self._mcp_post_sync(endpoint, payload)
        return response

    async def _wait_mcp_endpoints(self, handle: ServiceSandboxHandle) -> None:
        endpoints = [*handle.agent_mcp_servers, *handle.injection_mcp_servers]
        for endpoint in endpoints:
            deadline = time.time() + 30.0
            last: Dict[str, Any] = {}
            while time.time() < deadline:
                try:
                    health = await asyncio.to_thread(self._mcp_health_sync, endpoint)
                    response = await asyncio.to_thread(self._mcp_tools_list_sync, endpoint)
                    if isinstance(response, dict) and "result" in response:
                        last = {"ok": bool(health.get("ok")), "health": health}
                        break
                    last = {"ok": False, "response": response}
                except Exception as exc:
                    last = {"ok": False, "error": str(exc)}
                await asyncio.sleep(1.0)
            if not last.get("ok"):
                raise ServiceSandboxError(f"MCP endpoint health check failed for {endpoint.name} at {endpoint.url}: {last}")
            self._endpoint_health[endpoint.name] = last

    async def _refresh_mcp_tool_lists(self, handle: ServiceSandboxHandle) -> None:
        for endpoint in [*handle.agent_mcp_servers, *handle.injection_mcp_servers]:
            try:
                response = await asyncio.to_thread(self._mcp_tools_list_sync, endpoint)
                tools = response.get("result", {}).get("tools", []) if isinstance(response, dict) else []
                endpoint.tool_specs = [tool for tool in tools if isinstance(tool, dict) and tool.get("name")]
                endpoint.tools = [str(tool.get("name")) for tool in endpoint.tool_specs]
            except Exception as exc:
                logger.warning("failed to read tools/list for %s at %s: %s", endpoint.name, endpoint.url, exc)

    def _injection_endpoint_by_name(self, server_name: str) -> ServiceMCPServer:
        handle = self.handle
        if handle is None:
            raise ServiceSandboxError("sandbox has not been started")
        for endpoint in handle.injection_mcp_servers:
            if endpoint.name == server_name:
                return endpoint
        raise ServiceSandboxError(f"injection server is not active: {server_name}")

    def _agent_endpoint_by_name(self, server_name: str) -> ServiceMCPServer:
        handle = self.handle
        if handle is None:
            raise ServiceSandboxError("sandbox has not been started")
        for endpoint in handle.agent_mcp_servers:
            if endpoint.name == server_name:
                return endpoint
        raise ServiceSandboxError(f"agent server is not active: {server_name}")

    def call_mcp_tool_sync(self, endpoint: ServiceMCPServer, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_mcp_initialized_sync(endpoint)
        payload = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000) % 100000000,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {},
            },
        }
        try:
            response, _headers = self._mcp_post_sync(endpoint, payload)
            return response
        except Exception:
            raise

    def configure_tool_injection_sync(self, endpoint: ServiceMCPServer, tool_name: str, step: Dict[str, Any]) -> Dict[str, Any]:
        mode = str(step.get("mode") or "append")
        if mode not in {"append", "prefix", "override"}:
            raise ValueError("unsupported tool-output mode: %s" % mode)
        payload = {
            "tool": tool_name or "*",
            "content": str(step.get("content") or ""),
            "mode": mode,
            "once": bool(step.get("once", False)),
        }
        return _request_json("POST", self._mcp_management_url(endpoint, "/inject"), payload, timeout=10.0)

    async def materialize_service_seeds(self, seeds: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for index, seed in enumerate(seeds or []):
            seed_type = str(seed.get("type") or "")
            target = str(seed.get("injection_mcp_tool") or seed.get("injected_tool") or "")
            base = {
                "seed_index": index,
                "kind": "tool_output" if seed_type == "tool" else "resource",
                "target": target,
            }
            if ":" not in target:
                results.append({**base, "success": False, "error": f"invalid seed target: {target}"})
                continue
            server_name, tool_name = target.split(":", 1)
            if seed_type == "tool":
                try:
                    endpoint = self._agent_endpoint_by_name(server_name)
                    response = await asyncio.to_thread(self.configure_tool_injection_sync, endpoint, tool_name, seed)
                    row = {
                        **base,
                        "success": bool(response.get("ok")),
                        "server_name": server_name,
                        "tool_name": tool_name,
                        "response": response,
                    }
                except Exception as exc:
                    row = {
                        **base,
                        "success": False,
                        "server_name": server_name,
                        "tool_name": tool_name,
                        "error": str(exc),
                    }
            else:
                args = dict(seed.get("kwargs") or {})
                try:
                    endpoint = self._injection_endpoint_by_name(server_name)
                    response = await asyncio.to_thread(self.call_mcp_tool_sync, endpoint, tool_name, args)
                    row = {
                        **base,
                        "success": bool(response.get("ok", "error" not in response)),
                        "server_name": server_name,
                        "tool_name": tool_name,
                        "response": response,
                    }
                    if row["success"] and server_name == "external_files-injection":
                        file_path = str(args.get("file_path") or "")
                        prefix = "/srv/external-files/"
                        if file_path.startswith(prefix):
                            self._seeded_external_paths.add(file_path[len(prefix):].lstrip("/"))
                except Exception as exc:
                    row = {
                        **base,
                        "success": False,
                        "server_name": server_name,
                        "tool_name": tool_name,
                        "error": str(exc),
                    }
            results.append(row)
        return results

    def agent_service_exercise_report(self, evidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        handle = self.handle
        if handle is None:
            return {"required": False, "exercised": False, "agent_mcp_servers": [], "agent_actions": []}
        agent_names = {server.name for server in handle.agent_mcp_servers}
        injection_names = {server.name for server in handle.injection_mcp_servers}
        action_rows: List[Dict[str, Any]] = []
        seen_action_keys = set()

        def consider(rel: str, row: Dict[str, Any]) -> None:
            service = str(row.get("service") or "")
            if row.get("actor") != "agent" or service not in agent_names:
                return
            tool = row.get("tool")
            arguments = row.get("arguments") if isinstance(row.get("arguments"), dict) else {}
            if arguments.get("__harness_preflight"):
                return
            key = (service, str(tool), json.dumps(arguments, ensure_ascii=False, sort_keys=True), row.get("ts"))
            if key in seen_action_keys:
                return
            seen_action_keys.add(key)
            action_rows.append({"path": rel, "service": service, "tool": tool, "arguments": arguments})

        blobs: Dict[str, str] = {}
        if isinstance(evidence, dict):
            for bucket in ("action_ledger",):
                value = evidence.get(bucket)
                if isinstance(value, dict):
                    blobs.update({str(k): str(v) for k, v in value.items()})
        elif handle.evidence_dir:
            for path in handle.evidence_dir.rglob("*"):
                if path.is_file() and (path.name.endswith(".jsonl") or path.name.endswith("_events.jsonl")):
                    blobs[str(path.relative_to(handle.evidence_dir))] = path.read_text(encoding="utf-8", errors="replace")

        for rel, content in blobs.items():
            if not (rel.endswith(".jsonl") or rel.endswith("_events.jsonl")):
                continue
            for line in content.splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict):
                    consider(rel, row)
        return {
            "required": bool(agent_names),
            "exercised": bool(action_rows),
            "agent_mcp_servers": sorted(agent_names),
            "injection_mcp_servers": sorted(injection_names),
            "agent_actions": action_rows[:20],
            "agent_action_count": len(action_rows),
        }


    def external_records_drive_host_dir(self) -> Optional[Path]:
        handle = self.handle
        if handle is None or "external_files" not in set(handle.environments):
            return None
        if handle.evidence_dir is None:
            return None
        root = handle.evidence_dir / "service_state" / "external_files"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _seeded_external_relative_paths(self) -> set[str]:
        return set(self._seeded_external_paths)

    def _external_records_snapshot(self) -> Dict[str, str]:
        root = self.external_records_drive_host_dir()
        if root is None or not root.exists():
            return {}
        snapshot: Dict[str, str] = {}
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                snapshot[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            except Exception:
                continue
        return snapshot

    def capture_external_records_drive_baseline(self) -> Dict[str, str]:
        """Remember initialized records so later in-place edits remain observable."""
        self._external_records_baseline = self._external_records_snapshot()
        self._external_records_baseline_captured = True
        return dict(self._external_records_baseline)

    def _write_external_records_drive_outcomes(self, handle: ServiceSandboxHandle) -> None:
        root = self.external_records_drive_host_dir()
        if root is None or handle.evidence_dir is None or not root.exists():
            return
        state_dir = handle.evidence_dir / "service_state"
        ledger_path = state_dir / "outcome_ledger.jsonl"
        if ledger_path.exists() and "records_drive_mount_snapshot" in ledger_path.read_text(encoding="utf-8", errors="replace"):
            return
        seeded = self._seeded_external_relative_paths()
        rows: List[Dict[str, Any]] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            try:
                data = path.read_bytes()
            except Exception:
                continue
            digest = hashlib.sha256(data).hexdigest()
            if self._external_records_baseline_captured:
                if self._external_records_baseline.get(rel) == digest:
                    continue
            elif rel in seeded:
                # Compatibility for callers that collect without an explicit
                # post-initialization baseline.
                continue
            text = data[:16000].decode("utf-8", errors="replace")
            rows.append({
                "schema_version": "red-service-outcome-ledger-v1",
                "outcome_id": "out-mount-" + hashlib.sha256((rel + digest).encode()).hexdigest()[:12],
                "action_id": "mount-snapshot-" + hashlib.sha256(rel.encode()).hexdigest()[:12],
                "actor": "agent",
                "service": "external_files",
                "type": "file_write",
                "action": "write_file",
                "source": "records_drive_mount_snapshot",
                "path": "/srv/external-files/" + rel,
                "mount_path": "/mnt/records-drive/" + rel,
                "bytes": len(data),
                "fingerprints": {"sha256": digest},
                "content": text,
                "change_type": "modified" if rel in self._external_records_baseline else "created",
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
        if rows:
            state_dir.mkdir(parents=True, exist_ok=True)
            with ledger_path.open("a", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    async def collect_evidence(self) -> Dict[str, Any]:
        handle = self.handle
        if handle is None:
            raise ServiceSandboxError("sandbox has not been started")
        self._write_external_records_drive_outcomes(handle)
        evidence = {
            "schema_version": "red-agent-world-state-first-evidence-v1",
            "sandbox_id": handle.sandbox_id,
            "service_state": {},
            "action_ledger": {},
            "outcome_ledger": {},
            "manifest": str(handle.evidence_dir / "service_sandbox_manifest.json") if handle.evidence_dir else "",
        }
        if handle.evidence_dir:
            for path in sorted(handle.evidence_dir.rglob("*")):
                if not path.is_file():
                    continue
                if path.name == "service_sandbox_manifest.json":
                    continue
                try:
                    rel = str(path.relative_to(handle.evidence_dir))
                    content = path.read_text(encoding="utf-8", errors="replace")
                    if path.name == "outcome_ledger.jsonl":
                        evidence["outcome_ledger"][rel] = content
                    elif path.name == "action_ledger.jsonl":
                        evidence["action_ledger"][rel] = content
                    elif path.name == "mcp_action_logs.jsonl":
                        continue
                    else:
                        evidence["service_state"][rel] = content
                except Exception as exc:
                    evidence["service_state"][str(path)] = f"<read_error:{exc}>"
            (handle.evidence_dir / "state_first_evidence.json").write_text(
                json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return evidence

    async def cleanup(self) -> None:
        handle = self.handle
        for proc in self._processes:
            if proc.poll() is None:
                proc.terminate()
        await asyncio.sleep(0.5)
        for proc in self._processes:
            if proc.poll() is None:
                proc.kill()
        if handle is not None:
            await self._stop_compose_environments(handle)

    async def _stop_compose_environments(self, handle: ServiceSandboxHandle) -> None:
        env_defs = self.env_config.get("environments", {})
        profile_by_env = _profile_by_environment(self.profile_config)
        for env_name in handle.environments:
            profile = profile_by_env.get(env_name, {})
            env_def = env_defs.get(env_name, {})
            compose_rel = profile.get("compose") or env_def.get("docker_compose")
            if not compose_rel:
                continue
            compose_file = (REPO_ROOT / compose_rel).resolve() if profile.get("compose") else (self.service_world_root / compose_rel).resolve()
            env = os.environ.copy()
            env.update(handle.env_exports())
            cmd = ["docker", "compose", "-p", handle.project_names[env_name], "-f", str(compose_file), "down", "-v", "--remove-orphans"]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(compose_file.parent),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
