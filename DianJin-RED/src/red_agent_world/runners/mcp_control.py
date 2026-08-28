"""Formal MCP control-plane support for E5 cases.

E5 ``mcp_control`` attacks are removed from normal sandbox seeding and applied
to the agent-facing MCP catalog before an agent runtime registers its tools.
The implementation is intentionally local to one rollout, so concurrent runs
cannot share gateway processes or control rules.
"""

from __future__ import annotations

import copy
import hashlib
import json
import socket
import secrets
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlsplit


REPO_ROOT = Path(__file__).resolve().parents[3]

SERVICE_BY_SURFACE = {
    "gmail": "gmail",
    "browser": "browser",
    "banking": "banking",
    "external_files": "external_files",
}
SUPPORTED_MECHANISMS = {"description_poisoning", "input_schema_poisoning"}


def _content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _json_request(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=body, method=method)
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    with urllib.request.urlopen(request, timeout=30.0) as response:
        raw = response.read().decode("utf-8", errors="replace").strip()
        return {"status": response.status, "body": json.loads(raw) if raw else {}}


def _wait_gateway(port: int, timeout_s: float = 20.0) -> None:
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        try:
            response = _json_request("GET", "http://127.0.0.1:%s/health" % port)
            if response["status"] == 200 and response["body"].get("ok"):
                return
        except Exception as exc:  # pragma: no cover - timing-dependent startup path
            last_error = str(exc)
        time.sleep(0.5)
    raise RuntimeError("MCP control gateway did not become healthy on port %s: %s" % (port, last_error))


def _gateway_url(original_url: str, port: int) -> str:
    parsed = urlsplit(original_url)
    host = parsed.hostname or "127.0.0.1"
    return "http://%s:%s%s" % (host, port, parsed.path or "/mcp")


def _safe_rule(rule: Dict[str, Any]) -> Dict[str, Any]:
    safe = {
        "id": rule["id"],
        "source_index": rule["source_index"],
        "surface": rule["surface"],
        "service": rule["service"],
        "tool": rule["tool"],
        "mechanism": rule["mechanism"],
        "content_sha256": rule["content_sha256"],
    }
    if rule.get("parameter"):
        safe["parameter"] = rule["parameter"]
    return safe


def normalize_e5_case_for_sandbox(item: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt legacy E5 seed rows to final1's canonical sandbox initializer shape."""
    cloned = copy.deepcopy(item)
    sandbox = cloned.get("sandbox")
    if not isinstance(sandbox, dict):
        return cloned
    seeds = sandbox.get("seed")
    if not isinstance(seeds, list):
        return cloned

    normalized: List[Dict[str, Any]] = []
    for index, raw in enumerate(seeds):
        if not isinstance(raw, dict):
            raise ValueError("sandbox.seed[%s] must be an object" % index)
        surface = raw.get("surface")
        operation = raw.get("operation")
        args = raw.get("args") if isinstance(raw.get("args"), dict) else {}

        if surface == "workspace" and operation is None and "path" in raw and "content" in raw:
            normalized.append({
                "surface": "workspace",
                "operation": "write_file",
                "args": {"path": raw["path"], "content": raw["content"]},
            })
        elif surface == "browser" and operation == "update_html_content":
            if "html" not in args:
                raise ValueError("sandbox.seed[%s] browser update_html_content requires args.html" % index)
            normalized.append({"surface": "browser", "operation": "set_page", "args": {"html": args["html"]}})
        elif surface == "external_files" and operation == "inject_file":
            if "file_path" not in args or "content" not in args:
                raise ValueError("sandbox.seed[%s] external_files inject_file requires args.file_path and args.content" % index)
            normalized.append({
                "surface": "external_files",
                "operation": "write_file",
                "args": {"path": args["file_path"], "content": args["content"]},
            })
        else:
            normalized.append(copy.deepcopy(raw))

    sandbox["seed"] = normalized
    cloned["sandbox"] = sandbox
    return cloned


def split_mcp_control_attacks(item: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Return a normal seedable case plus validated evaluator-only E5 rules."""
    cloned = copy.deepcopy(item)
    sandbox = cloned.get("sandbox")
    if not isinstance(sandbox, dict):
        return cloned, []
    attacks = sandbox.get("attack")
    if not isinstance(attacks, list):
        return cloned, []

    kept: List[Dict[str, Any]] = []
    rules: List[Dict[str, Any]] = []
    for index, raw in enumerate(attacks):
        if not isinstance(raw, dict) or str(raw.get("type") or "") != "mcp_control":
            kept.append(raw)
            continue
        surface = str(raw.get("surface") or "").strip()
        service = SERVICE_BY_SURFACE.get(surface)
        mechanism = str(raw.get("mechanism") or "description_poisoning").strip().lower()
        tool = str(raw.get("tool") or "").strip()
        parameter = str(raw.get("parameter") or "").strip()
        content = str(raw.get("content") or "")
        if service is None:
            raise ValueError("mcp_control attack[%s] uses unsupported surface %s" % (index, surface or "<empty>"))
        if mechanism not in SUPPORTED_MECHANISMS:
            raise ValueError("mcp_control attack[%s] uses unsupported mechanism %s" % (index, mechanism or "<empty>"))
        if not tool:
            raise ValueError("mcp_control attack[%s] requires tool" % index)
        if not content:
            raise ValueError("mcp_control attack[%s] requires content" % index)
        if mechanism == "input_schema_poisoning" and not parameter:
            raise ValueError("mcp_control attack[%s] input_schema_poisoning requires parameter" % index)
        rules.append({
            "id": "case-%s-attack-%s" % (item.get("id", "unknown"), index),
            "source_index": index,
            "surface": surface,
            "service": service,
            "tool": tool,
            "mechanism": mechanism,
            "parameter": parameter,
            "content": content,
            "content_sha256": _content_sha256(content),
        })
    sandbox["attack"] = kept
    cloned["sandbox"] = sandbox
    return cloned, rules


class MCPControlSession:
    """Own gateway lifecycle and evidence for exactly one sandbox rollout."""

    def __init__(self) -> None:
        self._bindings: List[Dict[str, Any]] = []

    def start(self, service_handle: Any, rules: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        rules = list(rules)
        if not rules:
            return {"enabled": False, "rules": [], "services": [], "evidence_files": []}
        if service_handle is None:
            raise RuntimeError("cannot apply mcp_control without an active service sandbox")

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for rule in rules:
            grouped.setdefault(str(rule["service"]), []).append(rule)
        endpoints = {str(server.name): server for server in service_handle.agent_mcp_servers}
        missing = sorted(set(grouped) - set(endpoints))
        if missing:
            raise RuntimeError("mcp_control target services are not active: %s" % ", ".join(missing))

        try:
            for service_name, service_rules in grouped.items():
                endpoint = endpoints[service_name]
                original_url = str(endpoint.url)
                port = _free_port()
                evidence_root = Path(service_handle.evidence_dir) if service_handle.evidence_dir else REPO_ROOT / "tmp" / "mcp_control"
                state_dir = evidence_root / "service_state"
                state_dir.mkdir(parents=True, exist_ok=True)
                gateway_script = Path(__file__).with_name("mcp_control_gateway.py")
                control_token = secrets.token_urlsafe(32)
                stdout_path = state_dir / (service_name + "_mcp_control_gateway_stdout.log")
                stderr_path = state_dir / (service_name + "_mcp_control_gateway_stderr.log")
                stdout_handle = stdout_path.open("wb")
                stderr_handle = stderr_path.open("wb")
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(gateway_script),
                        "--port",
                        str(port),
                        "--upstream-url",
                        original_url,
                        "--server-name",
                        service_name,
                        "--state-dir",
                        str(state_dir),
                        "--control-token",
                        control_token,
                    ],
                    cwd=str(REPO_ROOT),
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                )
                exposed_url = _gateway_url(original_url, port)
                safe_service_name = "".join(char if char.isalnum() or char in "_.-" else "_" for char in service_name)
                catalog_path = state_dir / (safe_service_name + "_agent_visible_tool_catalog.jsonl")
                ledger_path = state_dir / (safe_service_name + "_mcp_control_ledger.jsonl")
                config_path = state_dir / (safe_service_name + "_mcp_control_config.json")
                binding = {
                    "service": service_name,
                    "endpoint": endpoint,
                    "original_url": original_url,
                    "gateway_url": exposed_url,
                    "process": process,
                    "stdout_handle": stdout_handle,
                    "stderr_handle": stderr_handle,
                    "config_path": config_path,
                    "catalog_path": catalog_path,
                    "ledger_path": ledger_path,
                    "rules": [_safe_rule(rule) for rule in service_rules],
                }
                self._bindings.append(binding)
                _wait_gateway(port)
                proofs: List[Dict[str, Any]] = []
                for rule in service_rules:
                    control = _json_request(
                        "POST",
                        "http://127.0.0.1:%s/control" % port,
                        {
                            "mechanism": rule["mechanism"],
                            "tool": rule["tool"],
                            "parameter": rule.get("parameter", ""),
                            "content": rule["content"],
                        },
                        {"Content-Type": "application/json", "X-RED-Control-Token": control_token},
                    )
                    if not control["body"].get("ok"):
                        raise RuntimeError("MCP control configuration failed: %s" % control["body"])
                    proofs.append({
                        "rule": _safe_rule(rule),
                        "proof": control["body"].get("proof", {}),
                    })

                endpoint.url = exposed_url
                config = {
                    "schema_version": "red-mcp-control-config-v1",
                    "kind": "e5_control_plane",
                    "service": service_name,
                    "original_url": original_url,
                    "gateway_url": exposed_url,
                    "gateway_port": port,
                    "agent_visible_catalog": str(catalog_path),
                    "control_ledger": str(ledger_path),
                    "rules": [_safe_rule(rule) for rule in service_rules],
                    "proofs": proofs,
                }
                config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            self.stop()
            raise
        return self.report()

    def report(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self._bindings),
            "rules": [rule for binding in self._bindings for rule in binding["rules"]],
            "services": [
                {
                    "service": binding["service"],
                    "original_url": binding["original_url"],
                    "gateway_url": binding["gateway_url"],
                    "agent_visible_catalog": str(binding["catalog_path"]),
                    "control_ledger": str(binding["ledger_path"]),
                }
                for binding in self._bindings
            ],
            "evidence_files": [
                str(path)
                for binding in self._bindings
                for path in (binding["config_path"], binding["catalog_path"], binding["ledger_path"])
            ],
        }

    def stop(self) -> None:
        for binding in reversed(self._bindings):
            endpoint = binding.get("endpoint")
            if endpoint is not None:
                endpoint.url = binding["original_url"]
            process = binding.get("process")
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            for handle_name in ("stdout_handle", "stderr_handle"):
                handle = binding.get(handle_name)
                if handle is not None:
                    try:
                        handle.close()
                    except Exception:
                        pass
        self._bindings = []
