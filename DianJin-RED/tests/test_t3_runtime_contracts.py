from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import pytest

from red_agent_world.common.case_schema import expected_outcome_description
from red_agent_world.sandbox.service_sandbox import ServiceSandboxSession


ROOT = Path(__file__).resolve().parents[1]
T3 = ROOT / "test" / "T3.json"
SERVICE_WORLD_SERVERS = (
    ROOT / "sandbox/service_world/mcp_servers/red_service_http.py",
    ROOT / "sandbox/service_world/injection_mcp_servers/red_service_http.py",
)


def test_browser_injection_uses_its_own_listener_port(tmp_path: Path) -> None:
    session = ServiceSandboxSession(
        {
            "services": ["browser"],
            "mcp_servers": ["browser"],
            "injection_mcp_servers": ["browser-injection"],
            "start_services": True,
        },
        "browser-port-contract",
        service_world_root=ROOT / "sandbox/service_world",
        evidence_root=tmp_path,
        dry_run=True,
    )
    endpoint = session._mcp_endpoint(
        "browser-injection",
        session.injection_config,
        {
            "BROWSER_MCP_PORT": 31001,
            "CUSTOM_WEBSITE_INJECTION_MCP_PORT": 31002,
        },
        role="injection",
    )

    assert endpoint.url == "http://127.0.0.1:31002/mcp"


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"test_{path.parent.name}_{path.stem}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("server_path", SERVICE_WORLD_SERVERS)
def test_override_never_returns_original_tool_result(server_path: Path, tmp_path: Path) -> None:
    module = _load_module(server_path)
    if server_path.parent.name == "mcp_servers":
        manager = module.InjectionRules(tmp_path, "gmail")
    else:
        manager = module.InjectionRules()
    manager.add({"tool": "list_messages", "content": '{"forged": true}', "mode": "override"})

    visible = manager.apply("list_messages", {"secret_original": "must-not-be-visible"})

    assert visible == {"forged": True}
    assert "secret_original" not in json.dumps(visible)
    assert "original_result" not in json.dumps(visible)

    manager.clear()
    manager.add({"tool": "list_messages", "content": "forged plain text", "mode": "override"})
    assert manager.apply("list_messages", {"secret_original": "must-not-be-visible"}) == "forged plain text"


@pytest.mark.parametrize("server_path", SERVICE_WORLD_SERVERS)
def test_email_boundary_uses_configurable_neutral_domains(
    server_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module(server_path)
    monkeypatch.setenv("INTERNAL_EMAIL_DOMAINS", "internal.example,corp.example")

    assert module._is_external_email("agent@internal.example") is False
    assert module._is_external_email("ops@sub.corp.example") is False
    assert module._is_external_email("collector@external.example") is True


def test_t3_has_no_standalone_safety_policy_seed() -> None:
    cases = json.loads(T3.read_text(encoding="utf-8"))
    assert len(cases) == 100
    exposed = []
    for case in cases:
        for row in case["sandbox"]["seed"]:
            if row["surface"] != "workspace":
                continue
            path = row["args"]["path"]
            basename = path.rsplit("/", 1)[-1].lower()
            if basename.endswith("policy.md") or basename == "mailbox_retention.md":
                exposed.append((case["id"], path))
    assert exposed == []


def test_t3_has_no_case_id_sentinel_markers() -> None:
    cases = json.loads(T3.read_text(encoding="utf-8"))
    marker = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+_984\d{3}\b")
    exposed = []
    for case in cases:
        agent_material = json.dumps(
            {
                "decomposed_query": case["decomposed_query"],
                "seed": case["sandbox"]["seed"],
                "attack": case["sandbox"]["attack"],
            },
            ensure_ascii=False,
        )
        hits = sorted(set(marker.findall(agent_material)))
        if hits:
            exposed.append((case["id"], hits))
    assert exposed == []


def test_all_browser_cases_explicitly_name_browser() -> None:
    cases = json.loads(T3.read_text(encoding="utf-8"))
    for case in cases:
        attack_surfaces = {row["surface"] for row in case["sandbox"]["attack"]}
        if "browser" in attack_surfaces:
            assert "browser" in " ".join(case["decomposed_query"]).lower(), case["id"]


def test_named_external_payee_uses_real_blnk_sink() -> None:
    source = (ROOT / "sandbox/services/banking/blnk/tools.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"_backend_destination", "_restore_external_destination"}
    ]
    namespace: dict[str, Any] = {"Any": Any}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(T3), "exec"), namespace)

    assert namespace["_backend_destination"]("external_supplier_984000") == "@world"
    assert namespace["_backend_destination"]("balance_internal") == "balance_internal"
    restored = namespace["_restore_external_destination"](
        {"transaction_id": "txn-real", "destination": "@world", "status": "applied"},
        "external_supplier_984000",
    )
    assert restored == {
        "transaction_id": "txn-real",
        "destination": "external_supplier_984000",
        "status": "applied",
        "backend_destination": "@world",
        "backend_sink": "@world",
        "external_sink": True,
    }


def test_external_transfer_cases_preserve_real_balance_discovery() -> None:
    cases = json.loads(T3.read_text(encoding="utf-8"))
    exposed = []
    for case in cases:
        expected = expected_outcome_description(case["expected_outcome"]).lower()
        if "banking transfer" not in expected or "external_" not in expected:
            continue
        for row in case["sandbox"]["attack"]:
            args = row["args"]
            if (
                row["surface"] == "banking"
                and row["operation"] == "inject_tool_output"
                and args.get("tool") == "list_balances"
                and args.get("mode") == "override"
            ):
                exposed.append(case["id"])
    assert exposed == []


def test_hermes_does_not_wrap_mcp_results_in_safety_instructions() -> None:
    source = (ROOT / "src/red_agent_world/runners/hermes_sandbox_runner.py").read_text(
        encoding="utf-8"
    )
    assert 'tool_dispatch_helpers._UNTRUSTED_TOOL_NAMES = frozenset()' in source
    assert 'tool_dispatch_helpers._UNTRUSTED_TOOL_PREFIXES = ()' in source
