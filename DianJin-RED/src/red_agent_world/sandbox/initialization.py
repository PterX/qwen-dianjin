"""Canonical case parsing and one-pass sandbox initialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional


WORKSPACE_SURFACE = "workspace"
SERVICE_BY_SURFACE = {
    "gmail": "gmail",
    "browser": "browser",
    "banking": "banking",
    "external_files": "external_files",
}
INJECTION_BY_SERVICE = {service: f"{service}-injection" for service in SERVICE_BY_SURFACE.values()}
REMOVED_SURFACES = {"memory", "skill"}
LEGACY_CASE_FIELDS = {"Setup", "Adversary", "Sandbox", "sandbox_spec", "service_sandbox", "attack_turns"}


def canonical_surface(value: Any) -> str:
    return str(value or "").strip()


@dataclass(frozen=True)
class InitializationOperation:
    phase: str
    index: int
    surface: str
    payload: Dict[str, Any]


@dataclass(frozen=True)
class InitializationPlan:
    surfaces: List[str]
    operations: List[InitializationOperation]
    runtime_spec: Dict[str, Any]

    @property
    def workspace_operations(self) -> List[InitializationOperation]:
        return [operation for operation in self.operations if operation.surface == WORKSPACE_SURFACE]

    @property
    def service_operations(self) -> List[InitializationOperation]:
        return [operation for operation in self.operations if operation.surface in SERVICE_BY_SURFACE]


@dataclass(frozen=True)
class CaseSandboxSpec:
    surfaces: List[str]
    seed: List[Dict[str, Any]]
    attack: List[Dict[str, Any]]

    @classmethod
    def parse(cls, item: Mapping[str, Any]) -> "CaseSandboxSpec":
        legacy = sorted(field for field in LEGACY_CASE_FIELDS if item.get(field) not in (None, {}, []))
        if legacy:
            raise ValueError("legacy case fields are not supported: %s" % ", ".join(legacy))
        sandbox = item.get("sandbox")
        if not isinstance(sandbox, dict):
            raise ValueError("sandbox must be an object")
        extra = sorted(set(sandbox) - {"surfaces", "seed", "attack"})
        if extra:
            raise ValueError("unsupported sandbox fields: %s" % ", ".join(extra))
        raw_surfaces = sandbox.get("surfaces")
        if not isinstance(raw_surfaces, list):
            raise ValueError("sandbox.surfaces must be a list")
        surfaces: List[str] = []
        for index, raw in enumerate(raw_surfaces):
            surface = canonical_surface(raw)
            cls._validate_surface(surface, "sandbox.surfaces[%s]" % index)
            if surface not in surfaces:
                surfaces.append(surface)
        phases: Dict[str, List[Dict[str, Any]]] = {}
        for phase in ("seed", "attack"):
            rows = sandbox.get(phase, [])
            if not isinstance(rows, list):
                raise ValueError("sandbox.%s must be a list" % phase)
            parsed: List[Dict[str, Any]] = []
            for index, raw in enumerate(rows):
                if not isinstance(raw, dict):
                    raise ValueError("sandbox.%s[%s] must be an object" % (phase, index))
                row = dict(raw)
                surface = canonical_surface(row.get("surface"))
                cls._validate_surface(surface, "sandbox.%s[%s].surface" % (phase, index))
                if surface not in surfaces:
                    raise ValueError("sandbox.%s[%s] uses undeclared surface %s" % (phase, index, surface))
                extra = sorted(set(row) - {"surface", "operation", "args"})
                if extra:
                    raise ValueError(
                        "unsupported sandbox.%s[%s] fields: %s; use surface, operation, and args"
                        % (phase, index, ", ".join(extra))
                    )
                operation = str(row.get("operation") or "").strip().lower()
                if not operation:
                    raise ValueError("sandbox.%s[%s].operation must be a non-empty string" % (phase, index))
                args = row.get("args")
                if not isinstance(args, dict):
                    raise ValueError("sandbox.%s[%s].args must be an object" % (phase, index))
                cls._validate_operation(surface, operation, args, "sandbox.%s[%s]" % (phase, index))
                parsed.append({"surface": surface, "operation": operation, "args": dict(args)})
            phases[phase] = parsed
        return cls(surfaces=surfaces, seed=phases["seed"], attack=phases["attack"])

    @staticmethod
    def _validate_surface(surface: str, label: str) -> None:
        if surface in REMOVED_SURFACES:
            raise ValueError("%s uses removed surface %s; use workspace memory/... or skills/..." % (label, surface))
        if surface != WORKSPACE_SURFACE and surface not in SERVICE_BY_SURFACE:
            raise ValueError("%s is unsupported: %s" % (label, surface or "<empty>"))

    @staticmethod
    def _validate_operation(surface: str, operation: str, args: Mapping[str, Any], label: str) -> None:
        allowed = {
            "workspace": {"write_file": {"path", "content"}},
            "gmail": {
                "inject_email": {"from_email", "to_email", "subject", "body", "attachments"},
                "inject_tool_output": {"tool", "content", "mode", "once"},
            },
            "browser": {
                "set_page": {"html"},
                "inject_tool_output": {"tool", "content", "mode", "once"},
            },
            "external_files": {
                "write_file": {"path", "content"},
                "append_file": {"path", "content"},
                "inject_tool_output": {"tool", "content", "mode", "once"},
            },
            "banking": {
                "seed_balance_pair": {"ledger_name", "amount"},
                "seed_pending_transfer": {
                    "source", "destination", "amount", "reason", "reference",
                    "queue_status", "approval_status", "approval_record",
                    "checkpoint_status", "checkpoint_record",
                    "authorization_status", "authorization_record",
                    "destination_status", "destination_record",
                    "override_status", "override_record", "queue_note",
                },
                "inject_tool_output": {"tool", "content", "mode", "once"},
            },
        }
        operation_schema = allowed.get(surface, {}).get(operation)
        if operation_schema is None:
            raise ValueError("%s uses unsupported %s operation %s" % (label, surface, operation))
        extra = sorted(set(args) - operation_schema)
        if extra:
            raise ValueError("unsupported %s.args fields: %s" % (label, ", ".join(extra)))
        required_by_operation = {
            ("workspace", "write_file"): {"path", "content"},
            ("gmail", "inject_email"): {"from_email", "to_email", "subject", "body"},
            ("browser", "set_page"): {"html"},
            ("external_files", "write_file"): {"path", "content"},
            ("external_files", "append_file"): {"path", "content"},
            ("banking", "seed_balance_pair"): {"ledger_name", "amount"},
            ("banking", "seed_pending_transfer"): {"reference", "source", "destination", "amount"},
        }
        required = (
            {"tool", "content"}
            if operation == "inject_tool_output"
            else required_by_operation.get((surface, operation), set())
        )
        missing = sorted(key for key in required if key not in args or args[key] is None)
        if missing:
            raise ValueError("%s.args is missing required fields: %s" % (label, ", ".join(missing)))

    def initialization_plan(self) -> InitializationPlan:
        operations: List[InitializationOperation] = []
        for phase, rows in (("seed", self.seed), ("attack", self.attack)):
            for index, row in enumerate(rows):
                operations.append(InitializationOperation(phase, index, row["surface"], dict(row)))
        services = [SERVICE_BY_SURFACE[surface] for surface in self.surfaces if surface in SERVICE_BY_SURFACE]
        runtime_spec = {
            "services": services,
            "mcp_servers": list(services),
            "injection_mcp_servers": [INJECTION_BY_SERVICE[service] for service in services],
            "start_services": bool(services),
        }
        return InitializationPlan(list(self.surfaces), operations, runtime_spec)


@dataclass
class InitializationReceipt:
    entries: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return all(bool(entry.get("success")) for entry in self.entries)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": "initialized" if self.success else "failed",
            "operation_count": len(self.entries),
            "entries": self.entries,
        }


def service_seed_for_operation(operation: InitializationOperation) -> Dict[str, Any]:
    row = operation.payload
    surface = operation.surface
    action = str(row["operation"])
    args = dict(row["args"])
    if action == "inject_tool_output":
        tool = str(args["tool"])
        service = SERVICE_BY_SURFACE[surface]
        mode = str(args.get("mode") or "append").strip().lower()
        if mode not in {"append", "prefix", "override"}:
            raise ValueError("unsupported tool-output mode: %s" % mode)
        return {
            "type": "tool",
            "injected_tool": "%s:%s" % (service, tool),
            "content": str(args["content"]),
            "mode": mode,
            "once": bool(args.get("once", False)),
        }
    if surface == "gmail":
        target = "gmail-injection:inject_email"
    elif surface == "browser":
        target = "browser-injection:update_html_content"
    elif surface == "external_files":
        target = "external_files-injection:%s" % ("inject_append" if action == "append_file" else "inject_file")
        args = {"file_path": args["path"], "content": args["content"]}
    elif surface == "banking":
        if action == "seed_balance_pair":
            target = "banking-injection:seed_balance_pair"
        elif action == "seed_pending_transfer":
            target = "banking-injection:seed_pending_transfer"
        else:
            raise ValueError("unsupported banking initialization operation: %s" % (action or "<empty>"))
    else:
        raise ValueError("not a service initialization operation: %s" % surface)
    if not args:
        raise ValueError("service initialization requires explicit args: %s" % target)
    return {"type": "environment", "injection_mcp_tool": target, "kwargs": args}


class SandboxInitializer:
    """Execute one canonical plan and return one auditable receipt."""

    def __init__(self, workspace_materializer: Any, service_session: Any) -> None:
        self.workspace_materializer = workspace_materializer
        self.service_session = service_session

    async def initialize(self, client: Any, plan: InitializationPlan, service_handle: Any = None) -> InitializationReceipt:
        receipt = InitializationReceipt()
        for phase in ("seed", "attack"):
            workspace = [operation for operation in plan.workspace_operations if operation.phase == phase]
            if workspace:
                receipt.entries.extend(await self.workspace_materializer.materialize(client, workspace, service_handle))
            service = [operation for operation in plan.service_operations if operation.phase == phase]
            if service:
                seeds = [service_seed_for_operation(operation) for operation in service]
                rows = await self.service_session.materialize_service_seeds(seeds)
                if len(rows) != len(service):
                    raise RuntimeError(
                        "service initializer returned %s receipts for %s operations"
                        % (len(rows), len(service))
                    )
                for operation, row in zip(service, rows):
                    receipt.entries.append({
                        "operation_id": "%s[%s]" % (operation.phase, operation.index),
                        "phase": operation.phase,
                        "index": operation.index,
                        "surface": operation.surface,
                        **row,
                    })
        if len(receipt.entries) != len(plan.operations):
            raise RuntimeError(
                "sandbox initializer returned %s receipts for %s operations"
                % (len(receipt.entries), len(plan.operations))
            )
        return receipt
