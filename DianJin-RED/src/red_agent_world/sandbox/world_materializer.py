#!/usr/bin/env python3
"""Materialize agent-visible runtime-world surfaces for workspace cases."""

import hashlib
import json
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional

from red_agent_world.sandbox.container_files import SANDBOX_WORKSPACE, put_text_file
from red_agent_world.sandbox.initialization import InitializationOperation
from red_agent_world.sandbox.local_docker import LocalDockerSandbox
from red_agent_world.sandbox.service_sandbox import ServiceSandboxHandle


class WorldMaterializer:
    """Write only the normalized workspace operations supplied by a plan."""

    def __init__(self, workspace: str = SANDBOX_WORKSPACE) -> None:
        self.workspace = workspace.rstrip("/") or "/workspace"

    async def materialize(
        self,
        client: LocalDockerSandbox,
        operations: List[InitializationOperation],
        service_handle: Optional[ServiceSandboxHandle] = None,
    ) -> List[Dict[str, Any]]:
        await client.execute_command("mkdir -p %s" % self.workspace)
        receipts: List[Dict[str, Any]] = []
        seen_by_phase: Dict[str, set[str]] = {}
        for operation in operations:
            args = operation.payload["args"]
            rel = self._target_relative_path(args)
            phase_seen = seen_by_phase.setdefault(operation.phase, set())
            if rel in phase_seen:
                raise ValueError("duplicate workspace %s path: %s" % (operation.phase, rel))
            phase_seen.add(rel)
            target = f"{self.workspace}/{rel}"
            content = self._content_for(args)
            mode = 0o755 if target.endswith((".sh", ".py")) or "/scripts/" in target else 0o644
            await put_text_file(client, target, content, mode=mode)
            receipts.append({
                "success": True,
                "operation_id": "%s[%s]" % (operation.phase, operation.index),
                "phase": operation.phase,
                "index": operation.index,
                "surface": "workspace",
                "status": "written",
                "path": target,
                "relative_path": rel,
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            })

        return receipts

    def _target_relative_path(self, seed: Dict[str, Any]) -> str:
        path = seed.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("workspace seed requires a non-empty path")
        return self._safe_relative_path(path)

    def _content_for(self, seed: Dict[str, Any]) -> str:
        if seed.get("content") is None:
            raise ValueError("workspace seed requires explicit content")
        return str(seed["content"])

    def _safe_relative_path(self, value: str) -> str:
        value = value.strip().lstrip("/") or "README.md"
        path = PurePosixPath(value)
        parts = [part for part in path.parts if part not in ("", ".")]
        if any(part == ".." for part in parts):
            raise ValueError("seed path escapes workspace: %s" % value)
        return "/".join(parts) or "README.md"
