"""Small evaluator-side helpers shared by active agent runtimes."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
_artifact_root = os.environ.get("RED_AGENT_WORLD_ARTIFACT_ROOT")
ARTIFACT_ROOT = Path(_artifact_root).expanduser().resolve() if _artifact_root else REPO_ROOT / "results"
SANDBOX_RESULT_FIELDS = [
    "id", "sandbox_status", "sandbox_template", "sandbox_error_type",
    "judge_status", "attack_succeeded", "score", "reason",
    "target", "category", "jailbreak_method", "error",
]
ROLLOUT_ERROR_PATTERNS = {
    "case_schema_error": ["legacy case fields", "unsupported sandbox", "sandbox.seed", "sandbox.attack"],
    "agent_model_incomplete": ["incomplete_result", "incomplete terminal response"],
    "agent_iteration_limit": ["iteration limit", "maximum iterations", "max turns"],
    "agent_tool_surface": ["toolset registration failed", "unknown toolsets"],
    "agent_noninteractive_prompt": ["interactive clarify", "clarify timed out"],
    "agent_gateway": ["GatewayClientRequestError", "FailoverError"],
    "openclaw_prompt_500": ["POST /prompt failed", "POST /prompt 失败", "500:"],
    "openclaw_timeout": ["timeout", "timed out", "超时"],
}
logger = logging.getLogger(__name__)


def resolve_arg_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else (Path.cwd() / candidate).resolve()


def classify_rollout_error(message: str) -> str:
    lowered = (message or "").lower()
    for error_type, needles in ROLLOUT_ERROR_PATTERNS.items():
        if any(needle.lower() in lowered for needle in needles):
            return error_type
    return "rollout_error"


def compact_json(value: Any, limit: int = 12000) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(text) <= limit:
        return text
    payload = {"truncated": True, "original_chars": len(text), "preview": text[: max(0, limit - 256)]}
    compacted = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    while len(compacted) > limit and payload["preview"]:
        payload["preview"] = payload["preview"][:-max(1, len(compacted) - limit + 16)]
        compacted = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return compacted


def _safe_part(value: Any) -> str:
    text = "".join(character if character.isalnum() or character in "_.-" else "_" for character in str(value))
    return text.strip("._-") or "unknown"


def attach_sandbox_bootstrap_sidecar(
    result: Dict[str, Any], export_dir: Path, item_id: Any, bootstrap_report: Dict[str, Any]
) -> Dict[str, Any]:
    export_root = Path(export_dir)
    root = export_root / "result_metadata" / ("item-%s" % _safe_part(item_id))
    root.mkdir(parents=True, exist_ok=True)
    path = root / "sandbox_bootstrap.json"
    data = (json.dumps(bootstrap_report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    temporary = path.with_suffix(".json.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    try:
        relative = path.relative_to(export_root).as_posix()
    except ValueError:
        relative = str(path)
    metadata = {"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    artifacts = result.setdefault("artifacts", {})
    artifacts["sandbox_bootstrap"] = {"storage": "sidecar_json", **metadata}
    return metadata


def sandbox_bootstrap_reference(result: Dict[str, Any]) -> Dict[str, Any]:
    artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), dict) else {}
    reference = artifacts.get("sandbox_bootstrap") if isinstance(artifacts.get("sandbox_bootstrap"), dict) else {}
    return dict(reference)


def constraint_for(item: Dict[str, Any]) -> str:
    taxonomy = item.get("taxonomy") if isinstance(item.get("taxonomy"), dict) else {}
    return str(taxonomy.get("constraint") or item.get("category") or "")


async def verify_canonical_workspace(client: Any) -> Dict[str, Any]:
    prepare = (
        "mkdir -p /workspace /root/.openclaw; "
        "rm -rf /root/.openclaw/workspace /root/project; "
        "ln -sfn /workspace /root/.openclaw/workspace; "
        "ln -sfn /workspace /root/project"
    )
    response = await client.execute_command(prepare)
    result = response.get("result", {})
    if result.get("exit_code") not in (0, None):
        raise RuntimeError("failed to prepare canonical workspace")
    check = await client.execute_command(
        "test \"$(readlink -f /root/.openclaw/workspace)\" = /workspace "
        "&& test \"$(readlink -f /root/project)\" = /workspace"
    )
    check_result = check.get("result", {})
    ok = check_result.get("exit_code") in (0, None)
    payload = {
        "ok": ok,
        "checks": {
            "/workspace": "/workspace",
            "/root/.openclaw/workspace": "/workspace",
            "/root/project": "/workspace",
        },
    }
    if not ok:
        raise RuntimeError("sandbox workspace contract failed: %s" % compact_json(payload))
    return payload


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def start_collector() -> Optional[subprocess.Popen]:
    if _port_open("127.0.0.1", 18080):
        return None
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = ARTIFACT_ROOT / "local_collector_runtime.log"
    log_file = open(log_path, "a", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("RED_AGENT_WORLD_COLLECTOR_LOG", str(ARTIFACT_ROOT / "local_collector_requests.jsonl"))
    # Starting the collector is an explicit CLI opt-in. Bind it for the
    # isolated runtime container only in that mode.
    env.setdefault("RED_AGENT_WORLD_COLLECTOR_HOST", "0.0.0.0")
    process = subprocess.Popen(
        [sys.executable, "-m", "red_agent_world.common.local_collector"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        env=env,
    )
    time.sleep(1.0)
    if not _port_open("127.0.0.1", 18080):
        process.terminate()
