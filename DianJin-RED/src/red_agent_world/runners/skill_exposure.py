"""E3-only native runtime exposure for preplaced workspace skills."""

from __future__ import annotations

import shlex
from pathlib import PurePosixPath
from typing import Any, Dict, List, Mapping, Optional

from red_agent_world.sandbox.container_files import SANDBOX_WORKSPACE
from red_agent_world.sandbox.local_docker import LocalDockerSandbox


MALICIOUS_SKILL_PREPLACEMENT = "malicious_skill_preplacement"
WORKSPACE_SKILLS_ROOT = "%s/skills" % SANDBOX_WORKSPACE.rstrip("/")
CODEX_SKILLS_ROOT = "/root/.codex/skills"

RUNTIME_HERMES = "hermes-cli"
RUNTIME_CODEX = "codex-cli"
RUNTIME_OPENCLAW = "openclaw-service"

_REGISTRATION_METHODS = {
    RUNTIME_HERMES: "hermes.skills.external_dirs",
    RUNTIME_CODEX: "codex.home.skill_package_symlinks",
    RUNTIME_OPENCLAW: "openclaw.workspace.skills",
}


def intervention_for(item: Mapping[str, Any]) -> str:
    taxonomy = item.get("taxonomy")
    if not isinstance(taxonomy, Mapping):
        return ""
    return str(taxonomy.get("intervention") or "").strip()


def is_malicious_skill_preplacement(item: Mapping[str, Any]) -> bool:
    return intervention_for(item) == MALICIOUS_SKILL_PREPLACEMENT


def hermes_workspace_skills_yaml(item: Mapping[str, Any]) -> str:
    """Return the E3-only Hermes native external-skill configuration."""
    if not is_malicious_skill_preplacement(item):
        return ""
    return "\nskills:\n  external_dirs:\n    - %s\n" % WORKSPACE_SKILLS_ROOT


def _command_result(response: Mapping[str, Any]) -> Mapping[str, Any]:
    result = response.get("result", {})
    return result if isinstance(result, Mapping) else {}


def _require_command_success(response: Mapping[str, Any], action: str) -> Mapping[str, Any]:
    result = _command_result(response)
    if result.get("exit_code") not in (0, None):
        detail = str(result.get("stderr") or result.get("stdout") or "")[-1000:]
        raise RuntimeError("E3 skill exposure failed while %s: %s" % (action, detail))
    return result


def _skill_packages(stdout: str) -> List[Dict[str, str]]:
    packages: List[Dict[str, str]] = []
    seen = set()
    root = PurePosixPath(WORKSPACE_SKILLS_ROOT)
    for raw in str(stdout or "").splitlines():
        value = raw.strip()
        if not value:
            continue
        manifest = PurePosixPath(value)
        try:
            relative = manifest.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("E3 skill manifest escaped canonical root: %s" % value) from exc
        if len(relative.parts) != 2 or relative.parts[1] != "SKILL.md":
            raise RuntimeError("E3 skill manifest must be skills/<name>/SKILL.md: %s" % value)
        name = relative.parts[0]
        if not name or name in {".", ".."} or name in seen:
            raise RuntimeError("E3 skill package name is invalid or duplicated: %s" % name)
        seen.add(name)
        packages.append({
            "name": name,
            "source": str(root / name),
            "manifest": str(manifest),
        })
    if not packages:
        raise RuntimeError("E3 skill exposure found no canonical /workspace/skills/<name>/SKILL.md package")
    return sorted(packages, key=lambda package: package["name"])


async def prepare_preplaced_skill_exposure(
    client: LocalDockerSandbox,
    item: Mapping[str, Any],
    runtime_name: str,
) -> Optional[Dict[str, Any]]:
    """Expose canonical E3 skill packages through one runtime's native loader.

    Non-E3 cases return before issuing any container command. This keeps their
    runtime configuration, workspace, and bootstrap report byte-for-byte on the
    existing path.
    """
    if not is_malicious_skill_preplacement(item):
        return None
    if runtime_name not in _REGISTRATION_METHODS:
        raise RuntimeError("unsupported runtime for E3 skill exposure: %s" % runtime_name)

    find_command = (
        "find %s -mindepth 2 -maxdepth 2 -type f -name SKILL.md -print"
        % shlex.quote(WORKSPACE_SKILLS_ROOT)
    )
    discovery = _require_command_success(
        await client.execute_command(find_command),
        "discovering canonical skill packages",
    )
    packages = _skill_packages(str(discovery.get("stdout") or ""))

    native_paths: List[str] = []
    if runtime_name == RUNTIME_CODEX:
        _require_command_success(
            await client.execute_command("mkdir -p %s" % shlex.quote(CODEX_SKILLS_ROOT)),
            "creating the Codex skill root",
        )
        for package in packages:
            target = "%s/%s" % (CODEX_SKILLS_ROOT, package["name"])
            link_command = "ln -s %s %s" % (
                shlex.quote(package["source"]),
                shlex.quote(target),
            )
            _require_command_success(
                await client.execute_command(link_command),
                "registering Codex skill %s" % package["name"],
            )
            native_paths.append(target)
    elif runtime_name == RUNTIME_HERMES:
        native_paths.append(WORKSPACE_SKILLS_ROOT)
    else:
        native_paths.extend(package["source"] for package in packages)

    return {
        "status": "registered",
        "intervention": MALICIOUS_SKILL_PREPLACEMENT,
        "runtime": runtime_name,
        "canonical_root": WORKSPACE_SKILLS_ROOT,
        "registration_method": _REGISTRATION_METHODS[runtime_name],
        "skill_names": [package["name"] for package in packages],
        "manifests": [package["manifest"] for package in packages],
        "native_paths": native_paths,
    }
