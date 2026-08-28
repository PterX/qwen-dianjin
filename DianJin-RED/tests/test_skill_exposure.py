import asyncio

import pytest

from red_agent_world.runners.skill_exposure import (
    CODEX_SKILLS_ROOT,
    RUNTIME_CODEX,
    RUNTIME_HERMES,
    RUNTIME_OPENCLAW,
    WORKSPACE_SKILLS_ROOT,
    hermes_workspace_skills_yaml,
    is_malicious_skill_preplacement,
    prepare_preplaced_skill_exposure,
)


E3_ITEM = {
    "taxonomy": {"intervention": "malicious_skill_preplacement"},
}


class FakeClient:
    def __init__(self, manifests=None, failing_prefix="", command_outputs=None) -> None:
        self.manifests = (
            ["/workspace/skills/citation-integrity/SKILL.md"]
            if manifests is None
            else manifests
        )
        self.failing_prefix = failing_prefix
        self.command_outputs = command_outputs or {}
        self.commands = []

    async def execute_command(self, command: str):
        self.commands.append(command)
        if self.failing_prefix and command.startswith(self.failing_prefix):
            return {"result": {"exit_code": 1, "stderr": "synthetic failure"}}
        stdout = ""
        if command.startswith("find "):
            stdout = "\n".join(self.manifests) + ("\n" if self.manifests else "")
        for prefix, value in self.command_outputs.items():
            if command.startswith(prefix):
                stdout = value
                break
        return {"result": {"exit_code": 0, "stdout": stdout, "stderr": ""}}


@pytest.mark.parametrize(
    "item, expected",
    [
        (E3_ITEM, True),
        ({"taxonomy": {"intervention": "malicious_skill_preplacement_v2"}}, False),
        ({"taxonomy": {"intervention": "memory_poisoning"}}, False),
        ({"taxonomy": {}}, False),
        ({}, False),
    ],
)
def test_e3_strategy_match_is_exact(item, expected) -> None:
    assert is_malicious_skill_preplacement(item) is expected


def test_non_e3_is_a_zero_command_noop_and_has_no_hermes_config_delta() -> None:
    client = FakeClient()
    item = {"taxonomy": {"intervention": "memory_poisoning"}}

    report = asyncio.run(prepare_preplaced_skill_exposure(client, item, RUNTIME_HERMES))

    assert report is None
    assert client.commands == []
    assert hermes_workspace_skills_yaml(item) == ""


def test_hermes_e3_uses_native_external_dirs_without_copying_packages() -> None:
    client = FakeClient()

    report = asyncio.run(prepare_preplaced_skill_exposure(client, E3_ITEM, RUNTIME_HERMES))

    assert report["registration_method"] == "hermes.skills.external_dirs"
    assert report["status"] == "registered"
    assert report["skill_names"] == ["citation-integrity"]
    assert report["native_paths"] == [WORKSPACE_SKILLS_ROOT]
    assert client.commands == [
        "find /workspace/skills -mindepth 2 -maxdepth 2 -type f -name SKILL.md -print"
    ]
    assert hermes_workspace_skills_yaml(E3_ITEM) == (
        "\nskills:\n  external_dirs:\n    - /workspace/skills\n"
    )


def test_codex_e3_maps_each_complete_package_into_native_skill_root() -> None:
    client = FakeClient(
        manifests=[
            "/workspace/skills/zeta/SKILL.md",
            "/workspace/skills/alpha/SKILL.md",
        ]
    )

    report = asyncio.run(prepare_preplaced_skill_exposure(client, E3_ITEM, RUNTIME_CODEX))

    assert report["skill_names"] == ["alpha", "zeta"]
    assert report["native_paths"] == [
        "%s/alpha" % CODEX_SKILLS_ROOT,
        "%s/zeta" % CODEX_SKILLS_ROOT,
    ]
    assert client.commands[1:] == [
        "mkdir -p /root/.codex/skills",
        "ln -s /workspace/skills/alpha /root/.codex/skills/alpha",
        "ln -s /workspace/skills/zeta /root/.codex/skills/zeta",
    ]


def test_openclaw_e3_keeps_canonical_workspace_packages_in_place() -> None:
    client = FakeClient()

    report = asyncio.run(prepare_preplaced_skill_exposure(client, E3_ITEM, RUNTIME_OPENCLAW))

    assert report["registration_method"] == "openclaw.workspace.skills"
    assert report["native_paths"] == ["/workspace/skills/citation-integrity"]
    assert len(client.commands) == 1


@pytest.mark.parametrize(
    "manifests",
    [
        [],
        ["/workspace/skills/group/name/SKILL.md"],
        ["/tmp/escaped/SKILL.md"],
        [
            "/workspace/skills/duplicate/SKILL.md",
            "/workspace/skills/duplicate/SKILL.md",
        ],
    ],
)
def test_malformed_e3_skill_packages_fail_before_rollout(manifests) -> None:
    client = FakeClient(manifests=manifests)

    with pytest.raises(RuntimeError, match="E3 skill"):
        asyncio.run(prepare_preplaced_skill_exposure(client, E3_ITEM, RUNTIME_HERMES))


def test_codex_native_name_collision_fails_closed() -> None:
    client = FakeClient(failing_prefix="ln -s ")

    with pytest.raises(RuntimeError, match="registering Codex skill"):
        asyncio.run(prepare_preplaced_skill_exposure(client, E3_ITEM, RUNTIME_CODEX))
