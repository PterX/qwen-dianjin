from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from red_agent_world.judges.action_grounded_judge import build_judge_payload, normalize_model_result
from red_agent_world.runners.agent_runtime_sandbox_runner import OpenClawCompatibleAgentRunner
from red_agent_world.runners.hermes_sandbox_runner import HermesSandboxRunner
from red_agent_world.runners.openclaw_service_sandbox_runner import OpenClawServiceSandboxRunner
from red_agent_world.runners.runtime_support import attach_sandbox_bootstrap_sidecar
from red_agent_world.sandbox.initialization import CaseSandboxSpec, service_seed_for_operation


ROOT = Path(__file__).resolve().parents[1]


class ContractCleanupTests(unittest.TestCase):
    def test_canonical_sandbox_row_is_strict(self) -> None:
        case = {
            "sandbox": {
                "surfaces": ["workspace", "browser", "external_files"],
                "seed": [
                    {
                        "surface": "workspace",
                        "operation": "write_file",
                        "args": {"path": "README.md", "content": ""},
                    },
                    {
                        "surface": "browser",
                        "operation": "set_page",
                        "args": {"html": "<h1>hello</h1>"},
                    },
                ],
                "attack": [
                    {
                        "surface": "external_files",
                        "operation": "inject_tool_output",
                        "args": {"tool": "read_file", "content": "injected", "mode": "append"},
                    }
                ],
            }
        }
        plan = CaseSandboxSpec.parse(case).initialization_plan()
        self.assertEqual(plan.runtime_spec["services"], ["browser", "external_files"])
        self.assertEqual(
            plan.runtime_spec["injection_mcp_servers"],
            ["browser-injection", "external_files-injection"],
        )
        seed = service_seed_for_operation(plan.service_operations[-1])
        self.assertEqual(seed["injected_tool"], "external_files:read_file")

    def test_sandbox_row_aliases_are_rejected(self) -> None:
        case = {
            "sandbox": {
                "surfaces": ["workspace"],
                "seed": [{"surface": "workspace", "path": "README.md", "content": "legacy"}],
                "attack": [],
            }
        }
        with self.assertRaisesRegex(ValueError, "use surface, operation, and args"):
            CaseSandboxSpec.parse(case)

    def test_judge_result_has_one_canonical_schema(self) -> None:
        normalized = normalize_model_result(
            {
                "verdict": "success",
                "evidence": ["outcome:1"],
            },
            ["outcome:1"],
        )
        self.assertEqual(
            set(normalized),
            {"status", "attack_succeeded", "score", "reason", "evidence_refs"},
        )
        payload = build_judge_payload(
            "hybrid",
            record_id=1,
            expected_outcome="sent",
            environment_state={},
            trajectory={},
        )
        self.assertNotIn("judge_type", payload)
        self.assertEqual(payload["evidence"], {})

    def test_result_bootstrap_is_stored_once(self) -> None:
        result = OpenClawCompatibleAgentRunner.__new__(OpenClawCompatibleAgentRunner)._base_result(
            {"id": 1, "decomposed_query": ["hello"], "taxonomy": {}}
        )
        with tempfile.TemporaryDirectory() as directory:
            attach_sandbox_bootstrap_sidecar(result, Path(directory), 1, {"status": "READY"})
        self.assertEqual(set(result["artifacts"]), {"sandbox_bootstrap"})
        for duplicate in (
            "sandbox_bootstrap",
            "sandbox_bootstrap_path",
            "sandbox_bootstrap_sha256",
            "sandbox_bootstrap_bytes",
            "harmful",
            "score",
            "evidence_type",
            "evidence_event_ids",
        ):
            self.assertNotIn(duplicate, result)

    def test_wire_names_are_runtime_specific(self) -> None:
        handle = SimpleNamespace(
            agent_mcp_servers=[SimpleNamespace(name="gmail", tools=["send_email"])]
        )
        openclaw = OpenClawServiceSandboxRunner.__new__(OpenClawServiceSandboxRunner)
        openclaw._active_service_sandbox = handle
        self.assertEqual(openclaw.service_tool_identity("gmail__send_email"), ("gmail", "send_email"))
        self.assertIsNone(openclaw.service_tool_identity("mcp__gmail__send_email"))
        hermes = HermesSandboxRunner.__new__(HermesSandboxRunner)
        hermes._active_service_sandbox = handle
        self.assertEqual(hermes.service_tool_identity("mcp__gmail__send_email"), ("gmail", "send_email"))
        self.assertIsNone(hermes.service_tool_identity("gmail.send_email"))

    def test_contract_snapshot_has_no_visible_tool_aliases(self) -> None:
        payload = json.loads((ROOT / "sandbox/contracts/service_mcp_tools_snapshot_v1.json").read_text())
        rendered = json.dumps(payload)
        for alias in ("fetch_page", "get_gmail_content", "send_reply", "get_attachment"):
            self.assertNotIn(f'"name": "{alias}"', rendered)


if __name__ == "__main__":
    unittest.main()
