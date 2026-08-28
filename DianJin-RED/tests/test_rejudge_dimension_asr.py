from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from scripts.rejudge_dimension_asr import evaluate_case


class RejudgeDimensionAsrTests(unittest.IsolatedAsyncioTestCase):
    async def test_judge_error_rollout_can_be_rejudged(self) -> None:
        class FakeJudge:
            async def evaluate_one(self, *_args, **_kwargs):
                return {"status": "JUDGED", "attack_succeeded": False}

        with tempfile.TemporaryDirectory() as directory:
            source_dir = Path(directory)
            exports = source_dir / "exports"
            exports.mkdir()
            (exports / "session_item-1.jsonl").write_text(
                '{"type": "message", "role": "assistant", "content": "done"}\n',
                encoding="utf-8",
            )
            row = await evaluate_case(
                judge=FakeJudge(),
                semaphore=asyncio.Semaphore(1),
                agent="hermes",
                source_dir=source_dir,
                item={"id": 1},
                source_result={"sandbox_status": "JUDGE_ERROR"},
                judge_types=("hybrid",),
            )

        self.assertEqual(row["judges"]["hybrid"]["status"], "JUDGED")

    async def test_unreadable_session_excludes_only_that_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_dir = Path(directory)
            exports = source_dir / "exports"
            exports.mkdir()
            (exports / "session_item-1.jsonl").write_text('{"type": "message"', encoding="utf-8")

            row = await evaluate_case(
                judge=object(),
                semaphore=asyncio.Semaphore(1),
                agent="codex",
                source_dir=source_dir,
                item={"id": 1},
                source_result={"sandbox_status": "JUDGED"},
                judge_types=("hybrid",),
            )

        self.assertEqual(row["judges"]["hybrid"]["status"], "SOURCE_EXCLUDED")
        self.assertIn("JSONDecodeError", row["judges"]["hybrid"]["reason"])


if __name__ == "__main__":
    unittest.main()
