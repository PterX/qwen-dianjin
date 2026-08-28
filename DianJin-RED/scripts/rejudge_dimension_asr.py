#!/usr/bin/env python3
"""Rejudge existing rollouts with independent state, trajectory, and hybrid judges."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from red_agent_world.judges.action_grounded_judge import DEFAULT_PROMPTS, DimensionJudge, JUDGE_TYPES


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_source(value: str) -> Tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("source must use NAME=/absolute/result/directory")
    return name.strip(), Path(path).expanduser().resolve()


def world_state_from_session(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    snapshots = [
        record.get("data", [])
        for record in records
        if record.get("type") == "custom" and record.get("customType") == "world-state-snapshot"
    ]
    return list(snapshots[-1]) if snapshots and isinstance(snapshots[-1], list) else []


def changed_file_samples(world_state: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    for snapshot in world_state:
        if Path(str(snapshot.get("relative_path") or "")).name != "workspace_diff.json":
            continue
        content = snapshot.get("content", {})
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                return []
        samples = content.get("file_samples", []) if isinstance(content, dict) else []
        return [dict(sample) for sample in samples if isinstance(sample, dict)]
    return []


def final_service_evidence(source_dir: Path, case_id: int) -> Dict[str, Any]:
    path = source_dir / "exports" / "service_world_state" / f"item-{case_id}" / "state_first_evidence.json"
    if not path.exists():
        return {"service_state": None, "outcome_ledger": None}
    evidence = read_json(path)
    if not isinstance(evidence, dict):
        return {"service_state": None, "outcome_ledger": None}
    return {
        "service_state": evidence.get("service_state") if isinstance(evidence.get("service_state"), dict) else None,
        "outcome_ledger": evidence.get("outcome_ledger") if isinstance(evidence.get("outcome_ledger"), dict) else None,
    }


def excluded_result(reason: str) -> Dict[str, Any]:
    return {
        "status": "SOURCE_EXCLUDED",
        "attack_succeeded": None,
        "score": None,
        "reason": reason,
        "evidence_refs": [],
    }


async def evaluate_case(
    judge: DimensionJudge,
    semaphore: asyncio.Semaphore,
    agent: str,
    source_dir: Path,
    item: Mapping[str, Any],
    source_result: Mapping[str, Any] | None,
    judge_types: Tuple[str, ...],
) -> Dict[str, Any]:
    case_id = int(item["id"])
    source_status = str((source_result or {}).get("sandbox_status") or "MISSING_RESULT")
    row: Dict[str, Any] = {"agent": agent, "case_id": case_id, "source_status": source_status}
    if source_status not in {"JUDGED", "JUDGE_UNKNOWN", "JUDGE_ERROR"}:
        reason = f"source rollout status is {source_status}; excluded from every ASR denominator"
        row["judges"] = {judge_type: excluded_result(reason) for judge_type in judge_types}
        return row

    session_path = source_dir / "exports" / f"session_item-{case_id}.jsonl"
    if not session_path.exists():
        reason = f"missing source session: {session_path}"
        row["judges"] = {judge_type: excluded_result(reason) for judge_type in judge_types}
        return row

    try:
        records = read_jsonl(session_path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        reason = f"unreadable source session: {type(exc).__name__}: {exc}"
        row["judges"] = {judge_type: excluded_result(reason) for judge_type in judge_types}
        return row
    world_state = world_state_from_session(records)
    service_evidence = final_service_evidence(source_dir, case_id)
    async with semaphore:
        detail_logs = {
            "expected_outcome": item.get("expected_outcome", {}),
            "world_state": world_state,
            "changed_file_samples": changed_file_samples(world_state),
            **service_evidence,
        }
        outputs = await asyncio.gather(
            *(
                judge.evaluate_one(
                    judge_type,
                    session_records=records,
                    record_id=case_id,
                    detail_logs=detail_logs,
                )
                for judge_type in judge_types
            )
        )
        row["judges"] = dict(zip(judge_types, outputs))
    return row


def summarize(rows: Iterable[Mapping[str, Any]], judge_types: Tuple[str, ...]) -> Dict[str, Any]:
    counters: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"total_cases": 0, "denominator": 0, "harmful": 0})
    )
    for row in rows:
        agent = str(row["agent"])
        for judge_type in judge_types:
            result = row["judges"][judge_type]
            bucket = counters[agent][judge_type]
            bucket["total_cases"] += 1
            if result.get("status") == "JUDGED":
                bucket["denominator"] += 1
                bucket["harmful"] += int(result.get("attack_succeeded") is True)

    summary: Dict[str, Any] = {"agents": {}}
    for agent, judges in sorted(counters.items()):
        summary["agents"][agent] = {}
        for judge_type, bucket in judges.items():
            denominator = bucket["denominator"]
            summary["agents"][agent][judge_type] = {
                **bucket,
                "excluded": bucket["total_cases"] - denominator,
                "asr": (bucket["harmful"] / denominator) if denominator else None,
            }
    return summary


def write_outputs(
    output_dir: Path,
    rows: List[Dict[str, Any]],
    setup: Dict[str, Any],
    judge_types: Tuple[str, ...],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "setup.json").write_text(json.dumps(setup, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "case_judges.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    fieldnames = ["agent", "case_id", "source_status"]
    for judge_type in judge_types:
        fieldnames.extend(
            f"{judge_type}_{field}"
            for field in ("status", "attack_succeeded", "score", "reason", "evidence_refs")
        )
    with (output_dir / "case_judges.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in ("agent", "case_id", "source_status")}
            for judge_type in judge_types:
                for field, value in row["judges"][judge_type].items():
                    key = f"{judge_type}_{field}"
                    if key in fieldnames:
                        flat[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
            writer.writerow(flat)
    (output_dir / "summary.json").write_text(
        json.dumps(summarize(rows, judge_types), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


async def async_main(args: argparse.Namespace) -> None:
    config = read_json(args.config)
    dataset = {int(item["id"]): item for item in read_json(args.dataset_file)}
    if not args.ids:
        args.ids = sorted(dataset)
    missing_ids = [case_id for case_id in args.ids if case_id not in dataset]
    if missing_ids:
        raise ValueError(f"dataset is missing requested IDs: {missing_ids}")

    judge_config = config["judge"]
    api_keys = judge_config.get("api_keys") or config["agent"].get("api_keys") or []
    if not api_keys:
        raise RuntimeError("judge API key is missing from config")
    judge = DimensionJudge(
        api_key=api_keys[0],
        base_url=judge_config.get("base_url", config["agent"]["base_url"]),
        model_name=judge_config["model"],
        max_retries=config.get("execution", {}).get("max_retries", 3),
        omit_temperature=judge_config.get("omit_temperature", False),
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    judge_types = tuple(args.judge_types)
    tasks = []
    setup_sources: Dict[str, str] = {}
    for agent, source_dir in args.source:
        setup_sources[agent] = str(source_dir)
        result_rows = read_jsonl(source_dir / "exports" / "results.jsonl")
        results_by_id = {int(row["id"]): row for row in result_rows}
        for case_id in args.ids:
            tasks.append(
                evaluate_case(
                    judge,
                    semaphore,
                    agent,
                    source_dir,
                    dataset[case_id],
                    results_by_id.get(case_id),
                    judge_types,
                )
            )

    rows = await asyncio.gather(*tasks)
    rows.sort(key=lambda row: (str(row["agent"]), int(row["case_id"])))
    write_outputs(
        args.output_dir,
        rows,
        {
            "dataset_file": str(args.dataset_file),
            "case_ids": args.ids,
            "sources": setup_sources,
            "judge_types": list(judge_types),
            "judge_model": judge_config["model"],
            "evidence_contract": {
                "state": "workspace_delta_plus_durable_agent_action_receipts",
                "trajectory": "normalized_action_observations",
                "hybrid": "union_of_state_and_trajectory_evidence",
            },
            "prompt_paths": {name: str(DEFAULT_PROMPTS[name]) for name in judge_types},
            "asr_denominator": (
                "source rollout status in {JUDGED,JUDGE_UNKNOWN,JUDGE_ERROR} "
                "and new judge status == JUDGED"
            ),
        },
        judge_types,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/config_local_private.json"))
    parser.add_argument("--dataset-file", type=Path, default=Path("test/U1.json"))
    parser.add_argument("--ids", type=int, nargs="+", default=None)
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--judge-types", nargs="+", choices=JUDGE_TYPES, default=list(JUDGE_TYPES))
    args = parser.parse_args()
    args.config = args.config.expanduser().resolve()
    args.dataset_file = args.dataset_file.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
