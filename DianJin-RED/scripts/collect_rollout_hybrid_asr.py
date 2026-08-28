#!/usr/bin/env python3
"""Collect Hybrid-only ASR directly from completed rollout result JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_source(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("source must use AGENT=/absolute/result/directory")
    return name.strip(), Path(path).expanduser().resolve()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-file", type=Path, default=Path("test/U1.json"))
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--judge-model",
        required=True,
        help="Exact judge model recorded in setup metadata.",
    )
    parser.add_argument("--system-prefix-file", type=Path, default=None)
    args = parser.parse_args()
    dataset_file = args.dataset_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    prefix_file = args.system_prefix_file.expanduser().resolve() if args.system_prefix_file else None
    case_ids = [int(item["id"]) for item in json.loads(dataset_file.read_text(encoding="utf-8"))]
    rows: list[dict] = []
    agents: dict[str, dict] = {}
    sources: dict[str, str] = {}

    for agent, source_dir in args.source:
        sources[agent] = str(source_dir)
        by_id = {int(row["id"]): row for row in read_jsonl(source_dir / "exports" / "results.jsonl")}
        counters = {"total_cases": len(case_ids), "denominator": 0, "harmful": 0, "excluded": 0}
        for case_id in case_ids:
            source = by_id.get(case_id, {})
            source_status = str(source.get("sandbox_status") or "MISSING_RESULT")
            hybrid = (source.get("judges") or {}).get("hybrid") or {}
            judge_status = str(hybrid.get("status") or "MISSING_JUDGE")
            included = source_status == "JUDGED" and judge_status == "JUDGED"
            attack_succeeded = bool(hybrid.get("attack_succeeded")) if included else None
            if included:
                counters["denominator"] += 1
                counters["harmful"] += int(attack_succeeded is True)
            else:
                counters["excluded"] += 1
            rows.append(
                {
                    "agent": agent,
                    "case_id": case_id,
                    "source_status": source_status,
                    "judges": {
                        "hybrid": {
                            "status": judge_status if source_status == "JUDGED" else "SOURCE_EXCLUDED",
                            "attack_succeeded": attack_succeeded,
                            "score": hybrid.get("score") if included else None,
                            "reason": hybrid.get("reason") or source.get("reason") or "",
                            "evidence_refs": hybrid.get("evidence_refs") or [],
                        }
                    },
                }
            )
        counters["asr"] = counters["harmful"] / counters["denominator"] if counters["denominator"] else None
        agents[agent] = {"hybrid": counters}

    rows.sort(key=lambda row: (row["agent"], row["case_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    setup = {
        "experiment": "system_prompt_prefix" if prefix_file else "no_additional_system_prompt_prefix",
        "dataset_file": str(dataset_file),
        "case_ids": case_ids,
        "sources": sources,
        "judge_types": ["hybrid"],
        "judge_model": args.judge_model,
        "evidence_contract": {"hybrid": "all_environment_final_state_plus_trajectory"},
        "system_prompt_prefix_enabled": prefix_file is not None,
        "system_prompt_prefix_file": str(prefix_file) if prefix_file else None,
        "system_prompt_prefix": prefix_file.read_text(encoding="utf-8").strip() if prefix_file else "",
        "asr_denominator": "source rollout status == JUDGED and Hybrid Judge status == JUDGED",
    }
    (output_dir / "setup.json").write_text(json.dumps(setup, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps({"agents": agents}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "case_judges.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    with (output_dir / "case_judges.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["agent", "case_id", "source_status", "hybrid_status", "hybrid_attack_succeeded", "hybrid_score", "hybrid_reason"],
        )
        writer.writeheader()
        for row in rows:
            hybrid = row["judges"]["hybrid"]
            writer.writerow(
                {
                    "agent": row["agent"],
                    "case_id": row["case_id"],
                    "source_status": row["source_status"],
                    "hybrid_status": hybrid["status"],
                    "hybrid_attack_succeeded": hybrid["attack_succeeded"],
                    "hybrid_score": hybrid["score"],
                    "hybrid_reason": hybrid["reason"],
                }
            )


if __name__ == "__main__":
    main()
