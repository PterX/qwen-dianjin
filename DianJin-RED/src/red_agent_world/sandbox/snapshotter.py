"""Host-side workspace snapshots kept outside the agent runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


MAX_SAMPLE_BYTES = 4096
MAX_SAMPLE_FILES = 20


@dataclass(frozen=True)
class WorkspaceSnapshot:
    files: Dict[str, str]
    samples: Dict[str, str]
    sizes: Dict[str, int]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "file_count": len(self.files),
            "files": dict(sorted(self.files.items())),
            "sizes": dict(sorted(self.sizes.items())),
        }


class SandboxSnapshotter:
    """Capture and compare the host directory bind-mounted at ``/workspace``."""

    def __init__(self, workspace_dir: Path, excluded_prefixes: Iterable[str] = (".git/",)) -> None:
        self.workspace_dir = workspace_dir.resolve()
        self.excluded_prefixes = tuple(excluded_prefixes)

    def capture(self) -> WorkspaceSnapshot:
        files: Dict[str, str] = {}
        samples: Dict[str, str] = {}
        sizes: Dict[str, int] = {}
        if not self.workspace_dir.exists():
            return WorkspaceSnapshot(files, samples, sizes)
        for path in sorted(self.workspace_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(self.workspace_dir).as_posix()
            if any(rel.startswith(prefix) for prefix in self.excluded_prefixes):
                continue
            try:
                content = path.read_bytes()
                files[rel] = hashlib.sha256(content).hexdigest()
                samples[rel] = content[:MAX_SAMPLE_BYTES].decode("utf-8", errors="replace")
                sizes[rel] = len(content)
            except OSError:
                files[rel] = "<read_error>"
                samples[rel] = "<read_error>"
                sizes[rel] = -1
        return WorkspaceSnapshot(files, samples, sizes)

    @staticmethod
    def diff(baseline: WorkspaceSnapshot, final: WorkspaceSnapshot) -> Dict[str, List[str]]:
        before = baseline.files
        after = final.files
        return {
            "created": sorted(path for path in after if path not in before),
            "modified": sorted(path for path, digest in after.items() if path in before and before[path] != digest),
            "deleted": sorted(path for path in before if path not in after),
        }

    def changed_file_samples(
        self,
        baseline: WorkspaceSnapshot,
        final: WorkspaceSnapshot,
    ) -> List[Dict[str, Any]]:
        changes = self.diff(baseline, final)
        rows: List[Dict[str, Any]] = []
        for change in ("created", "modified", "deleted"):
            for path in changes[change]:
                if len(rows) >= MAX_SAMPLE_FILES:
                    return rows
                before = baseline.samples.get(path)
                after = final.samples.get(path)
                rows.append({
                    "path": path,
                    "change": change,
                    "before": before,
                    "after": after,
                    "before_truncated": baseline.sizes.get(path, 0) > MAX_SAMPLE_BYTES,
                    "after_truncated": final.sizes.get(path, 0) > MAX_SAMPLE_BYTES,
                })
        return rows

    def evidence(self, baseline: WorkspaceSnapshot, final: WorkspaceSnapshot) -> List[Dict[str, Any]]:
        workspace_diff: Dict[str, Any] = self.diff(baseline, final)
        workspace_diff["file_samples"] = self.changed_file_samples(baseline, final)
        return [
            {
                "relative_path": "baseline.json",
                "path": str(self.workspace_dir),
                "content": json.dumps(baseline.as_dict(), ensure_ascii=False, sort_keys=True),
            },
            {
                "relative_path": "final_snapshot.json",
                "path": str(self.workspace_dir),
                "content": json.dumps(final.as_dict(), ensure_ascii=False, sort_keys=True),
            },
            {
                "relative_path": "workspace_diff.json",
                "path": str(self.workspace_dir),
                "content": json.dumps(workspace_diff, ensure_ascii=False, sort_keys=True),
            },
        ]
