import json
import os
from pathlib import Path

from fastmcp import FastMCP


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    async def list_pending_transfers() -> str:
        """List pending transfer records initialized for the current case."""
        state_dir = Path(os.getenv("RED_AGENT_WORLD_SERVICE_STATE_DIR", "/state"))
        path = state_dir / "banking_pending_transfers.jsonl"
        rows = []
        if path.exists():
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    rows.append({"raw": line})
        return json.dumps({"pending_transfers": rows}, ensure_ascii=False)
