"""Minimal container workspace helpers shared by active runtimes."""

from __future__ import annotations

import io
import shlex
import tarfile
from pathlib import Path

from red_agent_world.sandbox.local_docker import LocalDockerSandbox


SANDBOX_WORKSPACE = "/workspace"


async def put_text_file(
    client: LocalDockerSandbox,
    target_path: str,
    content: str,
    mode: int = 0o644,
) -> None:
    if not client.container:
        raise RuntimeError("container is not started")
    target = Path(target_path)
    target_dir = str(target.parent)
    response = await client.execute_command("mkdir -p %s" % shlex.quote(target_dir))
    result = response.get("result", {})
    if result.get("exit_code") not in (0, None):
        raise RuntimeError("failed to create container directory: %s" % target_dir)
    data = content.encode("utf-8")
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w") as archive:
        info = tarfile.TarInfo(name=target.name)
        info.size = len(data)
        info.mode = mode
        archive.addfile(info, io.BytesIO(data))
    tar_stream.seek(0)
    if not client.container.put_archive(target_dir, tar_stream):
        raise RuntimeError("failed to write container file: %s" % target_path)
