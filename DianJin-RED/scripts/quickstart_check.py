#!/usr/bin/env python3
"""Offline preflight for the public Codex Quickstart."""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import shlex
import tempfile
from pathlib import Path

import docker
import yaml

from red_agent_world.common.config_generator import ConfigGenerator
from red_agent_world.common.openai_proxy import OpenAIProxyServer
from red_agent_world.sandbox.local_docker import LocalDockerSandbox


ROOT = Path(__file__).resolve().parents[1]


async def network_smoke(image: str) -> None:
    """Prove that the runtime reaches the authenticated host proxy but not the Internet."""
    token = secrets.token_urlsafe(32)
    proxy = OpenAIProxyServer(
        host="0.0.0.0",
        port=0,
        upstream_base_url="http://127.0.0.1:1/v1",
        upstream_api_key="unused-smoke-upstream-token",
        client_api_key=token,
        model_name="quickstart-smoke-model",
    )
    proxy.start()
    port = int(proxy.server_address[1])
    try:
        async with LocalDockerSandbox(image=image, init_mode="exec", network_mode="isolated") as sandbox:
            await sandbox.initialize()
            url = "http://host.docker.internal:%d/models" % port
            unauthenticated = await sandbox.execute_command(
                "curl -sS -o /dev/null -w '%%{http_code}' %s" % shlex.quote(url)
            )
            if unauthenticated["result"]["stdout"].strip() != "401":
                raise RuntimeError("isolated proxy rejected-token check failed: %r" % unauthenticated)
            authenticated = await sandbox.execute_command(
                "curl -sS -o /dev/null -w '%%{http_code}' -H %s %s"
                % (shlex.quote("Authorization: Bearer " + token), shlex.quote(url))
            )
            if authenticated["result"]["stdout"].strip() != "200":
                raise RuntimeError("isolated proxy authenticated check failed: %r" % authenticated)
            external = await sandbox.execute_command(
                "curl -fsS --connect-timeout 2 --max-time 4 https://example.com >/dev/null"
            )
            if int(external["result"]["exit_code"]) == 0:
                raise RuntimeError("isolated runtime unexpectedly reached the public Internet")
    finally:
        proxy.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-image", action="store_true", help="Do not require the Codex runtime image.")
    parser.add_argument(
        "--network-smoke",
        action="store_true",
        help="Launch a temporary runtime and verify authenticated host access plus blocked Internet egress.",
    )
    args = parser.parse_args()
    if args.skip_image and args.network_smoke:
        parser.error("--network-smoke requires the runtime image")

    for relative in (
        "config/config_local_private.example.json",
        "config/openclaw.json",
        "sandbox/service_world/config/service_profiles.yaml",
        "taxonomy/registry.json",
        "test/E1.json",
    ):
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit("missing Quickstart asset: %s" % relative)

    json.loads((ROOT / "config/config_local_private.example.json").read_text(encoding="utf-8"))
    json.loads((ROOT / "taxonomy/registry.json").read_text(encoding="utf-8"))
    yaml.safe_load((ROOT / "sandbox/service_world/config/service_profiles.yaml").read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="redagentbench-quickstart-") as directory:
        output = Path(directory) / "openclaw.json"
        ConfigGenerator(str(ROOT / "config/openclaw.json")).generate_config(
            api_key="offline-test-token",
            base_url="http://host.docker.internal:12345/v1",
            model="offline-test-model",
            output_path=str(output),
        )
        generated = json.loads(output.read_text(encoding="utf-8"))
        assert generated["gateway"]["bind"] == "loopback"
        assert generated["gateway"]["auth"]["token"] not in {"", "GENERATED_PER_RUN"}

    image = "red-agent-world-codex:v1"
    if not args.skip_image:
        client = docker.from_env()
        client.ping()
        try:
            client.images.get(image)
        except docker.errors.ImageNotFound as exc:
            raise SystemExit(
                "missing red-agent-world-codex:v1; run 'bash docker/build-agent-images.sh codex'"
            ) from exc

    if args.network_smoke:
        asyncio.run(network_smoke(image))

    print("Quickstart preflight passed.")


if __name__ == "__main__":
    main()
