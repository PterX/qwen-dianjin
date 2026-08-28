import json
from pathlib import Path
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from red_agent_world.common.config_generator import ConfigGenerator
from red_agent_world.common.openai_proxy import OpenAIProxyServer
from red_agent_world.runners.codex_sandbox_runner import CodexSandboxRunner
from red_agent_world.runners.mcp_control_gateway import GatewayState, make_handler


def test_local_model_proxy_rejects_missing_token():
    proxy = OpenAIProxyServer(
        host="127.0.0.1",
        port=0,
        upstream_base_url="https://upstream.invalid/v1",
        upstream_api_key="real-upstream-secret",
        client_api_key="per-run-client-token",
        model_name="test-model",
    )
    proxy.start()
    host, port = proxy.server_address
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen("http://%s:%s/models" % (host, port), timeout=5)
        assert exc.value.code == 401

        request = urllib.request.Request(
            "http://%s:%s/models" % (host, port),
            headers={"Authorization": "Bearer per-run-client-token"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["data"][0]["id"] == "test-model"
    finally:
        proxy.stop()


def test_openclaw_gateway_tokens_are_unique_and_loopback_only(tmp_path):
    template = tmp_path / "template.json"
    template.write_text(
        json.dumps({
            "models": {"providers": {"DefaultProvider": {"models": [{}]}}},
            "agents": {"defaults": {}},
            "gateway": {"bind": "lan", "auth": {"token": "placeholder"}, "remote": {}},
        }),
        encoding="utf-8",
    )
    generator = ConfigGenerator(str(template))
    tokens = []
    for index in range(2):
        output = tmp_path / ("generated-%s.json" % index)
        generator.generate_config("proxy-token", "http://host.docker.internal:9000/v1", "model", str(output))
        config = json.loads(output.read_text(encoding="utf-8"))
        assert config["gateway"]["bind"] == "loopback"
        assert config["gateway"]["auth"]["token"] == config["gateway"]["remote"]["token"]
        tokens.append(config["gateway"]["auth"]["token"])
    assert tokens[0] != tokens[1]


def test_mcp_control_routes_require_control_token(tmp_path):
    state = GatewayState("browser", "http://127.0.0.1:9/mcp", tmp_path, "control-secret")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = "http://127.0.0.1:%s/control/rules" % server.server_port
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(url, timeout=5)
        assert exc.value.code == 401

        request = urllib.request.Request(url, headers={"X-RED-Control-Token": "control-secret"})
        with urllib.request.urlopen(request, timeout=5) as response:
            assert json.loads(response.read().decode("utf-8")) == {"ok": True, "rules": []}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_runtime_network_defaults_to_isolated():
    runner = CodexSandboxRunner.__new__(CodexSandboxRunner)
    runner.config = {"agent_runtimes": {"codex": {"image": "codex:test", "init_mode": "exec"}}}
    runner.runtime_args = SimpleNamespace(
        runtime_image=None,
        runtime_init_mode=None,
        runtime_container_api_port=None,
    )
    runner._active_service_sandbox = None
    assert runner.runtime_docker_config()["network_mode"] == "isolated"


def test_service_compose_files_do_not_use_host_networking():
    repository_root = Path(__file__).resolve().parents[1]
    compose_files = sorted((repository_root / "sandbox" / "services").glob("*/docker-compose.yml"))
    assert compose_files
    for compose_file in compose_files:
        contents = compose_file.read_text(encoding="utf-8")
        assert "network_mode: host" not in contents, compose_file
        assert 'com.docker.network.bridge.enable_ip_masquerade: "false"' in contents, compose_file
