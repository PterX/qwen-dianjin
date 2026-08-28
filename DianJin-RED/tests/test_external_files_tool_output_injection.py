import importlib.util
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


SERVER_PATH = (
    Path(__file__).resolve().parents[1]
    / "sandbox"
    / "services"
    / "os-filesystem"
    / "mcp_server"
    / "main.py"
)


def load_server_module():
    spec = importlib.util.spec_from_file_location("external_files_mcp_server", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def post_json(url, payload, method="POST"):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def test_external_files_agent_supports_once_override_via_inject_route(tmp_path, monkeypatch):
    module = load_server_module()
    monkeypatch.setenv("RED_AGENT_WORLD_SERVICE_STATE_DIR", str(tmp_path / "state"))

    class FakeAPIHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.dumps({
                "success": True,
                "ok": True,
                "returncode": 0,
                "stdout": "REAL_FILE_CONTENT",
                "stderr": "",
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    api_server = HTTPServer(("127.0.0.1", 0), FakeAPIHandler)
    agent_server = module.create_server(
        "agent",
        0,
        "http://127.0.0.1:%s" % api_server.server_port,
    )
    api_thread = threading.Thread(target=api_server.serve_forever, daemon=True)
    agent_thread = threading.Thread(target=agent_server.serve_forever, daemon=True)
    api_thread.start()
    agent_thread.start()
    agent_url = "http://127.0.0.1:%s" % agent_server.server_port

    try:
        injected = {"success": True, "stdout": "TAMPERED_FILE_CONTENT"}
        receipt = post_json(agent_url + "/inject", {
            "tool": "read_file",
            "content": json.dumps(injected),
            "mode": "override",
            "once": True,
        })
        assert receipt["ok"] is True

        call = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "read_file", "arguments": {"path": "notice.json"}},
        }
        first = post_json(agent_url + "/mcp", call)
        assert json.loads(first["result"]["content"][0]["text"]) == injected
        assert get_json(agent_url + "/inject/rules") == {"rules": []}

        second = post_json(agent_url + "/mcp", call)
        original = json.loads(second["result"]["content"][0]["text"])
        assert original["stdout"] == "REAL_FILE_CONTENT"
    finally:
        agent_server.shutdown()
        api_server.shutdown()
        agent_server.server_close()
        api_server.server_close()


def test_external_files_injection_modes_preserve_prefix_and_append_order():
    module = load_server_module()
    rules = module.ToolOutputInjectionRules()
    original = {"ok": True, "stdout": "REAL"}

    rules.add({"tool": "read_file", "content": "PREFIX", "mode": "prefix"})
    prefixed = rules.apply("read_file", original)
    assert prefixed.startswith("PREFIX\n")
    assert json.dumps(original, ensure_ascii=False) in prefixed

    rules.clear()
    rules.add({"tool": "read_file", "content": "APPEND", "mode": "append"})
    appended = rules.apply("read_file", original)
    assert appended.endswith("\nAPPEND")
    assert json.dumps(original, ensure_ascii=False) in appended
