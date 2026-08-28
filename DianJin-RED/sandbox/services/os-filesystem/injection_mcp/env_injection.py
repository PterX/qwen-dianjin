#!/usr/bin/env python3
import argparse
import json
import shlex
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import PurePosixPath


def api_execute(api_url, command):
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/api/v1/execute",
        data=json.dumps({"command": command}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode())


def run_server(kind, port, api_url):
    def tools():
        if kind == "agent":
            return [
                {"name": "list_directory", "description": "List a directory in the external file store.", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}}},
                {"name": "read_file", "description": "Read a file in the external file store.", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
                {"name": "write_file", "description": "Write a file in the external file store.", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
            ]
        return [
            {"name": "inject_file", "description": "Add a file to the external file store.", "inputSchema": {"type": "object", "additionalProperties": False, "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]}},
            {"name": "inject_append", "description": "Append content to a file in the external file store.", "inputSchema": {"type": "object", "additionalProperties": False, "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]}},
        ]

    def service_path(raw, default="/srv/external-files"):
        return str(raw or default).strip() or default

    def normalize_external_path(raw, default="/srv/external-files"):
        value = service_path(raw, default)
        if value in ("", "."):
            return "/srv/external-files"
        if value.startswith("./"):
            return "/srv/external-files/" + value[2:].lstrip("/")
        if not value.startswith("/"):
            return "/srv/external-files/" + value.lstrip("/")
        return value

    def write_command(raw_path, content, append=False):
        target = normalize_external_path(raw_path, "/srv/external-files/file_store_note.md")
        parent = str(PurePosixPath(target).parent) or "/srv/external-files"
        redirect = ">>" if append else ">"
        return "mkdir -p %s && cat %s %s <<'FILESTORE_EOF'\n%s\nFILESTORE_EOF" % (
            shlex.quote(parent), redirect, shlex.quote(target), str(content or "")
        )

    def call_tool(name, args):
        args = args or {}
        if name == "list_directory":
            path = shlex.quote(normalize_external_path(args.get("path")))
            return api_execute(api_url, f"ls -la {path}")
        if name == "read_file":
            path = shlex.quote(normalize_external_path(args.get("path"), ""))
            return api_execute(api_url, f"cat {path}")
        if name in ("write_file", "inject_file"):
            return api_execute(api_url, write_command(args.get("path") or args.get("file_path"), args.get("content") or ""))
        if name == "inject_append":
            return api_execute(api_url, write_command(args.get("file_path"), args.get("content") or "", append=True))
        return {"success": False, "error": "unknown tool: " + str(name)}

    def rpc(req, result=None, error=None):
        out = {"jsonrpc": "2.0", "id": req.get("id")}
        if error:
            out["error"] = error
        else:
            out["result"] = result
        return out

    class Handler(BaseHTTPRequestHandler):
        def _json(self, status, payload, session_id=None):
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if session_id:
                self.send_header("Mcp-Session-Id", session_id)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/health", "/mcp"):
                self._json(200, {"ok": True, "tools": [t["name"] for t in tools()]})
                return
            self._json(404, {"error": "not found"})

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("content-length", "0") or "0")).decode("utf-8", errors="replace")
            req = json.loads(raw or "{}")
            method = req.get("method")
            if method == "initialize":
                self._json(200, rpc(req, {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}, "serverInfo": {"name": "external_files-" + kind, "version": "0.1"}}), session_id="external_files-session")
                return
            if method == "notifications/initialized":
                self._json(202, {})
                return
            if method == "tools/list":
                self._json(200, rpc(req, {"tools": tools()}))
                return
            if method == "tools/call":
                params = req.get("params") or {}
                result = call_tool(params.get("name"), params.get("arguments") or {})
                self._json(200, rpc(req, {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}], "isError": not bool(result.get("success", result.get("ok", True)))}))
                return
            self._json(400, rpc(req, error={"message": "unsupported method"}))

        def log_message(self, *_args):
            return

    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["agent", "injection"], required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--api-url", required=True)
    args = parser.parse_args()
    run_server(args.kind, args.port, args.api_url)
