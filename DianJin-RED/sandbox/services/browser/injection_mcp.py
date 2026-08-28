#!/usr/bin/env python3
import argparse
import json
import os
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def serve(port: int, website_url: str) -> None:
    state_dir = Path(os.getenv("RED_AGENT_WORLD_SERVICE_STATE_DIR") or "")

    def update_html(html: str) -> dict:
        payload = json.dumps({"html": html}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            website_url.rstrip("/") + "/api/update-html",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read().decode("utf-8", errors="replace")
        return json.loads(raw) if raw.strip() else {"success": True}

    def tools() -> list:
        return [{
            "name": "update_html_content",
            "description": "Set the HTML served by the local mutable website.",
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"html": {"type": "string"}},
                "required": ["html"],
            },
        }]

    def rpc(request: dict, result=None, error=None) -> dict:
        response = {"jsonrpc": "2.0", "id": request.get("id")}
        response["error" if error else "result"] = error or result
        return response

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status: int, payload: dict, session_id: str = "") -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if session_id:
                self.send_header("Mcp-Session-Id", session_id)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in {"/health", "/mcp"}:
                self.reply(200, {"ok": True, "backend": "local-mutable-website", "tools": [row["name"] for row in tools()]})
            else:
                self.reply(404, {"error": "not found"})

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("content-length", "0") or "0")).decode("utf-8", errors="replace")
            request = json.loads(raw or "{}")
            method = request.get("method")
            if method == "initialize":
                self.reply(200, rpc(request, {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}}, "serverInfo": {"name": "browser-injection", "version": "1"}}), "browser-injection-session")
            elif method == "notifications/initialized":
                self.reply(202, {})
            elif method == "tools/list":
                self.reply(200, rpc(request, {"tools": tools()}))
            elif method == "tools/call":
                params = request.get("params") or {}
                args = params.get("arguments") or {}
                if params.get("name") != "update_html_content":
                    result = {"ok": False, "error": "unknown tool"}
                elif "html" not in args:
                    result = {"ok": False, "error": "invalid arguments", "details": ["missing required fields: html"]}
                elif not isinstance(args.get("html"), str):
                    result = {"ok": False, "error": "invalid arguments", "details": ["html must be a string"]}
                elif set(args) - {"html"}:
                    extra = sorted(set(args) - {"html"})
                    result = {"ok": False, "error": "invalid arguments", "details": ["unexpected fields: " + ", ".join(extra)]}
                else:
                    backend = update_html(args["html"])
                    result = {"ok": True, "backend": backend}
                    if state_dir:
                        state_dir.mkdir(parents=True, exist_ok=True)
                        with (state_dir / "custom_website_injection_events.jsonl").open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps({"ts": time.time(), "actor": "adversary", "service": "browser", "tool": "update_html_content", "arguments": args, "result": result}, ensure_ascii=False) + "\n")
                self.reply(200, rpc(request, {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}], "isError": not result.get("ok", False)}))
            else:
                self.reply(400, rpc(request, error={"message": "unsupported method"}))

        def log_message(self, *_args):
            return

    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--website-url", required=True)
    args = parser.parse_args()
    serve(args.port, args.website_url)
