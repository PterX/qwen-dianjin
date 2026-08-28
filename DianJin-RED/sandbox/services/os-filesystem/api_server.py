#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

API_PORT = int(os.getenv("API_PORT", "8034"))
STATE_DIR = Path(os.getenv("SERVICE_STATE_DIR", "/state"))
SERVICE_ROOT = Path(os.getenv("FILESYSTEM_ROOT", "/srv/external-files"))


def _write_jsonl(name, row):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with (STATE_DIR / name).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def reset_state():
    root = SERVICE_ROOT
    if str(root) in {"/", "/workspace"}:
        raise RuntimeError("external filesystem root must not be / or /workspace")
    if root.exists():
        for child in root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    _write_jsonl("os_filesystem_api_events.jsonl", {"ts": time.time(), "event": "reset", "filesystem_root": str(root)})


class Handler(BaseHTTPRequestHandler):
    def _json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": True, "service": "external_files"})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("content-length", "0") or "0")).decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw or "{}")
        except Exception:
            payload = {}
        if self.path == "/api/v1/reset":
            reset_state()
            self._json(200, {"ok": True})
            return
        if self.path == "/api/v1/permission-log":
            _write_jsonl("permission_log.jsonl", payload.get("entry", payload))
            self._json(200, {"ok": True})
            return
        if self.path == "/api/v1/execute":
            command = str(payload.get("command") or "")
            cwd = SERVICE_ROOT
            proc = subprocess.run(command, shell=True, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
            row = {"ts": time.time(), "event": "execute", "command": command, "returncode": proc.returncode, "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]}
            _write_jsonl("os_filesystem_api_events.jsonl", row)
            self._json(200, {"success": proc.returncode == 0, "stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode})
            return
        self._json(404, {"error": "not found"})

    def log_message(self, *_args):
        return


if __name__ == "__main__":
    reset_state()
    HTTPServer(("0.0.0.0", API_PORT), Handler).serve_forever()
