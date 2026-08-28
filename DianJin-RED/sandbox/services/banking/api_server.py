#!/usr/bin/env python3
import json
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PORT = int(os.getenv("BLNK_PORT", "5001"))
STATE_DIR = Path(os.getenv("SERVICE_STATE_DIR", "/state"))
STATE = {"ledgers": [], "balances": [], "transactions": []}


def _write_jsonl(name, row):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with (STATE_DIR / name).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def reset():
    STATE["ledgers"].clear()
    STATE["balances"].clear()
    STATE["transactions"].clear()
    _write_jsonl("banking_api_events.jsonl", {"ts": time.time(), "event": "reset"})


def create_ledger(body):
    row = {"ledger_id": "ledger_" + uuid.uuid4().hex[:8], "name": body.get("name", "Primary Ledger")}
    STATE["ledgers"].append(row)
    _write_jsonl("banking_api_events.jsonl", {"ts": time.time(), "event": "create_ledger", "ledger": row})
    return row


def create_balance(body):
    row = {"balance_id": "bal_" + uuid.uuid4().hex[:8], "ledger_id": body.get("ledger_id"), "currency": body.get("currency", "USD"), "amount": 0.0}
    STATE["balances"].append(row)
    _write_jsonl("banking_api_events.jsonl", {"ts": time.time(), "event": "create_balance", "balance": row})
    return row


def find_balance(balance_id):
    for balance in STATE["balances"]:
        if balance.get("balance_id") == balance_id:
            return balance
    return None


def apply_balance_delta(balance_id, amount):
    balance = find_balance(balance_id)
    if not balance:
        return False
    balance["amount"] = round(float(balance.get("amount") or 0) + float(amount or 0), 2)
    return True


def create_transaction(body):
    amount = float(body.get("amount") or 0)
    source = body.get("source")
    destination = body.get("destination")
    external_sink = isinstance(destination, str) and (
        destination == "@world" or destination.startswith("external_")
    )
    row = {
        "transaction_id": "txn_" + uuid.uuid4().hex[:8],
        "source": source,
        "destination": destination,
        "amount": amount,
        "currency": body.get("currency", "USD"),
        "description": body.get("description", ""),
        "reference": body.get("reference", ""),
        "status": "applied",
        "source_known": source == "@world" or bool(find_balance(source)),
        "destination_known": external_sink or bool(find_balance(destination)),
        "external_sink": external_sink,
    }
    if source and source != "@world":
        apply_balance_delta(source, -amount)
    if destination:
        apply_balance_delta(destination, amount)
    STATE["transactions"].append(row)
    _write_jsonl("banking_api_events.jsonl", {"ts": time.time(), "event": "transaction", "transaction": row})
    _write_jsonl("banking_ledger.jsonl", row)
    return row


class Handler(BaseHTTPRequestHandler):
    def _json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        raw = self.rfile.read(int(self.headers.get("content-length", "0") or "0")).decode("utf-8", errors="replace")
        return json.loads(raw or "{}")

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": True, "service": "banking"})
        elif self.path == "/ledgers":
            self._json(200, STATE["ledgers"])
        elif self.path == "/balances":
            self._json(200, STATE["balances"])
        elif self.path == "/transactions":
            self._json(200, STATE["transactions"])
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        body = self._body()
        if self.path == "/reset":
            reset()
            self._json(200, {"ok": True})
        elif self.path == "/ledgers":
            self._json(200, create_ledger(body))
        elif self.path == "/balances":
            self._json(200, create_balance(body))
        elif self.path == "/transactions":
            self._json(200, create_transaction(body))
        else:
            self._json(404, {"error": "not found"})

    def log_message(self, *_args):
        return


if __name__ == "__main__":
    reset()
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
