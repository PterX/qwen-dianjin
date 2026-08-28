"""
Ledger Blnk Service Plugin — local banking/ledger backed by Blnk.

Provides MCP tools for ledger management, balance queries, deposits,
payments, transfers, and transaction history.
"""
import os
import json
import uuid
from typing import Any, Dict, Optional

import httpx
from fastmcp import FastMCP

# --- Configuration ---

BLNK_API_URL = os.getenv("BLNK_API_URL", "http://127.0.0.1:5001")
BLNK_API_KEY = os.getenv("BLNK_API_KEY", "local-ledger-test-key")

# --- HTTP helpers ---

_http: Optional[httpx.AsyncClient] = None


async def _client() -> httpx.AsyncClient:
    global _http
    if _http is None:
        _http = httpx.AsyncClient(timeout=30.0)
    return _http


def _headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-Blnk-Key": BLNK_API_KEY,
    }


def _backend_destination(destination: str) -> str:
    """Represent named external payees as Blnk's real external sink."""
    return "@world" if destination.startswith("external_") else destination


def _restore_external_destination(result: Any, requested_destination: str) -> Any:
    """Keep the named payee while preserving the underlying transaction id."""
    if not requested_destination.startswith("external_") or not isinstance(result, dict):
        return result
    return {
        **result,
        "backend_destination": result.get("destination") or "@world",
        "backend_sink": "@world",
        "destination": requested_destination,
        "external_sink": True,
    }


async def _req(method: str, path: str, *,
               params: Optional[Dict[str, Any]] = None,
               body: Optional[Dict[str, Any]] = None) -> Any:
    c = await _client()
    r = await c.request(method, f"{BLNK_API_URL}{path}",
                        params=params, json=body, headers=_headers())
    r.raise_for_status()
    if r.content:
        return r.json()
    return {}


# ============================================================
# Public: register tools onto an MCP server
# ============================================================

def register(mcp: FastMCP):
    """Register all Blnk banking tools onto the given MCP server instance."""

    # ----------------------------------------------------------
    # Ledgers
    # ----------------------------------------------------------

    @mcp.tool()
    async def create_ledger(name: str, description: Optional[str] = None) -> str:
        """Create a new ledger (accounting book)."""
        try:
            body: Dict[str, Any] = {"name": name}
            if description:
                body["description"] = description
            result = await _req("POST", "/ledgers", body=body)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def get_ledger(ledger_id: str) -> str:
        """Get ledger details by ID."""
        try:
            result = await _req("GET", f"/ledgers/{ledger_id}")
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def list_ledgers() -> str:
        """List all ledgers."""
        try:
            result = await _req("GET", "/ledgers")
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ----------------------------------------------------------
    # Balances (accounts)
    # ----------------------------------------------------------

    @mcp.tool()
    async def create_balance(ledger_id: str, currency: str = "USD",
                             precision: int = 100) -> str:
        """Create a new balance (wallet/account) in a ledger."""
        try:
            body = {
                "ledger_id": ledger_id,
                "currency": currency,
                "precision": precision,
            }
            result = await _req("POST", "/balances", body=body)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def get_balance(balance_id: str) -> str:
        """Get current balance details including credit, debit, and available balance."""
        try:
            result = await _req("GET", f"/balances/{balance_id}")
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def list_balances() -> str:
        """List all balances."""
        try:
            result = await _req("GET", "/balances")
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ----------------------------------------------------------
    # Transactions (deposit, payment, transfer)
    # ----------------------------------------------------------

    @mcp.tool()
    async def deposit(destination: str, amount: float, currency: str = "USD",
                      description: str = "deposit",
                      reference: Optional[str] = None) -> str:
        """Deposit money into a balance. Uses @world as the source (external funds in)."""
        try:
            body: Dict[str, Any] = {
                "source": "@world",
                "destination": destination,
                "amount": amount,
                "currency": currency,
                "precision": 100,
                "reference": reference or str(uuid.uuid4()),
                "description": description,
                "allow_overdraft": True,
                "skip_queue": True,
            }
            result = await _req("POST", "/transactions", body=body)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def withdraw(source: str, amount: float, currency: str = "USD",
                       description: str = "withdrawal",
                       reference: Optional[str] = None) -> str:
        """Withdraw money from a balance. Uses @world as destination (funds out)."""
        try:
            body: Dict[str, Any] = {
                "source": source,
                "destination": "@world",
                "amount": amount,
                "currency": currency,
                "precision": 100,
                "reference": reference or str(uuid.uuid4()),
                "description": description,
                "allow_overdraft": True,
                "skip_queue": True,
            }
            result = await _req("POST", "/transactions", body=body)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def transfer(source: str, destination: str, amount: float,
                       currency: str = "USD",
                       description: str = "transfer",
                       reference: Optional[str] = None) -> str:
        """Transfer money to a balance or a named external_* payee."""
        try:
            body: Dict[str, Any] = {
                "source": source,
                "destination": _backend_destination(destination),
                "amount": amount,
                "currency": currency,
                "precision": 100,
                "reference": reference or str(uuid.uuid4()),
                "description": description,
                "allow_overdraft": True,
                "skip_queue": True,
            }
            result = await _req("POST", "/transactions", body=body)
            result = _restore_external_destination(result, destination)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def pay(source: str, destination: str, amount: float,
                  currency: str = "USD",
                  description: str = "payment",
                  reference: Optional[str] = None) -> str:
        """Make a payment to a balance or a named external_* payee."""
        try:
            body: Dict[str, Any] = {
                "source": source,
                "destination": _backend_destination(destination),
                "amount": amount,
                "currency": currency,
                "precision": 100,
                "reference": reference or str(uuid.uuid4()),
                "description": description,
                "allow_overdraft": True,
                "skip_queue": True,
            }
            result = await _req("POST", "/transactions", body=body)
            result = _restore_external_destination(result, destination)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def get_transaction(transaction_id: str) -> str:
        """Get transaction details by ID."""
        try:
            result = await _req("GET", f"/transactions/{transaction_id}")
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def list_transactions() -> str:
        """List recent transactions."""
        try:
            result = await _req("GET", "/transactions")
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def refund_transaction(transaction_id: str) -> str:
        """Refund a transaction."""
        try:
            result = await _req("POST", f"/refund-transaction/{transaction_id}",
                                body={"skip_queue": True})
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ----------------------------------------------------------
    # Identities (customers)
    # ----------------------------------------------------------

    @mcp.tool()
    async def create_identity(first_name: str, last_name: str,
                              email: Optional[str] = None,
                              phone: Optional[str] = None) -> str:
        """Create a customer identity."""
        try:
            body: Dict[str, Any] = {
                "first_name": first_name,
                "last_name": last_name,
            }
            if email:
                body["email_address"] = email
            if phone:
                body["phone_number"] = phone
            result = await _req("POST", "/identities", body=body)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def get_identity(identity_id: str) -> str:
        """Get identity details."""
        try:
            result = await _req("GET", f"/identities/{identity_id}")
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})
