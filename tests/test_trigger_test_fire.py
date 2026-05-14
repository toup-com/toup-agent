"""Test-fire UX pins — 2026-05-14 bug fix.

The user reported that clicking "Test" on the Gmail trigger dashboard
left the trigger looking broken: the event row showed `Failed` and the
trigger card showed both `Active` AND `Provisioning…` simultaneously.

Root cause: `EmailReceivedHandler._fetch_all` was calling MCP with the
test event's synthetic dedupe id (`test:<hex>`), getting a 404, and
returning a per-event `failed` status. The batch result was actually
`success` (no kept emails) which set `last_status="active"`, but the
visible event row was Failed. The "Check the chat for the resulting
message" banner was a lie — nothing landed in the chat.

This module pins the new behavior:

  1. `_synthetic_test_email` produces a complete `_FetchedEmail`
     with the marker headers/body the rest of the pipeline expects.
  2. `_fetch_all` detects `test:` prefix events and substitutes the
     synthetic email INSTEAD of calling MCP. No 404, no per-event
     failure.
  3. Source-grep guard: the `test:` short-circuit must stay in
     `_fetch_all`. Drop it and we re-introduce the misleading
     Failed-but-Active state.
"""
from __future__ import annotations

from pathlib import Path

import pytest


BACKEND = Path(__file__).resolve().parent.parent
_HANDLER_SRC = (BACKEND / "app/agent/triggers/email_received_handler.py").read_text()


# ── Source-grep guards ──────────────────────────────────────────────


def test_handler_has_synthetic_test_email_helper():
    """The synthetic email builder is the single source of truth for
    test-fire payloads. Removing it forces a refactor that would have
    to re-derive the headers/body shape from scratch."""
    assert "def _synthetic_test_email" in _HANDLER_SRC, (
        "`_synthetic_test_email` builder missing — test fires would "
        "fall through to the MCP fetch and 404, re-introducing the "
        "Failed-but-Active dashboard state from 2026-05-14."
    )


def test_fetch_all_short_circuits_test_events():
    """`_fetch_all` must detect the `test:` prefix BEFORE the MCP
    call. The order matters — once mcp.call_tool runs on a synthetic
    id we get a 404 and the test event lands as `failed`."""
    src = _HANDLER_SRC
    short_circuit_idx = src.find('gmail_id.startswith("test:")')
    mcp_call_idx = src.find('mcp.call_tool(')
    assert short_circuit_idx != -1, (
        "Missing `gmail_id.startswith(\"test:\")` short-circuit in "
        "_fetch_all. Test fires would re-route to MCP and fail."
    )
    assert short_circuit_idx < mcp_call_idx, (
        "Test-fire short-circuit must come BEFORE the MCP call in "
        "_fetch_all. Currently it's after — every test event would "
        "still hit Gmail and 404."
    )


# ── Behavior of the synthetic builder ───────────────────────────────


def test_synthetic_email_returns_complete_payload():
    """The synthetic email must have ALL the fields downstream code
    reads (subject, from, body, snippet) — incomplete payloads would
    crash the summariser with KeyError or render an empty chat
    message."""
    from app.agent.triggers.email_received_handler import (
        _synthetic_test_email,
    )
    fe = _synthetic_test_email("ev-1", "test:abc123")
    assert fe.event_id == "ev-1"
    assert fe.gmail_id == "test:abc123"
    assert "subject" in fe.headers and fe.headers["subject"]
    assert "from" in fe.headers and "triggers@toup.ai" in fe.headers["from"]
    assert fe.body and len(fe.body) > 100  # non-trivial body
    assert fe.snippet  # for the notify_only action's snippet field
    assert "INBOX" in fe.labels  # filters that check labelIds
    # raw_message must include the Gmail-shape `payload.headers` array
    # because filter_evaluator._flatten_headers reads from there.
    payload = fe.raw_message.get("payload") or {}
    headers_list = payload.get("headers") or []
    header_names = {h.get("name") for h in headers_list}
    assert "Subject" in header_names
    assert "From" in header_names


def test_synthetic_email_explicitly_marked_as_test():
    """The synthesised message must visibly identify itself as a test
    in the chat output, so the user doesn't mistake it for a real
    email summary."""
    from app.agent.triggers.email_received_handler import (
        _synthetic_test_email,
    )
    fe = _synthetic_test_email("ev-1", "test:abc123")
    assert "test" in fe.headers["subject"].lower() or "[Test]" in fe.headers["subject"]
    assert "synthetic" in fe.body.lower() or "test" in fe.body.lower()


# ── Test endpoint docstring ─────────────────────────────────────────


def test_test_trigger_endpoint_docstring_reflects_new_behavior():
    """The endpoint docstring is operator-facing reference. It must
    not still say "the handler will try to fetch … and fail" — that
    was the old behavior."""
    api_src = (BACKEND / "app/api/triggers.py").read_text()
    idx = api_src.index("async def test_trigger(")
    body = api_src[idx:idx + 2000]
    assert "synthetic email envelope" in body, (
        "test_trigger docstring missing reference to the synthetic "
        "envelope substitution — describes stale 'will fetch and fail' "
        "behavior."
    )
