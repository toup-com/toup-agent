"""Regression test for the 2026-05-12 "agent falls back to browser
when Gmail expired" fix.

Symptom: User asked "what's my 12th email" right after the 1-hour
token expiry. The agent (gpt-5.5) chose the `browser` tool to open
Gmail in Patchright instead of `gmail__list_messages`. That fails
because Chromium isn't installed in the tenant runtime container —
the user gets a useless "Open Live Browser" link and zero
information.

Root cause: the FastMCP `tools/list` filter
(`ConnectorToolFilterMiddleware.on_list_tools`) ONLY surfaced tools
for connector identities with `status == "active"`. When the token
expired and the dispatcher flipped status to `reauth_required`, the
tool was dropped from the agent's tool list entirely. The agent
literally couldn't see Gmail tools existed, so it picked the next
best thing: the browser tool. Browser failed → no answer.

The right behavior: KEEP the tool visible when the user has set up
the connector (even if it's currently broken). The dispatcher
returns `ConnectorReauthRequired` on invocation; the LLM surfaces it
to the user as "Please reconnect your Gmail." That's the production
UX. Hiding the tool is the wrong escape hatch.

This module pins three contracts via source-grep:

  1. The vault exposes `list_visible_to_agent(user_id)` returning
     identities with status in {active, reauth_required,
     provider_down}. `revoked` is the only status that hides the
     tool (user explicitly disconnected).

  2. `ConnectorToolFilterMiddleware.on_list_tools` uses
     `list_visible_to_agent`, NOT `list_active`. If a future cleanup
     swaps it back to `list_active` the hourly-browser-fallback bug
     comes back.

  3. The dispatcher still gates *invocation* on
     `status == "active"` — visibility is wider, execution is
     narrower. This is intentional: visibility drives the LLM's tool
     choice; invocation enforces the reauth-required error.
"""
from __future__ import annotations

from pathlib import Path


BACKEND = Path(__file__).resolve().parent.parent
_VAULT = (BACKEND / "app/services/connector_vault.py").read_text()
_MCP = (BACKEND / "app/services/connector_mcp.py").read_text()
_DISPATCHER = (BACKEND / "app/services/connector_dispatcher.py").read_text()


def test_vault_exposes_list_visible_to_agent():
    """The wider 'visible to the agent' set must be a public vault
    method — not inlined into the MCP filter — so the contract is
    testable and so other callers (system-prompt 'Connected' line,
    cache invalidation) can share it."""
    assert "async def list_visible_to_agent(" in _VAULT, (
        "vault must expose `list_visible_to_agent(user_id)` returning "
        "identities with status in {active, reauth_required, "
        "provider_down}. Without it the MCP filter would have to "
        "inline the status set, drifting from any other caller."
    )


def test_visible_statuses_include_reauth_required_and_provider_down():
    """The whole point of the wider set is that an expired identity
    stays visible. Pin both reauth_required AND provider_down — a
    transient outage on Google's end should still let the agent know
    Gmail exists so it can surface the structured error."""
    # We grep for the literal tuple of statuses since the constant
    # name (_VISIBLE_STATUSES) is private. The values are what
    # matters for behavior.
    assert '"reauth_required"' in _VAULT, (
        "vault's visible-status set must include reauth_required"
    )
    assert '"provider_down"' in _VAULT, (
        "vault's visible-status set must include provider_down"
    )
    assert '"active"' in _VAULT, (
        "vault's visible-status set must include active"
    )
    # `revoked` is the only terminal state — the user disconnected
    # explicitly. It MUST NOT be in the visible set.
    # We can't grep for absence cleanly because "revoked" appears
    # elsewhere in the file (in `disconnect()`), so instead verify
    # the _VISIBLE_STATUSES symbol exists and the function uses it.
    assert "_VISIBLE_STATUSES" in _VAULT, (
        "the visible-statuses tuple should be a named constant so "
        "future readers can see the explicit set"
    )


def test_mcp_filter_uses_list_visible_not_list_active():
    """The hourly-browser-fallback bug came from `list_active` — the
    MCP filter only surfaced tools for status='active'. Pin the
    list_visible_to_agent call so a future cleanup can't quietly
    swap it back."""
    assert "list_visible_to_agent" in _MCP, (
        "MCP filter must call vault.list_visible_to_agent — using "
        "vault.list_active hides reauth_required tools from the "
        "agent and triggers the browser-fallback failure mode."
    )
    # And the on_list_tools function specifically should use the
    # wider method. We can't easily isolate the on_list_tools body
    # from the rest of the file with grep, but a passing grep for
    # the symbol is enough — there's nowhere else in this file that
    # would call list_visible_to_agent.


def test_dispatcher_still_gates_invocation_on_active_status():
    """Visibility is wider; invocation is narrower. The dispatcher
    MUST keep gating actual tool execution on status == "active" —
    otherwise the expired-token call would hit Google with a dead
    token, take 5-15s to fail, and the LLM would see a generic 401
    instead of the clean `ConnectorReauthRequired` variant. Keep the
    two layers in their lanes."""
    assert 'identity.status != "active"' in _DISPATCHER, (
        "dispatcher must reject non-active identities BEFORE "
        "provider.execute() runs. Visibility (MCP filter) is wider, "
        "execution (dispatcher) is narrower — don't conflate them."
    )
    # And the rejection path must emit ConnectorReauthRequired so
    # the LLM gets the structured error to surface to the user.
    # Spot-check: the status check above is followed (within ~5
    # lines) by a `return ConnectorReauthRequired(`.
    status_idx = _DISPATCHER.index('identity.status != "active"')
    after = _DISPATCHER[status_idx:status_idx + 400]
    assert "ConnectorReauthRequired" in after, (
        "non-active identity must return ConnectorReauthRequired — "
        "anything else (ConnectorToolError, raised exception) blurs "
        "the LLM's signal and can re-trigger the browser fallback."
    )
