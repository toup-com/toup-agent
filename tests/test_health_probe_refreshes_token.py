"""Regression test for the 2026-05-11 overnight-Gmail bug.

The health-probe scheduler used to call `provider.health_probe(ctx)`
with whatever access token happened to be in the vault. Google
access tokens live 1 h; the dispatcher refreshes-on-expiring before
every tool call. But the probe path skipped that step entirely, so
any identity whose user hadn't touched the agent in >1 h ate a 401
on every sweep — three sweeps later (~15 min) the identity flipped
to `provider_down` and the agent lost the connector's tools until
the user manually clicked "Retry".

This test pins the contract: before invoking `provider.health_probe`,
`_probe_one` MUST run the same `needs_refresh` / `refresh_with_coalescing`
pair the dispatcher uses, and MUST short-circuit (no probe, no fail
count bump) when the refresh helper signals `ConnectorReauthRequired`
— otherwise we'd race ourselves toward provider_down on top of the
correct reauth_required state.

Source-grep style, same pattern as test_agent_runner_tool_events.py
and test_time_channel_fix.py — runs without booting the runtime."""
from __future__ import annotations

from pathlib import Path


_SRC = (
    Path(__file__).resolve().parent.parent
    / "app"
    / "services"
    / "connector_health_probe.py"
).read_text()


def test_imports_dispatcher_refresh_helpers():
    """The probe module must pull the SAME helpers the dispatcher uses
    (not its own duplicated refresh logic) so the per-identity lock
    is shared — otherwise a concurrent tool call and probe could
    both call provider.refresh() at the same instant."""
    assert "from app.services.connector_dispatcher import" in _SRC
    assert "needs_refresh" in _SRC, (
        "connector_health_probe.py must import `needs_refresh` from "
        "connector_dispatcher; otherwise the probe runs against the "
        "stale access_token and 401s after ~1 h."
    )
    assert "refresh_with_coalescing" in _SRC, (
        "connector_health_probe.py must import `refresh_with_coalescing` "
        "from connector_dispatcher; without it the probe can't refresh "
        "tokens, every overnight gap flips identities to provider_down."
    )


def _probe_one_body() -> str:
    """Slice out the body of `async def _probe_one(...)` so the test
    doesn't get false matches from the module docstring or imports."""
    start = _SRC.index("async def _probe_one(")
    # Body runs until the next top-level def at the same indent level.
    end = _SRC.index("\n    async def ", start + 1)
    return _SRC[start:end]


def test_probe_one_calls_refresh_before_provider():
    """`needs_refresh(...)` must appear in the source BEFORE the
    `provider.health_probe(` call inside `_probe_one`. Order
    matters: a probe against an expired token is what created the
    original bug."""
    body = _probe_one_body()
    probe_idx = body.index("provider.health_probe(")
    refresh_idx = body.index("needs_refresh(")
    assert refresh_idx < probe_idx, (
        "needs_refresh(...) must run BEFORE provider.health_probe(...) "
        "inside _probe_one. The original bug was the probe firing "
        "against a stale access_token and 401-ing on every sweep."
    )


def test_probe_one_handles_reauth_required_outcome():
    """When `refresh_with_coalescing` returns `ConnectorReauthRequired`,
    `_probe_one` must short-circuit — the helper already marked the
    identity reauth_required, counting it as a probe failure too
    would push it toward provider_down on the next sweep (wrong
    terminal state)."""
    assert "ConnectorReauthRequired" in _SRC, (
        "_probe_one needs to detect the ConnectorReauthRequired branch "
        "returned by refresh_with_coalescing — without that check, a "
        "failed refresh still counts as a probe-failure and we race "
        "toward provider_down on top of reauth_required."
    )
    # Find the chunk between `refresh_with_coalescing(` and the next
    # `provider.health_probe(` — that's the refresh-outcome branch
    # in `_probe_one`. It must NOT call `_record_fail` (which is what
    # bumps the consecutive-failure counter).
    body = _probe_one_body()
    refresh_call_idx = body.index("refresh_with_coalescing(")
    probe_call_idx = body.index("provider.health_probe(", refresh_call_idx)
    branch = body[refresh_call_idx:probe_call_idx]
    assert "_record_fail" not in branch, (
        "The refresh-failure branch of _probe_one must NOT call "
        "_record_fail. The identity is already on the correct "
        "terminal state (reauth_required); bumping the failure "
        "counter would race it toward provider_down."
    )


def test_run_passes_entry_not_provider():
    """`run_once` used to pass `entry.provider` to `_probe_one`. To
    refresh, `_probe_one` needs the full registry entry (so
    `refresh_with_coalescing` can call `entry.provider.refresh()`).
    Pin the wire format so a future cleanup doesn't accidentally
    drop back to `entry.provider`."""
    assert "self._probe_one(connector_id, ident, entry, sem)" in _SRC, (
        "run_once must pass `entry` (registry entry) — not "
        "`entry.provider` — into _probe_one so the refresh helper "
        "can call provider.refresh() with the right context."
    )
