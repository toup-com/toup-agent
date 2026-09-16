"""R46 F2 — the pairing route waits for a restarting adapter, it does not 503.

Incident 2026-09-15: the registry was empty for 2.6 s while the adapter was
replaced (14:48:20.99 → 14:48:24.86) and `_require_active_channel()` answered
503 within milliseconds of looking, twice. The user's client papered over it
with a blind fixed-interval retry loop, which is the client guessing at a fact
the server already holds.

F1 removes the restart that caused THAT hole. This is the bounded safety net
for the restarts that remain real (mode change, sick sidecar, boot), and the
shape matters: an EVENT published by the restart path, not a poll with a magic
interval, and a typed 503 carrying `retry_after_s` when the budget runs out.

Local run (from backend/):
    RUN_MODE=platform PYTHONPATH=. ENVIRONMENT=test \\
      pytest tests/test_whatsapp_qr_wait_for_adapter.py -q -p no:cacheprovider
"""
from __future__ import annotations

import ast
import asyncio
import pathlib
import time

import pytest
from fastapi import HTTPException

_HERE = pathlib.Path(__file__).resolve().parents[1]
_QR_SRC = (_HERE / "app" / "api" / "whatsapp_qr.py").read_text()
_BAILEYS_SRC = (_HERE / "app" / "agent" / "channels" / "whatsapp_baileys.py").read_text()
_AGENT_MAIN_SRC = (_HERE / "agent_main.py").read_text()


@pytest.fixture
def wb():
    """whatsapp_baileys with its readiness state reset around each case."""
    from app.agent.channels import whatsapp_baileys as mod

    saved = (
        mod._active_channel, mod._adapter_ready, mod._restart_in_flight,
        mod._ready_event, mod._ready_event_loop,
        mod._ever_linked, mod._restart_started_at,
    )
    mod._active_channel = None
    mod._adapter_ready = False
    mod._restart_in_flight = False
    mod._ready_event = None
    mod._ready_event_loop = None
    mod._ever_linked = False
    mod._restart_started_at = 0.0
    try:
        yield mod
    finally:
        (
            mod._active_channel, mod._adapter_ready, mod._restart_in_flight,
            mod._ready_event, mod._ready_event_loop,
            mod._ever_linked, mod._restart_started_at,
        ) = saved


class _Fake:
    pass


# ── the wait ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_ready_adapter_returns_immediately(wb):
    fake = _Fake()
    wb._active_channel = fake
    wb.mark_adapter_ready()

    t0 = time.monotonic()
    got = await wb.await_active_baileys_channel(5.0)

    assert got is fake
    assert time.monotonic() - t0 < 0.05, "a ready adapter was made to wait"


@pytest.mark.asyncio
async def test_a_late_registration_wakes_the_waiter_at_once(wb):
    """The point of an event: the waiter resumes when the restart path SAYS
    so, not on the next tick of somebody's interval."""
    wb.mark_adapter_restarting()
    fake = _Fake()

    async def register_later():
        await asyncio.sleep(0.05)
        wb._active_channel = fake
        wb.mark_adapter_ready()

    task = asyncio.create_task(register_later())
    t0 = time.monotonic()
    got = await wb.await_active_baileys_channel(5.0)
    elapsed = time.monotonic() - t0
    await task

    assert got is fake
    assert elapsed < 0.4, (
        f"waited {elapsed:.3f}s for a registration that landed at 0.05s — "
        "this reads like a polling interval, not an event"
    )


@pytest.mark.asyncio
async def test_the_budget_is_honoured_when_nothing_registers(wb):
    wb.mark_adapter_restarting()
    t0 = time.monotonic()
    got = await wb.await_active_baileys_channel(0.2)
    elapsed = time.monotonic() - t0

    assert got is None
    assert 0.15 <= elapsed < 1.0, f"budget not honoured (elapsed {elapsed:.3f}s)"


@pytest.mark.asyncio
async def test_a_failed_start_does_not_wake_waiters_into_an_empty_registry(wb):
    """`mark_adapter_start_failed` ends the restart WINDOW without claiming an
    adapter. A waiter must answer None — never adopt an empty registry — and
    it must answer it AT ONCE (R46 decision D1): the failure wakes the event,
    the waiter re-reads the registry, sees no restart in flight, and returns.
    Sitting out the rest of the budget only delays a 503 that is already
    decided, inside a request the user is watching."""
    wb.mark_adapter_restarting()

    async def fail_later():
        await asyncio.sleep(0.05)
        wb.mark_adapter_start_failed()

    task = asyncio.create_task(fail_later())
    t0 = time.monotonic()
    got = await wb.await_active_baileys_channel(5.0)
    elapsed = time.monotonic() - t0
    await task

    assert got is None
    assert wb.adapter_restart_in_flight() is False
    assert elapsed < 1.0, (
        f"waited {elapsed:.2f}s for an answer that was settled at 0.05s — a "
        "give-up must wake the waiters it stranded"
    )


# ── the route ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_route_503s_with_a_retry_hint_after_the_budget(wb, monkeypatch):
    from app.api import whatsapp_qr

    wb.mark_adapter_restarting()
    monkeypatch.setattr(whatsapp_qr, "_ADAPTER_WAIT_BUDGET_S", 0.1, raising=False)

    with pytest.raises(HTTPException) as exc:
        await whatsapp_qr._await_active_channel(0.1)

    assert exc.value.status_code == 503
    detail = exc.value.detail
    assert isinstance(detail, dict), "the 503 body must be typed, not prose"
    assert detail.get("retry_after_s"), "no retry_after_s for the client to honour"
    assert (exc.value.headers or {}).get("Retry-After")


@pytest.mark.asyncio
async def test_the_route_returns_the_adapter_when_one_is_ready(wb):
    from app.api import whatsapp_qr

    fake = _Fake()
    wb._active_channel = fake
    wb.mark_adapter_ready()

    assert await whatsapp_qr._await_active_channel(1.0) is fake


def test_the_budget_sits_below_the_platform_proxy_timeout():
    """WAIT + WORK, not the wait alone.

    This route waits here and only THEN calls the sidecar, whose mint is
    bounded by the agent's own `_SIDECAR_HTTP_TIMEOUT_S`. 6 + 10 = 16 > the
    10 s `_agent_qr_proxy` default, so the two platform call sites that can
    now wait must raise their own ceiling — otherwise the platform 504s on a
    request the agent is still holding and throws away a code the sidecar
    already minted.
    """
    from app.api import whatsapp_qr
    from app.agent.channels.whatsapp_baileys import _SIDECAR_HTTP_TIMEOUT_S

    assert whatsapp_qr._ADAPTER_WAIT_BUDGET_S <= 6.0

    src = (_HERE / "app" / "api" / "agent_setup.py").read_text()
    worst_case = whatsapp_qr._ADAPTER_WAIT_BUDGET_S + _SIDECAR_HTTP_TIMEOUT_S
    for fn, path in (
        ("whatsapp_pair_code", "/api/whatsapp/qr/pair-code"),
        ("whatsapp_qr_start", "/api/whatsapp/qr/start"),
    ):
        body = _fn_source(src, fn)
        assert path in body, f"{fn} no longer proxies {path}"
        m = __import__("re").search(r"timeout_s=([0-9.]+)", body)
        assert m, f"{fn} still relies on _agent_qr_proxy's 10 s default"
        assert float(m.group(1)) >= worst_case, (
            f"{fn} gives the agent {m.group(1)}s but the agent may spend "
            f"{worst_case}s (wait {whatsapp_qr._ADAPTER_WAIT_BUDGET_S} + "
            f"sidecar {_SIDECAR_HTTP_TIMEOUT_S})"
        )


# ── shape probes ──────────────────────────────────────────────────────


def _fn_source(src: str, name: str) -> str:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(f"{name} not found")


def test_the_wait_is_an_event_not_a_sleep_loop():
    """A `while ...: await asyncio.sleep(0.1)` would pass every behavioural
    case above and still be the thing the critic rejected."""
    body = _fn_source(_BAILEYS_SRC, "await_active_baileys_channel")
    assert "asyncio.sleep" not in body, (
        "await_active_baileys_channel polls with sleep — it must await an event"
    )
    assert "wait_for" in body and ".wait()" in body


def test_the_restart_path_publishes_before_it_stops_the_adapter():
    """ORDER, not presence. Marking the restart AFTER `existing.stop()` leaves
    a window in which a waiter sees the stale ready flag, takes the fast path,
    and gets the adapter that is being torn down."""
    body = _fn_source(_AGENT_MAIN_SRC, "_restart_whatsapp_locked")
    mark = body.find("_mark_wa_restarting()")
    stop = body.find("await existing.stop()")
    assert mark != -1, "the restart path never publishes that it is restarting"
    assert stop != -1
    assert mark < stop, "_mark_wa_restarting() must precede existing.stop()"
    # …and UNCONDITIONALLY. Nested inside `if existing is not None:` it would
    # still precede the stop while skipping the case that matters most: a COLD
    # start, where there is no adapter to replace and `channels_settled` would
    # keep saying "settled" all the way through the sidecar boot.
    marker_lines = [
        ln for ln in body.splitlines() if ln.strip() == "_mark_wa_restarting()"
    ]
    assert marker_lines, "the restart marker moved out of a statement of its own"
    assert any(len(ln) - len(ln.lstrip()) == 4 for ln in marker_lines), (
        "every _mark_wa_restarting() is nested in a branch — a restart that "
        "starts from an empty registry then never opens the window a "
        "pair-code request needs to wait on"
    )


def test_every_restart_exit_settles_the_window():
    """A restart that ends without an adapter must still END — otherwise every
    later pair-code request burns the full budget and `channels_settled` says
    'starting' forever."""
    body = _fn_source(_AGENT_MAIN_SRC, "_restart_whatsapp_locked")
    assert "_mark_wa_settled_if_pending()" in body
    assert "finally:" in body, "the settle is not on an unconditional exit"


def test_the_status_poll_does_not_wait():
    """`/qr/status` is polled by every client on every foreground. Making IT
    wait would turn a restart into a pile of held connections."""
    body = _fn_source(_QR_SRC, "qr_status")
    assert "_require_active_channel()" in body
    assert "_await_active_channel" not in body


# ── The ROUTES wait, not just the helper (fix lane A) ──────────────
#
# Every case above drives `_await_active_channel` / `await_active_baileys_channel`
# in isolation. Reverting `qr_pair_code` and `qr_start` to the pre-fix
# `_require_active_channel()` — literally the incident behaviour — passed all
# 181 WhatsApp tests. These are the positive twins of
# `test_the_status_poll_does_not_wait`, over the real handlers.


def _qr_handlers():
    from fastapi import FastAPI
    from app.api.whatsapp_qr import router

    app = FastAPI()
    app.include_router(router, prefix="/api")
    return {
        getattr(r, "path"): getattr(r, "endpoint")
        for r in app.routes
        if getattr(r, "path", None) and getattr(r, "endpoint", None)
    }


class _LateChannel:
    """Registered after the route has already started waiting."""

    def __init__(self):
        self.kicked = False
        self.phone = None

    async def kick_pair(self):
        self.kicked = True

    async def request_pairing_code(self, phone, idempotency_key=None):
        self.phone = phone
        return "ABCD1234"


async def _register_after(wb, delay_s: float, channel):
    await asyncio.sleep(delay_s)
    wb._active_channel = channel
    wb.mark_adapter_ready()


@pytest.mark.asyncio
async def test_the_pair_code_route_waits_for_a_restarting_adapter(wb):
    from app.api.whatsapp_qr import PairCodeRequest

    wb.mark_adapter_restarting()
    ch = _LateChannel()
    asyncio.get_event_loop().create_task(_register_after(wb, 0.05, ch))

    handler = _qr_handlers()["/api/whatsapp/qr/pair-code"]
    out = await handler(
        PairCodeRequest(phone="+14155552671"), _user=object(), idempotency_key=None,
    )
    assert out["pairing_code"] == "ABCD1234"
    assert ch.phone == "+14155552671"


@pytest.mark.asyncio
async def test_the_qr_start_route_waits_for_a_restarting_adapter(wb):
    wb.mark_adapter_restarting()
    ch = _LateChannel()
    asyncio.get_event_loop().create_task(_register_after(wb, 0.05, ch))

    handler = _qr_handlers()["/api/whatsapp/qr/start"]
    out = await handler(_user=object())
    assert out == {"ok": True}
    assert ch.kicked is True


def test_both_pairing_routes_await_rather_than_require():
    """The cheap structural twin of the two cases above: a revert to
    `_require_active_channel()` in either handler fails here even if the
    behavioural cases were ever stubbed out."""
    for name in ("qr_pair_code", "qr_start"):
        body = _fn_source(_QR_SRC, name)
        assert "_await_active_channel" in body, f"{name} stopped awaiting"
        assert "_require_active_channel()" not in body, (
            f"{name} answers 503 on an empty registry again"
        )


# ── `channels_settled` is a WHATSAPP fact, not a chat-readiness gate ──
#
# `serving` (agent_main) ANDs `channels_settled` in, and
# `agent_setup.agent_turn_readiness` refuses on it — so a WhatsApp-only
# transition became "this container cannot serve a turn". `/admin/bind` wakes
# the lazy channels on EVERY claim and `_wa_mode` defaults to `qr_link`
# whenever no cloud creds exist, so every new signup entered that window.


def test_a_never_linked_tenant_is_settled_through_a_restart(wb):
    wb.mark_adapter_restarting()
    assert wb.adapter_restart_in_flight() is True, "the waiter's flag must stand"
    assert wb.channels_settled() is True, (
        "a tenant who has never linked WhatsApp cannot be waiting on a "
        "WhatsApp restart — onboarding's readiness poll must not block on it"
    )
    assert wb.whatsapp_health_fields()["settled"] is True


def test_a_linked_tenant_is_not_settled_during_a_restart(wb):
    wb.mark_session_linked()
    wb.mark_adapter_restarting()
    assert wb.channels_settled() is False
    assert wb.whatsapp_health_fields()["settled"] is False
    wb.mark_adapter_ready()
    assert wb.channels_settled() is True


def test_a_wedged_restart_stops_counting(wb):
    """A `_teardown()` that hangs on a task ignoring cancellation may not pin
    `serving=False` for the life of the process."""
    wb.mark_session_linked()
    wb.mark_adapter_restarting()
    assert wb.channels_settled() is False
    wb._restart_started_at = (
        time.monotonic() - wb._SIDECAR_BOOT_TIMEOUT_S - wb._SETTLE_GRACE_S - 1
    )
    assert wb.channels_settled() is True


def test_the_link_write_path_publishes_ever_linked():
    """`channels_settled` reads a module flag because it is asked while the
    adapter is unregistered; only `_apply_sidecar_state` may set it."""
    body = _fn_source(_BAILEYS_SRC, "_apply_sidecar_state")
    assert "mark_session_linked()" in body
