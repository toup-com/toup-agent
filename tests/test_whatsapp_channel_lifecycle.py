"""Two sidecars, two SSE listeners, and one fail-open dedupe table (E6).

`restart_whatsapp_channel` has THREE callers and no lock between them —
`channel_init.wake_lazy_channels` (channel_init.py:65),
`tunnel_client`'s config_update (tunnel_client.py:375) and the agent_main
promote path (agent_main.py:1807) — all of which fire on the same
refresh-config storm. On 2026-09-14 two of them interleaved: the first
adapter's `stop()` ran while the second was already constructing, both
`start()`ed, both spawned a sidecar, both attached an SSE listener, and every
inbound WhatsApp message arrived twice. The only reason the user did not get
two replies is the fail-open `whatsapp_dedupe` claim at
whatsapp_baileys.py:442 — i.e. luck with a comment on it.

D8 fixes it structurally: an asyncio.Lock created lazily in the running loop,
coalescing, a config fingerprint, a generation token so a superseded instance
tears itself down, and `start()` returning a bool so the registry only ever
registers a started, current instance.

The dedupe cases below assert BOTH halves: the double is prevented by
SERIALIZATION, and dedupe stays FAIL-OPEN. A "fix" that makes dedupe
fail-closed trades a double reply for a dropped message, which is worse and
would otherwise read as a pass.

Local run (from backend/):
    RUN_MODE=platform PYTHONPATH=. ENVIRONMENT=test JWT_SECRET=x ENCRYPTION_KEY=y \
      pytest tests/test_whatsapp_channel_lifecycle.py -q --tb=short -p no:cacheprovider
"""
from __future__ import annotations

import asyncio
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agent.channels.base import BaseChannel, ChannelType, InboundMessage

_HERE = __import__("pathlib").Path(__file__).resolve().parents[1]
_BAILEYS_SRC = (_HERE / "app" / "agent" / "channels" / "whatsapp_baileys.py").read_text()
_SIDECAR_SRC = (_HERE / "whatsapp_sidecar" / "sidecar.mjs").read_text()


def _d8_landed() -> bool:
    """True once lane B2 has landed D8's lifecycle work.

    One probe for the whole group: the restart lock, the registry gate and the
    generation token ship together, and marking each case on its own private
    guess would leave a case xfailing after its fix arrived.
    """
    import agent_main as am
    from app.agent.channels.registry import ChannelRegistry

    return hasattr(am, "_wa_restart_lock") and hasattr(ChannelRegistry, "replace")


# ── fakes (tests/test_whatsapp_message_handler.py:31-88) ──────────────


class FakeChannel(BaseChannel):
    def __init__(self, channel_type: ChannelType = ChannelType.WHATSAPP):
        super().__init__(channel_type)
        self.sent: List[tuple] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def send_text(self, chat_id: str, text: str, parse_mode: Optional[str] = None) -> None:
        self.sent.append((chat_id, text))

    async def send_typing(self, chat_id: str) -> None:
        pass


def _fake_runner(response_text: str = "ok") -> MagicMock:
    resp = MagicMock()
    resp.text = response_text
    resp.session_id = "sess-1234"
    resp.tokens_total = 0
    resp.tool_calls = []
    resp.processing_time_ms = 1
    resp.persisted = {}
    runner = MagicMock()
    runner.run = AsyncMock(return_value=resp)
    return runner


def _inbound(text="hi", chat_id="+14155552671"):
    return InboundMessage(
        channel=ChannelType.WHATSAPP, channel_user_id=chat_id,
        channel_chat_id=chat_id, text=text, media_paths=[],
    )


@pytest.fixture(autouse=True)
def _clean_registry():
    """`ChannelRegistry._channels` and agent_main's restart state are module
    globals; they leak between tests and would make a serialization test pass
    by inheriting the previous one's instance."""
    from app.agent.channels.registry import ChannelRegistry

    ChannelRegistry._channels.clear()
    import agent_main as am
    try:
        import app.agent.channels.whatsapp_baileys as _wb
        _wb._generation = 0
    except Exception:
        pass
    for name in ("_wa_restart_lock", "_wa_config_fingerprint", "_wa_generation"):
        if hasattr(am, name):
            setattr(am, name, None)
    yield
    ChannelRegistry._channels.clear()


# ══════════════════════════════════════════════════════════════════════
# (a) restart is serialized
# ══════════════════════════════════════════════════════════════════════


class _SpyAdapter(BaseChannel):
    """Records construction, start and stop across every instance."""

    constructed: List["_SpyAdapter"] = []
    started: List["_SpyAdapter"] = []
    stopped: List["_SpyAdapter"] = []

    def __init__(self, allowed_numbers=None, **kw):
        super().__init__(ChannelType.WHATSAPP)
        self.allowed_numbers = allowed_numbers
        type(self).constructed.append(self)

    async def start(self):
        # A real Baileys start spawns a node sidecar and waits on its /health;
        # the window is what the three callers race through.
        await asyncio.sleep(0.05)
        type(self).started.append(self)
        return True

    async def stop(self):
        type(self).stopped.append(self)

    async def send_text(self, chat_id, text, parse_mode=None):
        pass

    async def send_typing(self, chat_id):
        pass

    @classmethod
    def reset(cls):
        cls.constructed, cls.started, cls.stopped = [], [], []


@pytest.fixture
def wa_restart(monkeypatch):
    """agent_main.restart_whatsapp_channel with a spy adapter behind it."""
    import agent_main as am
    import app.agent.channels.whatsapp_baileys as baileys
    from app.config import settings

    _SpyAdapter.reset()
    monkeypatch.setattr(baileys, "BaileysWhatsAppChannel", _SpyAdapter, raising=False)
    monkeypatch.setattr(am, "_agent_runner", _fake_runner(), raising=False)
    monkeypatch.setattr(settings, "whatsapp_mode", "qr_link", raising=False)
    monkeypatch.setattr(settings, "whatsapp_baileys_allowlist", "+14155552671", raising=False)
    return am.restart_whatsapp_channel


@pytest.mark.xfail(not _d8_landed(),
                   reason="RED until lane B2 serializes restart_whatsapp_channel (D8)",
                   strict=True)
@pytest.mark.asyncio
async def test_three_concurrent_restarts_produce_one_live_adapter(wa_restart):
    """The E6 reproduction, with the three real callers' shape. RED on the
    code this file was written against — that is the point.

    MUTATION (kill the guard): delete the lock in agent_main and this goes
    back to three constructions and three sidecars.
    """
    from app.agent.channels.registry import ChannelRegistry

    await asyncio.gather(wa_restart(), wa_restart(), wa_restart())

    started = len(_SpyAdapter.started)
    assert started == 1, f"{started} adapters started concurrently"
    live = ChannelRegistry.get(ChannelType.WHATSAPP)
    assert live is not None and live in _SpyAdapter.started


@pytest.mark.asyncio
async def test_a_restart_with_an_unchanged_fingerprint_is_a_no_op(wa_restart):
    """The refresh-config storm is mostly identical configs. Restarting on
    each one tears down a healthy socket for nothing."""
    import agent_main as am

    if not hasattr(am, "_wa_config_fingerprint"):
        pytest.skip("config fingerprint not landed yet — lane B2 (D8)")

    await wa_restart()
    n_start, n_stop = len(_SpyAdapter.started), len(_SpyAdapter.stopped)
    await wa_restart()
    assert (len(_SpyAdapter.started), len(_SpyAdapter.stopped)) == (n_start, n_stop)


@pytest.mark.asyncio
async def test_force_restarts_regardless_of_the_fingerprint(wa_restart):
    """ANTI-VACUITY for the case above, and the blue-green promote path's
    actual requirement: `force=True` must always cycle."""
    import agent_main as am
    import inspect as _i

    if "force" not in _i.signature(am.restart_whatsapp_channel).parameters:
        pytest.skip("restart_whatsapp_channel(force=) not landed yet — lane B2 (D8)")

    await wa_restart()
    n_start = len(_SpyAdapter.started)
    await am.restart_whatsapp_channel(force=True)
    assert len(_SpyAdapter.started) == n_start + 1


@pytest.mark.asyncio
async def test_a_changed_allowlist_is_hot_applied_and_does_not_restart(
    wa_restart, monkeypatch,
):
    """REVERSED by R46 F1 (2026-09-15), deliberately.

    This case used to assert that an allowlist change RESTARTS the channel.
    That is exactly the incident: the platform seeds the user's own number into
    the allowlist before minting a pairing code, the push reached the agent as
    a config change, and the fingerprint containing `frozenset(_allowlist)`
    SIGTERMed a sidecar that was mid-pairing — 14 s of "Waking Aria and
    creating your code…" and two 503s (toup-agent-pool-82, 14:48:19→14:48:26).

    The allowlist is not a socket parameter: the sidecar never sees it, and
    only the inbound dispatch gate and `health()` read it. It is now applied in
    place. `test_whatsapp_allowlist_hot_apply.py` holds the rest of the claim,
    including the source probe that says WHY an in-place apply is safe.
    """
    from app.config import settings
    import agent_main as am

    if not hasattr(am, "_wa_config_fingerprint"):
        pytest.skip("config fingerprint not landed yet — lane B2 (D8)")

    await wa_restart()
    n_start, n_stop = len(_SpyAdapter.started), len(_SpyAdapter.stopped)
    monkeypatch.setattr(settings, "whatsapp_baileys_allowlist", "+14155559999", raising=False)
    await wa_restart()
    assert (len(_SpyAdapter.started), len(_SpyAdapter.stopped)) == (n_start, n_stop), (
        "an allowlist change restarted the WhatsApp channel again"
    )


@pytest.mark.xfail(not _d8_landed(),
                   reason="RED until lane B2 gates register() on a started instance (D8)",
                   strict=True)
@pytest.mark.asyncio
async def test_the_registry_never_holds_an_adapter_that_failed_to_start(monkeypatch):
    """D8: `start()` returns a bool and the registry only registers a started,
    CURRENT instance. A registered-but-dead adapter answers `/qr/start` with
    a socket that does not exist."""
    import agent_main as am
    import app.agent.channels.whatsapp_baileys as baileys
    from app.agent.channels.registry import ChannelRegistry
    from app.config import settings

    class _Dead(_SpyAdapter):
        async def start(self):
            return False

    _SpyAdapter.reset()
    monkeypatch.setattr(baileys, "BaileysWhatsAppChannel", _Dead, raising=False)
    monkeypatch.setattr(am, "_agent_runner", _fake_runner(), raising=False)
    monkeypatch.setattr(settings, "whatsapp_mode", "qr_link", raising=False)

    await am.restart_whatsapp_channel()

    assert ChannelRegistry.get(ChannelType.WHATSAPP) is None, (
        "an adapter whose start() returned False is registered — /qr/start "
        "then answers over a socket that does not exist"
    )


def test_the_registry_can_replace_and_unregister():
    """D8 adds both, and removes the silent `register()` replace — a replace
    that only WARNS is how a stale adapter keeps answering after a restart."""
    from app.agent.channels.registry import ChannelRegistry

    for name in ("replace", "unregister"):
        if not hasattr(ChannelRegistry, name):
            pytest.skip(f"ChannelRegistry.{name} not landed yet — lane B2 (D8)")

    a, b = FakeChannel(), FakeChannel()
    ChannelRegistry.register(a)
    asyncio.run(ChannelRegistry.replace(b))
    assert ChannelRegistry.get(ChannelType.WHATSAPP) is b
    ChannelRegistry.unregister(ChannelType.WHATSAPP)
    assert ChannelRegistry.get(ChannelType.WHATSAPP) is None


# ══════════════════════════════════════════════════════════════════════
# (b)/(c) inbound dedupe — one dispatch, and fail OPEN
# ══════════════════════════════════════════════════════════════════════


def _real_baileys(monkeypatch, callback):
    """A REAL BaileysWhatsAppChannel, constructed but never started, with its
    message callback recorded. Driving `_handle_inbound_message` directly is
    the whole inbound path minus the sidecar."""
    from app.agent.channels.whatsapp_baileys import BaileysWhatsAppChannel
    from app.config import settings

    monkeypatch.setattr(settings, "user_id", "u-wa-owner", raising=False)
    ch = BaileysWhatsAppChannel(allowed_numbers=["+14155552671"])
    ch.set_message_callback(callback)
    return ch


@pytest.mark.asyncio
async def test_the_same_message_delivered_twice_dispatches_once(monkeypatch):
    """Two SSE listeners deliver the same event id twice. Even once the
    restart is serialized a reconnect can redeliver, so the claim stays —
    driven through the REAL handler, not a model of it."""
    import app.agent.channels.whatsapp_dedupe as dedupe

    got: List[InboundMessage] = []

    async def cb(msg):
        got.append(msg)

    claimed: set = set()

    async def claim(owner_user_id, message_id):
        if message_id in claimed:
            return False
        claimed.add(message_id)
        return True

    monkeypatch.setattr(dedupe, "claim", claim)
    ch = _real_baileys(monkeypatch, cb)

    payload = {"from": "+14155552671", "text": "hi", "message_id": "wamid.ABC",
               "push_name": "N"}
    await ch._handle_inbound_message(dict(payload))
    await ch._handle_inbound_message(dict(payload))

    assert len(got) == 1, f"the same message id dispatched {len(got)} times"


@pytest.mark.asyncio
async def test_a_raising_dedupe_store_still_delivers_the_message(monkeypatch):
    """FAIL-OPEN, asserted so a future 'hardening' cannot quietly make it
    fail-closed. A dropped WhatsApp message is invisible to everyone: there
    is no retry, no error and no row — the double is meant to be prevented by
    SERIALIZATION (the cases above), not by a stricter claim."""
    import app.agent.channels.whatsapp_dedupe as dedupe

    got: List[InboundMessage] = []

    async def cb(msg):
        got.append(msg)

    async def boom(*a, **k):
        raise RuntimeError("dedupe store down")

    monkeypatch.setattr(dedupe, "claim", boom)
    ch = _real_baileys(monkeypatch, cb)

    await ch._handle_inbound_message(
        {"from": "+14155552671", "text": "hi", "message_id": "wamid.DEF", "push_name": "N"}
    )
    assert len(got) == 1, "a dedupe-store failure swallowed the message"


@pytest.mark.asyncio
async def test_a_sender_outside_the_allowlist_is_never_dispatched(monkeypatch):
    """ANTI-VACUITY for both cases above: the handler really is the gate, so
    a green dedupe test is not just 'the callback always fires'."""
    import app.agent.channels.whatsapp_dedupe as dedupe

    got: List[InboundMessage] = []

    async def cb(msg):
        got.append(msg)

    async def claim(*a, **k):
        return True

    monkeypatch.setattr(dedupe, "claim", claim)
    ch = _real_baileys(monkeypatch, cb)

    await ch._handle_inbound_message(
        {"from": "+15559990000", "text": "hi", "message_id": "wamid.GHI", "push_name": "N"}
    )
    assert got == []


# ══════════════════════════════════════════════════════════════════════
# (d) the sidecar's own guards (D8)
# ══════════════════════════════════════════════════════════════════════


def test_the_sidecar_refuses_a_port_it_does_not_own():
    """Two sidecars on one port is the observable half of E6. `server.on
    ('error')` + exit 17 makes the SECOND one die loudly instead of silently
    losing its listen."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "whatsapp_sidecar" / "sidecar.mjs").read_text()
    if "EADDRINUSE" not in src:
        pytest.skip("sidecar listen guard not landed yet — lane B2 (D8)")
    assert "17" in src, "the EADDRINUSE path must exit with the agreed code 17"


def test_the_health_body_identifies_the_process_that_owns_the_port():
    """`spawn_token`/`pid` let the parent tell ITS sidecar from a stranger's.
    A MISSING token is UNKNOWN, never a stranger — an older sidecar image
    does not send one, and killing it on that basis is a cross-version
    outage dressed up as a safety check."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "whatsapp_sidecar" / "sidecar.mjs").read_text()
    if "spawn_token" not in src:
        pytest.skip("sidecar spawn_token not landed yet — lane B2 (D8)")

    parent = (Path(__file__).resolve().parents[1] / "app" / "agent" / "channels"
              / "whatsapp_baileys.py").read_text()
    assert "spawn_token" in parent
    assert "port_owned_by_stranger" in parent or "stranger" in parent


def test_health_started_means_the_sse_task_is_alive():
    """`_sidecar_proc is not None` is not aliveness: the recorded failure is a
    live process whose SSE loop had exited, so `/agent/health` said started
    while nothing was listening.

    The shipped attribute is `_event_task` (the SSE consumer). This used to
    grep for `_sse_task` under a strict xfail whose condition was the same
    grep — a satisfied requirement reported as a known failure, and a pin
    that a revert to `_sidecar_proc is not None` would have sailed past.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "app" / "agent" / "channels"
           / "whatsapp_baileys.py").read_text()
    # The dict KEY inside `health()` — not the comment above it that quotes the
    # word, and not the `channels.whatsapp` block R46 added at module level,
    # which reports `started` by READING health() rather than by deriving it.
    hstart = src.find("    def health(self) -> dict:")
    assert hstart != -1, "health() not found"
    idx = src.find('"started":', hstart)
    assert idx != -1, "health() shape not found"
    window = src[idx:idx + 300]
    assert ("_event_task" in window or "_sse_task" in window) and "done()" in window, (
        "health()['started'] does not read the SSE task's liveness — a live "
        "process whose listener has exited reports itself as started"
    )
    assert "_sidecar_proc is not None" not in window.split("\n")[0], (
        "'started' is read off the subprocess handle again"
    )


@pytest.mark.parametrize("unhealthy", [
    {"started": True, "is_current": True, "sidecar_alive": False},
    {"started": True, "is_current": True, "sidecar_alive": True, "adopted_sidecar": True},
])
@pytest.mark.asyncio
async def test_an_unchanged_config_still_restarts_a_dead_or_adopted_sidecar(monkeypatch, unhealthy):
    """Review R2 P2: `started` only says the SSE task object is alive — the
    consumer loop swallows connection errors forever — so the fingerprint
    no-op treated a dead sidecar as healthy and removed the only automatic
    recovery lever. An adopted (stranger's) sidecar can be neither restarted
    nor killed from inside and must be replaced on the next push too."""
    import agent_main as am
    import app.agent.channels.whatsapp_baileys as baileys
    from app.config import settings

    class _Sick(_SpyAdapter):
        def health(self):
            return dict(unhealthy)

        def is_current(self):
            return True

    _SpyAdapter.reset()
    monkeypatch.setattr(baileys, "BaileysWhatsAppChannel", _Sick, raising=False)
    monkeypatch.setattr(am, "_agent_runner", _fake_runner(), raising=False)
    monkeypatch.setattr(settings, "whatsapp_mode", "qr_link", raising=False)
    monkeypatch.setattr(settings, "whatsapp_baileys_allowlist", "+14155552671", raising=False)

    await am.restart_whatsapp_channel(force=True)
    assert len(_SpyAdapter.started) == 1
    await am.restart_whatsapp_channel()  # same config, unhealthy adapter
    assert len(_SpyAdapter.started) == 2, (
        "an unchanged config took the no-op path over a dead/adopted sidecar"
    )
    assert len(_SpyAdapter.stopped) >= 1


@pytest.mark.asyncio
async def test_an_unchanged_config_and_a_healthy_sidecar_is_still_a_noop(monkeypatch):
    """ANTI-VACUITY for the case above."""
    import agent_main as am
    import app.agent.channels.whatsapp_baileys as baileys
    from app.config import settings

    class _Healthy(_SpyAdapter):
        def health(self):
            return {"started": True, "is_current": True, "sidecar_alive": True,
                    "adopted_sidecar": False}

        def is_current(self):
            return True

    _SpyAdapter.reset()
    monkeypatch.setattr(baileys, "BaileysWhatsAppChannel", _Healthy, raising=False)
    monkeypatch.setattr(am, "_agent_runner", _fake_runner(), raising=False)
    monkeypatch.setattr(settings, "whatsapp_mode", "qr_link", raising=False)
    monkeypatch.setattr(settings, "whatsapp_baileys_allowlist", "+14155552671", raising=False)

    await am.restart_whatsapp_channel(force=True)
    await am.restart_whatsapp_channel()
    assert len(_SpyAdapter.started) == 1


@pytest.mark.asyncio
async def test_a_burst_of_restarts_runs_one_and_queues_one(monkeypatch):
    """Review R2 P3: the coalescing FLAG was cleared by the first runner's
    `finally`, which erased the flag the next waiter had set, so a later
    caller queued a redundant restart. With a waiter COUNT, a burst of N
    calls during one running restart produces exactly one more."""
    import agent_main as am
    import app.agent.channels.whatsapp_baileys as baileys
    from app.config import settings

    class _Slow(_SpyAdapter):
        async def start(self):
            await asyncio.sleep(0.15)
            type(self).started.append(self)
            return True

    _SpyAdapter.reset()
    monkeypatch.setattr(baileys, "BaileysWhatsAppChannel", _Slow, raising=False)
    monkeypatch.setattr(am, "_agent_runner", _fake_runner(), raising=False)
    monkeypatch.setattr(settings, "whatsapp_mode", "qr_link", raising=False)
    monkeypatch.setattr(settings, "whatsapp_baileys_allowlist", "+14155552671", raising=False)

    first = asyncio.create_task(am.restart_whatsapp_channel(force=True))
    await asyncio.sleep(0.02)  # first holds the lock
    burst = [asyncio.create_task(am.restart_whatsapp_channel(force=True)) for _ in range(5)]
    await asyncio.sleep(0.02)
    late = asyncio.create_task(am.restart_whatsapp_channel(force=True))  # arrives while the second waits
    await asyncio.gather(first, *burst, late)

    assert len(_SpyAdapter.started) == 2, (
        f"{len(_SpyAdapter.started)} restarts for a burst — expected the running one plus one queued"
    )
    assert am._wa_restart_waiting == 0


def test_shutdown_stops_the_registry_owned_adapter_when_the_global_is_unset():
    """Review R2 P3: on the deferred start paths (lobby bind, tunnel push,
    promote) the module global `whatsapp_channel` stays None and the
    registry owns the adapter — a shutdown that only read the global left a
    live sidecar behind for the next process to adopt."""
    import agent_main as am
    from app.agent.channels.registry import ChannelRegistry

    spy = _SpyAdapter()
    try:
        ChannelRegistry.register(spy)
        assert am._whatsapp_adapter_to_stop(None) is spy
        other = _SpyAdapter()
        assert am._whatsapp_adapter_to_stop(other) is other, "the boot path's own adapter must win"
    finally:
        try:
            ChannelRegistry.unregister(ChannelType.WHATSAPP)
        except Exception:
            pass
    assert am._whatsapp_adapter_to_stop(None) is None


def test_the_sidecar_exits_17_only_for_a_failed_listen():
    """Review R2 P3 (D8 amendment): exit 17 means "port owned by an
    incumbent" to the parent. Any later server error taking the process
    down would be a false ownership verdict on a healthy dispatcher."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "whatsapp_sidecar" / "sidecar.mjs").read_text()
    block = src[src.index("server.on('error'"):]
    block = block[:block.index("\n});") + 4]
    assert "e.syscall === 'listen'" in block and "server.listening" in block, (
        "the error handler no longer distinguishes a failed listen from a later server error"
    )
    assert block.count("process.exit(17)") == 1
    assert "server.error" in block, "a post-listen error must be logged, not swallowed"
