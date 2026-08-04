"""Behavioural tests for app.agent.channels.whatsapp_baileys.

WHY THIS FILE WAS REWRITTEN (2026-08-04)
----------------------------------------
Commit 056eaf25 ("feat(whatsapp-qr): replace neonize with Baileys Node
sidecar") deleted the constructor-time disk probe this file used to pin
(``self._session_status = "linked" if session_file_exists() else ...``)
along with the whole neonize SQLite store. Session truth now lives in the
Node sidecar and reaches Python over HTTP (``GET /health`` during
``start()``, ``GET /pair/status`` on every poll) and SSE. The old
``test_session_status_reflects_disk_state`` therefore asserted on a
mechanism that no longer exists, and the old ``TestAcl`` asserted set
membership on a helper module (``whatsapp_baileys_session``) that has
**zero production importers** after 056eaf25 — it never touched the real
ACL gate at all.

Worse, ``test_no_neonize_no_crash`` passed only by ACCIDENT: it relied on
``backend/whatsapp_sidecar/node_modules`` being absent from a fresh
checkout. On any machine where someone has run ``npm install`` (and
``node`` is on PATH, which it is on every dev Mac) that test spawned a
REAL Node process that outlived the test run. It is replaced by
``TestStartDegradesGracefully``, which drives each of the three bail-out
guards with a spawn spy that fails the test if it is ever called.

Everything here asserts on OBSERVABLE BEHAVIOUR — return values, the
``health()`` dict, recorded HTTP calls, dispatched ``InboundMessage``
objects. There is no source-text matching, and every assertion has been
mutation-checked against backend/app/agent/channels/whatsapp_baileys.py.

Lane: platform. Nothing here touches an AGENT_ONLY table — the dedupe
claim is stubbed rather than run against ``whatsapp_inbound_dedupe``,
which does not exist under RUN_MODE=platform sqlite.

    cd backend && RUN_MODE=platform PYTHONPATH=. pytest tests/test_whatsapp_baileys.py
"""

from __future__ import annotations

import asyncio
import signal
from typing import Any, Optional

import pytest


# ── Test doubles ─────────────────────────────────────────────────
#
# The adapter reaches the outside world through exactly three seams:
# `httpx` (module attribute on whatsapp_baileys), `asyncio.
# create_subprocess_exec`, and `shutil.which`. All three are replaced
# on the MODULE OBJECT, never globally, so nothing leaks into another
# test file.


class StubResponse:
    def __init__(self, status_code: int = 200, body: Any = None, text: str = ""):
        self.status_code = status_code
        self._body = {} if body is None else body
        self.text = text

    def json(self) -> Any:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class StubAsyncClient:
    """Scripted async HTTP client that RECORDS every request.

    ``routes`` maps ``"GET /health"`` → :class:`StubResponse`. Anything
    unrouted returns ``200 {"ok": true}`` so a test only has to script
    the calls it actually cares about.
    """

    def __init__(self, routes: Optional[dict[str, StubResponse]] = None):
        self.routes = routes or {}
        self.gets: list[str] = []
        self.posts: list[tuple[str, Any]] = []
        self.closed = False

    @property
    def post_paths(self) -> list[str]:
        return [p for p, _ in self.posts]

    def _resolve(self, method: str, path: str) -> StubResponse:
        return self.routes.get(f"{method} {path}", StubResponse(200, {"ok": True}))

    async def get(self, path: str, **_kw: Any) -> StubResponse:
        self.gets.append(path)
        return self._resolve("GET", path)

    async def post(self, path: str, json: Any = None, **_kw: Any) -> StubResponse:
        self.posts.append((path, json))
        return self._resolve("POST", path)

    async def aclose(self) -> None:
        self.closed = True


class StubSyncClient:
    """Blocking client used by ``get_pairing_status()``."""

    def __init__(self, routes: Optional[dict[str, StubResponse]] = None):
        self.routes = routes or {}
        self.gets: list[str] = []

    def __enter__(self) -> "StubSyncClient":
        return self

    def __exit__(self, *_exc: Any) -> None:
        return None

    def get(self, path: str, **_kw: Any) -> StubResponse:
        self.gets.append(path)
        return self.routes.get(f"GET {path}", StubResponse(200, {"ok": True}))


class FakeHttpxModule:
    """Stands in for the `httpx` module attribute on whatsapp_baileys.

    Default behaviour is "sidecar unreachable": constructing a client
    raises. That is the deliberate safety net — without it,
    ``get_pairing_status()`` on a developer machine that happens to be
    running a real sidecar on 127.0.0.1:8002 would silently talk to it.
    """

    def __init__(self) -> None:
        self.async_client: Optional[StubAsyncClient] = None
        self.sync_client: Optional[StubSyncClient] = None
        self.async_ctor_kwargs: list[dict] = []

    def AsyncClient(self, **kwargs: Any) -> StubAsyncClient:  # noqa: N802
        self.async_ctor_kwargs.append(kwargs)
        if self.async_client is None:
            raise ConnectionError("sidecar unreachable (test default)")
        return self.async_client

    def Client(self, **_kwargs: Any) -> StubSyncClient:  # noqa: N802
        if self.sync_client is None:
            raise ConnectionError("sidecar unreachable (test default)")
        return self.sync_client


class FakeProc:
    """Minimal asyncio.subprocess.Process stand-in."""

    def __init__(self, pid: int = 4242):
        self.pid = pid
        self.signals: list[int] = []
        self.killed = False

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        return 0


class SpawnSpy:
    """Records every subprocess spawn. Returns ``proc`` (or fails if
    ``proc is None`` — used by the "must never spawn" tests)."""

    def __init__(self, proc: Optional[FakeProc] = None):
        self.proc = proc
        self.calls: list[tuple[tuple, dict]] = []

    async def __call__(self, *args: Any, **kwargs: Any) -> FakeProc:
        self.calls.append((args, kwargs))
        if self.proc is None:
            raise AssertionError(
                "start() spawned a subprocess it should have refused to spawn: "
                f"argv={args!r}"
            )
        return self.proc


class AsyncioShim:
    """Proxies the real asyncio module, overriding only the spawn call.

    Patching `asyncio.create_subprocess_exec` globally would be visible
    to pytest-asyncio and to any other coroutine alive in the loop; this
    is scoped to the whatsapp_baileys module object only.
    """

    def __init__(self, spawn: SpawnSpy):
        self._spawn = spawn

    def __getattr__(self, name: str) -> Any:
        return getattr(asyncio, name)

    async def create_subprocess_exec(self, *args: Any, **kwargs: Any) -> Any:
        return await self._spawn(*args, **kwargs)


async def finish(coro, what: str, diag, timeout: float = 5.0):
    """Await ``coro`` with a hard ceiling.

    Never a bare await: a mutation that makes ``start()`` block must FAIL
    the test, not hang the suite. ``diag`` is called on timeout so the
    message reports what the adapter actually saw rather than just
    "timed out".
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise AssertionError(
            f"{what} did not finish within {timeout}s — observed: {diag()}"
        ) from exc


# ── Harness ──────────────────────────────────────────────────────


class Harness:
    def __init__(self, mod, monkeypatch, tmp_path):
        self.mod = mod
        self._mp = monkeypatch
        self._tmp = tmp_path
        self.httpx = FakeHttpxModule()
        self.spawn = SpawnSpy()
        self.dedupe_claims: list[tuple[str, str]] = []
        monkeypatch.setattr(mod, "httpx", self.httpx)
        monkeypatch.setattr(mod, "asyncio", AsyncioShim(self.spawn))
        # Belt and braces: even if a real client escaped the stub, point
        # it at the IANA discard port rather than the live sidecar port.
        monkeypatch.setattr(mod, "_SIDECAR_BASE", "http://127.0.0.1:9")

    # -- environment shaping ------------------------------------

    def sidecar_bundle(self, *, mjs: bool = True, node_modules: bool = True):
        d = self._tmp / "whatsapp_sidecar"
        d.mkdir(exist_ok=True)
        if mjs:
            (d / "sidecar.mjs").write_text("// stub\n")
        if node_modules:
            (d / "node_modules").mkdir(exist_ok=True)
        self._mp.setattr(self.mod, "_resolve_sidecar_dir", lambda: d)
        return d

    def node_on_path(self, present: bool = True):
        self._mp.setattr(
            self.mod.shutil, "which", lambda name: "/usr/bin/node" if present else None
        )

    def will_spawn(self, pid: int = 4242) -> FakeProc:
        proc = FakeProc(pid=pid)
        self.spawn.proc = proc
        return proc

    def sidecar_http(self, routes: Optional[dict] = None) -> StubAsyncClient:
        client = StubAsyncClient(routes)
        self.httpx.async_client = client
        return client

    def pair_status_http(self, routes: dict) -> StubSyncClient:
        client = StubSyncClient(routes)
        self.httpx.sync_client = client
        return client

    def quiet_background_tasks(self):
        """No SSE stream, no DB sweep — both are separate units."""
        async def _noop_events(_self):
            return None

        async def _noop_sweep():
            return None

        self._mp.setattr(
            self.mod.BaileysWhatsAppChannel, "_consume_events_forever", _noop_events
        )
        import app.agent.channels.whatsapp_dedupe as dedupe_mod
        self._mp.setattr(dedupe_mod, "run_sweep_loop", _noop_sweep)

    def stub_dedupe(self, results: Optional[list[bool]] = None, user_id: str = "u-test-1"):
        """Replace the DB-backed claim; record every (user_id, msg_id)."""
        import app.agent.channels.whatsapp_dedupe as dedupe_mod
        from app.config import settings

        queue = list(results) if results is not None else None

        async def _claim(uid: str, mid: str, **_kw):
            self.dedupe_claims.append((uid, mid))
            if queue is None:
                return True
            return queue.pop(0) if queue else True

        self._mp.setattr(dedupe_mod, "claim", _claim)
        self._mp.setattr(settings, "user_id", user_id)

    # -- convenience ---------------------------------------------

    def channel(self, allowed=None):
        return self.mod.BaileysWhatsAppChannel(allowed_numbers=allowed)

    async def drain(self, ch):
        """Let the (no-op) background tasks start() created complete, so
        the loop never reports 'Task was destroyed but it is pending'."""
        for task in (ch._event_task, ch._sweep_task):
            if task is not None:
                await task


@pytest.fixture
def wa(monkeypatch, tmp_path):
    from app.agent.channels import whatsapp_baileys as mod

    mod._active_channel = None
    h = Harness(mod, monkeypatch, tmp_path)
    try:
        yield h
    finally:
        mod._active_channel = None


class Recorder:
    """Captures dispatched InboundMessages."""

    def __init__(self):
        self.messages = []

    async def __call__(self, msg):
        self.messages.append(msg)


# ── _normalize_e164 ──────────────────────────────────────────────


class TestNormaliseE164:
    def test_empty(self):
        from app.agent.channels.whatsapp_baileys import _normalize_e164
        assert _normalize_e164("") == ""

    def test_plain_canonical(self):
        from app.agent.channels.whatsapp_baileys import _normalize_e164
        assert _normalize_e164("+14155552671") == "+14155552671"

    def test_missing_plus(self):
        from app.agent.channels.whatsapp_baileys import _normalize_e164
        assert _normalize_e164("14155552671") == "+14155552671"

    def test_with_spaces_dashes_parens(self):
        from app.agent.channels.whatsapp_baileys import _normalize_e164
        assert _normalize_e164("+1 (415) 555-2671") == "+14155552671"

    def test_double_zero_intl_prefix(self):
        from app.agent.channels.whatsapp_baileys import _normalize_e164
        assert _normalize_e164("00 1 415-555-2671") == "+14155552671"

    def test_only_garbage(self):
        from app.agent.channels.whatsapp_baileys import _normalize_e164
        assert _normalize_e164("abc def") == ""


# ── construction ─────────────────────────────────────────────────


class TestConstruction:
    def test_empty_allowlist(self, wa):
        assert wa.channel().allowed_numbers == set()

    def test_allowlist_normalised(self, wa):
        ch = wa.channel([
            "+14155552671",
            "14155552671",
            "+1 415-555-2671",
            "+1 (415) 555-2671",
            "00 1 415 555 2671",
        ])
        # All collapse to the same canonical form
        assert ch.allowed_numbers == {"+14155552671"}

    def test_allowlist_drops_empty_entries(self, wa):
        ch = wa.channel(["+14155552671", "", "   ", "abc"])
        assert ch.allowed_numbers == {"+14155552671"}

    def test_channel_type_is_whatsapp(self, wa):
        from app.agent.channels.base import ChannelType
        assert wa.channel().channel_type == ChannelType.WHATSAPP


# ── health() pre-start ───────────────────────────────────────────


class TestHealthPreStart:
    def test_shape(self, wa):
        """`/agent/health` merges this dict verbatim; every key is a
        contract with the Settings UI."""
        h = wa.channel(["+14155552671"]).health()
        assert h["mode"] == "qr_link"
        assert h["configured"] is True
        assert h["started"] is False
        assert h["connected"] is False
        assert h["self_e164"] is None
        assert h["allowed_numbers_count"] == 1
        assert h["inbound_count"] == 0
        assert h["last_inbound_at"] is None
        assert h["last_send_at"] is None
        assert h["last_send_error"] is None
        assert h["session_status"] == "not_linked"
        # Sidecar keys are new in 056eaf25 and were pinned nowhere.
        assert h["sidecar_pid"] is None
        assert h["sidecar_booted_at"] is None


# ── pairing snapshot ─────────────────────────────────────────────


class TestPairingSnapshot:
    def test_empty_when_sidecar_unreachable(self, wa):
        """Sidecar down → cached snapshot, never a crash and never a
        stale 'linked' invented out of nothing."""
        snap = wa.channel().get_pairing_status()
        assert snap["qr_data_url"] is None
        assert snap["qr_emitted_at"] is None
        assert snap["self_e164"] is None
        assert snap["connected"] is False
        assert snap["session_status"] == "not_linked"

    def test_poll_caches_sidecar_truth(self, wa):
        """The poll is the only thing that keeps `/agent/health` honest
        between SSE frames — it must WRITE BACK what the sidecar said,
        not just return it."""
        ch = wa.channel()
        sync = wa.pair_status_http({
            "GET /pair/status": StubResponse(200, {
                "session_status": "linked",
                "connected": True,
                "self_e164": "+14155552671",
                "qr_data_url": None,
                "qr_emitted_at": None,
            }),
        })
        body = ch.get_pairing_status()
        assert sync.gets == ["/pair/status"]
        assert body["session_status"] == "linked"
        # …and the cache took it, so health() agrees without another poll.
        h = ch.health()
        assert h["session_status"] == "linked"
        assert h["connected"] is True
        assert h["self_e164"] == "+14155552671"


# ── start(): sidecar boot ────────────────────────────────────────


class TestSidecarBoot:
    async def test_health_seeds_session_status_and_leaves_pairing_alone(self, wa):
        """A container restart on an ALREADY-PAIRED session must adopt
        the sidecar's 'linked' and must NOT fire /pair/start.

        /pair/start wipes the Baileys auth dir (sidecar.mjs
        handlePairStart → wipeAuthDir), so auto-kicking a linked session
        would destroy a working pairing and force the user to re-scan.
        """
        wa.sidecar_bundle()
        wa.node_on_path()
        proc = wa.will_spawn(pid=4242)
        wa.quiet_background_tasks()
        client = wa.sidecar_http({
            "GET /health": StubResponse(200, {
                "session_status": "linked",
                "connected": True,
                "self_e164": "+14155552671",
            }),
        })
        ch = wa.channel(["+14155552671"])

        await finish(ch.start(), "start()", lambda: f"health={ch.health()}")

        h = ch.health()
        assert h["session_status"] == "linked"
        assert h["connected"] is True
        assert h["self_e164"] == "+14155552671"
        assert h["started"] is True
        assert h["sidecar_pid"] == 4242
        assert h["sidecar_booted_at"] is not None
        assert client.gets == ["/health"]
        assert "/pair/start" not in client.post_paths, (
            "start() kicked the pair flow on an already-linked session — "
            "that wipes the auth dir and unpairs the user"
        )
        assert wa.spawn.calls, "the sidecar was never spawned"
        assert proc.signals == []
        await wa.drain(ch)

    async def test_boot_without_creds_auto_kicks_pair(self, wa):
        """Fresh container / wiped volume: the sidecar sits idle in
        not_linked until someone calls /pair/start. Without the kick the
        user sees no QR, messages silently go nowhere."""
        wa.sidecar_bundle()
        wa.node_on_path()
        wa.will_spawn()
        wa.quiet_background_tasks()
        client = wa.sidecar_http({
            "GET /health": StubResponse(200, {
                "session_status": "not_linked",
                "connected": False,
                "self_e164": None,
            }),
        })
        ch = wa.channel()

        await finish(ch.start(), "start()", lambda: f"posts={client.post_paths}")

        assert "/pair/start" in client.post_paths, (
            f"no pair kick on a credentialless boot; posts={client.post_paths}"
        )
        assert ch.health()["session_status"] == "linking"
        await wa.drain(ch)

    async def test_start_registers_and_stop_deregisters_the_active_channel(self, wa):
        """`/agent/health` and the QR endpoints read this module global.
        Construct must NOT register; a successful start must; stop must
        clear it AND SIGTERM the sidecar (an orphaned node process holds
        port 8002 and the next start silently can't bind)."""
        from app.agent.channels.whatsapp_baileys import get_active_baileys_channel

        wa.sidecar_bundle()
        wa.node_on_path()
        proc = wa.will_spawn()
        wa.quiet_background_tasks()
        client = wa.sidecar_http({
            "GET /health": StubResponse(200, {"session_status": "linked", "connected": True}),
        })

        ch = wa.channel()
        assert get_active_baileys_channel() is None

        await finish(ch.start(), "start()", lambda: f"health={ch.health()}")
        assert get_active_baileys_channel() is ch
        await wa.drain(ch)

        await finish(ch.stop(), "stop()", lambda: f"signals={proc.signals}")
        assert get_active_baileys_channel() is None
        assert proc.signals == [signal.SIGTERM], (
            f"sidecar was not SIGTERMed on stop; signals={proc.signals}"
        )
        assert client.closed is True
        assert ch.health()["started"] is False


class TestStartDegradesGracefully:
    """Replaces the old `test_no_neonize_no_crash`, which passed only
    because `backend/whatsapp_sidecar/node_modules` happens to be absent
    from a fresh checkout — on a machine where someone ran `npm install`
    it spawned a real Node process. Here the spawn spy RAISES if it is
    ever reached, so each guard is proved rather than assumed.
    """

    async def _assert_bailed(self, wa, ch):
        from app.agent.channels.whatsapp_baileys import get_active_baileys_channel

        await finish(ch.start(), "start()", lambda: f"spawns={wa.spawn.calls}")
        assert wa.spawn.calls == [], (
            f"start() tried to spawn the sidecar anyway: {wa.spawn.calls}"
        )
        assert ch.health()["started"] is False
        assert ch.health()["sidecar_pid"] is None
        assert get_active_baileys_channel() is None

    async def test_missing_sidecar_script(self, wa):
        # node_modules PRESENT on purpose: each guard has to be shown to
        # bail on its own. With both absent the node_modules guard masks
        # this one, and deleting the sidecar.mjs check stays green — a
        # mutation run caught exactly that.
        wa.sidecar_bundle(mjs=False, node_modules=True)
        wa.node_on_path()
        await self._assert_bailed(wa, wa.channel())

    async def test_missing_node_modules(self, wa):
        wa.sidecar_bundle(mjs=True, node_modules=False)
        wa.node_on_path()
        await self._assert_bailed(wa, wa.channel())

    async def test_node_not_installed(self, wa):
        wa.sidecar_bundle()
        wa.node_on_path(present=False)
        await self._assert_bailed(wa, wa.channel())


# ── SSE events drive the state machine ───────────────────────────


class TestSidecarEvents:
    async def test_connection_open_marks_linked(self, wa):
        ch = wa.channel()
        await ch._on_sidecar_event({
            "type": "connection_open", "self_e164": "+14155552671",
        })
        h = ch.health()
        assert h["session_status"] == "linked"
        assert h["connected"] is True
        assert h["self_e164"] == "+14155552671"

    async def test_logged_out_clears_identity(self, wa):
        """A remote unlink must drop the cached number too — otherwise
        the UI keeps showing a phone that is no longer paired."""
        ch = wa.channel()
        await ch._on_sidecar_event({
            "type": "connection_open", "self_e164": "+14155552671",
        })
        await ch._on_sidecar_event({"type": "logged_out"})
        h = ch.health()
        assert h["session_status"] == "logged_out"
        assert h["connected"] is False
        assert h["self_e164"] is None

    async def test_qr_marks_linking(self, wa):
        ch = wa.channel()
        await ch._on_sidecar_event({"type": "qr"})
        assert ch.health()["session_status"] == "linking"

    async def test_unknown_event_is_ignored_not_fatal(self, wa):
        ch = wa.channel()
        await ch._on_sidecar_event({"type": "presence.update", "x": 1})
        assert ch.health()["session_status"] == "not_linked"
        assert ch.health()["inbound_count"] == 0


# ── ACL gate, driven through the real inbound path ───────────────


class TestAcl:
    """Drives the REAL gate: a sidecar `message` SSE frame all the way to
    `dispatch()`. The wire shape is what sidecar.mjs actually emits —
    `from` is already canonical E.164 (`from: fromE164`), not a JID.
    """

    @staticmethod
    def _frame(sender: str, text: str = "hi", msg_id: str = "m1") -> dict:
        return {
            "type": "message",
            "from": sender,
            "text": text,
            "message_id": msg_id,
            "push_name": "Nariman",
        }

    async def test_allowlisted_sender_is_dispatched(self, wa):
        from app.agent.channels.base import ChannelType

        wa.stub_dedupe()
        rec = Recorder()
        # Allowlist stored in a messy human form; inbound arrives bare.
        ch = wa.channel(["+1 (415) 555-2671"])
        ch.set_message_callback(rec)

        await ch._on_sidecar_event(self._frame("14155552671"))

        assert len(rec.messages) == 1, f"expected 1 dispatch, got {rec.messages!r}"
        msg = rec.messages[0]
        assert msg.channel == ChannelType.WHATSAPP
        assert msg.channel_user_id == "+14155552671"
        assert msg.channel_chat_id == "+14155552671"
        assert msg.text == "hi"
        assert msg.display_name == "Nariman"
        h = ch.health()
        assert h["inbound_count"] == 1
        assert h["last_inbound_at"] is not None

    async def test_sender_outside_allowlist_is_dropped(self, wa):
        """Positive control first, so an always-empty recorder cannot
        make this test pass vacuously."""
        wa.stub_dedupe()
        rec = Recorder()
        ch = wa.channel(["+14155552671"])
        ch.set_message_callback(rec)

        await ch._on_sidecar_event(self._frame("+14155552671", msg_id="m1"))
        assert len(rec.messages) == 1  # control: the path works

        await ch._on_sidecar_event(self._frame("+12025550100", msg_id="m2"))
        assert len(rec.messages) == 1, (
            f"a non-allowlisted number got through: {rec.messages[-1]!r}"
        )
        assert ch.health()["inbound_count"] == 1

    async def test_empty_allowlist_blocks_everyone(self, wa):
        """Secure default: no allowlist configured = nobody gets in, not
        'anyone gets in'."""
        wa.stub_dedupe()
        blocked, admitted = Recorder(), Recorder()

        closed = wa.channel()          # empty allowlist
        closed.set_message_callback(blocked)
        open_ch = wa.channel(["+14155552671"])
        open_ch.set_message_callback(admitted)

        frame = self._frame("+14155552671")
        await closed._on_sidecar_event(frame)
        await open_ch._on_sidecar_event(frame)

        # Same frame, same code path — only the allowlist differs.
        assert admitted.messages, "positive control failed; the test proves nothing"
        assert blocked.messages == [], (
            f"empty allowlist admitted a message: {blocked.messages!r}"
        )
        assert closed.health()["inbound_count"] == 0

    async def test_media_only_message_is_not_dispatched(self, wa):
        """No text = nothing to run the LLM on. It must not reach
        dispatch(), and must not inflate inbound_count."""
        wa.stub_dedupe()
        rec = Recorder()
        ch = wa.channel(["+14155552671"])
        ch.set_message_callback(rec)

        await ch._on_sidecar_event(self._frame("+14155552671", text="   "))

        assert rec.messages == []
        assert ch.health()["inbound_count"] == 0


# ── dedupe ───────────────────────────────────────────────────────


class TestDedupe:
    async def test_retransmit_is_suppressed(self, wa):
        """Meta retransmits to linked devices on flaky networks. Without
        the dedupe claim a transport blip re-runs the whole LLM turn."""
        wa.stub_dedupe(results=[True, False], user_id="u-test-1")
        rec = Recorder()
        ch = wa.channel(["+14155552671"])
        ch.set_message_callback(rec)

        frame = TestAcl._frame("+14155552671", msg_id="wamid.ABC")
        await ch._on_sidecar_event(frame)
        await ch._on_sidecar_event(frame)

        assert wa.dedupe_claims == [
            ("u-test-1", "wamid.ABC"),
            ("u-test-1", "wamid.ABC"),
        ], f"claim was not consulted per delivery: {wa.dedupe_claims!r}"
        assert len(rec.messages) == 1, (
            f"the retransmit was dispatched a second time: {rec.messages!r}"
        )
        assert ch.health()["inbound_count"] == 1


# ── pairing controls (Settings UI) ───────────────────────────────


class TestPairingControls:
    async def test_kick_pair_resets_to_linking(self, wa):
        client = wa.sidecar_http()
        ch = wa.channel()
        ch._http = client
        # Pretend we were linked with a QR on screen.
        await ch._on_sidecar_event({"type": "connection_open", "self_e164": "+14155552671"})

        await finish(ch.kick_pair(), "kick_pair()", lambda: f"posts={client.post_paths}")

        assert "/pair/start" in client.post_paths, (
            f"kick_pair() never told the sidecar; posts={client.post_paths}"
        )
        snap = ch.get_pairing_status()   # sidecar poll unreachable → cached
        assert snap["session_status"] == "linking"
        assert snap["connected"] is False
        assert snap["qr_data_url"] is None
        assert snap["qr_emitted_at"] is None

    async def test_force_logout_clears_session(self, wa):
        client = wa.sidecar_http()
        ch = wa.channel()
        ch._http = client
        await ch._on_sidecar_event({"type": "connection_open", "self_e164": "+14155552671"})

        await finish(ch.force_logout(), "force_logout()", lambda: f"posts={client.post_paths}")

        assert "/pair/logout" in client.post_paths, (
            f"force_logout() never told the sidecar; posts={client.post_paths}"
        )
        h = ch.health()
        assert h["session_status"] == "not_linked", (
            "Disconnect must land on not_linked — 'logged_out' renders as an "
            "error banner rather than an invitation to pair again"
        )
        assert h["connected"] is False
        assert h["self_e164"] is None

    async def test_request_pairing_code_returns_code_and_enters_linking(self, wa):
        """Mobile onboarding path: the user can't scan their own screen,
        so they type an 8-char code instead."""
        client = wa.sidecar_http({
            "POST /pair/code": StubResponse(200, {"ok": True, "pairing_code": "WXYZ1234"}),
        })
        ch = wa.channel()
        ch._http = client

        code = await finish(
            ch.request_pairing_code("+14155552671"),
            "request_pairing_code()",
            lambda: f"posts={client.posts}",
        )

        assert code == "WXYZ1234"
        assert client.posts == [("/pair/code", {"phone": "+14155552671"})]
        assert ch.health()["session_status"] == "linking"
        assert ch.health()["connected"] is False

    async def test_request_pairing_code_surfaces_sidecar_error(self, wa):
        """A failed request must raise, not hand the UI an empty code the
        user would sit and stare at."""
        client = wa.sidecar_http({
            "POST /pair/code": StubResponse(200, {"ok": False, "error": "rate limited"}),
        })
        ch = wa.channel()
        ch._http = client

        with pytest.raises(RuntimeError, match="rate limited"):
            await finish(
                ch.request_pairing_code("+14155552671"),
                "request_pairing_code()",
                lambda: f"posts={client.posts}",
            )
