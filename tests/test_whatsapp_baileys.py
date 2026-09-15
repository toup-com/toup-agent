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


class FakeHttpxModule:
    """Stands in for the `httpx` module attribute on whatsapp_baileys.

    Default behaviour is "sidecar unreachable": constructing a client
    raises. That is the deliberate safety net — without it,
    ``get_pairing_status()`` on a developer machine that happens to be
    running a real sidecar on 127.0.0.1:8002 would silently talk to it.
    """

    def __init__(self) -> None:
        self.async_client: Optional[StubAsyncClient] = None
        self.async_ctor_kwargs: list[dict] = []

    def AsyncClient(self, **kwargs: Any) -> StubAsyncClient:  # noqa: N802
        self.async_ctor_kwargs.append(kwargs)
        if self.async_client is None:
            raise ConnectionError("sidecar unreachable (test default)")
        return self.async_client

    def Client(self, **_kwargs: Any) -> Any:  # noqa: N802
        # A TRIPWIRE, not a stub. `get_pairing_status()` used to open a
        # blocking client here, from inside a FastAPI route — every
        # ~1.5 s poll of the Settings modal stalled the agent's event
        # loop for as long as the sidecar took to answer. Nothing in the
        # adapter may construct a synchronous client again.
        raise AssertionError(
            "whatsapp_baileys constructed a BLOCKING httpx.Client — that "
            "runs on the agent's event loop"
        )


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
        # The reconciler never returns on its own — cancel it explicitly.
        recon = ch._reconcile_task
        if recon is not None and not recon.done():
            recon.cancel()
            try:
                await recon
            except asyncio.CancelledError:
                pass


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
    async def test_empty_when_sidecar_unreachable(self, wa):
        """Sidecar down → cached snapshot, never a crash and never a
        stale 'linked' invented out of nothing."""
        snap = await wa.channel().get_pairing_status()
        assert snap["qr_data_url"] is None
        assert snap["qr_emitted_at"] is None
        assert snap["self_e164"] is None
        assert snap["connected"] is False
        assert snap["session_status"] == "not_linked"

    async def test_poll_caches_sidecar_truth(self, wa):
        """The poll must WRITE BACK what the sidecar said, not just return
        it — and it must do so over the SHARED ASYNC client."""
        ch = wa.channel()
        client = wa.sidecar_http({
            "GET /pair/status": StubResponse(200, {
                "session_status": "linked",
                "connected": True,
                "self_e164": "+14155552671",
                "qr_data_url": None,
                "qr_emitted_at": None,
            }),
        })
        ch._http = client
        body = await ch.get_pairing_status()
        assert client.gets == ["/pair/status"]
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
        snap = await ch.get_pairing_status()
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


# ── the reconciler: the sidecar is the truth, SSE is the hurry ────
#
# Round 44. A `connection_open` emitted while no SSE consumer was attached
# reaches nobody — sidecar.mjs `emitEvent` writes to the currently-attached
# clients and keeps no replay — and `linked` used to be reachable ONLY from
# that frame. Observed on pool slot 18 on 2026-09-14: the sidecar was linked,
# the agent said `linking` forever, and every client rendered
# "Not connected / Connect" for a session that was working.


class TestSidecarReconciler:
    async def test_a_missed_connection_open_is_recovered_within_one_tick(self, wa):
        """The defect, end to end: boot sees `linking`, the SSE frame that
        would have said `linked` is never delivered, and the reconciler is
        the only thing that can find out."""
        from app.services import health_signals as hs

        hs.reset_for_tests()
        wa.sidecar_bundle()
        wa.node_on_path()
        wa.will_spawn()
        wa.quiet_background_tasks()
        client = wa.sidecar_http({
            "GET /health": StubResponse(200, {
                "session_status": "linking",
                "connected": False,
                "self_e164": None,
            }),
        })
        ch = wa.channel(["+14155552671"])
        await finish(ch.start(), "start()", lambda: f"health={ch.health()}")
        assert ch.health()["session_status"] == "linking"

        # The phone confirms. The sidecar flips; the frame reaches nobody.
        client.routes["GET /health"] = StubResponse(200, {
            "session_status": "linked",
            "connected": True,
            "self_e164": "+14155552671",
        })
        assert ch.health()["session_status"] == "linking", (
            "the cache moved with no event and no read — impossible"
        )

        assert await ch._reconcile_once("reconcile") is True
        h = ch.health()
        assert h["session_status"] == "linked", (
            f"the reconciler did not adopt the sidecar's truth: {h}"
        )
        assert h["connected"] is True
        assert h["self_e164"] == "+14155552671"
        assert h["session_status_source"] == "reconcile"
        assert hs.get("wa_status_reconciled") == 1, (
            "a status the SSE stream failed to deliver was recovered "
            "silently — nothing on the fleet can see that happening"
        )
        hs.reset_for_tests()
        await wa.drain(ch)

    async def test_the_running_loop_recovers_it_without_being_poked(self, wa):
        """Not the helper — the TASK `start()` creates. A reconcile method
        nobody schedules fixes nothing."""
        wa.sidecar_bundle()
        wa.node_on_path()
        wa.will_spawn()
        wa.quiet_background_tasks()
        client = wa.sidecar_http({
            "GET /health": StubResponse(200, {
                "session_status": "linking", "connected": False, "self_e164": None,
            }),
        })
        wa._mp.setattr(wa.mod, "_RECONCILE_INTERVAL_S", 0.01)
        ch = wa.channel(["+14155552671"])
        await finish(ch.start(), "start()", lambda: f"health={ch.health()}")
        assert ch._reconcile_task is not None and not ch._reconcile_task.done()

        client.routes["GET /health"] = StubResponse(200, {
            "session_status": "linked", "connected": True,
            "self_e164": "+14155552671",
        })
        for _ in range(200):
            await asyncio.sleep(0.01)
            if ch.health()["session_status"] == "linked":
                break
        assert ch.health()["session_status"] == "linked", (
            f"the reconcile task never re-read the sidecar: {ch.health()}"
        )
        await wa.drain(ch)

    async def test_a_logged_out_sidecar_is_adopted_and_stays(self, wa):
        """The user unlinked the device from their phone. That is a FACT,
        and re-reading the sidecar must not talk us out of it."""
        client = wa.sidecar_http({
            "GET /health": StubResponse(200, {
                "session_status": "logged_out", "connected": False,
                "self_e164": None,
            }),
        })
        ch = wa.channel()
        ch._http = client
        await ch._on_sidecar_event(
            {"type": "connection_open", "self_e164": "+14155552671"}
        )
        assert ch.health()["session_status"] == "linked"

        await ch._on_sidecar_event({"type": "logged_out"})
        h = ch.health()
        assert h["session_status"] == "logged_out"
        assert h["self_e164"] is None
        assert h["session_status_source"] == "sse"

        assert await ch._reconcile_once("reconcile") is True
        h = ch.health()
        assert h["session_status"] == "logged_out", (
            "a reconcile read undid an explicit logout"
        )
        assert h["self_e164"] is None

    async def test_an_unreachable_sidecar_neither_raises_nor_spins(self, wa):
        """A dead port must cost a handful of requests, not a hot loop for
        the life of the container."""
        class Boom(StubAsyncClient):
            def __init__(self):
                super().__init__()
                self.attempts = 0

            async def get(self, path: str, **_kw: Any) -> StubResponse:
                self.attempts += 1
                raise ConnectionError("sidecar down")

        ch = wa.channel()
        ch._http = Boom()

        # No raise, no state change, and the caller is told to back off.
        assert await ch._reconcile_once("reconcile") is False
        assert ch.health()["session_status"] == "not_linked"

        wa._mp.setattr(wa.mod, "_RECONCILE_INTERVAL_S", 0.01)
        wa._mp.setattr(wa.mod, "_RECONCILE_MAX_BACKOFF_S", 0.08)
        ch._reconcile_wake = asyncio.Event()
        task = asyncio.create_task(ch._reconcile_forever())
        await asyncio.sleep(0.3)
        assert not task.done(), (
            f"the reconcile loop died on an unreachable sidecar: "
            f"{task.exception() if task.done() else None}"
        )
        attempts = ch._http.attempts
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # 0.3 s at a flat 0.01 s interval would be ~30. With the geometric
        # backoff (0.01→0.02→0.04→0.08, capped) it is a handful.
        assert 2 <= attempts <= 10, (
            f"{attempts} reads in 0.3s — the loop is not backing off"
        )

    async def test_a_bad_sidecar_answer_is_not_state(self, wa):
        """Non-200 and unparseable bodies leave the cache alone."""
        class Bad(StubAsyncClient):
            def __init__(self, resp):
                super().__init__()
                self._resp = resp

            async def get(self, path: str, **_kw: Any):
                self.gets.append(path)
                return self._resp

        class Unparseable(StubResponse):
            def json(self):
                raise ValueError("not json")

        ch = wa.channel()
        await ch._on_sidecar_event(
            {"type": "connection_open", "self_e164": "+14155552671"}
        )

        ch._http = Bad(StubResponse(503, {}))
        assert await ch._reconcile_once("reconcile") is False
        ch._http = Bad(Unparseable(200))
        assert await ch._reconcile_once("reconcile") is False
        ch._http = None
        assert await ch._reconcile_once("reconcile") is False

        assert ch.health()["session_status"] == "linked", (
            "a failed read was treated as a state change"
        )

    async def test_health_reports_where_the_status_came_from(self, wa):
        """`session_status` is a cache; an operator has to be able to tell a
        fresh sidecar read from a boot value nothing has refreshed since."""
        ch = wa.channel()
        h = ch.health()
        assert h["session_status_source"] == "init"
        assert h["since_last_sidecar_read_s"] is None
        assert h["reconciler_running"] is False

        await ch._on_sidecar_event({"type": "connection_open"})
        h = ch.health()
        assert h["session_status_source"] == "sse"
        assert isinstance(h["session_status_stable_s"], int)

    async def test_a_push_write_does_not_pretend_the_sidecar_was_asked(self, wa):
        """Two clocks, two questions. `since_last_sidecar_read_s` answers
        "when did we last ASK" — an SSE frame or a local optimistic write is
        not an answer to that, and stamping it there let a cache that had
        not been confirmed in minutes look a second old."""
        client = wa.sidecar_http({
            "GET /health": StubResponse(200, {
                "session_status": "linked", "connected": True,
                "self_e164": "+14155552671",
            }),
        })
        ch = wa.channel()
        ch._http = client

        # Never read → never asked.
        await ch._on_sidecar_event({"type": "qr"})
        assert ch.health()["since_last_sidecar_read_s"] is None, (
            "an SSE frame was recorded as a sidecar read"
        )
        await ch.force_logout()
        assert ch.health()["since_last_sidecar_read_s"] is None, (
            "a local write was recorded as a sidecar read"
        )

        # A real read is.
        assert await ch._reconcile_once("reconcile") is True
        assert ch.health()["since_last_sidecar_read_s"] == 0

    async def test_the_stability_clock_moves_only_on_a_VALUE_change(self, wa):
        """`session_status_stable_s` is the platform's whole discriminator
        between a session whose credentials are gone and a reconnect in
        flight. A repeated read of the SAME status must not reset it."""
        client = wa.sidecar_http({
            "GET /health": StubResponse(200, {
                "session_status": "linking", "connected": False,
                "self_e164": None,
            }),
        })
        ch = wa.channel()
        ch._http = client
        await ch._reconcile_once("reconcile")

        # Backdate: the status has held for two minutes.
        ch._status_since -= 120
        assert ch.health()["session_status_stable_s"] >= 120

        for _ in range(3):
            await ch._reconcile_once("reconcile")
        assert ch.health()["session_status_stable_s"] >= 120, (
            "re-reading the same status reset the stability clock — every "
            "stale state would then look brand new"
        )

        # A real change restarts it.
        client.routes["GET /health"] = StubResponse(200, {
            "session_status": "linked", "connected": True,
            "self_e164": "+14155552671",
        })
        await ch._reconcile_once("reconcile")
        assert ch.health()["session_status_stable_s"] == 0

    async def test_the_reconciler_starts_with_the_channel_and_stops_with_it(self, wa):
        """A task that outlives `stop()` keeps polling a sidecar the next
        generation is about to replace."""
        wa.sidecar_bundle()
        wa.node_on_path()
        wa.will_spawn()
        wa.quiet_background_tasks()
        wa.sidecar_http({
            "GET /health": StubResponse(200, {"session_status": "linked", "connected": True}),
        })
        ch = wa.channel()
        assert ch._reconcile_task is None

        await finish(ch.start(), "start()", lambda: f"health={ch.health()}")
        task = ch._reconcile_task
        assert task is not None and not task.done()
        assert ch.health()["reconciler_running"] is True

        # Let the loop actually PARK in its wait. A task that has not run
        # its first step yet exits on the `_stopping` check for free, so
        # without this the case passes with no cancellation at all — and the
        # real one is always parked, holding the channel for a whole
        # interval (up to the backoff cap) past `stop()`.
        for _ in range(5):
            await asyncio.sleep(0)
        assert not task.done(), "the reconcile loop never parked"

        await finish(ch.stop(), "stop()", lambda: f"task={task}")
        assert task.done(), "stop() left the reconcile task running"
        assert ch._reconcile_task is None
        assert ch.health()["reconciler_running"] is False

    async def test_an_attaching_sse_stream_asks_for_a_read_at_once(self, wa):
        """The interval bounds the damage; the wake removes it. A stream
        that has JUST attached is the moment the cache is most likely to be
        stale, because anything emitted before it was dropped."""
        ch = wa.channel()
        ch._reconcile_wake = asyncio.Event()
        assert not ch._reconcile_wake.is_set()
        ch._wake_reconciler()
        assert ch._reconcile_wake.is_set(), (
            "_consume_events_forever's wake does nothing"
        )
        # …and it is actually called from the SSE loop, right after the
        # stream is established. Source probe: a wake nobody fires is dead.
        import inspect
        src = inspect.getsource(wa.mod.BaileysWhatsAppChannel._consume_events_forever)
        assert "_wake_reconciler()" in src, (
            "the SSE consumer does not wake the reconciler when it attaches"
        )

    async def test_the_pairing_snapshot_carries_its_own_provenance(self, wa):
        """The platform's downgrade rule reads `self_e164`,
        `session_status_stable_s` and `session_status_source` off THIS body.
        Ship it without them and the rule degrades to "never clear a link",
        silently — the exact defect the rule exists to bound."""
        keys = {"session_status_source", "session_status_stable_s",
                "since_last_sidecar_read_s"}

        # Served from the sidecar…
        client = wa.sidecar_http({
            "GET /pair/status": StubResponse(200, {
                "session_status": "linking", "connected": False,
                "self_e164": None, "qr_data_url": None,
            }),
        })
        ch = wa.channel()
        ch._http = client
        body = await ch.get_pairing_status()
        assert keys <= set(body), f"missing {keys - set(body)}"
        assert body["session_status_source"] == "pair_status"
        assert isinstance(body["session_status_stable_s"], int)

        # …and served from cache when the sidecar is down, which is when a
        # "how old is this?" answer matters most.
        class Dead(StubAsyncClient):
            async def get(self, path: str, **_kw: Any) -> StubResponse:
                raise ConnectionError("down")

        ch2 = wa.channel()
        ch2._http = Dead()
        snap = await ch2.get_pairing_status()
        assert keys <= set(snap), f"missing {keys - set(snap)}"

    def test_get_pairing_status_is_async_and_opens_no_blocking_client(self):
        """It runs inside a FastAPI route on the agent's event loop, and the
        Settings modal polls it every ~1.5 s."""
        import inspect
        from pathlib import Path
        from app.agent.channels.whatsapp_baileys import BaileysWhatsAppChannel

        assert inspect.iscoroutinefunction(
            BaileysWhatsAppChannel.get_pairing_status
        ), "get_pairing_status blocks the event loop"
        src = (Path(__file__).resolve().parents[1] / "app" / "agent" / "channels"
               / "whatsapp_baileys.py").read_text()
        assert "httpx.Client(" not in src, (
            "a synchronous httpx client is back in the adapter"
        )
        # And the caller awaits it — a coroutine returned to FastAPI is
        # serialised as an empty object, not a snapshot.
        qr = (Path(__file__).resolve().parents[1] / "app" / "api"
              / "whatsapp_qr.py").read_text()
        assert "await channel.get_pairing_status()" in qr


# ── the status read is a READ (round 44, client-side follow-up) ───
#
# The mobile Channels panel now polls the platform's `/whatsapp/qr-status`
# once per visit to Connectors → Channels and once per return to the
# foreground, for EVERY user whose stored status is not `linked` — including
# users who never set WhatsApp up. That turns this chain into a background
# read on a hot screen, so it has to be provably inert: no channel start, no
# sidecar spawn, no `/pair/start`, no QR minted, and a fast calm answer when
# the sidecar is down rather than a hang behind the panel.


class TestStatusReadIsInert:
    async def test_a_status_read_on_an_unstarted_channel_spawns_nothing(self, wa):
        """`wa.spawn` RAISES if the adapter tries to spawn — the channel was
        never started, so there is no sidecar and there must be no attempt
        to make one."""
        ch = wa.channel(["+14155552671"])
        snap = await ch.get_pairing_status()
        assert snap["session_status"] == "not_linked"
        assert snap["qr_data_url"] is None
        assert wa.spawn.calls == [], (
            "reading the pairing status spawned a sidecar"
        )
        assert ch._sidecar_proc is None
        assert ch._event_task is None and ch._reconcile_task is None

    async def test_a_status_read_never_kicks_the_pair_flow(self, wa):
        """`/pair/start` WIPES the Baileys auth dir (sidecar.mjs
        handlePairStart → wipeAuthDir). A poll that reached it would unpair
        a working session, and this poll now runs unprompted."""
        client = wa.sidecar_http({
            "GET /pair/status": StubResponse(200, {
                "session_status": "not_linked", "connected": False,
                "self_e164": None, "qr_data_url": None,
            }),
        })
        ch = wa.channel()
        ch._http = client

        for _ in range(3):
            await ch.get_pairing_status()

        assert client.posts == [], (
            f"the status read POSTed to the sidecar: {client.post_paths}"
        )
        assert client.gets == ["/pair/status"] * 3
        assert wa.spawn.calls == []
        assert ch.health()["session_status"] == "not_linked"

    async def test_a_down_sidecar_answers_from_cache_on_a_short_leash(self, wa):
        """A sick sidecar must cost a cached snapshot, not the shared
        client's full budget — this poll sits behind a screen the user is
        looking at."""
        seen: list[Optional[float]] = []

        class Slow(StubAsyncClient):
            async def get(self, path: str, **kw: Any) -> StubResponse:
                seen.append(kw.get("timeout"))
                raise ConnectionError("sidecar down")

        ch = wa.channel()
        ch._http = Slow()
        await ch._on_sidecar_event(
            {"type": "connection_open", "self_e164": "+14155552671"}
        )

        snap = await ch.get_pairing_status()
        assert snap["session_status"] == "linked", (
            "a down sidecar erased a link the adapter already knew about"
        )
        assert snap["self_e164"] == "+14155552671"
        assert seen == [wa.mod._PAIR_STATUS_TIMEOUT_S], (
            f"the user-facing read is not on its own short timeout: {seen}"
        )
        assert wa.mod._PAIR_STATUS_TIMEOUT_S <= 3.0

    async def test_the_agent_route_answers_503_without_touching_anything(self, wa):
        """No active channel is the ordinary case for a user who never set
        WhatsApp up. It must be a typed 503, not a start attempt."""
        from fastapi import FastAPI, HTTPException
        from app.api.whatsapp_qr import router

        app = FastAPI()
        app.include_router(router, prefix="/api")
        handler = {
            r.path: r.endpoint for r in app.routes
            if getattr(r, "path", None) and getattr(r, "endpoint", None)
        }["/api/whatsapp/qr/status"]

        wa.mod._active_channel = None
        try:
            await handler(_user=object())
            assert False, "expected a 503"
        except HTTPException as exc:
            assert exc.status_code == 503
        assert wa.spawn.calls == []
        assert wa.mod._active_channel is None


# ── a poll is a snapshot of the PAST (review F2) ──────────────────


class TestPolledBodiesLoseToPushes:
    async def test_an_in_flight_read_cannot_walk_back_a_newer_event(self, wa):
        """`/health` is requested, the phone confirms, `connection_open`
        lands, and only THEN does the answer — serialised before any of it —
        come back saying `linking`. Applying it would un-link a session that
        is open, and the next reconcile is 10 s away."""
        applied: list[str] = []

        class Slow(StubAsyncClient):
            """Answers only once the test says so."""

            def __init__(self, gate: asyncio.Event):
                super().__init__()
                self.gate = gate

            async def get(self, path: str, **_kw: Any) -> StubResponse:
                self.gets.append(path)
                await self.gate.wait()
                return StubResponse(200, {
                    "session_status": "linking", "connected": False,
                    "self_e164": None,
                })

        gate = asyncio.Event()
        ch = wa.channel()
        ch._http = Slow(gate)

        task = asyncio.create_task(ch._reconcile_once("reconcile"))
        for _ in range(5):
            await asyncio.sleep(0)   # let the read start

        # The event the read raced.
        await ch._on_sidecar_event(
            {"type": "connection_open", "self_e164": "+14155552671"}
        )
        assert ch.health()["session_status"] == "linked"

        gate.set()
        assert await task is True   # the read itself succeeded…
        applied.append(ch.health()["session_status"])
        assert applied == ["linked"], (
            "a /health body serialised BEFORE the connection_open was "
            "applied after it, un-linking a live session"
        )
        assert ch.health()["self_e164"] == "+14155552671"
        assert ch.health()["session_status_source"] == "sse"

    async def test_a_read_that_starts_after_the_event_still_applies(self, wa):
        """ANTI-VACUITY: the guard is about ORDER, not about ignoring polls.
        A read issued after the last push is exactly how a missed
        `connection_open` gets recovered, and it must still work."""
        client = wa.sidecar_http({
            "GET /health": StubResponse(200, {
                "session_status": "logged_out", "connected": False,
                "self_e164": None,
            }),
        })
        ch = wa.channel()
        ch._http = client
        await ch._on_sidecar_event(
            {"type": "connection_open", "self_e164": "+14155552671"}
        )
        assert await ch._reconcile_once("reconcile") is True
        assert ch.health()["session_status"] == "logged_out"


# ── the wake has a floor (review F5) ──────────────────────────────


class TestReconcileFloor:
    async def test_back_to_back_wakes_cannot_become_a_hot_loop(self, wa):
        """A sidecar that accepts an SSE connection and immediately EOFs
        fires one wake per round-trip. Without a floor, "re-derive on
        attach" turns into a read storm against the process that is
        already sick."""
        reads: list[float] = []

        class Counting(StubAsyncClient):
            async def get(self, path: str, **_kw: Any) -> StubResponse:
                reads.append(asyncio.get_event_loop().time())
                return StubResponse(200, {
                    "session_status": "linking", "connected": False,
                    "self_e164": None,
                })

        wa._mp.setattr(wa.mod, "_RECONCILE_INTERVAL_S", 30.0)
        wa._mp.setattr(wa.mod, "_RECONCILE_MIN_INTERVAL_S", 0.05)
        ch = wa.channel()
        ch._http = Counting()
        ch._reconcile_wake = asyncio.Event()
        task = asyncio.create_task(ch._reconcile_forever())

        # Hammer the wake the way a flapping stream would.
        deadline = asyncio.get_event_loop().time() + 0.25
        while asyncio.get_event_loop().time() < deadline:
            ch._wake_reconciler()
            await asyncio.sleep(0.005)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert reads, "the wake never produced a read at all"
        # 0.25 s at a 0.05 s floor is at most ~6; with no floor the loop
        # would turn over on every one of the ~50 wakes.
        assert len(reads) <= 8, (
            f"{len(reads)} reads from ~50 wakes in 0.25s — no floor"
        )
        gaps = [b - a for a, b in zip(reads, reads[1:])]
        assert all(g >= 0.04 for g in gaps), (
            f"two reads closer than the floor: {gaps}"
        )
