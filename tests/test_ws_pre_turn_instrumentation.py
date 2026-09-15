"""The platform's pre-turn block is measured, and its independent reads overlap.

WHY THIS FILE EXISTS

The agent half of a turn has been instrumented for a while — `[PERF]
phase1_total`, `llm_ttft`, `phase3_save`, and one `[TURN_WATERFALL]` line per
turn. The PLATFORM half had nothing. Everything between the frame landing on
the socket (`ws_chat.py`'s `receive_text`) and `create_task(_agent_runner.run(
…))` — roughly seven sequential database sessions — was dark, so a slow turn
could be attributed to the model, to the save, or to "the backend", and no
measurement could tell those apart.

So: one `[PERF] ws_pre_turn` line per turn with a millisecond figure for each
block, one `[PERF] ws_ttfb` line at the first text chunk the client receives
(measured from frame receipt, not from the LLM call), and one
`[PERF] ws_proxy_dial_ms` line for the other seconds-scale wait on the path.

Three of those blocks — the automation-session lookup, the reply-to target
lookup and the stored-timezone read — are independent read-only queries on
three separate sessions, and they now run concurrently. The tests below are
about the two properties that makes risky:

  * each one must keep its OWN session (two concurrent operations on one
    AsyncSession is undefined behaviour), and
  * the ORDER THE RESULTS ARE APPLIED must not move. The automation refusal
    still decides before the reply-to pointer resolves, which still decides
    before the timezone is persisted. Only the waiting overlaps.

Several checks read the SOURCE and assert ORDER rather than mere presence.
That is deliberate: a guard whose precondition something above it destroys is
invisible to a test that only asks whether the guard is there, and this
codebase has shipped exactly that shape before.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("ENVIRONMENT", "test")

WS_CHAT_SRC = (
    Path(__file__).resolve().parent.parent / "app/api/ws_chat.py"
).read_text()
WS_PROXY_SRC = (
    Path(__file__).resolve().parent.parent / "app/api/ws_chat_proxy.py"
).read_text()
DB_SRC = (
    Path(__file__).resolve().parent.parent / "app/db/database.py"
).read_text()


# ── The stopwatch ────────────────────────────────────────────────────────

def test_pre_turn_line_names_every_block_even_the_ones_that_did_not_run(caplog):
    """"Absent" and "fast" must not look alike.

    A key that is simply omitted when its block did not run turns every zero
    into an ambiguity — the failure that made the app's `intent:judged`
    counter unreadable for two releases. Every key is always printed.
    """
    from app.api.ws_chat import _PreTurn, _PRE_TURN_KEYS

    pt = _PreTurn()
    pt.start("presave")
    pt.end("presave")
    with caplog.at_level(logging.INFO, logger="app.api.ws_chat"):
        pt.emit(channel="app", mission="chatturn:deadbeef")

    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("[PERF] ws_pre_turn")]
    assert len(lines) == 1, f"expected exactly one line, got {lines}"
    line = lines[0]
    for key in _PRE_TURN_KEYS:
        assert f"{key}_ms=" in line, f"{key}_ms missing from {line}"
    assert "total_ms=" in line
    assert "channel=app" in line and "mission=chatturn:deadbeef" in line

    # …and the key list is derived from the CODE, not restated here. The two
    # halves have opposite failure modes and both are silent:
    #   * a block timed under a name that is not a key is measured and then
    #     dropped — it reads as a block that was fast;
    #   * a key with no timer behind it prints 0 forever — it reads as a block
    #     that was fast.
    # Equality is the only assertion that catches both.
    timed = set(re.findall(r'_pt\.start\("([a-z_]+)"\)', WS_CHAT_SRC))
    timed |= set(re.findall(r'_timed\(_pt, "([a-z_]+)"', WS_CHAT_SRC))
    assert timed, "no timed blocks found in ws_chat.py at all"
    assert timed == set(_PRE_TURN_KEYS), (
        f"timed but never printed: {sorted(timed - set(_PRE_TURN_KEYS))}; "
        f"printed but never timed: {sorted(set(_PRE_TURN_KEYS) - timed)}"
    )


def test_the_line_is_single_line_key_equals_value(caplog):
    """Loki and grep split it without a parser, so no spaces inside a value
    and no newlines anywhere."""
    from app.api.ws_chat import _PreTurn

    pt = _PreTurn()
    with caplog.at_level(logging.INFO, logger="app.api.ws_chat"):
        pt.emit(channel="app", mission="chatturn:beef", user="abcd1234")
    line = [r.getMessage() for r in caplog.records
            if r.getMessage().startswith("[PERF] ws_pre_turn")][0]
    assert "\n" not in line
    body = line[len("[PERF] ws_pre_turn "):]
    for token in body.split(" "):
        assert re.fullmatch(r"[A-Za-z0-9_]+=[^ ]*", token), token


def test_a_block_that_runs_twice_is_summed_not_overwritten():
    """The pre-save's reply_to retry runs the block a second time. Reporting
    only the last pass would under-count exactly the slow case."""
    from app.api.ws_chat import _PreTurn

    pt = _PreTurn()
    for _ in range(2):
        pt.start("presave")
        pt.ms["presave"] = pt.ms.get("presave", 0)  # no-op, keeps intent clear
        pt.end("presave")
    # Two closed passes recorded; the key exists exactly once and is a sum.
    assert "presave" in pt.ms
    pt.ms["presave"] = 5
    pt.start("presave")
    pt.end("presave")
    assert pt.ms["presave"] >= 5, "second pass replaced the first instead of adding"


def test_emit_never_raises_and_never_costs_a_turn(caplog):
    """Telemetry on the hot path must fail silent. A field whose repr blows up
    may not take the user's message with it."""
    from app.api.ws_chat import _PreTurn

    class _Boom:
        def __str__(self):  # noqa: D105
            raise RuntimeError("no")
        __repr__ = __str__

    pt = _PreTurn()
    with caplog.at_level(logging.INFO, logger="app.api.ws_chat"):
        pt.emit(channel=_Boom())  # must not raise


def test_end_on_an_unopened_block_is_zero_not_an_error():
    from app.api.ws_chat import _PreTurn

    assert _PreTurn().end("never_started") == 0


# ── The day chat, resolved once ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_day_chat_is_resolved_once_per_turn(monkeypatch):
    """`get_or_create_day_chat` is a round trip even on a cache hit — it
    re-SELECTs the cached id to prove the row still exists — so asking twice
    costs a turn two of them."""
    import app.api.ws_chat as ws

    calls = []

    async def _fake(db, user_id, tz_override=None):
        calls.append(user_id)
        return "dc-1"

    monkeypatch.setattr(ws, "_resolve_day_chat_id_for_now", _fake)
    once = ws._DayChatOnce("u1", "America/Toronto")
    assert await once.get(object()) == "dc-1"
    assert await once.get(object()) == "dc-1"
    assert len(calls) == 1, f"resolved {len(calls)} times, expected 1"


@pytest.mark.asyncio
async def test_forget_re_resolves_because_a_rollback_can_un_create_the_row(monkeypatch):
    """A caller that resolved inside a transaction and then ROLLED BACK has
    un-created the DayChat it may have just inserted. Stamping a message with
    that id is an FK violation; the un-memoised path self-heals by resolving
    again, and `forget()` is what preserves that property."""
    import app.api.ws_chat as ws

    seq = iter(["dc-rolled-back", "dc-real"])

    async def _fake(db, user_id, tz_override=None):
        return next(seq)

    monkeypatch.setattr(ws, "_resolve_day_chat_id_for_now", _fake)
    once = ws._DayChatOnce("u1")
    assert await once.get(object()) == "dc-rolled-back"
    once.forget()
    assert await once.get(object()) == "dc-real"


@pytest.mark.asyncio
async def test_a_failed_resolve_is_never_memoised(monkeypatch):
    """None is the graceful-degradation answer, and it is also the FAILURE
    answer — the resolver swallows its own error and returns None. Caching it
    would hand the pre-save a guaranteed `day_chat_id=NULL`, and a row with a
    NULL day chat is invisible to the day index AND to `load_day_context`:
    the "message not in canonical history" shape of the 2026-09-14 incident.

    Before the memo, `_ensure_presave_conversation` could fail and the
    pre-save would ask again a moment later, on a session that may have
    recovered. That second chance has to survive the memo.
    """
    import app.api.ws_chat as ws

    answers = iter([None, None, "dc-recovered"])
    n = {"c": 0}

    async def _fake(db, user_id, tz_override=None):
        n["c"] += 1
        return next(answers)

    monkeypatch.setattr(ws, "_resolve_day_chat_id_for_now", _fake)
    once = ws._DayChatOnce("u1")
    assert await once.get(object()) is None
    assert await once.get(object()) is None
    assert await once.get(object()) == "dc-recovered"
    assert n["c"] == 3, "a failed resolve was cached"
    # …and once it succeeds it is memoised, as before.
    assert await once.get(object()) == "dc-recovered"
    assert n["c"] == 3


def test_the_presave_and_the_conversation_creator_share_one_memo():
    """Both day-chat sites in this file read the same `_dc_once`. If either
    goes back to calling the resolver directly, the turn pays twice again."""
    src = WS_CHAT_SRC
    assert "_dc_once = _DayChatOnce(user_id, client_tz)" in src
    assert "_presave_dc_id = await _dc_once.get(_presave_db)" in src
    assert "client_tz=client_tz, day_chat=_dc_once," in src
    # …and the creator asks the memo, not the resolver.
    _i = src.index("async def _ensure_presave_conversation")
    ensure = src[_i:src.index("# ── CONTRACTS-R31 §4.1", _i)]
    assert "_day_chat_id = await _dc.get(db_session)" in ensure
    assert "_resolve_day_chat_id_for_now(" not in ensure.split("_dc.forget()")[0], \
        "the creator resolves directly again"
    assert "_dc.forget()" in ensure, \
        "the race rollback no longer drops the memo — a stale day-chat id can " \
        "now be stamped on a message"


def test_the_hand_off_to_the_runner_is_published_and_documented():
    """`agent_runner` resolves the day chat a THIRD time and is owned
    elsewhere. The ContextVar is the seam it can adopt; it is set on every
    turn so adoption needs no change here."""
    src = WS_CHAT_SRC
    assert "CURRENT_TURN_DAY_CHAT_ID" in src
    i_set = src.index("CURRENT_TURN_DAY_CHAT_ID.set(")
    i_task = src.index("agent_task = asyncio.create_task(_agent_runner.run(")
    assert i_set < i_task, (
        "the ContextVar must be set BEFORE create_task — a task copies the "
        "context at creation, so a later set is invisible to the run"
    )


# ── The three overlapped reads ───────────────────────────────────────────

def test_the_three_independent_reads_are_gathered_when_the_flag_is_on():
    src = WS_CHAT_SRC
    i_gather = src.index("_auto_id, _reply_lookup, _tz_lookup = await asyncio.gather(")
    tail = src[i_gather:i_gather + 1200]
    for fn in ("_automation_id_for_session(session_id)",
               "_reply_to_row(_reply_candidate)",
               "_stored_user_tz(user_id)"):
        assert fn in tail, f"{fn} is no longer part of the gather"
    assert "return_exceptions=True" in tail, (
        "one failing lookup would cancel its siblings; each of these already "
        "fails soft on its own"
    )


def test_the_overlap_is_opt_in_and_off_by_default():
    """Capacity, not correctness: the tenant Postgres host is still
    `max_connections=300` with no PgBouncer `max_db_connections` (host runbook
    steps 2+3 written, NOT applied) and sat at 297 backends in 15 of 95
    samples. Overlapping adds up to 2 concurrent connections per in-flight
    turn, which is the wrong thing to add to that."""
    from app.config import Settings

    assert Settings.model_fields["ws_pre_turn_overlap"].default is False
    doc = (
        Path(__file__).resolve().parent.parent / "app/config.py"
    ).read_text()
    i = doc.index("ws_pre_turn_overlap: bool = False")
    head = doc[max(0, i - 1800):i]
    assert "WS_PRE_TURN_OVERLAP" in head, "the env var is not named"
    assert "ENABLEMENT CONDITION" in head, (
        "a default-off flag with no written condition for turning it on is a "
        "flag nobody will ever turn on"
    )
    assert "host-capacity-and-alerting" in head

    src = WS_CHAT_SRC
    assert '_overlap = bool(getattr(settings, "ws_pre_turn_overlap", False))' in src
    i_flag = src.index("_overlap = bool(getattr(settings")
    i_gather = src.index("_auto_id, _reply_lookup, _tz_lookup = await asyncio.gather(")
    assert i_flag < i_gather
    assert "if _overlap:" in src[i_flag:i_gather]


def test_with_the_flag_off_each_lookup_fires_at_the_site_that_consumes_it():
    """OFF must be the pre-existing path, not "the gather with one member".
    The reply-to query ran after the automation gate and after its own log
    line; the timezone read ran after that. Reading them early would still be
    three sequential round trips, just moved."""
    src = WS_CHAT_SRC
    i_else = src.index('                else:\n                    _auto_id = (')
    tail = src[i_else:i_else + 700]
    assert "_reply_lookup = _UNREAD" in tail and "_tz_lookup = _UNREAD" in tail, (
        "the serial path resolves the other two eagerly"
    )
    # …and each is read at its own site, behind the sentinel.
    i_reply_read = src.index("if _reply_lookup is _UNREAD:")
    i_tz_read = src.index("if client_tz and _tz_lookup is _UNREAD:")
    i_refuse = src.index('"code": "use_thread_route"')
    i_reply_log = src.index('"[WS] reply_to received target=%s')
    assert i_refuse < i_reply_log < i_reply_read < i_tz_read, (
        "the serial reads no longer sit where main's did"
    )


def test_the_unread_sentinel_is_not_none():
    """None is a legitimate lookup ANSWER — no such conversation, no users
    row. A sentinel that collided with it would re-run the query every time
    the answer was 'nothing'."""
    from app.api.ws_chat import _UNREAD

    assert _UNREAD is not None
    assert not isinstance(_UNREAD, (bool, str, int))


def test_each_gathered_lookup_opens_its_own_session():
    """Two concurrent operations on ONE AsyncSession is undefined behaviour.
    Every member of the gather takes its own short-lived session from the
    sessionmaker."""
    import app.api.ws_chat as ws

    for fn in (ws._automation_id_for_session, ws._reply_to_row, ws._stored_user_tz):
        body = inspect.getsource(fn)
        assert "async_session_maker" in body, f"{fn.__name__} has no session of its own"
        assert "async with" in body, f"{fn.__name__} does not scope its session"


def test_the_reply_to_fetch_does_not_claim_an_authorization_it_never_does():
    """It takes no `user_id` and compares nothing. The ownership decision is
    at the call site, and a docstring describing it in both places would put
    the auth boundary where only one of them enforces it."""
    import app.api.ws_chat as ws

    assert "user_id" not in inspect.signature(ws._reply_to_row).parameters
    doc = inspect.getdoc(ws._reply_to_row) or ""
    assert "no authorization" in doc.lower()
    body = inspect.getsource(ws._reply_to_row)
    assert "== user_id" not in body, "it compares an owner after all — say so"


def test_the_results_are_applied_in_the_original_order():
    """Only the WAITING overlaps. The automation refusal still precedes the
    reply-to resolution, which still precedes the timezone write — and the
    refusal `continue`s, so on a refused turn the timezone is still not
    persisted and no reply_to line is logged."""
    src = WS_CHAT_SRC
    i_gather = src.index("_auto_id, _reply_lookup, _tz_lookup = await asyncio.gather(")
    i_refuse = src.index('"code": "use_thread_route"')
    i_reply = src.index('"[WS] reply_to authorized target=%s')
    i_tz_write = src.index("_user.timezone = client_tz")
    assert i_gather < i_refuse < i_reply < i_tz_write, (
        f"order moved: gather={i_gather} refuse={i_refuse} "
        f"reply={i_reply} tz_write={i_tz_write}"
    )


def test_the_lookup_helpers_are_silent_so_the_log_order_cannot_move():
    """Every log line these feed stays at the call site. A helper that logged
    from inside the gather would reorder the pre-turn log non-deterministically
    — and would log a reply_to line on a turn the automation gate refuses."""
    import app.api.ws_chat as ws

    for fn in (ws._reply_to_row, ws._stored_user_tz):
        body = inspect.getsource(fn)
        assert "logger." not in body, f"{fn.__name__} logs from inside the gather"


def test_a_cancellation_is_not_mistaken_for_a_failed_lookup():
    """`gather(return_exceptions=True)` captures BaseException, CancelledError
    included. Treating that as "this lookup failed" turns a turn the client
    cancelled into a turn that quietly runs on with a lookup missing."""
    from app.api.ws_chat import _lookup_result

    assert _lookup_result(None) is None
    assert _lookup_result(("x", "y")) == ("x", "y")
    err = RuntimeError("pool exhausted")
    assert _lookup_result(err) is err          # an ordinary failure passes through

    with pytest.raises(asyncio.CancelledError):
        _lookup_result(asyncio.CancelledError())
    with pytest.raises(KeyboardInterrupt):
        _lookup_result(KeyboardInterrupt())


def test_the_call_sites_test_for_exception_not_baseexception():
    """`isinstance(x, BaseException)` at a call site would re-swallow exactly
    what `_lookup_result` just re-raised."""
    src = WS_CHAT_SRC
    for name in ("_auto_id", "_reply_lookup", "_tz_lookup"):
        assert f"isinstance({name}, BaseException)" not in src, (
            f"{name} is still guarded on BaseException"
        )
    assert src.count("isinstance(_auto_id, Exception)") == 1
    assert src.count("isinstance(_reply_lookup, Exception)") == 1
    assert src.count("isinstance(_tz_lookup, Exception)") == 1


@pytest.mark.asyncio
async def test_a_failed_lookup_is_returned_not_raised(monkeypatch):
    """`gather` is only safe here because neither helper can raise into it."""
    import app.api.ws_chat as ws

    class _Boom:
        def __call__(self, *a, **k):
            raise RuntimeError("pool exhausted")

    monkeypatch.setattr("app.db.database.async_session_maker", _Boom())
    assert isinstance(await ws._reply_to_row("m1"), Exception)
    assert isinstance(await ws._stored_user_tz("u1"), Exception)


async def _make_user(uid: str, tz=None):
    from app.db.database import async_session_maker
    from app.db.models import User

    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid}@t.test", hashed_password="x",
                    timezone=tz))
        await db.commit()


@pytest.mark.asyncio
async def test_no_users_row_and_a_null_timezone_are_different_answers():
    """Collapsing them would queue a rebucket and drop the tz cache for a user
    that does not exist yet — which on a freshly-bound container is every
    first message."""
    import uuid as _uuid

    from app.api.ws_chat import _stored_user_tz

    assert await _stored_user_tz(str(_uuid.uuid4())) == (False, None)

    uid = str(_uuid.uuid4())
    await _make_user(uid)
    assert await _stored_user_tz(uid) == (True, None)


@pytest.mark.asyncio
async def test_the_stored_timezone_is_read_back():
    import uuid as _uuid

    from app.api.ws_chat import _stored_user_tz

    uid = str(_uuid.uuid4())
    await _make_user(uid, "America/Toronto")
    assert await _stored_user_tz(uid) == (True, "America/Toronto")


def test_the_timezone_write_still_re_reads_the_row_it_mutates():
    """The gathered value is a READ from another session. The write must be
    made against what the DB holds now, or a concurrent update is clobbered."""
    src = WS_CHAT_SRC
    i_gate = src.index("if _user_exists and _old_tz != client_tz:")
    block = src[i_gate:i_gate + 1400]
    assert "select(User).where(User.id == user_id)" in block, \
        "the write path no longer re-reads the User row"
    assert block.index("_old_tz = _user.timezone") < block.index("_user.timezone = client_tz")


# ── Invariants the overlap must not have touched ─────────────────────────

def test_the_exactly_once_ledger_still_commits_before_the_run_starts():
    """The ledger claim is the SOLE per-message guard against a second LLM
    call on a replay. It may not be folded into anything that could roll back
    after the turn starts."""
    src = WS_CHAT_SRC
    i_commit = src.index("await _pm_db.commit()")
    i_task = src.index("agent_task = asyncio.create_task(_agent_runner.run(")
    assert i_commit < i_task
    # …and the replay branch still drops the message instead of dispatching.
    i_replay = src.index('"[WS] Duplicate dropped (durable ledger)')
    assert i_commit < i_replay < i_task


def test_the_user_row_is_still_persisted_before_the_run():
    src = WS_CHAT_SRC
    i_presave = src.index("await _presave_db.refresh(_new_msg)")
    i_task = src.index("agent_task = asyncio.create_task(_agent_runner.run(")
    assert i_presave < i_task


def test_the_pre_turn_line_is_emitted_immediately_before_the_run_task():
    """It measures the block that ENDS there. Anything between the emit and
    the create_task is time the line silently does not account for."""
    src = WS_CHAT_SRC
    i_emit = src.index("_pt.emit(")
    i_task = src.index("agent_task = asyncio.create_task(_agent_runner.run(")
    between = src[src.index(")\n", src.index("att=len(_inbound_attachments),")) + 2:i_task]
    assert i_emit < i_task
    assert not [
        l for l in between.splitlines()
        if l.strip() and not l.strip().startswith("#")
    ], f"statements sit between the emit and the run task: {between!r}"


def test_the_ttfb_line_is_latched_so_a_long_answer_logs_once():
    """`on_text_chunk` fires thousands of times on a long answer."""
    src = WS_CHAT_SRC
    i_flag = src.index('_ttfb_seen = {"done": False}')
    i_cb = src.index("async def on_text_chunk(chunk: str):")
    assert i_flag < i_cb, "the latch is declared inside the callback it must outlive"
    body = src[i_cb:i_cb + 900]
    assert 'if not _ttfb_seen["done"]:' in body
    assert '_ttfb_seen["done"] = True' in body
    assert "[PERF] ws_ttfb ttfb_ms=" in body
    assert body.index('_ttfb_seen["done"] = True') < body.index("[PERF] ws_ttfb"), (
        "the latch must be set BEFORE the log, or an exception in formatting "
        "re-arms it for every chunk"
    )


def test_the_fast_media_external_request_is_behind_the_classifier():
    """`_fast_media_check` can spend up to 8 s on a YouTube request. A message
    the play-pattern does not match must never reach it — and it does not: the
    regexes are the first thing the function does and it returns None."""
    import app.api.ws_chat as ws

    body = inspect.getsource(ws._fast_media_check)
    head = body[:body.index("import httpx")]
    assert "_PLAY_PATTERNS" in head and "return None" in head, (
        "the external request is no longer gated on the play-pattern match"
    )
    assert head.index("if not query or len(query) < 2") < head.index("logger.info")


@pytest.mark.asyncio
async def test_a_non_media_message_makes_no_request(monkeypatch):
    import app.api.ws_chat as ws

    called = {"n": 0}

    class _Client:
        def __init__(self, *a, **k):
            called["n"] += 1

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr("httpx.AsyncClient", _Client)
    assert await ws._fast_media_check(
        "what is the weather tomorrow", "u1", asyncio.Queue()
    ) is None
    assert called["n"] == 0


# ── The proxy's dial ─────────────────────────────────────────────────────

def test_the_dial_reports_its_own_duration():
    src = WS_PROXY_SRC
    assert "[PERF] ws_proxy_dial_ms=" in src
    # The eager wait breadcrumb stays exactly where it is: it has to survive
    # the early returns between it and the dial.
    assert "[PERF] ws_proxy_agent_wait_ms=%d attempts=%d outcome=%s " in src
    i_wait = src.index("[PERF] ws_proxy_agent_wait_ms=")
    i_dial_call = src.index("agent_ws, _dial_failure, _dial_client_gone = await _dial_agent(")
    assert i_wait < i_dial_call
    assert "agent_wait_ms=_wait_ms" in src, \
        "the dial line cannot correlate itself with the wait it follows"


@pytest.mark.asyncio
async def test_the_dial_line_is_emitted_on_the_failure_path_too(caplog):
    """The interesting dials are the slow ones, and a slow one usually fails."""
    from app.api import ws_chat_proxy as proxy

    class _WS:
        async def receive(self):
            await asyncio.sleep(3600)

    class _Mod:
        @staticmethod
        async def connect(*a, **k):
            raise RuntimeError("connection refused")

    with caplog.at_level(logging.INFO, logger="app.api.ws_chat_proxy"):
        got, failure, gone = await proxy._dial_agent(
            _WS(), _Mod(), "wss://x/api/ws/chat", "k", [],
            attempts=1, connect_timeout_s=0.5, user_id="abcdefgh",
            agent_wait_ms=42,
        )
    assert got is None and failure is not None and gone is False
    lines = [r.getMessage() for r in caplog.records
             if "ws_proxy_dial_ms=" in r.getMessage()]
    assert len(lines) == 1, lines
    assert "agent_wait_ms=42" in lines[0]
    assert "outcome=error" in lines[0]
    assert "user=abcdefgh" in lines[0]


# ── init_db boot phases ──────────────────────────────────────────────────

def test_init_db_publishes_its_phase_split():
    """The conftest fixture has already run init_db against this test DB."""
    from app.db.database import init_db_timings

    t = init_db_timings()
    assert t, "init_db recorded no timings at all"
    for key in ("create_all_ms", "ddl_plan_ms", "ddl_exec_ms", "reconcile_ms",
                "seeds_ms", "total_ms", "alters", "skipped"):
        assert key in t, f"{key} missing from {sorted(t)}"
    assert all(isinstance(v, (int, str)) for v in t.values()), \
        "ints and short strings only — this rides an unauthenticated endpoint"


def test_the_ddl_pass_is_split_by_statement_kind():
    """"114 ALTERs" is not an explanation for 36 s. CREATE INDEX builds an
    index, a backfill scans a table, and an ADD COLUMN takes a lock — three
    different costs with three different fixes."""
    from app.db.database import init_db_timings

    t = init_db_timings()
    for key in ("alter_ms", "index_ms", "backfill_ms", "other_ms",
                "n_alter", "n_index", "n_backfill", "n_other", "slowest_ms"):
        assert key in t, f"{key} missing from {sorted(t)}"
    assert t["n_alter"] + t["n_index"] + t["n_backfill"] + t["n_other"] == t["alters"]
    # The buckets must actually SORT. A classifier that answers "other" for
    # everything keeps the sum honest and tells you nothing, which is the
    # state this split exists to leave behind.
    assert t["n_alter"] > 0, "no ALTER TABLE statement was recognised"
    assert t["n_index"] > 0, "no CREATE INDEX statement was recognised"


def test_the_boot_lines_other_tools_grep_for_are_byte_identical():
    """Three log lines are load-bearing for the pool walk and the runbooks."""
    src = DB_SRC
    assert '_logger.warning("[init_db] alter skipped: %s — %s", stmt[:80], str(_e)[:200])' in src
    assert '_logger.warning("[init_db] seed skipped: %s — %s", stmt[:80], str(_e)[:200])' in src
    agent_main = (
        Path(__file__).resolve().parent.parent / "agent_main.py"
    ).read_text()
    assert 'print("✅ Database initialized")' in agent_main
    assert 'print("🤖 Toup Agent ready.")' in agent_main


def test_the_alembic_attempt_and_the_port_bind_were_not_reordered():
    """Both change what "healthy" means to the pool reconciler; out of scope
    for a latency round."""
    agent_main = (
        Path(__file__).resolve().parent.parent / "agent_main.py"
    ).read_text()
    i_initdb = agent_main.index("await init_db()")
    i_ready = agent_main.index('print("🤖 Toup Agent ready.")')
    assert i_initdb < i_ready
    assert "_ALEMBIC_BOOT_MARKER_ENV" in agent_main


@pytest.mark.asyncio
async def test_health_carries_the_boot_split_next_to_alembic_boot(monkeypatch):
    """So the next pool walk yields a phase split from one poll instead of log
    archaeology across 99 containers."""
    import agent_main

    async def _version():
        return None

    monkeypatch.setattr(agent_main, "read_alembic_boot_marker", lambda: "failed")
    monkeypatch.setattr(agent_main, "read_alembic_version", _version)
    monkeypatch.setattr(agent_main, "_schema_status_cache", None)
    got = await agent_main.agent_schema_status(refresh=True)
    assert got["alembic_boot"] == "failed"
    assert got["alembic_ok"] is False
    assert "init_db" in got and "total_ms" in got["init_db"]


@pytest.mark.asyncio
async def test_the_boot_split_is_read_outside_the_cached_db_query(monkeypatch):
    """The cache exists to stop a DB QUERY per health poll — the bridge polls
    ~95 containers every few seconds. A process-local dict of ints is not one,
    and keeping it outside the cache means it is never stale."""
    import agent_main

    calls = {"n": 0}

    async def _counted():
        calls["n"] += 1
        return "101"

    monkeypatch.setattr(agent_main, "read_alembic_boot_marker", lambda: "ok")
    monkeypatch.setattr(agent_main, "read_alembic_version", _counted)
    monkeypatch.setattr(agent_main, "_schema_status_cache", None)
    monkeypatch.setattr("app.db.database.init_db_timings", lambda: {"total_ms": 1})
    for _ in range(5):
        got = await agent_main.agent_schema_status()
    assert calls["n"] == 1, f"queried {calls['n']} times for 5 health polls"
    assert got["init_db"] == {"total_ms": 1}


# ── The flag reaches the process ─────────────────────────────────────────

@pytest.mark.parametrize("env,expected", [("true", True), ("false", False),
                                          ("1", True), ("0", False)])
def test_the_overlap_flag_reads_its_env_var(monkeypatch, env, expected):
    """A default-off flag nobody can turn on is worse than no flag: the code
    it guards is then dead, and the measurement it was meant to enable never
    happens."""
    from app.config import Settings

    monkeypatch.setenv("WS_PRE_TURN_OVERLAP", env)
    assert Settings().ws_pre_turn_overlap is expected


@pytest.mark.parametrize("overlap", [False, True])
def test_both_branches_of_the_flag_produce_the_same_three_results(overlap):
    """Whichever branch runs, the handler continues on exactly three names
    with exactly the same meanings. A branch that bound a different shape
    would fail far downstream, at the tz unpack or the row attribute access."""
    src = WS_CHAT_SRC
    i = src.index("                _pt.start(\"lookups\")\n                if _overlap:")
    block = src[i:src.index("_pt.end(\"lookups\")", i)]
    on, off = block.split("                else:\n", 1)
    chosen = on if overlap else off
    for name in ("_auto_id", "_reply_lookup", "_tz_lookup"):
        assert name in chosen, f"{name} is unbound on the overlap={overlap} branch"
    if overlap:
        assert "asyncio.gather(" in chosen
        assert "_lookup_result(" in chosen
    else:
        assert "asyncio.gather(" not in chosen, "the OFF branch still gathers"
        assert "_UNREAD" in chosen


# ── The dial reports what it actually did ────────────────────────────────

@pytest.mark.asyncio
async def test_the_dial_logs_attempts_MADE_not_the_configured_cap(caplog):
    """A deadline, a client disconnect or a first-attempt success all stop the
    loop early. Logging the ceiling makes every one of those read as a full
    ladder, which is the opposite of what a latency breadcrumb is for."""
    from app.api import ws_chat_proxy as proxy

    class _WS:
        async def receive(self):
            await asyncio.sleep(3600)

    class _Mod:
        @staticmethod
        async def connect(*a, **k):
            raise RuntimeError("connection refused")

    with caplog.at_level(logging.INFO, logger="app.api.ws_chat_proxy"):
        await proxy._dial_agent(
            _WS(), _Mod(), "wss://x/api/ws/chat", "k", [],
            attempts=3, connect_timeout_s=0.2, backoff_s=0.0,
            user_id="abcdefgh", agent_wait_ms=7,
        )
    line = [r.getMessage() for r in caplog.records
            if "ws_proxy_dial_ms=" in r.getMessage()][0]
    assert "attempts=3" in line, line

    # …and a deadline already spent makes ZERO attempts, which must not be
    # reported as one.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="app.api.ws_chat_proxy"):
        await proxy._dial_agent(
            _WS(), _Mod(), "wss://x/api/ws/chat", "k", [],
            attempts=6, connect_timeout_s=8.0, deadline=0.0,
            user_id="abcdefgh",
        )
    line = [r.getMessage() for r in caplog.records
            if "ws_proxy_dial_ms=" in r.getMessage()][0]
    assert "attempts=0" in line, line


@pytest.mark.asyncio
async def test_the_redial_names_itself_so_its_missing_wait_is_not_a_mystery(caplog):
    """The identity re-dial has no row lookup of its own, so `agent_wait_ms`
    is -1 there. `phase=` is what stops that reading as a broken field."""
    from app.api import ws_chat_proxy as proxy

    class _WS:
        async def receive(self):
            await asyncio.sleep(3600)

    class _Mod:
        @staticmethod
        async def connect(*a, **k):
            raise RuntimeError("nope")

    with caplog.at_level(logging.INFO, logger="app.api.ws_chat_proxy"):
        await proxy._dial_agent(
            _WS(), _Mod(), "wss://x/api/ws/chat", "k", [],
            attempts=1, connect_timeout_s=0.2, user_id="abcdefgh",
            phase="identity_redial",
        )
    line = [r.getMessage() for r in caplog.records
            if "ws_proxy_dial_ms=" in r.getMessage()][0]
    assert "phase=identity_redial" in line and "agent_wait_ms=-1" in line, line
    assert "phase=\"identity_redial\"," in WS_PROXY_SRC, \
        "the re-dial call site does not label itself"


# ── The hand-off does not outlive its turn ───────────────────────────────

def test_the_day_chat_contextvar_is_reset_after_the_task_is_created():
    """The handler's context outlives the turn and is shared by every later
    iteration of the receive loop. A value left set is visible to anything
    spawned afterwards — including a turn that `continue`s out before ever
    setting it — so the previous turn's day chat would be handed to a later
    one. That is the exact class of bug this hand-off exists to help fix."""
    src = WS_CHAT_SRC
    i_set = src.index("_dc_ctx_token = CURRENT_TURN_DAY_CHAT_ID.set(")
    i_task = src.index("agent_task = asyncio.create_task(_agent_runner.run(")
    i_reset = src.index("CURRENT_TURN_DAY_CHAT_ID.reset(_dc_ctx_token)")
    assert i_set < i_task < i_reset, (
        f"set={i_set} task={i_task} reset={i_reset} — the reset must come "
        "AFTER create_task (the task copies the context) and the set BEFORE it"
    )
    between = src[src.index("\n", i_task + 40):i_reset]
    assert "await " not in between, (
        "something awaits between the task creation and the reset; anything "
        "scheduled there would inherit the value"
    )


def test_the_reset_survives_a_token_from_another_context():
    """`ContextVar.reset` raises ValueError for a token created in a different
    Context. It cannot happen on this path today, but a telemetry hand-off may
    not be the thing that kills a turn."""
    src = WS_CHAT_SRC
    i_reset = src.index("CURRENT_TURN_DAY_CHAT_ID.reset(_dc_ctx_token)")
    tail = src[i_reset:i_reset + 260]
    assert "except ValueError" in tail
    assert "CURRENT_TURN_DAY_CHAT_ID.set(None)" in tail


def test_the_contextvar_default_is_none_so_an_unset_turn_reads_as_absent():
    from app.api.ws_chat import CURRENT_TURN_DAY_CHAT_ID

    assert CURRENT_TURN_DAY_CHAT_ID.get() is None
