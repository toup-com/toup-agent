"""A turn the client cannot NAME is a turn the client cannot end.

Round 46, incident 2 (2026-09-15). The in-app "Hi" was answered at 14:50:44.7
and the app kept "Waking your agent…" and the stop control up for ~22 s. The
answer had been delivered: the turn's own socket was gone, so `ws_chat`
broadcast it to the reconnected one as `{type:'message'}` — a frame carrying no
`client_msg_id`, no `channel`, no `session_id`, and followed by no `done` at
all. `turn_ended`, the authoritative end the server DOES emit for a socket-less
turn, carried only a mission id. The client had nothing to match on and fell
back to a 12 s silence heuristic, twice.

This file pins the wire shapes that fix it, and the two module-level
mechanisms behind them — the duplicate ack's `running`, and the accept→register
replay window. It deliberately does NOT open a database: the identity columns
and `turn_receipt`'s ledger read are covered by the agent-mode test beside it.

Local run (from backend/):
    RUN_MODE=platform PYTHONPATH=. pytest tests/test_ws_turn_identity.py \
      -q -p no:cacheprovider
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
USER = "00000000-0000-4000-8000-000000000001"  # synthetic (IMPL_RULES §5)


def ws():
    # A plain import, never `importorskip`: FIRST-PARTY code that fails to
    # import must be a RED sweep, not silent skips reading green.
    from app.api import ws_chat as mod

    return mod


@pytest.fixture(autouse=True)
def _clean():
    m = ws()
    m._active_turns.clear()
    m._recent_broadcasts.clear()
    m._user_ws_queues.clear()
    yield
    m._active_turns.clear()
    m._recent_broadcasts.clear()
    m._user_ws_queues.clear()


# ── Every turn frame carries the turn's identity ─────────────────────────

def test_turn_frames_carry_client_msg_id():
    m = ws()
    m._set_active_turn(USER, mission_id="chatturn:abc", title="t", stage="thinking",
                       started_at=__import__("time").time(), client_msg_id="cm-1")
    frame = m._turn_frame("turn_active", m._get_active_turn(USER))
    assert frame["client_msg_id"] == "cm-1"
    assert frame["mission_id"] == "chatturn:abc"


def test_a_turn_with_no_client_msg_id_omits_the_key_rather_than_sending_null():
    """A channel turn (WhatsApp, a routine) has no client id. Sending
    `client_msg_id: null` would make an app's `=== null` comparison match every
    such frame."""
    m = ws()
    m._set_active_turn(USER, mission_id="chatturn:abc", stage="thinking",
                       started_at=__import__("time").time())
    assert "client_msg_id" not in m._turn_frame("turn_status", m._get_active_turn(USER))


# ── The duplicate ack answers the question the client actually has ───────

def _ack(m, *a, answered=None, **kw):
    """Run the real `_duplicate_ack` with its ONE database question answered
    by `answered` (the assistant row's id, or None for "no such row")."""
    async def _fake(_uid, _cmid):
        return answered

    old = m._answered_message_id
    m._answered_message_id = _fake
    try:
        return asyncio.run(m._duplicate_ack(*a, **kw))
    finally:
        m._answered_message_id = old


def test_duplicate_ack_reports_running_when_the_turn_is_in_flight():
    m = ws()
    m._set_active_turn(USER, mission_id="chatturn:abc", stage="thinking",
                       started_at=__import__("time").time(), client_msg_id="cm-1")
    ack = _ack(m, USER, "cm-1", "sess-1")
    assert ack["type"] == "user_message_persisted"
    assert ack["duplicate"] is True
    assert ack["running"] is True
    assert ack["mission_id"] == "chatturn:abc"


def test_duplicate_ack_reports_not_running_when_the_answer_IS_in_history():
    """THE 15 Sep FRAME. The agent drops the re-send because it already ran the
    message — and until this round it said only that, which proves liveness and
    settles nothing. `running: false` is the client's cue that its answer is
    already in history, so it is sent only when an assistant row proves it."""
    m = ws()
    ack = _ack(m, USER, "cm-1", "sess-1", answered="asst-1")
    assert ack["running"] is False
    assert ack["mission_id"] is None


def test_an_UNPROVEN_completion_omits_running_rather_than_claiming_false():
    """`_turn_running` answers False in two different situations — nothing in
    the registry, and an entry belonging to a DIFFERENT turn, because the
    registry holds ONE entry per user and a new turn replaces it. Only one of
    them means "finished". The client settles the turn and calls `onDropped`
    on `running: false`, so guessing it declares a turn that has not started
    over, with no answer in history. Absent ⇒ it falls back to
    `resume`/`turn_receipt`."""
    m = ws()
    ack = _ack(m, USER, "cm-1", "sess-1", answered=None)
    assert "running" not in ack, ack


def test_a_lookup_that_RAISES_omits_running_too():
    """The proof is a database read, and the database may be exactly what is
    broken — the 2026-09-15 shape. Unprovable is not false."""
    m = ws()

    async def _boom(_uid, _cmid):
        raise RuntimeError("pgbouncer is down")

    old = m._answered_message_id
    m._answered_message_id = _boom
    try:
        ack = asyncio.run(m._duplicate_ack(USER, "cm-1", "sess-1"))
    finally:
        m._answered_message_id = old
    assert "running" not in ack, ack


def test_a_different_turn_in_flight_is_not_evidence_about_this_one():
    m = ws()
    m._set_active_turn(USER, mission_id="chatturn:other", stage="thinking",
                       started_at=__import__("time").time(), client_msg_id="cm-OTHER")
    assert "running" not in _ack(m, USER, "cm-1", "s", answered=None)


def test_the_ack_carries_no_content():
    m = ws()
    ack = _ack(m, USER, "cm-1", "sess-1", answered="asst-1",
               server_msg_id="sid", day_chat_id="dc")
    assert set(ack) <= {"type", "client_msg_id", "session_id", "duplicate", "running",
                        "mission_id", "server_msg_id", "day_chat_id"}
    assert "text" not in ack and "content" not in ack


# ── The accept → register frame-loss window ──────────────────────────────

def test_a_frame_broadcast_before_registration_is_replayed():
    """On pool-82 the window was 1.6 s wide (accept 14:50:29.216,
    "Authenticated" 14:50:30.804) and a [BROADCAST] landed inside it — dropped,
    with a `queues=N sent=N` count that looked perfectly correct."""
    m = ws()
    accepted_at = __import__("time").monotonic()
    asyncio.run(m.broadcast_to_user(USER, {"type": "message", "id": "m1", "content": "hi"}))
    q: asyncio.Queue = asyncio.Queue(maxsize=10)
    replayed = m._register_ws_queue(USER, q, accepted_at)
    assert replayed == 1
    frame = q.get_nowait()
    assert frame["id"] == "m1"
    assert frame["replay"] is True, "a replayed frame must be marked, so a client can dedupe it"


def test_a_frame_broadcast_before_this_socket_was_accepted_is_NOT_replayed():
    m = ws()
    asyncio.run(m.broadcast_to_user(USER, {"type": "message", "id": "old"}))
    later = __import__("time").monotonic() + 1.0
    q: asyncio.Queue = asyncio.Queue(maxsize=10)
    assert m._register_ws_queue(USER, q, later) == 0


def test_narration_frames_are_never_replayed():
    """`turn_active`/`turn_status` are re-announced at registration from the
    live registry. Replaying a stale one would repaint a FINISHED turn as
    running — the opposite of the bug being fixed."""
    m = ws()
    accepted_at = __import__("time").monotonic()
    for t in ("turn_active", "turn_status", "job_update", "radio_state"):
        asyncio.run(m.broadcast_to_user(USER, {"type": t}))
    q: asyncio.Queue = asyncio.Queue(maxsize=10)
    assert m._register_ws_queue(USER, q, accepted_at) == 0


def test_the_replay_buffer_is_bounded_and_dropped_with_the_last_socket():
    m = ws()
    for i in range(200):
        asyncio.run(m.broadcast_to_user(USER, {"type": "message", "id": str(i)}))
    assert len(m._recent_broadcasts[USER]) <= m._REPLAY_MAX_PER_USER
    q: asyncio.Queue = asyncio.Queue(maxsize=10)
    m._register_ws_queue(USER, q)
    m._unregister_ws_queue(USER, q)
    assert USER not in m._recent_broadcasts


def test_registration_without_an_accept_time_replays_nothing():
    """Every other caller of `_register_ws_queue` is unchanged, and an
    unbounded replay is not a default anyone should get by accident."""
    m = ws()
    asyncio.run(m.broadcast_to_user(USER, {"type": "message", "id": "m1"}))
    q: asyncio.Queue = asyncio.Queue(maxsize=10)
    assert m._register_ws_queue(USER, q) == 0


# ── The late answer is a real ending ─────────────────────────────────────

def _late_block() -> str:
    """The late-answer lane, bounded by its own end anchor rather than a byte
    count — a slice sized in characters silently loses its tail the moment
    anything above it grows."""
    src = (BACKEND / "app/api/ws_chat.py").read_text()
    i = src.index("_late_frame = {")
    j = src.index("late answer delivered to", i)
    return src[i:j]


def test_the_late_frame_is_pairable():
    block = _late_block()
    for key in ('"client_msg_id": _client_msg_id_top', '"channel": channel',
                '"session_id": response.session_id', '"late": True'):
        assert key in block, f"{key} missing from the late answer frame"


def test_a_done_is_NEVER_broadcast_to_another_socket():
    """`done` is the one terminal frame every shipped client settles
    UNCONDITIONALLY — builds 123-126 and this round's build both run
    `case 'done': onDone(); _settleTurn()` with no identity check. Broadcast to
    the user's OTHER sockets it ends whatever DIFFERENT turn is pending there
    and renders this turn's text into it: turn A (a 70 s edit_image) loses its
    socket on backgrounding, the user foregrounds and sends B, A finishes, and
    B is answered with A's reply. The end-of-turn for a socket-less turn is
    `turn_ended`, which carries client_msg_id and is settled by identity."""
    src = (BACKEND / "app/api/ws_chat.py").read_text()
    assert "_done_payload" in src, "anchor moved — re-read this test"
    assert '{**_done_payload' not in src, (
        "a `done` is being mirrored to other sockets again"
    )
    # …and not written as a fresh dict literal either. The round's own
    # published contract (C1) told an implementer to write exactly that, so
    # forbidding ONE spelling of the same frame is not a guard, it is a
    # coincidence. Every `broadcast_to_user(` call in the file must be free of
    # a `"type": "done"` payload.
    for m in re.finditer(r"broadcast_to_user\(", src):
        window = src[m.start():m.start() + 700]
        assert not re.search(r'"type":\s*"done"', window), (
            "a `done` is being broadcast again:\n" + window[:400]
        )
    # Executed, not read: the replay buffer is the second way a stale `done`
    # reaches a socket that has moved on.
    assert "done" not in ws()._REPLAYABLE_TYPES


def test_an_error_is_never_replayed_onto_another_socket():
    """A fault is a property of the SOCKET it happened on. The new client's
    `case 'error'` else-branch calls `onError` and settles, so replaying one
    into a socket that has since started a different turn ends that turn with
    a stranger's failure. Nothing broadcasts one today — which is exactly when
    to take it out of the ring, not after a producer appears."""
    assert "error" not in ws()._REPLAYABLE_TYPES


def test_the_replay_ring_evicts_by_age_not_only_by_count():
    """Entries are whole message BODIES, and the ring is only dropped when the
    user's LAST queue unregisters. Without age eviction a user with no live
    socket kept up to 16 of them resident for the life of the process — and
    `_replay_missed` would not serve them anyway."""
    import time as _time

    m = ws()
    m._recent_broadcasts.pop(USER, None)
    try:
        m._remember_broadcast(USER, {"type": "message", "content": "old"})
        ring = m._recent_broadcasts[USER]
        assert len(ring) == 1
        # Age the entry past the window without sleeping through it.
        ring[0] = (_time.monotonic() - (m._REPLAY_WINDOW_S + 1.0), ring[0][1])
        m._remember_broadcast(USER, {"type": "message", "content": "new"})
        assert [e[1]["content"] for e in ring] == ["new"], list(ring)
    finally:
        m._recent_broadcasts.pop(USER, None)


def test_the_late_message_excludes_the_turns_own_socket():
    """A socket that is both owner and mirror would double-settle."""
    block = _late_block()
    i = block.index("_late_sent = await broadcast_to_user(")
    assert "exclude=broadcast_queue" in block[i:i + 200], block[i:i + 200]


def test_turn_ended_carries_the_identity():
    # Anchored on one stable token, never on formatted indentation: a cosmetic
    # reindent must not turn this into a ValueError.
    src = (BACKEND / "app/api/ws_chat.py").read_text()
    hits = [mm.start() for mm in re.finditer(r'"type":\s*"turn_ended"', src)]
    assert hits, "turn_ended is gone"
    assert any(
        '"client_msg_id": _client_msg_id_top' in src[i:i + 600] for i in hits
    ), "no turn_ended carries the turn's identity"


def test_the_status_ack_carries_the_identity():
    src = (BACKEND / "app/api/ws_chat.py").read_text()
    hits = [mm.start() for mm in re.finditer(r'"type":\s*"status",\s*"stage":\s*"received"', src)]
    assert hits, "the received ack is gone"
    assert any(
        '"client_msg_id": _client_msg_id_top' in src[i:i + 500] for i in hits
    ), "the received ack does not carry the turn's identity"


def test_every_status_stage_carries_the_identity():
    """C1 names `status` among the frames that carry `client_msg_id`; the
    `received` ack did and the `on_status` producer for every later stage
    (thinking, tool stages) did not — so a future identity check on `status`
    would answer "unknown" for every stage but the first."""
    src = (BACKEND / "app/api/ws_chat.py").read_text()
    hits = [mm.start() for mm in re.finditer(r'"type":\s*"status",\s*"stage":\s*stage,', src)]
    assert hits, "the on_status producer is gone"
    assert all(
        '"client_msg_id": _client_msg_id_top' in src[i:i + 500] for i in hits
    ), "a status stage does not carry the turn's identity"


# ── resume → turn_receipt ────────────────────────────────────────────────

def test_the_resume_branch_exists_and_answers_without_an_id():
    src = (BACKEND / "app/api/ws_chat.py").read_text()
    i = src.index('if msg_type == "resume":')
    block = src[i:i + 1400]
    assert '"type": "turn_receipt"' in block
    assert '"status": "unknown"' in block, "a resume with no id must be answered, not ignored"
    assert "continue" in block


def test_a_running_turn_is_reported_without_touching_the_database():
    """The in-process registry is checked FIRST: the common case (the user
    reconnects while their turn is still running) must not depend on a DB that
    may be exactly what is broken."""
    m = ws()
    m._set_active_turn(USER, mission_id="chatturn:abc", stage="thinking",
                       started_at=__import__("time").time(), client_msg_id="cm-1")
    receipt = asyncio.run(m._turn_receipt(USER, "cm-1"))
    assert receipt == {"type": "turn_receipt", "client_msg_id": "cm-1",
                       "status": "running", "mission_id": "chatturn:abc"}


def test_an_unprovable_receipt_is_unknown_never_failed():
    """With no database (this test has none reachable), the lookup raises. A
    receipt we cannot prove must be `unknown`: telling a client its turn FAILED
    because our DB is down is incident 3 in a different costume."""
    m = ws()
    receipt = asyncio.run(m._turn_receipt(USER, "cm-nope"))
    assert receipt["status"] == "unknown"
    assert receipt["client_msg_id"] == "cm-nope"
    assert "failed" not in str(receipt.get("status"))


def test_nothing_in_this_round_emits_a_failed_receipt():
    """`failed` stays in the vocabulary for a producer that can prove a
    failure. Guessing it — from "claimed, not running, no answer row" — would
    also describe every turn answered before this image shipped."""
    src = (BACKEND / "app/api/ws_chat.py").read_text()
    i = src.index("async def _turn_receipt")
    body = src[i:src.index("\n\ndef ", i + 10)]
    # Match the VALUE, not one assignment form: `receipt.update({"status":
    # "failed"})` and `return {..., "status": "failed"}` contain neither of
    # the two literals this used to forbid. The docstring above mentions
    # `failed` in backticks, which this deliberately still allows.
    assert not re.search(r"""["']failed["']""", body), body


def test_turns_completed_counter_exists_and_counts_only_ends():
    m = ws()
    before = m.turns_completed()
    m._set_active_turn(USER, mission_id="chatturn:x", stage="thinking",
                       started_at=__import__("time").time())
    assert m.turns_completed() == before, "starting a turn is not completing one"
    m._clear_active_turn(USER, "chatturn:x")
    assert m.turns_completed() == before + 1


def test_the_counter_is_an_int_and_carries_no_identity():
    m = ws()
    assert isinstance(m.turns_completed(), int)
    src = (BACKEND / "app/api/ws_chat.py").read_text()
    i = src.index("_turns_completed: int = 0")
    assert re.search(r"A count, never an id", src[i - 400:i])


# ── The id is client-controlled input ────────────────────────────────────

def test_the_client_msg_id_is_bounded_at_the_dispatch_head():
    """`messages.client_msg_id` is VARCHAR(100) and the value comes straight
    off the wire. Unbounded it surfaces as a DataError inside the pre-save —
    i.e. as a failed turn — for what is really a malformed field. Over-long is
    treated as ABSENT (the pre-identity path), never as a refusal."""
    m = ws()
    assert m._MAX_CLIENT_MSG_ID_LEN == 100
    src = (BACKEND / "app/api/ws_chat.py").read_text()
    i = src.index('_client_msg_id_top = msg.get("client_msg_id")')
    j = src.index("if _client_msg_id_top and not _is_system_action:", i)
    guard = src[i:j]
    assert "_MAX_CLIENT_MSG_ID_LEN" in guard, guard
    assert "_client_msg_id_top = None" in guard, guard
    assert 'msg.pop("client_msg_id"' in guard, (
        "the pre-save reads msg['client_msg_id'] again further down — clearing "
        "only the local binding leaves the over-long value on the INSERT"
    )
