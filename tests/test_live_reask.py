"""Reask with identity (v0.3 §F, feature `reask_turns`).

The shipped relay answered every `inject_text` with two provider writes: the
client's text as an unlabelled `session.thinking.append`, and a STANDING
instruction naming no turn — "The caller's last request is repeated in your
notes and has not been answered. Answer it now, briefly." Nothing checked
whether the words were stale, answered, already running or already re-asked,
nothing tied the reply to a turn, and nothing was logged. The V3 recording is
that defect end to end: an old stop-music fragment («قطع») was replayed after a
newer, unrelated English question, and the model later "answered" it in
Persian.

The repaired handler accepts only the NEWEST closed user turn that nothing has
answered, that owns no task, that was not re-asked before and that no newer
turn has superseded. An accepted reask is one tracked instruction that QUOTES
the relay's own words for that turn and scopes itself to them, its event id is
idempotent, and the reply is parented to the re-asked turn. Legacy frames (TF132,
text only) resolve by normalized text under the same fences.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

import test_live_harness as H


REASK_FEATURES = [
    "live_turns", "state_seq", "outcomes", "delegation_frames",
    "task_lifecycle", "playback_frames", "turn_timing", "media_control",
    "reask_turns",
]

#: The identity-free sentence the shipped relay sent for every inject_text.
IDENTITY_FREE = "has not been answered"


async def _wait_until(predicate, *, timeout=3.0, label="condition"):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError(f"timed out waiting for {label}")
        await asyncio.sleep(0.01)


def _finals(client) -> list[dict]:
    return [f for f in client.of("transcript") if f.get("final")]


def _send(client, frame) -> None:
    """Queue `frame` as the next thing the phone sends.

    The fake client's script is static; a turn id only exists once the relay
    has minted it, so a frame that names one is inserted at run time."""

    client._script.insert(0, frame)


def _reask_appends(provider) -> list[dict]:
    return [
        e for e in provider.of("session.instructions.append")
        if str(e.get("event_id") or "").startswith("toup-live-reask-")
    ]


def _no_identity_free_note(provider) -> None:
    for event in provider.of("session.instructions.append"):
        assert IDENTITY_FREE not in str(event.get("content") or ""), event
        assert "repeated in your notes" not in str(event.get("content") or ""), event


def _config_with(*features, **extra):
    frame = H.config(**extra)
    frame["features"] = list(frame["features"]) + list(features)
    return frame


# ── the legacy (TF132) frame: text only ──────────────────────────────


@pytest.mark.asyncio
async def test_legacy_reask_quotes_the_relays_own_words_once_and_scopes_itself(monkeypatch):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    question = "What is the best professor in UofT working on LLM?"

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta(question, 0, 900))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def u1_final():
        await _wait_until(lambda: _finals(client), label="U1 final")

    client = H.FakeClient([
        H.config(),                      # a v0.2 build: reask_turns NOT negotiated
        u1_final,
        {"type": "inject_text", "text": question},
        0.15,
        {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=6, db_session_id="db-reask-legacy")

    u1 = _finals(client)[0]["turn_id"]
    _no_identity_free_note(provider)
    # The words are never dropped into the notes as an unlabelled "request".
    assert not any(question in str(e.get("content")) for e in provider.of("session.thinking.append"))
    reasks = _reask_appends(provider)
    assert len(reasks) == 1, reasks
    assert reasks[0]["event_id"] == f"toup-live-reask-{u1}"
    content = reasks[0]["content"]
    assert f"«{question}»" in content
    assert "earlier request" in content          # self-scoping clause
    # A legacy client negotiated nothing, so it gets no new frame type.
    assert client.of("reask_result") == []


@pytest.mark.asyncio
async def test_a_stale_stop_fragment_is_never_reasked_after_a_newer_turn(monkeypatch):
    """V3: «قطع» (stop the music) was replayed after the caller had already
    moved on to an unrelated English question. The newer turn supersedes it."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    box = {}
    stop_words = "آهنگ رو قطع کن"
    question = "What is the best professor in UofT working on LLM?"

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta(stop_words, 0, 300))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def newer_turn():
        await _wait_until(lambda: len(_finals(client)) >= 1, label="U1 final")
        box["p"].push(H.user_delta(question, 1500, 2400))
        await _wait_until(lambda: len(_finals(client)) >= 2, label="U2 final")

    client = H.FakeClient([
        H.config(),
        newer_turn,
        {"type": "inject_text", "text": stop_words},     # the stale replay
        0.1,
        {"type": "inject_text", "text": stop_words},     # …and again
        0.1,
        {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=6, db_session_id="db-reask-stale")

    _no_identity_free_note(provider)
    for kind in ("session.thinking.append", "session.instructions.append"):
        for event in provider.of(kind):
            assert stop_words not in str(event.get("content")), event
    assert _reask_appends(provider) == []


# ── the negotiated frame: reask_of_user_turn_id ──────────────────────


@pytest.mark.asyncio
async def test_identified_reask_fences_and_answers_with_reask_result(monkeypatch):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    box = {}
    question = "What is the best professor in UofT working on LLM?"

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("قطع", 0, 300))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def two_turns():
        await _wait_until(lambda: len(_finals(client)) >= 1, label="U1 final")
        box["p"].push(H.user_delta(question, 1500, 2400))
        await _wait_until(lambda: len(_finals(client)) >= 2, label="U2 final")

    def reask(index, text):
        def _go():
            _send(client, {
                "type": "inject_text", "text": text, "reason": "no_response",
                "reask_of_user_turn_id": _finals(client)[index]["turn_id"],
            })
        return _go

    client = H.FakeClient([
        _config_with("reask_turns"),
        two_turns,
        reask(0, "قطع"),                  # superseded by U2
        0.1,
        reask(1, "LLM"),                  # the app's FRAGMENT of U2
        0.1,
        reask(1, "LLM"),                  # a retry of the same turn
        0.1,
        {"type": "inject_text", "text": "x", "reask_of_user_turn_id": "live-utt:nope:99"},
        0.1,
        {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=6, db_session_id="db-reask-id")

    u1, u2 = (f["turn_id"] for f in _finals(client)[:2])
    results = [(f["user_turn_id"], f["outcome"]) for f in client.of("reask_result")]
    assert results == [
        (u1, "stale"),
        (u2, "accepted"),
        (u2, "duplicate"),
        ("live-utt:nope:99", "unknown"),
    ], results
    _no_identity_free_note(provider)
    reasks = _reask_appends(provider)
    assert len(reasks) == 1
    assert reasks[0]["event_id"] == f"toup-live-reask-{u2}"
    # The RELAY's own full words for the turn, never the client's fragment.
    assert f"«{question}»" in reasks[0]["content"]
    assert "قطع" not in reasks[0]["content"]


@pytest.mark.asyncio
async def test_a_turn_that_owns_a_task_or_was_answered_is_not_reasked(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    box = {}

    async def think(_user_id, _task, _session_id, **_kwargs):
        await asyncio.sleep(1.2)
        return "Sunny.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("look up the weather", 0, 400))
            provider.push(H.delegation("d-weather", 450))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def running_then_answered():
        await _wait_until(
            lambda: any(f.get("state") == "running" for f in client.of("delegation")),
            label="task running",
        )
        _send(client, {
            "type": "inject_text", "text": "look up the weather",
            "reask_of_user_turn_id": _finals(client)[0]["turn_id"],
        })

    async def direct_answer():
        box["p"].push(H.user_delta("hello there", 3000, 3400))
        await _wait_until(lambda: len(_finals(client)) >= 2, label="U2 final")
        box["p"].push(H.out_text("Hi!", 3600, 3800))
        box["p"].push(H.out_audio())
        await _wait_until(
            lambda: any(f.get("parent_user_turn_id") == _finals(client)[1]["turn_id"]
                        for f in client.of("response_text")),
            label="direct answer",
        )
        await _wait_until(lambda: client.of("audio_delta"), label="direct answer audio")
        eid = client.of("audio_delta")[-1]["response_id"]
        reask = {
            "type": "inject_text", "text": "hello there",
            "reask_of_user_turn_id": _finals(client)[1]["turn_id"],
        }
        # Supervisor 2026-09-23 18:16: a reply with NO playback receipt is not
        # proof the caller heard it (UNKNOWN), so it no longer makes this turn
        # `answered` — the old pin expected "answered" here with no receipt.
        # New verdict for the no-receipt ask: `stale` (not_a_request — "hello
        # there" asks nothing, so it is never re-asked either way).  Once the
        # phone's `playback_idle` receipt says the reply WAS heard, the same
        # ask is `answered` (HEARD), which is what this pin is about.
        client._script[0:0] = [
            reask, 0.1,
            {"type": "playback_idle", "response_id": eid, "item_id": eid, "played_ms": 400},
            0.1, reask,
        ]

    client = H.FakeClient([
        _config_with("reask_turns"),
        running_then_answered,
        0.1,
        direct_answer,
        0.2,
        {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id="db-reask-fences")

    outcomes = [f["outcome"] for f in client.of("reask_result")]
    assert outcomes == ["in_flight", "stale", "answered"], outcomes
    assert _reask_appends(provider) == []
    _no_identity_free_note(provider)


@pytest.mark.asyncio
async def test_a_reask_while_a_newer_turn_is_still_open_is_in_flight(monkeypatch):
    """Pin changed by R2 addendum 5 A3 (was `…_is_stale`, expecting
    ["stale"]): an OPEN caller turn is UNJUDGED whatever its first words are —
    it supersedes only once it closes as a request (addendum 4 §4.1).  The
    old `stale` on "and then find me" was the same verdict that staled R on an
    open "can you" / «صدامو» / "thank you for" that then closed as a presence
    check, losing R for good.  Still nothing is re-asked now."""

    H.fast_clocks(
        monkeypatch, voice_live_utterance_gap_ms=3000, voice_live_utterance_hard_gap_ms=6000,
    )
    H.patch_relay(monkeypatch)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("stop the music", 0, 400))
            # Far enough past the gap to close U1 on arrival; U2 stays open.
            provider.push(H.user_delta("and then find me", 4000, 4400))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def u1_closed_u2_open():
        await _wait_until(lambda: _finals(client), label="U1 final")
        await asyncio.sleep(0.05)
        _send(client, {
            "type": "inject_text", "text": "stop the music",
            "reask_of_user_turn_id": _finals(client)[0]["turn_id"],
        })

    client = H.FakeClient([
        _config_with("reask_turns"), u1_closed_u2_open, 0.1, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=6, db_session_id="db-reask-open")

    assert [f["outcome"] for f in client.of("reask_result")] == ["in_flight"]
    assert _reask_appends(provider) == []


# ── the reply belongs to the re-asked turn ───────────────────────────


@pytest.mark.asyncio
async def test_a_reply_lost_in_the_interrupt_window_is_reasked_and_parented(monkeypatch):
    """The V3 U:88 shape: the model's reply to a turn is generated inside the
    post-interrupt suppression window and dropped, so the caller hears
    nothing — yet the relay had already fenced that turn as answered by direct
    output. The reask must be accepted (no epoch ever reached the phone) and
    its reply must carry the re-asked turn as its parent, even when the
    provider sends untimed audio before the timed transcript."""

    H.fast_clocks(
        monkeypatch, voice_live_interrupt_suppress_ms=400, voice_live_output_epoch_gap_ms=80,
    )
    H.patch_relay(monkeypatch)
    box = {}
    question = "what time is it"

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.out_text("Hello there.", 0, 300))
            provider.push(H.out_audio())
        elif (
            event["type"] == "session.instructions.append"
            and str(event.get("event_id") or "").startswith("toup-live-reask-")
        ):
            provider.push(H.out_audio("REASKED"))                 # untimed first
            provider.push(H.out_text("It is noon.", 1500, 1800))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def speak_then_barge_in():
        await _wait_until(lambda: client.of("response_text"), label="greeting")
        box["p"].push(H.user_delta(question, 500, 900))
        await _wait_until(
            lambda: any(f.get("text") == question for f in client.of("transcript")),
            label="user words seen",
        )
        _send(client, {"type": "interrupt"})

    async def lost_reply():
        await _wait_until(lambda: _finals(client), label="U1 final")
        # The answer starts before the resume point: dropped by suppression.
        box["p"].push(H.out_text("It is noon.", 600, 850))
        box["p"].push(H.out_audio())
        await asyncio.sleep(0.6)                                  # fence lifts

    client = H.FakeClient([
        H.config(),                                               # TF132 shape
        speak_then_barge_in,
        lost_reply,
        {"type": "inject_text", "text": question},
        0.4,
        {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=8, db_session_id="db-reask-parent")

    u1 = _finals(client)[0]["turn_id"]
    replies = [f for f in client.of("response_text") if "noon" in f.get("text", "")]
    assert replies, "the reask produced no audible reply"
    assert all(f.get("parent_user_turn_id") == u1 for f in replies), replies
    assert len(_reask_appends(provider)) == 1


# ── what is NOT a reask ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_app_authored_note_is_forwarded_once_and_quoted(monkeypatch):
    """The enrollment greeting and the web onboarding's `[COLOR_SELECTED: …]`
    also ride `inject_text`. They are app-authored notes — wholly bracketed,
    which speech transcripts never are — not a request of the caller's, so
    they are delivered as one labelled, self-scoping instruction and are
    never called an unanswered request."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    note = (
        "[The user has just finished teaching you their voice, so you now respond "
        "to them and to nobody else. Greet them in one short warm sentence, "
        "mention that briefly, and ask what they would like help with.]"
    )

    def on_send(provider, event):
        if event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.05, {"type": "inject_text", "text": note}, 0.1, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=6, db_session_id="db-reask-note")

    _no_identity_free_note(provider)
    assert not any(note in str(e.get("content")) for e in provider.of("session.thinking.append"))
    notes = [e for e in provider.of("session.instructions.append") if note in str(e.get("content"))]
    assert len(notes) == 1, notes
    assert f"«{note}»" in notes[0]["content"]
    assert "earlier request" in notes[0]["content"]


# ── language of the note ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_reask_instruction_is_written_in_the_delivery_language(monkeypatch):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    box = {}
    persian = "یک استاد خوب در دانشگاه تورنتو پیدا کن"

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta(persian, 0, 600))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def u1_final():
        await _wait_until(lambda: _finals(client), label="U1 final")

    client = H.FakeClient([
        _config_with("reask_turns"), u1_final,
        {"type": "inject_text", "text": persian}, 0.1, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=6, db_session_id="db-reask-fa")

    reasks = _reask_appends(provider)
    assert len(reasks) == 1
    content = reasks[0]["content"]
    assert f"«{persian}»" in content
    # Localised: an English instruction is the commonest way a Persian call
    # ends up answered in English.
    outside_quote = content.replace(f"«{persian}»", "")
    assert "فارسی" in outside_quote
    assert "Reply" not in outside_quote


# ── F4: causality you can read in the fleet logs, never the words ────


@pytest.mark.asyncio
async def test_reask_and_epoch_logs_carry_turns_and_codes_never_words(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)
    box = {}
    words = "یک استاد خوب پیدا کن"
    answer = "باشه، الان می‌گردم."

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta(words, 0, 600))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def answered_then_reasked():
        await _wait_until(lambda: _finals(client), label="U1 final")
        box["p"].push(H.out_text(answer, 800, 1200))
        box["p"].push(H.out_audio())
        await _wait_until(
            lambda: any("output epoch closed" in r.getMessage() for r in caplog.records),
            label="epoch closed",
        )
        _send(client, {
            "type": "inject_text", "text": words,
            "reask_of_user_turn_id": _finals(client)[0]["turn_id"],
        })

    client = H.FakeClient([
        _config_with("reask_turns"), answered_then_reasked, 0.1, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=6, db_session_id="db-reask-logs")

    u1 = _finals(client)[0]["turn_id"]
    messages = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[LIVE]")]
    assert any(
        m.startswith("[LIVE] output epoch opened") and f"parent={u1}" in m for m in messages
    ), messages
    assert any(
        m.startswith("[LIVE] output epoch closed") and f"parent={u1}" in m and "lang=fa" in m
        for m in messages
    ), messages
    # Supervisor 2026-09-23 18:16: the answer went out with NO playback
    # receipt (gap-retired only), which is UNKNOWN, not heard — the old pin
    # expected `outcome=answered` here.  New verdict: accepted with reason
    # `unconfirmed` (conditional on the wire).  The log contract this test is
    # about is unchanged: turn ids and codes, never the words.
    assert any(
        m.startswith("[LIVE] reask ") and f"turn={u1}" in m and "outcome=accepted" in m
        and "reason=unconfirmed" in m
        for m in messages
    ), messages
    for message in messages:
        assert words not in message and answer not in message, message
