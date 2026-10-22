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
    # D-P1/D-P3 (2026-10-05): it offers delegation and scopes itself to this
    # one reply — it no longer forbids the earlier request's work, which is
    # what «چرا سرچ نمی کنی» asks for.
    assert "or delegate it if it needs work" in content
    assert "This note is for this one reply only." in content
    assert "earlier request" not in content
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
        # D-P1 (e) quotes the request SPAN; U2 starts past `request_span`'s
        # 8 s gap so «قطع» stays its own (superseded) request here.
        box["p"].push(H.user_delta(question, 9500, 10400))
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


@pytest.mark.parametrize("mode", ["", "unheard"])
@pytest.mark.parametrize("lang", ["en", "fa"])
def test_the_plain_and_unheard_reask_offer_delegation_and_no_earlier_request_ban(mode, lang):
    """D-P1/D-P3 (2026-10-05): the conditional line always said "or delegate
    it if it needs work"; the plain and unheard lines said only "reply" and
    then forbade acting on any earlier request — so «چرا سرچ نمی کنی» was
    pushed to a spoken guess."""

    from app.services.live_voice_protocol import reask_instruction

    content = reask_instruction("چرا سرچ نمی کنی", lang, mode=mode)
    assert "«چرا سرچ نمی کنی»" in content
    if lang == "fa":
        assert "یا اگر کار لازم دارد، آن را واگذار کن" in content
        assert "این یادداشت فقط برای همین یک جواب است" in content
        assert "درخواست قبلی" not in content
    else:
        assert "or delegate it if it needs work" in content
        assert "This note is for this one reply only." in content
        assert "earlier request" not in content


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
    assert "یا اگر کار لازم دارد، آن را واگذار کن" in outside_quote
    assert "این یادداشت فقط برای همین یک جواب است" in outside_quote
    assert "هیچ درخواست قبلی" not in outside_quote


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
        m.startswith("[LIVE] output epoch opened") and f"parent={u1}" in m
        # D-P1: how long GPT-Live's input had been still when it answered.
        and " since_client_audio_ms=" in m and " since_input_audio_ms=" in m
        for m in messages
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


# ── D-P1 (2026-10-05 call): the relay asks for the reply it is owed ───


def _relay_reply_ids(client) -> list[str]:
    from app.services.live_voice_protocol import RELAY_REPLY_PREFIX

    seen: list[str] = []
    for frame in client.of("delegation"):
        did = str(frame.get("delegation_id") or "")
        if did.startswith(RELAY_REPLY_PREFIX) and did not in seen:
            seen.append(did)
    return seen


def _ladder_on(monkeypatch, *, regen=0.6, backstop=True, keeper=True, **extra):
    """The reply ladder as production runs it (regen → backstop), on the
    harness's fast clocks — `fast_clocks` pins both rungs OFF for every older
    suite (voice round 2026-10-06)."""

    H.fast_clocks(
        monkeypatch, voice_live_uplink_keeper_enabled=keeper,
        voice_live_reply_regen_after_s=regen,
        voice_live_reply_backstop_enabled=backstop,
        voice_live_delegation_ttl_s=2.0, **extra,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("client_build", ["142", "141", "140"])
async def test_a_reply_still_owed_is_reasked_by_the_relay_quoting_the_whole_request(
    monkeypatch, caplog, client_build,
):
    """0:20 / 1:04 / 3:38 of the 2026-10-05 call: a request closed, the model
    said nothing, and the only ask-again was the app's — 12 s later, quoting
    ONE fragment.  Now the relay itself owes the reply: it says `thinking`,
    runs the keeper (input must move for GPT-Live to answer), and after
    `voice_live_reply_regen_after_s` files ONE reask quoting the WHOLE
    request span, `reask_result{source:'relay'}`.  The app's own reask of the
    same turn later is a `duplicate`: the relay owns it.

    OC-1: build 140 (no `task_job`) predates relay reasks — it ignores the
    relay's `accepted` and reads `duplicate` as "covered", dropping the
    request it would hand off to chat. It is told `accepted` instead, and the
    relay injects nothing a second time.

    Re-baselined, voice round 2026-10-06 (D-R1): the 08:38 call's ACCEPTED
    note produced nothing for 10 s, so the ladder does not stop at one note —
    one regen interval after it the relay hands the WHOLE request to the
    caller's agent itself (`relay-reply:`), and still files exactly ONE note
    (each is permanent session state, D11).  A `reply_status` client (142) is
    told each step; builds 141/140 get no such frame."""

    from app.services import live_voice_protocol as live

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    _ladder_on(monkeypatch)
    calls: list[dict] = []

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append({"did": kw.get("delegation_id"), "task": task, **kw})
        await asyncio.sleep(0.1)
        return "Zero to One, by Peter Thiel.", "m"

    H.patch_relay(monkeypatch, think=think)
    consumed_at_first_await: list = []
    original_bind = live._LiveSession.bind_task_language

    async def bind_spy(self, task):
        if task.delegation_id.startswith(live.RELAY_REPLY_PREFIX):
            # The FIRST await of `fire_reply_backstop`: the whole span must
            # already be consumed (at-most-once; a provider delegation landing
            # now is absorbed, never bound to the head fragment).
            consumed_at_first_await.append(
                [(u.turn_id, u.consumed_by) for u in self.utterances.closed_queue]
            )
        return await original_bind(self, task)

    monkeypatch.setattr(live._LiveSession, "bind_task_language", bind_spy)
    box = {}
    head, tail = "what is the best book for a founder", "to raise money from investors"

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta(head, 0, 900))
            provider.push(H.user_delta(tail, 1500, 2400))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def relay_reasked():
        await _wait_until(lambda: len(_finals(client)) >= 2, label="two finals")
        await _wait_until(lambda: _reask_appends(provider), timeout=4.0, label="relay reask")
        _send(client, {
            "type": "inject_text", "text": tail,
            "reask_of_user_turn_id": _finals(client)[1]["turn_id"],
        })

    async def backstopped():
        await _wait_until(lambda: calls, timeout=4.0, label="the reply backstop's task")
        await asyncio.sleep(0.4)

    features = {
        "142": ("reask_turns", "task_job", "reply_status"),
        "141": ("reask_turns", "task_job"),
        "140": ("reask_turns",),
    }[client_build]
    client = H.FakeClient([
        _config_with(*features), relay_reasked, backstopped, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id=f"db-reask-relay-{client_build}")

    u1, u2 = (f["turn_id"] for f in _finals(client)[:2])
    reasks = _reask_appends(provider)
    assert [r["event_id"] for r in reasks] == [f"toup-live-reask-{u2}"], reasks
    assert head in reasks[0]["content"] and tail in reasks[0]["content"], reasks[0]
    results = [
        (f["user_turn_id"], f["outcome"], f.get("source")) for f in client.of("reask_result")
    ]
    second = "accepted" if client_build == "140" else "duplicate"
    assert results == [(u2, "accepted", "relay"), (u2, second, None)], results
    states = [f["state"] for f in client.of("state")]
    assert "thinking" in states, states
    keeper = [
        e for e in provider.of("session.input_audio.append")
    ]
    assert keeper, "the keeper must move GPT-Live's input while a reply is owed"
    messages = [r.getMessage() for r in caplog.records]
    assert any("[LIVE] uplink keeper on reason=reply" in m for m in messages), messages
    relay_line = [m for m in messages if m.startswith("[LIVE] reask ") and "via=relay:regen" in m]
    assert relay_line and "since_client_audio_ms=" in relay_line[0], messages
    # D-R1 (b): the app's ask of a turn the relay already re-asked is refused
    # as RELAY-OWNED (OC-1 still tells build 140 `accepted`), never filed.
    app_line = [m for m in messages if m.startswith("[LIVE] reask ") and " via=id " in m]
    assert len(app_line) == 1 and "reason=relay_owned" in app_line[0], app_line
    acked = [m for m in messages if m.startswith("[LIVE] append acked") and "delivery=reask" in m]
    assert acked, messages
    assert " chars=0 " not in acked[0], acked
    # D-R1 (d): the last rung — ONE relay-opened task for the WHOLE request.
    (rid,) = _relay_reply_ids(client)
    assert [c["did"] for c in calls] == [rid], calls
    assert head in calls[0]["task"] and tail in calls[0]["task"], calls[0]["task"]
    created = [
        f for f in client.of("delegation")
        if f.get("delegation_id") == rid and f.get("phase") == "created"
    ]
    assert len(created) == 1 and created[0]["parent_user_turn_id"] == u2, created
    assert consumed_at_first_await == [[(u1, rid), (u2, rid)]], consumed_at_first_await
    assert any(m.startswith("[LIVE] reply backstop fired") for m in messages), messages
    # Told ONCE, by an instruction — never commentary on an id it never minted.
    started = [
        e for e in provider.of("session.instructions.append")
        if str(e.get("event_id") or "").startswith("toup-live-reply-")
    ]
    assert len(started) == 1, started
    assert all(e.get("delegation_id") != rid for e in provider.sent)
    ready_caps = client.of("ready")[0]["capabilities"]
    statuses = [(f["user_turn_id"], f["outcome"]) for f in client.of("reply_status")]
    if client_build == "142":
        assert statuses == [(u2, "reasked"), (u2, "delegated")], client.of("reply_status")
        reasked, delegated = client.of("reply_status")
        assert reasked["attempt"] == 1 and reasked["next_action"] == "delegate"
        assert reasked["next_in_ms"] == 600 and reasked["since_ms"] >= 600, reasked
        assert delegated["task_id"] == rid and delegated["next_action"] == "none"
        assert "next_in_ms" not in delegated, delegated
        assert ready_caps["reply_status"] is True
        assert ready_caps["reply_pending_max_ms"] == int(live._REPLY_PENDING_MAX_S * 1000)
        assert ready_caps["reply_regen_after_ms"] == 600
        assert ready_caps["reply_backstop_after_ms"] == 8000
    else:
        assert statuses == [], client.of("reply_status")
        for key in ("reply_status", "reply_pending_max_ms", "reply_regen_after_ms",
                    "reply_backstop_after_ms"):
            assert key not in ready_caps, ready_caps
    for message in messages:
        assert head not in message and tail not in message, message
    assert u1 != u2


@pytest.mark.asyncio
async def test_a_refused_regen_is_logged_and_retried_next_tick(monkeypatch, caplog):
    """The 08:38 call: the relay's ONE regen was spent before it was judged —
    a refusal consumed it too, and nothing logged why.  Now a refusal
    (`input_still`: no input reaches GPT-Live, so it could not take the note)
    is logged ONCE however many 0.5 s retries repeat it, and the reask goes
    out the moment input moves again."""

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    _ladder_on(monkeypatch, regen=0.3, backstop=False, keeper=False)
    H.patch_relay(monkeypatch)
    box = {}
    words = "what is the tallest building in toronto"

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta(words, 0, 900))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    def refused_lines():
        return [
            r.getMessage() for r in caplog.records
            if r.getMessage().startswith("[LIVE] reask ") and "reason=input_still" in r.getMessage()
        ]

    async def quiet_then_audio():
        await _wait_until(lambda: _finals(client), label="final")
        # Well past the due moment and two retry intervals: still refused.
        await _wait_until(refused_lines, timeout=3.0, label="refusal logged")
        await asyncio.sleep(1.2)
        box["before_audio"] = len(_reask_appends(provider))

    async def reasked():
        await _wait_until(lambda: _reask_appends(provider), timeout=3.0, label="the reask")

    # The phone's audio frames are read only once the script step before them
    # returns: input starts moving right after the long quiet.
    audio = [{"type": "audio", "data": "AAAA"}] * 6
    client = H.FakeClient([
        _config_with("reask_turns"), quiet_then_audio, *audio, reasked, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-reask-refused")

    turn = _finals(client)[0]["turn_id"]
    assert box["before_audio"] == 0
    assert [r["event_id"] for r in _reask_appends(provider)] == [f"toup-live-reask-{turn}"]
    refused = refused_lines()
    assert len(refused) == 1, refused
    assert f"turn={turn} " in refused[0] and "outcome=in_flight" in refused[0]
    assert "via=relay:regen" in refused[0] and " since_close_ms=" in refused[0]
    accepted = [
        r.getMessage() for r in caplog.records
        if r.getMessage().startswith("[LIVE] reask ") and "outcome=accepted" in r.getMessage()
    ]
    assert len(accepted) == 1 and " repeat=" in accepted[0], accepted
    for record in caplog.records:
        assert words not in record.getMessage()


@pytest.mark.asyncio
async def test_the_relay_reask_cap_is_real(monkeypatch, caplog):
    """`_REASK_MAX_PER_TURN` was dead code: the `once_per_turn` fence refused
    every second ask before the counter was read.  Now the relay's own fence
    IS the counter — two notes per turn from anyone, each with its own event
    id — and the third attempt is `duplicate max_per_turn`.  With the
    backstop off the relay then says the turn is `unanswered`."""

    from app.services import live_voice_protocol as live

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    _ladder_on(monkeypatch, regen=0.3, backstop=False)
    monkeypatch.setattr(live, "_REPLY_REGEN_ATTEMPTS", 9)
    H.patch_relay(monkeypatch)
    box = {}
    words = "how far is the moon from earth"

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta(words, 0, 900))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    def capped():
        return any(
            "outcome=duplicate" in r.getMessage() and "reason=max_per_turn" in r.getMessage()
            for r in caplog.records
        )

    async def until_capped():
        await _wait_until(capped, timeout=4.0, label="the cap")
        await asyncio.sleep(0.6)

    client = H.FakeClient([
        _config_with("reask_turns", "task_job", "reply_status"), until_capped, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-reask-cap")

    turn = _finals(client)[0]["turn_id"]
    assert [r["event_id"] for r in _reask_appends(provider)] == [
        f"toup-live-reask-{turn}", f"toup-live-reask-{turn}-2",
    ], _reask_appends(provider)
    relay_accepts = [
        f for f in client.of("reask_result")
        if f.get("source") == "relay" and f["outcome"] == "accepted"
    ]
    assert len(relay_accepts) == 2, client.of("reask_result")
    statuses = [(f["outcome"], f.get("attempt"), f.get("reason")) for f in client.of("reply_status")]
    assert statuses == [
        ("reasked", 1, None), ("reasked", 2, None), ("unanswered", 2, "backstop_off"),
    ], statuses
    assert any("[LIVE] reply unanswered" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_an_unacked_reask_is_logged(monkeypatch, caplog):
    """D-R1 (h), the H1/H3 discriminator the 08:38 call lacked: a reask the
    provider never acked logs `append unacked … delivery=reask` at the next
    decision point — once."""

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    _ladder_on(monkeypatch, regen=0.3, backstop=False)
    H.patch_relay(monkeypatch)
    box = {}

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("what is the weather in oslo", 0, 900))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    def unacked():
        return [
            r.getMessage() for r in caplog.records
            if r.getMessage().startswith("[LIVE] append unacked")
        ]

    async def until_unacked():
        await _wait_until(unacked, timeout=4.0, label="unacked line")
        await asyncio.sleep(0.8)

    client = H.FakeClient([_config_with("reask_turns"), until_unacked, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=False)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-reask-unacked")

    turn = _finals(client)[0]["turn_id"]
    lines = unacked()
    assert len(lines) == 2, lines            # one per unacked APPEND, never per tick
    assert all(
        "kind=instructions delivery=reask" in line and f"turn={turn} " in line
        and " waited_ms=" in line and " since_input_audio_ms=" in line
        for line in lines
    ), lines
    assert "attempt=1" in lines[0] and "attempt=2" in lines[1], lines


@pytest.mark.asyncio
async def test_a_reply_dropped_in_the_interrupt_fence_stays_bindable_and_is_reasked(
    monkeypatch,
):
    """D-P1 fence-after-mint: the turn whose reply the interrupt fence drops
    was marked answered BEFORE the drop, so nothing could bind it again.  Now
    it is fenced only once an epoch exists — and the relay, not the app, asks
    for the lost reply after the grace."""

    from app.services import live_voice_protocol as live

    H.fast_clocks(
        monkeypatch, voice_live_interrupt_suppress_ms=400, voice_live_output_epoch_gap_ms=80,
    )
    monkeypatch.setattr(live, "_DROPPED_REPLY_GRACE_S", 0.3)
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
            provider.push(H.out_text("It is noon.", 1500, 1800))
            provider.push(H.out_audio())
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
        box["p"].push(H.out_text("It is noon.", 600, 850))      # inside the fence
        box["p"].push(H.out_audio())
        await _wait_until(lambda: _reask_appends(provider), timeout=3.0, label="relay reask")
        await asyncio.sleep(0.3)

    client = H.FakeClient([
        _config_with("reask_turns"), speak_then_barge_in, lost_reply, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id="db-reask-fence-relay")

    u1 = _finals(client)[0]["turn_id"]
    assert [r["event_id"] for r in _reask_appends(provider)] == [f"toup-live-reask-{u1}"]
    assert [(f["outcome"], f.get("source")) for f in client.of("reask_result")] == [
        ("accepted", "relay"),
    ]
    replies = [f for f in client.of("response_text") if "noon" in f.get("text", "")]
    assert replies and all(f.get("parent_user_turn_id") == u1 for f in replies), replies


@pytest.mark.asyncio
async def test_a_delegation_after_a_fenced_drop_still_binds_the_request(monkeypatch):
    """Fence-after-mint, the binding half: the model's reply to a request was
    dropped inside the interrupt fence and the model then delegated the
    work.  The delegation must bind THAT request — before D-P1 the dropped
    reply had already fenced it as answered and the delegation expired with
    no causal turn."""

    H.fast_clocks(
        monkeypatch, voice_live_interrupt_suppress_ms=400, voice_live_output_epoch_gap_ms=80,
        voice_live_delegation_ttl_s=1.5,
    )
    seen = []

    async def think(_user_id, _task, _session_id, **kwargs):
        seen.append(kwargs.get("delegation_id"))
        return "Done.", "test-model"

    H.patch_relay(monkeypatch, think=think)
    box = {}
    question = "find the best professor for LLMs"

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.out_text("Hello there.", 0, 300))
            provider.push(H.out_audio())
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def barge_in():
        await _wait_until(lambda: client.of("response_text"), label="greeting")
        box["p"].push(H.user_delta(question, 500, 900))
        await _wait_until(
            lambda: any(f.get("text") == question for f in client.of("transcript")),
            label="user words seen",
        )
        _send(client, {"type": "interrupt"})      # sent when this step returns

    async def drop_then_delegate():
        await _wait_until(lambda: _finals(client), label="U1 final")
        box["p"].push(H.out_text("Let me", 600, 700))           # dropped by the fence
        await asyncio.sleep(0.1)
        box["p"].push(H.delegation("d1", 950))
        await _wait_until(lambda: seen, timeout=3.0, label="delegation ran")

    client = H.FakeClient([
        H.config(), barge_in, drop_then_delegate, 0.3, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id="db-fence-bind")

    assert seen == ["d1"], seen
    created = [f for f in client.of("delegation") if f.get("phase") == "created"]
    assert created and created[0]["turn_id"] == _finals(client)[0]["turn_id"], created


# ── D-R1 (e)/(g): a promise is not an answer; an acknowledgement answers its own turn ──


@pytest.mark.asyncio
@pytest.mark.parametrize("heard", [False, True])
async def test_a_reply_that_only_promised_the_action_is_owed_again(monkeypatch, caplog, heard):
    """1:20 of the 08:38 call: «اوکی برا می آهنگ پخش کن» → «حتماً، دارم پخش
    می‌کنم.» and nothing was delegated — the turn counted as answered and no
    music could ever come from it.  When the reply CLAIMS the action
    (`action_claim_evidence`, the media lane's reader — stubbed here by
    name) over a play/media ask and nothing binds the turn within
    `voice_live_claim_grace_ms`, the reply is owed again: `thinking`, and a
    reask in mode `promise`.  `heard`: the phone's receipt said the promise
    WAS heard — on a real call it is — and that still does not answer the
    turn (the relay's reask fence must not read it as `answered`)."""

    from app.services import live_voice_protocol as live

    if heard:
        monkeypatch.setattr(
            live._LiveSession, "turn_output_evidence",
            lambda self, turn_id, **kw: live.OUTPUT_HEARD,
        )

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    _ladder_on(monkeypatch, regen=0.3, backstop=False, voice_live_claim_grace_ms=200)
    monkeypatch.setattr(
        live, "action_claim_evidence", lambda text: "playing it now" in text.lower(),
        raising=False,
    )
    H.patch_relay(monkeypatch)
    box = {}
    ask = "play a song by drake"

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta(ask, 0, 900))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def claimed():
        await _wait_until(lambda: _finals(client), label="final")
        box["p"].push(H.out_text("Sure, playing it now.", 1000, 1500))
        box["p"].push(H.out_audio())
        await _wait_until(lambda: _reask_appends(provider), timeout=4.0, label="promise reask")
        await asyncio.sleep(0.2)

    client = H.FakeClient([_config_with("reask_turns"), claimed, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id=f"db-reask-promise-{heard}")

    turn = _finals(client)[0]["turn_id"]
    messages = [r.getMessage() for r in caplog.records]
    promise = [m for m in messages if m.startswith("[LIVE] reply was a promise")]
    assert len(promise) == 1 and f"turn={turn} " in promise[0], messages
    reasks = _reask_appends(provider)
    assert [r["event_id"] for r in reasks] == [f"toup-live-reask-{turn}"], reasks
    assert "you did not delegate it" in reasks[0]["content"], reasks[0]
    assert "playing it now" in reasks[0]["content"] and ask in reasks[0]["content"]
    states = [f["state"] for f in client.of("state")]
    assert states[-1] == "thinking" or "thinking" in states[states.index("speaking"):], states
    for message in messages:
        assert ask not in message, message


@pytest.mark.asyncio
async def test_a_zero_claim_grace_turns_promise_mode_off(monkeypatch, caplog):
    """Merge seam RA × RB: `voice_live_claim_grace_ms = 0` is the switch the
    setting documents and `fast_clocks` pins ("promise mode off") — never "read
    the promise with no grace".  Read as no grace it re-opened every older
    suite's acknowledged media turn (test_live_round7_media f3)."""

    from app.services import live_voice_protocol as live

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    _ladder_on(monkeypatch, regen=0.3, backstop=False, voice_live_claim_grace_ms=0)
    monkeypatch.setattr(
        live, "action_claim_evidence", lambda text: "playing it now" in text.lower(),
        raising=False,
    )
    H.patch_relay(monkeypatch)
    box = {}

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("play a song by drake", 0, 900))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def claimed():
        await _wait_until(lambda: _finals(client), label="final")
        box["p"].push(H.out_text("Sure, playing it now.", 1000, 1500))
        box["p"].push(H.out_audio())
        await asyncio.sleep(1.2)

    client = H.FakeClient([_config_with("reask_turns"), claimed, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-reask-promise-off")

    assert _reask_appends(provider) == []
    assert not any("reply was a promise" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_reply_that_answers_is_never_read_as_a_promise(monkeypatch, caplog):
    """The fallback reader (no media lane) reads nothing as a promise, and a
    real reader reads nothing into a reply to a NON-media ask."""

    from app.services import live_voice_protocol as live

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    _ladder_on(monkeypatch, regen=0.3, backstop=False, voice_live_claim_grace_ms=200)
    monkeypatch.setattr(live, "action_claim_evidence", lambda text: True, raising=False)
    H.patch_relay(monkeypatch)
    box = {}

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("what time is it in tokyo", 0, 900))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def answered():
        await _wait_until(lambda: _finals(client), label="final")
        box["p"].push(H.out_text("It is nine in the morning there.", 1000, 1500))
        box["p"].push(H.out_audio())
        await asyncio.sleep(1.2)

    client = H.FakeClient([_config_with("reask_turns"), answered, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-reask-not-promise")

    assert _reask_appends(provider) == []
    assert not any("reply was a promise" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_delegations_acknowledgement_answers_its_own_turn_not_an_older_one(
    monkeypatch, caplog,
):
    """The 08:38 call, by inference (RR §3.1): the epoch after «ها»'s
    delegation was attributed to the stuck «فورمولوان» — `consume_direct_turn`
    took the newest UNCONSUMED turn, and «ها» was already consumed — and
    silently "answered" it.  Output opening right after a delegation was
    bound to a newer turn is that delegation's acknowledgement (D-R1 (g))."""

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    calls: list = []

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append(kw.get("delegation_id"))
        await asyncio.sleep(0.8)
        return "The next race is in Austin.", "m"

    H.patch_relay(monkeypatch, think=think)
    box = {}

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("formula one", 0, 700))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def second_request_delegated_then_acknowledged():
        await _wait_until(lambda: _finals(client), label="first final")
        await asyncio.sleep(0.3)
        # More than `voice_live_request_span_max_gap_ms` later (the call's
        # «ها» came 19 s after «فورمولوان»), so its span is its own words.
        box["p"].push(H.user_delta("when is the next race", 9000, 9800))
        box["p"].push(H.delegation("d-race", 9850))
        await _wait_until(lambda: calls, label="the delegation ran")
        box["p"].push(H.out_text("Let me check that for you.", 9900, 10300))
        box["p"].push(H.out_audio())
        await asyncio.sleep(0.6)

    client = H.FakeClient([
        _config_with("reask_turns"), second_request_delegated_then_acknowledged, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-reask-ack-attribution")

    first, second = (f["turn_id"] for f in _finals(client)[:2])
    acks = [f for f in client.of("response_text") if "Let me check" in str(f.get("text") or "")]
    assert acks, client.of("response_text")
    assert {f.get("parent_user_turn_id") for f in acks} == {second}, acks
    assert any(
        r.getMessage().startswith("[LIVE] output attributed to delegation ack")
        and f"turn_id={second} " in r.getMessage() and f"not={first}" in r.getMessage()
        for r in caplog.records
    )


def test_a_check_in_rebinds_to_a_turn_whose_reply_was_only_a_promise():
    """D-R1 (e): «چی شد» after «…آهنگ پخش کن» → «دارم پخش می‌کنم» (nothing
    delegated) walks back PAST the promised turn's direct reply — it was not
    an answer — so the check-in carries the request (`unanswered_before`)."""

    from app.config import settings
    from app.services.live_voice_protocol import (
        DelegatedTask, Utterance, UtteranceAssembler, _LiveSession,
    )

    session = _LiveSession.__new__(_LiveSession)
    session.adapter = None
    session.settings = settings
    session.utterances = UtteranceAssembler(provider_session_id=H.PSID)
    asked = Utterance(
        turn_id="t1", ordinal=1, start_ms=0, end_ms=900, text="play a song by drake",
        closed=True, causal_speech=True, dispatched=True, asks_nothing=False,
    )
    check_in = Utterance(
        turn_id="t2", ordinal=2, start_ms=5000, end_ms=5500, text="so what happened",
        closed=True, causal_speech=True, consumed_by="d1", asks_nothing=False,
    )
    session.utterances.closed_queue.extend([asked, check_in])
    task = DelegatedTask(
        delegation_id="d1", offset_ms=5500, transcript=check_in.text, turn_id="t2",
        created_ms=5500,
    )

    assert session.unanswered_before(task, "") == []      # a direct reply ends the walk
    asked.promised = True
    assert session.unanswered_before(task, "") == [asked]  # …a promise does not


@pytest.mark.asyncio
async def test_a_run_of_refused_regens_still_reaches_the_reply_backstop_at_the_ceiling(monkeypatch):
    """Phase D (tests-guards-3): with input still the whole call, every regen
    is REFUSED (`in_flight / input_still`, the H3 shape) — never accepted, so
    the "one regen interval after the accepted reask" rung never arms.  The
    only way the request still gets a card is `voice_live_reply_backstop_after_s`
    from the close; nothing exercised it."""

    from app.services import live_voice_protocol as live

    _ladder_on(monkeypatch, regen=0.3, backstop=True, keeper=False,
               voice_live_reply_backstop_after_s=1.2)
    calls: list = []

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append(kw.get("delegation_id"))
        return "Zero to One.", "m"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("what is the best book for a founder", 0, 900))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def wait():
        await _wait_until(lambda: _finals(client), label="the final")
        await asyncio.sleep(2.5)

    client = H.FakeClient([
        _config_with("reask_turns", "task_job", "reply_status"), wait, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-reask-ceiling")

    assert _reask_appends(provider) == []
    reply_ids = {
        str(f.get("delegation_id") or "") for f in client.of("delegation")
        if str(f.get("delegation_id") or "").startswith(live.RELAY_REPLY_PREFIX)
    }
    assert len(reply_ids) == 1, client.of("delegation")
    st = [(f["outcome"], f.get("reason")) for f in client.of("reply_status")]
    assert st and st[0] == ("pending", "input_still"), st
    assert ("delegated", None) in st, st
