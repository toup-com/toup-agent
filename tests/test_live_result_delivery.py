"""A delegated result is not delivered until it has been SAID (R12).

The owner symptom this file exists for: a research task runs, its card
completes, its answer lands in the thread — and the agent never tells the
caller anything.  A completed card, a persisted row and a progress placeholder
are all compatible with total silence, so every scenario here asserts the
ACTUAL answer text reaching the provider, and what the relay wrote down about
whether it was heard.

Numbers in the docstrings are R12.6's scenario numbers.

NOT EXERCISABLE HERE (owner device test only): whether GPT-Live really
paraphrases a commentary append aloud, how it words it, and whether any audio
reaches the caller's ear.  The fake provider models the documented contract —
acks carry `client_event_id`, an `error` can carry one too, an accepted append
is followed by output deltas — and nothing more.
"""

import asyncio
import time

import pytest

import test_live_harness as H
import test_live_no_receipt_r4 as NR


def _tool_start(relay, call_id="i1", name="web_search", query="admissions"):
    return relay.on_event({
        "type": "tool.start", "call_id": call_id, "name": name,
        "args": {"query": query},
    })


def _tool_end(relay, call_id="i1", name="web_search"):
    return relay.on_event({
        "type": "tool.end", "call_id": call_id, "name": name,
        "ok": True, "preview": "found", "elapsed_ms": 20,
    })


def _commentary(provider) -> list[str]:
    return [e["content"] for e in provider.of("session.commentary.append")]


def _instructions(provider) -> list[str]:
    return [e["content"] for e in provider.of("session.instructions.append")]


def _delegated_rows(saves, ref_prefix="live-delegation:"):
    return [s for s in saves if str(s.get("assistant_ref") or "").startswith(ref_prefix)]


def _counter_names(counters) -> list[str]:
    return [name for name, _fields in counters]


class _Persistence:
    """Everything `persist_spoken` needs of the queue: where a record went."""

    def __init__(self):
        self.records = []

    def submit(self, record):
        self.records.append(record)

    def keys(self) -> list[str]:
        return [r.key for r in self.records]


def _bare_session():
    """A `_LiveSession` with only the attributes the persistence seam reads.

    The claim/epoch guards below are ORDERING guards: each one is the other's
    backstop, so a relay scenario that exercises one leaves the other free to
    be deleted (the state it defends against is the state the first guard has
    already cleaned up). Constructing the state directly is the only way each
    one can be failed on its own, which is the whole point of pinning it.
    """

    from datetime import datetime, timezone

    from app.config import settings as app_settings
    from app.services.live_voice_protocol import LiveClientAdapter, _LiveSession

    session = _LiveSession.__new__(_LiveSession)
    session.settings = app_settings
    session.user_id = "user-1"
    session.adapter = LiveClientAdapter(session_id=H.PSID)
    session.persistence = _Persistence()
    session.spoken_rows = {}
    session.provider_session_id = H.PSID
    session.deliveries = {}
    session.active_delivery = None
    session.anchor_wall = datetime.now(timezone.utc)
    session.last_occurred = None
    session.provider_now_ms = 0
    session.provider_now_at = time.monotonic()
    return session


# ── the two seams this round widened ──────────────────────────────────

def test_the_delivery_verdict_survives_the_tenant_allowlist():
    """`sessions._clean_voice` drops any key it does not recognise, so a relay
    that sends `spoken` without it there would believe it wrote a field nobody
    stored — and `spoken:false` would be indistinguishable from delivered."""

    from app.api.sessions import _VOICE_KEYS, _clean_voice

    assert "spoken" in _VOICE_KEYS
    sent = {
        "source": "delegated", "delegation_id": "d1", "model": "m",
        "superseded": False, "cancelled": False, "spoken": False,
    }
    assert _clean_voice(dict(sent)) == sent


def test_every_line_this_round_added_exists_in_both_languages():
    """The relay never authors an English sentence for the user outside the
    fixed table, and a line that exists only in English is the same defect as
    one written inline."""

    from app.services.live_voice_protocol import relay_line

    fmt = {
        "step": "searching", "seconds": 4, "result": "the answer",
        "request": "the question", "first": "one", "second": "two",
        "title": "the title", "text": "the addendum",
    }
    for key in (
        "status_running", "status_running_nostep", "no_task",
        "result_nudge", "status_result", "detached_result",
        # Voice round 2026-10-07 (D-R1): the merge, continuation and G6 cues.
        "result_pair_cue", "result_continue_cue", "result_addendum_note",
    ):
        en = relay_line(key, "en", **fmt)
        fa = relay_line(key, "fa", **fmt)
        assert en and fa and en != fa, key


# ── the `status` class itself ─────────────────────────────────────────

def test_a_status_question_beats_the_overlap_rule_that_used_to_kill_the_task():
    """R12.3. A status question repeats the task's own nouns, so the overlap
    rule read it as a CORRECTION: the running research was cancelled and an
    identical one started, every time the caller asked how it was going."""

    from app.services.live_voice_protocol import classify_followup, token_overlap

    running = "search the toronto admissions deadline"
    question = "what did you find about the toronto admissions"
    assert token_overlap(question, running) >= 0.4, (
        "the scenario requires an overlap the correction rule would fire on"
    )
    assert classify_followup(question, running) == "status"


def test_a_long_request_that_merely_contains_a_status_phrase_is_still_a_request():
    """The markers are substrings, so length is what keeps "tell me what
    happened to the order I placed last week" a request. It fails towards
    doing the work."""

    from app.services.live_voice_protocol import classify_followup

    assert classify_followup("what happened", "reading the report") == "status"
    assert classify_followup(
        "tell me what happened to the order i placed last week", "reading the report",
    ) == "unrelated"


@pytest.mark.asyncio
async def test_a_status_shaped_first_request_is_dispatched_not_refused(monkeypatch):
    """…and with NOTHING to report it is not a status question at all. "What
    happened with my order" as the first thing said on a call is a request;
    answering "nothing is running right now" would turn a working feature into
    a refusal."""

    H.fast_clocks(monkeypatch)
    thinks = []

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        thinks.append(task)
        return "It shipped on Tuesday.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("what happened with my order", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 1.0, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=on_send, auto_ack=True, speak_on_commentary=True,
    )
    await H.run_relay(client, provider, timeout=8, db_session_id="db-r12-first")

    assert len(thinks) == 1, "there was nothing to report, so this was a request"
    assert any("It shipped on Tuesday." in c for c in _commentary(provider))


# ── (1) the ordinary case, end to end ─────────────────────────────────

@pytest.mark.asyncio
async def test_a_two_round_research_answer_is_spoken_and_marked_spoken(monkeypatch):
    """(1) Two tool rounds, then commentary carrying the REAL answer, an ack,
    an output epoch — and only then `spoken:true` on the row."""

    H.fast_clocks(monkeypatch)
    counters = H.capture_counters(monkeypatch)
    answer = "Admissions open on the twelfth of September."

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        await _tool_start(relay, "i1", "web_search")
        await _tool_end(relay, "i1", "web_search")
        await _tool_start(relay, "i2", "web_fetch")
        await _tool_end(relay, "i2", "web_fetch")
        return answer, "test-model"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("when do admissions open", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 1.2, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=on_send, auto_ack=True, speak_on_commentary=True,
    )
    await H.run_relay(
        client, provider, timeout=8, db_session_id="db-r12-1",
    )

    assert any(answer in c for c in _commentary(provider)), (
        "the caller has to be told the answer itself, not that a task finished"
    )
    assert provider.acked, "the provider acked the append the relay tracked"
    rows = _delegated_rows(recorded["saves"])
    assert rows and rows[0]["assistant_text"] == answer
    assert rows[-1]["assistant_voice"].get("spoken") is True, rows[-1]["assistant_voice"]
    assert rows[-1]["assistant_voice"]["source"] == "delegated", (
        "delivery does not change provenance (R12.5)"
    )
    assert "completed" in client.phases()
    assert "live_result_unspoken" not in _counter_names(counters)


# ── (2) asked about, mid-flight ───────────────────────────────────────

@pytest.mark.asyncio
async def test_a_status_question_is_answered_from_state_and_never_re_runs(monkeypatch):
    """(2) "what did you find" WHILE the task runs: classified `status`, not
    dispatched, not a supersession, no second AgentRunner turn — and the real
    answer still arrives afterwards."""

    H.fast_clocks(monkeypatch)
    thinks = []
    provider_box = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        thinks.append(task)
        await _tool_start(relay, "i1", "web_search")
        provider = provider_box["p"]
        # The caller asks how it is going — and the model delegates that too.
        provider.push(H.user_delta("what did you find so far", 4000, 4600))
        provider.push(H.delegation("d2", 4700))
        await asyncio.sleep(0.5)
        await _tool_end(relay, "i1", "web_search")
        return "Tuition is nineteen thousand dollars.", "test-model"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        provider_box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("what is the tuition", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 1.8, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=on_send, auto_ack=True, speak_on_commentary=True,
    )
    await H.run_relay(client, provider, timeout=10, db_session_id="db-r12-2")

    assert len(thinks) == 1, "a status question must not start a second turn"
    # INVERTED in R2 §C. This used to pin a `completed`/`status_answered`
    # frame for d2 — i.e. a task card for a question the relay answered
    # itself, which the app drew as a Done card with no body (V1+04:14,
    # «چی شد میگم»). The status question is answered in speech and is not a
    # job: its provider delegation emits no frame at all.
    assert [
        f for f in client.of("delegation") if f.get("delegation_id") == "d2"
    ] == [], "a relay-answered status question is not a task card"
    assert "superseded" not in client.phases()
    progress = [
        e for e in provider.of("session.commentary.append")
        if e["delegation_id"] == "d1" and "Still working" in e["content"]
    ]
    assert progress, _commentary(provider)
    assert "Searching the web" in progress[0]["content"], (
        "the progress line names the real current step, not a placeholder"
    )
    assert "admissions" not in progress[0]["content"], (
        "…by its kind, never by the raw query (R2 §E, contract §7)"
    )
    assert any(
        "Tuition is nineteen thousand dollars." in c for c in _commentary(provider)
    ), "the result still has to arrive after the status answer"
    rows = _delegated_rows(recorded["saves"])
    assert rows[-1]["assistant_voice"].get("spoken") is True


# ── (3) something unrelated happened in between ───────────────────────

@pytest.mark.asyncio
async def test_an_unrelated_utterance_before_the_answer_does_not_lose_it(monkeypatch):
    """(3) D9 again, from the delivery side: a completion is never discarded
    because somebody spoke, and it is announced exactly once."""

    H.fast_clocks(monkeypatch)
    provider_box = {}
    answer = "It is fourteen degrees and clear."

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        provider_box["p"].push(H.user_delta("nice weather we are having", 4000, 4900))
        await asyncio.sleep(0.25)
        return answer, "test-model"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        provider_box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("what is the temperature", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 1.5, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=on_send, auto_ack=True, speak_on_commentary=True,
    )
    await H.run_relay(client, provider, timeout=8, db_session_id="db-r12-3")

    carrying = [c for c in _commentary(provider) if answer in c]
    assert len(carrying) == 1, carrying
    rows = _delegated_rows(recorded["saves"])
    assert rows[-1]["assistant_voice"].get("spoken") is True


# ── (4) two answers at once ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_two_completions_are_serialized_and_each_spoken_once(monkeypatch):
    """(4) Attribution is "the first epoch after the append", so two answers
    appended inside one round-trip could each claim the other's epoch. They are
    serialized instead: two appends, two epochs, two `spoken:true` rows."""

    H.fast_clocks(monkeypatch)
    # Deliberately share no tokens: `classify_followup`'s overlap rule reads
    # two requests that share even their stopwords as a correction, and this
    # scenario is about two INDEPENDENT tasks finishing together.
    answers = {
        "how much is tuition": "Tuition is nineteen thousand.",
        "who runs admissions": "Doctor Vance runs admissions.",
    }

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        display = kwargs.get("display_request") or task
        await asyncio.sleep(0.05)
        return answers[display], "test-model"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("how much is tuition", 0, 400))
            provider.push(H.delegation("d1", 420))
            provider.push(H.user_delta("who runs admissions", 3000, 3400))
            provider.push(H.delegation("d2", 3450))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 2.0, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=on_send, auto_ack=True, speak_on_commentary=True,
    )
    await H.run_relay(client, provider, timeout=10, db_session_id="db-r12-4")

    spoken = _commentary(provider)
    for text in answers.values():
        assert len([c for c in spoken if text in c]) == 1, (text, spoken)
    flags = {
        (s["assistant_voice"]["delegation_id"], s["assistant_voice"].get("spoken"))
        for s in _delegated_rows(recorded["saves"])
        if "spoken" in (s.get("assistant_voice") or {})
    }
    assert flags == {("d1", True), ("d2", True)}, flags


@pytest.mark.asyncio
async def test_a_second_result_is_not_appended_until_the_first_is_accounted_for(
        monkeypatch):
    """(4, the other half) Serialization is a TIMING property, and the FIFO of
    epoch claims hides it: with both appends in flight the claims still bind in
    order, so nothing observable breaks until one of them is rejected or times
    out and its claim is withdrawn out from under the other. Pinned directly —
    the second append does not leave until the first has a verdict."""

    H.fast_clocks(monkeypatch)
    sent_at: list[float] = []
    answers = {
        "how much is tuition": "Tuition is nineteen thousand.",
        "who runs admissions": "Doctor Vance runs admissions.",
    }

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        return answers[kwargs.get("display_request") or task], "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("how much is tuition", 0, 400))
            provider.push(H.delegation("d1", 420))
            provider.push(H.user_delta("who runs admissions", 3000, 3400))
            provider.push(H.delegation("d2", 3450))
        elif event["type"] == "session.commentary.append":
            sent_at.append(time.monotonic())
        elif event["type"] == "session.close":
            provider.push(H.closed())

    # Acked, never spoken: delivery one only resolves when its speak timeout
    # expires, which is the window the second one must wait out.
    client = H.FakeClient([_config_141(), 3.0, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=12, db_session_id="db-r12-4b")

    # Voice round 2026-10-07 (D-R1 (2)): no second delivery — the requeue
    # (D-P1) is gone.  One commentary, one nudge, a verdict; the second answer
    # still waits for that verdict (merge pinned off in `fast_clocks`): two
    # speak budgets (0.5 s each) after the first append, never less.
    said = _commentary(provider)
    assert said == [
        "Tuition is nineteen thousand.", "Doctor Vance runs admissions.",
    ], said
    assert sent_at[1] - sent_at[0] >= 0.9, (
        "the second answer was appended while the first still had no verdict"
    )


# ── (5) the provider refused the append ───────────────────────────────

@pytest.mark.asyncio
async def test_a_rejected_append_is_recovered_by_one_nudge_not_reported_as_a_fault(
        monkeypatch):
    """(5) An `error` carrying our own `client_event_id` is a REJECTION of one
    append, not a session fault. The relay nudges once and still delivers — and
    the phone is never shown a generic "voice hit a problem" it cannot act on."""

    H.fast_clocks(monkeypatch)
    counters = H.capture_counters(monkeypatch)
    answer = "The library closes at eleven."

    async def think(*_args, **_kwargs):
        return answer, "test-model"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("when does the library close", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 1.5, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=on_send,
        auto_ack=True,
        reject_appends={"session.commentary.append"},
        reject_first=1,
        speak_on_commentary=True,
        speak_on_instructions=True,
    )
    await H.run_relay(client, provider, timeout=8, db_session_id="db-r12-5")

    assert provider.refused, "the scenario requires the first append to be refused"
    nudges = [c for c in _instructions(provider) if answer in c]
    assert len(nudges) == 1, _instructions(provider)
    assert "live_commentary_rejected" in _counter_names(counters)
    assert not [
        f for f in client.of("error") if f.get("code") == "append_rejected"
    ], "a rejected append is not a client-visible fault"
    rows = _delegated_rows(recorded["saves"])
    assert rows[-1]["assistant_voice"].get("spoken") is True


# ── (6) nothing came back at all ──────────────────────────────────────

@pytest.mark.asyncio
async def test_an_append_that_never_produces_speech_is_reported_as_unspoken(
        monkeypatch):
    """(6) No ack, no epoch: one nudge, then the honest verdict — `spoken:false`
    on the row, a `delegation` frame carrying the result text so the call
    surface can show it, and the counter. Exactly once."""

    H.fast_clocks(monkeypatch)
    counters = H.capture_counters(monkeypatch)
    answer = "The bus leaves at six."

    async def think(*_args, **_kwargs):
        return answer, "test-model"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("when does the bus leave", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 2.0, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=8, db_session_id="db-r12-6")

    unspoken = [f for f in client.of("delegation") if "spoken" in f]
    assert len(unspoken) == 1, unspoken
    assert unspoken[0]["phase"] == "completed"
    assert unspoken[0]["spoken"] is False
    # OC-2: the verdict carries the answer even on an open call — no card
    # holds it. Nothing reached the provider's input, so no second delivery
    # either: an unacked append is consumed LATE once input moves, and a copy
    # would be voiced twice.
    assert unspoken[0].get("result_text") == answer, unspoken[0]
    assert "live_result_requeued" not in _counter_names(counters), counters
    assert len([c for c in _instructions(provider) if answer in c]) == 1, (
        "exactly ONE nudge — a retry loop would talk over the caller"
    )
    # The REASON, not just the name: `no_ack` is the provider dropping our
    # append and `no_epoch` is the model declining to say it — different faults
    # with nothing in common to fix. This provider never acks, and a verdict
    # that always answered `no_epoch` would satisfy a bare name check just as
    # well, so the half of the split that names the PROVIDER is only verified
    # from this direction.
    assert ("live_result_nudged", {"reason": "no_ack"}) in counters, counters
    assert ("live_result_unspoken", {"reason": "no_ack_after_nudge"}) in counters, counters
    flags = [
        s["assistant_voice"]["spoken"] for s in _delegated_rows(recorded["saves"])
        if "spoken" in (s.get("assistant_voice") or {})
    ]
    assert flags == [False], flags


# ── (7) the caller hung up and came back ──────────────────────────────

@pytest.mark.asyncio
async def test_a_detached_result_is_announced_once_on_the_next_session(monkeypatch):
    """(7) Hang up mid-task: it detaches, finishes, persists — and the NEXT
    Live session on the same DB session is told what it found, as a thinking
    note, so the caller can ask instead of having it researched again."""

    H.fast_clocks(monkeypatch)
    release = asyncio.Event()
    answer = "Their office is in the Bahen Centre."

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        await release.wait()
        return answer, "test-model"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("where is their office", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.6, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=8, db_session_id="db-r12-7")
    release.set()
    await H.drain_detached()

    rows = _delegated_rows(recorded["saves"])
    assert rows and rows[-1]["assistant_text"] == answer, (
        "detached work persists its REAL result, not a placeholder"
    )
    assert not any(answer in c for c in _commentary(provider)), (
        "and says nothing into a closed session"
    )

    # …now reconnect. The tenant hands back the row it just wrote, so the seed
    # and the note both carry the answer.
    from app.api import ws_realtime as rt

    async def vps(_user_id):
        return ("https://tenant.example", "key")

    async def vps_api(_url, _key, _method, path, **_kwargs):
        if "/messages" in path:
            return [
                {"role": "user", "content": "where is their office"},
                {"role": "assistant", "content": answer},
            ]
        return None

    H.fast_clocks(monkeypatch, voice_live_history_rows=10)
    H.patch_relay(monkeypatch, vps=vps)
    monkeypatch.setattr(rt, "_vps_api", vps_api)

    client2 = H.FakeClient([H.config(), 0.4, {"type": "stop"}])
    provider2 = H.FakeProvider(
        on_send=lambda p, e: p.push(H.closed()) if e["type"] == "session.close" else None,
    )
    await H.run_relay(client2, provider2, timeout=8, db_session_id="db-r12-7")

    seeded = provider2.of("session.start")[0]["session"].get("input") or []
    assert any(
        answer in item["content"][0]["text"] for item in seeded
    ), "the history seed carries the finished answer"
    notes = [c for c in [
        e["content"] for e in provider2.of("session.thinking.append")
    ] if answer in c]
    assert len(notes) == 1, notes
    assert "do not redo it" in notes[0]

    # Announced ONCE: a third session says nothing about it.
    client3 = H.FakeClient([H.config(), 0.3, {"type": "stop"}])
    provider3 = H.FakeProvider(
        on_send=lambda p, e: p.push(H.closed()) if e["type"] == "session.close" else None,
    )
    await H.run_relay(client3, provider3, timeout=8, db_session_id="db-r12-7")
    assert not [
        e for e in provider3.of("session.thinking.append") if answer in e["content"]
    ]


# ── (8) ended, not cancelled ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_ending_the_call_before_completion_is_not_a_cancellation(monkeypatch):
    """(8) Ending voice DETACHES. The row must not be stamped `cancelled` — the
    job card reads that flag, and a cancelled card for work that finished is
    the same lie as a completed card for work that did not."""

    H.fast_clocks(monkeypatch)
    release = asyncio.Event()
    answer = "Forty-one people are registered."

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        await release.wait()
        return answer, "test-model"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("how many are registered", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.6, {"type": "stop"}])
    await H.run_relay(
        client, H.FakeProvider(on_send=on_send), timeout=8, db_session_id="db-r12-8",
    )
    release.set()
    await H.drain_detached()

    rows = _delegated_rows(recorded["saves"])
    assert rows, recorded["saves"]
    voice = rows[-1]["assistant_voice"]
    assert rows[-1]["assistant_text"] == answer
    assert voice["source"] == "delegated" and not voice["cancelled"]
    assert "spoken" not in voice, (
        "nothing was ever appended, so there is no delivery verdict to write"
    )


# ── (9) the caller cancelled it ───────────────────────────────────────

@pytest.mark.asyncio
async def test_an_explicitly_cancelled_task_never_speaks_its_answer(monkeypatch):
    """(9) `cancel_task` is the only cancellation. Its answer is never spoken,
    and the row says cancelled."""

    H.fast_clocks(monkeypatch)
    started = asyncio.Event()
    answer = "You should take the streetcar."

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        started.set()
        await asyncio.sleep(5)
        return answer, "test-model"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("how should i get there", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([
        H.config(), 0.5, {"type": "cancel_task", "delegation_id": "d1"},
        0.3, {"type": "stop"},
    ])
    provider = H.FakeProvider(
        on_send=on_send, auto_ack=True, speak_on_commentary=True,
    )
    await H.run_relay(client, provider, timeout=8, db_session_id="db-r12-9")

    assert started.is_set()
    assert not any(answer in c for c in _commentary(provider))
    assert "cancelled" in client.phases()
    assert not [f for f in client.of("delegation") if f.get("spoken") is True]


# ── (10) nothing came back from the agent ─────────────────────────────

@pytest.mark.asyncio
async def test_an_empty_answer_keeps_the_record_provenance_and_no_spoken_flag(
        monkeypatch):
    """(10) An empty or failed turn speaks a CODE line, and its row stays
    `delegated_record` (L1V2-R1) — a record of what happened, never delivery
    proof, and never carrying a delivery verdict it did not earn."""

    H.fast_clocks(monkeypatch)

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        await _tool_start(relay)
        await _tool_end(relay)
        return "", "test-model"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("look that up", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 1.0, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=on_send, auto_ack=True, speak_on_commentary=True,
    )
    await H.run_relay(client, provider, timeout=8, db_session_id="db-r12-10")

    assert [f for f in client.of("error") if f.get("code") == "delegation_empty"]
    rows = _delegated_rows(recorded["saves"])
    assert rows, recorded["saves"]
    assert rows[-1]["assistant_voice"]["source"] == "delegated_record"
    assert "spoken" not in rows[-1]["assistant_voice"]


# ── (11) asked about, after the fact ──────────────────────────────────

@pytest.mark.asyncio
async def test_a_follow_up_after_completion_replays_the_real_result(monkeypatch):
    """(11) "what did you find?" once the work is done is answered from the
    ledger with the ACTUAL text, and starts no second turn."""

    H.fast_clocks(monkeypatch)
    thinks = []
    answer = "There are nine seats left."

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        thinks.append(task)
        return answer, "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("how many seats are left", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def ask_again(provider):
        provider.push(H.user_delta("so what did you find", 5000, 5600))
        provider.push(H.delegation("d2", 5700))

    provider = H.FakeProvider(
        on_send=on_send, auto_ack=True, speak_on_commentary=True,
    )
    client = H.FakeClient([
        H.config(), 0.8, lambda: ask_again(provider), 0.8, {"type": "stop"},
    ])
    await H.run_relay(client, provider, timeout=10, db_session_id="db-r12-11")

    assert len(thinks) == 1, "the answer is already known; nothing may re-run it"
    replay = [c for c in _instructions(provider) if answer in c]
    assert replay, _instructions(provider)
    # INVERTED in R2 §C: this pinned the `status_answered` completion frame,
    # the bodiless Done card. The replay above IS the answer; d2 is no job.
    assert not [f for f in client.of("delegation") if f.get("delegation_id") == "d2"]


# ── (12) asked about, after reconnecting ──────────────────────────────

@pytest.mark.asyncio
async def test_a_follow_up_after_a_reconnect_replays_it_from_the_ledger(monkeypatch):
    """(12) The accepted ledger survives the socket, so the same question on a
    NEW provider session is still answered from the result rather than by
    researching it a second time."""

    H.fast_clocks(monkeypatch)
    thinks = []
    answer = "The deadline is the first of March."

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        thinks.append(task)
        return answer, "test-model"

    H.patch_relay(monkeypatch, think=think)

    def first(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("when is the deadline", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.9, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=first, auto_ack=True, speak_on_commentary=True,
    )
    await H.run_relay(client, provider, timeout=8, db_session_id="db-r12-12")
    assert len(thinks) == 1

    def second(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("what did you find earlier", 0, 600))
            provider.push(H.delegation("d9", 650))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client2 = H.FakeClient([H.config(), 0.9, {"type": "stop"}])
    provider2 = H.FakeProvider(on_send=second, auto_ack=True)
    await H.run_relay(client2, provider2, timeout=8, db_session_id="db-r12-12")

    assert len(thinks) == 1, "the reconnected session must not re-run the task"
    assert [c for c in _instructions(provider2) if answer in c], _instructions(provider2)


# ── (13) it finished while they were still talking ────────────────────

@pytest.mark.asyncio
async def test_a_result_landing_mid_utterance_waits_for_the_caller_to_stop(
        monkeypatch):
    """(13) R12.1. The append is held while the caller is still speaking, then
    sent — bounded, and never dropped. Interrupting somebody mid-sentence to
    read them a search result is the behaviour this defers around."""

    H.fast_clocks(
        monkeypatch,
        # The turn must still be OPEN when the answer lands, so the boundary is
        # decided by recency rather than by the gap closing it first.
        voice_live_utterance_gap_ms=60,
        voice_live_utterance_hard_gap_ms=4000,
        voice_live_result_defer_max_ms=3000,
    )
    counters = H.capture_counters(monkeypatch)
    answer = "It is a twenty minute walk."
    provider_box = {}
    marks = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        # …and the caller starts a NEW sentence just as the answer arrives.
        provider_box["p"].push(H.user_delta("and also i wanted to ask and", 4000, 4800))
        await asyncio.sleep(0.1)
        marks["answered"] = time.monotonic()
        return answer, "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        provider_box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("how far is it", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.commentary.append" and answer in event["content"]:
            marks["spoke"] = time.monotonic()
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 2.5, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=on_send, auto_ack=True, speak_on_commentary=True,
    )
    await H.run_relay(client, provider, timeout=10, db_session_id="db-r12-13")

    assert "spoke" in marks, "deferral must never turn into a drop"
    assert marks["spoke"] - marks["answered"] >= 0.5, (
        "the answer was appended while the caller was still mid-sentence"
    )
    assert "live_result_deferred" in _counter_names(counters)
    assert "live_result_defer_expired" not in _counter_names(counters), (
        "the boundary released it, not the deadline"
    )


# ── attribution: WHICH append did that epoch answer? (L6V-1/L6V-3) ────

def test_an_epoch_is_attributed_by_claim_identity_never_by_delegation_id():
    """The epoch claim is a FIFO, and `speak()` files one for EVERY append —
    including the progress line about a running task, the cancel line and the
    failure line, all of which name a delegation that may also own a result
    delivery. Keyed by that NAME, the epoch those lines caused resolves the
    waiting result; keyed by the claim OBJECT, only the append that filed it
    can. The id is how the candidate is found, never how it is proved."""

    from app.services.live_voice_protocol import (
        DelegatedTask, ResultDelivery, _LiveSession,
    )

    session = _LiveSession.__new__(_LiveSession)
    task = DelegatedTask(delegation_id="d1", offset_ms=0, transcript="find the deadline")
    waiting = ResultDelivery(
        task=task,
        answer="The first of March.",
        claim={
            "ref": "live-delegation:p:d1", "payload": {"assistant_text": "x"},
            "revision": 1, "delegation_id": "d1",
        },
    )
    session.deliveries = {"d1": waiting}
    # Set for the mutation's benefit: "resolve whatever delivery is active"
    # must fail here for the reason this test names, not with an AttributeError.
    session.active_delivery = waiting

    bare = {"ref": "", "payload": {}, "revision": 1, "delegation_id": "d1"}
    assert session.delivery_for_claim(bare) is None, (
        "a progress line about d1 is not d1's answer being spoken"
    )
    assert session.delivery_for_claim(waiting.claim) is waiting
    waiting.resolved = True
    assert session.delivery_for_claim(waiting.claim) is None, (
        "resolved exactly once; a second epoch may not re-resolve it"
    )


@pytest.mark.asyncio
async def test_a_progress_lines_epoch_never_marks_another_answer_spoken(monkeypatch):
    """The blocker (L6V-1), end to end. Two tasks: one still running, one
    finished and appended but NOT voiced. The caller asks how it is going, the
    relay speaks a progress line about the RUNNING one — and that is the only
    epoch in the session.

    Attribution used to be "the next epoch, whoever caused it", so the progress
    line's epoch resolved the waiting answer `spoken:true`, folded its own
    playback onto that answer's row, sent no `delegation{spoken:false}` frame
    and fired no counter. The caller was never told the result and nothing
    anywhere said so."""

    H.fast_clocks(monkeypatch, voice_live_speak_timeout_s=0.6)
    counters = H.capture_counters(monkeypatch)
    answer = "The tuition is nineteen thousand dollars."
    release = asyncio.Event()

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        display = kwargs.get("display_request") or task
        if "flight" in display:
            # The long-running one: a real step, so the progress line has a
            # real thing to name, and no answer until the test releases it.
            await _tool_start(relay, "i1", "web_search")
            await release.wait()
            return "Six hours.", "test-model"
        return answer, "test-model"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        kind = event["type"]
        if kind == "session.start":
            provider.push(H.user_delta("how long is the flight", 0, 400))
            provider.push(H.delegation("d1", 420))
            provider.push(H.user_delta("who wrote this book", 3000, 3400))
            provider.push(H.delegation("d2", 3450))
        elif kind == "session.commentary.append":
            content = event.get("content") or ""
            if answer in content:
                # The answer is with the provider and has not been voiced. NOW
                # the caller asks how the other task is going.
                provider.push(H.user_delta("what did you find", 6000, 6600))
                provider.push(H.delegation("d3", 6700))
            elif "Still working" in content:
                # …and this — the progress line — is the one thing the model
                # actually says out loud.
                provider.push(H.out_text("Still working on it.", 900, 1500))
                provider.push(H.out_audio("SPEAKING"))
        elif kind == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 3.0, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=14, db_session_id="db-r12-attr", drain=False,
    )
    release.set()
    await H.drain_detached()

    d2_rows = [
        s for s in _delegated_rows(recorded["saves"])
        if (s.get("assistant_voice") or {}).get("delegation_id") == "d2"
    ]
    assert d2_rows, recorded["saves"]
    verdicts = [
        r["assistant_voice"]["spoken"] for r in d2_rows
        if "spoken" in r["assistant_voice"]
    ]
    assert verdicts == [False], (
        "the epoch belonged to the progress line about d1, so nothing about d2 "
        "was ever said"
    )
    assert not [
        r for r in d2_rows if "played_ms" in (r.get("assistant_voice") or {})
    ], "the progress line's playback was folded onto an answer nobody heard"

    unspoken = [f for f in client.of("delegation") if f.get("spoken") is False]
    assert len(unspoken) == 1, unspoken
    assert unspoken[0]["delegation_id"] == "d2"
    assert "live_result_unspoken" in _counter_names(counters)
    # D-P1: the progress line no longer SPOILS the waiting answer — it is held
    # behind it and said after its verdict, so its epoch can only be its own.
    assert ("live_relay_line_held", {"reason": "interposed_speech"}) in counters, counters
    assert "live_result_attribution_spoiled" not in _counter_names(counters), counters


# ── the ack is evidence, not decoration (L6V-2) ───────────────────────

@pytest.mark.asyncio
async def test_the_speak_budget_starts_when_the_provider_accepts_the_append(
        monkeypatch):
    """R12.2 words it as "an output epoch begins within N seconds of
    ACCEPTANCE". `TrackedAppend.acked` was write-only, so the clock ran from
    the send instead — a provider that took half the budget to answer ate half
    the time the model had to start speaking — and the verdict could not tell
    "the append was dropped" from "the model declined to say it", which are
    different faults with nothing in common to fix."""

    H.fast_clocks(monkeypatch, voice_live_speak_timeout_s=0.6)
    counters = H.capture_counters(monkeypatch)
    answer = "The gate closes at nine."
    marks = {}

    async def think(*_args, **_kwargs):
        return answer, "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        kind = event["type"]
        if kind == "session.start":
            provider.push(H.user_delta("when does the gate close", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif kind == "session.commentary.append" and answer in event["content"]:
            marks["appended"] = time.monotonic()

            async def late_ack():
                # A provider that is slow to admit it has the append, and then
                # never says anything.
                await asyncio.sleep(0.5)
                marks.setdefault("acked", time.monotonic())
                provider.push({
                    "type": "session.commentary.appended",
                    "client_event_id": event["event_id"],
                })

            asyncio.get_running_loop().create_task(late_ack())
        elif kind == "session.instructions.append":
            marks.setdefault("nudged", time.monotonic())
        elif kind == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([_config_141(), 3.0, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=14, db_session_id="db-r12-ack")

    assert "acked" in marks and "nudged" in marks, marks
    assert marks["nudged"] - marks["acked"] >= 0.3, (
        "the six-second speak budget ran from the send, not from acceptance"
    )
    reasons = [f.get("reason") for name, f in counters if name == "live_result_nudged"]
    assert reasons and reasons[0] == "no_epoch", reasons
    # Voice round 2026-10-07 (D-R1 (2)): acked and never voiced is ONE
    # nudge and then the verdict — the requeue (a second copy) is gone.
    assert ("live_result_unspoken", {"reason": "no_epoch_after_nudge"}) in counters, counters
    assert not [name for name, _f in counters if name == "live_result_requeued"], counters


# ── R12.4's note is a sentence aimed at the model, so it is localised ──

@pytest.mark.asyncio
async def test_a_detached_result_is_announced_in_the_callers_language(monkeypatch):
    """(7), the Persian half. The reconnect note asks the model to TELL the
    caller something, which is exactly the shape `result_nudge` and
    `status_result` are localised for: an English instruction is the commonest
    way a Persian call ends up answered in English. The originating task's
    language rides the ledger row, because by announcement time no utterance of
    that task's exists to detect it from."""

    from app.services.live_voice_protocol import relay_line

    H.fast_clocks(monkeypatch)
    release = asyncio.Event()
    answer = "دفترشان در ساختمان بهن است."

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        await release.wait()
        return answer, "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("دفترشان کجاست", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.6, {"type": "stop"}])
    await H.run_relay(
        client, H.FakeProvider(on_send=on_send), timeout=8, db_session_id="db-r12-fa",
    )
    release.set()
    await H.drain_detached()

    client2 = H.FakeClient([H.config(), 0.4, {"type": "stop"}])
    provider2 = H.FakeProvider(
        on_send=lambda p, e: p.push(H.closed()) if e["type"] == "session.close" else None,
    )
    await H.run_relay(client2, provider2, timeout=8, db_session_id="db-r12-fa")

    notes = [
        e["content"] for e in provider2.of("session.thinking.append")
        if answer in e["content"]
    ]
    assert len(notes) == 1, notes
    tail = relay_line("detached_result", "fa", request="x", result="y").split("y")[-1]
    assert tail and tail in notes[0], notes[0]
    assert "do not redo it" not in notes[0], (
        "a Persian call got the one instruction that decides the reply language "
        "in English"
    )


# ── the two bare-claim guards, each on its own (L6V2-4) ───────────────

@pytest.mark.asyncio
async def test_a_bare_claims_binding_is_released_when_its_epoch_is_settled():
    """`speak()` files a claim for EVERY append, and a progress, cancel or
    failure line's claim names no row — it is only holding that append's place
    in the FIFO. The epoch it binds is an ordinary spoken reply, so the binding
    has to be undone: left in `commentary_epochs` it is what `persist_spoken`
    reads to decide the epoch belongs to somebody's delegated row."""

    from app.services.live_voice_protocol import ClosedEpoch  # noqa: F401  (shape doc)

    session = _bare_session()
    bare = session.build_commentary_claim("", {}, delegation_id="d1")
    session.note_commentary_epoch_claim(bare)
    session.adapter.provider_event(H.out_text("Still working on it.", 900, 1500))
    assert session.adapter.commentary_epochs == {1: bare}, (
        "the scenario requires the bare claim to have bound to a real epoch"
    )

    await session.settle_bound_claims()

    assert session.adapter.commentary_epochs == {}, (
        "a claim naming no row must not stay bound to an epoch"
    )


@pytest.mark.asyncio
async def test_an_epoch_bound_to_a_bare_claim_is_never_persisted_under_an_empty_key():
    """The second half of the same pair, and it has to be asserted with the
    FIRST half not yet run — that is precisely the ordering it exists for.

    `settle_bound_claims` releases a bare binding from inside the provider loop,
    in the same iteration that minted the epoch; this guard is what stands
    between a reordering of that and a persistence record keyed on the EMPTY
    STRING, which the receiving route upserts by. With both guards gone the
    relay submits exactly that."""

    from app.services.live_voice_protocol import ClosedEpoch

    session = _bare_session()
    bare = session.build_commentary_claim("", {}, delegation_id="d1")
    session.note_commentary_epoch_claim(bare)
    session.adapter.provider_event(H.out_text("Still working on it.", 900, 1500))
    assert session.adapter.commentary_epochs == {1: bare}

    await session.persist_spoken(ClosedEpoch(
        epoch=1, output_id=f"live:{H.PSID}:1", text="Still working on it.",
        interrupted=False, played_ms=900,
    ))

    keys = session.persistence.keys()
    assert all(keys), f"a persistence record keyed on nothing: {keys}"
    assert keys == [f"live-output:{H.PSID}:1"], keys


# ── a spoiled delivery's row stays consistent with itself (L6V2-5) ────

@pytest.mark.asyncio
@pytest.mark.parametrize("told", [True, False])
async def test_a_spoiled_deliverys_epoch_folds_only_when_its_words_prove_it(told):
    """Voice round 2026-10-07 (D-R1 §6.3, a flipped pin): "spoiled" existed
    because the epoch FIFO could not tell two outstanding appends apart, so a
    spoiled delivery's epoch was NEVER folded onto its row.  The words can tell
    them apart: an epoch whose words prove the answer folds onto the delegated
    row even if another speakable append went out in between; one that does
    not prove it stays an ordinary spoken reply (its own row) — and nothing of
    it reaches the delegated row."""

    from app.services.live_voice_protocol import (
        ClosedEpoch, DelegatedTask, ResultDelivery,
    )

    session = _bare_session()
    ref = f"live-delegation:{H.PSID}:d1"
    payload = {
        "assistant_text": "The deadline is the first of March.",
        "assistant_voice": {"source": "delegated", "delegation_id": "d1"},
    }
    claim = session.build_commentary_claim(ref, payload, delegation_id="d1")
    delivery = ResultDelivery(
        task=DelegatedTask(delegation_id="d1", offset_ms=0, transcript="the deadline"),
        answer="The deadline is the first of March.",
        claim=claim,
    )
    session.deliveries["d1"] = delivery
    session.active_delivery = delivery
    delivery.sent_monotonic = delivery.started_monotonic = time.monotonic()

    # Something else asked the model to speak before this delivery's epoch.
    session.spoil_active_attribution(
        "interposed_speech",
        own=session.build_commentary_claim("", {}, delegation_id="d2"),
    )
    assert delivery.spoiled

    text = "It is the first of March." if told else "Still working on it."
    session.adapter.provider_event(H.out_text(text, 900, 1500))
    await session.persist_spoken(ClosedEpoch(
        epoch=1, output_id=f"live:{H.PSID}:1", text=text,
        interrupted=False, played_ms=1234, born_monotonic=time.monotonic(),
    ))

    folds = [r for r in session.persistence.records if r.key == ref]
    if told:
        assert folds and folds[-1].payload["assistant_voice"]["epoch"] == 1, (
            "the epoch that SAID the answer was not folded onto its row"
        )
        assert f"live-output:{H.PSID}:1" not in session.persistence.keys()
        assert delivery.resolved and delivery.spoken and delivery.words_reason == "words"
    else:
        assert not [
            r for r in folds if "epoch" in (r.payload.get("assistant_voice") or {})
        ], f"an unrelated utterance was recorded on the delegated row: {folds}"
        assert f"live-output:{H.PSID}:1" in session.persistence.keys()
        assert not delivery.resolved


# ── a withdrawn claim is really withdrawn (L6V2-3) ────────────────────

@pytest.mark.asyncio
async def test_a_withdrawn_claim_never_captures_the_next_unrelated_epoch(monkeypatch):
    """A delivery that resolves `spoken:false` has to take its claim OUT of the
    FIFO, or the next epoch — some ordinary reply, minutes later — binds to it.

    Nothing downstream would catch that: the delivery is already resolved, so
    `delivery_for_claim` answers None while the claim's `ref` is truthy, which
    is exactly the pair `settle_bound_claims` reads as "not mine, leave it
    bound". `persist_spoken` then folds that reply's playback onto a row
    already stamped `spoken:false` — a row recording delivery of the thing the
    relay had just finished deciding was never delivered."""

    H.fast_clocks(monkeypatch)
    counters = H.capture_counters(monkeypatch)
    answer = "The bus leaves at six."
    unrelated = "Anything else I can do?"

    async def think(*_args, **_kwargs):
        return answer, "test-model"

    recorded = H.patch_relay(monkeypatch, think=think)

    async def speak_about_something_else(provider):
        # Only AFTER the verdict: an epoch that arrives while the delivery is
        # still waiting is the ordinary delivery proof, which is a different
        # test.
        for _ in range(300):
            if "live_result_unspoken" in _counter_names(counters):
                break
            await asyncio.sleep(0.01)
        else:
            return
        provider.push(H.out_text(unrelated, 5000, 5600))
        provider.push(H.out_audio("SPEAKING"))

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("when does the bus leave", 0, 400))
            provider.push(H.delegation("d1", 420))
            asyncio.get_running_loop().create_task(
                speak_about_something_else(provider),
            )
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 2.5, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=14, db_session_id="db-r12-drop")

    assert "live_result_unspoken" in _counter_names(counters), (
        "the scenario requires the delivery to have been given up on first"
    )
    rows = _delegated_rows(recorded["saves"])
    assert rows, recorded["saves"]
    assert not [
        r for r in rows if "played_ms" in (r.get("assistant_voice") or {})
    ], "an unrelated reply's playback was recorded on an answer nobody heard"
    spoken_rows = [
        s for s in recorded["saves"]
        if str(s.get("assistant_ref") or "").startswith("live-output:")
    ]
    assert [s["assistant_text"] for s in spoken_rows] == [unrelated], (
        "the unrelated reply has to become its own row, not somebody's fold"
    )


# ── the INSTRUCTIONS half of the attribution seam (L6V2-1) ────────────

@pytest.mark.asyncio
async def test_a_status_instruction_never_marks_another_answer_spoken(monkeypatch):
    """The other way the relay makes the model speak, and it is not `speak()`.

    `answer_status`'s "tell them now" branch sends an `instructions.append`
    directly — no commentary, no claim, nothing in the FIFO — and the epoch the
    model opens to obey it is the first epoch of the session. The FIFO hands
    that epoch to its head, which is still the OTHER answer's own claim, so
    claim-identity passes and only the spoil refuses it.

    Without the spoil this writes `spoken:true` on an answer the caller was
    never told, on the strength of a sentence that was about a different
    request, with no frame, no counter and nothing anywhere recording that the
    relay had stopped being able to attribute — the exact false success R12.2
    and R12.5 forbid. The commentary half of the same seam is pinned by
    `test_a_progress_lines_epoch_never_marks_another_answer_spoken`; this is
    the instructions half."""

    H.fast_clocks(monkeypatch, voice_live_speak_timeout_s=0.4)
    counters = H.capture_counters(monkeypatch)
    first = "The cover is green."
    second = "It was written by Ursula Le Guin."
    voiced: list[str] = []

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        display = kwargs.get("display_request") or task
        if "colour" in display:
            return first, "test-model"

        async def later():
            # The caller asks what was found AFTER both tasks are done, so
            # nothing is running and `answer_status` has no delegation to hang
            # commentary on — which is the branch that instructs instead.
            await asyncio.sleep(0.4)
            provider.push(H.user_delta("what did you find", 6000, 6600))
            provider.push(H.delegation("d3", 6700))

        asyncio.ensure_future(later())
        return second, "test-model"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        kind = event["type"]
        content = str(event.get("content") or "")
        if kind == "session.start":
            provider.push(H.user_delta("what colour is it", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif kind == "session.commentary.append" and first in content:
            # d1's answer is with the provider, accepted and NOT voiced — the
            # delivery is active and waiting for an epoch that never comes.
            provider.push(H.user_delta("who wrote this book", 3000, 3400))
            provider.push(H.delegation("d2", 3450))
        elif kind == "session.instructions.append" and second in content and not voiced:
            # The one thing the model says out loud all session: the OTHER
            # answer, replayed because the caller asked for it.
            voiced.append(content)
            provider.push(H.out_text("It was Ursula Le Guin.", 900, 1500))
            provider.push(H.out_audio("SPEAKING"))
        elif kind == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 4.5, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=18, db_session_id="db-r12-statusattr")

    assert voiced, "the scenario requires the status instruction to be spoken"
    # D-P1: held behind d1's delivery (never spoiling it) and said only once
    # d1 had its verdict — so its epoch is unambiguously its own.
    status_at = next(
        i for i, f in enumerate(provider.sent)
        if f["type"] == "session.instructions.append" and second in str(f.get("content"))
    )
    d1_appends = [
        i for i, f in enumerate(provider.sent)
        if f["type"] == "session.commentary.append" and first in str(f.get("content"))
    ]
    assert d1_appends and max(d1_appends) < status_at, (d1_appends, status_at)
    d1_rows = [
        s for s in _delegated_rows(recorded["saves"])
        if (s.get("assistant_voice") or {}).get("delegation_id") == "d1"
    ]
    assert d1_rows, recorded["saves"]
    verdicts = [
        r["assistant_voice"]["spoken"] for r in d1_rows
        if "spoken" in r["assistant_voice"]
    ]
    assert verdicts == [False], (
        "the only epoch in the session answered an instruction about the OTHER "
        f"request, so nothing about d1 was ever said: {verdicts}"
    )
    assert not [
        r for r in d1_rows if "played_ms" in (r.get("assistant_voice") or {})
    ], "that utterance's playback was folded onto an answer nobody heard"

    unspoken = [
        f for f in client.of("delegation")
        if f.get("spoken") is False and f.get("delegation_id") == "d1"
    ]
    assert len(unspoken) == 1, unspoken
    assert "live_result_unspoken" in _counter_names(counters)
    assert ("live_relay_line_held", {"reason": "status_instruction"}) in counters, counters
    assert "live_result_attribution_spoiled" not in _counter_names(counters), counters


# ── PR-1 instrumentation: what the 2026-10-02 call could not show ────

@pytest.mark.asyncio
async def test_an_ack_logs_its_latency_and_the_epoch_logs_what_it_carried(
        monkeypatch, caplog):
    """PR-1. On 2026-10-02 none of the call's eight relay appends was voiced
    in budget, and the log could not say whether GPT-Live had even ACCEPTED
    them: acks were never logged. Nor could it say whether an epoch was audio
    only (a backchannel, a held tail) or carried words. And a provider schema
    change would have been invisible. Three lines, ids and counts only:

    - `append acked … delivery=result … ack_ms=N` (send → ack),
    - `output epoch closed … audio_chunks=N text_chars=N`,
    - `provider event shape n=1..3 type=… keys=…` (key NAMES, never values).
    """

    import logging
    import re

    H.fast_clocks(monkeypatch, voice_live_speak_timeout_s=2.0)
    answer = "The gate closes at nine."
    spoken = "It closes at nine."
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")

    async def think(*_args, **_kwargs):
        return answer, "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        kind = event["type"]
        if kind == "session.start":
            provider.push(H.user_delta("when does the gate close", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif kind == "session.commentary.append" and answer in event["content"]:
            async def late_ack_then_speech():
                await asyncio.sleep(0.3)
                provider.push({
                    "type": "session.commentary.appended",
                    "client_event_id": event["event_id"],
                })
                provider.push(H.out_text(spoken, 900, 1500))
                provider.push(H.out_audio("AQID"))
                provider.push(H.out_audio("BBBB"))

            asyncio.get_running_loop().create_task(late_ack_then_speech())
        elif kind == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 2.0, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-pr1-ack")

    messages = [r.getMessage() for r in caplog.records]

    acks = [m for m in messages if m.startswith("[LIVE] append acked ")]
    result_acks = [m for m in acks if "kind=commentary delivery=result delegation=d1 " in m]
    assert result_acks, acks
    ack_ms = int(re.search(r" ack_ms=(\d+)$", result_acks[0]).group(1))
    assert 250 <= ack_ms <= 3000, result_acks[0]
    assert f"chars={len(answer)} " in result_acks[0]

    closed = [m for m in messages if m.startswith("[LIVE] output epoch closed ")]
    carried = [m for m in closed if m.endswith(f"audio_chunks=2 text_chars={len(spoken)}")]
    assert carried, closed
    # The existing fields stay where they were; the counts are only appended.
    assert re.match(
        r"\[LIVE\] output epoch closed epoch=\d+ reason=\w+ parent=\S+ lang=\S+ audio_chunks=",
        carried[0],
    ), carried[0]

    shapes = [m for m in messages if m.startswith("[LIVE] provider event shape ")]
    assert [re.search(r" n=(\d+) ", m).group(1) for m in shapes] == ["1", "2", "3"], shapes
    for line in shapes:
        keys = re.search(r" keys=(\S+)$", line).group(1).split(",")
        assert "type" in keys, line
        assert "gate" not in line and "nine" not in line, "names only, never values"


# ── D-P1: "Result in chat" only when the call is really gone ──────────

def _config_141():
    """This round's app: `task_job` is the feature build 141 added."""
    frame = H.config()
    frame["features"] = list(frame["features"]) + ["task_job"]
    return frame


@pytest.mark.asyncio
@pytest.mark.parametrize("stay", [True, False])
async def test_an_open_call_gets_a_second_delivery_before_result_in_chat(monkeypatch, stay):
    """2026-10-05: a delegation's answer that was not voiced fell to "Result
    in chat" although the call was still open.  Voice round 2026-10-07 (D-R1
    (2), a flipped pin): the second delivery (the requeue) is GONE — it is
    what told the 10-07 Academic answer a third time.  Nothing said: ONE
    commentary, ONE nudge, then the verdict with the answer's text (OC-2: no
    card holds it).  A call that ends after the nudge gets the same verdict,
    text included."""

    H.fast_clocks(monkeypatch)
    counters = H.capture_counters(monkeypatch)
    answer = "The museum opens at ten."

    async def think(*_args, **_kwargs):
        return answer, "test-model"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("when does the museum open", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    # Acked and never voiced: `no_epoch` — the provider has it, so no copy
    # of it can be consumed late.
    async def nudge_out():
        for _ in range(400):
            if [c for c in _instructions(provider) if answer in c]:
                return
            await asyncio.sleep(0.01)

    client = H.FakeClient([
        _config_141(), *([4.0] if stay else [nudge_out]), {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id=f"db-dp1-requeue-{stay}")

    appended = [c for c in _commentary(provider) if answer in c]
    nudges = [c for c in _instructions(provider) if answer in c]
    unspoken = [f for f in client.of("delegation") if f.get("spoken") is False]
    assert len(unspoken) == 1, unspoken
    assert not [name for name, _f in counters if name == "live_result_requeued"], counters
    assert len(appended) == 1, appended
    assert len(nudges) == 1, nudges
    assert unspoken[0]["result_text"] == answer
    if stay:
        assert ("live_result_unspoken", {"reason": "no_epoch_after_nudge"}) in counters
    else:
        assert ("live_result_unspoken", {"reason": "session_closed"}) in counters
    flags = [
        s["assistant_voice"]["spoken"] for s in _delegated_rows(recorded["saves"])
        if "spoken" in (s.get("assistant_voice") or {})
    ]
    assert flags == [False], flags


@pytest.mark.asyncio
async def test_the_delivery_language_cue_is_sent_once_per_call_per_language(monkeypatch):
    """D-P1 notes bounded: an instructions append is permanent session state,
    and the cue scopes itself to every backend result of the call — so two
    Persian results after "answer in English" carry ONE cue, not two."""

    H.fast_clocks(monkeypatch)
    answers = {
        "how much is tuition": "شهریه نوزده هزار دلار است.",
        "who runs admissions": "دکتر ونس مسئول پذیرش است.",
    }

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        return answers[kwargs.get("display_request") or task], "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("how much is tuition", 0, 400))
            provider.push(H.delegation("d1", 420))
            provider.push(H.user_delta("who runs admissions", 3000, 3400))
            provider.push(H.delegation("d2", 3450))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([
        H.config(), {"type": "language_preference", "lang": "en"}, 2.0, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True, speak_on_commentary=True)
    await H.run_relay(client, provider, timeout=8, db_session_id="db-dp1-cue")

    said = _commentary(provider)
    assert any(answers["how much is tuition"] in c for c in said), said
    assert any(answers["who runs admissions"] in c for c in said), said
    cues = [
        e for e in provider.of("session.instructions.append")
        if "toup-live-langcue-" in str(e.get("event_id"))
    ]
    assert len(cues) == 1, cues
    assert "in English" in cues[0]["content"] and "this call" in cues[0]["content"]


@pytest.mark.asyncio
async def test_build_140_gets_one_delivery_ladder_and_the_answer_text(monkeypatch):
    """OC-2: build 140 (no `task_job`) drops a toolless task card after a 30 s
    grace, and a requeued second ladder outlasts it. It keeps the single
    ladder this round inherited, and its verdict carries the answer's text
    (it drew an EMPTY "Result in chat" without it)."""

    H.fast_clocks(monkeypatch)
    counters = H.capture_counters(monkeypatch)
    answer = "The museum opens at ten."

    async def think(*_args, **_kwargs):
        return answer, "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("when does the museum open", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 4.0, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-oc2-build140")

    assert "live_result_requeued" not in [name for name, _ in counters], counters
    assert len([c for c in _commentary(provider) if answer in c]) == 1
    unspoken = [f for f in client.of("delegation") if f.get("spoken") is False]
    assert len(unspoken) == 1 and unspoken[0]["result_text"] == answer, unspoken


# ── Voice round 2026-10-06 (DESIGN-R2 D-R7, D-6b): one answer, once, in
#    order, each on its OWN epoch ───────────────────────────────────────

def _live_output_texts(saves) -> list[str]:
    return [
        str(s.get("assistant_text") or "") for s in saves
        if str(s.get("assistant_ref") or "").startswith("live-output:")
    ]


@pytest.mark.asyncio
async def test_a_result_never_cuts_the_models_own_sentence(monkeypatch, caplog):
    """D-6b (1)+(2)+(3).  The UFC answer of the 2026-10-06 call: `speak()`
    force-closed the model's running epoch at the append («…الان همزمان»), the
    model's NEXT words minted a fresh epoch, and the result's claim bound IT —
    "spoken" on the model's sentence, and the sentence saved as two rows split
    mid-phrase.  Now the append waits for the model's sentence to end (bounded
    — here it is NOT, so the append lands mid-sentence), the model's epoch is
    never cut, and (voice round 2026-10-07, D-R1) the answer is bound by its
    WORDS: "Sure, I am checking UFC and also Inter Miami right now." shares
    one content word with it and proves nothing, "UFC is on Saturday in Las
    Vegas." proves it."""

    import logging

    H.fast_clocks(
        monkeypatch, voice_live_output_epoch_gap_ms=400, voice_live_result_defer_max_ms=60,
    )
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    answer = "The next UFC event is on Saturday in Las Vegas."
    box: dict = {}

    async def think(*_args, **_kwargs):
        # The model is already talking when the answer lands.
        box["p"].push(H.out_text("Sure, I am checking UFC and also", 1000, 1800))
        box["p"].push(H.out_audio())
        await asyncio.sleep(0.15)
        return answer, "test-model"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("what is the next ufc event", 0, 600))
            provider.push(H.delegation("d1", 650))
        elif event["type"] == "session.commentary.append":
            async def finish_then_answer():
                # The model finishes ITS sentence after the append was taken…
                await asyncio.sleep(0.1)
                provider.push(H.out_text(" Inter Miami right now.", 1800, 2500))
                provider.push(H.out_audio())
                # …and only after a pause voices the answer.
                await asyncio.sleep(0.7)
                provider.push(H.out_text("UFC is on Saturday in Las Vegas.", 3200, 4500))
                provider.push(H.out_audio("ANSWER"))
            asyncio.get_running_loop().create_task(finish_then_answer())
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([_config_141(), 2.4, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-r7-nocut")

    messages = [r.getMessage() for r in caplog.records]
    assert any("result append deferred reason=model_speaking" in m for m in messages), messages
    texts = _live_output_texts(recorded["saves"])
    assert "Sure, I am checking UFC and also Inter Miami right now." in texts, texts
    assert not any(t.endswith("and also") for t in texts), "the model's sentence was cut"
    assert "UFC is on Saturday in Las Vegas." not in texts, (
        "the answer's own epoch must fold onto the result row, not stand alone"
    )
    flags = [
        s["assistant_voice"].get("spoken") for s in _delegated_rows(recorded["saves"])
        if "spoken" in (s.get("assistant_voice") or {})
    ]
    assert flags[-1] is True, flags


@pytest.mark.asyncio
async def test_the_next_answer_waits_for_the_one_being_voiced(monkeypatch):
    """D-6b (5).  A delivery resolved `spoken` at its epoch's BEGIN, and the
    pump appended the next answer at once — into the answer being voiced.
    Now the next append waits for that epoch to retire."""

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=400)
    sent_at: dict = {}
    spoke_at: dict = {}
    answers = {
        "how much is tuition": "Tuition is nineteen thousand.",
        "who runs admissions": "Doctor Vance runs admissions.",
    }

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        return answers[kwargs.get("display_request") or task], "test-model"

    H.patch_relay(monkeypatch, think=think)
    clock = {"ms": 5000}

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("how much is tuition", 0, 400))
            provider.push(H.delegation("d1", 420))
            provider.push(H.user_delta("who runs admissions", 3000, 3400))
            provider.push(H.delegation("d2", 3450))
        elif event["type"] == "session.commentary.append":
            key = event["delegation_id"]
            sent_at.setdefault(key, time.monotonic())
            clock["ms"] += 3000
            provider.push(H.out_text(event["content"], clock["ms"], clock["ms"] + 2000))
            provider.push(H.out_audio())
            spoke_at.setdefault(key, time.monotonic())
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([_config_141(), 2.5, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-r7-onevoice")

    assert set(sent_at) == {"d1", "d2"}, sent_at
    first, second = sorted(sent_at, key=sent_at.get)
    assert sent_at[second] - spoke_at[first] >= 0.3, (
        "the second answer was appended while the first was still being voiced"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("wait_s", [3.0, 0.0])
async def test_two_answers_are_spoken_in_the_order_they_were_asked(monkeypatch, caplog, wait_s):
    """D-6b (6).  The queue was FINISH order: a later question whose answer
    came back first was spoken first.  Now it waits (bounded by
    `voice_live_result_order_wait_s`, 0 disables) for the earlier one."""

    import logging

    H.fast_clocks(monkeypatch, voice_live_result_order_wait_s=wait_s)
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        display = kwargs.get("display_request") or task
        if display == "how much is tuition":
            await asyncio.sleep(0.6)
            return "Tuition is nineteen thousand.", "test-model"
        return "Doctor Vance runs admissions.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("how much is tuition", 0, 400))
            provider.push(H.delegation("d1", 420))
            provider.push(H.user_delta("who runs admissions", 3000, 3400))
            provider.push(H.delegation("d2", 3450))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([_config_141(), 2.0, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True, speak_on_commentary=True)
    await H.run_relay(client, provider, timeout=10, db_session_id=f"db-r7-order-{wait_s}")

    said = [c for c in _commentary(provider) if c in (
        "Tuition is nineteen thousand.", "Doctor Vance runs admissions.",
    )]
    held = [r.getMessage() for r in caplog.records if "result held for order" in r.getMessage()]
    if wait_s:
        assert said[:2] == ["Tuition is nineteen thousand.", "Doctor Vance runs admissions."], said
        assert held and "behind=d1" in held[0], held
    else:
        # 0 disables the wait: finish order, as before.
        assert said[:2] == ["Doctor Vance runs admissions.", "Tuition is nineteen thousand."], said
        assert held == []


# ══════════════════════════════════════════════════════════════════════
# Voice round 2026-10-07 (D-R1): the delivery proof reads the WORDS
# ══════════════════════════════════════════════════════════════════════
#
# The corpus below is the phone's own day cache, verbatim: the 2026-10-07
# call ($S/evidence/phone/chat-day-cache.json — ACAD/GEN are the two IELTS
# answers, TELLINGS the five tellings cut out of the 977-character row that
# epoch 7 wrote, the rest the call's other rows) and the 2026-10-03/10-06
# days of the previous round's cache (each delegated answer still has its
# spoken paraphrase saved as a `live_transcript` sibling there — a labelled
# corpus).  `relay-delivery.md` §3 and `critic.md` N1/N2 measured W on it.
_CORPUS = {
 "d1007": {
  "ACAD": "آره، اگه منظورت نمرهٔ ۸ ریدینگه: توی آکادمیک معمولاً باید ۳۵ یا ۳۶ تا از ۴۰ سؤال رو درست بزنی؛ یعنی می‌تونی ۴ یا ۵ تا غلط داشته باشی. برای جنرال معمولاً ۳۷ یا ۳۸ تا درسته، یعنی ۲ یا ۳ غلط. پس برای نمرهٔ ۸، جنرال از نظر تعداد جواب درست سخت‌گیرتره. این عددها ممکنه بسته به نسخهٔ آزمون کمی تغییر کنن؛ این بازه‌ها رو آی‌دی‌پی منتشر کرده.",
  "GEN": "آها، منظورت ریدینگ آیلتس جنرال برای نمرهٔ ۸ بود. از ۴۰ سؤال، معمولاً باید ۳۷ یا ۳۸ تا رو درست بزنی؛ یعنی می‌تونی حدود ۲ تا ۳ تا غلط داشته باشی. طبق توضیح آی‌دی‌پی، این بازه معمولاً بند ۸ می‌شه، ولی مرز دقیق نمره ممکنه بین نسخه‌های آزمون کمی فرق کنه.",
  "TELLINGS": [
   "ببین، برای آیلتس آکادمیک، معمولاً با ۳۵ یا ۳۶ جواب درست از چهل تا، نمره هشت",
   "ببین، برای آیلتس آکادمیک، معمولاً با ۳۵ یا ۳۶ جواب درست از چهل تا، نمره هشت می‌گیری. یعنی حدود ۴ یا ۵ تا می‌تونی غلط داشته باشی. برای جنرال معمولاً باید ۳۷ یا ۳۸ تا درست باشه، یعنی فقط ۲ یا ۳ تا غلط. پس اگه هدف نمره ۸ باشه،",
   "ببین، برای آیلتس آکادمیک، معمولاً با ۳۵ یا ۳۶ جواب درست از چهل تا، نمرهٔ هشت می‌گیری. برای جنرال اما معمولاً باید ۳۷ یا ۳۸ تا درست باشه، یعنی فقط حدود ۲ یا ۳ تا غلط.",
   "اوکی، برای ریدینگ آیلتس جنرال اگه بخوای نمرهٔ هشت بگیری، معمولاً از چهل سؤال باید حدود ۳۷ یا ۳۸ تا رو درست بزنی؛ یعنی می‌تونی تقریباً ۲ تا ۳ تا غلط داشته باشی. آی‌دی‌پی همین حدود رو گفته، ولی مرز دقیق نمره ممکنه بین نسخه‌های آزمون یه کم بالا و پایین بشه.",
   "ببین، برای ریدینگ آیلتس جنرال اگه هدفت نمرهٔ ۸ه، معمولاً باید حدود ۳۷ یا ۳۸ تا از ۴۰ تا سؤال رو درست بزنی. یعنی می‌تونی تقریباً ۲ تا ۳ تا غلط داشته باشی؛ همون بازه‌ایه که آی‌دی‌پی گفته. ولی این مرز دقیق ممکنه بین نسخه‌های مختلف آزمون یه کم بالا پایین بشه."
  ],
  "RAFFEL": "اگه بخوام برای پژوهش روی خودِ مدل‌های زبانی یک نفر رو پیشنهاد بدم، **کالین رَفِل** انتخاب اولمه؛ طبق فهرست فعلی دانشکدهٔ کامپیوتر دانشگاه تورنتو، استاد دانشیاره و «مدل‌های زبانی بزرگ» جزو حوزه‌های پژوهشیشه. ولی «قوی‌ترین» رتبهٔ رسمی نداره. **ژِجینگ جین** هم گزینهٔ خیلی مرتبطیه، مخصوصاً اگه به ایمنی، ارزیابی و عامل‌های مبتنی بر ال‌ال‌ام علاقه داری؛ اون هم در فهرست فعلی دانشکده است.",
  "KIE_EN": "I had to stop here — this took more steps than I can run in one go. Here's where it stands; say \"keep going\" to continue.",
  "KIE_FA": "این دنبال‌کردن بیشتر از حدی بود که بشه یه‌جا تمومش کرد. فعلاً همین اطلاعات رو دارم؛ اگه می‌خوای ادامه بدم،",
  "DRAKE_ROW": "«Drake - Come Thru» رو برات گذاشتم 🎵",
  "DRAKE_EP17": "«Drake - Come Thru» رو برات گذاشتم 🎵",
  "NAME_ROW": "اسم آهنگ «Come Thru» از دریکه.",
  "NAME_EP20": "اسم آهنگ «Come Thru» از دریکه.",
  "STOP_ROW": "موزیک رو قطع کردم.",
  "STOP_EP22": "قطعش کردم.",
  "NEXT_ROW": "دارم می‌رم روی «La Isla Bonita».",
  "NEXT_EP26": "چشم، می‌ذارم بعدیش بیاد.",
  "MAD_ROW": "«Madonna - Live to Tell» داره شروع می‌شه.",
  "PROG": "همین الان دارم یه صفحه رو دقیق می‌خونم ببینم چی میگه.",
  "FRAG": "از این نیاز به",
  "REQ_GEN": "یه چیزی برام سرچ کن. ببین آیاس جنرال. خب؟ اه ریدینگ برای هج شدن میتونی چند تا غلط بزنی",
  "REQ_ACAD": "اوکی. بعد به نظرت همین توی آیسآکادمیک چجوریه",
  "OTHER_ROWS": [
   "Hey, Chatty, تو دیگه خوبی",
   "آره بابا، من خوبم. چه خبرا؟",
   "سلامتی! چی کارا میکنی",
   "همین دور هم نشستیم، یه ذره موزیک گوش میدم اگه تو کاری داری بگو باهم بریم سراغش.",
   "باشه چه",
   "یه چیزی برام سرچ کن. ببین",
   "آیاس",
   "جنرال. خب؟ اه",
   "ریدینگ برای هج شدن میتونی چند تا غلط بزنی",
   "باشه، می‌گردم ببینم چی دقیقاً گفتن.",
   "هنوز دارم می‌گردم، چند ثانیه دیگه بده.",
   "دارم دقیق نگاه می‌کنم، اوکی اینو هم می‌ذارم کنارش که با هم چکشون کنم، یه چند ثانیه بده.",
   "اوکی، الان یه سؤال",
   "... یه سرچ کن ببین اه",
   ". چیز اه قوی‌ترین استاد... اه یو‌اف‌تی توی بحث ال‌ال‌ام",
   "کیه",
   "باشه",
   "، حتماً، دارم تو وب می‌گردم.",
   "هنوز دارم دنبال جواب می‌گردم، یه کم دیگه صبر کن.",
   "هنوز دارم روش کار می‌کنم.",
   "حالا یه دنه آهنگ دریک واسم پخش کن",
   "هنوز دارم روش کار می‌کنم، یک لحظه.",
   "اسم صبحش آهنگ",
   "یک لحظه ببینم منظورت دقیقاً چیه.",
   "استاپش کن",
   "یه آهنگ از مدونا برام پخش کن",
   "می‌رم دنبالش.",
   "بعدی رو پخش کن"
  ]
 },
 "d1003": {
  "tasks": [
   {
    "id": "item_EV1T1eu5n",
    "answer": "آها، منظورت فقط فهمیدنِ مفهوم استارتاپ نیست؛ می‌خوای بدانی عملاً از کجا شروع کنی و چه قدم‌هایی برداری. برای این هدف، پیشنهاد قبلی‌ام را عوض می‌کنم: «Disciplined Entrepreneurship» نوشتهٔ بیل آولت از «The Lean Startup» مرحله‌به‌مرحله‌تر است. کمک می‌کند مشتری مشخصی انتخاب کنی، مشکلش را بسنجی، محصول اولیه بسازی و راه فروش را پیدا کنی. چون Toup را راه انداخته‌ای، مهم‌ترین قدم برای تو احتمالاً صحبت با کاربران واقعی و پیدا کردن آن مشکلی است که حاضرند برای حلش پول بدهند.",
    "request": "خب اگر که بخوام ... چیز کنم بفهمم که چجوری باید استارتاپمو درست کنم",
    "title": "خب اگر که بخوام ... چیز کنم بفهمم که چجوری باید استارتاپمو درست کنم",
    "summary": "روش‌های راه‌اندازی استارتاپ"
   },
   {
    "id": "item_EV1UnwZrr",
    "answer": "آها، منظورت کتاب «How to Start Your Own Business» بود؛ من سؤال قبلیت رو اشتباه فهمیدم. اگر منظورت نسخهٔ انتشارات DK باشه، اون هم گزینهٔ خوبیه: تصویری و ساده توضیح می‌ده چطور یک کسب‌وکار راه بندازی. ولی برای ساختن یک استارتاپ فناوری مثل Toup، هنوز «Disciplined Entrepreneurship» از بیل آولت رو پیشنهاد می‌کنم، چون مسیر پیدا کردن مشتری و ساخت محصول رو قدم‌به‌قدم‌تر پیش می‌بره. راستی «The Lean Startup» کتاب جداگانه‌ایه، نوشتهٔ اریک ریس، نه بیل آولت.",
    "request": "How to start your own business چی",
    "title": "How to start your own business چی",
    "summary": ""
   },
   {
    "id": "item_EV1clAJur",
    "answer": "آره، سؤال اصلیت اینه: حالا که Toup رو ساختی، اول دنبال مشتری بری یا کارمند؟ به نظرم اول دنبال مشتری یا کاربر واقعی برو، نه اینکه فوری تیم رو بزرگ کنی. با چند نفر که فکر می‌کنی Toup مشکلشون رو حل می‌کنه صحبت کن، بذار واقعاً ازش استفاده کنن، و ببین آیا حاضرن دوباره برگردن یا براش پول بدن. وقتی فهمیدی کدوم نیاز تکرار می‌شه و خودت دیگه از پس انجامش برنمیای، اون‌وقت برای همون نقش مشخص نیرو بگیر؛ نه صرفاً برای اینکه استارتاپ «تیم» داشته باشه.",
    "request": "Okay, ببین یه سوال دار الان من یه کتاب ... الان که تو پختم ، به بعدش، به هم کمک کنی. که یعنی ... برای استارتاپم ساختن، حالا بعد دنبال مشتری بگردم یعنی بگم دنبال کارمند بگردم، یعنی",
    "title": "Okay, ببین یه سوال دار الان من یه کتاب ... الان که تو پختم ، به بعدش، به هم کمک…",
    "summary": "کمک به استارتاپ و جذب مشتری"
   }
  ],
  "tellings": [
   {
    "task": "item_EV1T1eu5n",
    "text": "اه، فهمیدم دنبال اینی که بدونی از کجا شروع کنی. به جای اون کتاب قبلی، الان پیشنهادم «Disciplined Entrepreneurship» اثر بیل آولته، چون مرحله به مرحله راه نشونت می‌ده: چطور یه مشتری مشخص پیدا کنی، مشکل اصلی‌ش رو بفهمی، نسخه اولیه محصول رو بسازی، و بفهمی چطوری بفروشیش. چون Toup رو راه انداختی، اولین قدمت بهتره حرف زدن با آدمای واقعی باشه تا بفهمی دقیقاً دنبال حل چه مشکلی هستن."
   },
   {
    "task": "item_EV1UnwZrr",
    "text": "چک می‌کنم.آها، متوجه شدم. منظورِت «How to Start Your Own Business» بود. اگر نسخه‌ی DK مد نظرته، اونم خوبه و به شکل تصویری و ساده توضیح می‌ده. ولی برای اینکه Toup رو بسازی، «Disciplined Entrepreneurship» از بیل آولت هنوز بهتره، چون دقیق‌تر راه می‌ده که چطور مشتری پیدا کنی و محصول بسازی.و یه نکته هم این که «The Lean Startup» یه کتاب جداست، نوشته‌ی اریک ریسه، نه آولت."
   },
   {
    "task": "item_EV1clAJur",
    "text": "آره، سؤال اصلیت اینه: حالا که Toup رو ساختی، اول دنبال مشتری بری یا کارمند؟ به نظرم اول دنبال مشتری یا کاربر واقعی برو. چند نفر که حاضرن از محصولت استفاده کنن پیدا کن، ببین براشون واقعاً ارزش داره یا نه. تا وقتی ندونی دقیقاً"
   }
  ]
 },
 "d1006": {
  "tasks": [
   {
    "id": "item_EVyjLb8UR",
    "answer": "اگه منظورت دانشگاه تورنتو، یوآف‌تیه، به نظرم قوی‌ترین اسم در تقاطع کامپیوترساینس و رباتیک «راکل اورتاسون»ه. استاد دپارتمان کامپیوترساینزه؛ روی بینایی ماشین، یادگیری ماشین و خودروهای خودران کار می‌کنه، شرکت Waabi رو تأسیس کرده و امسال هم عضو انجمن سلطنتی بریتانیا شده. البته «قوی‌ترین» رتبه‌بندی رسمی نداره: اگه منظورت ربات‌های بازویی و دست‌کاری اشیا باشه، انتخابم ممکنه فرق کنه. طبق صفحهٔ خود دانشگاه تورنتو، برای مسیر هوش مصنوعی و رباتیک خودران، اورتاسون نقطهٔ شروع خیلی محکمیه.",
    "request": "منم هستم میگم یه سوال برام سرچ کن ببین قوی ترین استاد روباتیک کامپیوتر ساینس تو یافتی کیه",
    "title": "منم هستم میگم یه سوال برام سرچ کن ببین قوی ترین استاد روباتیک کامپیوتر ساینس تو…",
    "summary": "قوی ترین استاد روباتیک کامپیوتر ساینس"
   },
   {
    "id": "item_EVylJI3XM",
    "answer": "ببخشید، قبلش قاطی کردم. الان «Shadmehr Aghili - Taghdir» داره پخش می‌شه.",
    "request": "نه یه آهنگی برام پخش کن چی شد",
    "title": "نه یه آهنگی برام پخش کن",
    "summary": "پخش آهنگ"
   },
   {
    "id": "item_EVylgDP4P",
    "answer": "ببخشید، از اینجا دکمهٔ توقف پخش رو در اختیار ندارم. لطفاً روی مکثِ پخش‌کننده بزن.",
    "request": "استاپش کن",
    "title": "استاپش کن",
    "summary": "استاپ پخش آهنگ"
   },
   {
    "id": "item_EVymUeDbp",
    "answer": "منظورت مصاحبهٔ کدوم عرفانه؟ اسم کاملش یا لینک خبر رو بگی، محل برگزاریش رو برات پیدا می‌کنم.",
    "request": "اوکی یه چیزی میخواستم ببینم که این مصاحبه عرفان کجا برگزار داره میشه الان",
    "title": "اوکی یه چیزی میخواستم ببینم که این مصاحبه عرفان کجا برگزار داره میشه الان",
    "summary": "مصاحبه عرفان کجا برگزار می‌شود"
   },
   {
    "id": "item_EVymbZyRe",
    "answer": "آها، منظورت «عرفان رضوان» بود. گشتم، ولی با این اسم نتونستم محل مصاحبه‌ای که الان برگزار می‌شه رو تأیید کنم. لینک یا عکسِ اعلامیهٔ مصاحبه رو داری بفرستی؟",
    "request": "رضوان",
    "title": "رضوان",
    "summary": "محل برگزاری مصاحبه عرفان"
   },
   {
    "id": "item_EVynM30re",
    "answer": "آها، فرمول یک! ببخشید، من «مسابقه» رو «مصاحبهٔ عرفان» شنیده بودم. امروز ششم اکتبر مسابقه‌ای در جریان نیست؛ مسابقهٔ بعدی ۹ تا ۱۱ اکتبر توی پیست خیابانی مارینا بِیِ سنگاپور برگزار می‌شه. این رو از تقویم خود فرمول یک چک کردم.",
    "request": "ها",
    "title": "ها",
    "summary": "ها"
   },
   {
    "id": "item_EVyntxJec",
    "answer": "آها، یو‌اف‌سی رو می‌گفتی، نه فرمول یک! جواب قبلیم دربارهٔ فرمول یک بود. طبق برنامهٔ خود یو‌اف‌سی، رویداد بعدی شنبه ۱۰ اکتبر، مبارزهٔ اَلن و دانکن، در سالن متا اِیپکسِ لاس‌وگاس برگزار می‌شه؛ ساعت شروعش هم ۸ شب به وقت تورنتو اعلام شده.",
    "request": "یو‌اف‌سی چی",
    "title": "یو‌اف‌سی چی",
    "summary": "یو‌اف‌سی"
   },
   {
    "id": "item_EVyo79Wd2",
    "answer": "آره، اگه منظورت «میامی اف‌سی» باشه، سه بازی آخرش رو برده: ۳ اکتبر تامپا بِی رو ۲–۱، ۳۰ سپتامبر اسپورتینگ جکس رو ۴–۱، و ۲۶ سپتامبر ایندی الون رو ۲–۰. این نتایج رو از گزارش‌های خود باشگاه چک کردم. اگر منظورت «اینتر میامیِ مسی» بود، اون یه تیم دیگه‌ست؛ آخرین بازیش ۲–۱ به کلمبوس باخته.",
    "request": "میتونی برام یه چیز دیگه هم چک کنی؟ مسابقات... آخرین مسابقاتی که ف... فوتبال میامی اف‌سی میامی",
    "title": "میتونی برام یه چیز دیگه هم چک کنی؟ مسابقات... آخرین مسابقاتی که ف... فوتبال…",
    "summary": "آخرین مسابقات فوتبال میامی اف‌سی"
   },
   {
    "id": "item_EVzu3xmR1",
    "answer": "آهان، احتمالاً اصطلاحی که دنبالش هستی «de-escalate the situation» ـه؛ یعنی پلیس تنشِ موقعیت رو کم کنه تا اوضاع از کنترل خارج نشه. «Defuse the situation» هم نزدیکه، یعنی قضیه رو آروم و جمع‌وجور کنه.",
    "request": "نه اصطلاح دیگه به من بگو درباره",
    "title": "نه اصطلاح دیگه به من بگو درباره",
    "summary": "اصطلاحات دیگر"
   }
  ],
  "tellings": [
   {
    "task": "item_EVyjLb8UR",
    "text": "اگه داریم درباره یوآف‌تی حرف می‌زنیم، یه اسم خیلی مطرح «راکل اورتاسون»ه. استاد کامپیوترساینسه، روی بینایی ماشین و هوش مصنوعی و خودروهای خودران کار می‌کنه و همین امسال هم عضو انجمن سلطنتی بریتانیا شده. البته «قوی‌ترین» رتبه‌بندی رسمی نداره، ولی اگر دنبال مسیر هوش مصنوعی و رباتیک خودران باشی، اورتاسون نقطه شروع خیلی خوبیه."
   },
   {
    "task": "item_EVylJI3XM",
    "text": "از شادمهر عقیلی داره پخش می‌شه."
   },
   {
    "task": "item_EVylgDP4P",
    "text": "ببخشید، از اینجا نمی‌تونم متوقفش کنم؛ لطفاً خودت روی دکمه مکث بزن."
   },
   {
    "task": "item_EVylvv80s",
    "text": "ظه، دارم می‌برم روی بعدی.آماده‌ست، الان «همیشگی» پخش می‌شه."
   },
   {
    "task": "item_EVym3kx7J",
    "text": "می‌ذارم.الان «Hotline Bling» از دریک داره پخش می‌شه."
   },
   {
    "task": "item_EVymUeDbp",
    "text": "منظورت مصاحبهٔ کدوم عرفانه؟ اسم کاملش یا یه لینک خبر رو بدی، سریع محل برگزاریش رو برات پیدا می‌کنم."
   },
   {
    "task": "item_EVynM30re",
    "text": "آها، فرمول یک! ببخشید، من «مسابقه» رو «مصاحبه عرفان» شنیده بودم. امروز ششم اکتبر مسابقه‌ای در جریان نیست؛ مسابقه بعدی نهم تا یازدهم اکتبر توی پیست خیابانی مارینا بی سنگاپور برگزار می‌شه،"
   },
   {
    "task": "item_EVyntxJec",
    "text": "دارم هم UFC هم بازی آخر اینتر میامی رو چک می‌کنم، یه لحظه دیگه. آها، UFC رو می‌گفتی، نه فرمول یک! جواب قبلیم درباره فرمول یک بود. طبق برنامه خود UFC، رویداد بعدی شنبه ۱۰ اکتبره، مبارزه الن و دانکن، توی سالن متا اپکس در لاس وگاس، ساعت ۸ شب به وقت تورنتو."
   },
   {
    "task": "item_EVyo79Wd2",
    "text": "آره، اگه منظورت «میامی اف سی» باشه، سه بازی آخرش رو برده: ۳ اکتبر تامپا بی رو ۲-۱، ۳۰ سپتامبر اسپورتینگ جکس رو ۴-۱، و ۲۶ سپتامبر ایندی الون رو ۲-۰. این نتایج رو از گزارش"
   },
   {
    "task": "item_EVzu3xmR1",
    "text": "اوه، فهمیدم. معمولاً می‌گن «de-escalate the situation»، یعنی تنش رو کم کنن که اوضاع بدتر نشه. یه اصطلاح نزدیک دیگه هم هست: «defuse the situation»، یعنی موقعیت رو آروم کنن و جمعش کنن. این دنبالش بودی؟"
   }
  ]
 }
}


def _w(spoken, answer, others=(), exclude=()):
    from app.services.live_voice_protocol import words_verdict

    return words_verdict(spoken, answer, others, exclude=exclude, every_window=True)


def test_words_verdict_on_the_recorded_corpus():
    """W proves each telling of the recorded calls for its OWN answer and for
    nothing else — the verdicts relay-delivery §3.2 and critic N1 measured."""

    from app.services.live_voice_protocol import (
        words_answer_set, words_content_tokens, words_judge, words_language_mismatch,
    )

    c = _CORPUS["d1007"]
    acad, gen, tellings = c["ACAD"], c["GEN"], c["TELLINGS"]
    req_acad, req_gen = [c["REQ_ACAD"]], [c["REQ_GEN"]]

    # (1) Tellings 1–3 told the Academic answer, 4–5 the General one; with the
    # other answer live each proves ONLY its own.
    for told in tellings[:3]:
        assert _w(told, acad, [gen], req_acad).ok, told[:40]
        assert not _w(told, gen, [acad], req_gen).ok, told[:40]
    for told in tellings[3:]:
        assert _w(told, gen, [acad], req_gen).ok, told[:40]
        assert not _w(told, acad, [gen], req_acad).ok, told[:40]
    # …and why `distinct` exists: the Academic answer restated General's
    # numbers, so ALONE the General telling also "proves" Academic.
    assert _w(tellings[3], acad, (), req_acad).ok

    # (2) Critic C12: inside ONE merged group the fellow member is not an
    # `other` — a telling that says the overlap once proves both members.
    for merged in (tellings[3] + " " + tellings[1], tellings[3] + " " + tellings[0]):
        assert _w(merged, gen, (), req_gen).ok and _w(merged, acad, (), req_acad).ok
        # Phase D words-proof-1 (C12 REVERSED): and with the fellow as an
        # `other` — each member's own facts are in the merged telling.
        assert _w(merged, gen, [acad], req_gen).ok and _w(merged, acad, [gen], req_acad).ok
    # …while ONE member's telling never proves its fellow: General's first
    # sentence alone (the caller cutting in) does not prove Academic.
    assert not _w(tellings[3].split(".")[0], acad, [gen], req_acad).ok

    # (3) Epoch 7 as it grew (the incremental last-window check): Academic is
    # proven inside telling 1; General NEVER during tellings 1–3.
    tokens = words_content_tokens(" ".join(tellings))
    a_set, g_set = words_answer_set(acad, req_acad), words_answer_set(gen, req_gen)
    a_others, g_others = frozenset(words_content_tokens(gen)), frozenset(words_content_tokens(acad))
    first_a = next(n for n in range(1, len(tokens) + 1) if words_judge(tokens[:n], a_set, a_others).ok)
    first_g = next(n for n in range(1, len(tokens) + 1) if words_judge(tokens[:n], g_set, g_others).ok)
    lengths = [len(words_content_tokens(t)) for t in tellings]
    assert first_a <= lengths[0], (first_a, lengths)
    assert first_g > sum(lengths[:3]), (first_g, lengths)
    # The check is a WINDOW, not the whole text: at the end of epoch 7 the
    # last window (telling 5) proves General; the 977 characters as one bag
    # do not (precision 0.48 — Academic's words dilute it).
    assert words_judge(tokens, g_set, g_others).ok

    # (4) Short answers: the play confirmation and the song name are proven by
    # their tellings; the stop/next relay lines are not (the media fold
    # window owns them).
    assert _w(c["DRAKE_EP17"], c["DRAKE_ROW"]).ok
    assert _w(c["NAME_EP20"], c["NAME_ROW"]).ok
    assert not _w(c["STOP_EP22"], c["STOP_ROW"]).ok
    assert not _w(c["NEXT_EP26"], c["NEXT_ROW"]).ok

    # (5) An English answer told in Persian shares no word with it: the
    # cross-language rule (`timing_lang`) is what binds it.
    assert not _w(c["KIE_FA"], c["KIE_EN"]).ok
    assert words_language_mismatch(c["KIE_FA"], c["KIE_EN"])
    assert not words_language_mismatch(c["DRAKE_EP17"], c["DRAKE_ROW"])

    # (6) The epoch-14 false positive («از این نیاز به»), the progress line, and
    # every other row of the call — user rows included — prove no answer.
    answers = {
        "ACAD": acad, "GEN": gen, "RAFFEL": c["RAFFEL"], "KIE_EN": c["KIE_EN"],
        "DRAKE": c["DRAKE_ROW"], "NAME": c["NAME_ROW"], "STOP": c["STOP_ROW"],
        "NEXT": c["NEXT_ROW"], "MAD": c["MAD_ROW"],
    }
    for name, answer in answers.items():
        for row in c["OTHER_ROWS"] + [c["PROG"], c["FRAG"]]:
            assert not _w(row, answer).ok, (name, row[:40])

    # (7) 10-03/10-06: every own telling proves its task under W-novel —
    # except the Shadmehr play, a Latin title told in Persian (critic N2: the
    # media send window binds it, `test_a_media_result_told_in_another_script…`).
    proven = {}
    for day in ("d1003", "d1006"):
        tasks = {t["id"]: t for t in _CORPUS[day]["tasks"]}
        for told in _CORPUS[day]["tellings"]:
            own = tasks.get(told["task"])
            if own is None:
                continue          # a media-control act: no delegated row
            proven[told["task"]] = _w(
                told["text"], own["answer"], (), [own["request"], own["title"], own["summary"]],
            ).ok
    assert len(proven) == 11, proven
    assert proven.pop("item_EVylJI3XM") is False
    assert all(proven.values()), proven
    # (8) Critic N1: the 10-06 UFC telling ("I'm checking UFC AND the last
    # Inter Miami game…") repeats the Miami REQUEST's nouns.  With the
    # request subtracted it does not prove the Miami answer (UFC live); W
    # without the subtraction did.
    tasks = {t["id"]: t for t in _CORPUS["d1006"]["tasks"]}
    ufc, miami = tasks["item_EVyntxJec"], tasks["item_EVyo79Wd2"]
    ufc_told = next(t["text"] for t in _CORPUS["d1006"]["tellings"] if t["task"] == "item_EVyntxJec")
    miami_req = [miami["request"], miami["title"], miami["summary"]]
    assert not _w(ufc_told, miami["answer"], [ufc["answer"]], miami_req).ok
    assert _w(ufc_told, miami["answer"], [ufc["answer"]]).ok

    # (9) …and a telling proves no OTHER task of its day — except the one pair
    # whose answers restate each other (10-03: the second answer repeated the
    # first's book recommendation word for word; the send-time rule is what
    # keeps those apart, since the first was told before the second existed).
    crossed = set()
    for day in ("d1003", "d1006"):
        for told in _CORPUS[day]["tellings"]:
            for task in _CORPUS[day]["tasks"]:
                if task["id"] != told["task"] and _w(
                    told["text"], task["answer"], (),
                    [task["request"], task["title"], task["summary"]],
                ).ok:
                    crossed.add((told["task"], task["id"]))
    assert crossed == {
        ("item_EV1T1eu5n", "item_EV1UnwZrr"), ("item_EV1UnwZrr", "item_EV1T1eu5n"),
    }, crossed
    # (10) Phase D words-proof-7: production ALSO judges the OPEN epoch at
    # every delta (`judge_epoch(retiring=False)`: the last
    # `_WORDS_TAIL_TOKENS` of the words so far, `partial=True`).  The same
    # sweep in that form may cross nothing (9) does not — a prefix shorter
    # than the window was judged as ONE window, so 10-06 «منظورت مصاحبهٔ کدوم
    # عرفانه؟…» proved the never-told item_EVymbZyRe at its 11th word.
    from app.services.live_voice_protocol import _WORDS_TAIL_TOKENS, words_answer_sets

    open_crossed = set()
    for day in ("d1003", "d1006"):
        for told in _CORPUS[day]["tellings"]:
            tokens = words_content_tokens(told["text"])
            for task in _CORPUS[day]["tasks"]:
                if task["id"] == told["task"]:
                    continue
                answer_set, neutral = words_answer_sets(
                    task["answer"], [task["request"], task["title"], task["summary"]],
                )
                if any(
                    words_judge(
                        tokens[:n][-_WORDS_TAIL_TOKENS:], answer_set, every_window=True,
                        neutral=neutral, partial=True,
                    ).ok
                    for n in range(1, len(tokens) + 1)
                ):
                    open_crossed.add((told["task"], task["id"]))
    assert open_crossed <= crossed, open_crossed - crossed
    # …while every own telling is still proven in that form before it ends.
    for day in ("d1003", "d1006"):
        tasks = {t["id"]: t for t in _CORPUS[day]["tasks"]}
        for told in _CORPUS[day]["tellings"]:
            own = tasks.get(told["task"])
            if own is None or told["task"] == "item_EVylJI3XM":
                continue
            tokens = words_content_tokens(told["text"])
            answer_set, neutral = words_answer_sets(
                own["answer"], [own["request"], own["title"], own["summary"]],
            )
            assert words_judge(
                tokens, answer_set, every_window=True, neutral=neutral,
            ).ok, told["task"]


@pytest.mark.parametrize(("spoken", "tokens"), [
    ("۳۵", ["35"]),
    ("سی و پنج", ["35"]),
    ("سی‌وپنج", ["35"]),                       # ZWNJ + «و», joined
    ("thirty-five", ["35"]),
    ("thirty five", ["35"]),
    ("سی", ["30"]),                            # a tens word alone stays what it is
    ("سی و", ["30"]),                          # …dangling «و» too
    ("twenty one", ["21"]),                    # «one» after a tens word is a digit
    ("بیست و یک", ["21"]),
    ("سی و نه تا", ["39"]),                    # «نه» after a tens word is 9, not "no"
    ("ninety-nine", ["99"]),
    ("سی و هفت یا سی و هشت", ["37", "یا", "38"]),
    ("نه", ["نه"]),                            # alone: still "no"
    ("سی وقت", ["30", "وقت"]),                 # «و…» that is not a unit
    # Phase D3 ruling 61 (Phase D2 F6): never across sentence punctuation —
    # these were `32`, `21` and `39`.  A lone «one» / «نه» after the stop is
    # what it is alone (ruling 44): an article, "no".
    ("Listening is thirty. Two sections are harder.", ["listening", "30", "2", "sections", "harder"]),
    ("twenty, one of them is hard", ["20", "one", "hard"]),
    ("سی، نه، چهل", ["30", "نه", "40"]),
    ("سی؛ پنج تا", ["30", "5"]),               # the Persian semicolon too
])
def test_a_compound_number_is_one_number(spoken, tokens):
    """Phase D2 ruling 44 (words-proof-5): «سی و پنج» folded to `30`,`5` and
    never matched an answer's `35` — a telling of the 10-07 Academic answer
    with its numbers spelled out could not prove it."""

    from app.services.live_voice_protocol import words_content_tokens

    assert words_content_tokens(spoken) == tokens


def test_a_telling_with_spelled_out_compound_numbers_proves_its_answer():
    """Ruling 44 on the recorded answer: telling 1 with «۳۵ یا ۳۶» spoken as
    words proves Academic as the digits do (General live — the same `distinct`
    test)."""

    c = _CORPUS["d1007"]
    told = c["TELLINGS"][0].replace("۳۵", "سی و پنج").replace("۳۶", "سی‌وشش")
    assert "۳۵" not in told and "سی و پنج" in told
    assert _w(told, c["ACAD"], [c["GEN"]], [c["REQ_ACAD"]]).ok
    assert _w(c["TELLINGS"][0], c["ACAD"], [c["GEN"]], [c["REQ_ACAD"]]).ok


def test_an_open_epochs_short_prefix_is_not_a_whole_window():
    """Phase D words-proof-7: `judge_epoch` judges the OPEN epoch at every
    delta.  A prefix shorter than the window was judged as one window, so
    the 10-06 «منظورت مصاحبهٔ کدوم عرفانه؟ اسم کاملش یا یه لینک خبر رو بدی…»
    — the model asking which Erfan — proved item_EVymbZyRe (5 of its first
    10 words, precision 0.5), an answer never told.  While the epoch is open
    the words still to come count against it; retired, a short epoch is
    judged on its own length."""

    from app.services.live_voice_protocol import words_content_tokens

    d = _CORPUS["d1006"]
    task = next(t for t in d["tasks"] if t["id"] == "item_EVymbZyRe")
    told = next(t["text"] for t in d["tellings"] if t["task"] == "item_EVymUeDbp")
    session = _bare_session()
    delivery, _ref = _session_delivery(
        session, "item_EVymbZyRe", task["answer"], transcript=task["request"],
    )
    tokens = words_content_tokens(told)
    born = delivery.sent_monotonic + 0.1
    for n in range(1, len(tokens) + 1):
        judged = session.judge_epoch(
            delivery, 9, tokens[:n], born=born, parent="", retiring=False,
        )
        assert judged is None or judged[0] != "words", (n, judged)


def test_a_short_answer_is_not_held_apart_from_an_answer_already_told():
    """Phase D words-proof-3 (P1): a short answer (need ≤ 3) needed every
    proving word absent from every answer resolved in the last 60 s.  The
    10-07 «اسم آهنگ «Come Thru» از دریکه.», told byte for byte 18 s after the
    Drake play was heard, shared {come, thru} with it and kept only «دریکه»
    apart: `unmatched_output`, «نتیجه در گفتگو» and a second row — the 10-07
    double row again.  An answer already TOLD (its epoch retired) cannot be
    the one being told now, so a short answer needs one word of its own
    beyond it; an answer still owed keeps the full `distinct` bar."""

    from app.services.live_voice_protocol import words_content_tokens

    c = _CORPUS["d1007"]
    session = _bare_session()
    drake, _ = _session_delivery(
        session, "relay-play:9", c["DRAKE_ROW"], transcript="حالا یه دنه آهنگ دریک واسم پخش کن",
        media={"type": "youtube", "video_id": "x", "title": "Drake - Come Thru"},
    )
    name, _ = _session_delivery(session, "d-name", c["NAME_ROW"], transcript="اسم صبحش آهنگ")
    born = name.sent_monotonic + 0.5

    def judged(text, epoch):
        out = session.judge_epoch(
            name, epoch, words_content_tokens(text), born=born, parent="", retiring=True,
        )
        return out[0] if out else None

    # The play still OWED: its words keep the name's telling apart from it.
    # (Phase D2 ruling 43: the window is the name's answer word for word and
    # carries none of the play's own words, so the `distinct` TIE-BREAK
    # proves it — `words_tiebreak`, never the plain `words` this row pins.)
    assert judged(c["NAME_EP20"], 20) != "words"
    # Heard 18 s ago, its epoch retired: the telling proves the name.
    drake.resolved, drake.spoken, drake.resolved_at = True, True, time.monotonic() - 18
    assert judged(c["NAME_EP20"], 20) == "words"
    # …and a REPEAT of the play's own line is still not the name.
    assert judged(c["DRAKE_EP17"], 21) != "words"


@pytest.mark.parametrize(("opener", "task_id"), [
    ("آره، یعنی می‌خوای", "ACAD"), ("آره، اگه منظورت اینه که", "ACAD"),
    # real 10-06 acknowledgements that tell nothing, against the 10-06 answers
    ("باشه، اون رو هم می‌گردم.", "item_EVyo79Wd2"),
    ("الان برات چک می‌کنم.", "item_EVymbZyRe"),
    ("الان دارم روش فکر می‌کنم.", "item_EVymbZyRe"),
])
def test_a_caller_cut_filler_opener_never_proves_a_result(opener, task_id):
    """Phase D words-proof-4 (P2): the words proof's stoplist held only
    standard written function words, so spoken-Persian ones («می» — split out
    of every ZWNJ verb —, «آره», «یعنی», «اگه», «منظورت», «الان», «باشه»…)
    counted as content.  A four-word opener cut by the caller met the
    caller-cut floor (3 shared, precision 0.75) on filler alone: the result
    recorded heard (`spoken_interrupted`), no «نتیجه در گفتگو», while the
    caller heard nothing of it.  A real opener of the answer still binds."""

    from app.services.live_voice_protocol import words_content_tokens

    c = _CORPUS["d1007"]
    session = _bare_session()
    if task_id == "ACAD":
        answer, request = c["ACAD"], c["REQ_ACAD"]
    else:
        task = next(t for t in _CORPUS["d1006"]["tasks"] if t["id"] == task_id)
        answer, request = task["answer"], task["request"]
    delivery, _ref = _session_delivery(session, "d-cut", answer, transcript=request)
    born = delivery.sent_monotonic + 0.1
    judged = session.judge_epoch(
        delivery, 3, words_content_tokens(opener), born=born, parent="", retiring=True, cut=True,
    )
    assert judged is None or judged[0] not in ("words", "caller_cut"), (opener, judged)
    if task_id == "ACAD":
        real = session.judge_epoch(
            delivery, 4, words_content_tokens("آکادمیک ۳۵ یا ۳۶ تا درست"),
            born=born, parent="", retiring=True, cut=True,
        )
        assert real and real[0] in ("words", "caller_cut"), real


def _session_delivery(session, delegation_id, answer, *, transcript="", media=None, turn_id=""):
    """A delivery as the pump leaves it: sent, its row's claim NOT in the FIFO."""

    from app.services.live_voice_protocol import DelegatedTask, ResultDelivery

    ref = f"live-delegation:{H.PSID}:{delegation_id}"
    payload = {
        "assistant_text": answer, "media": media,
        "assistant_voice": {"source": "delegated", "delegation_id": delegation_id},
    }
    claim = session.build_commentary_claim(ref, payload, delegation_id=delegation_id)
    delivery = ResultDelivery(
        task=DelegatedTask(
            delegation_id=delegation_id, offset_ms=0, transcript=transcript, turn_id=turn_id,
        ),
        answer=answer, claim=claim, media=bool(media),
    )
    session.deliveries[delegation_id] = delivery
    delivery.sent_monotonic = delivery.started_monotonic = time.monotonic()
    return delivery, ref


@pytest.mark.asyncio
@pytest.mark.parametrize("window_ms", [2500, 0])
async def test_a_media_result_told_in_another_script_binds_by_send_window(monkeypatch, window_ms):
    """Critic N2: «از شادمهر عقیلی داره پخش می‌شه.» told the 10-06 Shadmehr
    play — whose result names the track in LATIN letters — and shares 3 of
    its words (need 5).  A media result binds the first epoch born within
    `voice_live_result_bind_window_ms` of its send instead; with no window it
    is an unproven telling (its own row)."""

    from app.config import settings as app_settings
    from app.services.live_voice_protocol import ClosedEpoch

    tasks = {t["id"]: t for t in _CORPUS["d1006"]["tasks"]}
    play = tasks["item_EVylJI3XM"]
    told = next(t["text"] for t in _CORPUS["d1006"]["tellings"] if t["task"] == "item_EVylJI3XM")
    monkeypatch.setattr(app_settings, "voice_live_result_bind_window_ms", window_ms)
    session = _bare_session()
    delivery, ref = _session_delivery(
        session, "d1", play["answer"], transcript=play["request"],
        media={"type": "youtube", "video_id": "v1", "title": "Shadmehr Aghili - Taghdir"},
    )
    session.active_delivery = delivery
    born = time.monotonic()
    session.adapter.provider_event(H.out_text(told, 900, 1500))
    await session.persist_spoken(ClosedEpoch(
        epoch=1, output_id=f"live:{H.PSID}:1", text=told, interrupted=False,
        played_ms=900, born_monotonic=born,
    ))

    if window_ms:
        assert delivery.spoken and delivery.words_reason == "send_window"
        assert f"live-output:{H.PSID}:1" not in session.persistence.keys()
        assert [r for r in session.persistence.records if r.key == ref]
    else:
        assert not delivery.resolved
        assert f"live-output:{H.PSID}:1" in session.persistence.keys()


@pytest.mark.asyncio
async def test_an_answer_voiced_before_its_ack_is_spoken_once_and_folded(monkeypatch, caplog):
    """10-07 problem 1: GPT-Live started voicing the Academic answer 118 ms
    BEFORE the relay processed its ack, so "the first epoch after the ack"
    never came: nudge, requeue, «نتیجه در گفتگو» — all inside the telling.
    The words prove it in the epoch that says it: spoken, no nudge, and the
    epoch folds onto the task row (no `live-output:` copy of the answer)."""

    import logging

    H.fast_clocks(monkeypatch)
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    answer = "The Robarts Library closes at eleven tonight."
    recorded = H.patch_relay(monkeypatch, think=_fixed_think(answer))

    def on_send(provider, event):
        kind = event["type"]
        if kind == "session.start":
            provider.push(H.user_delta("when does robarts close", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif kind == "session.commentary.append" and answer in event["content"]:
            provider.push(H.out_text("Robarts Library closes at eleven tonight.", 900, 1500))
            provider.push(H.out_audio("ANSWER"))
            asyncio.get_running_loop().call_later(0.15, provider.push, {
                "type": "session.commentary.appended", "client_event_id": event["event_id"],
            })
        elif kind in ("session.instructions.append", "session.commentary.append"):
            provider.push({
                "type": kind.replace(".append", ".appended"),
                "client_event_id": event["event_id"],
            })
        elif kind == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([_config_141(), 2.0, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-w-preack")

    messages = [r.getMessage() for r in caplog.records]
    assert [m for m in messages if "result spoken id=d1" in m and "reason=words" in m], messages
    assert not [c for c in _instructions(provider) if answer in c], "a telling was nudged"
    assert not [m for m in messages if "reason=unacked" in m]
    assert not [t for t in _live_output_texts(recorded["saves"]) if "Robarts" in t], (
        "the telling was saved as a second row"
    )
    rows = _delegated_rows(recorded["saves"])
    assert rows[-1]["assistant_voice"].get("spoken") is True
    assert any("epoch" in (r.get("assistant_voice") or {}) for r in rows), rows
    assert not [f for f in client.of("delegation") if f.get("spoken") is False]


def _fixed_think(answer, delay=0.0):
    async def think(*_args, **_kwargs):
        if delay:
            await asyncio.sleep(delay)
        return answer, "test-model"
    return think


_TUITION_ANSWER = "Tuition is nineteen thousand dollars a year."
_WEATHER_REPLY = "It is sunny and warm in Toronto today."


async def _newer_request_after_a_result(
    monkeypatch, caplog, *, tell, db, reply_after=0.35, pieces=1, request="how much is tuition",
    answer=_TUITION_ANSWER, newer="what is the weather today", client=None, receipt_at=None,
):
    """d1's answer is appended; the caller then asks something NEW (turn 2,
    closed before the model speaks).  `tell` = what the model says next: the
    reply to turn 2 (the 10-07 §1(f) Drake shape) or the RESULT itself.  The
    production reply ladder is on (regen scaled to 0.6 s, no backstop).
    `pieces` > 1 streams `tell` in that many deltas inside ONE epoch.
    `answer` = d1's answer, `newer` = turn 2's words.  `client` replaces the
    phone (default: build 141's, no receipts); `receipt_at` = seconds into
    the call at which that default phone acks epoch 1 as played."""

    import logging

    H.fast_clocks(
        monkeypatch, voice_live_uplink_keeper_enabled=True, voice_live_reply_regen_after_s=0.6,
        **({"voice_live_output_epoch_gap_ms": 300} if pieces > 1 else {}),
    )
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    recorded = H.patch_relay(monkeypatch, think=_fixed_think(answer))

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta(request, 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.commentary.append" and answer[:24] in event["content"]:
            async def caller_asks_something_else():
                provider.push(H.user_delta(newer, 1000, 1500))
                await asyncio.sleep(reply_after)
                if pieces == 1:
                    provider.push(H.out_text(tell, 1700, 2300))
                    provider.push(H.out_audio("NEXT"))
                    return
                words = tell.split(" ")
                step = -(-len(words) // pieces)
                t = 1700
                for i in range(0, len(words), step):
                    last = i + step >= len(words)
                    provider.push(H.out_text(" ".join(words[i:i + step]) + ("" if last else " "), t, t + 150))
                    provider.push(H.out_audio("NEXT"))
                    t += 150
                    await asyncio.sleep(0.05)
            asyncio.get_running_loop().create_task(caller_asks_something_else())
        elif event["type"] == "session.close":
            provider.push(H.closed())

    if client is None:
        script: list = [_config_141(), 2.8, {"type": "stop"}]
        if receipt_at is not None:
            script[1:2] = [receipt_at, NR._idle(f"live:{H.PSID}:1"), 2.8 - receipt_at]
        client = H.FakeClient(script)
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=12, db_session_id=db)
    messages = [r.getMessage() for r in caplog.records]
    turn2 = f"live-utt:{H.PSID}:2"
    reasked = [m for m in messages if f"reask turn={turn2} outcome=accepted" in m]
    return client, recorded, messages, turn2, reasked


@pytest.mark.asyncio
@pytest.mark.parametrize("reply_after", [0.35, 0.0], ids=["after_close", "while_open"])
async def test_a_reply_to_a_request_asked_after_a_result_keeps_its_own_turn(
    monkeypatch, caplog, reply_after,
):
    """delivery-ladder-2 (P1): a result's bare parent claim is armed at SEND
    and `on_provider_output` preferred the FIFO head over the causal parent —
    so the model's answer to a request asked AFTER the append (10-07 §1(f):
    the Drake ask during the «کیه» result) was parented on the RESULT's turn,
    on the row and on the wire, counted as claimed, left owed, and 0.6 s
    (3 s in production) later the ladder asked the model to answer it AGAIN.
    The newer request releases the claim: the reply keeps its turn."""

    client, recorded, messages, turn2, reasked = await _newer_request_after_a_result(
        monkeypatch, caplog, tell=_WEATHER_REPLY, db=f"db-parent-newer-reply-{reply_after}",
        reply_after=reply_after,
    )
    rows = {
        str(s.get("assistant_text") or ""): (s.get("assistant_voice") or {}).get("parent_user_turn_id")
        for s in recorded["saves"] if str(s.get("assistant_ref") or "").startswith("live-output:")
    }
    assert rows.get(_WEATHER_REPLY) == turn2, rows
    wire = {f.get("parent_user_turn_id") for f in client.of("response_text")
            if _WEATHER_REPLY[:12] in str(f.get("text") or "")}
    assert wire == {turn2}, wire
    assert any(f"result parent claim released id=d1 turn_id={turn2}" in m for m in messages), messages
    assert reasked == [], reasked


@pytest.mark.asyncio
async def test_a_result_told_after_a_newer_request_is_not_that_requests_reply(monkeypatch, caplog):
    """delivery-ladder-2, the other order: the model tells the RESULT first.
    With the parent claim released, that epoch took turn 2 as its cause and was
    counted as turn 2's reply — but a result telling is relay speech, never a
    reply.  Its words bind it to d1 and turn 2 is owed again, so the ladder may
    still ask for it (what the claimed epoch gave it before the release)."""

    client, recorded, messages, turn2, reasked = await _newer_request_after_a_result(
        monkeypatch, caplog, tell=_TUITION_ANSWER, db="db-parent-newer-result",
    )
    assert any("result spoken id=d1" in m and "reason=words" in m for m in messages), messages
    turn1 = f"live-utt:{H.PSID}:1"
    assert any(f"result epoch re-attributed id=d1 epoch=1 from={turn2} to={turn1}" in m
               for m in messages), messages
    assert any(f"reply owed reopened turn_id={turn2}" in m for m in messages), messages
    assert reasked, [m for m in messages if "reask" in m]


@pytest.mark.asyncio
@pytest.mark.parametrize(("tell", "pieces"), [
    (_WEATHER_REPLY + " And " + _TUITION_ANSWER, 1),      # the reply, then the result
    (_TUITION_ANSWER + " And " + _WEATHER_REPLY, 1),      # the result, then the reply
    # …streamed: the result's words prove it BEFORE the reply's words exist.
    (_TUITION_ANSWER + " And " + _WEATHER_REPLY, 4),
], ids=["reply_then_result", "result_then_reply", "result_then_reply_streamed"])
async def test_one_epoch_that_answers_the_newer_request_and_tells_the_result_is_not_asked_again(
    monkeypatch, caplog, tell, pieces,
):
    """Phase D3 ruling 59 (Phase D2 F5): delivery-ladder-2's re-attribution
    re-opened the newer request whenever the result's words proved an epoch
    taken from it — even when that SAME epoch also answered it (GPT-Live
    addressing both in one go).  The ladder then asked the model to answer
    turn 2 again: an answer the caller had just heard, told twice.  The
    epoch is still re-parented to the result's turn; the turn stays answered
    (`reply_kept=1`).  A streamed telling is judged WHOLE — at the bind its
    reply words have not arrived yet.  `result_only`
    (`test_a_result_told_after_a_newer_request_is_not_that_requests_reply`)
    still re-asks."""

    client, recorded, messages, turn2, reasked = await _newer_request_after_a_result(
        monkeypatch, caplog, tell=tell, pieces=pieces,
        db=f"db-dl2-combined-{pieces}-{abs(hash(tell)) % 10**6}",
    )
    turn1 = f"live-utt:{H.PSID}:1"
    assert any("result spoken id=d1" in m and "reason=words" in m for m in messages), messages
    assert any(
        m.startswith(f"[LIVE] result epoch re-attributed id=d1 epoch=1 from={turn2} to={turn1} ")
        and m.endswith(" reply_kept=1") for m in messages
    ), [m for m in messages if "re-attributed" in m]
    assert not [m for m in messages if "reply owed reopened" in m], messages
    assert reasked == [], reasked


_KEPT_SHAPES = {
    # the control: an ordinary reply to turn 2, nothing re-attributed
    "reply_only": _WEATHER_REPLY,
    # ruling 59's two kept shapes: ONE epoch answers turn 2 AND tells d1
    "reply_then_result": _WEATHER_REPLY + " And " + _TUITION_ANSWER,
    "result_then_reply": _TUITION_ANSWER + " And " + _WEATHER_REPLY,
    # …and the result alone: re-opened, re-asked by the relay (ruling 59)
    "result_only": _TUITION_ANSWER,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("heard", [False, True], ids=["no_receipt", "heard_receipt"])
@pytest.mark.parametrize("shape", list(_KEPT_SHAPES))
async def test_a_kept_reply_is_the_newer_turns_answer_for_the_phone(monkeypatch, caplog, shape, heard):
    """Phase D4 ruling 64 (Phase D3 review F2 = R3-APP-1): ruling 59 kept the
    newer turn answered for the relay's own ladder only.  The evidence every
    other recovery path reads (`turn_output_evidence`) still dropped the
    re-parented epoch, and the settled `output_attribution` — sent at the
    BIND, before the relay had decided — named the RESULT's turn: the app's
    turn watch withdrew turn 2's heard coverage, asked for it by id 12 s
    later, and the relay ACCEPTED it (`reason=unconfirmed`, "your earlier
    answer was sent, but the caller's phone never confirmed playing it…
    recap") — the second answer ruling 59 removed, through the phone.

    The app is in the loop here (its feature list): once epoch 1 retires the
    phone acks it as played (`heard_receipt`) or not, then asks turn 2 by id
    — what the watch sends.  A kept epoch is now turn 2's answer exactly like
    an ordinary reply (`reply_only`, the control): HEARD → `answered`; with
    no receipt both are UNKNOWN → `accepted reason=unconfirmed` (addendum 4
    §3.2, supervisor 18:16 — only a HEARD answer answers a turn; the recap
    instruction's premise is then true).  The settled attribution is sent
    after the decision, at the epoch's retirement (after its segment
    completes), naming turn 2 when kept and the result's turn otherwise
    (`result_only`, which the relay itself re-asks)."""

    turn1, turn2 = f"live-utt:{H.PSID}:1", f"live-utt:{H.PSID}:2"

    async def script():
        # epoch 1 retires (the gap: this phone sends nothing on its own) …
        assert await NR._wait(lambda: [
            f for f in phone.of("speech_segment_complete") if f.get("epoch") == 1
        ], 6), phone.frames
        await asyncio.sleep(0.1)
        if heard:                                  # …a late "played" receipt
            phone.push_frame(NR._idle(f"live:{H.PSID}:1"))
        await asyncio.sleep(0.8)
        phone.reask(turn2, "what is the weather today")
        assert await NR._wait(lambda: phone.of("reask_result"), 4)

    phone = NR.Phone([NR._config(NR.APP_FEATURES), script, 0.5, {"type": "stop"}], tape=[])
    _client, _recorded, messages, _turn2, _reasked = await _newer_request_after_a_result(
        monkeypatch, caplog, tell=_KEPT_SHAPES[shape], client=phone,
        db=f"db-r64-reask-{shape}-{int(heard)}",
    )
    kept = shape in ("reply_then_result", "result_then_reply")
    assert any("result spoken id=d1" in m and "reason=words" in m for m in messages) is (shape != "reply_only")
    assert any("reply_kept=1" in m for m in messages) is kept, [m for m in messages if "re-attributed" in m]
    by_id = [m for m in messages if f"reask turn={turn2} " in m and " via=id " in m]
    results = [(f.get("user_turn_id"), f.get("outcome")) for f in phone.of("reask_result")]
    if shape == "result_only":
        assert any(f"reply owed reopened turn_id={turn2}" in m for m in messages), messages
        assert any(f"reask turn={turn2} outcome=accepted" in m and "via=relay:regen" in m
                   for m in messages), messages
        assert len(by_id) == 1 and " outcome=duplicate reason=relay_owned " in by_id[0], by_id
    elif heard:
        assert len(by_id) == 1 and " outcome=answered reason=epoch " in by_id[0], by_id
        assert results == [(turn2, "answered")], results
        # …and no reask instruction (no "never confirmed playing it" recap)
        assert not [m for m in messages if f"reask turn={turn2} outcome=accepted" in m], messages
        assert not [m for m in messages if "delivery=reask" in m], messages
    else:
        assert len(by_id) == 1 and " outcome=accepted reason=unconfirmed " in by_id[0], by_id
    settled = [
        (index, f.get("parent_user_turn_id")) for index, f in enumerate(phone.frames)
        if f.get("type") == "output_attribution" and f.get("settled") and f.get("epoch") == 1
    ]
    if shape == "reply_only":
        assert not [p for _i, p in settled if p != turn2], settled
        return
    retired = next(index for index, f in enumerate(phone.frames)
                   if f.get("type") == "speech_segment_complete" and f.get("epoch") == 1)
    assert [p for _i, p in settled] == [turn2 if kept else turn1], settled
    assert settled[0][0] > retired, (settled, retired)


@pytest.mark.asyncio
@pytest.mark.parametrize("heard", [False, True], ids=["no_receipt", "heard_receipt"])
@pytest.mark.parametrize("shape", ["reply_only", "reply_then_result", "result_then_reply"])
async def test_a_kept_reply_is_not_carried_as_unanswered(monkeypatch, caplog, shape, heard):
    """Phase D4 ruling 64, the teardown half (Phase D3 review F2's carry):
    `stranded_turn_record` reads the same evidence.  A kept epoch the phone
    acked as played answers turn 2, so nothing is carried to the next
    socket; with no receipt it is UNKNOWN and carried `conditional` —
    exactly as the ordinary reply (`reply_only`) is."""

    turn2 = f"live-utt:{H.PSID}:2"
    _client, _recorded, messages, _turn2, reasked = await _newer_request_after_a_result(
        monkeypatch, caplog, tell=_KEPT_SHAPES[shape], receipt_at=2.2 if heard else None,
        db=f"db-r64-carry-{shape}-{int(heard)}",
    )
    assert any("reply_kept=1" in m for m in messages) is (shape != "reply_only")
    assert reasked == [], reasked
    carried = [m for m in messages if f"unanswered turn carried turn={turn2}" in m]
    assert bool(carried) is (not heard), carried


_D1007 = _CORPUS["d1007"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("answer", "asked", "newer", "tell", "outside", "shared"), [
    # a telling with the framing a voice model gives it (Phase D3 review F1's
    # `framed_en`): 4 words outside the result, none of them the weather
    (_TUITION_ANSWER, "how much is tuition", "what is the weather today",
     "Okay, I checked their official website. " + _TUITION_ANSWER, 4, 0),
    # the founder's own: the RECORDED 10-07 General answer, told by its
    # RECORDED telling (T3) after the caller asked something new
    (_D1007["GEN"], _D1007["REQ_GEN"], "هوای تورنتو امروز چطوره؟", _D1007["TELLINGS"][3], 11, 0),
    # …and after a bare «کیه؟», which names nothing the telling says
    (_D1007["GEN"], _D1007["REQ_GEN"], "کیه؟", _D1007["TELLINGS"][3], 11, 0),
    # the framing shares "what" with the newer request — a word that names
    # nothing (`_FUNCTION_WORDS`)
    (_TUITION_ANSWER, "how much is tuition", "what is the weather today",
     "Okay, here's what I found on their official website. " + _TUITION_ANSWER, 6, 0),
    # the recorded telling shares «اگه» with the newer request — a spoken
    # Persian function word (`_WORDS_SPOKEN_STOPWORDS`)
    (_D1007["GEN"], _D1007["REQ_GEN"], "اگه امروز بارون بیاد چی؟", _D1007["TELLINGS"][3], 11, 0),
    # it NAMES the newer request («weather») but only acknowledges it: two
    # words outside the result — ruling 59's bound still re-opens
    (_TUITION_ANSWER, "how much is tuition", "what is the weather today",
     "Weather in a moment. " + _TUITION_ANSWER, 2, 1),
], ids=["framed_en", "recorded_general", "recorded_general_kiye", "framing_what",
        "recorded_general_if", "acknowledged_only"])
async def test_a_framed_result_telling_after_a_newer_request_still_reasks_it(
    monkeypatch, caplog, answer, asked, newer, tell, outside, shared,
):
    """Phase D3 review F1 (P1): ruling 59 read "≤2 content tokens outside the
    result" as "the result telling and nothing more" — but a REAL telling
    carries framing (16 of the 20 recorded own-proven tellings have >2 tokens
    outside), so a result-only epoch read as "also answered the newer
    request": `reply_kept=1`, never re-asked — dropped on the web (no reask
    of its own), and on the phone asked 12 s later as a "recap" of an answer
    that never existed.  Kept now needs a word outside the result that the
    NEWER turn itself said and that names something; these re-open and
    re-ask.  `test_one_epoch_that_answers_the_newer_request_and_tells_the_result_is_not_asked_again`
    (they share «today») stays kept."""

    client, recorded, messages, turn2, reasked = await _newer_request_after_a_result(
        monkeypatch, caplog, tell=tell, answer=answer, request=asked, newer=newer,
        db=f"db-dl2-framed-{abs(hash((newer, tell))) % 10**6}",
    )
    turn1 = f"live-utt:{H.PSID}:1"
    assert any("result spoken id=d1" in m and "reason=words" in m for m in messages), messages
    line = [m for m in messages if "result epoch re-attributed id=d1" in m]
    assert line == [
        f"[LIVE] result epoch re-attributed id=d1 epoch=1 from={turn2} to={turn1}"
        f" outside={outside} newer_shared={shared}"
    ], line
    assert any(f"reply owed reopened turn_id={turn2}" in m for m in messages), messages
    assert reasked, [m for m in messages if "reask" in m]


@pytest.mark.asyncio
@pytest.mark.parametrize("newer", [
    "what is the weather today",
    # a follow-up on the same subject: «students» is a word of the newer
    # request AND of the result's own — it is the result's, so it does not
    # make the telling a reply (Phase D3 review F1's newer-request test)
    "and what about domestic students",
], ids=["unrelated", "same_subject_follow_up"])
async def test_a_result_telling_that_restates_its_request_still_reopens_the_newer_one(
    monkeypatch, caplog, newer,
):
    """Ruling 59's other half: the words a result telling may say beyond
    its answer include its OWN request's («Tuition at the University of
    Toronto for international students…» for "how much is tuition at the
    university of toronto for international students") — that epoch is the
    result telling and nothing more, so the newer request is still owed and
    re-asked."""

    request = "how much is tuition at the university of toronto for international students"
    tell = (
        "Tuition at the University of Toronto for international students is nineteen "
        "thousand dollars a year."
    )
    client, recorded, messages, turn2, reasked = await _newer_request_after_a_result(
        monkeypatch, caplog, tell=tell, request=request, newer=newer,
        db=f"db-dl2-restated-request-{abs(hash(newer)) % 10**6}",
    )
    assert any("result spoken id=d1" in m and "reason=words" in m for m in messages), messages
    assert not [m for m in messages if "reply_kept=1" in m], messages
    assert any(f"reply owed reopened turn_id={turn2}" in m for m in messages), messages
    assert reasked, [m for m in messages if "reask" in m]


@pytest.mark.asyncio
async def test_a_media_results_send_window_never_binds_a_reply_to_a_newer_request(
    monkeypatch, caplog,
):
    """words-proof-8 (P2): a MEDIA result binds by send window the first epoch
    born ≤ 2.5 s after its send "not parented on a newer caller turn" (D-R1
    (1), critic N2) — but `judge_epoch` read that from the epoch's parent,
    which the result's send-armed parent claim had just overwritten with the
    play's turn.  The reply to a request asked right after the play's append
    was bound to the play with ZERO shared words and folded onto its row: its
    words in no row of the day.  With the claim released for the newer
    request, the reply keeps its turn, the send window refuses it, and it is
    its own row."""

    import logging

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        if out is not None:
            out["media"] = {"type": "youtube", "video_id": "HALO0000001", "title": "Beyonce - Halo"}
        return "Playing Halo by Beyonce for you now.", "m"

    recorded = H.patch_relay(monkeypatch, think=think)
    fired = {"n": 0}

    async def caller_asks_something_else(provider):
        provider.push(H.user_delta("what is the weather today", 1000, 1500))
        await asyncio.sleep(0.35)
        provider.push(H.out_text(_WEATHER_REPLY, 1700, 2300))
        provider.push(H.out_audio("REPLY"))

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("play something by beyonce", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.commentary.append" and "Halo" in event["content"] and not fired["n"]:
            fired["n"] = 1
            asyncio.get_running_loop().create_task(caller_asks_something_else(provider))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([_config_141(), 2.5, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-media-window-newer")
    messages = [r.getMessage() for r in caplog.records]
    assert not [m for m in messages if "result spoken id=d1" in m and "reason=send_window" in m], messages
    rows = {
        str(s.get("assistant_text") or ""): (s.get("assistant_voice") or {}).get("parent_user_turn_id")
        for s in recorded["saves"] if str(s.get("assistant_ref") or "").startswith("live-output:")
    }
    assert rows.get(_WEATHER_REPLY) == f"live-utt:{H.PSID}:2", rows


@pytest.mark.asyncio
async def test_unrelated_output_after_the_ack_ends_unmatched_without_a_nudge(
        monkeypatch, caplog):
    """10-07 epochs 14/21: an epoch born after the ack that is NOT the answer
    used to settle it ("spoken"), and the task row was revised with it.  Now
    it proves nothing, stays its own row, and — because the model DID speak —
    the verdict is `unmatched_output` with no nudge (a nudge is what made the
    model drop a sentence and restart)."""

    import logging

    H.fast_clocks(monkeypatch)
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    counters = H.capture_counters(monkeypatch)
    answer = "Tuition is nineteen thousand dollars a year."
    recorded = H.patch_relay(monkeypatch, think=_fixed_think(answer))
    unrelated = "Sure, I can also play some music for you later."

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("how much is tuition", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.commentary.append" and answer in event["content"]:
            async def say_something_else():
                await asyncio.sleep(0.1)
                provider.push(H.out_text(unrelated, 900, 1500))
                provider.push(H.out_audio("OTHER"))
            asyncio.get_running_loop().create_task(say_something_else())
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([_config_141(), 2.5, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-w-unmatched")

    messages = [r.getMessage() for r in caplog.records]
    assert not [m for m in messages if "result spoken id=d1" in m], messages
    assert not [c for c in _instructions(provider) if answer in c], "nudged after the model spoke"
    assert ("live_result_unspoken", {"reason": "unmatched_output"}) in counters, counters
    assert [m for m in messages if "result epoch unmatched id=d1" in m], messages
    assert unrelated in _live_output_texts(recorded["saves"])
    assert not [
        r for r in _delegated_rows(recorded["saves"]) if "epoch" in (r.get("assistant_voice") or {})
    ], "the task row was revised with an epoch that did not tell it"
    unspoken = [f for f in client.of("delegation") if f.get("spoken") is False]
    assert len(unspoken) == 1 and unspoken[0]["result_text"] == answer


@pytest.mark.asyncio
async def test_a_long_telling_is_never_nudged_requeued_or_written_off(monkeypatch, caplog):
    """The 83-second epoch 7, compressed: an answer told over more than two
    speak budgets in ONE open epoch.  It is proven by its words early; no
    nudge, no second copy, no `spoken:false`, one folded row."""

    import logging

    H.fast_clocks(monkeypatch)
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    answer = (
        "Reading for an eight in General needs thirty seven or thirty eight correct "
        "answers out of forty, so you can miss two or three questions at most."
    )
    recorded = H.patch_relay(monkeypatch, think=_fixed_think(answer))

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("how many mistakes for an eight", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.commentary.append" and answer in event["content"]:
            async def tell_slowly():
                words = answer.split()
                t = 900
                for index in range(0, len(words), 2):
                    provider.push(H.out_text(" ".join(words[index:index + 2]) + " ", t, t + 100))
                    provider.push(H.out_audio("TELL"))
                    t += 100
                    await asyncio.sleep(0.05)
            asyncio.get_running_loop().create_task(tell_slowly())
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([_config_141(), 2.5, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-w-long")

    messages = [r.getMessage() for r in caplog.records]
    assert not _instructions(provider), _instructions(provider)
    assert len([c for c in _commentary(provider) if answer in c]) == 1
    assert len([m for m in messages if "result spoken id=d1" in m]) == 1, messages
    assert not [f for f in client.of("delegation") if f.get("spoken") is False]
    assert not [t for t in _live_output_texts(recorded["saves"]) if "General" in t]
    rows = _delegated_rows(recorded["saves"])
    assert rows[-1]["assistant_voice"].get("spoken") is True


@pytest.mark.asyncio
@pytest.mark.parametrize("window_ms", [2500, 100])
async def test_a_cross_language_telling_binds_by_timing(monkeypatch, caplog, window_ms):
    """The 10-07 «کیه» result was English and told in Persian: no shared
    word, so W alone would write it off.  An epoch in another SCRIPT born
    within `voice_live_result_bind_window_ms` of the send binds it
    (`timing_lang`); born outside the window it does not."""

    import logging

    H.fast_clocks(monkeypatch, voice_live_result_bind_window_ms=window_ms)
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    answer = "The museum opens at ten in the morning."
    H.patch_relay(monkeypatch, think=_fixed_think(answer))

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("when does the museum open", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.commentary.append" and answer in event["content"]:
            async def tell_in_persian():
                await asyncio.sleep(0.3)
                provider.push(H.out_text("موزه صبح ساعت ده باز می‌شه.", 900, 1500))
                provider.push(H.out_audio("FA"))
            asyncio.get_running_loop().create_task(tell_in_persian())
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([_config_141(), 2.0, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id=f"db-w-lang-{window_ms}")

    spoken = [
        m for m in (r.getMessage() for r in caplog.records)
        if "result spoken id=d1" in m and "reason=timing_lang" in m
    ]
    if window_ms > 300:
        assert spoken
    else:
        assert not spoken
        assert [f for f in client.of("delegation") if f.get("spoken") is False]


_TUITION, _VANCE = "Tuition is nineteen thousand.", "Doctor Vance runs admissions."


def _two_asks(d1_delay, d2_delay):
    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        display = kwargs.get("display_request") or task
        if display == "how much is tuition":
            await asyncio.sleep(d1_delay)
            return _TUITION, "test-model"
        await asyncio.sleep(d2_delay)
        return _VANCE, "test-model"
    return think


def _two_asks_on_send(provider, event):
    if event["type"] == "session.start":
        provider.push(H.user_delta("how much is tuition", 0, 400))
        provider.push(H.delegation("d1", 420))
        provider.push(H.user_delta("who runs admissions", 3000, 3400))
        provider.push(H.delegation("d2", 3450))
    elif event["type"] == "session.close":
        provider.push(H.closed())


@pytest.mark.asyncio
async def test_ask_order_survives_the_finishing_window(monkeypatch, caplog):
    """10-07 C3: the order hold released the moment the earlier task's agent
    turn RETURNED (`entry.ended`), while its `deliver_result` was still
    several awaits away at the end of `finish_delegation` — and the
    later-asked Academic answer went out first.  Slow that window down: the
    earlier-asked answer still goes first (`reason=finishing`)."""

    import logging

    import app.services.live_voice_protocol as live

    H.fast_clocks(monkeypatch, voice_live_result_order_wait_s=3.0)
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    H.patch_relay(monkeypatch, think=_two_asks(0.6, 0.0))
    original = live._LiveSession.release_dependents

    async def slow_release(self, task, outcome, accepted):
        # The awaits between `supervisor.running.pop` (the top of
        # `finish_delegation`) and its last line, `deliver_result`.
        if task.delegation_id == "d1":
            await asyncio.sleep(0.4)
        return await original(self, task, outcome, accepted)

    monkeypatch.setattr(live._LiveSession, "release_dependents", slow_release)
    client = H.FakeClient([_config_141(), 2.5, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_two_asks_on_send, auto_ack=True, speak_on_commentary=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-w-finishing")

    said = [c for c in _commentary(provider) if c in (_TUITION, _VANCE)]
    assert said[:2] == [_TUITION, _VANCE], said
    held = [r.getMessage() for r in caplog.records if "result held for order" in r.getMessage()]
    assert held and "behind=d1" in held[0], held


@pytest.mark.asyncio
@pytest.mark.parametrize("speaking,second_delay", [(False, 0), (True, 0), (False, 0.4), (False, 0.6)])
async def test_two_results_finishing_together_go_out_as_one_merged_append_in_ask_order(
        monkeypatch, caplog, speaking, second_delay):
    """D-R1 (3): two answers finishing within the merge window go out as ONE
    append group in ask order — a pair cue, then each commentary — and one
    epoch telling both folds onto BOTH rows, each keeping its own parent.
    Critic U1: with the model mid-sentence the cue is NOT sent (an
    instruction then is what made it drop the sentence and restart); the two
    commentaries still go out together."""

    import logging

    H.fast_clocks(
        monkeypatch, voice_live_result_order_wait_s=3.0, voice_live_result_merge_window_s=5.0,
        voice_live_output_epoch_gap_ms=300,
    )
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    # The later-asked answer finishes first and waits for the earlier one
    # (ask order); the earlier lands 0.5 s later — inside the merge window.
    recorded = H.patch_relay(monkeypatch, think=_two_asks(0.5, 0.0))
    box = {"t": 5000}

    def on_send(provider, event):
        _two_asks_on_send(provider, event)
        if event["type"] == "session.commentary.append" and event["content"] in (_TUITION, _VANCE):
            content = event["content"]

            async def tell():
                if second_delay and content == _VANCE:
                    # …or two consecutive tellings: the second member's words
                    # arrive after the first's epoch RETIRED (gap 0.3 s) — the
                    # group is still owed, so nothing else may take the turn
                    # meanwhile, and the first member's epoch is not "output
                    # that did not tell the second" (0.6 s: past the 0.5 s
                    # budget — one nudge, then its telling still proves it).
                    await asyncio.sleep(second_delay)
                provider.push(H.out_text(content + " ", box["t"], box["t"] + 400))
                provider.push(H.out_audio("PAIR"))
                box["t"] += 400
            asyncio.get_running_loop().create_task(tell())

    async def model_talks():
        if speaking:
            # The model's own sentence spans the moment both results land.
            await asyncio.sleep(0.4)
            for index, word in enumerate("Let me think about that for a second or two".split()):
                provider.push(H.out_text(word + " ", 4000 + index * 100, 4100 + index * 100))
                provider.push(H.out_audio("OWN"))
                await asyncio.sleep(0.1)

    client = H.FakeClient([_config_141(), model_talks, 2.5, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=10, db_session_id=f"db-w-merge-{speaking}-{second_delay}",
    )

    messages = [r.getMessage() for r in caplog.records]
    assert [m for m in messages if "result merged ids=d1,d2" in m], messages
    sent = [
        (e["type"], e["content"]) for e in provider.sent
        if e["type"] in ("session.commentary.append", "session.instructions.append")
    ]
    commentaries = [content for kind, content in sent if kind.endswith("commentary.append")]
    assert commentaries[:2] == [_TUITION, _VANCE], sent
    cues = [content for kind, content in sent if "Two results follow" in content]
    if speaking:
        assert cues == [], cues
        assert [m for m in messages if "cue=paircue reason=model_speaking" in m], messages
    else:
        assert len(cues) == 1 and sent.index(("session.instructions.append", cues[0])) < sent.index(
            ("session.commentary.append", _TUITION),
        ), sent
    spoken = [m for m in messages if "result spoken id=d" in m]
    assert {m.split("id=")[1].split()[0] for m in spoken} == {"d1", "d2"}, spoken
    rows = {
        (s.get("assistant_voice") or {}).get("delegation_id"): s
        for s in _delegated_rows(recorded["saves"]) if "spoken" in (s.get("assistant_voice") or {})
    }
    assert rows["d1"]["assistant_voice"]["spoken"] is True
    assert rows["d2"]["assistant_voice"]["spoken"] is True
    folded = [
        s for s in _delegated_rows(recorded["saves"]) if "epoch" in (s.get("assistant_voice") or {})
    ]
    parents = {
        (s["assistant_voice"]["delegation_id"], s["assistant_voice"]["parent_user_turn_id"])
        for s in folded
    }
    assert ("d1", f"live-utt:{H.PSID}:1") in parents, parents
    assert ("d2", f"live-utt:{H.PSID}:2") in parents, parents
    assert not [t for t in _live_output_texts(recorded["saves"]) if "Vance" in t or "Tuition" in t]
    if second_delay:
        assert len({m.split("epoch=")[1].split()[0] for m in spoken}) == 2, spoken
    assert not [m for m in messages if "unmatched_output" in m], messages
    # Phase C (tapes): the telling's epoch is parented to ONE member's turn,
    # so each member the words proved is told to the phone as heard — or its
    # card's delivery hold reads "writing the answer" for 45 s after it.
    heard = [f.get("delegation_id") for f in client.of("delegation")
             if f.get("phase") == "completed" and f.get("spoken") is True]
    assert sorted(heard) == ["d1", "d2"], heard
    # Phase D (delivery-ladder-10): the companion's own append is not an
    # interposed one — no "attribution spoiled" line for a pair told right.
    assert not [m for m in messages if "result attribution spoiled" in m], messages


@pytest.mark.asyncio
@pytest.mark.parametrize("academic_told", [True, False], ids=["told_after", "never_told"])
async def test_one_members_telling_never_proves_its_merged_fellow(
    monkeypatch, caplog, academic_told,
):
    """Phase D words-proof-1 (P1): the founder's two IELTS answers finish
    within the 5 s merge window (2.7 s apart on 10-07) and go out as ONE
    append.  With the fellow excluded from `others` (critic C12), the General
    telling alone proved Academic on the words the two answers share: the
    Academic answer was resolved heard at General's epoch, its own telling
    then became an extra `live-output:` row (the chat showed it twice), and
    when it was never told the phone still got `completed{spoken:true}` for
    it — no «نتیجه در گفتگو», no nudge."""

    import logging

    c = _CORPUS["d1007"]
    gen, acad = c["GEN"], c["ACAD"]
    gen_told = c["TELLINGS"][3]
    acad_told = (
        "ببین، برای آیلتس آکادمیک، معمولاً با ۳۵ یا ۳۶ جواب درست از چهل تا، نمره هشت می‌گیری. "
        "یعنی حدود ۴ یا ۵ تا می‌تونی غلط داشته باشی."
    )
    H.fast_clocks(
        monkeypatch, voice_live_result_order_wait_s=3.0, voice_live_result_merge_window_s=5.0,
        voice_live_output_epoch_gap_ms=300,
    )
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        if "جنرال" in str(kwargs.get("display_request") or task):
            await asyncio.sleep(0.5)
            return gen, "test-model"
        return acad, "test-model"

    recorded = H.patch_relay(monkeypatch, think=think)
    box = {"t": 5000}

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta(c["REQ_GEN"], 0, 400))
            provider.push(H.delegation("d1", 420))
            provider.push(H.user_delta(c["REQ_ACAD"], 3000, 3400))
            provider.push(H.delegation("d2", 3450))
        elif event["type"] == "session.close":
            provider.push(H.closed())
        elif event["type"] == "session.commentary.append" and event["content"] in (gen, acad):
            content = event["content"]
            if content == acad and not academic_told:
                return

            async def tell():
                if content == acad:
                    await asyncio.sleep(0.4)
                told = gen_told if content == gen else acad_told
                provider.push(H.out_text(told + " ", box["t"], box["t"] + 400))
                provider.push(H.out_audio("PAIR"))
                box["t"] += 400
            asyncio.get_running_loop().create_task(tell())

    client = H.FakeClient([_config_141(), 3.0, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=12, db_session_id=f"db-wp1-{academic_told}")
    messages = [r.getMessage() for r in caplog.records]
    assert [m for m in messages if "result merged ids=d1,d2" in m], "precondition: one merged append"
    proven = {m.split("result spoken id=")[1].split()[0]: m.split("epoch=")[1].split()[0]
              for m in messages if "result spoken id=d" in m}
    assert proven.get("d1") == "1", proven
    heard = [f.get("delegation_id") for f in client.of("delegation")
             if f.get("phase") == "completed" and f.get("spoken") is True]
    if academic_told:
        assert proven.get("d2") == "2", proven
        assert not [o for o in _live_output_texts(recorded["saves"]) if "۳۵ یا ۳۶" in o]
    else:
        assert "d2" not in proven, proven
        assert "d2" not in heard, heard


# Phase D2 ruling 43 (words-proof-2, P1): the `distinct` gate is a TIE-BREAK.
# The 10-07 Academic answer RESTATED General's numbers, so a faithful concise
# telling of General («…۳۷ یا ۳۸ … ۲ تا ۳ تا غلط…», shared 15, precision 0.94)
# had `distinct` 0 against it and ended `unmatched_output` — «نتیجه در گفتگو»
# for an answer the caller had just heard.
_CONCISE_GEN = (
    "و در مورد جنرال، از چهل سؤال باید ۳۷ یا ۳۸ تا رو درست بزنی؛ یعنی می‌تونی ۲ تا ۳ تا غلط داشته باشی."
)
#: A repeat of Academic's own telling, said tightly: every window that holds
#: General's numbers also holds Academic's own words.
_COMPACT_ACAD = (
    "برای آکادمیک ۳۵ یا ۳۶ تا، یعنی ۴ یا ۵ غلط؛ برای جنرال از چهل سؤال باید ۳۷ یا ۳۸ تا رو "
    "درست بزنی یعنی ۲ یا ۳ غلط داشته باشی."
)
#: General's numbers inside a sentence NEARER to Academic (two of its own words,
#: «آکادمیک» and «تعداد» — under the (ii) bar of 3, so only (i) refuses it).
_NEARER_ACAD = (
    "برای جنرال یا آکادمیک از چهل سؤال باید ۳۷ یا ۳۸ تا رو درست بزنی؛ یعنی می‌تونی ۲ تا ۳ تا "
    "غلط داشته باشی، تعداد."
)
#: Phase D2 F1: Academic's own telling, said FIRST in the epoch that then
#: tells General — every General window's lookback used to reach its words.
_ACAD_TOLD = (
    "ببین، برای آیلتس آکادمیک، معمولاً با ۳۵ یا ۳۶ جواب درست از چهل تا، نمره هشت می‌گیری. "
    "یعنی حدود ۴ یا ۵ تا می‌تونی غلط داشته باشی."
)
#: …and an aside after it that tells nothing of General: with Academic's
#: closing words («…حدود ۴ یا ۵ تا می‌تونی غلط داشته باشی») it reaches
#: General's `need` at precision 0.5.
_ASIDE = "سؤال دیگه‌ای داری؟ می‌تونم کمکت کنم."


def _tiebreak_pair(order: str, *, academic_told: bool):
    """General (asked first, d1) and Academic (d2) as one merged append
    group, `order` = which one heads it ("ga" | "ag")."""

    c = _CORPUS["d1007"]
    session = _bare_session()
    gen, _ = _session_delivery(session, "d1", c["GEN"], transcript=c["REQ_GEN"])
    acad, _ = _session_delivery(session, "d2", c["ACAD"], transcript=c["REQ_ACAD"])
    gen.ask_ordinal, acad.ask_ordinal = 1, 2
    if academic_told:
        # Told 20 s ago, in an epoch of its own: no longer a candidate, still
        # a live other (≤60 s) whose own words block.
        acad.resolved = acad.spoken = True
        acad.resolved_at, acad.words_epoch = time.monotonic() - 20, 0
        session.active_delivery = gen
    else:
        head, member = (gen, acad) if order == "ga" else (acad, gen)
        head.companions, member.group_head = [member], head
        session.active_delivery = head
    return session, gen, acad


@pytest.mark.asyncio
@pytest.mark.parametrize(("case", "text", "order", "academic_told", "want"), [
    # (1) the concise General telling, Academic owed: General by tie-break,
    # Academic stays owed — whichever member the group iterates first.
    ("concise", _CONCISE_GEN, "ga", False, {"d1": "words_tiebreak"}),
    ("concise_reversed", _CONCISE_GEN, "ag", False, {"d1": "words_tiebreak"}),
    # …and Academic's own later telling (its exclusive words) proves it.
    ("concise_then_academic", _CONCISE_GEN + " " + _CORPUS["d1007"]["TELLINGS"][0]
     + " می‌گیری. یعنی حدود ۴ یا ۵ تا می‌تونی غلط داشته باشی.", "ga", False,
     {"d1": "words_tiebreak", "d2": "words"}),
    # (2) wp-1: General's FULL first sentence — a symmetric tie, General asked
    # first; never Academic (reversed: the group iterates Academic first).
    ("first_sentence", "@T3_FIRST", "ga", False, {"d1": "words_tiebreak"}),
    ("first_sentence_reversed", "@T3_FIRST", "ag", False, {"d1": "words_tiebreak"}),
    ("full_general", "@T3", "ga", False, {"d1": "words"}),
    # (3) both recorded merged tellings still prove BOTH members (normal W).
    ("merged_1", "@T3+T1", "ga", False, {"d1": "words", "d2": "words"}),
    ("merged_0", "@T3+T0", "ga", False, {"d1": "words", "d2": "words"}),
    # (4a) (ii): a REPEAT of Academic's own telling never proves General —
    # the recorded tellings 2 and 3 (General's numbers right after
    # Academic's) and a compact one.
    ("academic_repeat_1", "@T1", "ga", True, {}),
    ("academic_repeat_2", "@T2", "ga", True, {}),
    ("academic_repeat_compact", _COMPACT_ACAD, "ga", True, {}),
    # (4b) (i): a window nearer to Academic never proves General.
    ("nearer_to_academic", _NEARER_ACAD, "ga", False, {"d2": "words_tiebreak"}),
    # (5) not merged: General told 20 s after Academic was heard.
    ("general_after_academic_told", _CONCISE_GEN, "ga", True, {"d1": "words_tiebreak"}),
    # (6) Phase D2 F1: Academic told FIRST, then the concise General telling,
    # in ONE epoch.  (ii)'s lookback stops where Academic's proof ended, so
    # Academic's own words no longer refuse every General window…
    ("academic_first_then_concise", _ACAD_TOLD + " " + _CONCISE_GEN, "ga", False,
     {"d1": "words_tiebreak", "d2": "words"}),
    ("academic_first_then_concise_reversed", _ACAD_TOLD + " " + _CONCISE_GEN, "ag", False,
     {"d1": "words_tiebreak", "d2": "words"}),
    # …and Academic's own telling still never proves General: a window
    # nearer to an answer THIS epoch proved is that answer's telling, so (i)
    # still reads it.
    ("academic_first_alone", _ACAD_TOLD, "ga", False, {"d2": "words"}),
    ("academic_first_then_aside", _ACAD_TOLD + " " + _ASIDE, "ga", False, {"d2": "words"}),
    ("academic_first_then_aside_reversed", _ACAD_TOLD + " " + _ASIDE, "ag", False,
     {"d2": "words"}),
    # (7) Phase D3 ruling 56 (accepts fixup §3): the RECORDED Academic tellings
    # 1 and 2 also voice General's numbers («برای جنرال معمولاً باید ۳۷ یا ۳۸ تا
    # درست باشه…»).  Told while BOTH answers are owed, in one merged epoch,
    # they credit General by tie-break: the caller heard General's facts, and
    # the un-voiced IDP caveat is a smaller loss than a repeat telling
    # (problem 1).  The same tellings said as a REPEAT after Academic was
    # heard in ANOTHER epoch still never prove General (`academic_repeat_*`).
    ("merged_academic_telling_1", "@T1", "ga", False, {"d1": "words_tiebreak", "d2": "words"}),
    ("merged_academic_telling_1_reversed", "@T1", "ag", False,
     {"d1": "words_tiebreak", "d2": "words"}),
    ("merged_academic_telling_2", "@T2", "ga", False, {"d1": "words_tiebreak", "d2": "words"}),
    ("merged_academic_telling_2_reversed", "@T2", "ag", False,
     {"d1": "words_tiebreak", "d2": "words"}),
])
async def test_the_distinct_gate_is_a_tiebreak_on_the_recorded_corpus(
    caplog, case, text, order, academic_told, want,
):
    """Phase D2 ruling 43 (amends D-R1(3)): a window that says enough of an
    answer but holds fewer than `min(3, need)` of its OWN words still proves
    it when, against every other live answer sharing those words, it is (i)
    not nearer to that answer and (ii) not a telling of that answer's own
    content; a symmetric tie goes to the answer asked first, and the words
    one proof spent are never credited to a second."""

    import logging

    from app.services.live_voice_protocol import ClosedEpoch

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    tellings = _CORPUS["d1007"]["TELLINGS"]
    text = {
        "@T3_FIRST": tellings[3].split(".")[0], "@T3": tellings[3],
        "@T3+T1": tellings[3] + " " + tellings[1], "@T3+T0": tellings[3] + " " + tellings[0],
        "@T1": tellings[1], "@T2": tellings[2],
    }.get(text, text)
    session, gen, acad = _tiebreak_pair(order, academic_told=academic_told)
    session.adapter.provider_event(H.out_text(text, 900, 9000))
    await session.persist_spoken(ClosedEpoch(
        epoch=1, output_id=f"live:{H.PSID}:1", text=text, interrupted=False,
        played_ms=8000, born_monotonic=time.monotonic(),
    ))
    got = {
        d.task.delegation_id: d.words_reason for d in (gen, acad)
        if d.words_epoch == 1 and d.spoken
    }
    assert got == want, (case, got)
    # An answer the epoch proved is never ALSO reported unmatched by it (F1:
    # the verdict of the pass that bound it is the one that stands).
    unmatched = [
        r.getMessage().split("id=")[1].split()[0] for r in caplog.records
        if "result epoch unmatched id=" in r.getMessage()
    ]
    assert not set(unmatched) & set(want), (case, unmatched)


@pytest.mark.asyncio
@pytest.mark.parametrize("academic_told", [True, False], ids=["told_after", "never_told"])
async def test_a_concise_telling_of_a_restated_answer_is_heard_by_tiebreak(
    monkeypatch, caplog, academic_told,
):
    """Phase D2 ruling 43, the founder's own scenario end to end: General and
    the RESTATING 10-07 Academic answer go out as one merged append; the model
    tells General concisely (shared 15, precision 0.94, `distinct` 0).  It is
    heard by tie-break — not `unmatched_output`, not «نتیجه در گفتگو» — and
    Academic stays owed: its own later telling proves it the normal way."""

    import logging

    c = _CORPUS["d1007"]
    gen, acad = c["GEN"], c["ACAD"]
    acad_told = (
        "ببین، برای آیلتس آکادمیک، معمولاً با ۳۵ یا ۳۶ جواب درست از چهل تا، نمره هشت می‌گیری. "
        "یعنی حدود ۴ یا ۵ تا می‌تونی غلط داشته باشی."
    )
    H.fast_clocks(
        monkeypatch, voice_live_result_order_wait_s=3.0, voice_live_result_merge_window_s=5.0,
        voice_live_output_epoch_gap_ms=300,
    )
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        if "جنرال" in str(kwargs.get("display_request") or task):
            await asyncio.sleep(0.5)
            return gen, "test-model"
        return acad, "test-model"

    recorded = H.patch_relay(monkeypatch, think=think)
    box = {"t": 5000}

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta(c["REQ_GEN"], 0, 400))
            provider.push(H.delegation("d1", 420))
            provider.push(H.user_delta(c["REQ_ACAD"], 3000, 3400))
            provider.push(H.delegation("d2", 3450))
        elif event["type"] == "session.close":
            provider.push(H.closed())
        elif event["type"] == "session.commentary.append" and event["content"] in (gen, acad):
            content = event["content"]
            if content == acad and not academic_told:
                return

            async def tell():
                if content == acad:
                    await asyncio.sleep(0.4)
                told = _CONCISE_GEN if content == gen else acad_told
                provider.push(H.out_text(told + " ", box["t"], box["t"] + 400))
                provider.push(H.out_audio("PAIR"))
                box["t"] += 400
            asyncio.get_running_loop().create_task(tell())

    client = H.FakeClient([_config_141(), 3.0, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=12, db_session_id=f"db-r43-{academic_told}")
    messages = [r.getMessage() for r in caplog.records]
    assert [m for m in messages if "result merged ids=d1,d2" in m], "precondition: one merged append"
    spoken = {m.split("result spoken id=")[1].split()[0]: m for m in messages if "result spoken id=d" in m}
    assert "d1" in spoken and " epoch=1 " in spoken["d1"], spoken
    assert "reason=words_tiebreak" in spoken["d1"] and spoken["d1"].endswith(" vs=d2"), spoken["d1"]
    assert not [m for m in messages if "result NOT spoken id=d1" in m], messages
    heard = [f.get("delegation_id") for f in client.of("delegation")
             if f.get("phase") == "completed" and f.get("spoken") is True]
    assert "d1" in heard, heard
    assert not [f for f in client.of("delegation") if f.get("delegation_id") == "d1"
                and f.get("phase") == "completed" and f.get("spoken") is False]
    # One row per answer: the concise telling folded onto General's row.
    assert not [o for o in _live_output_texts(recorded["saves"]) if "۳۷ یا ۳۸" in o]
    if academic_told:
        assert " epoch=2 " in spoken.get("d2", "") and "reason=words " in spoken["d2"], spoken
    else:
        assert "d2" not in spoken, spoken
        assert "d2" not in heard, heard


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ["general_first", "academic_first"])
async def test_a_merged_pair_told_in_one_epoch_is_heard_in_either_order(
    monkeypatch, caplog, order,
):
    """Phase D2 F1 (ruling 43, the founder's own pair end to end): General
    and the RESTATING Academic answer go out as ONE merged append and the
    model tells both in ONE epoch — Academic first, or General first.  The
    pair cue is not sent while the model speaks, and the model may ignore
    it anyway.  Academic first used to leave General `result epoch
    unmatched` (its lookback read Academic's words), then the one nudge — a
    second telling — or «نتیجه در گفتگو».  Both are heard, once."""

    import logging

    c = _CORPUS["d1007"]
    gen, acad = c["GEN"], c["ACAD"]
    H.fast_clocks(
        monkeypatch, voice_live_result_order_wait_s=3.0, voice_live_result_merge_window_s=5.0,
        voice_live_output_epoch_gap_ms=300,
    )
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        if "جنرال" in str(kwargs.get("display_request") or task):
            await asyncio.sleep(0.5)
            return gen, "test-model"
        return acad, "test-model"

    H.patch_relay(monkeypatch, think=think)
    box = {"t": 5000, "told": False}
    telling = (
        _CONCISE_GEN + " " + _ACAD_TOLD if order == "general_first"
        else _ACAD_TOLD + " " + _CONCISE_GEN
    )

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta(c["REQ_GEN"], 0, 400))
            provider.push(H.delegation("d1", 420))
            provider.push(H.user_delta(c["REQ_ACAD"], 3000, 3400))
            provider.push(H.delegation("d2", 3450))
        elif event["type"] == "session.close":
            provider.push(H.closed())
        elif event["type"] == "session.commentary.append" and event["content"] in (gen, acad):
            if box["told"]:
                return
            box["told"] = True

            async def tell():
                await asyncio.sleep(0.2)   # both appends are in
                words = telling.split(" ")
                step = max(1, len(words) // 4)
                for i in range(0, len(words), step):
                    provider.push(H.out_text(
                        " ".join(words[i:i + step]) + " ", box["t"], box["t"] + 300,
                    ))
                    provider.push(H.out_audio("PAIR"))
                    box["t"] += 300
                    await asyncio.sleep(0.05)
            asyncio.get_running_loop().create_task(tell())

    client = H.FakeClient([_config_141(), 3.5, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=14, db_session_id=f"db-f1-{order}")
    messages = [r.getMessage() for r in caplog.records]
    assert [m for m in messages if "result merged ids=d1,d2" in m], "precondition: one merged append"
    spoken = {m.split("result spoken id=")[1].split()[0]: m for m in messages if "result spoken id=d" in m}
    assert " epoch=1 " in spoken.get("d1", "") and " epoch=1 " in spoken.get("d2", ""), spoken
    assert "reason=words_tiebreak" in spoken["d1"] and spoken["d1"].endswith(" vs=d2"), spoken["d1"]
    assert "reason=words " in spoken["d2"], spoken["d2"]
    assert not [m for m in messages if "result nudge sent" in m or "result NOT spoken" in m
                or "result epoch unmatched" in m], messages
    completed = [(f.get("delegation_id"), f.get("spoken")) for f in client.of("delegation")
                 if f.get("phase") == "completed"]
    assert ("d1", True) in completed and ("d2", True) in completed, completed
    assert not [f for f in completed if f[1] is False], completed


_MUSEUM, _ADMISSIONS = "The museum opens at ten in the morning.", "Doctor Vance runs admissions."


@pytest.mark.asyncio
@pytest.mark.parametrize(("mode", "speak_s"), [("staccato", 0.5), ("lively", 2.0)])
async def test_an_untold_result_is_bounded_however_lively_the_call(
    monkeypatch, caplog, mode, speak_s,
):
    """Phase D delivery-ladder-7: `voice_live_result_verdict_max_s` bounded
    only the open-epoch wait.  The quiet branch re-armed to the LAST output
    plus a speak budget and the mid-utterance branch to "now", so a model
    replying in short sentences under one budget apart ("staccato"), or a
    caller and model taking turns ("lively"), kept an untold result owed for
    as long as the conversation flowed — its group busy, the next result
    queued behind it, relay lines held, progress suppressed.  The bound now
    covers every wait: the verdict comes at the bound (once the caller is not
    mid-sentence), and the next result goes out."""

    import logging

    bound = 3.0
    H.fast_clocks(
        monkeypatch, voice_live_speak_timeout_s=speak_s, voice_live_result_verdict_max_s=bound,
    )
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    answers = {"when does the museum open": _MUSEUM, "who runs admissions": _ADMISSIONS}
    box: dict = {"sends": {}, "verdicts": []}

    # Measured on the relay's OWN clock (`time.monotonic`, what the bound is
    # written in), never the wall clock: a host that sleeps mid-run (this Mac
    # in maintenance sleep, 2026-10-08: send at 02:54:56.557, Sleep at
    # 02:54:56, DarkWake at 02:56:09, verdict at 02:56:10.9) adds its sleep to
    # `time.time()` and to `LogRecord.created` but not to the monotonic clock.
    class _VerdictClock(logging.Handler):
        def emit(self, record):
            if "result NOT spoken id=d1" in record.getMessage():
                box["verdicts"].append(time.monotonic())

    clock = _VerdictClock(level=logging.INFO)
    relay_logger = logging.getLogger("app.services.live_voice_protocol")
    relay_logger.addHandler(clock)

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        display = kwargs.get("display_request") or task
        await asyncio.sleep(1.0 if answers.get(display) == _ADMISSIONS else 0.05)
        return answers[display], "test-model"

    H.patch_relay(monkeypatch, think=think)

    async def chatter(provider):
        loop = asyncio.get_running_loop()
        end, t, u = loop.time() + 7.0, 5000, 9000
        while loop.time() < end:
            if mode == "staccato":
                await asyncio.sleep(0.3)
                provider.push(H.out_text(" Okay.", t, t + 150))
                provider.push(H.out_audio("TALK"))
                t += 300
                continue
            for word in ("so", "what", "about", "this"):
                provider.push(H.user_delta(" " + word, u, u + 100))
                u += 100
                await asyncio.sleep(0.1)
            await asyncio.sleep(0.2)
            for word in (" Sure", " thing", " friend."):
                provider.push(H.out_text(word, t, t + 100))
                provider.push(H.out_audio("TALK"))
                t += 100
                await asyncio.sleep(0.1)
            u += 800
            await asyncio.sleep(0.3)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("when does the museum open", 0, 400))
            provider.push(H.delegation("d1", 420))
            provider.push(H.user_delta("who runs admissions", 3000, 3400))
            provider.push(H.delegation("d2", 3450))
        elif event["type"] == "session.commentary.append":
            content = event.get("content") or ""
            for did, answer in (("d1", _MUSEUM), ("d2", _ADMISSIONS)):
                if answer in content and did not in box["sends"]:
                    box["sends"][did] = time.monotonic()
                    if did == "d1":
                        asyncio.get_running_loop().create_task(chatter(provider))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([_config_141(), 9.0, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    try:
        await H.run_relay(client, provider, timeout=16, db_session_id=f"db-bound-{mode}")
    finally:
        relay_logger.removeHandler(clock)
    assert "d1" in box["sends"] and "d2" in box["sends"], box
    verdict = box["verdicts"]
    assert verdict, "d1 never got its verdict"
    # At the bound — plus at most one caller burst — never at the end of the
    # 7 s of talk; and d2 (ready ~1 s after d1's send) went out before it ended.
    assert verdict[0] - box["sends"]["d1"] <= bound + 1.0, (verdict[0] - box["sends"]["d1"])
    assert box["sends"]["d2"] - box["sends"]["d1"] < 6.5, box


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["quiet", "caller_speaking"])
async def test_the_verdict_bound_covers_the_quiet_and_caller_waits(monkeypatch, state):
    """Phase D delivery-ladder-7, each re-arm on its own: past
    `voice_live_result_verdict_max_s` from the send, the QUIET branch no
    longer re-arms to the last output + a speak budget (it gives the verdict
    now), and the caller mid-sentence no longer pushes the deadline forward
    (the verdict comes the moment they stop)."""

    from app.config import settings as app_settings

    monkeypatch.setattr(app_settings, "voice_live_result_verdict_max_s", 3.0)
    monkeypatch.setattr(app_settings, "voice_live_speak_timeout_s", 6.0)
    session = _bare_session()
    delivery, _ref = _session_delivery(session, "d1", _MUSEUM)
    now = time.monotonic()
    delivery.sent_monotonic = delivery.appended_at = delivery.acked_at = now - 4.0
    delivery.deadline = now - 0.1
    session.adapter._last_delta_monotonic = now - 0.1   # the model spoke 0.1 s ago
    verdicts: list = []

    async def resolve(d, *, spoken, reason):
        verdicts.append((d.task.delegation_id, spoken, reason))

    async def nudge(d, *, reason):
        verdicts.append((d.task.delegation_id, "nudge", reason))

    session.resolve_delivery = resolve
    session.nudge_delivery = nudge
    session.hold_for_input = lambda d: False
    session.born_since = lambda d, moment: True
    session.user_is_mid_utterance = lambda: state == "caller_speaking"
    await session.check_delivery_timeout(delivery)
    if state == "quiet":
        assert verdicts == [("d1", False, "unmatched_output")], verdicts
    else:
        assert verdicts == [] and delivery.deadline < now, (verdicts, delivery.deadline - now)


_LIBRARY_TWO = (
    "The library on Bloor Street closes at eleven tonight. "
    "The cafeteria on the second floor stops serving at nine."
)
_LIBRARY_ONE = "The library closes at eleven tonight."


async def _refused(monkeypatch, caplog, *, answer, ceiling, reject_kinds, reject_first,
                   model_talking=False, db):
    """One result whose commentary append(s) the provider REFUSES; the fake
    model says whatever it is then asked (the nudge's text)."""

    import logging

    H.fast_clocks(
        monkeypatch, voice_live_append_token_ceiling=ceiling, voice_live_result_defer_max_ms=60,
    )
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    box: dict = {}

    async def think(*_a, **_k):
        if model_talking:
            # (before the fake's echo of the nudge, which it places at 900 ms)
            box["p"].push(H.out_text("Okay so while that runs let me tell you", 450, 650))
            box["p"].push(H.out_audio())
            await asyncio.sleep(0.12)
        return answer, "test-model"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("when does the library close", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif model_talking and event["type"] == "session.commentary.append":
            async def more():
                t = 650
                for w in (" about", " the", " weather", " up", " north."):
                    await asyncio.sleep(0.04)
                    provider.push(H.out_text(w, t, t + 40))
                    t += 40
                    provider.push(H.out_audio())
            asyncio.get_running_loop().create_task(more())
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([_config_141(), 2.5, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=on_send, auto_ack=True, reject_appends=set(reject_kinds),
        reject_first=reject_first, speak_on_commentary=True, speak_on_instructions=True,
    )
    await H.run_relay(client, provider, timeout=10, db_session_id=db)
    messages = [r.getMessage() for r in caplog.records]
    flags = [(r.get("assistant_voice") or {}).get("spoken") for r in _delegated_rows(recorded["saves"])]
    return messages, flags, provider


@pytest.mark.asyncio
async def test_a_later_chunk_refusal_is_not_the_nudges(monkeypatch, caplog):
    """Phase D delivery-ladder-8: a long answer goes out as several commentary
    appends and the nudge goes out on the FIRST refusal — so the refusal of
    the SAME attempt's second chunk was counted as "the nudge was refused
    too": `rejected_twice`, «نتیجه در گفتگو», while the accepted nudge was
    being told, and that telling was never credited (no late window)."""

    messages, flags, provider = await _refused(
        monkeypatch, caplog, answer=_LIBRARY_TWO, ceiling=20,
        reject_kinds={"session.commentary.append"}, reject_first=2, db="db-reject-chunks",
    )
    assert len(provider.refused) == 2 and not any("nudge" in r for r in provider.refused)
    assert not [m for m in messages if "reason=rejected_twice" in m], messages
    assert any("result spoken id=d1" in m for m in messages), messages
    assert flags and flags[-1] is True, flags


@pytest.mark.asyncio
async def test_a_refused_results_nudge_waits_for_the_models_sentence(monkeypatch, caplog):
    """Phase D delivery-ladder-8: an append refused while the model is
    mid-sentence nudged at once — an instruction "say it now" into its open
    sentence, which it drops and restarts (10-07 problem 1).  The nudge now
    waits for that epoch to end, then goes out (the refused answer was never
    the model's, so nothing it said can be its telling)."""

    messages, flags, provider = await _refused(
        monkeypatch, caplog, answer=_LIBRARY_ONE, ceiling=450,
        reject_kinds={"session.commentary.append"}, reject_first=1, model_talking=True,
        db="db-reject-mid-sentence",
    )
    deferred = [i for i, m in enumerate(messages) if "result nudge deferred id=d1" in m]
    sent = [i for i, m in enumerate(messages) if "result nudge sent id=d1 reason=rejected" in m]
    closed = [i for i, m in enumerate(messages) if "output epoch closed epoch=1" in m]
    assert deferred and sent and closed, messages
    assert deferred[0] < closed[0] < sent[0], (deferred, closed, sent)
    assert flags and flags[-1] is True, flags


class _Respin(Exception):
    """A held relay line re-held this many times in ONE flush: the loop that
    froze the replica (delivery-ladder-1).  Raised so a regression fails
    instead of hanging the runner."""


def _count_reholds(monkeypatch, live, box, limit=50):
    original_hold = live._LiveSession.hold_behind_delivery
    original_flush = live._LiveSession.flush_held_speech

    def counted(self, line, reason):
        if box.get("in_flush"):
            box["reholds"] = box.get("reholds", 0) + 1
            if box["reholds"] > limit:
                raise _Respin(f"re-held {box['reholds']} times in one flush")
        return original_hold(self, line, reason)

    async def flush(self):
        box["in_flush"] = True
        try:
            await original_flush(self)
        finally:
            box["in_flush"] = False

    monkeypatch.setattr(live._LiveSession, "hold_behind_delivery", counted)
    monkeypatch.setattr(live._LiveSession, "flush_held_speech", flush)


@pytest.mark.asyncio
async def test_a_held_line_never_spins_the_flush_while_a_merged_companion_is_owed(monkeypatch):
    """delivery-ladder-1 (P0): `hold_behind_delivery` holds on the whole
    append GROUP (`delivery_group_busy`) while `flush_held_speech` let a line
    go once the HEAD resolved.  With a merged pair's head proven and retired
    and its companion still owed, a held `speak()` line was popped, re-held
    by its own lambda and popped again forever — no suspension point, every
    call on the replica frozen.  The flush now waits on the same predicate,
    and one pass never re-runs a line that re-held itself."""

    import app.services.live_voice_protocol as live
    from app.services.live_voice_protocol import DelegatedTask, ResultDelivery

    box: dict = {}
    _count_reholds(monkeypatch, live, box)
    session = _bare_session()
    session.held_speech = []
    session.held_model_since = 0.0
    monkeypatch.setattr(session, "may_speak", lambda: True)

    def _delivery(did):
        return ResultDelivery(
            task=DelegatedTask(delegation_id=did, offset_ms=0, transcript=""),
            answer="x", claim={},
        )

    head, companion = _delivery("d1"), _delivery("d2")
    head.companions, companion.group_head = [companion], head
    session.active_delivery = head
    assert await session.speak("d3", "Still on it.") == []
    assert len(session.held_speech) == 1
    head.resolved = True  # the head proven, its epoch retired; the companion owed
    await asyncio.wait_for(session.flush_held_speech(), timeout=5)
    assert len(session.held_speech) == 1 and box.get("reholds", 0) == 0, box


@pytest.mark.asyncio
async def test_a_line_held_behind_a_merged_pair_goes_out_after_the_companion(monkeypatch, caplog):
    """delivery-ladder-1 (P0), end to end: the recorded merged-pair shape (the
    companion told 0.6 s after the head's epoch retired) plus one bare relay
    line requested while the pair goes out — the line waits for the whole
    group and is then said once, after the companion."""

    import logging
    import app.services.live_voice_protocol as live

    box: dict = {}
    _count_reholds(monkeypatch, live, box)
    H.fast_clocks(
        monkeypatch, voice_live_result_order_wait_s=3.0, voice_live_result_merge_window_s=5.0,
        voice_live_output_epoch_gap_ms=300,
    )
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    H.patch_relay(monkeypatch, think=_two_asks(0.5, 0.0))
    original_cue = live._LiveSession.cue_merged_pair

    async def cue_then_line(self, head, companion):
        await original_cue(self, head, companion)
        box["held_ret"] = await self.speak("d1", "One more moment.", parent_user_turn_id="")

    monkeypatch.setattr(live._LiveSession, "cue_merged_pair", cue_then_line)
    t = {"t": 5000}

    def on_send(provider, event):
        _two_asks_on_send(provider, event)
        if event["type"] == "session.commentary.append" and event["content"] in (_TUITION, _VANCE):
            content = event["content"]

            async def tell():
                if content == _VANCE:
                    await asyncio.sleep(0.6)
                provider.push(H.out_text(content + " ", t["t"], t["t"] + 400))
                provider.push(H.out_audio("PAIR"))
                t["t"] += 400
            asyncio.get_running_loop().create_task(tell())

    client = H.FakeClient([_config_141(), 2.5, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-flush-merged-held")
    sent = [e.get("content") for e in provider.sent if e["type"] == "session.commentary.append"]
    assert box.get("held_ret") == [], "precondition: the line was held behind the pair"
    assert box.get("reholds", 0) <= 1, box
    assert sent.count("One more moment.") == 1, sent
    assert sent.index("One more moment.") > sent.index(_VANCE), sent


@pytest.mark.asyncio
async def test_no_nudge_and_no_verdict_while_an_epoch_is_open(monkeypatch, caplog):
    """D-R1 (2): the model mid-sentence — an epoch held open by the hard gap
    after an unfinished phrase, no deltas arriving — is not "quiet".  The
    10-07 nudges all went out inside the one epoch telling the answer.  Here
    the speak budget passes twice over while the model's own sentence is
    open; no nudge, `result verdict waits`, and its continuation proves it."""

    import logging

    H.fast_clocks(monkeypatch, voice_live_result_defer_max_ms=60)
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    answer = "The museum opens at ten in the morning."
    box: dict = {}

    async def think(*_args, **_kwargs):
        box["p"].push(H.out_text("Let me check that for you and also", 1000, 1800))
        box["p"].push(H.out_audio())
        await asyncio.sleep(0.15)
        return answer, "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("when does the museum open", 0, 600))
            provider.push(H.delegation("d1", 650))
        elif event["type"] == "session.commentary.append" and answer in event["content"]:
            async def finish_late():
                await asyncio.sleep(1.3)    # > 2 × the 0.5 s budget, < the hard gap
                provider.push(H.out_text(" it opens at ten in the morning.", 1800, 2600))
                provider.push(H.out_audio())
            asyncio.get_running_loop().create_task(finish_late())
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([_config_141(), 2.4, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-w-open-epoch")

    messages = [r.getMessage() for r in caplog.records]
    assert not [c for c in _instructions(provider) if answer in c], "nudged into an open epoch"
    assert [m for m in messages if "result verdict waits id=d1" in m], messages
    assert [m for m in messages if "result spoken id=d1" in m and "reason=words" in m], messages


@pytest.mark.asyncio
async def test_a_result_landing_while_another_is_voiced_continues_after_it(monkeypatch, caplog):
    """D-R1 (3) continuation: an answer that arrives while another is being
    voiced waits for that epoch to retire, then goes out after ONE "continue,
    do not start over" cue."""

    import logging

    H.fast_clocks(
        monkeypatch, voice_live_result_continue_cue_enabled=True,
        voice_live_output_epoch_gap_ms=200,
    )
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    H.patch_relay(monkeypatch, think=_two_asks(0.0, 0.45))
    marks: dict = {}

    def on_send(provider, event):
        _two_asks_on_send(provider, event)
        kind, content = event["type"], str(event.get("content") or "")
        if kind == "session.commentary.append" and content == _TUITION:
            async def tell_slowly():
                for index, word in enumerate(_TUITION.split()):
                    provider.push(H.out_text(word + " ", 5000 + index * 100, 5100 + index * 100))
                    provider.push(H.out_audio("ONE"))
                    await asyncio.sleep(0.12)
                marks["first_told"] = time.monotonic()
            asyncio.get_running_loop().create_task(tell_slowly())
        elif kind == "session.instructions.append" and "Continue after" in content:
            marks["cue"] = time.monotonic()
        elif kind == "session.commentary.append" and content == _VANCE:
            marks["second"] = time.monotonic()
            provider.push(H.out_text(_VANCE, 9000, 9600))
            provider.push(H.out_audio("TWO"))

    client = H.FakeClient([_config_141(), 2.5, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-w-continue")

    messages = [r.getMessage() for r in caplog.records]
    assert [m for m in messages if "result continued id=d2 after=d1" in m], messages
    assert "cue" in marks and "second" in marks, marks
    assert marks["first_told"] <= marks["cue"] <= marks["second"], marks


def test_a_relay_opened_delivery_is_never_merged(monkeypatch):
    """Critic U2: a relay-opened task's result is spoken as a `relay_task_say`
    INSTRUCTION ("Tell them now: …"); two of those back to back are two
    say-it-now restarts.  It is never merged, on either side."""

    from app.config import settings as app_settings
    from app.services.live_voice_protocol import DelegatedTask, ResultDelivery

    monkeypatch.setattr(app_settings, "voice_live_result_merge_window_s", 5.0)
    monkeypatch.setattr(app_settings, "voice_live_result_order_wait_s", 0.0)
    session = _bare_session()

    def delivery(delegation_id):
        return ResultDelivery(
            task=DelegatedTask(delegation_id=delegation_id, offset_ms=0, transcript="x"),
            answer="An answer.", claim={}, created_monotonic=time.monotonic(),
        )

    for head_id, other_id, merged in (
        ("d1", "d2", True),
        ("relay-research:abc", "d2", False),
        ("d1", "relay-play:abc", False),
    ):
        head, other = delivery(head_id), delivery(other_id)
        session.delivery_queue = [other]
        companion = session.merge_companion(head)
        assert (companion is other) is merged, (head_id, other_id)
        assert (session.delivery_queue == []) is merged


def test_a_relay_line_holds_a_waiting_result_only_while_it_can_still_be_voiced(
        monkeypatch, caplog):
    """F14 (critic C7): a relay line's claim held a waiting result until it
    bound or its 6 s deadline passed — the 10-07 progress lines withdrew
    `reason=no_epoch` 6 s later while answers waited behind them.  Now: hold
    while its epoch is open (bounded by the result's defer budget) or for
    `voice_live_relay_line_hold_ms` after its send, then withdraw it
    (`reason=result_waiting`)."""

    import logging

    from app.config import settings as app_settings
    from app.services.live_voice_protocol import DelegatedTask, ResultDelivery

    monkeypatch.setattr(app_settings, "voice_live_relay_line_hold_ms", 1500)
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    session = _bare_session()
    session.withdrawn_relay_claims = {}
    session.appends = {}
    waiting = ResultDelivery(
        task=DelegatedTask(delegation_id="d9", offset_ms=0, transcript="x"),
        answer="An answer.", claim={}, defer_until=time.monotonic() + 4.0,
    )
    fresh = session.build_commentary_claim("", {}, delegation_id="d1")
    stale = session.build_commentary_claim("", {}, delegation_id="d2")
    session.note_commentary_epoch_claim(fresh)
    session.note_commentary_epoch_claim(stale)
    now = time.monotonic()
    session.relay_claims = [
        {"claim": fresh, "event_ids": ["e1"], "deadline": now + 6, "sent_at": now - 0.2},
        {"claim": stale, "event_ids": ["e2"], "deadline": now + 6, "sent_at": now - 2.0},
    ]
    assert session.relay_line_holds_result(waiting) is True, "a line sent 0.2 s ago may still bind"
    assert [r["claim"] for r in session.relay_claims] == [fresh]
    assert session.adapter._commentary_claims == [fresh], "the stale claim left the FIFO"
    withdrawn = [r.getMessage() for r in caplog.records if "reason=result_waiting" in r.getMessage()]
    assert withdrawn and "age_ms=" in withdrawn[0], withdrawn

    session.relay_claims[0]["sent_at"] = now - 1.6
    assert session.relay_line_holds_result(waiting) is False
    assert session.relay_claims == [] and session.adapter._commentary_claims == []

    # (a) the line's own epoch is being voiced: held — until the result's
    # defer budget is spent.
    session.adapter.provider_event(H.out_text("Still searching for that.", 900, 1500))
    session.relay_line_epoch = session.adapter.epoch
    assert session.relay_line_holds_result(waiting) is True
    waiting.defer_until = time.monotonic() - 0.1
    assert session.relay_line_holds_result(waiting) is False


_F14_LINE_DELAY = 2.0


@pytest.mark.asyncio
@pytest.mark.parametrize("held", [True, False], ids=["held_for_order", "goes_out_now"])
async def test_a_relay_lines_claim_is_withdrawn_only_by_the_append_it_held(
    monkeypatch, caplog, held,
):
    """Phase D2 ruling 45 (delivery-ladder-5): F14 withdrew an unbound relay
    line's claim as a SIDE EFFECT of its gate, which ran BEFORE the ask-order
    and boundary holds — so a result that could not go out anyway (held for
    an earlier-asked task) still took the claim away.  The progress line,
    acked at once and voiced 2 s later (a slow provider; the hold is 1.5 s),
    then opened an epoch with no claim: parented on the caller's unrelated
    remark in between («okay take your time») and recorded as its answer.
    Now F14 runs only in the pass that appends: the line binds its OWN
    claim, on its task's turn.  Control (`goes_out_now`): a result ready to
    go out still withdraws the stale claim (`reason=result_waiting`).  The
    results' sends do not move.  Production clocks for every knob read."""

    import logging

    from app.config import settings as app_settings

    H.fast_clocks(
        monkeypatch, voice_live_delegation_ttl_s=6.0, voice_live_speak_timeout_s=6.0,
        voice_live_result_defer_max_ms=4000, voice_live_result_order_wait_s=4.0,
        voice_live_output_epoch_gap_ms=300,
    )
    monkeypatch.setattr(app_settings, "voice_live_relay_line_hold_ms", 1500)
    monkeypatch.setattr(app_settings, "voice_live_progress_speak_after_s", 0.2)
    monkeypatch.setattr(app_settings, "voice_live_progress_min_remaining_s", 0.0)
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    marks: dict = {}
    box: dict = {}

    def line_sent() -> bool:
        return "line_sent" in marks

    async def wait(pred, timeout):
        for _ in range(int(timeout / 0.02)):
            if pred():
                return True
            await asyncio.sleep(0.02)
        return False

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        display = kw.get("display_request") or task
        if display == "how much is tuition":
            await relay.on_event({"type": "tool.start", "call_id": "c1", "name": "web_search",
                                  "args": {"query": "tuition"}})
            await wait(line_sent, 4.0)
            await asyncio.sleep(3.0 if held else 1.7)
            await relay.on_event({"type": "tool.end", "call_id": "c1", "name": "web_search",
                                  "ok": True, "preview": "found", "elapsed_ms": 20})
            marks["d1_done"] = time.monotonic()
            return _TUITION, "test-model"
        await wait(line_sent, 4.0)
        await asyncio.sleep(0.2)
        marks["d2_done"] = time.monotonic()
        return _VANCE, "test-model"

    def on_send(provider, event):
        kind = event["type"]
        if kind == "session.start":
            provider.push(H.user_delta("how much is tuition", 0, 400))
            provider.push(H.delegation("d1", 420))
            if held:
                provider.push(H.user_delta("who runs admissions", 3000, 3400))
                provider.push(H.delegation("d2", 3450))
        elif kind == "session.close":
            provider.push(H.closed())
        elif kind == "session.commentary.append":
            content = str(event.get("content") or "")
            if content in (_TUITION, _VANCE):
                marks[content] = time.monotonic()
                provider.push(H.out_text(content, 9000, 9600))
                provider.push(H.out_audio("ANS"))
                return
            marks.setdefault("line_sent", time.monotonic())
            box["line"] = content

            async def voice_late():
                # The caller says something after the hold ran out and before
                # the line is voiced.
                await asyncio.sleep(1.65)
                provider.push(H.user_delta("okay take your time", 4000, 4300))
                await asyncio.sleep(_F14_LINE_DELAY - 1.65)
                provider.push(H.out_text(content, 5000, 5600))
                provider.push(H.out_audio("LINE"))
            asyncio.get_running_loop().create_task(voice_late())

    H.patch_relay(monkeypatch, think=think)
    client = H.FakeClient([_config_141(), 7.5, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=15, db_session_id=f"db-r45-{held}")

    messages = [r.getMessage() for r in caplog.records]
    withdrawn = [m for m in messages if "relay line claim withdrawn reason=result_waiting" in m]
    rows: dict = {}
    for frame in client.of("response_text"):
        row = rows.setdefault(frame.get("assistant_turn_id"), {
            "parent": frame.get("parent_user_turn_id"), "text": ""})
        row["text"] += frame.get("text") or ""
    line_rows = [r["parent"] for r in rows.values() if box.get("line") and box["line"] in r["text"]]
    spoken = {m.split("result spoken id=")[1].split()[0]: m.split("epoch=")[1].split()[0]
              for m in messages if "result spoken id=d" in m}
    if held:
        assert [m for m in messages if "result held for order id=d2 behind=d1" in m], messages
        assert withdrawn == [], withdrawn
        # The line's epoch is the LINE's: on its task's turn, never on the
        # remark («okay take your time» is turn 3).
        assert line_rows == [f"live-utt:{H.PSID}:1"], rows
        # …and the results go out exactly as before: the earlier-asked one the
        # moment it is done, the held one after it.
        assert spoken == {"d1": "2", "d2": "3"}, spoken
        assert marks[_TUITION] - marks["d1_done"] < 0.5, marks
        assert marks[_VANCE] >= marks[_TUITION], marks
    else:
        # The control: a result that goes out now still withdraws a claim
        # past its hold (F14's own rule, unchanged), and is sent at once.
        assert len(withdrawn) == 1, messages
        assert spoken.get("d1"), spoken
        assert marks[_TUITION] - marks["d1_done"] < 0.5, marks


def test_a_finished_tasks_unapplied_addendum_rides_with_its_result():
    """G6: the caller added to a task while it ran and the running turn never
    applied it (the addendum route answered `finished`/`answer_written`; the
    turns lane stores it on `task.addenda_unapplied`).  The result's delivery
    carries ONE bounded model-facing line naming it; absent means none."""

    from app.services.live_voice_protocol import DelegatedTask, relay_line

    session = _bare_session()
    task = DelegatedTask(delegation_id="d1", offset_ms=0, transcript="ielts general", reply_language="en")
    assert session.result_addendum_note(task) == ""
    task.addenda_unapplied = ["and what about academic?", {"text": "x" * 500}]
    note = session.result_addendum_note(task)
    assert note.startswith("While this ran the caller also asked «and what about academic?")
    assert len(note) < len(relay_line("result_addendum_note", "en", text="")) + 245
    task.reply_language = "fa"
    task.addenda_unapplied = ["آکادمیکش چی؟"]
    assert session.result_addendum_note(task).startswith("در این مدت تماس‌گیرنده «آکادمیکش چی؟»")


@pytest.mark.asyncio
async def test_an_unapplied_addendum_note_goes_out_before_its_result(monkeypatch):
    """…and the pump sends it as an instruction right before the commentary
    (no claim; the result's words prove the delivery as always)."""

    import app.services.live_voice_protocol as live

    H.fast_clocks(monkeypatch)
    answer = "The museum opens at ten."
    H.patch_relay(monkeypatch, think=_fixed_think(answer))
    original = live._LiveSession.deliver_result

    async def with_addendum(self, task, answer_text, claim):
        task.addenda_unapplied = ["and on sundays?"]
        return await original(self, task, answer_text, claim)

    monkeypatch.setattr(live._LiveSession, "deliver_result", with_addendum)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("when does the museum open", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([_config_141(), 1.5, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True, speak_on_commentary=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-w-addnote")

    sent = [
        e["content"] for e in provider.sent
        if e["type"] in ("session.commentary.append", "session.instructions.append")
    ]
    notes = [c for c in sent if "also asked «and on sundays?»" in c]
    assert len(notes) == 1, sent
    assert sent.index(notes[0]) < sent.index(answer), sent


@pytest.mark.asyncio
async def test_an_unspoken_answer_row_still_says_it_is_the_task_result(monkeypatch):
    """G7: the `spoken:false` revision of a delegated row carries
    `record_kind: task_result` like its `spoken:true` twin (10-07 rows 12/17
    had nothing to say what they were)."""

    H.fast_clocks(monkeypatch)
    answer = "The museum opens at ten."
    recorded = H.patch_relay(monkeypatch, think=_fixed_think(answer))

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("when does the museum open", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    config = _config_141()
    config["features"] = list(config["features"]) + ["heard_text"]
    client = H.FakeClient([config, 2.5, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-w-g7")

    flags = [
        s["assistant_voice"] for s in _delegated_rows(recorded["saves"])
        if "spoken" in (s.get("assistant_voice") or {})
    ]
    assert flags and flags[-1]["spoken"] is False, flags
    assert flags[-1].get("record_kind") == "task_result", flags[-1]


@pytest.mark.asyncio
async def test_a_media_result_already_told_is_not_appended_again(monkeypatch):
    """D-R1 (4): the play the model already announced BY NAME is not
    appended a second time (one telling) — media results only."""

    from app.config import settings as app_settings
    from app.services.live_voice_protocol import ClosedEpoch

    monkeypatch.setattr(app_settings, "voice_live_result_order_wait_s", 0.0)
    session = _bare_session()
    told = "«Drake - Come Thru» رو برات گذاشتم 🎵"
    delivery, ref = _session_delivery(
        session, "d1", told, transcript="حالا یه دنه آهنگ دریک واسم پخش کن",
        media={"type": "youtube", "video_id": "v1", "title": "Drake - Come Thru"},
    )
    session.dispatched_at = {"d1": time.monotonic() - 5.0}
    session.adapter._history.append(ClosedEpoch(
        epoch=3, output_id=f"live:{H.PSID}:3", text=told, interrupted=False,
        played_ms=900, born_monotonic=time.monotonic() - 1.0,
    ))
    sent: list = []

    async def send_delegation(phase, task, **extra):
        sent.append((phase, task.delegation_id, extra.get("spoken")))

    session.send_delegation = send_delegation
    assert await session.result_already_said(delivery) is True
    assert delivery.spoken and delivery.words_reason == "already_said"
    assert [r for r in session.persistence.records if r.key == ref]
    # Ruling 35 (F3): the telling was BEFORE the append, i.e. before the
    # task's terminal frame — inside the card's delivery floor — so the phone
    # is told the play was heard (`completed{spoken:true}`).
    assert sent == [("completed", "d1", True)], sent

    other, _ref = _session_delivery(
        session, "d2", "The museum opens at ten.", transcript="museum hours",
    )
    session.dispatched_at["d2"] = time.monotonic() - 5.0
    assert await session.result_already_said(other) is False


def test_a_merged_members_fellow_is_an_other_like_everyone_else():
    """Phase D words-proof-1 REVERSES critic C12: a fellow member of the same
    merged append IS an `other` — excluded, the General telling proved the
    Academic answer on their shared words.  Measured on the recorded corpus,
    a merged telling still proves both members (each member's own facts are
    in it)."""

    from app.services.live_voice_protocol import DelegatedTask, ResultDelivery

    session = _bare_session()

    def delivery(delegation_id, answer):
        made = ResultDelivery(
            task=DelegatedTask(delegation_id=delegation_id, offset_ms=0, transcript="x"),
            answer=answer, claim={},
        )
        session.deliveries[delegation_id] = made
        return made

    head = delivery("d1", "General needs thirty seven correct answers.")
    member = delivery("d2", "Academic needs thirty five correct answers.")
    bystander = delivery("d3", "Robarts closes at eleven tonight.")
    head.companions, member.group_head = [member], head
    others = session.live_answer_tokens(member)
    assert "general" in others, others
    assert {"robarts", "11"} <= others, others
    alone = delivery("d4", "Toronto is cold in January.")
    assert {"general", "academic", "robarts"} <= session.live_answer_tokens(alone)
