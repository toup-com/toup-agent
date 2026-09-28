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
        "request": "the question",
    }
    for key in (
        "status_running", "status_running_nostep", "no_task",
        "result_nudge", "status_result", "detached_result",
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
    client = H.FakeClient([H.config(), 3.0, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=12, db_session_id="db-r12-4b")

    assert len(sent_at) == 2, _commentary(provider)
    assert sent_at[1] - sent_at[0] >= 0.4, (
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
    assert unspoken[0]["result_text"] == answer
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
    assert unspoken[0]["result_text"] == answer
    assert "live_result_unspoken" in _counter_names(counters)
    assert "live_result_attribution_spoiled" in _counter_names(counters), (
        "the relay has to record that it STOPPED being able to attribute"
    )


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
                marks["acked"] = time.monotonic()
                provider.push({
                    "type": "session.commentary.appended",
                    "client_event_id": event["event_id"],
                })

            asyncio.get_running_loop().create_task(late_ack())
        elif kind == "session.instructions.append":
            marks.setdefault("nudged", time.monotonic())
        elif kind == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 3.0, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=14, db_session_id="db-r12-ack")

    assert "acked" in marks and "nudged" in marks, marks
    assert marks["nudged"] - marks["acked"] >= 0.3, (
        "the six-second speak budget ran from the send, not from acceptance"
    )
    reasons = [f.get("reason") for name, f in counters if name == "live_result_nudged"]
    assert reasons == ["no_epoch"], reasons
    verdicts = [
        f.get("reason") for name, f in counters if name == "live_result_unspoken"
    ]
    assert verdicts == ["no_epoch_after_nudge"], verdicts


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
async def test_a_spoiled_deliverys_epoch_is_not_folded_onto_its_delegated_row():
    """A spoiled delivery always ends `spoken:false`, so nothing about that
    epoch may reach its row: a row saying "nobody heard this" and carrying 1234
    ms of playback of it is a diagnostic that contradicts itself, and the one
    row R12.5 makes the record of truth.

    `settle_bound_claims` normally releases the binding first and the question
    never arises. It arises when the epoch CLOSES first — `speak()` closes the
    open epoch and persists it (and can be entered from another task inside the
    single await between the mint and the settle), while only the settle
    releases a spoiled binding."""

    from app.services.live_voice_protocol import (
        ClosedEpoch, DelegatedTask, ResultDelivery,
    )

    session = _bare_session()
    ref = f"live-delegation:{H.PSID}:d1"
    payload = {
        "assistant_text": "The first of March.",
        "assistant_voice": {"source": "delegated", "delegation_id": "d1"},
    }
    claim = session.build_commentary_claim(ref, payload, delegation_id="d1")
    delivery = ResultDelivery(
        task=DelegatedTask(delegation_id="d1", offset_ms=0, transcript="the deadline"),
        answer="The first of March.",
        claim=claim,
    )
    session.deliveries["d1"] = delivery
    session.active_delivery = delivery
    session.note_commentary_epoch_claim(claim)
    session.adapter.provider_event(H.out_text("Still working on it.", 900, 1500))
    assert session.adapter.commentary_epochs == {1: claim}

    # Something else asked the model to speak before this delivery's epoch, so
    # the relay can no longer tell which append the epoch answers.
    session.spoil_active_attribution(
        "interposed_speech",
        own=session.build_commentary_claim("", {}, delegation_id="d2"),
    )
    assert delivery.spoiled

    await session.persist_spoken(ClosedEpoch(
        epoch=1, output_id=f"live:{H.PSID}:1", text="Still working on it.",
        interrupted=False, played_ms=1234,
    ))

    folds = [r for r in session.persistence.records if r.key == ref]
    assert not folds, (
        "an utterance the relay refuses to attribute was recorded as playback "
        f"of the delegated answer: {[r.payload for r in folds]}"
    )
    assert session.persistence.keys() == [f"live-output:{H.PSID}:1"], (
        "the epoch is an ordinary spoken reply — the same row the released "
        "binding produces a moment later"
    )


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

    H.fast_clocks(monkeypatch, voice_live_speak_timeout_s=1.5)
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

    client = H.FakeClient([H.config(), 3.6, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=18, db_session_id="db-r12-statusattr")

    assert voiced, "the scenario requires the status instruction to be spoken"
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
    assert unspoken[0]["result_text"] == first
    assert "live_result_unspoken" in _counter_names(counters)
    assert ("live_result_attribution_spoiled", {"reason": "status_instruction"}) in counters, (
        "the relay has to record WHICH interposed append cost it the attribution"
    )
