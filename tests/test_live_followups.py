"""Follow-ups, commands and relay-authored speech (R2 §C, §D, §E).

Every scenario here was a defect on the 16ee4958 relay, reproduced offline
first (`/private/tmp/toup-voice-r2-scratch/relay-tasks/`) and on the recorded
2026-09-22 call where a production log line exists:

  §C  a status question or a cancel command the relay answers ITSELF became a
      task card ("چی شد میگم", Done, no body — V1+04:15), and the relay's own
      lines were parented onto whatever unrelated turn was newest;
  RT-2  "what happened?" replayed an already-HEARD result over a newer question
      nobody had answered, so that question was never worked on;
  §D  an unrelated question containing "not"/"actually"/«با» — or a related one
      that merely shared words — cancelled healthy running work, and a real
      correction lost the request it was correcting;
  RT-7  a tied "another professor" arrived at the agent with no context at all;
  §E  nothing was ever said during a long tool-backed wait, and the status line
      read the English tool label and the raw search query in Persian;
  RT-9  nothing told the agent that earlier constraints still hold, that a
      person's CURRENT affiliation needs the organisation's own page, or that a
      contradiction with an accepted answer has to be named.

Driven through the real relay (`test_live_harness`) with provider-shaped events
wherever the behaviour is a relay decision; the pure seams are called directly.
"""

import asyncio

import pytest

import test_live_harness as H


def _frames_for(client, did):
    return [f for f in client.of("delegation") if f.get("delegation_id") == did]


def _terminal(client, did):
    return any(
        f.get("phase") in {"completed", "failed", "cancelled", "superseded", "expired"}
        for f in _frames_for(client, did)
    )


def _lifecycle(frames) -> list[dict]:
    """The task_lifecycle observations only — not the historical repeated
    `completed` (spoken:false) verdict, which deliberately carries none of the
    lifecycle fields."""

    return [f for f in frames if "task_revision" in f]


def _commentary(provider) -> list[dict]:
    return provider.of("session.commentary.append")


def _instructions(provider) -> list[str]:
    return [e["content"] for e in provider.of("session.instructions.append")]


def _status_replays(provider, answer) -> list[str]:
    """The status-from-result instruction carrying `answer` — never the
    result NUDGE, which carries the same text for a different reason."""

    return [
        c for c in _instructions(provider)
        if answer in c and ("asked what you found" in c or "پرسید چه چیزی پیدا کردی" in c)
    ]


def _final_turns(client) -> dict[str, str]:
    return {f["turn_id"]: f["text"] for f in client.of("transcript") if f.get("final")}


def _turn_id(client, text) -> str:
    return next(k for k, v in _final_turns(client).items() if v == text)


def _parents_of(client, needle) -> list[str]:
    """The causal parent of every response_text delta carrying `needle`."""

    return [
        f.get("parent_user_turn_id") or ""
        for f in client.of("response_text")
        if needle in (f.get("text") or "")
    ]


async def _wait(pred, timeout=4.0):
    for _ in range(int(timeout / 0.02)):
        if pred():
            return True
        await asyncio.sleep(0.02)
    return False


def _tool_start(relay, call_id="i1", name="web_search", query="toronto robotics faculty"):
    return relay.on_event({
        "type": "tool.start", "call_id": call_id, "name": name, "args": {"query": query},
    })


def _tool_end(relay, call_id="i1", name="web_search"):
    return relay.on_event({
        "type": "tool.end", "call_id": call_id, "name": name,
        "ok": True, "preview": "found", "elapsed_ms": 20,
    })


def _progress_clocks(monkeypatch, after_s):
    """`fast_clocks` plus the §E threshold. Tolerant of the setting being
    absent so that, on a relay without §E, the test fails on its ASSERTION
    (nothing was said) rather than on setup; with the setting present and
    renamed, the 6 s default applies and the assertion fails just the same."""

    from app.config import settings

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    try:
        monkeypatch.setattr(settings, "voice_live_progress_speak_after_s", after_s)
    except (AttributeError, ValueError):
        pass


def _closer(box, first_turn, first_offset, first_id="d1"):
    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.start":
            p.push(H.user_delta(first_turn, 0, first_offset))
            p.push(H.delegation(first_id, first_offset + 50))
        elif e["type"] == "session.close":
            p.push(H.closed())
    return on_send


# ══════════════════════════════════════════════════════════════════════
# §C — a command the relay answers itself is not a job
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_a_status_answer_from_the_result_emits_no_task_card(monkeypatch):
    """V1+04:14.6: `delegation ready item_ER7OsUT chars=10` → `status answered
    from=result` → a Done card titled «چی شد میگم» with no body. The answer is
    speech; the provider's delegation needs no client frame at all."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    thinks = []
    box = {}
    answer = "Professor Ada Lovelace works on robotics."

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        thinks.append(task)
        return answer, "m"

    H.patch_relay(monkeypatch, think=think)

    async def ask_status():
        await _wait(lambda: _terminal(client, "d1"))
        await asyncio.sleep(0.4)
        box["p"].push(H.user_delta("چی شد میگم", 8000, 8800))
        box["p"].push(H.delegation("d2", 8850))
        await _wait(lambda: bool(_status_replays(provider, answer)))

    client = H.FakeClient([H.config(), ask_status, 0.4, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=_closer(box, "find a robotics professor at toronto", 900),
        auto_ack=True, speak_on_commentary=True,
    )
    await H.run_relay(client, provider, timeout=10, db_session_id="db-fu-status-card")

    assert len(thinks) == 1, "a status question must not run a second agent turn"
    assert _status_replays(provider, answer), "it was still answered — from the real result"
    assert _frames_for(client, "d2") == [], (
        "a relay-answered status question must not become a task card"
    )


@pytest.mark.asyncio
async def test_a_cancel_command_is_not_its_own_card(monkeypatch):
    """The TARGET's `cancelled` frame is the outcome. A second card titled
    "cancel that", created then cancelled, is work that never existed."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    started = asyncio.Event()
    box = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        started.set()
        await asyncio.sleep(5)
        return "x", "m"

    H.patch_relay(monkeypatch, think=think)

    async def ask():
        await started.wait()
        box["p"].push(H.user_delta("cancel that", 3000, 3500))
        box["p"].push(H.delegation("d2", 3550))
        await _wait(lambda: _terminal(client, "d1"))

    client = H.FakeClient([H.config(), ask, 0.4, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=_closer(box, "research the toronto robotics faculty", 900), auto_ack=True,
    )
    await H.run_relay(client, provider, timeout=10, db_session_id="db-fu-cancel-card")

    assert [f.get("phase") for f in _frames_for(client, "d1")][-1] == "cancelled"
    assert _frames_for(client, "d2") == [], (
        "the cancel command is not a job; the target's cancelled frame is the outcome"
    )
    spoken = [e for e in _commentary(provider) if e["delegation_id"] == "d2"]
    assert spoken, "the command is still acknowledged out loud"


@pytest.mark.asyncio
async def test_an_ambiguous_cancel_reports_the_code_but_is_not_a_card(monkeypatch):
    """Two tasks running and "cancel that" names neither: the relay asks which
    one (`cancel_target_required` + the spoken line), cancels NOTHING, and the
    command itself leaves no failed card behind."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    running = asyncio.Event()
    cancelled: list[str] = []
    box = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        did = kw.get("delegation_id")
        if did == "d2":
            running.set()
        try:
            await asyncio.sleep(1.5)
        except asyncio.CancelledError:
            cancelled.append(did)
            raise
        return f"answer for {did}", "m"

    H.patch_relay(monkeypatch, think=think)

    async def script():
        await _wait(lambda: any(f.get("phase") == "started" for f in _frames_for(client, "d1")))
        box["p"].push(H.user_delta("what is the weather in paris", 2000, 2600))
        box["p"].push(H.delegation("d2", 2650))
        await running.wait()
        box["p"].push(H.user_delta("cancel that", 4000, 4400))
        box["p"].push(H.delegation("d3", 4450))
        await _wait(lambda: any(
            f.get("code") == "cancel_target_required" for f in client.of("error")
        ))

    client = H.FakeClient([H.config(), script, 1.8, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=_closer(box, "research the toronto robotics faculty", 900), auto_ack=True,
    )
    await H.run_relay(client, provider, timeout=10, db_session_id="db-fu-ambiguous")

    assert [f for f in client.of("error") if f.get("code") == "cancel_target_required"]
    assert cancelled == [], "an ambiguous cancel has no mutation authority"
    assert _frames_for(client, "d3") == [], "no failed card for the command itself"
    assert [e for e in _commentary(provider) if e["delegation_id"] == "d3"], (
        "the caller is asked which task, out loud"
    )


# ══════════════════════════════════════════════════════════════════════
# RT-2 — a heard result is not replayed over a newer unanswered question
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_status_does_not_replay_a_heard_result_over_a_newer_unanswered_turn(monkeypatch):
    """V1+03:37→04:14: the new LLM question was never delegated, and 33 s later
    "what happened?" re-read the robotics result the caller had already heard.
    The newer question has to be worked on instead."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    thinks = []
    box = {}
    first_answer = "Professor Ada Lovelace and Professor Alan Turing work on robotics."

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        thinks.append(task)
        if len(thinks) == 1:
            return first_answer, "m"
        return "Yes, Professor Turing works on large language models.", "m"

    H.patch_relay(monkeypatch, think=think)

    async def script():
        await _wait(lambda: _terminal(client, "d1"))
        await asyncio.sleep(0.5)
        p = box["p"]
        p.push(H.user_delta("are any of them working on llms", 6000, 7000))
        await asyncio.sleep(0.5)
        # Far apart on the provider timeline (> the request-span gap), so the
        # status words are their own request, as on the recorded call (33 s).
        p.push(H.user_delta("what happened", 20000, 20500))
        p.push(H.delegation("d2", 20550))
        await _wait(lambda: _terminal(client, "d2"))

    client = H.FakeClient([H.config(), script, 0.5, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=_closer(box, "find robotics professors at toronto", 900),
        auto_ack=True, speak_on_commentary=True,
    )
    await H.run_relay(client, provider, timeout=10, db_session_id="db-fu-rt2")

    assert not _status_replays(provider, first_answer), (
        "an already-heard result must not be re-announced as 'what you found'"
    )
    assert len(thinks) == 2, "the unanswered newer ask has to be worked on"


@pytest.mark.asyncio
async def test_status_still_replays_an_unheard_result_parented_on_the_status_turn(
        monkeypatch):
    """The other half of RT-2, and RT-3 on the instruction path. The result was
    never SAID (`spoken:false`), so "what happened?" still replays it — and the
    epoch that replay produces answers the status turn, never the unrelated
    (older, still unanswered) question it would otherwise have fenced.

    Pin changed in fix round 2 (addendum 2 item 1): the unrelated question used
    to be asked AFTER the unheard result, and the replay was pinned "even
    though a newer turn exists".  A NEWER unanswered request now overtakes the
    check-in in every branch — the unheard-result one included — and is
    dispatched as itself (tests/test_live_round2_classification.py).  The
    parenting property this test exists for is kept with the question asked
    BEFORE the research: still the newest unanswered turn when the replay is
    voiced, but not newer than the work, so it does not overtake."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    thinks = []
    box = {}
    first_answer = "Professor Ada Lovelace and Professor Alan Turing work on robotics."

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        thinks.append(task)
        return first_answer, "m"

    H.patch_relay(monkeypatch, think=think)

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.start":
            # Separate requests: further apart than the request-span gap.
            p.push(H.user_delta("what time does the library close", 0, 800))
            p.push(H.user_delta("find robotics professors at toronto", 12000, 12900))
            p.push(H.delegation("d1", 12950))
        elif e["type"] == "session.close":
            p.push(H.closed())

    async def script():
        # No speech for the commentary: nudge, then an honest spoken:false.
        await _wait(lambda: any(
            f.get("spoken") is False for f in _frames_for(client, "d1")
        ))
        p = box["p"]
        p.push(H.user_delta("what happened", 20000, 20500))
        p.push(H.delegation("d2", 20550))
        await _wait(lambda: bool(_status_replays(provider, first_answer)))
        # The model obeys the status instruction: a new output epoch.
        p.push(H.out_text("They are Ada Lovelace and Alan Turing.", 20800, 21400))
        p.push(H.out_audio("X"))
        await asyncio.sleep(0.4)

    client = H.FakeClient([H.config(), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-fu-rt2-unheard")

    assert len(thinks) == 1
    assert _status_replays(provider, first_answer), (
        "an unheard result is still what 'what happened?' asks for"
    )
    parents = _parents_of(client, "They are")
    assert parents, client.of("response_text")
    library_turn = _turn_id(client, "what time does the library close")
    status_turn = _turn_id(client, "what happened")
    assert library_turn not in parents, (
        "a replayed old result must not be recorded as the answer to the library question"
    )
    assert set(parents) == {status_turn}


# ══════════════════════════════════════════════════════════════════════
# RT-3 — relay-authored speech carries its own causal parent
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_a_status_replay_does_not_fence_an_older_unanswered_turn(monkeypatch):
    """A question asked before the research, never answered, is the newest
    unconsumed turn when "what did you find" is replayed. Without an explicit
    parent the replay's epoch was recorded as ITS answer and fenced it."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    box = {}
    answer = "Professor Ada Lovelace works on robotics."

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        return answer, "m"

    H.patch_relay(monkeypatch, think=think)

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.start":
            # Separate requests: further apart than the request-span gap.
            p.push(H.user_delta("what time does the library close", 0, 800))
            p.push(H.user_delta("find robotics professors at toronto", 12000, 12900))
            p.push(H.delegation("d1", 12950))
        elif e["type"] == "session.close":
            p.push(H.closed())

    async def script():
        await _wait(lambda: _terminal(client, "d1"))
        await asyncio.sleep(0.5)
        p = box["p"]
        p.push(H.user_delta("what did you find", 20000, 20500))
        p.push(H.delegation("d2", 20550))
        await _wait(lambda: bool(_status_replays(provider, answer)))
        p.push(H.out_text("It is Ada Lovelace.", 20800, 21400))
        p.push(H.out_audio("X"))
        await asyncio.sleep(0.4)

    client = H.FakeClient([H.config(), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True, speak_on_commentary=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-fu-rt3-older")

    parents = _parents_of(client, "It is Ada Lovelace.")
    assert parents
    library = _turn_id(client, "what time does the library close")
    assert library not in parents, "the replay is not the library question's answer"
    assert set(parents) == {_turn_id(client, "what did you find")}


@pytest.mark.asyncio
async def test_the_status_running_line_is_parented_on_the_status_turn(monkeypatch):
    """The progress line built on a bare claim had an EMPTY parent, so
    `consume_direct_turn` recorded it as the answer to the newest unanswered
    turn — here an unrelated library question.

    Pin changed in fix round 2 (addendum 2 item 1, which wins over this pin's
    old `d2 == []` for a question asked WHILE the research ran): a newer
    unanswered request overtakes a check-in on running work and is dispatched
    as itself (tests/test_live_round2_classification.py).  The parenting
    property is kept with the library question asked BEFORE the research —
    still the newest unanswered turn when the line is voiced, but not newer
    than the work, so the check-in is answered from state."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    box = {}
    started = asyncio.Event()

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        await _tool_start(relay, query="toronto robotics")
        started.set()
        await asyncio.sleep(2.0)
        return "Robotics list.", "m"

    H.patch_relay(monkeypatch, think=think)

    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.start":
            # Separate requests: further apart than the request-span gap.
            p.push(H.user_delta("what time does the library close", 0, 800))
            p.push(H.user_delta("research the toronto robotics faculty", 12000, 12900))
            p.push(H.delegation("d1", 12950))
        elif e["type"] == "session.close":
            p.push(H.closed())

    async def script():
        await started.wait()
        p = box["p"]
        await asyncio.sleep(0.4)
        p.push(H.user_delta("what did you find so far", 14000, 14600))
        p.push(H.delegation("d2", 14650))
        await _wait(lambda: any("Still working" in e["content"] for e in _commentary(provider)))
        # The model voices the relay's status commentary.
        p.push(H.out_text("Still searching, a few seconds in.", 14900, 15500))
        p.push(H.out_audio("X"))
        await asyncio.sleep(0.5)

    client = H.FakeClient([H.config(), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-fu-rt3-running")

    parents = _parents_of(client, "Still searching")
    assert parents
    assert _turn_id(client, "what time does the library close") not in parents
    assert set(parents) == {_turn_id(client, "what did you find so far")}
    assert _frames_for(client, "d2") == []


@pytest.mark.asyncio
async def test_a_failure_line_answers_its_own_task_turn(monkeypatch):
    """The failure line ("That didn't go through") is about the failed request.
    With no parent it was recorded as the answer to the newest unanswered turn
    instead — here a question the caller asked while the task was running."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    box = {}
    asked = asyncio.Event()

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        await asked.wait()
        raise RuntimeError("agent container gone")

    H.patch_relay(monkeypatch, think=think)

    async def script():
        await _wait(lambda: any(f.get("phase") == "started" for f in _frames_for(client, "d1")))
        box["p"].push(H.user_delta("what time is it", 3000, 3500))
        await asyncio.sleep(0.3)
        asked.set()
        await _wait(lambda: any(e["delegation_id"] == "d1" for e in _commentary(provider)))
        box["p"].push(H.out_text("Sorry, that one failed.", 4000, 4600))
        box["p"].push(H.out_audio("X"))
        await asyncio.sleep(0.4)

    client = H.FakeClient([H.config(), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=_closer(box, "book a table for two", 900), auto_ack=True,
    )
    await H.run_relay(client, provider, timeout=10, db_session_id="db-fu-rt3-failed")

    parents = _parents_of(client, "that one failed")
    assert parents
    assert _turn_id(client, "what time is it") not in parents
    assert set(parents) == {_turn_id(client, "book a table for two")}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["silent", "rejected"])
async def test_an_unvoiced_relay_line_does_not_capture_a_later_reply(monkeypatch, mode):
    """The price of an explicit parent is that a claim nobody voices would hand
    the NEXT epoch — the model's direct reply to a new question — to the status
    turn. So the claim is bounded: dropped on rejection, and withdrawn once the
    speak budget passes without an epoch. A guard for the new behaviour."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0, voice_live_speak_timeout_s=0.3)
    box = {}
    started = asyncio.Event()

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        await _tool_start(relay)
        started.set()
        await asyncio.sleep(2.5)
        return "Robotics list.", "m"

    H.patch_relay(monkeypatch, think=think)

    async def script():
        await started.wait()
        p = box["p"]
        p.push(H.user_delta("what did you find so far", 3000, 3600))
        p.push(H.delegation("d2", 3650))
        await _wait(lambda: any("Still working" in e["content"] for e in _commentary(provider)))
        # Never voiced. Past the budget, the caller asks something else and
        # the model answers it directly.
        await asyncio.sleep(0.9)
        p.push(H.user_delta("what time is it", 6000, 6500))
        await asyncio.sleep(0.4)
        p.push(H.out_text("It is three o'clock.", 6800, 7300))
        p.push(H.out_audio("X"))
        await asyncio.sleep(0.4)

    client = H.FakeClient([H.config(), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=_closer(box, "research the toronto robotics faculty", 900),
        auto_ack=True,
        reject_appends=("session.commentary.append",) if mode == "rejected" else (),
    )
    await H.run_relay(client, provider, timeout=10, db_session_id=f"db-fu-rt3-{mode}")

    parents = _parents_of(client, "three o'clock")
    assert parents
    assert set(parents) == {_turn_id(client, "what time is it")}


# ══════════════════════════════════════════════════════════════════════
# §D / RT-4 — only an anchored, explicit modification may supersede
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("followup", [
    "is it not going to rain today",
    "what is actually the capital of france",
    "با کی قرار دارم فردا؟",
])
async def test_an_unrelated_question_does_not_cancel_running_work(monkeypatch, followup):
    """A correction word in mid-sentence, or «با» opening an ordinary question,
    killed the running task: 'explicit targeted cancellation only' was false."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    thinks = []
    started = asyncio.Event()
    cancelled = asyncio.Event()
    box = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        thinks.append(task)
        if len(thinks) == 1:
            started.set()
            try:
                await asyncio.sleep(1.2)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return "Robotics faculty list.", "m"
        return "Some other answer.", "m"

    H.patch_relay(monkeypatch, think=think)

    async def ask():
        await started.wait()
        box["p"].push(H.user_delta(followup, 3000, 3800))
        box["p"].push(H.delegation("d2", 3850))

    client = H.FakeClient([H.config(), ask, 2.0, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=_closer(box, "research the toronto robotics faculty", 900), auto_ack=True,
    )
    await H.run_relay(client, provider, timeout=10, db_session_id="db-fu-unrelated")

    assert not cancelled.is_set(), "only an explicit, targeted cancel may stop running work"
    assert "superseded" not in [f.get("phase") for f in _frames_for(client, "d1")]
    assert len(thinks) == 2, "the new question runs as its own task"


@pytest.mark.asyncio
async def test_a_related_but_independent_request_runs_separately(monkeypatch):
    """"find robotics professors at the university of toronto" while the LLM
    search runs shares 80% of its words. Overlap is not a replacement
    instruction: both run, neither is cancelled, and no relation is claimed."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    thinks = []
    started = asyncio.Event()
    cancelled = asyncio.Event()
    box = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        thinks.append(task)
        if len(thinks) == 1:
            started.set()
            try:
                await asyncio.sleep(1.2)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return "LLM professors: A, B.", "m"
        return "Robotics professors: C, D.", "m"

    H.patch_relay(monkeypatch, think=think)

    async def ask():
        await started.wait()
        box["p"].push(H.user_delta(
            "find robotics professors at the university of toronto", 3000, 3800,
        ))
        box["p"].push(H.delegation("d2", 3850))

    client = H.FakeClient([H.config(), ask, 2.0, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=_closer(box, "find llm professors at the university of toronto", 900),
        auto_ack=True,
    )
    await H.run_relay(client, provider, timeout=10, db_session_id="db-fu-related")

    assert not cancelled.is_set(), "overlap is not an explicit replacement instruction"
    assert [f.get("phase") for f in _frames_for(client, "d1")][-1] == "completed"
    assert len(thinks) == 2
    assert all("relation" not in f for f in _frames_for(client, "d2"))


def test_correction_and_refinement_markers_are_anchored_whole_words():
    """The marker tests were substring/prefix tests on a STRIPPED marker, so
    "no " matched "notify", "add " matched "addresses", «نه» matched «نهار»,
    «با» and «هم» opened ordinary questions, and "not"/"actually" anywhere in
    a sentence made it a correction. Only whole LEADING words count now — and
    only they carry the authority to supersede (`explicit_modification`)."""

    from app.services.live_voice_protocol import classify_followup, explicit_modification

    running = "research the toronto robotics faculty"
    for text in (
        "is it not going to rain today",
        "what is actually the capital of france",
        "notify me when my package arrives",
        "addresses of the robotics labs",
        "without the hotels please",
        "با کی قرار دارم فردا؟",
        "هم اتاقیم کجاست",
        "نهار چی بخورم",
    ):
        assert explicit_modification(text) == "", text
    for text in (
        "is it not going to rain today",
        "what is actually the capital of france",
        "با کی قرار دارم فردا؟",
    ):
        assert classify_followup(text, running) == "unrelated", text
    assert explicit_modification("no, only the downtown campus") == "correction"
    assert explicit_modification("actually make it masters") == "correction"
    assert explicit_modification("نه، فقط پردیس مرکزی") == "correction"
    assert explicit_modification("also only show associate professors") == "refinement"
    assert explicit_modification("فقط استادهای ایرانی") == "refinement"
    assert classify_followup("no, only the downtown campus", running) == "correction"
    assert classify_followup("نه، فقط پردیس مرکزی", running) == "correction"


# ══════════════════════════════════════════════════════════════════════
# §D / RT-5 — a correction carries what it corrects, and says so
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_a_correction_replaces_and_carries_the_original_request(monkeypatch):
    """The replacement agent turn used to receive only "no, only the downtown
    campus" — the institution and the subject were gone."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    calls = []
    started = asyncio.Event()
    box = {}
    original = "find iranian computer science professors at the university of toronto"

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append(task)
        if len(calls) == 1:
            started.set()
            await asyncio.sleep(5)
        return "Downtown: Professor X.", "m"

    recorded = H.patch_relay(monkeypatch, think=think)

    async def correct():
        await started.wait()
        box["p"].push(H.user_delta("no, only the downtown campus", 3000, 3800))
        box["p"].push(H.delegation("d2", 3850))
        await _wait(lambda: _terminal(client, "d2"))

    client = H.FakeClient([H.config(), correct, 0.4, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_closer(box, original, 900), auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-fu-correction")

    assert len(calls) == 2
    assert original in calls[1], "the replacement must carry the request it corrects"
    assert calls[1].rstrip().endswith("Reply in the caller's language."), (
        "the caller's own words stay last"
    )
    assert "Current caller request: no, only the downtown campus" in calls[1]
    assert "superseded" in [f.get("phase") for f in _frames_for(client, "d1")]
    d2 = _lifecycle(_frames_for(client, "d2"))
    assert d2 and all(
        f.get("relation") == {"kind": "replaces", "task_id": "d1"} for f in d2
    ), d2
    assert d2[0]["title"] == "no, only the downtown campus", "the title is still the caller's words"
    rows = [
        s for s in recorded["saves"]
        if str(s.get("assistant_ref") or "").endswith(":d2")
    ]
    assert rows and rows[-1]["assistant_voice"]["related_task_id"] == "d1"


@pytest.mark.asyncio
async def test_the_relation_rides_only_on_the_lifecycle_family(monkeypatch):
    """Additive under `task_lifecycle`: a client that did not negotiate it gets
    byte-for-byte the frames it always got; the agent still gets the request."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    calls = []
    started = asyncio.Event()
    box = {}
    original = "find iranian computer science professors at the university of toronto"

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append(task)
        if len(calls) == 1:
            started.set()
            await asyncio.sleep(5)
        return "Downtown: Professor X.", "m"

    H.patch_relay(monkeypatch, think=think)

    async def correct():
        await started.wait()
        box["p"].push(H.user_delta("no, only the downtown campus", 3000, 3800))
        box["p"].push(H.delegation("d2", 3850))
        await _wait(lambda: _terminal(client, "d2"))

    features = [f for f in H.config()["features"] if f != "task_lifecycle"]
    client = H.FakeClient([H.config(features=features), correct, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_closer(box, original, 900), auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-fu-legacy-relation")

    assert len(calls) == 2 and original in calls[1]
    assert _frames_for(client, "d2")
    assert all("relation" not in f for f in client.of("delegation"))


@pytest.mark.asyncio
async def test_a_correction_after_the_answer_links_without_cancelling(monkeypatch):
    """Nothing is running: "no, only the downtown campus" after the answer was
    heard is a correction of THAT request. It is linked and carries the
    original words; the finished task is not touched."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    calls = []
    box = {}
    original = "find iranian computer science professors at the university of toronto"

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append(task)
        return "Professor X, Professor Y.", "m"

    H.patch_relay(monkeypatch, think=think)

    async def correct():
        await _wait(lambda: _terminal(client, "d1"))
        await asyncio.sleep(0.3)
        box["p"].push(H.user_delta("no, only the downtown campus", 5000, 5800))
        box["p"].push(H.delegation("d2", 5850))
        await _wait(lambda: _terminal(client, "d2"))

    client = H.FakeClient([H.config(), correct, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=_closer(box, original, 900), auto_ack=True, speak_on_commentary=True,
    )
    await H.run_relay(client, provider, timeout=10, db_session_id="db-fu-after")

    assert len(calls) == 2 and original in calls[1]
    d1_phases = [f.get("phase") for f in _frames_for(client, "d1")]
    assert "superseded" not in d1_phases and d1_phases.count("completed") >= 1
    d2 = _lifecycle(_frames_for(client, "d2"))
    assert d2 and all(
        f.get("relation") == {"kind": "replaces", "task_id": "d1"} for f in d2
    ), d2


@pytest.mark.asyncio
async def test_a_continuation_is_linked_to_its_source(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    calls = []
    box = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append(task)
        if len(calls) == 1:
            return "Professor Ada Lovelace works on robotics at Toronto.", "m"
        return "Professor Grace Hopper.", "m"

    H.patch_relay(monkeypatch, think=think)

    async def script():
        await _wait(lambda: _terminal(client, "d1"))
        box["p"].push(H.user_delta("another professor", 2000, 2300))
        box["p"].push(H.delegation("d2", 2350))
        await _wait(lambda: _terminal(client, "d2"))

    client = H.FakeClient([H.config(), script, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=_closer(box, "find a robotics professor at toronto", 600), auto_ack=True,
    )
    await H.run_relay(client, provider, timeout=10, db_session_id="db-fu-continues")

    assert len(calls) == 2
    d2 = _lifecycle(_frames_for(client, "d2"))
    assert d2 and all(
        f.get("relation") == {"kind": "continues", "task_id": "d1"} for f in d2
    ), d2


def test_the_tenant_allowlist_keeps_the_related_task_id():
    """`sessions._clean_voice` drops unknown keys; the link would be written by
    the relay and stored by nobody."""

    from app.api.sessions import _VOICE_KEYS, _clean_voice

    assert "related_task_id" in _VOICE_KEYS
    sent = {"source": "delegated", "delegation_id": "d2", "related_task_id": "d1"}
    assert _clean_voice(dict(sent)) == sent


def test_every_new_scaffold_line_is_blocked_from_a_user_visible_answer():
    """No scaffold reaches a title or a row: an answer that echoes any line of
    the new blocks — whole or truncated — is dropped like the old ones."""

    from app.services.live_voice_protocol import (
        delegated_agent_input,
        sanitize_user_visible_answer,
    )

    for kind in ("replaces", "refines"):
        message = delegated_agent_input(
            "no, only the downtown campus",
            [("find professors at toronto", "Professor X.")],
            related_request="find iranian professors at the university of toronto",
            relation_kind=kind,
        )
        scaffold = message.split("Current caller request:")[0]
        lines = [line for line in scaffold.splitlines() if line.strip()]
        assert lines
        for line in lines:
            if line.startswith("Prior caller request") or line.startswith("Accepted backend"):
                continue
            assert sanitize_user_visible_answer(line) == "", line
            assert sanitize_user_visible_answer(line[: max(24, len(line) * 2 // 3)]) == "", line
    # Ordinary prose that shares a word or two is untouched.
    ok = "Professor Ada Lovelace is currently at the University of Toronto."
    assert sanitize_user_visible_answer(ok) == ok


# ══════════════════════════════════════════════════════════════════════
# RT-7 — a tied continuation keeps the ordinary ledger
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_a_tied_continuation_keeps_the_prior_context(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    calls = []
    box = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append(task)
        if len(calls) == 1:
            return "Professor Ada Lovelace works on robotics at Toronto.", "m"
        if len(calls) == 2:
            return "Professor Alan Turing works on language models at Toronto.", "m"
        return "Professor Grace Hopper.", "m"

    H.patch_relay(monkeypatch, think=think)

    async def script():
        p = box["p"]
        await _wait(lambda: _terminal(client, "d1"))
        p.push(H.user_delta("find a language model professor at toronto", 1000, 1500))
        p.push(H.delegation("d2", 1550))
        await _wait(lambda: _terminal(client, "d2"))
        p.push(H.user_delta("another professor", 2000, 2300))
        p.push(H.delegation("d3", 2350))
        await _wait(lambda: _terminal(client, "d3"))

    client = H.FakeClient([H.config(), script, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=_closer(box, "find a robotics professor at toronto", 600), auto_ack=True,
    )
    await H.run_relay(client, provider, timeout=10, db_session_id="db-fu-tied")

    assert len(calls) == 3
    assert "toronto" in calls[2].lower(), "a follow-up must not arrive with no prior context"
    assert "Continuation contract" not in calls[2], "a tie still inherits no contract"
    assert all("relation" not in f for f in _frames_for(client, "d3"))


# ══════════════════════════════════════════════════════════════════════
# RT-9 — the follow-up grounding contract
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_a_follow_up_carries_the_prior_request_and_the_grounding_rules(monkeypatch):
    """V2: "where is he a professor now" came back with a former affiliation
    that contradicted the accepted answer, and nothing said so."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    calls = []
    box = {}
    original = "find iranian computer science professors at the university of toronto downtown"

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append(task)
        return "Professor X is at the University of Toronto.", "m"

    H.patch_relay(monkeypatch, think=think)

    async def script():
        await _wait(lambda: _terminal(client, "d1"))
        box["p"].push(H.user_delta("where is he a professor now", 3000, 3600))
        box["p"].push(H.delegation("d2", 3650))
        await _wait(lambda: _terminal(client, "d2"))

    client = H.FakeClient([H.config(), script, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_closer(box, original, 900), auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-fu-grounding")

    assert len(calls) == 2
    first, follow_up = calls
    assert "Follow-up rules" not in first, "a first request carries no follow-up scaffold"
    assert original in follow_up
    low = follow_up.lower()
    assert "still apply unless the caller changes them" in low
    assert "current position on the organisation's own current page" in low
    assert "former role" in low
    assert "differs from an accepted backend result" in low
    assert follow_up.rstrip().endswith("Reply in the caller's language.")


@pytest.mark.parametrize(
    ("follow_up", "correction"),
    [
        # The 2026-10-06 call: a NEW question right after the F1 answer.  It
        # shares no entity with it and opens with no correction marker.
        ("یو‌اف‌سی چی", False),
        # …and the explicit opener the brief names keeps the correction rule.
        ("نه منظورم UFC", True),
    ],
)
@pytest.mark.asyncio
async def test_a_new_question_after_an_answer_is_not_framed_as_its_correction(
    monkeypatch, follow_up, correction,
):
    """Voice round 2026-10-06 (DESIGN-R2 D-R7 / D-6a).  Every follow-up after
    an accepted answer got rule 3 — "If your answer differs from an accepted
    backend result above, say so explicitly" — and the tenant model obeyed it:
    the UFC answer opened «آها، UFC رو می‌گفتی، نه فرمول یک!».  An UNLINKED
    follow-up gets the new-question rule instead; the accepted ledger stays
    (pronoun resolution needs it)."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    calls = []
    box = {}
    f1 = "مسابقه بعدی فرمول یک کی و کجاست"

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append(task)
        return "مسابقه بعدی فرمول یک، جایزه بزرگ سنگاپور است.", "m"

    H.patch_relay(monkeypatch, think=think)

    async def script():
        await _wait(lambda: _terminal(client, "d1"))
        box["p"].push(H.user_delta(follow_up, 3000, 3600))
        box["p"].push(H.delegation("d2", 3650))
        await _wait(lambda: _terminal(client, "d2"))

    client = H.FakeClient([H.config(), script, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_closer(box, f1, 900), auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id=f"db-fu-new-{correction}")

    assert len(calls) == 2
    second = calls[1]
    assert "Prior caller request: " + f1 in second, "the ledger stays for pronoun resolution"
    low = second.lower()
    if correction:
        assert "differs from an accepted backend result" in low
        assert "still apply unless the caller changes them" in low
        assert "The current request is a NEW question" not in second
    else:
        assert "differs from an accepted backend result" not in low
        assert "still apply unless the caller changes them" not in low
        assert "The current request is a NEW question" in second
        # Rule 2 is about answering ANY role question and stays.
        assert "former role" in low


def test_the_voice_research_rule_allows_one_authoritative_fetch_for_current_affiliation():
    """Rule 1 pushed the agent to answer from snippets, which is how a stale
    Georgia Tech snippet could outrank the university's own faculty page."""

    from app.agent.agent_runner import CHANNEL_GUIDANCE

    rule = CHANNEL_GUIDANCE["voice"].split("  2. ")[0]
    assert "at most ONE source" in rule, "the latency budget stays"
    assert "CURRENT" in rule and "affiliation" in rule
    assert "own current page" in rule
    assert "former" in rule


# ══════════════════════════════════════════════════════════════════════
# §E — grounded progress speech
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_one_grounded_progress_line_is_spoken_after_a_real_tool_start(monkeypatch):
    """Nothing was ever said during a 5-16 s delegation. After a REAL tool.start
    and a noticeable silence the caller hears one short line — never before the
    tool starts, never with the raw query, and parented on the task's turn."""

    _progress_clocks(monkeypatch, 0.3)
    box = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        box["before_tool"] = list(_commentary(box["p"]))
        await _tool_start(relay, query="toronto robotics faculty")
        await asyncio.sleep(1.0)
        box["during"] = list(_commentary(box["p"]))
        await _tool_end(relay)
        return "Professor Ada Lovelace.", "m"

    H.patch_relay(monkeypatch, think=think)

    client = H.FakeClient([H.config(), 1.6, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=_closer(box, "research the toronto robotics faculty", 900),
        auto_ack=True, speak_on_commentary=True,
    )
    await H.run_relay(client, provider, timeout=10, db_session_id="db-fu-progress")

    assert box.get("before_tool") == []
    during = box.get("during", [])
    assert len(during) == 1, "one grounded, spoken progress line during a noticeable wait"
    line = during[0]
    assert line["delegation_id"] == "d1"
    assert "toronto robotics faculty" not in line["content"], "no raw query aloud"
    assert "Searching the web" in line["content"], "named from the real step's kind"
    first_turn = _turn_id(client, "research the toronto robotics faculty")
    # The fake model says what it is given (voice round 2026-10-07): the
    # progress line's own words are what was voiced, parented on the task.
    assert _parents_of(client, line["content"])[0] == first_turn
    assert [e for e in _commentary(provider) if "Professor Ada Lovelace." in e["content"]], (
        "the answer still arrives"
    )


@pytest.mark.asyncio
async def test_no_progress_is_spoken_without_a_real_tool_start(monkeypatch):
    """R6: a `tool.intent` is a plan, not evidence of work. Silence stays
    silence until something real has started. A guard for §E."""

    _progress_clocks(monkeypatch, 0.2)
    box = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        await relay.on_event({"type": "tool.intent", "name": "web_search"})
        await asyncio.sleep(0.9)
        box["during"] = list(_commentary(box["p"]))
        return "Done.", "m"

    H.patch_relay(monkeypatch, think=think)
    client = H.FakeClient([H.config(), 1.3, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=_closer(box, "research this for me", 600),
        auto_ack=True, speak_on_commentary=True,
    )
    await H.run_relay(client, provider, timeout=10, db_session_id="db-fu-no-progress")

    assert box.get("during") == []


@pytest.mark.asyncio
async def test_persian_progress_and_status_lines_carry_no_english_label_or_query(monkeypatch):
    """«هنوز دارم روش کار می‌کنم — Searching the web: toronto robotics» was the
    spoken Persian status line. The label is localized from the tool's kind and
    the query never reaches speech; the thinking note keeps the detail."""

    _progress_clocks(monkeypatch, 0.3)
    box = {}
    query = "toronto robotics professors"

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        await _tool_start(relay, query=query)
        await asyncio.sleep(0.8)
        box["p"].push(H.user_delta("چی شد؟", 4000, 4400))
        box["p"].push(H.delegation("d2", 4450))
        await asyncio.sleep(0.6)
        await _tool_end(relay)
        return "استاد ایدا.", "m"

    H.patch_relay(monkeypatch, think=think)
    client = H.FakeClient([H.config(), 2.0, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=_closer(box, "استادهای رباتیک دانشگاه تورنتو رو پیدا کن", 900),
        auto_ack=True, speak_on_commentary=True,
    )
    await H.run_relay(client, provider, timeout=10, db_session_id="db-fu-fa-progress")

    relay_lines = [
        e["content"] for e in _commentary(provider)
        if e["delegation_id"] == "d1" and "استاد ایدا" not in e["content"]
    ]
    assert len(relay_lines) >= 2, relay_lines   # the progress line and the status line
    for content in relay_lines:
        assert "Searching" not in content and "web" not in content.lower(), content
        assert query not in content, content
        assert "جستجو" in content, content
    notes = [e["content"] for e in provider.of("session.thinking.append")]
    assert any(query in n for n in notes), "the model's own notes keep the detail"


def test_every_new_relay_line_exists_in_both_languages():
    from app.services.live_voice_protocol import relay_line

    for key in ("progress_step", "progress_nostep"):
        en = relay_line(key, "en", step="x")
        fa = relay_line(key, "fa", step="x")
        assert en and fa and en != fa, key


# ══════════════════════════════════════════════════════════════════════
# D-R5 (2026-10-06 call, 4:19–4:40): the recorded tail, replayed
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_d_r5_the_recorded_huh_carries_the_owed_request_not_itself(monkeypatch, caplog):
    """«فورمولوان» closed and got no reply; 18 s later the caller said «ها»
    and the model delegated.  On build 141 the delegation's request — and its
    card title — was «ها», and the F1 reply owed was superseded.  Now «ها» is
    phatic: it supersedes nothing, and the delegation is rebound to the owed
    request (`phatic_anchor`) — its words, its turn, its title."""

    import logging

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    asked: list[str] = []
    seen: list[dict] = []

    async def think(_user_id, task_text, _session_id, **kw):
        asked.append(task_text)
        seen.append(kw)
        return "The next Formula One race is the Mexico City Grand Prix.", "m"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("فورمولوان", 0, 800))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def script():
        await asyncio.sleep(0.5)
        box["p"].push(H.user_delta("ها", 18000, 18300))
        box["p"].push(H.delegation("ep3kr", 18350))
        for _ in range(200):
            if _terminal(client, "ep3kr"):
                break
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.2)

    import app.services.live_voice_protocol as live

    titled: dict = {}
    original_start = live._LiveSession.start_delegation

    async def spy_start(self, task, *args, **kwargs):
        # What RC's `task_title` reads: the request the task now IS.
        titled[task.delegation_id] = task.title_request
        return await original_start(self, task, *args, **kwargs)

    monkeypatch.setattr(live._LiveSession, "start_delegation", spy_start)
    client = H.FakeClient([H.config(), script, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    with caplog.at_level(logging.INFO):
        await H.run_relay(client, provider, timeout=8, db_session_id="db-d-r5-huh")

    owed = _turn_id(client, "فورمولوان")
    huh = _turn_id(client, "ها")
    finals = {f["turn_id"]: f for f in client.of("transcript") if f.get("final")}
    assert finals[huh].get("phatic") is True
    created = [f for f in _frames_for(client, "ep3kr") if f.get("phase") == "created"]
    assert created, _frames_for(client, "ep3kr")
    assert created[0]["parent_user_turn_id"] == owed
    assert "ها" not in (created[0].get("title") or "").split(), created[0].get("title")
    assert owed in created[0]["request_turn_ids"]
    assert asked and "فورمولوان" in asked[0] and asked[0].strip() != "ها"
    # Phase D (tests-guards-2): the D-R8 (c) / D-R5 BODY keys reach the tenant
    # — deleting the `delegated_turn_fields` spread from `_think`'s call
    # passed every suite (the stub swallowed **kwargs; the frame assertion
    # above fell back to `[owed]`).
    kw = [k for k in seen if k.get("delegation_id") == "ep3kr"][-1]
    assert kw["parent_user_turn_id"] == owed
    assert owed in kw["request_turn_ids"]
    assert kw["task_title"] == "فورمولوان"
    assert kw["anchor_occurred_at"]
    assert titled.get("ep3kr") == "فورمولوان", titled
    assert "status overtaken" in caplog.text and "reason=phatic_anchor" in caplog.text
    superseded = [
        line for line in caplog.text.splitlines()
        if "unanswered_superseded" in line and owed in line
    ]
    assert superseded == [], superseded


# ── Voice round 2026-10-07 (D-R3(2)/(3)): the ADDENDUM join ───────────────
# A question tail, a restatement or a continuation that starts within
# `voice_live_addendum_window_ms` of a running (or queued) request's last word
# completes THAT request: one card, forwarded to the running tenant turn on
# `POST /api/v1/internal/voice-delegations/addendum`.  Every other short
# follow-up keeps today's path (the controls above stay green unchanged).

from app.config import settings as _settings  # noqa: E402
import app.services.live_voice_protocol as _live  # noqa: E402

PROF = "قوی‌ترین استاد یوافتی توی بحث ال‌ال‌ام رو پیدا کن"


def _addenda_on(monkeypatch, window=5000, **kw):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0, **kw)
    monkeypatch.setattr(_settings, "voice_live_addendum_window_ms", window)


def _tenant(monkeypatch, answer=None, status=None):
    from app.api import ws_realtime as rt

    posts: list[dict] = []

    async def api(_url, _key, _method, path, json_body=None, timeout=None, outcome=None, **_kw):
        if path.endswith("/voice-delegations/addendum"):
            posts.append(dict(json_body or {}))
            if status is not None:
                if outcome is not None:
                    outcome.update(status=status)
                return None
            return answer if answer is not None else {"accepted": True, "state": "queued"}
        return None

    async def vps(_user_id):
        return ("https://tenant.example", "agent-key")

    monkeypatch.setattr(rt, "_vps_api", api)
    return posts, vps


def _boot(box):
    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.close":
            p.push(H.closed())
    return on_send


@pytest.mark.asyncio
async def test_a_tail_joins_the_models_own_running_task_and_is_forwarded(monkeypatch, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    _addenda_on(monkeypatch)
    posts, vps = _tenant(monkeypatch)
    calls: list[dict] = []

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append({"did": kw.get("delegation_id"), "task": task})
        await asyncio.sleep(1.0)
        if out is not None:
            out["addenda_applied"] = [box["kie"]] if box.get("kie") else []
        return "کالین رافل.", "m"

    H.patch_relay(monkeypatch, think=think, vps=vps)
    box: dict = {}

    async def script():
        p = box["p"]
        p.push(H.user_delta(PROF, 1000, 2500))
        p.push(H.delegation("d1", 2550))
        await _wait(lambda: calls, timeout=3.0)
        p.push(H.user_delta("کیه", 3900, 4100))
        await _wait(lambda: "کیه" in _final_turns(client).values(), timeout=4.0)
        box["kie"] = _turn_id(client, "کیه")
        # The model delegates the tail on its own as well — absorbed.
        p.push(H.delegation("d2", 4150))
        await _wait(lambda: _terminal(client, "d1"), timeout=4.0)
        await asyncio.sleep(0.3)

    client = H.FakeClient([
        H.config(features=H.config()["features"] + ["reply_status"]), 0.1, script, 0.2,
        {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=_boot(box), auto_ack=True)
    await H.run_relay(client, provider, timeout=15, db_session_id="db-r3-addendum-d1")

    kie = box["kie"]
    assert [c["did"] for c in calls] == ["d1"], calls
    assert [f["delegation_id"] for f in client.of("delegation") if f["phase"] == "created"] == ["d1"]
    logs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith(f"[LIVE] addendum joined id=d1 turn_id={kie} kind=question_tail ")
               for m in logs), [m for m in logs if "addendum" in m]
    assert any(m.startswith(f"[LIVE] delegation absorbed id=d2 turn_id={kie} reason=addendum owner=d1")
               for m in logs), [m for m in logs if "absorbed" in m]
    assert [(p["task_id"], p["addendum_id"], p["text"]) for p in posts] == [("d1", kie, "کیه")]
    frames = _lifecycle(_frames_for(client, "d1"))
    held = frames[0]["request_turn_ids"]
    grown = next(f for f in frames if kie in (f.get("request_turn_ids") or []))
    assert grown["request_turn_ids"][:len(held)] == held and grown["title"].endswith("کیه")
    assert ("delegated", "d1") in [(f["outcome"], f.get("task_id"))
                                   for f in client.of("reply_status")
                                   if f.get("user_turn_id") == kie]


@pytest.mark.asyncio
@pytest.mark.parametrize("wall_s", [4.0, 40.0])
@pytest.mark.parametrize("path", ["addendum", "seam"])
async def test_a_tail_long_after_its_request_on_the_wall_clock_is_its_own_task(
    monkeypatch, caplog, path, wall_s,
):
    """Phase D2 ruling 49, end to end: the request, then «کیه» 1.4 s later on
    the provider timeline — the relay heard the request `wall_s` before the
    tail.  4 s: one card, as before (the tail joins; the model's own
    delegation for it is absorbed — after the tail by the addendum, at its
    onset by the seam).  40 s: not joined, not absorbed — d2 is a task of
    its own."""

    import logging

    import app.services.live_voice_protocol as live

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    _addenda_on(monkeypatch, voice_live_seam_onset_ms=400)
    original_add = live.UtteranceAssembler.add

    def add(self, text, start_ms, end_ms):
        closed, current = original_add(self, text, start_ms, end_ms)
        if text == PROF:
            # The relay heard the request `wall_s` ago (its input timeline
            # stalled in between).
            current.last_heard_monotonic -= wall_s
        return closed, current

    monkeypatch.setattr(live.UtteranceAssembler, "add", add)
    calls: list[str] = []

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append(kw.get("delegation_id"))
        await asyncio.sleep(1.5 if len(calls) == 1 else 0.05)
        return "کالین رافل.", "m"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}

    async def script():
        p = box["p"]
        p.push(H.user_delta(PROF, 1000, 2500))
        p.push(H.delegation("d1", 2550))
        await _wait(lambda: calls, timeout=3.0)
        p.push(H.user_delta("کیه", 3900, 4100))
        if path == "seam":
            p.push(H.delegation("d2", 3900))          # decided at the tail's onset
        await _wait(lambda: "کیه" in _final_turns(client).values(), timeout=4.0)
        if path == "addendum":
            p.push(H.delegation("d2", 4150))
        await asyncio.sleep(1.6)

    client = H.FakeClient([H.config(), 0.1, script, 0.2, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_boot(box), auto_ack=True)
    await H.run_relay(client, provider, timeout=15, db_session_id=f"db-r49-{path}-{int(wall_s)}")
    logs = [r.getMessage() for r in caplog.records]
    if wall_s < 15:
        assert calls == ["d1"], calls
        assert [m for m in logs if "[LIVE] addendum joined id=d1" in m], logs
        assert [m for m in logs if m.startswith("[LIVE] delegation absorbed id=d2")
                and f"reason={path} owner=d1" in m], logs
        assert not [m for m in logs if "wall_gap" in m], logs
        return
    assert calls == ["d1", "d2"], calls
    assert not [m for m in logs if "[LIVE] addendum joined" in m], logs
    assert [m for m in logs if "[LIVE] addendum refused id=d1" in m and "reason=wall_gap" in m], logs
    if path == "seam":
        assert [m for m in logs if "[LIVE] seam skipped id=d2" in m and "reason=wall_gap" in m], logs


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["late", "finished"])
async def test_a_tail_outside_the_window_or_after_the_answer_is_its_own_request(
    monkeypatch, caplog, case,
):
    """5001 ms after the request, or after its task FINISHED: today's path —
    the model's delegation for the tail is a task of its own."""

    import logging

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    _addenda_on(monkeypatch, window=1000)
    calls: list[str] = []

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append(kw.get("delegation_id"))
        await asyncio.sleep(1.0 if case == "late" and len(calls) == 1 else 0.05)
        return "یک جواب.", "m"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}

    async def script():
        p = box["p"]
        p.push(H.user_delta(PROF, 1000, 2500))
        p.push(H.delegation("d1", 2550))
        await _wait(lambda: calls, timeout=3.0)
        if case == "finished":
            await _wait(lambda: _terminal(client, "d1"), timeout=3.0)
        start = 3501 if case == "late" else 2900
        p.push(H.user_delta("کیه", start, start + 200))
        await asyncio.sleep(0.2)
        p.push(H.delegation("d2", start + 250))
        await _wait(lambda: len(calls) >= 2, timeout=3.0)
        await asyncio.sleep(1.2)

    client = H.FakeClient([H.config(), 0.1, script, 0.2, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_boot(box), auto_ack=True)
    await H.run_relay(client, provider, timeout=15, db_session_id=f"db-r3-addendum-{case}")
    assert calls == ["d1", "d2"], calls
    assert not any("[LIVE] addendum joined" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_tail_joined_to_a_QUEUED_task_rides_its_request_and_its_placement(monkeypatch, caplog):
    """G2: a queued task is not running yet — the tail goes into the request it
    will run (no forward), and the saved task is placed after the LAST request
    turn (`anchor_occurred_at` = the tail's own row)."""

    import logging

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    _addenda_on(monkeypatch, voice_live_max_concurrent_delegations=1)
    posts, vps = _tenant(monkeypatch)
    calls: list[dict] = []

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append({"did": kw.get("delegation_id"), "task": task, **kw})
        await asyncio.sleep(1.2 if len(calls) == 1 else 0.05)
        return "یک جواب.", "m"

    saves = H.patch_relay(monkeypatch, think=think, vps=vps)["saves"]
    box: dict = {}

    async def script():
        p = box["p"]
        p.push(H.user_delta("the weather in toronto please", 1000, 2000))
        p.push(H.delegation("d1", 2050))
        await _wait(lambda: calls, timeout=3.0)
        p.push(H.user_delta(PROF, 2600, 4000))
        p.push(H.delegation("d2", 4050))
        await asyncio.sleep(0.3)
        p.push(H.user_delta("کیه", 4800, 5000))
        await _wait(lambda: len(calls) >= 2, timeout=4.0)
        await asyncio.sleep(0.4)

    client = H.FakeClient([H.config(), 0.1, script, 0.2, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_boot(box), auto_ack=True)
    await H.run_relay(client, provider, timeout=15, db_session_id="db-r3-addendum-queued")

    assert [c["did"] for c in calls] == ["d1", "d2"], calls
    logs = [r.getMessage() for r in caplog.records]
    assert any("[LIVE] addendum joined id=d2 " in m and "target=queued" in m for m in logs)
    assert posts == [], "a queued task carries the tail in its request; nothing to forward"
    d2 = calls[1]
    assert "کیه" in d2["task"] and d2["display_request"].endswith("کیه")
    kie_row = next(s for s in saves if s.get("user_text") == "کیه")
    stamp = kie_row["user_occurred_at"]
    from datetime import timezone

    assert d2["anchor_occurred_at"] == (
        stamp.astimezone(timezone.utc).replace(tzinfo=None).isoformat(timespec="milliseconds")
    )


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("کیه", "question_tail"), ("چیه", "question_tail"), ("کدومه", "question_tail"),
        ("کجاست", "question_tail"), ("چنده", "question_tail"), ("کی بود", "question_tail"),
        ("خب کیه", "question_tail"), ("who is it", "question_tail"),
        ("which one", "question_tail"), ("how much", "question_tail"),
        ("قوی‌ترین استاد یو‌اف‌تی", "restatement"),
        ("توی دانشگاه تورنتو", "continuation"), ("که روی ال‌ال‌ام کار می‌کنه", "continuation"),
        # Never an addendum: repair initiators, D-R6 corrections, an answer to
        # the agent, new requests with their own predicate or content, a
        # consent, a new-request opener, a status question, a correction.
        ("چی؟", ""), ("what?", ""), ("رضوان", ""), ("افوانه", ""), ("هوای تورنتو", ""),
        ("weather?", ""), ("حتما", ""), ("تورنتو", ""), ("با کی قرار دارم فردا؟", ""),
        ("برای فردا هم یه قرار بذار", ""), ("and the weather in toronto", ""),
        ("سرچ کن", ""), ("یه چیز دیگه هم بگو", ""), ("وضعیت چی شد", ""),
        ("نه، منظورم تورنتو بود", ""),
        # Phase D (turns-startup-1): a NEW request opening with a preposition
        # carries an imperative verb the closed predicate list did not know
        # — joined as a "continuation", it was never done — and "you" is no
        # filler outside "thank you"; a bare "why" / «چرا» / "how" is a
        # question of its own.
        ("برای مامانم پیام بفرست", ""), ("با علی تماس بگیر", ""), ("تو تقویمم بنویس", ""),
        ("با صدای بلند بخون", ""), ("از یخچال آب بیار", ""), ("برای فردا یه یادآور بساز", ""),
        ("at 9 wake me up", ""), ("in the calendar add it", ""),
        ("where are you", ""), ("who are you", ""), ("why", ""), ("چرا", ""), ("how", ""),
        ("برای نمره هشت", "continuation"), ("in toronto", "continuation"),
    ],
)
def test_the_addendum_kinds_are_closed(text, kind):
    request = ". چیز اه قوی‌ترین استاد... اه یو‌اف‌تی توی بحث ال‌ال‌ام"
    assert _live.addendum_kind(text, request) == kind


def test_a_related_but_independent_request_is_never_a_restatement():
    """The `:626` control: 75–80 % of the running request is a different
    request; only EVERY content word held is a restatement."""

    assert _live.addendum_kind(
        "find robotics professors at the university of toronto",
        "research the toronto LLM faculty at the university of toronto",
    ) == ""
    assert _live.addendum_kind(
        "addresses of the robotics labs", "research the toronto robotics faculty",
    ) == ""


def _target_session(monkeypatch, *, window=5000):
    """A `__new__`-built session with exactly what `addendum_target` reads."""

    monkeypatch.setattr(_settings, "voice_live_addendum_window_ms", window)
    session = _live._LiveSession.__new__(_live._LiveSession)
    session.settings = _settings
    session.utterances = _live.UtteranceAssembler("p-target", gap_ms=400, hard_gap_ms=900)
    session.supervisor = _live.DelegationSupervisor(max_concurrent=2)
    session.results_owed = set()
    session.deliveries = {}
    request = session.utterances
    request.add(PROF, 1000, 2500)
    turn = request.close("gap")
    turn.causal_speech = True
    task = _live.DelegatedTask(delegation_id="d1", offset_ms=2550, transcript=PROF,
                               turn_id=turn.turn_id, request_turn_ids=[turn.turn_id])
    turn.consumed_by, turn.dispatched = "d1", True
    task.span = [turn]
    session.supervisor.running["d1"] = _live.RunningDelegation(task=task, started_monotonic=0.0)
    return session, task


def _tail(session, text, start, end):
    session.utterances.add(text, start, end)
    turn = session.utterances.close("gap")
    turn.causal_speech = True
    return turn


def test_the_addendum_bounds_hold(monkeypatch):
    """Each bound on its own session (a tail left unconsumed in the queue is
    itself a request turn between the task and the next tail)."""

    def target(text, start, end, **task_fields):
        session, task = _target_session(monkeypatch)
        for key, value in task_fields.items():
            setattr(task, key, value)
        return session.addendum_target(_tail(session, text, start, end))

    assert target("کیه", 3900, 4100)[1:] == ("question_tail", 1400)
    # ≤3 per task.
    assert target("کیه", 3900, 4100,
                  addenda=[{"text": "کیه"}, {"text": "کجاست"}, {"text": "چنده"}]) is None
    assert target("کیه", 3900, 4100, addenda=[{"text": "کیه"}, {"text": "کجاست"}]) is not None
    # ≤240 characters together.
    assert target("کجاست", 3900, 4100, addenda=[{"text": "x" * 118}, {"text": "y" * 118}]) is None
    # ≤120 characters each (a 130-character restatement).
    long = ("قوی‌ترین استاد یو‌اف‌تی " * 6).strip()
    assert len(long) > 120
    assert target(long, 3900, 4100) is None
    # The window: 5000 ms joins, 5001 ms does not.
    assert target("کیه", 7500, 7600) is not None
    assert target("کیه", 7501, 7600) is None
    # The result already owed (in delivery).
    session, _task = _target_session(monkeypatch)
    session.results_owed.add("d1")
    assert session.addendum_target(_tail(session, "کیه", 3900, 4100)) is None


def test_the_seam_owner_rules(monkeypatch):
    """D-R3(1): decided at the ONSET (≤400 ms, never past the turn's end), the
    turn before consumed by a task STILL WORKING, within the window."""

    from types import SimpleNamespace

    monkeypatch.setattr(_settings, "voice_live_seam_onset_ms", 400)

    def owner(offset, start=3500, end=4700, **running):
        session, task = _target_session(monkeypatch)
        entry = session.supervisor.running["d1"]
        for key, value in running.items():
            setattr(entry, key, value)
        latest = _tail(session, "هوای تورنتو چطوره؟", start, end)
        got = session.seam_owner(SimpleNamespace(offset_ms=offset), latest)
        return got and got[0].delegation_id

    assert owner(3500) == "d1"
    assert owner(3900) == "d1"
    assert owner(3901) is None                       # past the onset bound
    assert owner(3600, start=3500, end=3550) is None  # past the turn's own end
    assert owner(3499) is None                       # before the turn: not its seam
    assert owner(7501, start=7501, end=8000) is None  # 5001 ms after the request
    assert owner(3500, ended=True) is None           # the owner finished
    assert owner(3500, cancelled=True) is None
    monkeypatch.setattr(_settings, "voice_live_seam_onset_ms", 0)
    assert owner(3500) is None                       # the switch


@pytest.mark.parametrize("wall_s", [4.0, 40.0])
def test_the_join_and_the_seam_are_bounded_on_the_relays_own_clock(monkeypatch, caplog, wall_s):
    """Phase D2 ruling 49 (turns-startup-9): both windows are on the
    INPUT-AUDIO timeline, which stalls whenever the phone uplinks nothing
    (silent idle under music before the request's final; 17.6 s of skew on
    10-07).  «کیه» 1.4 s after the request on that timeline but 40 s after it
    on the relay's clock is not its tail, and a model decision arriving that
    late is not the request's own.  The relay's clock bounds both: 3 × the
    5 s window (15 s) from the request's last word as the relay heard it."""

    import logging
    import time
    from types import SimpleNamespace

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    monkeypatch.setattr(_settings, "voice_live_seam_onset_ms", 400)
    session, task = _target_session(monkeypatch)
    session.user_id = "user-1"
    request = task.span[0]
    tail = _tail(session, "کیه", 3900, 4100)
    request.last_heard_monotonic = tail.seen_monotonic - wall_s
    got = session.addendum_target(tail)
    seam = session.seam_owner(SimpleNamespace(
        delegation_id="d-model", offset_ms=3900, created_monotonic=request.last_heard_monotonic + wall_s,
    ), tail)
    logs = [r.getMessage() for r in caplog.records]
    if wall_s < 15:
        assert got is not None and got[1:] == ("question_tail", 1400), got
        assert seam is not None and seam[0].delegation_id == "d1", seam
        assert not [m for m in logs if "wall_gap" in m], logs
        return
    assert got is None and seam is None, (got, seam)
    assert any(m.startswith(
        f"[LIVE] addendum refused id=d1 turn_id={tail.turn_id} reason=wall_gap gap_ms=40000"
    ) for m in logs), logs
    assert any(m.startswith(
        f"[LIVE] seam skipped id=d-model turn_id={request.turn_id} reason=wall_gap gap_ms=40000"
    ) for m in logs), logs
    # Judged again on the next pass: logged once.
    assert session.addendum_target(tail) is None
    assert len([m for m in caplog.records if "addendum refused" in m.getMessage()]) == 1


def test_a_request_turn_between_the_task_and_the_tail_stops_the_join(monkeypatch):
    session, _task = _target_session(monkeypatch)
    _tail(session, "the weather in toronto please", 2800, 3500)
    assert session.addendum_target(_tail(session, "کیه", 3900, 4100)) is None


def test_media_work_is_never_an_addendum_target(monkeypatch):
    session, task = _target_session(monkeypatch)
    task.intent = "media_control"
    assert session.addendum_target(_tail(session, "کیه", 3900, 4100)) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer", "status", "outcome"),
    [
        ({"accepted": True, "state": "queued"}, None, "accepted"),
        ({"accepted": True, "state": "duplicate"}, None, "duplicate"),
        ({"accepted": False, "reason": "finished"}, None, "finished"),
        ({"accepted": False, "reason": "answer_written"}, None, "answer_written"),
        ({"accepted": False, "reason": "not_running"}, None, "not_running"),
        ({"accepted": False, "reason": "limit"}, None, "limit"),
        ({"accepted": False, "reason": "mystery"}, None, "failed"),
        (None, 404, "unsupported"),
        (None, 500, "failed"),
    ],
)
async def test_the_addendum_forward_outcomes(monkeypatch, answer, status, outcome):
    posts, vps = _tenant(monkeypatch, answer=answer, status=status)
    from app.api import ws_realtime as rt

    monkeypatch.setattr(rt, "_get_vps_info", vps)
    session = _live._LiveSession.__new__(_live._LiveSession)
    session.user_id, session.db_session_id, session.tenant_addenda = "u1", "db-1", True
    task = _live.DelegatedTask(delegation_id="d1", offset_ms=0, transcript=PROF, reply_language="fa")
    assert await session.post_tenant_addendum(task, "live-utt:p:5", "کیه") == outcome
    assert posts[0] == {"session_id": "db-1", "task_id": "d1", "addendum_id": "live-utt:p:5",
                        "text": "کیه", "lang": "fa"}
    if outcome == "unsupported":
        # An image without the route: the relay stops posting for the session.
        assert await session.post_tenant_addendum(task, "live-utt:p:6", "کجاست") == "unsupported"
        assert len(posts) == 1
    # The late title rides the same route with empty text.
    posts.clear()
    session.tenant_addenda = True
    await session.post_tenant_addendum(task, "title:d1", "", task_title="استاد ال‌ال‌ام یوافتی")
    assert posts and posts[0]["text"] == "" and posts[0]["task_title"] == "استاد ال‌ال‌ام یوافتی"


def test_the_tenants_applied_list_decides_what_the_delivery_must_still_answer():
    session = _live._LiveSession.__new__(_live._LiveSession)
    task = _live.DelegatedTask(delegation_id="d1", offset_ms=0, transcript=PROF)
    task.addenda = [
        {"id": "t5", "text": "کیه", "forwarded": True},
        {"id": "t6", "text": "کجاست", "forwarded": True},
        {"id": "t7", "text": "توی تورنتو", "forwarded": False},   # rode a queued request
    ]
    task.addenda_unapplied = ["کیه", "کجاست"]
    # An image without the route sends no list: the presumption stands.
    session.note_addenda_applied(task, {})
    assert task.addenda_unapplied == ["کیه", "کجاست"]
    session.note_addenda_applied(task, {"addenda_applied": ["t5"]})
    assert task.addenda_unapplied == ["کجاست"]
    # Phase C seam: what RB stores is exactly what RA's delivery line reads.
    task.reply_language = "fa"
    note = session.result_addendum_note(task)
    assert "«کجاست»" in note and "کیه" not in note, note
    session.note_addenda_applied(task, {"addenda_applied": ["t5", "t6"]})
    assert task.addenda_unapplied == []


@pytest.mark.asyncio
@pytest.mark.parametrize("older", [False, True], ids=["alone", "older_task_running"])
async def test_a_tail_that_closed_before_its_requests_delegation_still_joins(
    monkeypatch, caplog, older,
):
    """The model's delegation for the request can arrive AFTER the tail closed
    (both pending at once).  The request binds first; the tail joins it in the
    same pass, and the tail's own delegation is absorbed — one card.

    Phase D turns-startup-10: ALSO while an unrelated older task runs — the
    tail was judged at its close against that older task only, stamped seen,
    and never judged again once its own request bound: a second card."""

    import logging

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    _addenda_on(monkeypatch)
    calls: list[str] = []

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append(kw.get("delegation_id"))
        await asyncio.sleep(6.0 if kw.get("delegation_id") == "d0" else 0.8)
        return "کالین رافل.", "m"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}

    async def script():
        p = box["p"]
        if older:
            p.push(H.user_delta("هوای تورنتو رو سرچ کن", 100, 600))
            p.push(H.delegation("d0", 650))
            await _wait(lambda: calls, timeout=3.0)
        p.push(H.user_delta(PROF, 1000, 2500))
        await _wait(lambda: PROF in _final_turns(client).values(), timeout=4.0)
        p.push(H.user_delta("کیه", 3900, 4100))
        await _wait(lambda: "کیه" in _final_turns(client).values(), timeout=4.0)
        p.push(H.delegation("d1", 2550))
        p.push(H.delegation("d2", 4150))
        await _wait(lambda: _terminal(client, "d1"), timeout=4.0)
        await asyncio.sleep(0.3)

    client = H.FakeClient([H.config(), 0.1, script, 0.2, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_boot(box), auto_ack=True)
    await H.run_relay(
        client, provider, timeout=15, db_session_id=f"db-r3-addendum-late-target-{older}",
    )
    kie = _turn_id(client, "کیه")
    assert [c for c in calls if c != "d0"] == ["d1"], calls
    logs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith(f"[LIVE] addendum joined id=d1 turn_id={kie} ") for m in logs)
    assert any(m.startswith(f"[LIVE] delegation absorbed id=d2 turn_id={kie} reason=addendum")
               for m in logs), [m for m in logs if "absorbed" in m or "addendum" in m]
