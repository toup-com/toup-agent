"""Classification may not grant authority the caller's words do not carry.

Fix round after the R2 adversarial review (findings F7, F8, F9, F11, F12 and
the critic's addendum items 3, 4 and 5).  Every scenario here was reproduced on
the integrated relay first (probes under
`/private/tmp/toup-voice-r2-review/probes/verify-relay-tasks-*` and
`.../completeness-critic/`), and each test asserts the CORRECT behaviour:

  F7   "cancel my three pm meeting with john" / «جلسه فردا با علی رو لغو کن»
       cancelled the one running research task and never ran;
  F8   a short request holding a status marker ("what happened in the raptors
       game last night", «هنوز بارون میاد تو تورنتو؟») was answered from state —
       "still working" about other work, or the old heard result re-read;
  F9   "actually …", "also …", «فقط …» in front of an unrelated request
       superseded the only running task and carried it as "the request the
       caller is correcting";
  F11  a second correction carried only the first correction's words — the
       subject (Iranian CS professors at UofT) was gone;
  F12  "never mind" with nothing running became a card and a full agent turn;
  add. 3  a status check-in that found a newer unanswered request dispatched
       the bare status words as research (V1: a card titled «چی شد میگم»);
  add. 4  an unmarked qualifier after dispatch («توی پردیس سنت جورج») ran with
       no context at all;
  add. 5  "after that, email it to me" ran at once, concurrently, without the
       first result.

Controls (marked CONTROL) pin what must NOT change: bare cancels still cancel,
check-ins are still answered from state, real corrections still supersede, and a
cancel with an object still runs when nothing is running.

Driven through the real relay (`test_live_harness`) with provider-shaped
events; the pure seams are called directly.
"""

from __future__ import annotations

import asyncio

import pytest

import test_live_harness as H


RESEARCH = "research the toronto robotics faculty"
ORIGINAL = "find iranian computer science professors at the university of toronto"
SUBJECT = "iranian computer science professors"


def _frames_for(client, did):
    return [f for f in client.of("delegation") if f.get("delegation_id") == did]


def _phases(client, did):
    return [f.get("phase") for f in _frames_for(client, did)]


def _lifecycle(frames):
    return [f for f in frames if "task_revision" in f]


def _terminal(client, did):
    return any(
        f.get("phase") in {"completed", "failed", "cancelled", "superseded", "expired"}
        for f in _frames_for(client, did)
    )


def _commentary(provider):
    return [e["content"] for e in provider.of("session.commentary.append")]


def _instructions(provider):
    return [e["content"] for e in provider.of("session.instructions.append")]


def _finals(client) -> dict[str, str]:
    return {f["turn_id"]: f["text"] for f in client.of("transcript") if f.get("final")}


def _turn_id(client, text) -> str:
    return next(k for k, v in _finals(client).items() if v == text)


async def _wait(pred, timeout=4.0):
    for _ in range(int(timeout / 0.02)):
        if pred():
            return True
        await asyncio.sleep(0.02)
    return False


def _opener(box, first_turn, first_offset=900, first_id="d1"):
    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.start":
            p.push(H.user_delta(first_turn, 0, first_offset))
            p.push(H.delegation(first_id, first_offset + 50))
        elif e["type"] == "session.close":
            p.push(H.closed())
    return on_send


async def _while_running(monkeypatch, followup, *, running=RESEARCH, db, hold=1.5):
    """d1 (`running`) is mid-think when the caller says `followup` (d2)."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    inputs: dict[str, str] = {}
    displays: dict[str, str] = {}
    killed: list[str] = []
    started = asyncio.Event()
    box: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        did = kw.get("delegation_id")
        inputs[did] = task
        displays[did] = kw.get("display_request") or ""
        if did == "d1":
            started.set()
            try:
                await asyncio.sleep(hold)
            except asyncio.CancelledError:
                killed.append(did)
                raise
            return "Robotics faculty: Professor Ada Lovelace.", "m"
        return "Done: " + (kw.get("display_request") or ""), "m"

    H.patch_relay(monkeypatch, think=think)

    async def script():
        await started.wait()
        await asyncio.sleep(0.1)
        box["p"].push(H.user_delta(followup, 3000, 3800))
        box["p"].push(H.delegation("d2", 3850))
        await _wait(lambda: _terminal(client, "d1"), timeout=hold + 2.0)
        await _wait(lambda: _terminal(client, "d2") or not _frames_for(client, "d2"), 1.0)
        await asyncio.sleep(0.2)

    client = H.FakeClient([H.config(), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_opener(box, running), auto_ack=True)
    await H.run_relay(client, provider, timeout=12, db_session_id=db)
    return client, provider, inputs, displays, killed


# ══════════════════════════════════════════════════════════════════════
# F7 — a cancel VERB about something else is a request, not a command
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("followup", [
    "cancel my three pm meeting with john",
    "جلسه فردا با علی رو لغو کن",
    "cancel my netflix subscription",
])
async def test_a_cancel_verb_about_something_else_runs_and_the_work_survives(monkeypatch, followup):
    client, provider, inputs, displays, killed = await _while_running(
        monkeypatch, followup, db=f"db-auth-f7-{abs(hash(followup))}",
    )
    assert killed == [], "a cancel verb about another object killed the running research"
    assert _phases(client, "d1")[-1] == "completed", _phases(client, "d1")
    assert displays.get("d2") == followup, "the caller's own request never ran"
    assert _phases(client, "d2")[0] == "created"
    assert not any("Stopped." in c or "متوقف شد" in c for c in _commentary(provider))


@pytest.mark.asyncio
@pytest.mark.parametrize("followup", [
    "cancel that", "stop the search", "جستجو رو لغو کن", "cancel the robotics research",
])
async def test_control_a_bare_or_naming_cancel_still_cancels_the_running_task(monkeypatch, followup):
    """CONTROL: an empty residue (or one the task holds) is still a command."""

    client, provider, inputs, displays, killed = await _while_running(
        monkeypatch, followup, db=f"db-auth-f7c-{abs(hash(followup))}",
    )
    assert killed == ["d1"], killed
    assert _phases(client, "d1")[-1] == "cancelled"
    assert "d2" not in inputs and _frames_for(client, "d2") == []


def test_the_cancel_residue_is_what_the_cancel_names():
    from app.services.live_voice_protocol import cancel_residue, classify_followup

    for bare in (
        "cancel that", "cancel it", "stop the search", "never mind", "بی‌خیال", "ولش کن",
        "جستجو رو لغو کن", "کنسلش کن", "please cancel it", "cancel that search",
        "wait wait never mind", "cancel the second one",
    ):
        assert classify_followup(bare, RESEARCH) == "cancel", bare
        assert cancel_residue(bare) == set(), bare
    assert cancel_residue("cancel the robotics research") == {"robotics"}
    assert {"meeting", "john"} <= cancel_residue("cancel my three pm meeting with john")
    assert {"جلسه", "علی"} <= cancel_residue("جلسه فردا با علی رو لغو کن")


# ══════════════════════════════════════════════════════════════════════
# F8 — a status marker inside a request of its own is not a check-in
# ══════════════════════════════════════════════════════════════════════

MARKER_REQUESTS = [
    "what happened in the raptors game last night",
    "هنوز بارون میاد تو تورنتو؟",
    "what's the status of my amazon order",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("followup", MARKER_REQUESTS)
async def test_a_request_holding_a_status_marker_is_dispatched_while_work_runs(monkeypatch, followup):
    client, provider, inputs, displays, killed = await _while_running(
        monkeypatch, followup, db=f"db-auth-f8a-{abs(hash(followup))}",
    )
    assert displays.get("d2") == followup, (
        f"never dispatched; the relay said {_commentary(provider)!r}"
    )
    assert not any(
        "Still working" in c or "هنوز دارم روش کار می‌کنم" in c for c in _commentary(provider)
    )
    assert killed == []


async def _after_heard_result(monkeypatch, second, *, db):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    thinks: list[str] = []
    answer = "Professor Ada Lovelace works on robotics at Toronto."
    box: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        thinks.append(kw.get("display_request") or "")
        return (answer if len(thinks) == 1 else "Second answer."), "m"

    H.patch_relay(monkeypatch, think=think)

    async def script():
        await _wait(lambda: _terminal(client, "d1"))
        await asyncio.sleep(0.6)
        box["p"].push(H.user_delta(second, 20000, 21000))
        box["p"].push(H.delegation("d2", 21050))
        await _wait(lambda: len(thinks) == 2 or bool(_replays(provider, answer)), 2.0)
        await asyncio.sleep(0.3)

    client = H.FakeClient([H.config(), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=_opener(box, "find robotics professors at toronto"),
        auto_ack=True, speak_on_commentary=True,
    )
    await H.run_relay(client, provider, timeout=10, db_session_id=db)
    return client, provider, thinks, answer


def _replays(provider, answer):
    return [
        c for c in _instructions(provider)
        if answer in c and ("asked what you found" in c or "پرسید چه چیزی پیدا کردی" in c)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("second", MARKER_REQUESTS)
async def test_a_request_holding_a_status_marker_after_a_heard_result_is_dispatched(monkeypatch, second):
    client, provider, thinks, answer = await _after_heard_result(
        monkeypatch, second, db=f"db-auth-f8b-{abs(hash(second))}",
    )
    assert thinks == ["find robotics professors at toronto", second], thinks
    assert not _replays(provider, answer), "the old heard result was re-read instead"


@pytest.mark.asyncio
async def test_control_a_check_in_after_a_heard_result_is_still_answered_from_it(monkeypatch):
    """CONTROL: "so what did you find" is a check-in; with nothing newer it
    replays the real result and runs no second agent turn."""

    client, provider, thinks, answer = await _after_heard_result(
        monkeypatch, "so what did you find", db="db-auth-f8c",
    )
    assert len(thinks) == 1 and _replays(provider, answer)
    assert _frames_for(client, "d2") == []


@pytest.mark.asyncio
async def test_control_a_check_in_while_work_runs_is_still_answered_from_state(monkeypatch):
    client, provider, inputs, displays, killed = await _while_running(
        monkeypatch, "what did you find so far", db="db-auth-f8d",
    )
    assert "d2" not in inputs and _frames_for(client, "d2") == []
    assert any("Still working" in c for c in _commentary(provider))


def test_a_status_question_is_a_check_in_on_its_own_words():
    from app.services.live_voice_protocol import classify_followup, is_status_question

    persian_task = "یه استاد برای یادگیری ماشین در دانشگاه تورنتو پیدا کن"
    check_ins = [
        ("what happened", ""), ("what did you find so far", RESEARCH),
        ("so what did you find", RESEARCH), ("any update on the robotics search", RESEARCH),
        ("did you find anything yet or not", RESEARCH), ("how's it going", RESEARCH),
        ("what did you find about the toronto admissions", "search the toronto admissions deadline"),
        ("what's the status of my amazon order", "track my amazon order"),
        ("چی شد میگم", ""), ("خب چی شد", persian_task), ("هنوز؟", persian_task),
        ("هنوز داری میگردی؟", persian_task), ("پیدا نکردی؟", persian_task),
        ("استاد برای یادگیری ماشین پیدا نکردی؟", persian_task),
        ("what did you find earlier", "when is the deadline"),
        ("did you find anything useful", RESEARCH), ("هنوز تموم نشد؟", persian_task),
        ("وضعیت چطوره؟", persian_task),
    ]
    for text, subject in check_ins:
        assert is_status_question(text, subject), (text, subject)
    requests = [
        ("what happened in the raptors game last night", RESEARCH),
        ("هنوز بارون میاد تو تورنتو؟", "استادهای رباتیک تورنتو رو پیدا کن"),
        ("what's the status of my amazon order", RESEARCH),
        ("what happened with my order", RESEARCH),
        ("کارشون ال‌ال‌امم چی شد میگم", "find robotics professors at toronto"),
    ]
    for text, subject in requests:
        assert not is_status_question(text, subject), (text, subject)
    # The classifier's default subject is the running text (R12.3 still holds).
    assert classify_followup(
        "what did you find about the toronto admissions", "search the toronto admissions deadline",
    ) == "status"
    assert classify_followup("what happened in the raptors game last night", RESEARCH) == "unrelated"


# ══════════════════════════════════════════════════════════════════════
# F9 — an opener is not a correction of work it does not name
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("followup", [
    "actually what time is it in tokyo right now",
    "also set a timer for ten minutes",
    "فقط یه سوال، ساعت توکیو چنده؟",
    "actually what is the weather in the city",
    "no worries, what time is it in tokyo",
])
async def test_an_opener_before_an_unrelated_request_never_supersedes(monkeypatch, followup):
    client, provider, inputs, displays, killed = await _while_running(
        monkeypatch, followup, db=f"db-auth-f9-{abs(hash(followup))}",
    )
    assert killed == [], "healthy running work was cancelled by an opener"
    assert not {"superseded", "cancelled"} & set(_phases(client, "d1"))
    assert _phases(client, "d1")[-1] == "completed"
    assert all("relation" not in f for f in _frames_for(client, "d2"))
    second = inputs.get("d2", "")
    assert displays.get("d2") == followup
    assert "Request the caller is correcting" not in second
    assert "Request the caller is refining" not in second


@pytest.mark.asyncio
@pytest.mark.parametrize(("followup", "running", "kind"), [
    ("no, only the downtown campus", ORIGINAL, "replaces"),
    ("no, book thursday with the dentist", "book friday with the dentist", "replaces"),
    # V2: the negating correction shares no word with what it corrects.
    ("نه، کجایی هستش استاده؟ کدوم شهر دنیا", "اوکی، اه کجای استاد", "replaces"),
    ("also only the robotics faculty at the downtown campus", RESEARCH, "refines"),
    ("actually only the robotics faculty", RESEARCH, "replaces"),
])
async def test_control_real_modifications_still_supersede(monkeypatch, followup, running, kind):
    """CONTROL: a negating correction of the one running task, or an opener
    that names what the task holds, still replaces/refines it."""

    client, provider, inputs, displays, killed = await _while_running(
        monkeypatch, followup, running=running, db=f"db-auth-f9c-{abs(hash(followup))}",
    )
    assert "superseded" in _phases(client, "d1"), _phases(client, "d1")
    d2 = _lifecycle(_frames_for(client, "d2"))
    assert d2 and all(f.get("relation") == {"kind": kind, "task_id": "d1"} for f in d2), d2
    assert running in inputs["d2"]


def test_only_a_negating_correction_that_says_something_needs_no_named_target():
    from app.services.live_voice_protocol import explicit_modification, negating_correction

    for yes in (
        "no, only the downtown campus", "no, thursday", "instead only associate professors",
        "نه، کجایی هستش استاده؟ کدوم شهر دنیا", "i meant the downtown campus",
    ):
        assert negating_correction(yes), yes
    for no in (
        "actually what time is it in tokyo right now", "also set a timer for ten minutes",
        "فقط یه سوال، ساعت توکیو چنده؟", "no worries, what time is it", "نه مرسی، ساعت چنده",
        "no", "نه", "not that one",
    ):
        assert not negating_correction(no), no
    # The classifier is unchanged: these still READ as corrections/refinements;
    # only the authority they carry needs evidence.
    assert explicit_modification("actually make it masters") == "correction"
    assert explicit_modification("also set a timer for ten minutes") == "refinement"


# ══════════════════════════════════════════════════════════════════════
# F11 — a chain of corrections keeps the request that holds the subject
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize(("second", "third", "kind"), [
    ("no, only the downtown campus", "actually only associate professors", "replaces"),
    ("no, only the downtown campus", "and only associate professors", "refines"),
])
async def test_a_second_correction_still_carries_the_original_request(monkeypatch, second, third, kind):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    inputs: list[str] = []
    box: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        inputs.append(task)
        if len(inputs) < 3:
            await asyncio.sleep(5)   # still running when the next correction lands
        return "Dr. A, associate professor, St. George.", "m"

    H.patch_relay(monkeypatch, think=think)

    async def script():
        await _wait(lambda: len(inputs) == 1)
        box["p"].push(H.user_delta(second, 12000, 12800))
        box["p"].push(H.delegation("d2", 12850))
        await _wait(lambda: len(inputs) == 2)
        box["p"].push(H.user_delta(third, 24000, 24800))
        box["p"].push(H.delegation("d3", 24850))
        await _wait(lambda: len(inputs) == 3)
        await asyncio.sleep(0.3)

    client = H.FakeClient([H.config(), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_opener(box, ORIGINAL), auto_ack=True)
    await H.run_relay(
        client, provider, timeout=14, db_session_id=f"db-auth-f11-{abs(hash((second, third)))}",
    )

    assert len(inputs) == 3, inputs
    assert SUBJECT in inputs[1]
    d3 = inputs[2]
    assert SUBJECT in d3, f"the second correction lost the subject:\n{d3}"
    assert d3.index(ORIGINAL) < d3.index(second), "the root request comes first"
    assert d3.rstrip().endswith("Current caller request: " + third) or (
        "Current caller request: " + third in d3
    )
    relations = [f.get("relation") for f in _lifecycle(_frames_for(client, "d3"))]
    assert relations and all(r == {"kind": kind, "task_id": "d2"} for r in relations), relations
    assert "superseded" in _phases(client, "d1") and "superseded" in _phases(client, "d2")


@pytest.mark.asyncio
async def test_a_correction_of_a_finished_correction_still_carries_the_original_request(monkeypatch):
    """The chain survives the task that carried it finishing: d2 (a correction
    of d1) completes, then "actually only associate professors" links to d2
    through the outcome record — and still names the subject."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    inputs: list[str] = []
    box: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        inputs.append(task)
        if len(inputs) == 1:
            await asyncio.sleep(5)
        return "Downtown: Dr. A and Dr. B.", "m"

    H.patch_relay(monkeypatch, think=think)

    async def script():
        await _wait(lambda: len(inputs) == 1)
        box["p"].push(H.user_delta("no, only the downtown campus", 12000, 12800))
        box["p"].push(H.delegation("d2", 12850))
        await _wait(lambda: _terminal(client, "d2"))
        await asyncio.sleep(0.4)
        box["p"].push(H.user_delta("actually only associate professors", 24000, 24800))
        box["p"].push(H.delegation("d3", 24850))
        await _wait(lambda: len(inputs) == 3)
        await asyncio.sleep(0.3)

    client = H.FakeClient([H.config(), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_opener(box, ORIGINAL), auto_ack=True, speak_on_commentary=True)
    await H.run_relay(client, provider, timeout=14, db_session_id="db-auth-f11-finished")

    assert len(inputs) == 3, inputs
    assert _phases(client, "d2")[-1] == "completed"
    assert SUBJECT in inputs[2].split("Current caller request:")[0].split(
        "Request the caller is correcting",
    )[-1], inputs[2]
    relations = [f.get("relation") for f in _lifecycle(_frames_for(client, "d3"))]
    assert relations and all(r == {"kind": "replaces", "task_id": "d2"} for r in relations)


def test_the_request_chain_keeps_the_root_and_the_newest():
    from app.services.live_voice_protocol import (
        chain_requests, delegated_agent_input, sanitize_user_visible_answer,
    )

    chain = chain_requests(chain_requests("", ORIGINAL), "no, only the downtown campus")
    assert chain.split("\n") == [ORIGINAL, "no, only the downtown campus"]
    long_chain = chain
    for i in range(40):
        long_chain = chain_requests(long_chain, f"also only people hired in {1980 + i} or later")
    lines = long_chain.split("\n")
    assert lines[0] == ORIGINAL and lines[-1].endswith("2019 or later")
    assert len(long_chain) <= 1200
    message = delegated_agent_input(
        "actually only associate professors", [],
        related_request=chain, relation_kind="replaces",
    )
    assert ORIGINAL in message and "no, only the downtown campus" in message
    for line in message.split("Current caller request:")[0].splitlines():
        if line.strip():
            assert sanitize_user_visible_answer(line) == "", line


# ══════════════════════════════════════════════════════════════════════
# F12 — a dismissal with nothing in flight is not a job
# ══════════════════════════════════════════════════════════════════════

async def _after_finished(monkeypatch, prior, second, *, db, prior_answer="Professor Ada Lovelace works on robotics."):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    thinks: list[str] = []
    box: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        thinks.append(task)
        return (prior_answer if len(thinks) == 1 else "Okay, done."), "m"

    H.patch_relay(monkeypatch, think=think)

    async def script():
        await _wait(lambda: _terminal(client, "d1"))
        await asyncio.sleep(0.5)
        box["p"].push(H.user_delta(second, 8000, 8600))
        box["p"].push(H.delegation("d2", 8650))
        await _wait(lambda: _terminal(client, "d2"), timeout=2.0)
        await asyncio.sleep(0.2)

    client = H.FakeClient([H.config(), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=_opener(box, prior), auto_ack=True, speak_on_commentary=True,
    )
    await H.run_relay(client, provider, timeout=10, db_session_id=db)
    return client, provider, thinks


@pytest.mark.asyncio
@pytest.mark.parametrize(("text", "line"), [
    ("never mind", "Nothing is running right now."),
    ("بی‌خیال", "الان هیچ کاری در حال اجرا نیست."),
])
async def test_a_bare_dismissal_with_nothing_running_is_answered_not_run(monkeypatch, text, line):
    client, provider, thinks = await _after_finished(
        monkeypatch, "find a robotics professor at toronto", text,
        db=f"db-auth-f12-{len(text)}",
    )
    assert len(thinks) == 1, "a dismissal ran a full agent turn"
    assert _frames_for(client, "d2") == [], "a dismissal became a task card"
    spoken = [e for e in provider.of("session.commentary.append") if e["delegation_id"] == "d2"]
    assert [e["content"] for e in spoken] == [line]


@pytest.mark.asyncio
@pytest.mark.parametrize(("prior", "text"), [
    ("find a robotics professor at toronto", "cancel my 3pm meeting"),
    ("find a robotics professor at toronto", "لغو جلسه فردا"),
    ("set a reminder for 5pm to call mom", "cancel that"),
])
async def test_control_a_cancel_with_an_object_with_nothing_running_still_runs(monkeypatch, prior, text):
    """CONTROL: with nothing running, "cancel my 3pm meeting" is a request,
    and "cancel that" after a reminder is an undo only the agent can do."""

    client, provider, thinks = await _after_finished(
        monkeypatch, prior, text, db=f"db-auth-f12c-{abs(hash((prior, text)))}",
    )
    assert len(thinks) == 2
    assert _phases(client, "d2")[0] == "created"


# ══════════════════════════════════════════════════════════════════════
# addendum 3 — a check-in overtaken by an unanswered request dispatches THAT
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_v1_the_second_status_dispatches_the_unanswered_llm_question_itself(monkeypatch):
    """The production-faithful V1 replay: «کارشون ال‌ال‌امم» barges into the
    heard robotics replay and is never delegated; 12 s of input timeline later
    (outside the request span) «چی شد میگم» is.  The relay used to dispatch the
    bare status words — a card titled «چی شد میگم» and an agent turn that never
    saw the question.  It dispatches the QUESTION: its words, turn and title."""

    import test_live_three_videos as TV

    call = TV.Call(monkeypatch, features=TV.TF132, name="auth-v1")
    TV._v1_tenant(call)
    seen: dict = {}

    async def scenario(call):
        await TV._v1_until_first_status(call, seen)
        await call.utter(TV.LLM_QUESTION, after={0: lambda: call.barge_in(19800)})
        seen["llm_turn"] = await call.final(TV._words(TV.LLM_QUESTION))
        await call.idle(1.5, input_ms=12000)
        await call.utter(TV.STATUS)
        call.delegate("d-status-again")
        await call.done("d-status-again")
        await call.quiet()

    await call.run(scenario, timeout=60)
    question = TV._words(TV.LLM_QUESTION)
    record = call.tenant.call("d-status-again")
    assert record["display"] == question, record["display"]
    assert record["task"].rstrip().split("Current caller request: ")[-1].startswith(question)
    created = call.phone.created_for("d-status-again")
    assert created["title"] == question
    assert created["turn_id"] == seen["llm_turn"]
    status_turns = [
        f["turn_id"] for f in call.phone.finals() if f["text"] == TV._words(TV.STATUS)
    ]
    assert len(status_turns) == 2, status_turns
    ids = created.get("request_turn_ids") or []
    # The check-in is folded into the task that answers it (the app covers it),
    # and the question is the anchor.  In SPOKEN order (addendum 2 item 2):
    # this pin read [status, question] ("anchor last") before request_turn_ids
    # became chronological; the question was asked first.
    assert ids == [seen["llm_turn"], status_turns[-1]], ids
    assert call.instructions("toup-live-status-")[1:] == [], "the heard result was re-read"
    assert call.counted("live_status_dispatched", reason="newer_turn") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("check_in", ["what happened", "پس چی شد؟"])
async def test_a_check_in_joined_to_an_unanswered_question_dispatches_the_question(
        monkeypatch, check_in):
    """Inside the span: "are any of them working on llms" (no answer) and,
    after a pause, "what happened" — one delegation.  After a HEARD result the
    question is dispatched as itself; the status words are not its request,
    and a Persian check-in does not make the English question's reply Persian."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    thinks: list[str] = []
    langs: list[str] = []
    answer = "Professor Ada Lovelace and Professor Alan Turing work on robotics."
    box: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        thinks.append(kw.get("display_request") or "")
        langs.append(kw.get("reply_language") or "")
        return (answer if len(thinks) == 1 else "Yes, Professor Turing works on LLMs."), "m"

    H.patch_relay(monkeypatch, think=think)

    async def script():
        await _wait(lambda: _terminal(client, "d1"))
        await asyncio.sleep(0.5)
        p = box["p"]
        p.push(H.user_delta("are any of them working on llms", 6000, 7000))
        await asyncio.sleep(0.4)
        p.push(H.user_delta(check_in, 8000, 8500))
        p.push(H.delegation("d2", 8550))
        await _wait(lambda: _terminal(client, "d2"))
        await asyncio.sleep(0.2)

    client = H.FakeClient([H.config(), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=_opener(box, "find robotics professors at toronto"),
        auto_ack=True, speak_on_commentary=True,
    )
    await H.run_relay(
        client, provider, timeout=10, db_session_id=f"db-auth-a3-span-{len(check_in)}",
    )

    assert thinks == ["find robotics professors at toronto", "are any of them working on llms"]
    assert langs[1] == "en", langs
    assert not _replays(provider, answer)
    created = [f for f in _frames_for(client, "d2") if f.get("phase") == "created"]
    assert created and created[0]["title"] == "are any of them working on llms"
    assert created[0]["turn_id"] == _turn_id(client, "are any of them working on llms")
    # Spoken order, the anchor wherever it falls (addendum 2 item 2; this pin
    # read [check-in, question] — "anchor last" — before).
    assert created[0].get("request_turn_ids") == [
        _turn_id(client, "are any of them working on llms"), _turn_id(client, check_in),
    ]


@pytest.mark.asyncio
async def test_a_check_in_joined_to_a_question_while_work_runs_dispatches_the_question(monkeypatch):
    """The same inside a running task: "who is the dean there?" is ignored, and
    "what happened?" follows within the span.  "Still working on it" would
    consume the question; it is dispatched, and the research keeps running."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    displays: list[str] = []
    killed: list[str] = []
    started = asyncio.Event()
    box: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        displays.append(kw.get("display_request") or "")
        if kw.get("delegation_id") == "d1":
            started.set()
            try:
                await asyncio.sleep(1.5)
            except asyncio.CancelledError:
                killed.append("d1")
                raise
            return "Robotics faculty list.", "m"
        return "The dean is Professor X.", "m"

    H.patch_relay(monkeypatch, think=think)

    async def script():
        await started.wait()
        p = box["p"]
        p.push(H.user_delta("who is the dean there?", 3000, 3800))
        await asyncio.sleep(0.4)
        p.push(H.user_delta("what happened?", 5000, 5500))
        p.push(H.delegation("d2", 5550))
        await _wait(lambda: _terminal(client, "d1") and _terminal(client, "d2"), 4.0)

    client = H.FakeClient([H.config(), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_opener(box, RESEARCH), auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-auth-a3-running")

    assert displays == [RESEARCH, "who is the dean there?"], displays
    assert killed == [] and _phases(client, "d1")[-1] == "completed"
    assert not any("Still working" in c for c in _commentary(provider))


@pytest.mark.asyncio
async def test_a_turn_the_model_answered_directly_does_not_overtake_a_check_in(monkeypatch):
    """Unanswered means NOTHING answered it (the §F reask rule): "how are you"
    that the model answered out loud is not a request the check-in must
    dispatch instead — "what did you find" still hears the real result."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    thinks: list[str] = []
    answer = "Professor Ada Lovelace works on robotics at Toronto."
    box: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        thinks.append(kw.get("display_request") or "")
        return answer, "m"

    H.patch_relay(monkeypatch, think=think)

    async def script():
        await _wait(lambda: _terminal(client, "d1"))
        await asyncio.sleep(0.6)
        p = box["p"]
        p.push(H.user_delta("how are you doing today", 6000, 7000))
        await asyncio.sleep(0.3)
        p.push(H.out_text("I'm doing well, thanks for asking.", 7200, 7900))
        p.push(H.out_audio("X"))
        await asyncio.sleep(0.5)
        p.push(H.user_delta("what did you find", 20000, 20500))
        p.push(H.delegation("d2", 20550))
        await _wait(lambda: bool(_replays(provider, answer)) or len(thinks) == 2, 2.0)
        await asyncio.sleep(0.2)

    client = H.FakeClient([H.config(), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=_opener(box, "find robotics professors at toronto"),
        auto_ack=True, speak_on_commentary=True,
    )
    await H.run_relay(client, provider, timeout=10, db_session_id="db-auth-a3-direct")

    assert thinks == ["find robotics professors at toronto"], thinks
    assert _replays(provider, answer)
    assert _frames_for(client, "d2") == []


# ══════════════════════════════════════════════════════════════════════
# addendum 4 — an unmarked follow-up keeps the work it may modify, non-binding
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("qualifier", ["توی پردیس سنت جورج", "in the computer science department"])
async def test_an_unmarked_qualifier_after_dispatch_carries_the_running_request(monkeypatch, qualifier):
    client, provider, inputs, displays, killed = await _while_running(
        monkeypatch, qualifier, running="استادای رباتیک یوآفتی رو پیدا کن",
        db=f"db-auth-a4-{abs(hash(qualifier))}",
    )
    second = inputs["d2"]
    assert displays["d2"] == qualifier, "the qualifier is still the card's request"
    assert "Work still in progress when the caller said this" in second
    assert "non-binding context — apply it only if the current request modifies it" in second
    assert "استادای رباتیک یوآفتی رو پیدا کن" in second
    assert second.rstrip().split("Current caller request: ")[-1].startswith(qualifier)
    assert all("relation" not in f for f in _frames_for(client, "d2")), "no relation without evidence"
    assert killed == [] and _phases(client, "d1")[-1] == "completed"


@pytest.mark.asyncio
async def test_a_follow_up_right_after_an_answer_carries_that_request_as_context(monkeypatch):
    client, provider, thinks = await _after_finished(
        monkeypatch, "find a robotics professor at toronto", "what about waterloo",
        db="db-auth-a4-previous",
    )
    assert len(thinks) == 2
    assert "Request made just before this one" in thinks[1]
    assert "find a robotics professor at toronto" in thinks[1]
    assert all("relation" not in f for f in _frames_for(client, "d2"))


# ══════════════════════════════════════════════════════════════════════
# addendum 5 — "after that …" waits for the work it follows
# ══════════════════════════════════════════════════════════════════════

async def _sequenced(monkeypatch, followup, *, db, first_fails=False, correction=None):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    order: list[str] = []
    inputs: dict[str, str] = {}
    gate = asyncio.Event()
    started = asyncio.Event()
    box: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        did = kw.get("delegation_id")
        order.append(did)
        inputs[did] = task
        if did == "d1":
            started.set()
            await gate.wait()
            if first_fails:
                raise RuntimeError("agent container gone")
            return "Professor Ada Lovelace, ada@toronto.example.", "m"
        if did == "d3":
            await gate.wait()
            return "Downtown: Professor Grace Hopper.", "m"
        return "Emailed it to you.", "m"

    H.patch_relay(monkeypatch, think=think)

    async def script():
        await started.wait()
        p = box["p"]
        p.push(H.user_delta(followup, 3000, 3800))
        p.push(H.delegation("d2", 3850))
        await _wait(lambda: len(_frames_for(client, "d2")) >= 2)
        box["d2_before_release"] = list(order)
        if correction:
            p.push(H.user_delta(correction, 6000, 6800))
            p.push(H.delegation("d3", 6850))
            await _wait(lambda: "d3" in order)
            box["d2_after_correction"] = list(order)
        await asyncio.sleep(0.3)
        gate.set()
        await _wait(lambda: _terminal(client, "d2"), 4.0)
        await asyncio.sleep(0.2)

    client = H.FakeClient([H.config(), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_opener(box, RESEARCH), auto_ack=True, speak_on_commentary=True)
    await H.run_relay(client, provider, timeout=12, db_session_id=db)
    return client, provider, order, inputs, box


@pytest.mark.asyncio
@pytest.mark.parametrize("followup", [
    "after that email me the list",
    "email me the list when it's done",
    "وقتی تموم شد لیستشو برام ایمیل کن",
])
async def test_a_sequenced_request_waits_and_runs_with_the_first_result(monkeypatch, followup):
    client, provider, order, inputs, box = await _sequenced(
        monkeypatch, followup, db=f"db-auth-a5-{abs(hash(followup))}",
    )
    assert box["d2_before_release"] == ["d1"], "the dependent ran before the work it follows"
    assert order == ["d1", "d2"]
    d2 = _frames_for(client, "d2")
    assert [f.get("phase") for f in d2][:2] == ["created", "pending"]
    assert d2[1].get("reason") == "waiting_for_task"
    relations = [f.get("relation") for f in _lifecycle(d2)]
    assert relations and all(r == {"kind": "continues", "task_id": "d1"} for r in relations)
    assert _phases(client, "d2")[-1] == "completed"
    assert "Accepted backend result: Professor Ada Lovelace, ada@toronto.example." in inputs["d2"]
    assert "Request this one follows (it has finished; its result is above)" in inputs["d2"]
    assert RESEARCH in inputs["d2"]


@pytest.mark.asyncio
async def test_a_sequenced_request_fails_honestly_when_its_task_fails(monkeypatch):
    client, provider, order, inputs, box = await _sequenced(
        monkeypatch, "after that email me the list", db="db-auth-a5-fail", first_fails=True,
    )
    assert order == ["d1"], "the dependent ran on nothing"
    d2 = _frames_for(client, "d2")
    assert d2[-1].get("phase") == "failed" and d2[-1].get("reason") == "dependency_failed", d2
    spoken = [e["content"] for e in provider.of("session.commentary.append") if e["delegation_id"] == "d2"]
    assert spoken == [
        "I didn't do the follow-up you asked for, because the task it was waiting on didn't finish.",
    ]


@pytest.mark.asyncio
async def test_a_sequenced_request_follows_the_correction_of_its_task(monkeypatch):
    """"…then email it" waits for d1; "no, only the downtown campus" replaces
    d1 with d3; the email waits for d3 and runs after it — not failed with d1."""

    client, provider, order, inputs, box = await _sequenced(
        monkeypatch, "then email me the list", db="db-auth-a5-follow",
        correction="no, only the downtown campus",
    )
    assert "superseded" in _phases(client, "d1")
    assert "d2" not in box["d2_after_correction"]
    assert order[-1] == "d2" and _phases(client, "d2")[-1] == "completed"
    relations = [f.get("relation") for f in _lifecycle(_frames_for(client, "d2"))]
    assert relations[-1] == {"kind": "continues", "task_id": "d3"}, relations


def test_sequencing_markers_are_anchored():
    from app.services.live_voice_protocol import sequencing_marker

    for yes in (
        "after that email it to me", "email it to me when it's done", "then email me the list",
        "بعدش ایمیلش کن", "وقتی تموم شد ایمیلش کن", "ایمیلش کن وقتی تموم شد",
    ):
        assert sequencing_marker(yes), yes
    for no in (
        "then what", "and then?", "what happened afterwards", "بعد از ظهر جلسه دارم",
        "after that", "is it done", "what time is it then",
    ):
        assert not sequencing_marker(no), no


def test_every_new_relay_line_and_label_is_bilingual_and_private():
    from app.services.live_voice_protocol import (
        delegated_agent_input, relay_line, sanitize_user_visible_answer,
    )

    assert relay_line("dependency_failed", "en") != relay_line("dependency_failed", "fa")
    assert relay_line("dependency_failed", "fa")
    for kind in ("running", "previous"):
        message = delegated_agent_input(
            "in the computer science department", [],
            context_request=ORIGINAL, context_kind=kind,
        )
        scaffold = message.split("Current caller request:")[0]
        for line in scaffold.splitlines():
            if line.strip() and not line.startswith("- "):
                assert sanitize_user_visible_answer(line) == "", line
    message = delegated_agent_input(
        "email me the list", [("find x", "X.")],
        related_request=ORIGINAL, relation_kind="continues",
    )
    label = next(line for line in message.splitlines() if line.startswith("Request this one follows"))
    assert sanitize_user_visible_answer(label) == ""
