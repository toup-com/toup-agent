"""Second fix round: classification authority (spec addendum 2, items 1-5).

Re-verification (`/private/tmp/toup-voice-r2-review/reverify.json`, groups
w1-span and w2a-classify) left these open or partial on the integrated relay:

  item 1  F1/F10 merge gap: a question ending in "?"/«؟» swept into a check-in's
          span while work ran (or a result was unheard) was consumed and
          answered "still working" — `begin()` looked for swept-in requests
          only when the anchor phrase did NOT read as status.  Addendum 3
          outside the span: a newer unanswered request never overtook a
          check-in while work ran, or with an unheard result (or an outcome
          that failed).  The dead `overtake_status` path is gone.
  item 2  `request_turn_ids` is in spoken order, the anchor wherever it falls.
  item 3  F7: "cancel the whole thing", "cancel it right away", «کلاً کنسلش
          کن», «اصلاً ولش کن» were dispatched as new requests (and a card),
          the research kept running.
  item 4  F8: "did you find anyone", «کسی رو پیدا کردی؟», "did you find the
          professors yet" spawned a duplicate agent run while work ran.
  item 5  F9: "no, what time is it in tokyo right now", «نه بابا، ساعت توکیو
          چنده؟» superseded the only running task with no target evidence.

Each test asserts the CORRECT behaviour and failed on the pre-fix snapshot
(`integrated-snap3`); tests marked CONTROL pin what must not change.  Driven
through the real relay (`test_live_harness`) with provider-shaped events; the
pure seams are called directly.
"""

from __future__ import annotations

import asyncio

import pytest

import test_live_harness as H


RESEARCH = "research the toronto robotics faculty"
PERSIAN_TASK = "یه استاد برای یادگیری ماشین در دانشگاه تورنتو پیدا کن"
PROFESSOR_TASK = "find a machine learning professor at the university of toronto"
LIBRARY = "what time does the library close"


def _frames_for(client, did):
    return [f for f in client.of("delegation") if f.get("delegation_id") == did]


def _phases(client, did):
    return [f.get("phase") for f in _frames_for(client, did)]


def _created(client, did):
    return [f for f in _frames_for(client, did) if f.get("phase") == "created"]


def _terminal(client, did):
    return any(
        f.get("phase") in {"completed", "failed", "cancelled", "superseded", "expired"}
        for f in _frames_for(client, did)
    )


def _commentary(provider):
    return [e["content"] for e in provider.of("session.commentary.append")]


def _instructions(provider):
    return [e["content"] for e in provider.of("session.instructions.append")]


def _status_lines(provider):
    return [
        c for c in _commentary(provider)
        if "Still working" in c or "هنوز دارم روش کار" in c
    ]


def _replays(provider, answer):
    return [
        c for c in _instructions(provider)
        if answer in c and ("asked what you found" in c or "پرسید چه چیزی پیدا کردی" in c)
    ]


def _turn_id(client, text) -> str:
    return next(
        f["turn_id"] for f in client.of("transcript") if f.get("final") and f["text"] == text
    )


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
# item 1 — a check-in never answers for the request the caller is waiting on
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize(("question", "check_in"), [
    ("Who is the dean of engineering?", "What happened?"),
    ("رئیس دانشکده مهندسی کیه؟", "چی شد؟"),
])
async def test_a_punctuated_question_swept_into_a_check_in_while_work_runs_is_dispatched(
        monkeypatch, question, check_in):
    """F1/F10: the question an ASR renders WITH its "?"/«؟» ends its own phrase,
    so the anchor phrase alone is the check-in.  The span still swept the
    question in, and it — not "still working" — is what runs: its words, its
    title, its turn; the check-in's turn follows it in `request_turn_ids`."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    displays: list[str] = []
    killed: list[str] = []
    started = asyncio.Event()
    box: dict = {}
    counters = H.capture_counters(monkeypatch)

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        displays.append(kw.get("display_request") or "")
        if kw.get("delegation_id") == "d1":
            started.set()
            try:
                await asyncio.sleep(1.8)
            except asyncio.CancelledError:
                killed.append("d1")
                raise
            return "Robotics faculty list.", "m"
        return "The dean is Professor X.", "m"

    H.patch_relay(monkeypatch, think=think)

    async def script():
        await started.wait()
        p = box["p"]
        p.push(H.user_delta(question, 3000, 3900))
        await asyncio.sleep(0.4)
        p.push(H.user_delta(check_in, 7000, 7500))
        p.push(H.delegation("d2", 7550))
        await _wait(lambda: _terminal(client, "d1") and _terminal(client, "d2"), 5.0)

    client = H.FakeClient([H.config(), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_opener(box, RESEARCH), auto_ack=True)
    await H.run_relay(
        client, provider, timeout=12, db_session_id=f"db-r2c-f1-{abs(hash(question))}",
    )

    assert displays == [RESEARCH, question], displays
    assert killed == [] and _phases(client, "d1")[-1] == "completed"
    assert not _status_lines(provider), _commentary(provider)
    created = _created(client, "d2")
    assert [c["title"] for c in created] == [question], created
    q_turn, s_turn = _turn_id(client, question), _turn_id(client, check_in)
    assert created[0]["turn_id"] == q_turn
    assert created[0]["request_turn_ids"] == [q_turn, s_turn]
    assert ("live_status_dispatched", {"reason": "request_in_span"}) in counters
    assert not any(name == "live_status_answered" for name, _f in counters)


@pytest.mark.asyncio
async def test_a_newer_unanswered_request_outside_the_span_overtakes_a_check_in_on_running_work(
        monkeypatch):
    """Addendum 3 / addendum 2 item 1, running branch: "what time does the
    library close" (asked while the research runs, never answered) and, 11 s
    of input later, "what did you find so far".  The library question runs as
    itself; the research is left alone; nothing says "still working"."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    displays: list[str] = []
    started = asyncio.Event()
    box: dict = {}
    counters = H.capture_counters(monkeypatch)

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        displays.append(kw.get("display_request") or "")
        if len(displays) == 1:
            started.set()
            await asyncio.sleep(2.0)
            return "Robotics list.", "m"
        return "The library closes at 9.", "m"

    H.patch_relay(monkeypatch, think=think)

    async def script():
        await started.wait()
        p = box["p"]
        p.push(H.user_delta(LIBRARY, 3000, 3800))
        await asyncio.sleep(0.4)
        p.push(H.user_delta("what did you find so far", 14000, 14600))
        p.push(H.delegation("d2", 14650))
        await _wait(lambda: _terminal(client, "d1") and _terminal(client, "d2"), 5.0)

    client = H.FakeClient([H.config(), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_opener(box, RESEARCH), auto_ack=True)
    await H.run_relay(client, provider, timeout=12, db_session_id="db-r2c-add3-running")

    assert displays == [RESEARCH, LIBRARY], displays
    assert _phases(client, "d1")[-1] == "completed"
    assert not _status_lines(provider), _commentary(provider)
    created = _created(client, "d2")
    assert [c["title"] for c in created] == [LIBRARY]
    lib_turn, s_turn = _turn_id(client, LIBRARY), _turn_id(client, "what did you find so far")
    assert created[0]["turn_id"] == lib_turn
    assert created[0]["request_turn_ids"] == [lib_turn, s_turn]
    assert ("live_status_dispatched", {"reason": "newer_turn"}) in counters


@pytest.mark.asyncio
async def test_a_newer_unanswered_request_outside_the_span_overtakes_an_unheard_result(monkeypatch):
    """Addendum 2 item 1, unheard-result branch: the result was never SAID, the
    caller asked something newer that nothing answered, and 13 s later "what
    happened".  The newer question runs; the old result is not re-read over it."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    displays: list[str] = []
    box: dict = {}
    first_answer = "Professor Ada Lovelace and Professor Alan Turing work on robotics."
    question = "are any of them working on llms"

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        displays.append(kw.get("display_request") or "")
        return (first_answer if len(displays) == 1 else "Yes, Professor Turing."), "m"

    H.patch_relay(monkeypatch, think=think)

    async def script():
        # No speech for the commentary: nudge, then an honest spoken:false.
        await _wait(lambda: any(f.get("spoken") is False for f in _frames_for(client, "d1")))
        p = box["p"]
        p.push(H.user_delta(question, 6000, 7000))
        await asyncio.sleep(0.5)
        p.push(H.user_delta("what happened", 20000, 20500))
        p.push(H.delegation("d2", 20550))
        await _wait(lambda: _terminal(client, "d2") or bool(_replays(provider, first_answer)), 4.0)
        await asyncio.sleep(0.2)

    client = H.FakeClient([H.config(), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=_opener(box, "find robotics professors at toronto"), auto_ack=True,
    )
    await H.run_relay(client, provider, timeout=12, db_session_id="db-r2c-add3-unheard")

    assert displays == ["find robotics professors at toronto", question], displays
    assert not _replays(provider, first_answer), "the unheard result was re-read over the question"
    created = _created(client, "d2")
    assert created and created[0]["title"] == question
    assert created[0]["turn_id"] == _turn_id(client, question)


@pytest.mark.asyncio
async def test_a_newer_unanswered_request_overtakes_a_check_in_after_a_failed_task(monkeypatch):
    """One rule, every branch: a failed outcome on file is no reason to leave the
    caller's newer question unanswered behind "that didn't go through"."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    displays: list[str] = []
    box: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        displays.append(kw.get("display_request") or "")
        if len(displays) == 1:
            raise RuntimeError("agent container gone")
        return "The library closes at 9.", "m"

    H.patch_relay(monkeypatch, think=think)

    async def script():
        await _wait(lambda: _terminal(client, "d1"))
        await asyncio.sleep(0.4)
        p = box["p"]
        p.push(H.user_delta(LIBRARY, 6000, 6800))
        await asyncio.sleep(0.5)
        p.push(H.user_delta("what happened", 20000, 20500))
        p.push(H.delegation("d2", 20550))
        await _wait(lambda: _terminal(client, "d2"), 3.0)
        await asyncio.sleep(0.2)

    client = H.FakeClient([H.config(), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_opener(box, RESEARCH), auto_ack=True)
    await H.run_relay(client, provider, timeout=12, db_session_id="db-r2c-add3-failed")

    assert _phases(client, "d1")[-1] == "failed"
    assert displays == [RESEARCH, LIBRARY], displays
    created = _created(client, "d2")
    assert created and created[0]["title"] == LIBRARY
    assert created[0]["turn_id"] == _turn_id(client, LIBRARY)


@pytest.mark.asyncio
@pytest.mark.parametrize("between", ["okay thanks", "how are you doing today"])
async def test_control_a_check_in_on_running_work_is_still_answered_when_nothing_newer_waits(
        monkeypatch, between):
    """CONTROL: a content-free backchannel is not a request, and a question the
    model ANSWERED out loud is not unanswered — neither overtakes the check-in,
    which is answered from state with no card and no second run."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    displays: list[str] = []
    started = asyncio.Event()
    box: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        displays.append(kw.get("display_request") or "")
        started.set()
        await asyncio.sleep(2.0)
        return "Robotics list.", "m"

    H.patch_relay(monkeypatch, think=think)

    async def script():
        await started.wait()
        p = box["p"]
        p.push(H.user_delta(between, 3000, 3800))
        await asyncio.sleep(0.3)
        if between.startswith("how"):
            p.push(H.out_text("I'm doing well, thanks for asking.", 4000, 4600))
            p.push(H.out_audio("X"))
        await asyncio.sleep(0.4)
        p.push(H.user_delta("what did you find so far", 14000, 14600))
        p.push(H.delegation("d2", 14650))
        await _wait(lambda: bool(_status_lines(provider)), 3.0)
        await asyncio.sleep(0.2)

    client = H.FakeClient([H.config(), script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_opener(box, RESEARCH), auto_ack=True)
    await H.run_relay(
        client, provider, timeout=12, db_session_id=f"db-r2c-add3-ctl-{len(between)}",
    )

    assert displays == [RESEARCH], displays
    assert _frames_for(client, "d2") == []
    assert _status_lines(provider)


def test_the_dead_overtake_status_path_is_gone():
    """`begin()` decides through `status_overtaken_by` → `rebind_to_requests`;
    the second, never-called path could only drift from it."""

    from app.services.live_voice_protocol import _LiveSession

    assert not hasattr(_LiveSession, "overtake_status")


# ══════════════════════════════════════════════════════════════════════
# item 3 — a cancel that only says how or when is a cancel (F7)
# ══════════════════════════════════════════════════════════════════════

CANCEL_WITH_MODIFIER = [
    "cancel the whole thing", "cancel it right away", "cancel that immediately",
    "stop it completely", "forget it altogether",
    "کلاً کنسلش کن", "کلا لغو کن", "بی‌خیال شو", "اصلا ولش کن", "سریع لغو کن",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("followup", CANCEL_WITH_MODIFIER)
async def test_an_explicit_cancel_with_an_intensifier_cancels_the_running_task(monkeypatch, followup):
    client, provider, inputs, displays, killed = await _while_running(
        monkeypatch, followup, db=f"db-r2c-f7-{abs(hash(followup))}",
    )
    assert killed == ["d1"], killed
    assert _phases(client, "d1")[-1] == "cancelled", _phases(client, "d1")
    assert "d2" not in inputs, "the cancel phrase was dispatched as a request"
    assert _frames_for(client, "d2") == [], "a card for the cancel command itself"


@pytest.mark.asyncio
@pytest.mark.parametrize("followup", ["بی‌خیال شو", "اصلاً ولش کن", "forget it altogether"])
async def test_a_dismissal_with_an_intensifier_and_nothing_running_is_no_job(monkeypatch, followup):
    """F12's twin: with nothing in flight, «بی‌خیال شو» is a dismissal — an
    honest line, no card, no agent run."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    thinks: list[str] = []

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        thinks.append(task)
        return "An answer.", "m"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}
    client = H.FakeClient([H.config(), 1.2, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_opener(box, followup), auto_ack=True)
    await H.run_relay(
        client, provider, timeout=10, db_session_id=f"db-r2c-f7-idle-{abs(hash(followup))}",
    )

    assert thinks == []
    assert _frames_for(client, "d1") == []
    assert [e for e in provider.of("session.commentary.append") if e.get("delegation_id") == "d1"]


def test_a_cancel_residue_ignores_intensifiers_and_adverbs():
    from app.services.live_voice_protocol import cancel_residue, classify_followup, is_idle_dismissal

    for text in CANCEL_WITH_MODIFIER + ["همین الان کنسلش کن", "cancel all of it"]:
        assert classify_followup(text, RESEARCH) == "cancel", text
        assert cancel_residue(text) == set(), (text, cancel_residue(text))
    # CONTROL: an object still names something (F7).
    assert {"meeting", "john"} <= cancel_residue("cancel my three pm meeting with john right away")
    assert {"جلسه", "علی"} <= cancel_residue("جلسه فردا با علی رو همین الان لغو کن")
    assert cancel_residue("cancel the robotics research immediately") == {"robotics"}
    assert is_idle_dismissal("بی‌خیال شو") and is_idle_dismissal("اصلاً ولش کن")
    assert not is_idle_dismissal("cancel it right away")   # an undo only the agent can do


# ══════════════════════════════════════════════════════════════════════
# item 4 — a natural check-in on running work is a check-in (F8, R12.3)
# ══════════════════════════════════════════════════════════════════════

CHECK_INS = [
    ("did you find anyone", RESEARCH),
    ("did you find someone", RESEARCH),
    ("did you find the professors yet", RESEARCH),
    ("any luck with the professors?", RESEARCH),
    ("did you find the professor", PROFESSOR_TASK),
    ("کسی رو پیدا کردی؟", PERSIAN_TASK),
    ("چی شد؟ کسی پیدا شد؟", PERSIAN_TASK),
    ("هنوز داری دنبالش میگردی؟", PERSIAN_TASK),
    ("استاده رو پیدا کردی؟", PERSIAN_TASK),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("check_in", "running"), CHECK_INS)
async def test_a_natural_check_in_while_work_runs_is_answered_from_state(monkeypatch, check_in, running):
    client, provider, inputs, displays, killed = await _while_running(
        monkeypatch, check_in, running=running, db=f"db-r2c-f8-{abs(hash(check_in))}",
    )
    assert "d2" not in inputs, f"a duplicate agent run for {check_in!r}"
    assert _frames_for(client, "d2") == []
    assert killed == [] and _phases(client, "d1")[-1] == "completed"
    assert _status_lines(provider), _commentary(provider)


@pytest.mark.asyncio
async def test_control_a_find_question_with_a_subject_of_its_own_is_dispatched(monkeypatch):
    """CONTROL: two words of its own are a request, not a check-in."""

    followup = "did you find out when the library closes"
    client, provider, inputs, displays, killed = await _while_running(
        monkeypatch, followup, db="db-r2c-f8-ctl",
    )
    assert displays.get("d2") == followup
    assert not _status_lines(provider)
    assert killed == []


def test_a_check_in_is_read_on_its_own_words_and_the_work_it_asks_about():
    from app.services.live_voice_protocol import is_status_question

    for text, subject in CHECK_INS:
        assert is_status_question(text, subject), (text, subject)
    # CONTROL (F8): a marker inside a request of its own is still a request.
    for text, subject in [
        ("what happened in the raptors game last night", RESEARCH),
        ("هنوز بارون میاد تو تورنتو؟", PERSIAN_TASK),
        ("what's the status of my amazon order", RESEARCH),
        ("what happened with my order", RESEARCH),
        ("did you find out when the library closes", RESEARCH),
        ("کارشون ال‌ال‌امم چی شد میگم", "find robotics professors at toronto"),
    ]:
        assert not is_status_question(text, subject), (text, subject)
    # The object rule needs work to ask about.
    assert not is_status_question("did you find the professors yet", "")


# ══════════════════════════════════════════════════════════════════════
# item 5 — a negating opener supersedes only with target evidence (F9)
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("followup", [
    "no, what time is it in tokyo right now",
    "نه بابا، ساعت توکیو چنده؟",
    "not now, set a timer for ten minutes",
])
async def test_a_negating_opener_before_an_unrelated_request_never_supersedes(monkeypatch, followup):
    client, provider, inputs, displays, killed = await _while_running(
        monkeypatch, followup, db=f"db-r2c-f9-{abs(hash(followup))}",
    )
    assert killed == [], "healthy running work was destroyed by an opener"
    assert not {"superseded", "cancelled"} & set(_phases(client, "d1"))
    assert _phases(client, "d1")[-1] == "completed"
    assert all("relation" not in f for f in _frames_for(client, "d2"))
    assert displays.get("d2") == followup
    second = inputs.get("d2", "")
    assert "non-binding" in second and RESEARCH in second, second
    assert "Request the caller is correcting" not in second


def test_negation_needs_evidence_of_what_it_modifies():
    from app.services.live_voice_protocol import _content_words, negation_targets

    original = "find iranian computer science professors at the university of toronto"
    evidenced = [
        ("no, only the downtown campus", original),                       # restriction
        ("no no, only the downtown campus", original),
        ("نه، فقط پردیس مرکزی", original),
        ("no, book thursday with the dentist", "book friday with the dentist"),  # shared words
        ("نه، کجایی هستش استاده؟ کدوم شهر دنیا", "اوکی، اه کجای استاد"),   # V2
        ("no, thursday", "book friday with the dentist"),                 # a bare fragment
        ("i meant the downtown campus", original),                        # says it modifies
        ("instead only associate professors", original),
    ]
    for text, work in evidenced:
        assert negation_targets(text, _content_words(work)), text
    for text in (
        "no, what time is it in tokyo right now", "نه بابا، ساعت توکیو چنده؟",
        "not now, set a timer for ten minutes", "no, call mom", "no worries, what time is it",
        "no, no, what time is it",
    ):
        assert not negation_targets(text, _content_words(RESEARCH)), text
