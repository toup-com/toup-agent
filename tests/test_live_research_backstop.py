"""D-P3 (2026-10-05 call): an explicit search request is always work.

The recording asked four times — «…برام یه چیزی رو سرچ کن ببین…», «خب سرچ کن
برام», «سرچ کن بر», «چرا سرچ نمی کنی». GPT-Live answered the book question from
memory, said «من چیزی رو سرچ نکردم», and never delegated: nothing searched and
no card appeared. The relay now reads a closed search grammar on every closed
caller turn, and when no model delegation claims such a turn within
`voice_live_research_backstop_grace_ms` it opens the web-research task itself
(`relay-research:<hex>`, the card's Retry recipe): the delegated body carries
`require_web`, an object-less ask searches the request before it, the model is
told once by an INSTRUCTION (GPT-Live refuses commentary for an id it did not
mint), and a late provider delegation for the same words is absorbed.

Everything drives the real relay through `test_live_harness`; the tenant is
faked only at `_think`.
"""

import asyncio

import pytest

import test_live_harness as H
from app.config import settings
from app.services import live_voice_protocol as live
from app.services.live_voice_protocol import (
    RELAY_RESEARCH_PREFIX,
    relay_line,
    relay_opened_delegation,
    research_request_evidence,
)


FEATURES = H.config()["features"]
BOOK = "بهترین کتابی که میشه یه آدم بخونه برای بیزینس خودش رو راه بندازه چیه؟"
GUESS = "به‌نظرم The Lean Startup از اریک ریس."
SEARCH_IT = "خب سرچ کن برام"
WHY_NOT = "چرا سرچ نمی کنی"
NAMED = "سرچ کن ببین بهترین کتاب برای جذب سرمایه چیه"


def _on(monkeypatch, grace_ms: int = 250, **overrides) -> None:
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0, **overrides)
    monkeypatch.setattr(settings, "voice_live_research_backstop_enabled", True)
    monkeypatch.setattr(settings, "voice_live_research_backstop_grace_ms", grace_ms)


def _research_ids(client) -> list[str]:
    seen: list[str] = []
    for frame in client.of("delegation"):
        did = str(frame.get("delegation_id") or "")
        if did.startswith(RELAY_RESEARCH_PREFIX) and did not in seen:
            seen.append(did)
    return seen


def _instructions(provider) -> list[str]:
    return [str(e.get("content") or "") for e in provider.of("session.instructions.append")]


async def _wait(pred, timeout=5.0):
    for _ in range(int(timeout / 0.02)):
        if pred():
            return True
        await asyncio.sleep(0.02)
    return False


def _boot(box):
    def on_send(p, e):
        box["p"] = p
        if e["type"] == "session.close":
            p.push(H.closed())
    return on_send


def test_the_backstop_is_on_by_default_with_a_short_grace():
    assert settings.voice_live_research_backstop_enabled is True
    assert settings.voice_live_research_backstop_grace_ms == 800


@pytest.mark.parametrize("text, verdict", [
    # The recording.
    ("برام یه چیزی رو سرچ کن ببین " + BOOK, ("search", True)),
    (SEARCH_IT, ("search", False)),
    ("سرچ کن بر", ("search", False)),
    (WHY_NOT, ("complaint", False)),
    ("چرا سرچ نمی‌کنی", ("complaint", False)),
    ("چرا سرچ نمیکنی", ("complaint", False)),
    ("سرچ می‌کنی برام؟", ("search", False)),
    ("گوگلش کن", ("search", False)),
    ("جست و جو کن قیمت طلا", ("search", True)),
    ("تو اینترنت ببین قیمت دلار چنده", ("search", True)),
    ("بگرد ببین کی برنده شد", ("search", True)),
    ("چرا سرچ نکردی", ("complaint", False)),
    ("چرا نمی‌گردی", ("complaint", False)),
    ("می‌گردی ببینی کی برنده شد؟", ("search", True)),
    ("search for the best business book", ("search", True)),
    ("can you look it up", ("search", False)),
    ("google it", ("search", False)),
    ("find out who won the game", ("search", True)),
    ("check online for flights", ("search", True)),
    ("why didn't you search", ("complaint", False)),
    ("why don't you look it up", ("complaint", False)),
    # Never: provenance, refusal, first person, status, cancel, retry, media,
    # a place that is not the web, and ordinary talk.
    ("اینو از کجا سرچ کردی", (None, False)),
    ("سرچ نکردی", (None, False)),
    ("سرچ کردم", (None, False)),
    ("سرچ نکن", (None, False)),
    ("نگرد", (None, False)),
    ("so you found out who won?", (None, False)),
    ("چی شد", (None, False)),
    ("دوباره سرچ کن", (None, False)),
    ("جستجو رو متوقف کن", (None, False)),
    ("ایمیلامو سرچ کن", (None, False)),
    ("where did you search", (None, False)),
    ("did you search", (None, False)),
    ("don't search", (None, False)),
    ("I'll search later", (None, False)),
    ("stop the search", (None, False)),
    ("what are you searching for", (None, False)),
    # RO-5: the Persian present progressive and a question word are a status
    # or provenance check-in, exactly like "are you searching" above.
    ("الان داری سرچ میکنی؟", (None, False)),
    ("داری سرچ می‌کنی؟", (None, False)),
    ("کجا رو سرچ میکنی", (None, False)),
    ("داری چی رو سرچ میکنی؟", (None, False)),
    ("داری می‌گردی؟", (None, False)),
    ("are you searching", (None, False)),
    ("search my email for the invoice", (None, False)),
    ("search for a song by Ebi and play it", (None, False)),
    ("سلام، چطوری", (None, False)),
    ("what is the capital of France", (None, False)),
])
def test_the_search_grammar(text, verdict):
    assert research_request_evidence(text) == verdict


def test_a_research_task_is_relay_opened():
    assert relay_opened_delegation(f"{RELAY_RESEARCH_PREFIX}abc")
    assert not relay_opened_delegation("d1")


async def _call(monkeypatch, *, ask, model_delegates_at=None, think_delay=0.0,
                book_delegated=False, db):
    """The book question (answered from memory, or delegated as `d-book`),
    then `ask`; the model replies with text and — unless `model_delegates_at`
    — never delegates `ask`."""

    calls: list[dict] = []
    counters = H.capture_counters(monkeypatch)
    started = asyncio.Event()

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append({"did": kw.get("delegation_id"), "task": task, **kw})
        started.set()
        if think_delay:
            await asyncio.sleep(think_delay)
        return "منبع: goodreads.com — The Lean Startup.", "m"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}

    async def script():
        p = box["p"]
        p.push(H.user_delta(BOOK, 1000, 2500))
        if book_delegated:
            p.push(H.delegation("d-book", 2550))
            await _wait(lambda: any(c["did"] == "d-book" for c in calls))
        else:
            await asyncio.sleep(0.3)
            p.push(H.out_text(GUESS, 2700, 3300))
            p.push(H.out_audio("GUESS"))
        await asyncio.sleep(0.4)
        p.push(H.user_delta(ask, 6000, 6800))
        if model_delegates_at is not None:
            p.push(H.delegation("d-model", model_delegates_at))
        await asyncio.sleep(0.2)
        p.push(H.out_text("باشه.", 7000, 7300))
        p.push(H.out_audio("OK"))
        await asyncio.sleep(1.2)

    client = H.FakeClient([H.config(features=FEATURES), 0.1, script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_boot(box), auto_ack=True)
    await H.run_relay(client, provider, timeout=15, db_session_id=db)
    return client, provider, calls, counters


@pytest.mark.asyncio
async def test_an_objectless_search_ask_searches_the_request_before_it(monkeypatch):
    """«خب سرچ کن برام» after the book question was answered from memory: the
    relay opens ONE web-research task for the BOOK question — the card is
    titled from it, the agent gets the caller's words with the book question
    as the request they refer back to, and the body requires the web."""

    _on(monkeypatch)
    client, provider, calls, counters = await _call(
        monkeypatch, ask=SEARCH_IT, db="db-research-objectless",
    )

    (rid,) = _research_ids(client)
    assert [c["did"] for c in calls] == [rid], calls
    call = calls[0]
    assert call.get("require_web") is True
    assert call["display_request"] == live.display_request_text(BOOK)
    assert BOOK in call["task"] and SEARCH_IT in call["task"]
    assert "refer back to" in call["task"]
    frames = [f for f in client.of("delegation") if f.get("delegation_id") == rid]
    assert frames[0]["phase"] == "created" and frames[-1]["phase"] == "completed", frames
    assert {f.get("title") for f in frames if f.get("title")} == {live.request_title(BOOK)}
    # Told ONCE, by an instruction quoting what is searched — never commentary
    # naming an id GPT-Live did not mint.
    started = relay_line("research_started", "fa", request=live.display_request_text(BOOK))
    assert _instructions(provider).count(started) == 1, _instructions(provider)
    assert all(e.get("delegation_id") != rid for e in provider.sent)
    assert ("live_research_backstop", {"kind": "search"}) in counters


@pytest.mark.asyncio
async def test_a_named_search_request_carries_its_own_words(monkeypatch):
    _on(monkeypatch)
    client, provider, calls, _counters = await _call(
        monkeypatch, ask=NAMED, db="db-research-named",
    )

    (rid,) = _research_ids(client)
    assert [c["did"] for c in calls] == [rid], calls
    assert calls[0].get("require_web") is True
    assert calls[0]["display_request"] == live.display_request_text(NAMED)
    assert "refer back to" not in calls[0]["task"]
    assert BOOK not in calls[0]["task"]


@pytest.mark.asyncio
async def test_a_delegation_in_time_is_the_path_and_the_backstop_stays_quiet(monkeypatch):
    _on(monkeypatch)
    client, provider, calls, counters = await _call(
        monkeypatch, ask=NAMED, model_delegates_at=6850, db="db-research-in-time",
    )

    assert _research_ids(client) == []
    assert [c["did"] for c in calls] == ["d-model"], calls
    # Voice round 2026-10-07 (D-R4 (1)) REVERSES "only the backstop sets
    # require_web": the model's own delegation of an explicit search ask is a
    # search too (`decide_web` source `evidence`) — the backstop still stays
    # quiet.
    assert calls[0].get("require_web") is True, calls[0]
    assert not any("live_research_backstop" == name for name, _f in counters)


@pytest.mark.asyncio
async def test_a_late_model_delegation_for_the_same_words_is_absorbed(monkeypatch):
    """The model catches up after the relay fired: its delegation is retired,
    never run (a second search) and never rebound to the older book question;
    its function call is completed under ITS id with what the app is doing."""

    _on(monkeypatch)
    calls: list[dict] = []
    counters = H.capture_counters(monkeypatch)

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append({"did": kw.get("delegation_id"), **kw})
        await asyncio.sleep(0.5)
        return "منبع: goodreads.com", "m"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}

    async def script():
        p = box["p"]
        p.push(H.user_delta(BOOK + ".", 1000, 2500))
        await asyncio.sleep(0.3)
        p.push(H.out_text(GUESS, 2700, 3300))
        p.push(H.out_audio("GUESS"))
        await asyncio.sleep(0.4)
        p.push(H.user_delta(NAMED, 6000, 6800))
        await _wait(lambda: _research_ids(client))
        p.push(H.delegation("d-late", 6900))
        await asyncio.sleep(1.2)

    client = H.FakeClient([H.config(features=FEATURES), 0.1, script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_boot(box), auto_ack=True)
    await H.run_relay(client, provider, timeout=15, db_session_id="db-research-absorbed")

    (rid,) = _research_ids(client)
    assert [c["did"] for c in calls] == [rid], calls
    assert ("live_delegation_absorbed", {"reason": "relay_research"}) in counters
    notes = [
        e for e in provider.of("session.thinking.append") if e.get("delegation_id") == "d-late"
    ]
    assert notes and "already being searched" in notes[0]["content"], notes
    assert not [f for f in client.of("delegation") if f.get("delegation_id") == "d-late"]


@pytest.mark.asyncio
@pytest.mark.parametrize("ask", ["اینو از کجا سرچ کردی", "چی شد", "الان داری سرچ میکنی؟"])
async def test_a_provenance_question_or_check_in_never_searches(monkeypatch, ask):
    _on(monkeypatch)
    client, _provider, calls, _counters = await _call(
        monkeypatch, ask=ask, db=f"db-research-never-{ask.encode().hex()[:24]}",
    )

    assert _research_ids(client) == []
    assert calls == []


@pytest.mark.asyncio
async def test_a_complaint_while_that_search_runs_joins_it(monkeypatch):
    """The book question was delegated and is still running when the caller
    says «چرا سرچ نمی کنی»: the work it asks for is in flight, so nothing new
    starts."""

    _on(monkeypatch)
    client, _provider, calls, counters = await _call(
        monkeypatch, ask=WHY_NOT, book_delegated=True, think_delay=2.5,
        db="db-research-joins",
    )

    assert _research_ids(client) == []
    assert [c["did"] for c in calls] == ["d-book"], calls
    assert ("live_research_backstop_skipped", {"reason": "already_running"}) in counters


@pytest.mark.asyncio
async def test_repeated_asks_join_the_running_search_and_never_become_its_subject(monkeypatch):
    """The recording's 3:07–3:38 shape: «خب سرچ کن برام» starts the search of
    the book question; «چرا سرچ نمی کنی» while it runs joins it (nothing new
    starts); «سرچ کن بر» after it finished searches the BOOK question again —
    never one of the earlier object-less asks, which name no subject."""

    _on(monkeypatch)
    calls: list[dict] = []
    counters = H.capture_counters(monkeypatch)

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append({"did": kw.get("delegation_id"), **kw})
        await asyncio.sleep(1.5 if len(calls) == 1 else 0.1)
        return "منبع: goodreads.com", "m"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}

    def done(rid):
        return any(
            f.get("delegation_id") == rid and f.get("phase") == "completed"
            for f in client.of("delegation")
        )

    async def script():
        p = box["p"]
        p.push(H.user_delta(BOOK, 1000, 2500))
        await asyncio.sleep(0.3)
        p.push(H.out_text(GUESS, 2700, 3300))
        p.push(H.out_audio("GUESS"))
        await asyncio.sleep(0.4)
        p.push(H.user_delta(SEARCH_IT, 6000, 6800))
        await _wait(lambda: calls)
        await asyncio.sleep(0.2)
        p.push(H.user_delta(WHY_NOT, 9000, 9800))
        await asyncio.sleep(0.8)
        await _wait(lambda: done(calls[0]["did"]))
        await asyncio.sleep(0.3)
        p.push(H.user_delta("سرچ کن بر", 20000, 20600))
        await _wait(lambda: len(calls) == 2)
        await asyncio.sleep(0.5)

    client = H.FakeClient([H.config(features=FEATURES), 0.1, script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_boot(box), auto_ack=True)
    await H.run_relay(client, provider, timeout=20, db_session_id="db-research-repeat")

    assert len(_research_ids(client)) == 2, _research_ids(client)
    assert [c["display_request"] for c in calls] == [live.display_request_text(BOOK)] * 2
    assert ("live_research_backstop_skipped", {"reason": "already_running"}) in counters


@pytest.mark.asyncio
async def test_switched_off_it_never_fires(monkeypatch):
    _on(monkeypatch)
    monkeypatch.setattr(settings, "voice_live_research_backstop_enabled", False)
    client, _provider, calls, _counters = await _call(
        monkeypatch, ask=SEARCH_IT, db="db-research-off",
    )

    assert _research_ids(client) == []
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("require_web", [True, False])
async def test_the_agent_turn_body_carries_require_web_only_when_asked(monkeypatch, require_web):
    """The tenant forces its web tools on `require_web` (D-P3, tenant half).
    Only a relay-research task sets it; every other body is byte-identical."""

    from app.api import ws_realtime as rt

    bodies: list[tuple[str, dict]] = []

    async def vps(_user_id):
        return ("http://tenant.invalid", "key")

    async def stream(_url, _key, path, json_body, relay, cause=None):
        bodies.append((path, dict(json_body)))
        return "stream", {"text": "Found it on goodreads.com."}, 1

    class Relay:
        async def close_open(self):
            return None

    monkeypatch.setattr(rt, "_get_vps_info", vps)
    monkeypatch.setattr(rt, "_vps_api_stream", stream)
    monkeypatch.setattr(rt, "_agent_runner", None)
    monkeypatch.setattr(settings, "voice_realtime_v2", True)
    monkeypatch.setattr(settings, "voice_realtime_tool_events", True)
    extra = {"require_web": True} if require_web else {}
    text, _model = await rt._think(
        "user-1", "find the best book", "db-session", relay=Relay(),
        delegation_id=f"{RELAY_RESEARCH_PREFIX}x", **extra,
    )

    assert text == "Found it on goodreads.com."
    ((path, body),) = bodies
    assert path == "/api/v1/internal/agent-turn/stream"
    assert body.get("require_web") is (True if require_web else None)


# ── voice round 2026-10-06 (D-R1 (d)/(f)): the REPLY backstop, its sibling ──
#
# A request the model never answers — no reply, no delegation, an accepted
# reask that produced nothing (the 08:38 call's «فورمولوان», 10 s of silence
# while input moved) — is handed to the caller's agent by the relay itself:
# `relay-reply:<hex>`, the research recipe without the search.


def test_a_reply_backstop_task_is_relay_opened():
    assert relay_opened_delegation(f"{live.RELAY_REPLY_PREFIX}abc")
    assert relay_opened_delegation(f"{live.RELAY_PLAY_PREFIX}abc")
    # Voice round 2026-10-07 (D-R4 (4)): the relay's own now-playing answer is
    # the fifth relay-opened kind.
    assert set(live.RELAY_OPENED_PREFIXES) == {
        live.RELAY_RETRY_PREFIX, RELAY_RESEARCH_PREFIX, live.RELAY_REPLY_PREFIX,
        live.RELAY_PLAY_PREFIX, live.RELAY_STATUS_PREFIX,
    }
    assert not relay_opened_delegation("relay-replyx")


def _reply_on(monkeypatch, *, regen=0.3, research=False, **overrides) -> None:
    _on(monkeypatch, **overrides)
    monkeypatch.setattr(settings, "voice_live_research_backstop_enabled", research)
    monkeypatch.setattr(settings, "voice_live_reply_regen_after_s", regen)
    monkeypatch.setattr(settings, "voice_live_reply_backstop_enabled", True)
    monkeypatch.setattr(settings, "voice_live_uplink_keeper_enabled", True)


def _reply_ids(client) -> list[str]:
    seen: list[str] = []
    for frame in client.of("delegation"):
        did = str(frame.get("delegation_id") or "")
        if did.startswith(live.RELAY_REPLY_PREFIX) and did not in seen:
            seen.append(did)
    return seen


async def _silent_call(monkeypatch, *, ask, db, late_delegation=False, after=1.6):
    """`ask`, then a model that never answers it (and, with
    `late_delegation`, delegates it only after the relay already did)."""

    calls: list[dict] = []
    counters = H.capture_counters(monkeypatch)

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append({"did": kw.get("delegation_id"), "task": task, **kw})
        await asyncio.sleep(0.4)
        return "Robarts closes at 11 pm.", "m"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}

    async def script():
        p = box["p"]
        p.push(H.user_delta(ask, 1000, 2500))
        if late_delegation:
            await _wait(lambda: _reply_ids(client))
            p.push(H.delegation("d-late", 2600))
        await asyncio.sleep(after)

    client = H.FakeClient([H.config(features=FEATURES), 0.1, script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_boot(box), auto_ack=True)
    await H.run_relay(client, provider, timeout=15, db_session_id=db)
    return client, provider, calls, counters


ROBARTS = "when does robarts library close tonight"


@pytest.mark.asyncio
async def test_a_request_the_model_never_answers_is_handed_to_the_agent_by_the_relay(monkeypatch):
    _reply_on(monkeypatch)
    client, provider, calls, counters = await _silent_call(
        monkeypatch, ask=ROBARTS, db="db-reply-backstop",
    )

    (rid,) = _reply_ids(client)
    assert [c["did"] for c in calls] == [rid], calls
    assert ROBARTS in calls[0]["task"]
    assert "require_web" not in calls[0]
    frames = [f for f in client.of("delegation") if f.get("delegation_id") == rid]
    assert frames[0]["phase"] == "created" and frames[-1]["phase"] == "completed", frames
    started = relay_line("reply_started", "en", request=live.display_request_text(ROBARTS))
    assert _instructions(provider).count(started) == 1, _instructions(provider)
    assert all(e.get("delegation_id") != rid for e in provider.sent)
    assert ("live_reply_backstop", {}) in counters
    # ONE reask came first, and only one.
    reasks = [i for i in provider.of("session.instructions.append")
              if str(i.get("event_id") or "").startswith("toup-live-reask-")]
    assert len(reasks) == 1, reasks


@pytest.mark.asyncio
async def test_a_late_model_delegation_for_the_backstopped_words_is_absorbed(monkeypatch):
    """The model catches up after the relay handed the request on: its
    delegation is retired, never run (the agent twice) and never rebound to
    an older turn; its function call is completed under ITS id."""

    _reply_on(monkeypatch)
    client, provider, calls, counters = await _silent_call(
        monkeypatch, ask=ROBARTS, db="db-reply-absorbed", late_delegation=True, after=1.2,
    )

    (rid,) = _reply_ids(client)
    assert [c["did"] for c in calls] == [rid], calls
    assert ("live_delegation_absorbed", {"reason": "relay_reply"}) in counters
    notes = [
        e for e in provider.of("session.thinking.append") if e.get("delegation_id") == "d-late"
    ]
    assert notes and "already handed to their agent" in notes[0]["content"], notes
    assert not [f for f in client.of("delegation") if f.get("delegation_id") == "d-late"]


@pytest.mark.asyncio
async def test_the_reply_backstop_switched_off_never_fires(monkeypatch):
    _reply_on(monkeypatch)
    monkeypatch.setattr(settings, "voice_live_reply_backstop_enabled", False)
    client, _provider, calls, _counters = await _silent_call(
        monkeypatch, ask=ROBARTS, db="db-reply-off",
    )

    assert _reply_ids(client) == []
    assert calls == []


@pytest.mark.asyncio
async def test_a_search_request_is_the_research_backstops_not_the_reply_backstops(monkeypatch):
    """An explicit search ask the model ignores: the RESEARCH backstop owns it
    (its card, `require_web`) even when the reply ladder reaches its backstop
    rung first — the reply backstop stands down (`research_armed`)."""

    _reply_on(monkeypatch, regen=0.1, research=True, grace_ms=900)
    client, _provider, calls, counters = await _silent_call(
        monkeypatch, ask=NAMED, db="db-reply-vs-research", after=2.0,
    )

    assert _reply_ids(client) == []
    (rid,) = _research_ids(client)
    assert [c["did"] for c in calls] == [rid], calls
    assert calls[0].get("require_web") is True
    # Phase D2: an ARMED backstop is a hold the ladder re-decides, never a skip.
    assert ("live_reply_backstop_held", {"reason": "research_armed"}) in counters
    assert not any(
        name == "live_reply_backstop_skipped" and f.get("reason") == "research_armed"
        for name, f in counters
    ), counters


# ── Phase D2 (correctness-relay-6): ARMED is not FIRED ──────────────────────
#
# The reply ladder used to mark an owed reply DONE the moment another relay
# backstop was merely armed for the turn.  When that backstop then declined
# (`stale` after its max wait, `no_subject`, `already_running`, the play
# stand-down), nothing answered the turn, no `reply_status unanswered` was
# sent, and `thinking` stood until the 60 s expiry — which skipped a done owed.

ENGLISH_SEARCH = "search for the best business book"


def _statuses(client, turn_id) -> list[tuple]:
    return [
        (f["outcome"], f.get("reason")) for f in client.of("reply_status")
        if f.get("user_turn_id") == turn_id
    ]


@pytest.mark.asyncio
async def test_an_armed_research_backstop_that_goes_stale_hands_the_owed_reply_to_the_reply_backstop(
    monkeypatch,
):
    """The caller asks for a search and goes on humming («um …») without a
    pause: the research backstop HOLDS `wait` (caller mid-utterance) when the
    reply ladder reaches its backstop rung — `reply_status pending
    reason=research_armed`, no `relay-reply:` yet — then gives up `stale` at
    its max wait.  The owed reply is not stranded: the reply backstop fires
    once for the turn and says `delegated`."""

    _reply_on(
        monkeypatch, regen=0.3, research=True, grace_ms=250,
        voice_live_reply_backstop_after_s=0.6,
        # A hum must hold the turn open under a loaded test host: a turn
        # closes only after this much input silence.
        voice_live_utterance_gap_ms=800, voice_live_utterance_hard_gap_ms=1600,
    )
    monkeypatch.setattr(live, "_RESEARCH_BACKSTOP_MAX_WAIT_S", 1.2)
    calls: list[dict] = []
    counters = H.capture_counters(monkeypatch)

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append({"did": kw.get("delegation_id"), "task": task, **kw})
        await asyncio.sleep(0.2)
        return "The Lean Startup.", "m"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}
    seen: dict = {}

    async def script():
        p = box["p"]
        p.push(H.user_delta(ENGLISH_SEARCH, 1000, 2500))
        await asyncio.sleep(0.15)
        t = 4500
        for _ in range(60):          # ~2.4 s of «um …», no pause long enough to close
            p.push(H.user_delta(" um", t, t + 40))
            t += 40
            await asyncio.sleep(0.04)
            if not seen.get("held") and any(
                f.get("outcome") == "pending" and f.get("reason") == "research_armed"
                for f in client.of("reply_status")
            ):
                seen["held"] = True
                seen["reply_ids_when_held"] = list(_reply_ids(client))
        await _wait(lambda: _reply_ids(client), timeout=4.0)
        await asyncio.sleep(0.4)

    client = H.FakeClient([
        H.config(features=FEATURES + ["reply_status"]), 0.1, script, 0.3, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=_boot(box), auto_ack=True)
    await H.run_relay(client, provider, timeout=20, db_session_id="db-d2-research-stale")

    finals = [f for f in client.of("transcript") if f.get("final")]
    asked = next(f for f in finals if f["text"] == ENGLISH_SEARCH)
    st = _statuses(client, asked["turn_id"])
    assert seen.get("held") and seen["reply_ids_when_held"] == [], (seen, st)
    assert ("live_reply_backstop_held", {"reason": "research_armed"}) in counters, counters
    assert ("live_research_backstop_skipped", {"reason": "stale"}) in counters, counters
    assert _research_ids(client) == []
    (rid,) = _reply_ids(client)
    assert [c["did"] for c in calls] == [rid], calls
    assert ENGLISH_SEARCH in calls[0]["task"], calls[0]["task"]
    created = [f for f in client.of("delegation")
               if f.get("delegation_id") == rid and f.get("phase") == "created"]
    assert len(created) == 1 and created[0]["parent_user_turn_id"] == asked["turn_id"], created
    assert st.index(("pending", "research_armed")) < st.index(("delegated", None)), st
    assert not [s for s in st if s[0] == "unanswered"], st


@pytest.mark.asyncio
async def test_an_armed_research_backstop_that_fires_leaves_the_reply_ladder_nothing_to_send(
    monkeypatch,
):
    """The other branch of the same hold: the research backstop FIRES after
    the reply ladder reached its rung.  The turn is then claimed by the
    research task (`consumed_by`) — the ladder sends nothing more for it: no
    `relay-reply:`, no `delegated`, no `unanswered`."""

    _reply_on(monkeypatch, regen=0.1, research=True, grace_ms=900)
    calls: list[dict] = []
    counters = H.capture_counters(monkeypatch)

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append({"did": kw.get("delegation_id"), "task": task, **kw})
        await asyncio.sleep(0.4)
        return "The Lean Startup.", "m"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}

    async def script():
        box["p"].push(H.user_delta(ENGLISH_SEARCH, 1000, 2500))
        await _wait(lambda: _research_ids(client), timeout=4.0)
        await asyncio.sleep(2.6)     # > two of the ladder's armed re-decisions

    client = H.FakeClient([
        H.config(features=FEATURES + ["reply_status"]), 0.1, script, 0.3, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=_boot(box), auto_ack=True)
    await H.run_relay(client, provider, timeout=15, db_session_id="db-d2-research-fires")

    finals = [f for f in client.of("transcript") if f.get("final")]
    asked = next(f for f in finals if f["text"] == ENGLISH_SEARCH)
    st = _statuses(client, asked["turn_id"])
    assert ("pending", "research_armed") in st, st
    assert ("live_reply_backstop_held", {"reason": "research_armed"}) in counters, counters
    assert _reply_ids(client) == []
    (sid,) = _research_ids(client)
    assert [c["did"] for c in calls] == [sid], calls
    created = [f for f in client.of("delegation")
               if f.get("delegation_id") == sid and f.get("phase") == "created"]
    assert created and created[0]["parent_user_turn_id"] == asked["turn_id"], created
    after = st[st.index(("pending", "research_armed")) + 1:]
    assert not [s for s in after if s[0] in ("delegated", "unanswered")], st


NAMED_FA = "برام سرچ کن ببین بهترین کتاب برای جذب سرمایه چیه"
SEARCH_BARE = "سرچ کن"


@pytest.mark.asyncio
@pytest.mark.parametrize("bare", [
    # No content word: the reply ladder re-asked the model about it and then
    # gave it up `unanswered not_a_request` — the phone handed «سرچ کن» off.
    SEARCH_BARE,
    # A content word: the reply backstop opened a `relay-reply:` card.
    "سرچش کن", "google it",
])
async def test_an_objectless_ask_for_a_search_already_running_joins_it_and_is_never_reply_backstopped(
    monkeypatch, caplog, bare,
):
    """Phase D3 (supervisor ruling 2): «برام سرچ کن ببین …» is being searched
    (the research backstop's task, still running) when the caller says
    «سرچ کن».  The research backstop skips it `already_running` — and used to
    leave it claimed by nothing, so the reply ladder re-asked the model and
    then handed «سرچ کن» to the agent as a `relay-reply:` card: a second card
    for the search already running.  The ask is the RUNNING task's now:
    consumed by it, in its `request_turn_ids` (its frames and its saved row),
    `reply_status delegated` naming it — and no reply card, no second search."""

    import logging

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    _reply_on(monkeypatch, regen=0.8, research=True, grace_ms=250)
    calls: list[dict] = []
    counters = H.capture_counters(monkeypatch)

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append({"did": kw.get("delegation_id"), "task": task, **kw})
        await asyncio.sleep(3.0 if len(calls) == 1 else 0.1)
        return "منبع: goodreads.com — The Lean Startup.", "m"

    saves = H.patch_relay(monkeypatch, think=think)["saves"]
    box: dict = {}

    def done(rid):
        return any(
            f.get("delegation_id") == rid and f.get("phase") == "completed"
            for f in client.of("delegation")
        )

    async def script():
        p = box["p"]
        p.push(H.user_delta(NAMED_FA, 1000, 2500))
        await _wait(lambda: calls, timeout=4.0)
        await asyncio.sleep(0.2)
        p.push(H.user_delta(bare, 6000, 6600))
        await asyncio.sleep(2.4)     # past the ladder's regen (0.8) and backstop rung (1.6)
        await _wait(lambda: done(calls[0]["did"]), timeout=4.0)
        await asyncio.sleep(0.3)

    client = H.FakeClient([
        H.config(features=FEATURES + ["reply_status"]), 0.1, script, 0.3, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=_boot(box), auto_ack=True)
    await H.run_relay(
        client, provider, timeout=20, db_session_id=f"db-d3-research-joins-{bare.encode().hex()[:12]}",
    )

    finals = [f for f in client.of("transcript") if f.get("final")]
    asked = next(f for f in finals if f["text"] == NAMED_FA)["turn_id"]
    joined = next(f for f in finals if f["text"] == bare)["turn_id"]
    assert ("live_research_backstop_skipped", {"reason": "already_running"}) in counters, counters
    (sid,) = _research_ids(client)
    assert [c["did"] for c in calls] == [sid], calls
    assert _reply_ids(client) == []
    reasks = [e for e in provider.of("session.instructions.append")
              if str(e.get("event_id") or "").startswith("toup-live-reask-")]
    assert reasks == [], reasks
    logs = [r.getMessage() for r in caplog.records]
    assert not any("reply backstop fired" in m for m in logs)
    assert any(
        m.startswith(f"[LIVE] research backstop joined id={sid[:24]} turn_id={joined} ") for m in logs
    ), [m for m in logs if "research backstop" in m]
    # The phone: `reply_status delegated` names the running task (the app keeps
    # a task's FIRST request_turn_ids as its identity, so a later frame alone
    # would never cover the ask); nothing says the ask went unanswered.
    st = [(f["outcome"], f.get("task_id")) for f in client.of("reply_status")
          if f.get("user_turn_id") == joined]
    assert ("delegated", sid) in st, st
    assert not [o for o, _t in st if o == "unanswered"], st
    created = next(f for f in client.of("delegation")
                   if f.get("delegation_id") == sid and f.get("phase") == "created")
    assert created["parent_user_turn_id"] == asked
    assert joined not in created["request_turn_ids"], created
    wired = [f for f in client.of("delegation")
             if f.get("delegation_id") == sid and "request_turn_ids" in f]
    assert wired[-1]["phase"] == "completed" and joined in wired[-1]["request_turn_ids"], wired
    rows = [r for r in saves if (r.get("assistant_voice") or {}).get("delegation_id") == sid]
    assert rows and joined in rows[-1]["assistant_voice"]["request_turn_ids"], rows


@pytest.mark.asyncio
@pytest.mark.parametrize("ask", [
    "next song", "چی شد",
    # Merge seam RA × RB: bare playback commands the context-free grammar does
    # not read (D-R4).  Nothing plays here, so the media backstop declines them
    # and the model stays silent; a `relay-reply:` task would hand "stop" to the
    # agent, whose `media_control` tool would halt whatever plays next.
    "stop", "بسه", "استاپش کن", "pause", "resume the music",
])
async def test_a_status_checkin_or_media_command_is_never_reply_backstopped(monkeypatch, ask):
    _reply_on(monkeypatch)
    client, _provider, calls, counters = await _silent_call(
        monkeypatch, ask=ask, db=f"db-reply-never-{ask.encode().hex()[:16]}",
    )

    assert _reply_ids(client) == []
    assert not [c for c in calls if str(c["did"]).startswith(live.RELAY_REPLY_PREFIX)]
    assert not any(name == "live_reply_backstop" for name, _f in counters)


# ── Phase D (acceptance-2): a caller who silenced the agent is owed silence ──

@pytest.mark.asyncio
@pytest.mark.parametrize("silence", ["ساکت شو", "بسه", "stop", "be quiet"])
@pytest.mark.parametrize("over_answer", [False, True])
async def test_silencing_the_agent_is_never_owed_a_reply(monkeypatch, silence, over_answer):
    """«ساکت شو» / "stop" said to the AGENT (nothing playing, no task running)
    was opened as an owed reply: 3 s later the regen made the agent talk
    again, and at ~6 s «ساکت شو» became a `relay-reply:` card (or, for
    "stop", `reply_status unanswered` → «برای «stop» پاسخی نیامد»).  It is
    `phatic` on the wire, owes nothing, and — said over an answer — the
    answer it cut off is not asked again either."""

    _reply_on(monkeypatch)
    calls: list = []

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append(kw.get("delegation_id"))
        return "ok", "m"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}

    async def script():
        p = box["p"]
        if over_answer:
            p.push(H.user_delta("tell me about the program", 200, 900))
            await asyncio.sleep(0.3)
            p.push(H.out_text("The program covers the whole", 1000, 1600))
            p.push(H.out_audio("A1"))
            await asyncio.sleep(0.2)
            p.push(H.user_delta(silence, 1700, 2100))
        else:
            p.push(H.user_delta(silence, 1000, 1500))
        await asyncio.sleep(2.4)

    features = FEATURES + ["reply_status"]
    client = H.FakeClient([H.config(features=features), 0.1, script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_boot(box), auto_ack=True)
    await H.run_relay(client, provider, timeout=15,
                      db_session_id=f"db-silence-{over_answer}-{silence.encode().hex()[:12]}")

    finals = [f for f in client.of("transcript") if f.get("final")]
    mine = next(f for f in finals if f["text"] == silence)
    assert mine.get("phatic") is True, mine
    reasks = [i for i in provider.of("session.instructions.append")
              if str(i.get("event_id") or "").startswith("toup-live-reask-")]
    assert reasks == [], reasks
    assert _reply_ids(client) == [] and calls == []
    assert not [f for f in client.of("reply_status")
                if f["outcome"] in ("reasked", "delegated")], client.of("reply_status")
    assert not [f for f in client.of("reply_status")
                if f["user_turn_id"] == mine["turn_id"]], client.of("reply_status")


@pytest.mark.parametrize("text,want", [
    ("ساکت شو", True), ("Be quiet.", True), ("stop", True), ("بسه", True),
    ("stop the timer", False), ("استاپش کن", False), ("be quiet for a second and listen", False),
    ("enough about that, what's the weather", False),
])
def test_silence_directive_is_the_whole_utterance(text, want):
    assert live.silence_directive(text) is want


@pytest.mark.asyncio
async def test_a_late_delegation_after_a_filler_is_absorbed_into_the_backstopped_request(
    monkeypatch,
):
    """Phase D (tests-guards-6): the absorber's "newest causal turn" skips a
    turn that asks nothing (critic G9) — a «ها» spoken over a backstopped
    request, then the model's late delegation, is still about the request.
    Nothing pinned the skip: without it the late call was told "No task was
    started for the caller's words «ها» … Reply briefly yourself"."""

    _reply_on(monkeypatch)
    calls: list = []
    counters = H.capture_counters(monkeypatch)

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append(kw.get("delegation_id"))
        await asyncio.sleep(0.4)
        return "Robarts closes at 11 pm.", "m"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}

    async def script():
        p = box["p"]
        p.push(H.user_delta(ROBARTS, 1000, 2500))
        await _wait(lambda: _reply_ids(client))
        p.push(H.user_delta("ها", 2700, 2900))
        await asyncio.sleep(0.4)
        p.push(H.delegation("d-late", 3000))
        await asyncio.sleep(1.2)

    client = H.FakeClient([H.config(features=FEATURES), 0.1, script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_boot(box), auto_ack=True)
    await H.run_relay(client, provider, timeout=15, db_session_id="db-reply-absorbed-filler")

    (rid,) = _reply_ids(client)
    assert calls == [rid], calls
    assert ("live_delegation_absorbed", {"reason": "relay_reply"}) in counters
    late = " ".join(str(e.get("content")) for e in provider.sent
                    if e.get("delegation_id") == "d-late")
    assert "already handed to their agent" in late and "No task was started" not in late, late


# ── Voice round 2026-10-07 (D-R3(1)/(2)): one question, one card ──────────
# The 12:04 call: the research backstop fired on «… قوی‌ترین استاد یو‌اف‌تی توی
# بحث ال‌ال‌ام» (consuming it) and 2 ms later GPT-Live's own delegation arrived
# with `offset_ms` = the ONSET of «کیه», said 1.4 s after the request ended.
# The relay bound it to «کیه» alone: a second card, a 252k-token run, the step
# budget's boilerplate.  «کیه» now JOINS the running search (an addendum) and
# the model's delegation is absorbed; a decision at the onset of a genuinely
# new request is absorbed as the running task's own (the seam).

KIE_REQUEST = "سرچ کن ببین قوی‌ترین استاد یوافتی توی بحث ال‌ال‌ام"
KIE = "کیه"


def _addendum_on(monkeypatch, *, window=5000, onset=400):
    monkeypatch.setattr(settings, "voice_live_addendum_window_ms", window)
    monkeypatch.setattr(settings, "voice_live_seam_onset_ms", onset)


def _tenant_routes(monkeypatch, answer=None):
    """The tenant's addendum route, recorded.  Every other tenant read (the
    durable-cancel status poll) answers nothing, as an unreachable tenant."""

    from app.api import ws_realtime as rt

    posts: list[dict] = []

    async def api(_url, _key, _method, path, json_body=None, timeout=None, outcome=None, **_kw):
        if path.endswith("/voice-delegations/addendum"):
            posts.append(dict(json_body or {}))
            return answer if answer is not None else {"accepted": True, "state": "queued"}
        return None

    async def vps(_user_id):
        return ("https://tenant.example", "agent-key")

    monkeypatch.setattr(rt, "_vps_api", api)
    return posts, vps


def _created_ids(client) -> list[str]:
    return [f["delegation_id"] for f in client.of("delegation") if f.get("phase") == "created"]


@pytest.mark.asyncio
@pytest.mark.parametrize("onset, reason", [(0, "addendum"), (400, "seam")])
async def test_a_question_tail_joins_the_running_search_and_the_models_delegation_is_absorbed(
    monkeypatch, caplog, onset, reason,
):
    """With the seam OFF the model's delegation (decided at the tail's onset)
    waits for the tail to close and is absorbed because the tail JOINED the
    search (`reason=addendum`); with it ON — production — it is absorbed at
    the onset, before the tail closes (`reason=seam`).  One card either way."""

    import logging

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    _on(monkeypatch)
    _addendum_on(monkeypatch, onset=onset)
    posts, vps = _tenant_routes(monkeypatch)
    calls: list[str] = []

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append(kw.get("delegation_id"))
        await asyncio.sleep(1.2)
        return "کالین رافل، استاد علوم کامپیوتر یوافتی.", "m"

    H.patch_relay(monkeypatch, think=think, vps=vps)
    box: dict = {}

    async def script():
        p = box["p"]
        p.push(H.user_delta(KIE_REQUEST, 1000, 2500))
        await _wait(lambda: calls, timeout=3.0)
        # 1.4 s after the request ended (provider timeline), and GPT-Live's
        # delegation decided at the first sample of the tail.
        p.push(H.user_delta(KIE, 3900, 4100))
        p.push(H.delegation("d-model", 3900))
        await _wait(lambda: any(
            f.get("phase") == "completed" for f in client.of("delegation")
        ), timeout=6.0)
        await asyncio.sleep(0.3)

    client = H.FakeClient([
        H.config(features=FEATURES + ["reply_status"]), 0.1, script, 0.2, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=_boot(box), auto_ack=True)
    await H.run_relay(client, provider, timeout=20, db_session_id="db-r3-addendum-research")

    (sid,) = _research_ids(client)
    assert _created_ids(client) == [sid], "the tail became a card of its own"
    assert calls == [sid], calls
    finals = {f["text"]: f["turn_id"] for f in client.of("transcript") if f.get("final")}
    kie = finals[KIE]
    logs = [r.getMessage() for r in caplog.records]
    assert any(
        m.startswith(f"[LIVE] addendum joined id={sid[:24]} turn_id={kie} kind=question_tail "
                     "since_request_ms=1400 n=1 chars=3 target=running")
        for m in logs
    ), [m for m in logs if "addendum" in m]
    absorbed_on = kie if reason == "addendum" else finals[KIE_REQUEST]
    assert any(
        m.startswith(f"[LIVE] delegation absorbed id=d-model turn_id={absorbed_on} "
                     f"reason={reason} owner={sid[:24]}")
        for m in logs
    ), [m for m in logs if "absorbed" in m]
    assert not any("سرچ" in m or "استاد" in m for m in logs if "[LIVE] addendum" in m)
    # The card: ONE card whose request now ends with the tail, the ids growing
    # from the held list (the app adopts a longer list that starts with it).
    lifecycle = [f for f in client.of("delegation")
                 if f.get("delegation_id") == sid and "task_revision" in f]
    created = next(f for f in lifecycle if f["phase"] == "created")
    grown = next(f for f in lifecycle if kie in (f.get("request_turn_ids") or []))
    assert grown["phase"] == "started" and grown["task_revision"] > created["task_revision"]
    assert grown["request_turn_ids"][:len(created["request_turn_ids"])] == created["request_turn_ids"]
    assert grown["title"].endswith(KIE), grown["title"]
    assert ("delegated", sid) in [
        (f["outcome"], f.get("task_id")) for f in client.of("reply_status")
        if f.get("user_turn_id") == kie
    ]
    # The model is told quietly; the running tenant turn gets the words.
    assert any(KIE in str(e.get("content") or "") and "added" in str(e.get("content") or "")
               for e in provider.of("session.thinking.append"))
    assert posts == [{
        "session_id": "db-r3-addendum-research", "task_id": sid,
        "addendum_id": kie, "text": KIE, "lang": "fa",
    }], posts
    assert any(m.startswith(f"[LIVE] addendum forwarded id={sid[:24]} turn_id={kie} outcome=accepted ")
               for m in logs)


@pytest.mark.asyncio
async def test_a_decision_at_the_onset_of_a_new_request_belongs_to_the_running_task(
    monkeypatch, caplog,
):
    """The seam: while the search runs the caller starts a NEW question and
    GPT-Live's delegation arrives at its first sample — about the request it
    had heard, not these new words.  Absorbed as the running task's; the new
    question gets its own (later) delegation.  Two cards, never three."""

    import logging

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    _on(monkeypatch)
    _addendum_on(monkeypatch)
    calls: list[str] = []

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls.append(kw.get("delegation_id"))
        await asyncio.sleep(1.2 if len(calls) == 1 else 0.1)
        return "یک جواب.", "m"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}
    weather = "هوای تورنتو چطوره؟"

    async def script():
        p = box["p"]
        p.push(H.user_delta(KIE_REQUEST, 1000, 2500))
        await _wait(lambda: calls, timeout=3.0)
        p.push(H.user_delta(weather, 3500, 4700))
        p.push(H.delegation("d-seam", 3500))
        await asyncio.sleep(0.4)
        p.push(H.delegation("d-weather", 4800))
        await _wait(lambda: len(calls) >= 2, timeout=4.0)
        await asyncio.sleep(1.5)

    client = H.FakeClient([H.config(), 0.1, script, 0.2, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_boot(box), auto_ack=True)
    await H.run_relay(client, provider, timeout=20, db_session_id="db-r3-seam")

    (sid,) = _research_ids(client)
    assert _created_ids(client) == [sid, "d-weather"], _created_ids(client)
    finals = {f["text"]: f["turn_id"] for f in client.of("transcript") if f.get("final")}
    logs = [r.getMessage() for r in caplog.records]
    assert any(
        m.startswith(f"[LIVE] delegation absorbed id=d-seam turn_id={finals[KIE_REQUEST]} "
                     f"reason=seam owner={sid[:24]} onset_ms=0")
        for m in logs
    ), [m for m in logs if "absorbed" in m]
    assert not any("[LIVE] addendum joined" in m for m in logs), "a new question is not an addendum"
    weather_frames = [f for f in client.of("delegation") if f.get("delegation_id") == "d-weather"]
    assert weather_frames and weather_frames[0]["turn_id"] == finals[weather]


# ══════════════════════════════════════════════════════════════════════
# Voice round 2026-10-07 (D-R4 (1)): ONE web decision per task
# ══════════════════════════════════════════════════════════════════════
# The 10-07 call: card A (the relay's research backstop) had `require_web` and
# its first search 0.9 s after dispatch; card 1 (the MODEL delegated
# «…یه چیزی برام سرچ کن. ببین آیاس جنرال…») and card 2 (a follow-up question
# about the same subject) had none, so their first tool was the model's own,
# 5–7 s in.  `decide_web` now reads every task at its first `created`.

SEARCH_SPAN = "یه چیزی برام سرچ کن. ببین اه آیاس جنرال ریدینگ برای نمره هشت چند تا غلط مجازه"
FOLLOWUP = "اوکی. بعد به نظرت همین توی آیسآکادمیک چجوریه"


async def _two_asks(monkeypatch, first, second, *, db, first_slow=1.6, gap=0.5):
    """`first` delegated as d1 (its agent turn runs `first_slow` s), then
    `second` delegated as d2 while d1 still runs (`second` None: only d1).
    Returns {delegation_id: the `_think` kwargs}."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    calls: dict = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        calls[kw.get("delegation_id")] = kw
        if kw.get("delegation_id") == "d1":
            await asyncio.sleep(first_slow)
        return "جواب.", "m"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}

    async def script():
        p = box["p"]
        p.push(H.user_delta(first, 1000, 2400))
        p.push(H.delegation("d1", 2450))
        await _wait(lambda: "d1" in calls)
        if second is not None:
            await asyncio.sleep(gap)
            p.push(H.user_delta(second, 5000, 6200))
            p.push(H.delegation("d2", 6250))
            await _wait(lambda: "d2" in calls)
        await asyncio.sleep(0.3)

    client = H.FakeClient([H.config(features=FEATURES), 0.1, script, 0.3, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_boot(box), auto_ack=True)
    await H.run_relay(client, provider, timeout=15, db_session_id=db)
    return calls


@pytest.mark.asyncio
async def test_a_model_delegation_of_a_search_ask_carries_require_web(monkeypatch, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    calls = await _two_asks(monkeypatch, SEARCH_SPAN, None, db="db-web-evidence", first_slow=0.1)
    d1 = calls["d1"]
    assert d1.get("require_web") is True, d1
    # The speculative search's query is the request, not the span: no
    # hesitation, no frame words.
    query = d1.get("search_query") or ""
    assert "آیاس" in query and "جنرال" in query, query
    for frame_word in ("اه", "ببین", "سرچ", "برام", "چیزی"):
        assert frame_word not in query.split(), query
    assert len(query.split()) <= live.SEARCH_QUERY_MAX_WORDS
    assert any(
        "delegation web id=d1 require_web=1 source=evidence" in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_a_followup_question_with_a_back_reference_inherits_require_web(monkeypatch, caplog):
    """Card 2 of the 10-07 call: a question that refers back («همین») while
    the research it refers to still runs is research too."""

    import logging

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    calls = await _two_asks(monkeypatch, SEARCH_SPAN, FOLLOWUP, db="db-web-followup")
    assert calls["d1"].get("require_web") is True
    d2 = calls["d2"]
    assert d2.get("require_web") is True, d2
    # Phase D (search-play-title-1): the cleaner leaves card 2 ONE word
    # («آیسآکادمیک»), which the tenant's t0 search refuses — so the relay
    # sends no query and the tenant searches its own cleaning of the display.
    assert len((d2.get("search_query") or "a b").split()) >= 2, d2
    assert any(
        "delegation web id=d2 require_web=1 source=followup" in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("second", [
    "what time is it in tokyo right now?",       # a question with no back-reference
    "این آهنگ چیه؟",                              # a media question
    "همین رو تو تقویمم چک کن؟",                   # the caller's own data (owned-data guard)
    # Phase D2 ruling 51 (search-play-title-2): an ACTION request ending in
    # "?" that refers to the work («it») is not research…
    "can you set a reminder for it tomorrow?",
    # …and «همین الان» is "right now", not a back-reference.
    "همین الان ساعت چنده؟",
    # Phase D3 ruling 57: a Persian verb of ACTING (send) is refused too.
    "همین رو برام بفرست؟",
    # Phase D4 ruling 66 (Phase D3 review F4): …and in the polite
    # subjunctive a Persian speaker asks it in ("can you send it?"), stem +
    # a personal ending.  The review's row («همینو» is no back word, so it is
    # off the web either way — its verb is pinned in
    # `test_the_persian_action_verb_takes_the_polite_subjunctive`), and the
    # same with «همین رو», which only the ending keeps off the web.
    "میشه همینو برام بفرستی؟",
    "میشه همین رو برام بفرستی؟",
], ids=["no_back_reference", "media", "owned_data", "action_request", "right_now_idiom",
        "action_request_fa", "action_request_fa_polite", "action_request_fa_polite_back"])
async def test_a_question_that_is_not_about_the_research_stays_off_the_web(monkeypatch, second):
    calls = await _two_asks(
        monkeypatch, SEARCH_SPAN, second, db=f"db-web-neg-{abs(hash(second)) % 10**6}",
    )
    assert calls["d1"].get("require_web") is True
    assert "require_web" not in calls["d2"], calls["d2"]
    assert "search_query" not in calls["d2"]
    if second.startswith("can you set a reminder"):
        # …and no web plan on its card.
        assert not set(calls["d2"].get("planned_steps") or ()) & {"web", "fetch"}, calls["d2"]


@pytest.mark.asyncio
@pytest.mark.parametrize("second", [
    FOLLOWUP,                                    # card 2 of the 10-07 call («همین توی …»)
    "how did they do it?",                       # a research question about the work
    "what did they find about it?",              # it ASKS about the work
    # Ruling 57: a verb of ACTING (write) that still ASKS about the work — the
    # research interrogative keeps it research.
    "what did they write about it?",
    # Phase D3 ruling 57 (Phase D2 F3): card 2 asked as an IMPERATIVE.  A verb
    # of ASKING (tell, explain, read, know, find, give) is research about the
    # work, not an action — through ts-1's addendum tables these lost their
    # web search, the 10-07 shape in which Academic was answered from
    # General's context.
    "همین رو برای آکادمیک هم بگو؟",
    "همین رو برای آکادمیک هم توضیح بده؟",
    "بعد همین توی آکادمیک رو هم بگو؟",
    "همین رو برای آکادمیک هم بخون؟",
    "can you tell me the same for academic?",
    "do you know the same for academic?",
    "can you find the same for academic?",
    "can you give me the same numbers for academic?",
], ids=["card_2", "how_did_they", "what_did_they_find", "what_did_they_write", "fa_tell", "fa_explain", "fa_then_tell",
        "fa_read", "en_tell", "en_know", "en_find", "en_give"])
async def test_a_research_question_about_the_running_work_stays_web(monkeypatch, caplog, second):
    """Phase D2 ruling 51's controls: what source 4 exists for still goes to
    the web — a back-reference question about research still running."""

    import logging

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    calls = await _two_asks(
        monkeypatch, SEARCH_SPAN, second, db=f"db-web-pos-{abs(hash(second)) % 10**6}",
    )
    assert calls["d2"].get("require_web") is True, calls["d2"]
    assert any(
        "delegation web id=d2 require_web=1 source=followup" in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.parametrize(("text", "action"), [
    # the review's sentence, and ruling 57's own Persian rows
    ("میشه همینو برام بفرستی؟", True), ("همین رو برام بفرست؟", True),
    ("همین رو برای آکادمیک هم بگو؟", False), ("میشه همین رو برای آکادمیک هم بگی؟", False),
    ("بفرست", True), ("نفرست", True),            # ruling 57: the bare imperative / negative
    ("بفرستش", True), ("بفرستشون", True), ("نفرستش", True),  # ruling 69: + an object clitic
    ("بفرستی", True), ("بفرستید", True), ("بفرستین", True),  # ruling 66: + a personal ending…
    ("بنویسی", True), ("نزنی", True), ("بگذارید", True),
    # …in the SECOND person only (ruling 69 (a)): first and third person ask
    # about someone else, not for something.
    ("بفرستیم", False), ("بفرسته", False), ("بفرستن", False), ("بخره", False),
    # verbs of ASKING keep their subjunctive research (ruling 57's never-list)
    ("بگی", False), ("بدی", False), ("بخونی", False), ("بگردی", False), ("بپرسی", False),
    # a stem + an ending that is a word of its own is not a verb
    ("بنده", False), ("نخیر", False),
    # …nor is ب/ن + «ی» + a stem (Phase D5 ruling 70: no action stem starts
    # with a vowel, so that form is never a verb here — «نیزن», a ney player)
    ("نیزن", False),
])
def test_the_persian_action_verb_takes_the_polite_subjunctive(text, action):
    """Phase D4 ruling 66 (Phase D3 review F4): ruling 57's Persian action
    matcher saw only the BARE imperative («بفرست»), so «میشه همینو برام
    بفرستی؟» — how a Persian speaker politely asks "can you send it?" — was
    not an action, and after a research task it went out `require_web` with
    a web plan seed.  ب/ن + an action stem + a personal ending is an action
    now; the information stems stay out of it (only `_followup_action_verb`:
    `_addendum_verb` is untouched).  Phase D5 ruling 69 narrowed the endings
    to the second person (ی ید ین) and gave the bare imperative its object
    clitic (ش شون)."""

    assert live._followup_action_verb(live._leading_words(text)) is action


#: Phase D5 ruling 69 (Phase D4 review P3-2): the reviewer's rows — the
#: founder's Academic question in its most natural Persian forms — and one
#: row per condition that ONLY that condition refuses (each reviewer row is
#: refused by two of them at once, so it alone cannot pin any one).
_R69_REQUESTS = [
    "میشه همینو برام بفرستی؟", "میشه همین رو برام بفرستی؟",
    "میشه همین رو برام بفرستین؟", "لطفا همین رو برام بفرستید",
    "همین رو برام بفرستش؟", "همون کتابا رو برام بفرستشون؟",
    # (b) and (c) read the SUBJUNCTIVE only: an imperative stays a request
    "همین رو برام بفرست ببینم چقدر میشه",
]
_R69_QUESTIONS = {
    # the reviewer's rows
    "برای آکادمیک هم باید همین نمره رو بگیری؟": "b",
    "برای آکادمیک هم همین نمره رو باید بگیرن؟": "ab",
    "اونجا هم همین کتابا رو باید بخری؟": "b",
    "همین آزمون رو کجا میشه بخریم؟": "ac",
    "برای آکادمیک هم چند باید بگیری؟": "bc",
    "برای آکادمیک هم چقدر باید بگیری؟": "bc",
    "برای آکادمیک هم ریدینگ چقدر باید بگیری؟": "bc",
    # (a) alone: first / third person, no deontic, no question word
    "همین کتابا رو اونجا هم بخریم؟": "a",
    "برای آکادمیک هم همین نمره رو بگیرن کافیه؟": "a",
    "برای آکادمیک هم همین نمره رو بگیره کافیه؟": "a",
    # (b) alone: «نباید» / «بایست»
    "برای آکادمیک هم نباید همین نمره رو بگیری؟": "b",
    "برای آکادمیک هم بایست همین نمره رو بگیری؟": "b",
    # (c) alone: a second-person request shape under «کجا / چند / چقدر / کی»
    "همین کتابا رو کجا بخری ارزونتره؟": "c",
    "برای آکادمیک هم ریدینگ چند بگیری کافیه؟": "c",
    "برای آکادمیک هم ریدینگ چقدر بگیری کافیه؟": "c",
    "همین آزمون رو کی بخری بهتره؟": "c",
}


@pytest.mark.parametrize(("text", "action"), [
    *[(t, True) for t in _R69_REQUESTS],
    *[(t, False) for t in _R69_QUESTIONS],
], ids=[
    *[f"request{i}" for i in range(len(_R69_REQUESTS))],
    *[f"{refused}{i}" for i, refused in enumerate(_R69_QUESTIONS.values())],
])
def test_the_persian_subjunctive_is_an_action_only_as_a_request(text, action):
    """Phase D5 ruling 69 (Phase D4 review P3-2): ruling 66 read every
    subjunctive of an action stem as a request, so the Academic question asked
    the way a Persian speaker asks it — «برای آکادمیک هم باید همین نمره رو
    بگیری؟», «همین آزمون رو کجا میشه بخریم؟» — was an ACTION and left the web:
    the 10-07 shape in which Academic was answered from General's context.
    The subjunctive is a request only in the second person (a), with no
    «باید / نباید / بایست» (b) and no «کجا / چند / چقدر / کی» (c); the bare
    imperative also takes «ش / شون» («بفرستش»)."""

    assert live._followup_action_verb(live._leading_words(text)) is action


@pytest.mark.asyncio
@pytest.mark.parametrize(("second", "source"), [
    # «همینو» is no back word: off the web either way (ruling 66's note);
    # the «همین رو» row is the one only the matcher keeps off the web.
    ("میشه همینو برام بفرستی؟", "none"),
    ("میشه همین رو برام بفرستی؟", "none"),
    ("همین رو برام بفرستش؟", "none"),
    ("همون کتابا رو برام بفرستشون؟", "none"),
    *[(t, "followup") for t in (
        "برای آکادمیک هم باید همین نمره رو بگیری؟",
        "برای آکادمیک هم همین نمره رو باید بگیرن؟",
        "اونجا هم همین کتابا رو باید بخری؟",
        "همین آزمون رو کجا میشه بخریم؟",
        "برای آکادمیک هم چند باید بگیری؟",
        # The reviewer's «برای آکادمیک هم چقدر باید بگیری؟» carries no back
        # word and no word of the research, so it is off the web whatever the
        # matcher says; «ریدینگ» is the research's own subject.
        "برای آکادمیک هم ریدینگ چقدر باید بگیری؟",
        "همین کتابا رو اونجا هم بخریم؟",
        "برای آکادمیک هم همین نمره رو بگیرن کافیه؟",
        "همین کتابا رو کجا بخری ارزونتره؟",
        "برای آکادمیک هم ریدینگ چند بگیری کافیه؟",
        "همین آزمون رو کی بخری بهتره؟",
    )],
], ids=["polite_fa", "polite_fa_back", "clitic", "clitic_plural",
        "must_get", "must_get_3pl", "must_buy", "where_buy_1pl", "how_many_must", "how_much_must",
        "a_buy_1pl", "a_get_3pl", "c_where", "c_how_many", "c_when"])
async def test_a_persian_question_about_the_research_is_web_and_a_request_is_not(
    monkeypatch, caplog, second, source,
):
    """Ruling 69 end to end: a research task running, then the ask.  A
    request TO the agent goes `source=none`; a requirement / where / how-many
    question in the subjunctive stays `source=followup`, `require_web=1`."""

    import logging

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    calls = await _two_asks(
        monkeypatch, SEARCH_SPAN, second, db=f"db-web-r69-{abs(hash(second)) % 10**6}",
    )
    web = source == "followup"
    assert calls["d1"].get("require_web") is True
    assert (calls["d2"].get("require_web") is True) is web, calls["d2"]
    assert any(
        f"delegation web id=d2 require_web={int(web)} source={source}" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records if "delegation web id=d2" in r.getMessage()]


@pytest.mark.asyncio
@pytest.mark.parametrize("ask", [
    "تو اسلک سرچ کن ببین علی دیروز چی گفت",
    "تو واتساپ سرچ کن ببین مریم چی فرستاده",
    "تو جلسه‌هام سرچ کن ببین فردا با کی قرار دارم",
], ids=["slack", "whatsapp", "meetings"])
async def test_an_explicit_search_of_the_callers_own_data_is_never_web(monkeypatch, ask):
    """Phase D search-play-title-3 (P1): only `decide_web`'s source 4 checked
    for the caller's own data; sources 2 (evidence) and 3 (linked) had no
    guard, and the relay's not-web word list lacks اسلک / واتساپ / جلسه — so a
    Slack / WhatsApp / meetings search went out `require_web` with a web plan
    seed.  Every source now asks the tenant's own predicate."""

    calls = await _two_asks(
        monkeypatch, ask, None, db=f"db-web-owned-{abs(hash(ask)) % 10**6}", first_slow=0.1,
    )
    d1 = calls["d1"]
    assert "require_web" not in d1, d1
    assert not set(d1.get("planned_steps") or ()) & {"web", "fetch"}, d1


@pytest.mark.asyncio
@pytest.mark.parametrize("ask", [
    "تو اسلک سرچ کن ببین علی دیروز چی گفت",
    "تو واتساپ سرچ کن ببین مریم چی فرستاده",
    NAMED,                                       # control: a real web ask
], ids=["slack", "whatsapp", "web_control"])
async def test_the_research_backstop_never_opens_a_web_search_of_the_callers_own_data(
    monkeypatch, caplog, ask,
):
    """Phase D3 ruling 58 (Phase D2 F4): deviation 24 says own data is never
    web "at EVERY source", but `decide_web` read the backstop's
    `require_web` BEFORE the owned-data test.  An explicit search of the
    caller's own Slack / WhatsApp arms the research backstop; when the model
    did not delegate in time the relay opened it `require_web=1
    source=backstop` with a web plan seed and told the model «… یک جستجوی وب
    شروع کرد».  Now own data is tested first: no `require_web`, no query, no
    web/fetch rows, and the model is told the request is being looked into.
    The control — a real web ask through the same backstop — is unchanged."""

    import logging

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    _on(monkeypatch)
    client, provider, calls, _counters = await _call(
        monkeypatch, ask=ask, db=f"db-backstop-owned-{abs(hash(ask)) % 10**6}",
    )
    (rid,) = _research_ids(client)
    assert [c["did"] for c in calls] == [rid], calls
    call = calls[0]
    decided = [r.getMessage() for r in caplog.records
               if f"delegation web id={rid[:12]} " in r.getMessage()]
    plan = next((f["plan"] for f in client.of("delegation")
                 if f.get("delegation_id") == rid and f.get("plan")), [])
    request = " ".join(call["display_request"].split())[:120]
    instructions = _instructions(provider)
    if ask == NAMED:
        assert call.get("require_web") is True, call
        assert decided == [f"[LIVE] delegation web id={rid[:12]} require_web=1 source=backstop"]
        assert instructions.count(relay_line("research_started", "fa", request=request)) == 1
        return
    assert "require_web" not in call and "search_query" not in call, call
    assert not set(call.get("planned_steps") or ()) & {"web", "fetch"}, call
    assert not {row.get("kind") for row in plan} & {"web", "fetch"}, plan
    assert decided == [f"[LIVE] delegation web id={rid[:12]} require_web=0 source=none"], decided
    assert not [i for i in instructions if "جستجوی وب" in i or "web search" in i], instructions
    assert instructions.count(relay_line("research_started_own", "fa", request=request)) == 1, (
        instructions
    )


@pytest.mark.asyncio
async def test_a_plain_request_with_no_research_around_it_stays_off_the_web(monkeypatch):
    calls = await _two_asks(
        monkeypatch, "find me a dentist near union station", None, db="db-web-none",
        first_slow=0.1,
    )
    assert "require_web" not in calls["d1"], calls["d1"]


def test_a_retry_or_correction_of_research_is_research():
    from types import SimpleNamespace

    session = live._LiveSession.__new__(live._LiveSession)
    session.settings = settings
    session.user_id = "user-1"
    session.supervisor = SimpleNamespace(running={}, queued=[])
    research = live.DelegatedTask(delegation_id="d1", offset_ms=0, transcript=SEARCH_SPAN)
    assert session.decide_web(research) == "evidence"
    retry = live.DelegatedTask(delegation_id="d2", offset_ms=0, transcript="retry")
    retry.relation_kind, retry.related_task_id = live.RELATION_RETRIES, "d1"
    assert session.decide_web(retry) == "linked" and retry.require_web
    correction = live.DelegatedTask(delegation_id="d3", offset_ms=0, transcript="no, academic")
    correction.bound_task_id = "d1"
    assert session.decide_web(correction) == "linked" and correction.require_web
    unrelated = live.DelegatedTask(delegation_id="d4", offset_ms=0, transcript="retry")
    unrelated.relation_kind, unrelated.related_task_id = live.RELATION_RETRIES, "d9"
    assert session.decide_web(unrelated) == "none" and not unrelated.require_web
    # ONE decision per task: asking again changes nothing.
    assert session.decide_web(retry) == "linked"


@pytest.mark.asyncio
async def test_the_agent_turn_body_carries_the_search_query_and_the_planned_steps(monkeypatch):
    """D-R4 (1)/(6): additive body keys an older image ignores — the cleaned
    first search query and the plan the card was seeded with."""

    from app.api import ws_realtime as rt

    bodies: list[dict] = []

    async def vps(_user_id):
        return ("http://tenant.invalid", "key")

    async def stream(_url, _key, path, json_body, relay, cause=None):
        bodies.append(dict(json_body))
        return "stream", {"text": "Found it."}, 1

    class Relay:
        async def close_open(self):
            return None

    monkeypatch.setattr(rt, "_get_vps_info", vps)
    monkeypatch.setattr(rt, "_vps_api_stream", stream)
    monkeypatch.setattr(rt, "_agent_runner", None)
    monkeypatch.setattr(settings, "voice_realtime_v2", True)
    monkeypatch.setattr(settings, "voice_realtime_tool_events", True)
    await rt._think(
        "user-1", "find the best book", "db-session", relay=Relay(), delegation_id="d1",
        require_web=True, search_query="best  business book",
        planned_steps=["web", "fetch", "answer"],
    )
    await rt._think("user-1", "hello", "db-session", relay=Relay(), delegation_id="d2")
    assert bodies[0]["search_query"] == "best business book"
    assert bodies[0]["planned_steps"] == ["web", "fetch", "answer"]
    assert "search_query" not in bodies[1] and "planned_steps" not in bodies[1]
