"""R2 fix round (w2b): grounded progress speech as revised by addendum 7, and F14.

§E shipped one line per task, only after a real tool.start. The evidence's
5–16 s toolless Deep turns got no spoken feedback at all, and a search that
moved on to reading a page said nothing about it (critic gaps, Req 12/14).
Addendum 7: up to 2 lines per task — one after a real tool.start once the wait
is noticeable, a second only on a real change to a DIFFERENT tool kind; a
toolless running task gets at most one honest "still working on «title»" line
after `voice_live_progress_idle_after_s` (never "searching"); never over the
caller; never counts as delivery; claims bound to their own epoch.

F14 (verify-relay-tasks-7-0): a result appended while the progress line's
bare claim was still unbound had its epoch captured by that claim — the answer
the caller heard was recorded `spoken:false` ("Result in chat"), nudged into a
second telling, and stored twice.

Driven through the real relay (`test_live_harness`); the model is a fake that
voices exactly what the test says it voices, on a provider clock that runs
forward.
"""

import asyncio

import pytest

import test_live_harness as H
from app.config import settings
from app.services.live_voice_protocol import relay_line, request_title


ANSWER = "Professor Ada Lovelace works on robotics at Toronto."
REQUEST = "research the toronto robotics faculty"


def _set(monkeypatch, **values):
    """Tolerant of a setting being absent: on a relay without it the test
    fails on its assertion, not on setup."""

    for key, value in values.items():
        try:
            monkeypatch.setattr(settings, key, value)
        except (AttributeError, ValueError):
            pass


def _commentary(provider) -> list[dict]:
    return provider.of("session.commentary.append")


def _progress_lines(provider, did="d1") -> list[str]:
    return [
        e["content"] for e in _commentary(provider)
        if e.get("delegation_id") == did and ANSWER not in e["content"]
        and "Answer" not in e["content"]
    ]


def _card_title(request_text: str) -> str:
    """The card's title the idle line names (2026-10-07): the fallback title,
    else PR-10's stripped request."""

    from app.services import live_voice_protocol as live

    task = live.DelegatedTask(delegation_id="x", offset_ms=0, transcript=request_text)
    return live.fallback_task_title(task) or request_title(request_text, 60)


def _frames_for(client, did):
    return [f for f in client.of("delegation") if f.get("delegation_id") == did]


def _parents_of(client, needle) -> list:
    rows: dict = {}
    for frame in client.of("response_text"):
        row = rows.setdefault(
            frame.get("assistant_turn_id"),
            {"parent": frame.get("parent_user_turn_id"), "text": ""},
        )
        row["text"] += frame.get("text") or ""
    return [r["parent"] for r in rows.values() if needle in r["text"]]


async def _wait(pred, timeout=4.0):
    for _ in range(int(timeout / 0.02)):
        if pred():
            return True
        await asyncio.sleep(0.02)
    return False


def _model(box, *, voice_progress=True, answer_nudge=False, first=REQUEST, first_end=900):
    """A GPT-Live fake: voices the result commentary, voices relay progress
    lines only when `voice_progress`, and answers a result NUDGE only when
    `answer_nudge`. Every voicing is a fresh output epoch."""

    box.setdefault("t", 5000)
    box.setdefault("voiced", [])

    def voice(p, text):
        t = box["t"]
        box["t"] = t + 1500
        box["voiced"].append(text)
        p.push(H.out_text(text, t, t + 600))
        p.push(H.out_audio("SPEAKING"))

    def on_send(p, e):
        box["p"] = p
        kind = e["type"]
        content = str(e.get("content") or "")
        if kind == "session.start":
            p.push(H.user_delta(first, 0, first_end))
            p.push(H.delegation("d1", first_end + 50))
        elif kind == "session.close":
            p.push(H.closed())
        elif kind == "session.commentary.append":
            if ANSWER in content:
                voice(p, "Ada Lovelace works on robotics at Toronto.")
            elif voice_progress:
                voice(p, "Still at it: " + content[:40])
        elif kind == "session.instructions.append" and answer_nudge and ANSWER in content:
            voice(p, "Again: Ada Lovelace works on robotics at Toronto.")

    return on_send


def _tool(relay, phase, call_id, name):
    if phase == "start":
        return relay.on_event({
            "type": "tool.start", "call_id": call_id, "name": name,
            "args": {"query": "toronto robotics"},
        })
    return relay.on_event({
        "type": "tool.end", "call_id": call_id, "name": name,
        "ok": True, "preview": "found", "elapsed_ms": 20,
    })


def test_progress_defaults_follow_addendum_7():
    # Voice round 2026-10-07 (D-R4 (2)): ONE spoken line per task, two model
    # notes, none within 8 s of the expected finish.
    assert settings.voice_live_progress_spoken_max == 1
    assert settings.voice_live_progress_max_per_delegation == 2
    assert settings.voice_live_progress_min_remaining_s == 8.0
    assert settings.voice_live_progress_idle_after_s == 8.0
    assert settings.voice_live_progress_speak_after_s == 6.0


# ══════════════════════════════════════════════════════════════════════
# F14 — a pending progress claim never captures the result's epoch
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("answer_nudge", [False, True], ids=["ignores-nudge", "answers-nudge"])
async def test_a_result_finishing_behind_an_unvoiced_progress_line_is_heard_once(
    monkeypatch, answer_nudge,
):
    """The model accepts the progress line and does not voice it (or voices it
    late); the task finishes inside that window and the model voices the
    result once. It is recorded as heard, never nudged into a second telling,
    and stored once — on its delegated row, not also as a loose spoken row."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0, voice_live_speak_timeout_s=1.0)
    _set(monkeypatch, voice_live_progress_speak_after_s=0.2)
    counters = H.capture_counters(monkeypatch)
    box = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        await _tool(relay, "start", "c1", "web_search")
        await _wait(lambda: any(
            "working" in (e.get("content") or "").lower() for e in _commentary(box["p"])
        ), 3)
        await asyncio.sleep(0.1)
        await _tool(relay, "end", "c1", "web_search")
        return ANSWER, "m"

    rec = H.patch_relay(monkeypatch, think=think)
    client = H.FakeClient([H.config(), 4.0, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=_model(box, voice_progress=False, answer_nudge=answer_nudge), auto_ack=True,
    )
    await H.run_relay(client, provider, timeout=12, db_session_id=f"db-f14-{answer_nudge}")

    names = [name for name, _fields in counters]
    assert "live_progress_spoken" in names, "the progress line was never appended"
    d1 = _frames_for(client, "d1")
    assert not [f for f in d1 if f.get("spoken") is False], (
        f"a result the caller heard was recorded NOT heard; counters={names}"
    )
    assert "live_result_nudged" not in names
    assert [v for v in box["voiced"] if "Ada Lovelace" in v] == [
        "Ada Lovelace works on robotics at Toronto."
    ], box["voiced"]
    loose = [
        s for s in rec["saves"]
        if (s.get("assistant_voice") or {}).get("source") == "live_spoken"
        and "Ada Lovelace" in (s.get("assistant_text") or "")
    ]
    assert loose == [], "the heard answer was stored a second time as its own row"


# ══════════════════════════════════════════════════════════════════════
# Addendum 7 — a second line only on a real change of tool kind
# ══════════════════════════════════════════════════════════════════════

async def _run_tools(monkeypatch, steps, *, db, lang_request=REQUEST, spoken_max=None):
    """`steps`: [(tool_name, seconds it runs)], executed one after another.
    `spoken_max` overrides the per-task cap (production: ONE line since
    2026-10-07; the kind-change rule is exercised with the cap raised)."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    _set(monkeypatch, voice_live_progress_speak_after_s=0.25)
    if spoken_max is not None:
        _set(monkeypatch, voice_live_progress_spoken_max=spoken_max)
    box = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        for index, (name, seconds) in enumerate(steps):
            await _tool(relay, "start", f"c{index}", name)
            await asyncio.sleep(seconds)
            await _tool(relay, "end", f"c{index}", name)
        return ANSWER, "m"

    H.patch_relay(monkeypatch, think=think)
    total = sum(seconds for _name, seconds in steps) + 1.2
    client = H.FakeClient([H.config(), total, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_model(box, first=lang_request), auto_ack=True)
    await H.run_relay(client, provider, timeout=15, db_session_id=db)
    return client, provider


@pytest.mark.asyncio
async def test_a_second_line_follows_a_real_change_to_a_different_tool_kind(monkeypatch):
    # The cap is ONE in production (2026-10-07); the kind-change rule still
    # decides what a raised cap may say.
    client, provider = await _run_tools(
        monkeypatch, [("web_search", 0.7), ("web_fetch", 0.9)], db="db-prog-two",
        spoken_max=2,
    )
    lines = _progress_lines(provider)
    assert len(lines) == 2, lines
    assert relay_line("progress_step", "en", step="Searching the web") == lines[0]
    assert relay_line("progress_step", "en", step="Reading a page") == lines[1]
    turn = f"live-utt:{H.PSID}:1"
    assert set(_parents_of(client, "Still at it")) == {turn}


@pytest.mark.asyncio
async def test_the_same_kind_again_earns_no_second_line(monkeypatch):
    _client, provider = await _run_tools(
        monkeypatch, [("web_search", 0.7), ("web_search", 0.9)], db="db-prog-same",
        spoken_max=2,
    )
    assert len(_progress_lines(provider)) == 1


@pytest.mark.asyncio
async def test_never_more_than_two_lines_per_task(monkeypatch):
    _client, provider = await _run_tools(
        monkeypatch, [("web_search", 0.6), ("web_fetch", 0.7), ("read_file", 0.8)],
        db="db-prog-cap", spoken_max=2,
    )
    assert len(_progress_lines(provider)) == 2


# ══════════════════════════════════════════════════════════════════════
# Addendum 7 — one honest line for a toolless task
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize(("request_text", "lang"), [
    ("explain how large language models are trained", "en"),
    ("توضیح بده مدل‌های زبانی بزرگ چطور آموزش می‌بینن", "fa"),
])
async def test_a_toolless_task_gets_one_honest_line_naming_the_request(
    monkeypatch, caplog, request_text, lang,
):
    """The 5–16 s Deep turns with no tool said nothing at all. Now: one line,
    after the idle threshold, naming the request — never "searching".

    Phase D2 ruling 52: at PRODUCTION clocks (idle 8 s, min_remaining 8 s,
    cap 1 — `fast_clocks` pins min_remaining to 0, which is how this line's
    unreachability in production went unseen): a generic task (12 s
    expected) is near its finish from 4 s, so the idle line due at 8 s is
    withheld — and falls due once the task is OVERDUE, at 12 s.  (The English
    request is a question, which seeds the SEARCH plan — 20 s expected,
    window 12–20 s — so its idle line goes out at the 8 s threshold.)"""

    import logging
    import time

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")

    H.fast_clocks(
        monkeypatch, voice_live_delegation_ttl_s=3.0, voice_live_progress_min_remaining_s=8.0,
    )
    box = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        box["started"] = time.monotonic()
        await asyncio.sleep(0.3)
        box["early"] = list(_progress_lines(box["p"]))
        await asyncio.sleep(9.7)
        box["at_10s"] = list(_progress_lines(box["p"]))
        await asyncio.sleep(5.0)
        return ANSWER, "m"

    inner = _model(box, first=request_text)

    def on_send(p, e):
        if e["type"] == "session.commentary.append" and ANSWER not in str(e.get("content") or ""):
            box.setdefault("line_at", time.monotonic())
        inner(p, e)

    H.patch_relay(monkeypatch, think=think)
    client = H.FakeClient([H.config(), 16.5, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=30, db_session_id=f"db-prog-idle-{lang}")

    assert box.get("early") == [], "spoken before the idle threshold"
    messages = [r.getMessage() for r in caplog.records]
    kind = next(m.split(" kind=")[1].split()[0] for m in messages if "delegation plan id=d1" in m)
    assert (lang, kind) in {("en", "search"), ("fa", "generic")}, (lang, kind)
    line_s = box["line_at"] - box["started"]
    if kind == "generic":
        assert box.get("at_10s") == [], "spoken inside the near-finish window (4–12 s)"
        assert [m for m in messages if "progress suppressed id=d1 reason=near_finish" in m]
        assert 11.5 <= line_s <= 14.5, line_s
    else:
        assert 7.5 <= line_s <= 9.5, line_s
    lines = _progress_lines(provider)
    # Voice round 2026-10-07 (D-R4 (2)): «{title}» is the card's title — here
    # the deterministic fallback (the stub summarizer answers nothing) — never
    # the raw request read back.
    title = _card_title(request_text)
    assert title and title != request_text
    expected = relay_line("progress_idle", lang, title=title)
    assert lines == [expected], lines
    assert "Searching" not in lines[0] and "جستجو" not in lines[0]
    assert set(_parents_of(client, "Still at it")) == {f"live-utt:{H.PSID}:1"}
    # The answer still arrives, after the line.
    assert any(ANSWER in e["content"] for e in _commentary(provider))


@pytest.mark.asyncio
async def test_the_idle_line_waits_out_the_caller_and_is_never_the_delivery(monkeypatch):
    """Addendum 7: the toolless line is unprompted speech. It never starts
    while the caller is talking — it waits for the turn to end — and voicing
    it is not hearing the answer: here the model voices the idle line and
    never the result, so the result is recorded NOT heard."""

    H.fast_clocks(
        monkeypatch,
        voice_live_delegation_ttl_s=3.0,
        voice_live_utterance_gap_ms=400,
        voice_live_utterance_hard_gap_ms=800,
        voice_live_speak_timeout_s=0.6,
    )
    _set(
        monkeypatch,
        voice_live_progress_speak_after_s=0.1,
        voice_live_progress_idle_after_s=0.4,
    )
    request_text = "explain how large language models are trained"
    idle = relay_line("progress_idle", "en", title=_card_title(request_text))
    box = {"talking": False, "idle_appends": [], "t": 6000}
    started = asyncio.Event()

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        started.set()
        await asyncio.sleep(2.6)
        return ANSWER, "m"

    H.patch_relay(monkeypatch, think=think)

    def on_send(p, e):
        box["p"] = p
        kind = e["type"]
        content = str(e.get("content") or "")
        if kind == "session.start":
            p.push(H.user_delta(request_text, 0, 900))
            p.push(H.delegation("d1", 950))
        elif kind == "session.close":
            p.push(H.closed())
        elif kind == "session.commentary.append" and ANSWER not in content:
            # A relay progress line: the model voices it. The result it is
            # handed later is never voiced.
            box["idle_appends"].append((content, box["talking"]))
            p.push(H.out_text("Still at it.", box["t"], box["t"] + 500))
            p.push(H.out_audio("SPEAKING"))
            box["t"] += 1500

    async def script():
        await _wait(started.is_set)
        p = box["p"]
        # The caller talks right through the moment the idle line falls due.
        box["talking"] = True
        start = 3000
        for word in "so while you look at that I was also wondering about".split():
            p.push(H.user_delta(" " + word, start, start + 80))
            start += 100
            await asyncio.sleep(0.1)
        box["talking"] = False
        await _wait(
            lambda: any(
                f.get("phase") == "completed" for f in _frames_for(client, "d1")
            ),
            6.0,
        )
        await asyncio.sleep(0.2)

    client = H.FakeClient([H.config(), script, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=14, db_session_id="db-prog-idle-caller")

    assert [text for text, _talking in box["idle_appends"]] == [idle], box["idle_appends"]
    assert [talking for _text, talking in box["idle_appends"]] == [False], (
        "the idle line was spoken over the caller"
    )
    d1 = _frames_for(client, "d1")
    assert not [f for f in d1 if f.get("spoken") is True], (
        "the idle line's epoch was counted as hearing the answer"
    )
    assert [f for f in d1 if f.get("phase") == "completed" and f.get("spoken") is False]


@pytest.mark.asyncio
async def test_a_media_play_in_flight_earns_no_idle_line(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    _set(
        monkeypatch,
        voice_live_progress_speak_after_s=0.1,
        voice_live_progress_idle_after_s=0.3,
    )
    box = {}

    async def play(_u, _q, _variety=False):
        await asyncio.sleep(1.2)
        return "Starting Halo.", {"type": "youtube", "video_id": "v1", "title": "Halo"}

    H.patch_relay(monkeypatch, play=play)
    client = H.FakeClient([H.config(), 2.0, {"type": "stop"}])
    provider = H.FakeProvider(
        on_send=_model(box, first="play halo by beyonce", first_end=400), auto_ack=True,
    )
    await H.run_relay(client, provider, timeout=10, db_session_id="db-prog-media")

    lines = [e["content"] for e in _commentary(provider)]
    assert lines == [relay_line("media_playing", "en", title="Halo")], lines



# ══════════════════════════════════════════════════════════════════════
# Voice round 2026-10-07 (D-R4 (2)): ONE line per task, none once the answer
# is being written, none within 8 s of the expected finish
# ══════════════════════════════════════════════════════════════════════

def _suppressed(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "progress suppressed" in r.getMessage()]


@pytest.mark.asyncio
async def test_one_progress_line_per_task(monkeypatch, caplog):
    """The 10-07 call: four relay lines in five tasks, one of them 5 s before
    its result.  Search → reading a page used to earn a second line; at the
    production cap it earns none, and the cap's suppression is logged."""

    import logging

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    _client, provider = await _run_tools(
        monkeypatch, [("web_search", 0.7), ("web_fetch", 0.9)], db="db-prog-one",
    )
    lines = _progress_lines(provider)
    assert lines == [relay_line("progress_step", "en", step="Searching the web")], lines
    assert [m for m in _suppressed(caplog) if "reason=cap" in m], _suppressed(caplog)


@pytest.mark.asyncio
async def test_no_progress_line_after_the_answer_beat(monkeypatch, caplog):
    """The tenant's round-start answer beat (`step.event` reasoning, final)
    means the answer is being written: the toolless "still working on «…»"
    line it would otherwise earn is suppressed, and the suppression logged."""

    import logging

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    _set(
        monkeypatch,
        voice_live_progress_speak_after_s=0.1,
        voice_live_progress_idle_after_s=0.4,
    )
    box = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        await asyncio.sleep(0.1)
        await relay.on_event({
            "type": "step.event", "kind": "reasoning", "iteration": 1,
            "final": True, "provisional": True,
        })
        await asyncio.sleep(1.4)
        return ANSWER, "m"

    H.patch_relay(monkeypatch, think=think)
    request_text = "explain how large language models are trained"
    client = H.FakeClient([H.config(), 2.2, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_model(box, first=request_text), auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-prog-beat")

    assert _progress_lines(provider) == [], "a progress line over the answer being written"
    assert [m for m in _suppressed(caplog) if "reason=answer_started" in m], _suppressed(caplog)
    assert any(ANSWER in e["content"] for e in _commentary(provider))


@pytest.mark.asyncio
async def test_a_tool_after_a_provisional_beat_earns_the_line_again(monkeypatch):
    """A provisional beat is retracted by a later tool start: the task is
    working again, so the one tool line it may earn is not suppressed."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    _set(monkeypatch, voice_live_progress_speak_after_s=0.25)
    box = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        await relay.on_event({
            "type": "step.event", "kind": "reasoning", "iteration": 1,
            "final": True, "provisional": True,
        })
        await _tool(relay, "start", "c0", "web_search")
        await asyncio.sleep(0.8)
        await _tool(relay, "end", "c0", "web_search")
        return ANSWER, "m"

    H.patch_relay(monkeypatch, think=think)
    client = H.FakeClient([H.config(), 2.0, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_model(box), auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-prog-beat-tool")

    assert _progress_lines(provider) == [
        relay_line("progress_step", "en", step="Searching the web"),
    ]


def test_the_expected_durations_are_the_measured_table():
    from app.services import live_voice_protocol as live

    # Phase D2 ruling 52: a connector read is expected at 20 s (it was mapped
    # to generic's 12 s, whose window swallowed its 6–12 s tool line).
    assert live._PROGRESS_EXPECTED_S == {
        "search": 20.0, "media": 8.0, "connector": 20.0, "generic": 12.0,
    }
    assert live.progress_expected_total_s("connector") == 20.0
    assert live.plan_kind_of(["connector_read", "answer"]) == "connector"
    # A media task (8 s expected) is inside the 8 s window from its first
    # second until it is overdue: the play's own line is the outcome.
    assert live.progress_expected_total_s("media") - settings.voice_live_progress_min_remaining_s <= 0


@pytest.mark.parametrize(("kind", "tool", "quiet_s", "line_s"), [
    ("generic", False, 0.0, 12.0),    # the idle line: due at 8 s, near the finish until 12 s
    ("connector", True, 0.0, 6.0),    # its tool line at 6 s: before the 12–20 s window
    ("search", True, 7.0, 20.0),      # due at 13 s: near the finish — withheld until overdue
    ("search", True, 19.0, 25.0),     # due at 25 s: overdue — allowed
], ids=["generic_idle", "connector_tool", "search_13s", "search_25s"])
def test_the_one_progress_line_at_production_clocks(caplog, kind, tool, quiet_s, line_s):
    """Phase D2 ruling 52 (search-play-title-6), at PRODUCTION settings
    (speak_after 6, idle 8, min_remaining 8, cap 1 — nothing pinned to 0):
    `near_finish` held with no upper bound, so a generic or connector task
    could never speak its one line (generic's window opened at 4 s, before
    any line can fall due) and a search line first due after 20 s was
    silenced as "near" a finish already past.  Stepped on a clock: when does
    the ONE line go out?"""

    import logging
    from types import SimpleNamespace

    from app.services import live_voice_protocol as live

    assert (settings.voice_live_progress_speak_after_s, settings.voice_live_progress_idle_after_s,
            settings.voice_live_progress_min_remaining_s, settings.voice_live_progress_spoken_max) == (
        6.0, 8.0, 8.0, 1)
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    t0 = 1000.0
    session = live._LiveSession.__new__(live._LiveSession)
    session.settings, session.user_id = settings, "user-1"
    session.adapter = SimpleNamespace(_last_delta_monotonic=t0 + quiet_s)
    entry = live.RunningDelegation(
        task=live.DelegatedTask(delegation_id="p1", offset_ms=0, transcript="x"),
        started_monotonic=t0,
    )
    entry.plan_kind = kind
    if tool:
        entry.tool_started, entry.tool_kind = True, kind if kind == "search" else ""
        entry.active_tool_calls.add("c1")
    spoken_at = None
    for step in range(0, 30 * 4 + 1):
        now = t0 + step / 4
        due = session.progress_speech_due(entry)
        if not due or now < due:
            continue
        withheld = session.progress_suppression(entry, now)
        if withheld:
            session.note_progress_suppressed(entry, withheld)
            continue
        spoken_at = step / 4
        break
    assert spoken_at == line_s, (kind, spoken_at)
    near = [r.getMessage() for r in caplog.records if "progress suppressed id=p1 reason=near_finish" in r.getMessage()]
    assert bool(near) == (kind != "connector" and line_s != 25.0), near


@pytest.mark.asyncio
async def test_no_progress_line_within_eight_seconds_of_the_expected_finish(monkeypatch, caplog):
    """The line would fall due inside the window before the expected finish
    (scaled down here: every kind expected at 1.0 s, a 0.8 s window, the line
    due at 0.25 s): it is never spoken, and `elapsed=`/`expected=` are logged."""

    import logging

    from app.services import live_voice_protocol as live

    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    monkeypatch.setattr(
        live, "_PROGRESS_EXPECTED_S", {"search": 1.0, "media": 1.0, "generic": 1.0},
    )
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    _set(
        monkeypatch,
        voice_live_progress_speak_after_s=0.25,
        voice_live_progress_min_remaining_s=0.8,
    )
    box = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        await _tool(relay, "start", "c0", "web_search")
        await asyncio.sleep(0.9)
        await _tool(relay, "end", "c0", "web_search")
        return ANSWER, "m"

    H.patch_relay(monkeypatch, think=think)
    client = H.FakeClient([H.config(), 2.0, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_model(box), auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-prog-near")

    assert _progress_lines(provider) == []
    near = [m for m in _suppressed(caplog) if "reason=near_finish" in m]
    assert len(near) == 1, _suppressed(caplog)
    assert "expected=1.0" in near[0] and "elapsed=" in near[0]


# ── Phase C seam (DESIGN-R3 §0, ruling 6): the busy test reads the GROUP ──
# RA's merge appends two results as one group; the head can resolve while
# its companion is still owed.  A progress line spoken in that gap would sit
# between the two tellings and make the companion's epoch unattributable.

@pytest.mark.asyncio
async def test_no_progress_line_while_a_merged_companion_is_still_owed(monkeypatch):
    import time as _time
    from types import SimpleNamespace

    from app.services.live_voice_protocol import (
        DelegatedTask, LiveClientAdapter, ResultDelivery, _LiveSession,
    )

    session = _LiveSession.__new__(_LiveSession)
    session.settings = settings
    session.user_id = "user-1"
    session.adapter = LiveClientAdapter(session_id=H.PSID)
    session.utterances = None
    session.deferred_output = None
    session.delivery_queue = []
    session.results_owed = {}
    session.relay_claims = []
    for gate, value in (
        ("may_speak", True), ("interrupt_fence_up", False),
        ("user_is_mid_utterance", False), ("starved_relay_line", False),
    ):
        monkeypatch.setattr(session, gate, (lambda v: lambda *a, **k: v)(value))
    monkeypatch.setattr(session, "progress_speech_due", lambda entry: 1.0)
    monkeypatch.setattr(session, "progress_suppression", lambda entry, now: None)
    spoken: list = []

    async def speak(delegation_id, text, **kw):
        spoken.append((delegation_id, kw.get("progress")))
        return []

    monkeypatch.setattr(session, "speak", speak)
    task = DelegatedTask(delegation_id="p1", offset_ms=0, transcript="museum hours",
                         turn_id="u1", summary="Museum hours")
    entry = SimpleNamespace(
        ended=False, task=task, started_monotonic=_time.monotonic() - 10,
        progress_spoken=0, tool_started=False, progress_idle_spoken=False,
    )
    session.supervisor = SimpleNamespace(running={"p1": entry})

    def _delivery(did):
        return ResultDelivery(
            task=DelegatedTask(delegation_id=did, offset_ms=0, transcript=""),
            answer="x", claim={},
        )

    head, companion = _delivery("d1"), _delivery("d2")
    head.companions, companion.group_head = [companion], head
    head.resolved = True
    session.active_delivery = head

    await session.maybe_speak_progress(_time.monotonic())
    assert spoken == [], "a progress line went out between a merged pair's two tellings"

    companion.resolved = True
    await session.maybe_speak_progress(_time.monotonic())
    assert spoken == [("p1", True)], spoken


# ══════════════════════════════════════════════════════════════════════
# Phase D2 ruling 46 (delivery-ladder-11): the relay's own line is never
# "something born"
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["acked", "late_ack", "chained", "late_ack_chained", "stale"])
async def test_a_progress_line_voiced_after_the_result_is_not_the_model_saying_something_else(
    monkeypatch, caplog, mode,
):
    """The progress line was sent and not voiced; the result was appended;
    the model then voiced ONLY that line and went quiet.  The line's epoch was
    counted as "something born that never matched", so the one nudge was
    never sent and an answer the model never had a turn to say ended
    `unmatched_output` («نتیجه در گفتگو») — the relay's own speech cost the
    caller the retry.  Now the line is recognised and the nudge goes out
    (`relay_line_only=1`); the model answers it.

    `acked`: the line's late epoch binds the RESULT's parent claim — only its
    words say it is the line (ruling 46 (b)).  `late_ack`: the line was
    withdrawn unacked BEFORE the result existed and its ack came late — the
    re-filed claim binds its epoch (a).  `chained` (the control): the line
    runs straight on into a below-W paraphrase of the answer (more than two
    words beyond the line) — that IS output that did not tell it:
    `unmatched_output`, no nudge.  `stale` (the other control): the line was
    acked and withdrawn BEFORE the result existed — the same words after the
    result's append are not a line the relay has pending, they are the model
    talking: `unmatched_output`, no nudge.  `late_ack_chained` (Phase D3
    ruling 60, Phase D2 F2): the same chained telling on the LATE-ACK path —
    binding the line's claim made (a) wave the epoch through whatever else it
    carried, so the answer got the nudge and was told again; (a) now carries
    (b)'s "≤2 words outside the line" bound."""

    import logging

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    _set(monkeypatch, voice_live_progress_speak_after_s=0.2)
    if mode in ("late_ack", "late_ack_chained", "stale"):
        # The line is withdrawn (speak budget 0.5 s) before the result lands
        # at ~1.2 s — unacked (`late_ack`) or acked (`stale`).
        _set(monkeypatch, voice_live_speak_timeout_s=0.5)
    else:
        _set(monkeypatch, voice_live_speak_timeout_s=2.0, voice_live_relay_line_hold_ms=600)
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    box: dict = {"t": 5000}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        await _tool(relay, "start", "c1", "web_search")
        await _wait(lambda: box.get("line") is not None, 3)
        await asyncio.sleep(1.0 if mode in ("late_ack", "late_ack_chained", "stale") else 0.1)
        await _tool(relay, "end", "c1", "web_search")
        return ANSWER, "m"

    H.patch_relay(monkeypatch, think=think)

    def voice(p, text):
        t = box["t"]
        box["t"] = t + 1500
        p.push(H.out_text(text, t, t + 600))
        p.push(H.out_audio("SPEAKING"))

    def on_send(p, e):
        kind, content = e["type"], str(e.get("content") or "")
        if kind == "session.start":
            p.push(H.user_delta(REQUEST, 0, 900))
            p.push(H.delegation("d1", 950))
        elif kind == "session.close":
            p.push(H.closed())
        elif kind == "session.commentary.append":
            if ANSWER in content:
                async def late():
                    await asyncio.sleep(0.05)
                    if box.get("held_ack"):
                        p.push({"type": "session.commentary.appended",
                                "client_event_id": box["held_ack"], "start_ms": 0, "end_ms": 0})
                        await asyncio.sleep(0.05)
                    told = box["line"]
                    if mode.endswith("chained"):
                        told += " A well known professor there, I believe."
                    voice(p, told)
                asyncio.get_running_loop().create_task(late())
            elif box.get("line") is None:
                box["line"] = content
        elif kind == "session.instructions.append" and ANSWER in content:
            voice(p, "Ada Lovelace works on robotics at Toronto.")

    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    if mode.startswith("late_ack"):
        auto = provider._auto_respond

        def hold_the_lines_ack(event):
            content = str(event.get("content") or "")
            if (event.get("type") == "session.commentary.append" and ANSWER not in content
                    and box.get("line") is None):
                box["held_ack"] = str(event.get("event_id") or "")
                return
            auto(event)

        provider._auto_respond = hold_the_lines_ack
    client = H.FakeClient([H.config(), 6.0, {"type": "stop"}])
    await H.run_relay(client, provider, timeout=14, db_session_id=f"db-r46-{mode}")

    messages = [r.getMessage() for r in caplog.records]
    nudges = [m for m in messages if "result nudge sent id=d1" in m]
    unspoken = [m for m in messages if "result NOT spoken id=d1" in m]
    spoken = [m for m in messages if "result spoken id=d1" in m]
    if mode in ("chained", "late_ack_chained", "stale"):
        if mode == "late_ack_chained":
            # The path under test: the late ack re-filed the line's claim.
            assert [m for m in messages if "relay line consumed late" in m], messages
        assert nudges == [], nudges
        assert unspoken and "reason=unmatched_output" in unspoken[0], messages
        return
    if mode == "late_ack":
        assert [m for m in messages if "relay line consumed late" in m], messages
    assert nudges == ["[LIVE] result nudge sent id=d1 reason=no_epoch relay_line_only=1"], messages
    assert unspoken == [], unspoken
    assert spoken and "reason=words " in spoken[0], spoken
    assert not [f for f in _frames_for(client, "d1") if f.get("spoken") is False]


# Phase D4 ruling 65 (Phase D3 review F3): a relay line filed WITHOUT text.
_TEXTLESS_ASK = "سرچ کن ببین بهترین کتاب برای جذب سرمایه چیه"
_TEXTLESS_ANSWER = (
    "بهترین کتاب برای جذب سرمایه، استارتاپ ناب نوشته اریک ریس است؛ روش آزمون سریع ایده را یاد می‌دهد."
)
_TEXTLESS_LINE_SAID = "باشه، دارم توی اینترنت می‌گردم ببینم بهترین کتاب در این زمینه کدومه."


@pytest.mark.asyncio
async def test_a_late_relay_line_filed_without_text_voiced_alone_still_gets_the_nudge(monkeypatch, caplog):
    """Ruling 60 gave path (a) of ruling 46 the "≤2 words outside the line"
    bound, against the words `note_relay_line` recorded for the line — but
    only `speak()` files a line WITH its words.  The instruction lines
    (`research_started`, the status answers, the confirmations, the reply
    backstop) are filed with `text=""`, so every content word the model
    voiced for one was "outside the line": the late-acked «I'm looking it
    up» epoch was "something born", and an answer the model never had a turn
    to say ended `unmatched_output` with no nudge («نتیجه در گفتگو») —
    ruling 46's defect, back for exactly the relay-opened research the
    founder's General/Academic asks open.  A line with no recorded words
    keeps ruling 46's answer on (a); the bound applies where the relay knows
    the line's words (`late_ack_chained` above is unchanged).

    The research backstop opens a search the model did not delegate and
    sends its `research_started` instruction; its ack is held past the speak
    budget, so the claim is withdrawn unacked; the result is delivered; the
    ack lands late and the model voices ONLY that line (Phase D3 review F3's
    reproduction, `r60/test_r60_textless_line.py`)."""

    import logging

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    _set(
        monkeypatch, voice_live_research_backstop_enabled=True,
        voice_live_research_backstop_grace_ms=250, voice_live_speak_timeout_s=0.5,
    )
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        await asyncio.sleep(1.2)
        return _TEXTLESS_ANSWER, "m"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {"t": 5000, "answers": 0}

    def voice(p, text):
        t = box["t"]
        box["t"] = t + 1500
        p.push(H.out_text(text, t, t + 600))
        p.push(H.out_audio("SPEAKING"))

    def on_send(p, e):
        kind, content = e["type"], str(e.get("content") or "")
        if kind == "session.start":
            p.push(H.user_delta(_TEXTLESS_ASK, 0, 900))
        elif kind == "session.close":
            p.push(H.closed())
        elif kind == "session.instructions.append" and _TEXTLESS_ANSWER in content:
            box["answers"] += 1
            if box["answers"] == 1:              # the result's delivery
                async def late():
                    await asyncio.sleep(0.05)
                    if box.get("held_ack"):
                        p.push({"type": "session.instructions.appended",
                                "client_event_id": box["held_ack"], "start_ms": 0, "end_ms": 0})
                        await asyncio.sleep(0.05)
                    voice(p, _TEXTLESS_LINE_SAID)   # ONLY the "started" line
                asyncio.get_running_loop().create_task(late())
            else:                                # the nudge: now it tells the answer
                voice(p, _TEXTLESS_ANSWER)

    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    auto = provider._auto_respond

    def hold_the_lines_ack(event):
        if (event.get("type") == "session.instructions.append"
                and str(event.get("event_id") or "").startswith("toup-live-research-")):
            box["held_ack"] = str(event.get("event_id") or "")
            return
        auto(event)

    provider._auto_respond = hold_the_lines_ack
    client = H.FakeClient([H.config(), 6.0, {"type": "stop"}])
    await H.run_relay(client, provider, timeout=14, db_session_id="db-r65-textless-line")

    messages = [r.getMessage() for r in caplog.records]
    assert box.get("held_ack"), "the research_started line was never sent"
    assert [m for m in messages if "research backstop fired" in m], messages
    assert [m for m in messages if "relay line claim withdrawn reason=no_epoch" in m], messages
    # The path under test: the late ack re-filed the line's (text-less) claim.
    assert [m for m in messages if "relay line consumed late" in m], messages
    nudges = [m for m in messages if "result nudge sent id=relay-resear" in m]
    assert nudges == ["[LIVE] result nudge sent id=relay-resear reason=no_epoch relay_line_only=1"], messages
    spoken = [m for m in messages if "result spoken id=relay-resear" in m]
    assert spoken and "reason=words " in spoken[0], messages
    assert not [m for m in messages if "reason=unmatched_output" in m], messages
