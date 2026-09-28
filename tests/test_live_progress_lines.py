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
    assert settings.voice_live_progress_spoken_max == 2
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

async def _run_tools(monkeypatch, steps, *, db, lang_request=REQUEST):
    """`steps`: [(tool_name, seconds it runs)], executed one after another."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    _set(monkeypatch, voice_live_progress_speak_after_s=0.25)
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
    client, provider = await _run_tools(
        monkeypatch, [("web_search", 0.7), ("web_fetch", 0.9)], db="db-prog-two",
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
    )
    assert len(_progress_lines(provider)) == 1


@pytest.mark.asyncio
async def test_never_more_than_two_lines_per_task(monkeypatch):
    _client, provider = await _run_tools(
        monkeypatch, [("web_search", 0.6), ("web_fetch", 0.7), ("read_file", 0.8)],
        db="db-prog-cap",
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
    monkeypatch, request_text, lang,
):
    """The 5–16 s Deep turns with no tool said nothing at all. Now: one line,
    after the idle threshold, naming the request — never "searching"."""

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    _set(
        monkeypatch,
        voice_live_progress_speak_after_s=0.1,
        voice_live_progress_idle_after_s=0.5,
    )
    box = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        await asyncio.sleep(0.3)
        box["early"] = list(_progress_lines(box["p"]))
        await asyncio.sleep(1.8)
        return ANSWER, "m"

    H.patch_relay(monkeypatch, think=think)
    client = H.FakeClient([H.config(), 3.0, {"type": "stop"}])
    provider = H.FakeProvider(on_send=_model(box, first=request_text), auto_ack=True)
    await H.run_relay(client, provider, timeout=12, db_session_id=f"db-prog-idle-{lang}")

    assert box.get("early") == [], "spoken before the idle threshold"
    lines = _progress_lines(provider)
    expected = relay_line("progress_idle", lang, title=request_title(request_text, 60))
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
    idle = relay_line("progress_idle", "en", title=request_title(request_text, 60))
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
