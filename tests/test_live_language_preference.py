"""Reply-language preference (v0.3 §G, feature `language_pref`).

Before this, the Live path had no language preference at all: a static
turn-by-turn mirroring rule, a URL `lang` pin the Live branch never passed on,
and a per-task language fixed by the SCRIPT of the words at dispatch. The V3
recording is the result: the caller asked in Persian to be spoken to in English
only, the model agreed in English, and the next request — spoken in Persian —
was dispatched `lang=fa` with a Persian directive, its relay lines and nudges
were Persian, and nothing remembered the request across a reconnect.

The preference comes from three sources: an explicit spoken request matched by
a closed, anchored grammar; the URL pin (`run_live_voice_session(language_pin=…)`);
and the app's optional `language_preference` frame. Once set it holds for this
DB session (in-process, bounded), outranks turn-by-turn mirroring, and every
live, queued and detached task follows it for delivery.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

import test_live_harness as H


async def _wait_until(predicate, *, timeout=3.0, label="condition"):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError(f"timed out waiting for {label}")
        await asyncio.sleep(0.01)


def _finals(client) -> list[dict]:
    return [f for f in client.of("transcript") if f.get("final")]


def _config_with(*features, **extra):
    frame = H.config(**extra)
    frame["features"] = list(frame["features"]) + list(features)
    return frame


PERSIAN_TASK = "یک استاد خوب در دانشگاه تورنتو پیدا کن"
V3_REQUEST = "می‌تونی باهام انگلیسی حرف بزنی فقط"


# ── the grammar ───────────────────────────────────────────────────────

@pytest.mark.parametrize("text, expected", [
    # The V3 words, and the shapes the spec names.
    (V3_REQUEST, "en"),
    ("می‌تونی باهام فقط انگلیسی حرف بزنی", "en"),
    ("باهام انگلیسی حرف بزن", "en"),
    ("فقط انگلیسی", "en"),
    ("لطفا فارسی جواب بده", "fa"),
    ("از این به بعد فارسی صحبت کن", "fa"),
    ("speak English", "en"),
    ("English please", "en"),
    ("Can you speak English with me?", "en"),
    ("reply in Persian", "fa"),
    ("Reply in Farsi.", "fa"),
    ("I want you to talk to me in English from now on", "en"),
    ("Only English, please.", "en"),
    # Not requests: a bare name, a question about a language, a negation, a
    # task that mentions a language, two languages, a statement.
    ("English", None),
    ("in English", None),
    ("انگلیسی", None),
    ("don't speak English", None),
    ("انگلیسی حرف نزن", None),
    ("چرا انگلیسی حرف می‌زنی؟", None),
    ("translate this to English please", None),
    ("speak English and find me a professor", None),
    ("speak English or Persian", None),
    ("I speak English", None),
    ("من انگلیسی حرف می‌زنم", None),
    ("what is the best professor in UofT working on LLM", None),
    (PERSIAN_TASK, None),
    ("", None),
])
def test_an_explicit_language_request_is_a_closed_anchored_grammar(text, expected):
    from app.services.live_voice_protocol import detect_language_request

    assert detect_language_request(text) == expected


# ── V3, end to end ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_v3_a_persian_task_after_english_only_is_delivered_in_english(monkeypatch):
    """Persian history → «می‌تونی باهام انگلیسی حرف بزنی فقط» → the model agrees
    in English → a Persian-script request. The request is dispatched in
    English, and when the model does not voice the result the relay's own
    nudge is English too."""

    H.fast_clocks(
        monkeypatch, voice_live_delegation_ttl_s=2.0, voice_live_history_rows=20,
        voice_live_speak_timeout_s=0.3,
    )
    calls = []

    async def think(_user_id, task, _session_id, **kwargs):
        calls.append((task, kwargs))
        return "Professor A works on LLMs.", "test-model"

    async def vps(_user_id):
        return ("http://agent.local", "key")

    async def vps_api(*_args, **_kwargs):
        return [
            {"role": "user", "content": "سلام، امروز هوا چطوره؟"},
            {"role": "assistant", "content": "سلام! امروز آفتابیه."},
        ]

    from app.api import ws_realtime as rt
    H.patch_relay(monkeypatch, think=think, vps=vps)
    monkeypatch.setattr(rt, "_vps_api", vps_api)
    box = {}

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("سلام، چه خبر؟", 0, 400))
            provider.push(H.out_text("سلام! خوبم.", 600, 900))
            provider.push(H.user_delta(V3_REQUEST, 1500, 2100))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def conversation():
        await _wait_until(lambda: len(_finals(client)) >= 2, label="the English-only request")
        box["p"].push(H.out_text("Sure. I'll speak only English with you.", 2300, 2900))
        await asyncio.sleep(0.2)
        box["p"].push(H.user_delta(PERSIAN_TASK, 4000, 4600))
        await asyncio.sleep(0.15)
        box["p"].push(H.delegation("d-fa-after-en", 4700))
        await _wait_until(
            lambda: any(f.get("phase") == "completed" for f in client.of("delegation")),
            timeout=4.0, label="delegation done",
        )
        await _wait_until(
            lambda: any("toup-live-nudge-" in str(e.get("event_id"))
                        for e in box["p"].of("session.instructions.append")),
            timeout=3.0, label="result nudge",
        )

    client = H.FakeClient([
        _config_with("language_pref"), conversation, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=12, db_session_id="db-lang-v3")

    assert provider.of("session.start")[0]["session"]["input"], "Persian history was seeded"
    assert len(calls) == 1, calls
    task, kwargs = calls[0]
    assert PERSIAN_TASK in task
    assert kwargs.get("reply_language") == "en"
    assert "Persian/Farsi" not in task
    assert "Reply in English" in task
    # The one relay-authored model instruction about the result is English.
    nudges = [e for e in provider.of("session.instructions.append")
              if "toup-live-nudge-" in str(e.get("event_id"))]
    assert nudges and nudges[0]["content"].startswith("The backend finished"), nudges
    # One tracked instruction recorded the preference; the app was told.
    preference = [e for e in provider.of("session.instructions.append")
                  if "toup-live-language-" in str(e.get("event_id"))]
    assert len(preference) == 1
    assert "English" in preference[0]["content"]
    assert client.of("language") == [{"type": "language", "lang": "en", "source": "explicit"}]


# ── the URL pin ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_url_language_pin_reaches_live_dispatch(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    calls = []

    async def think(_user_id, task, _session_id, **kwargs):
        calls.append((task, kwargs))
        return "Done.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta(PERSIAN_TASK, 0, 600))
            provider.push(H.delegation("d-pinned", 650))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def done():
        await _wait_until(
            lambda: any(f.get("phase") == "completed" for f in client.of("delegation")),
            timeout=4.0, label="delegation done",
        )

    client = H.FakeClient([_config_with("language_pref"), done, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(
        client, provider, timeout=8, db_session_id="db-lang-pin", language_pin="en",
    )

    assert calls and calls[0][1].get("reply_language") == "en"
    assert "Persian/Farsi" not in calls[0][0]
    assert client.of("language") == [{"type": "language", "lang": "en", "source": "pin"}]


def test_ws_realtime_passes_the_lang_pin_into_the_live_branch():
    """Source probe: the Live branch returns hundreds of lines before
    `_pinned_lang` is computed for Realtime, so the pin has to be handed over
    at the call itself."""

    import pathlib

    src = pathlib.Path("app/api/ws_realtime.py").read_text()
    call = src.split("await run_live_voice_session(", 1)
    assert len(call) == 2, "marker gone, rewrite this probe"
    args = call[1].split(")\n", 1)[0]
    assert "language_pin=lang if lang in _VOICE_LANG_PINS else None" in args


# ── the app frame ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_app_frame_sets_and_auto_clears_the_preference(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    calls = []

    async def think(_user_id, task, _session_id, **kwargs):
        calls.append(kwargs.get("reply_language"))
        return "Done.", "test-model"

    H.patch_relay(monkeypatch, think=think)
    box = {}

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.close":
            provider.push(H.closed())

    def task(index, start):
        async def _go():
            box["p"].push(H.user_delta(PERSIAN_TASK, start, start + 600))
            await asyncio.sleep(0.15)
            box["p"].push(H.delegation(f"d-{index}", start + 700))
            await _wait_until(
                lambda: len([f for f in client.of("delegation")
                             if f.get("phase") == "completed"]) >= index,
                timeout=4.0, label=f"delegation {index}",
            )
        return _go

    client = H.FakeClient([
        _config_with("language_pref"),
        0.05,
        {"type": "language_preference", "lang": "en"},
        0.05,
        task(1, 0),
        {"type": "language_preference", "lang": "auto"},
        0.05,
        task(2, 3000),
        {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-lang-frame")

    assert calls == ["en", "fa"], calls
    # Set is echoed with its source; a clear came FROM the app, so it is not.
    assert client.of("language") == [{"type": "language", "lang": "en", "source": "client"}]


@pytest.mark.asyncio
async def test_a_running_task_follows_a_preference_set_while_it_runs(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)

    async def think(_user_id, _task, _session_id, **_kwargs):
        await asyncio.sleep(0.6)
        return "", "test-model"                     # empty → the relay's own line

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta(PERSIAN_TASK, 0, 600))
            provider.push(H.delegation("d-running", 650))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def running():
        await _wait_until(
            lambda: any(f.get("state") == "running" for f in client.of("delegation")),
            label="running",
        )

    async def finished():
        await _wait_until(
            lambda: any(f.get("phase") == "failed" for f in client.of("delegation")),
            timeout=4.0, label="finished empty",
        )

    client = H.FakeClient([
        H.config(), running, {"type": "language_preference", "lang": "en"},
        finished, 0.1, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=8, db_session_id="db-lang-running")

    lines = [e["content"] for e in provider.of("session.commentary.append")]
    assert "I didn't get an answer back for that. Say it again?" in lines, lines
    assert not any("جوابی برنگشت" in line for line in lines), lines


# ── persistence across sockets ───────────────────────────────────────


@pytest.mark.asyncio
async def test_a_detached_result_is_announced_in_the_current_preference(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)

    async def think(_user_id, _task, _session_id, **_kwargs):
        await asyncio.sleep(0.5)
        return "Professor A works on LLMs.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    def first_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta(PERSIAN_TASK, 0, 600))
            provider.push(H.delegation("d-detached", 650))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def running():
        await _wait_until(
            lambda: any(f.get("state") == "running" for f in first.of("delegation")),
            label="running",
        )

    first = H.FakeClient([
        H.config(), running, {"type": "language_preference", "lang": "en"}, 0.05,
        {"type": "stop"},                                  # hang up mid-task
    ])
    await H.run_relay(first, H.FakeProvider(on_send=first_send, auto_ack=True),
                      timeout=8, db_session_id="db-lang-detached")

    def second_send(provider, event):
        if event["type"] == "session.close":
            provider.push(H.closed())

    second = H.FakeClient([_config_with("language_pref"), 0.1, {"type": "stop"}])
    provider = H.FakeProvider(on_send=second_send, auto_ack=True)
    await H.run_relay(second, provider, timeout=6, db_session_id="db-lang-detached")

    notes = [e["content"] for e in provider.of("session.thinking.append")
             if "Professor A" in str(e.get("content"))]
    assert notes and notes[0].startswith("A task the caller started earlier finished"), notes
    # …and the preference itself came back with the DB session.
    assert second.of("language") == [{"type": "language", "lang": "en", "source": "client"}]


@pytest.mark.asyncio
async def test_a_spoken_preference_survives_a_reconnect_and_outranks_the_pin(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)
    calls: dict[str, list[str]] = {}

    async def think(_user_id, _task, session_id, **kwargs):
        calls.setdefault(session_id, []).append(kwargs.get("reply_language"))
        return "Done.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    def asks_for_english(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("speak English please", 0, 500))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    first = H.FakeClient([H.config(), 0.3, {"type": "stop"}])
    await H.run_relay(first, H.FakeProvider(on_send=asks_for_english, auto_ack=True),
                      timeout=6, db_session_id="db-lang-reconnect")

    def persian_task(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta(PERSIAN_TASK, 0, 600))
            provider.push(H.delegation("d-after-reconnect", 650))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    for db_session_id, pin in (
        ("db-lang-reconnect", "fa"),     # same conversation, an old pin on the URL
        ("db-lang-elsewhere", None),     # a different conversation
    ):
        client = H.FakeClient([H.config()])

        async def done(client=client):
            await _wait_until(
                lambda: any(f.get("phase") == "completed" for f in client.of("delegation")),
                timeout=4.0, label="delegation done",
            )

        client._script.extend([done, {"type": "stop"}])
        await H.run_relay(
            client, H.FakeProvider(on_send=persian_task, auto_ack=True), timeout=8,
            db_session_id=db_session_id, language_pin=pin,
        )

    assert calls["db-lang-reconnect"] == ["en"], calls
    assert calls["db-lang-elsewhere"] == ["fa"], calls


# ── negotiation and logs ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_frames_are_negotiated_and_the_log_names_only_codes(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="app.services.live_voice_protocol")
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta(V3_REQUEST, 0, 600))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    legacy = H.FakeClient([H.config(), 0.3, {"type": "stop"}])
    await H.run_relay(legacy, H.FakeProvider(on_send=on_send, auto_ack=True),
                      timeout=6, db_session_id="db-lang-legacy")

    ready = legacy.of("ready")[0]["capabilities"]
    assert ready.get("language_pref") is True and ready.get("reask_turns") is True
    # A v0.2 client never negotiated the frame and never receives it.
    assert legacy.of("language") == []
    messages = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[LIVE]")]
    assert any(
        m.startswith("[LIVE] language preference set") and "lang=en" in m
        and "source=explicit" in m
        for m in messages
    ), messages
    for message in messages:
        assert V3_REQUEST not in message and "انگلیسی" not in message, message
