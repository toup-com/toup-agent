"""Fix round for §F (reask with identity) and §G (reply-language preference).

Every test here is a verifier's reproduction from the R2 adversarial review,
kept as a permanent regression test (fix-round findings F16, F17, F19–F24,
F44, F45):

* F16 — an accepted reask answered by a delegation WITHOUT speech first used
  to expire `no_causal_turn`: the dropped reply's fence was the last word.
* F17 — «؟ ، ؛» are inside the block the tokenizer keeps as word characters,
  so the ordinary Persian question «…انگلیسی حرف بزنی؟» was never a request.
* F19 — a switch back that names the language being left ("speak Persian, not
  English") was declined, and nothing spoken returned to mirroring.
* F20 — the grammar judged each closed turn alone, so "Reply in English" +
  pause + "to the email from Sara" locked the call to English.
* F21 — «به انگلیسی بگو» ("say it in English") switched the whole call while
  the English "say it in English" did not.
* F22 — (relay half) a repair that lands on a process with no memory of the
  preference must take the app's re-stated `language_preference`.
* F23 — a task already running when the caller asked for English was voiced
  from its Persian answer text with nothing telling the model the language.
* F24 — an un-rolled tenant still renders "Persian in, Persian out — every
  time" next to the relay's rule.
* F44 — TF132's replay of a question stranded by a socket drop was dropped
  silently by the new relay.
* F45 — an explicit preference outlived its call and answered a later call of
  the same day in the wrong language.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
from fastapi import WebSocketDisconnect

import test_live_harness as H


PERSIAN_TASK = "یک استاد خوب در دانشگاه تورنتو پیدا کن"
PERSIAN_ANSWER = "استاد الف در دانشگاه تورنتو روی مدل‌های زبانی کار می‌کند."
QUESTION = "find me a good LLM professor at UofT"
TF132 = [
    "delegation_frames", "heard_text", "live_turns", "media_control",
    "playback_frames", "task_lifecycle", "turn_timing",
]
REASK_FEATURES = [
    "live_turns", "state_seq", "outcomes", "delegation_frames",
    "task_lifecycle", "playback_frames", "turn_timing", "media_control",
    "reask_turns",
]


async def _until(predicate, *, timeout=4.0, label="condition"):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise AssertionError(f"timed out waiting for {label}")
        await asyncio.sleep(0.01)


def _finals(client) -> list[dict]:
    return [f for f in client.of("transcript") if f.get("final")]


def _completed(client) -> int:
    return sum(1 for f in client.of("delegation") if f.get("phase") == "completed")


def _config_with(*features):
    frame = H.config()
    frame["features"] = list(frame["features"]) + list(features)
    return frame


def _language_notes(provider) -> list[str]:
    return [
        e["content"] for e in provider.of("session.instructions.append")
        if "toup-live-language-" in str(e.get("event_id"))
    ]


@pytest.fixture(autouse=True)
def _fresh_process_stores():
    """Every test is its own process as far as the per-DB-session stores go."""

    import app.services.live_voice_protocol as live

    stores = (
        live._LANGUAGE_PREF_BY_SESSION, live._LANGUAGE_PREF_LRU,
        getattr(live, "_LANGUAGE_PREF_RELEASED", {}),
        getattr(live, "_LANGUAGE_PREF_OWNER", {}),
        getattr(live, "_STRANDED_BY_SESSION", {}),
    )
    for store in stores:
        store.clear()
    yield
    for store in stores:
        store.clear()


# ── F17 · F19 · F21: the grammar ──────────────────────────────────────


@pytest.mark.parametrize("text, expected", [
    # F17: Arabic-script punctuation is a boundary, not part of a word.
    ("می‌تونی باهام انگلیسی حرف بزنی؟", "en"),
    ("می‌تونی انگلیسی حرف بزنی؟", "en"),            # the production fragment, joined
    ("میشه انگلیسی حرف بزنی؟", "en"),
    ("انگلیسی حرف بزن، لطفا", "en"),
    ("باشه، فقط فارسی", "fa"),
    ("لطفا فارسی جواب بده؛", "fa"),
    ("لطفا؛ فارسی جواب بده", "fa"),
    ("باشه، انگلیسی حرف بزن", "en"),
    ("انگلیسی؟", None),
    ("نه، انگلیسی حرف نزن", None),
    ("انگلیسی حرف نزن؟", None),
    ("چرا انگلیسی؟", None),
    ("انگلیسی یا فارسی؟", None),
    ("من انگلیسی حرف می‌زنم.", None),
    # F19: a switch back may name the language it leaves.
    ("speak Persian, not English", "fa"),
    ("فارسی حرف بزن نه انگلیسی", "fa"),
    ("دیگه انگلیسی حرف نزن، فارسی حرف بزن", "fa"),
    ("به جای انگلیسی فارسی حرف بزن", "fa"),
    ("instead of English, answer in Farsi", "fa"),
    ("stop speaking English, speak Persian", "fa"),
    ("no English, speak Persian", "fa"),
    ("دوباره فارسی حرف بزن", "fa"),
    ("go back to Persian", "fa"),
    ("let's go back to Farsi", "fa"),
    ("برگرد به فارسی", "fa"),
    ("نه انگلیسی حرف بزن", "en"),                   # «نه،» the interjection
    ("لطفا، انگلیسی حرف بزن", "en"),
    ("not English", None),
    ("no English, no Persian", None),
    ("no more English", None),                      # a release, not a request
    # F21: a SAY verb is a one-off in both languages …
    ("به انگلیسی بگو", None),
    ("انگلیسی بگو", None),
    ("فارسی بگو", None),
    ("میشه انگلیسی بگی", None),
    ("برام انگلیسی بگو", None),
    ("لطفا انگلیسی بگو", None),
    ("say it in English", None),
    ("tell me in English", None),
    # … unless the caller says it is for good.
    ("فقط انگلیسی بگو", "en"),
    ("از این به بعد فقط فارسی بگو", "fa"),
])
def test_the_grammar_handles_punctuation_switch_back_and_one_off_requests(text, expected):
    from app.services.live_voice_protocol import detect_language_request

    assert detect_language_request(text) == expected


@pytest.mark.parametrize("text, current, expected", [
    ("no more English", "en", True),
    ("دیگه انگلیسی نه", "en", True),
    ("don't speak English", "en", True),
    ("انگلیسی حرف نزن", "en", True),
    ("stop speaking English", "en", True),
    ("reply in the language I speak", "en", True),
    ("answer in whatever language I speak", "fa", True),
    ("match my language", "en", True),
    ("answer me in the same language as me", "en", True),
    ("به همون زبونی که حرف می‌زنم جواب بده", "en", True),
    # Nothing to release, or not the language in force, or not a request.
    ("no more English", "fa", False),
    ("no more English", "", False),
    ("reply in the language I speak", "", False),
    ("فارسی حرف نزن", "en", False),
    ("I speak English", "en", False),
    ("what language do I speak", "en", False),
])
def test_a_spoken_release_returns_to_mirroring(text, current, expected):
    from app.services.live_voice_protocol import detect_language_release

    assert detect_language_release(text, current) is expected


# ── shared end-to-end scenario ────────────────────────────────────────


async def _call(monkeypatch, steps, *, db, features=("language_pref",), think_answer="Done.",
                think_delay=0.0, **clocks):
    """One call: `steps` are (kind, payload) run in order.

    kinds: "say" (text, start, end) — a caller turn, waited to its final;
    "reply" (text, start, end) — the model speaks; "task" (text, start, end) —
    a caller turn the model delegates; "frame" dict — the app sends it.
    Returns (client, provider, think calls).
    """

    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0, **clocks)
    calls: list[dict] = []

    async def think(_user_id, task, _session_id, **kwargs):
        calls.append({"task": task, **kwargs})
        if think_delay:
            await asyncio.sleep(think_delay)
        return think_answer, "test-model"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.close":
            provider.push(H.closed())

    def step(index, kind, payload):
        # Each step is its own client-script item, so an app frame is READ
        # between the caller's turns, in order, not after the whole script.
        if kind == "frame":
            return payload
        if kind == "wait":
            return payload

        async def run():
            p = box["p"]
            if kind in {"say", "task"}:
                text, start, end = payload
                seen = len(_finals(client))
                p.push(H.user_delta(text, start, end))
                await _until(lambda: len(_finals(client)) > seen, label=f"final {text!r}")
                if kind == "task":
                    done = _completed(client)
                    p.push(H.delegation(f"d-{db}-{index}", end + 100))
                    await _until(lambda: _completed(client) > done, label=f"task {text!r}")
            elif kind == "reply":
                text, start, end = payload
                # A separate response: the previous output epoch retires first.
                await asyncio.sleep(0.15)
                p.push(H.out_text(text, start, end))
                p.push(H.out_audio())
                await asyncio.sleep(0.4)
        return run

    items = [0.05] + [step(i, kind, payload) for i, (kind, payload) in enumerate(steps)]
    client = H.FakeClient([_config_with(*features), *items, 0.2, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=20, db_session_id=db)
    return client, provider, calls


# ── F17 end to end ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_v3_asked_as_a_persian_question_sets_english(monkeypatch):
    client, provider, calls = await _call(monkeypatch, [
        ("say", ("می‌تونی باهام انگلیسی حرف بزنی؟", 0, 900)),
        ("reply", ("Sure, I'll speak English with you.", 1100, 1800)),
        ("task", (PERSIAN_TASK, 4000, 4600)),
    ], db="db-f17-question")

    assert client.of("language") == [{"type": "language", "lang": "en", "source": "explicit"}]
    assert [c.get("reply_language") for c in calls] == ["en"]
    assert "Persian/Farsi" not in calls[0]["task"]


# ── F19 end to end ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_switching_back_while_naming_english_dispatches_in_persian(monkeypatch):
    client, provider, calls = await _call(monkeypatch, [
        ("say", ("speak English please", 0, 700)),
        ("reply", ("Sure.", 900, 1200)),
        ("say", ("فارسی حرف بزن نه انگلیسی", 3000, 3900)),
        ("reply", ("باشه.", 4100, 4400)),
        ("task", (PERSIAN_TASK, 6000, 6600)),
    ], db="db-f19-switch-back")

    assert [f["lang"] for f in client.of("language")] == ["en", "fa"]
    assert [c.get("reply_language") for c in calls] == ["fa"]
    assert "Reply in English" not in calls[0]["task"]


@pytest.mark.asyncio
async def test_a_spoken_clear_returns_the_call_to_mirroring(monkeypatch):
    client, provider, calls = await _call(monkeypatch, [
        ("say", ("speak English please", 0, 700)),
        ("reply", ("Sure.", 900, 1200)),
        ("say", ("reply in the language I speak", 3000, 3900)),
        ("reply", ("Okay.", 4100, 4400)),
        ("task", (PERSIAN_TASK, 6000, 6600)),
    ], db="db-f19-clear")

    # The app is told, so it stops re-stating English after a repair.
    assert [f["lang"] for f in client.of("language")] == ["en", "auto"]
    assert [c.get("reply_language") for c in calls] == ["fa"]
    notes = _language_notes(provider)
    assert len(notes) == 2 and notes[1].startswith("The caller no longer asks"), notes


# ── F20: a request is the span, not the turn ──────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("fragments", [
    [("Reply in English", 0, 900), (" to the email from Sara about the Monday meeting", 2400, 4600)],
    [("به ایمیل سارا درباره جلسه دوشنبه", 0, 1800), (" جواب انگلیسی بده", 3300, 4300)],
], ids=["english-head", "persian-tail"])
async def test_a_language_phrase_inside_one_task_does_not_lock_the_call(monkeypatch, fragments):
    steps = [("say", f) for f in fragments[:-1]] + [("task", fragments[-1])]
    steps.append(("task", (PERSIAN_TASK, 20000, 20600)))
    client, provider, calls = await _call(monkeypatch, steps, db="db-f20-" + fragments[0][0][:3])

    first = calls[0]["task"]
    assert all(f[0].strip() in first for f in fragments), first      # one task, both turns
    assert client.of("language") == []
    assert _language_notes(provider) == []
    assert calls[1].get("reply_language") == "fa"


@pytest.mark.asyncio
async def test_a_language_request_the_model_answered_still_switches(monkeypatch):
    """Control for F20: the same first words, answered on their own."""

    client, provider, calls = await _call(monkeypatch, [
        ("say", ("speak English please", 0, 700)),
        ("reply", ("Sure.", 900, 1200)),
        ("task", (PERSIAN_TASK, 3000, 3600)),
    ], db="db-f20-control")

    assert client.of("language") == [{"type": "language", "lang": "en", "source": "explicit"}]
    assert [c.get("reply_language") for c in calls] == ["en"]


@pytest.mark.asyncio
async def test_an_unanswered_language_request_settles_after_the_span_gap(monkeypatch):
    client, provider, calls = await _call(monkeypatch, [
        ("say", ("speak English please", 0, 700)),
        ("wait", 0.9),
        ("task", (PERSIAN_TASK, 9000, 9600)),
    ], db="db-f20-silence", voice_live_request_span_max_gap_ms=500)

    assert client.of("language") == [{"type": "language", "lang": "en", "source": "explicit"}]
    assert [c.get("reply_language") for c in calls] == ["en"]


# ── F21 end to end ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_say_it_in_english_is_a_one_off_not_a_call_wide_switch(monkeypatch):
    client, provider, calls = await _call(monkeypatch, [
        ("say", ("سیب به انگلیسی چی میشه؟", 0, 900)),
        ("reply", ("سیب.", 1100, 1400)),
        ("say", ("به انگلیسی بگو", 3000, 3700)),
        ("reply", ("Apple.", 3900, 4200)),
        ("task", (PERSIAN_TASK, 6000, 6600)),
    ], db="db-f21")

    assert client.of("language") == []
    assert _language_notes(provider) == []
    assert [c.get("reply_language") for c in calls] == ["fa"]


# ── F22 (relay half): the app's re-stated preference after a repair ───


@pytest.mark.asyncio
async def test_a_repair_on_a_process_without_the_preference_takes_the_apps_restatement(monkeypatch):
    import app.services.live_voice_protocol as live

    await _call(monkeypatch, [
        ("say", ("speak English please", 0, 700)),
        ("reply", ("Sure.", 900, 1200)),
    ], db="db-f22")
    assert live._LANGUAGE_PREF_BY_SESSION.get("db-f22") == ("en", "explicit")
    # The repair lands on the OTHER replica: nothing in its memory.
    live._LANGUAGE_PREF_BY_SESSION.clear()
    live._LANGUAGE_PREF_LRU.clear()

    client, provider, calls = await _call(monkeypatch, [
        ("frame", {"type": "language_preference", "lang": "en"}),
        ("task", (PERSIAN_TASK, 1000, 1600)),
    ], db="db-f22")

    assert client.of("language") == [{"type": "language", "lang": "en", "source": "client"}]
    assert [c.get("reply_language") for c in calls] == ["en"]


# ── F23: a task running across the switch ─────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("switch", ["frame", "spoken"])
async def test_a_result_written_before_the_switch_is_cued_in_the_new_language(monkeypatch, switch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=3.0)
    seen: list[str] = []

    async def think(_user_id, _task, _session_id, **kwargs):
        seen.append(kwargs.get("reply_language"))
        await asyncio.sleep(1.2)
        return PERSIAN_ANSWER, "test-model"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta(PERSIAN_TASK, 0, 600))
            provider.push(H.delegation("d-running", 650))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def running():
        await _until(lambda: any(f.get("state") == "running" for f in client.of("delegation")),
                     label="running")

    async def speak_english():
        box["p"].push(H.user_delta("speak English please", 3000, 3700))
        await _until(lambda: len(_finals(client)) >= 2, label="request final")
        await asyncio.sleep(0.15)
        box["p"].push(H.out_text("Sure.", 3900, 4200))
        box["p"].push(H.out_audio())

    async def finished():
        await _until(lambda: any(f.get("phase") == "completed" for f in client.of("delegation")),
                     timeout=5.0, label="completed")

    step = {"type": "language_preference", "lang": "en"} if switch == "frame" else speak_english
    client = H.FakeClient([H.config(), running, step, finished, 0.5, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True)
    await H.run_relay(client, provider, timeout=12, db_session_id=f"db-f23-{switch}")

    assert seen == ["fa"]                     # dispatched before the switch
    order = [
        (e["type"], str(e.get("event_id") or ""), e.get("content") or "")
        for e in provider.sent
        if e["type"] in ("session.instructions.append", "session.commentary.append")
    ]
    answer_at = next(i for i, (t, _id, c) in enumerate(order)
                     if t == "session.commentary.append" and c.startswith(PERSIAN_ANSWER[:12]))
    cues = [i for i, (t, eid, _c) in enumerate(order) if "toup-live-langcue-" in eid]
    assert cues and cues[0] < answer_at, order
    assert "in English" in order[cues[0]][2]


@pytest.mark.asyncio
async def test_no_cue_when_the_result_is_already_in_the_callers_language(monkeypatch):
    client, provider, calls = await _call(monkeypatch, [
        ("frame", {"type": "language_preference", "lang": "en"}),
        ("task", (PERSIAN_TASK, 1000, 1600)),
    ], db="db-f23-match", think_answer="Professor A works on LLMs.")
    client2, provider2, _ = await _call(monkeypatch, [
        ("task", (PERSIAN_TASK, 1000, 1600)),
    ], db="db-f23-nopref", think_answer=PERSIAN_ANSWER)

    for p in (provider, provider2):
        assert not [e for e in p.of("session.instructions.append")
                    if "toup-live-langcue-" in str(e.get("event_id"))]


# ── F24: an un-rolled tenant's language lines ─────────────────────────

#: The three reply-language bullets of the pre-R2 tenant render
#: (`voice_context._VOICE_MODE_COMMON` at the base snapshot), verbatim.
PRE_R2_TENANT_LINES = (
    "- Match the user's language. Speak EVERY language with a natural, NATIVE "
    "accent and native pronunciation — never a foreign or English-accented one.",
    "- REPLY LANGUAGE: answer in the language the user JUST SPOKE, turn by "
    "turn, even for a two-word turn. Persian in, Persian out — every time; the "
    "same rule holds for every other language. This prompt being written in "
    "English is never a reason to reply in English.",
    "- When the user speaks Persian/Farsi, reply in fluent, natural Farsi with a "
    "native Tehrani accent, pronouncing every Persian sound correctly (خ، غ، ق، ژ, "
    "and the tapped ر) exactly as a native speaker from Tehran would — NOT with an "
    "English accent. In Persian: «فارسی را کاملاً روان و طبیعی صحبت کن، با لهجهٔ "
    "بومیِ تهرانی و تلفّظِ درستِ فارسی، بدون هیچ لهجهٔ خارجی یا انگلیسی.»",
)


def test_an_unrolled_tenants_mirroring_rules_are_rewritten_to_the_r2_wording():
    from app.agent.voice_context import render_voice_mode
    from app.services.live_voice_protocol import adapt_instructions_for_live

    old = "# Voice Conversation Mode\n- Be brief.\n" + "\n".join(PRE_R2_TENANT_LINES) + "\n- Keep it short."
    served = adapt_instructions_for_live(old)
    assert "every time" not in served
    assert "Match the user's language." not in served
    assert "When the user speaks Persian/Farsi, reply" not in served
    rule = served.split("- REPLY LANGUAGE", 1)[1].split("\n", 1)[0]
    assert rule.index("explicitly asked") < rule.index("JUST SPOKE")
    assert "native Tehrani accent" in served and "- Keep it short." in served
    # Word for word what the R2 tenant renders, so relay and tenant cannot drift.
    current = render_voice_mode(datetime(2026, 9, 20, 18, 47, tzinfo=timezone.utc), live=True)
    rendered = set(current.split("\n"))
    for line in served.split("\n"):
        if line.startswith(("- REPLY LANGUAGE", "- Speak EVERY language", "- When you reply in Persian")):
            assert line in rendered, line
    # An R2 tenant render passes through unchanged.
    assert adapt_instructions_for_live(current).startswith(current.rstrip())


# ── F45: a preference is the call's, not the day's ────────────────────


def test_the_preference_store_is_scoped_to_a_call_and_its_repairs():
    import app.services.live_voice_protocol as live

    first, repair, later = object(), object(), object()
    assert live.start_language_pref("db-f45-unit", None, owner=first, repair_s=60) == ("", "")
    live.remember_language_pref("db-f45-unit", "en", "explicit")
    # A repair socket opens while the first is still tearing down …
    assert live.start_language_pref("db-f45-unit", None, owner=repair, repair_s=60) == ("en", "explicit")
    # … so the first socket's release must not start the clock on it.
    live.release_language_pref("db-f45-unit", first)
    assert "db-f45-unit" not in live._LANGUAGE_PREF_RELEASED
    live.release_language_pref("db-f45-unit", repair)
    assert "db-f45-unit" in live._LANGUAGE_PREF_RELEASED
    # A reconnect inside the window is a repair …
    assert live.start_language_pref("db-f45-unit", None, owner=later, repair_s=60) == ("en", "explicit")
    live.release_language_pref("db-f45-unit", later)
    # … and one after it is a later call.
    live._LANGUAGE_PREF_RELEASED["db-f45-unit"] -= 3600
    assert live.start_language_pref("db-f45-unit", None, owner=object(), repair_s=60) == ("", "")


@pytest.mark.asyncio
async def test_a_later_call_that_day_starts_mirroring(monkeypatch):
    import app.services.live_voice_protocol as live

    await _call(monkeypatch, [
        ("say", ("speak English please", 0, 700)),
        ("reply", ("Sure.", 900, 1200)),
    ], db="db-f45-day", features=())
    assert live._LANGUAGE_PREF_BY_SESSION.get("db-f45-day") == ("en", "explicit")
    # Hours later — past any repair window.
    monkeypatch.setattr(live, "_LANGUAGE_PREF_REPAIR_S", 0.0, raising=False)
    await asyncio.sleep(0.01)

    client, provider, calls = await _call(monkeypatch, [
        ("task", (PERSIAN_TASK, 1000, 1600)),
    ], db="db-f45-day", features=())
    assert _language_notes(provider) == []
    assert [c.get("reply_language") for c in calls] == ["fa"]


# ── F16: an accepted reask answered by a bare delegation ──────────────


async def _lost_reply_then_reask(monkeypatch, *, answers, db):
    H.fast_clocks(
        monkeypatch, voice_live_interrupt_suppress_ms=400,
        voice_live_output_epoch_gap_ms=80, voice_live_delegation_ttl_s=1.2,
    )
    calls: list[str] = []

    async def think(_user_id, task, _session_id, **kwargs):
        calls.append(kwargs.get("display_request") or task)
        return "Professor A works on LLMs.", "test-model"

    H.patch_relay(monkeypatch, think=think)
    box: dict = {}

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.out_text("Hi, how can I help?", 0, 300))
            provider.push(H.out_audio())
        elif event["type"] == "session.instructions.append" and str(
            event.get("event_id") or ""
        ).startswith("toup-live-reask-"):
            provider.push({"type": "session.instructions.appended",
                           "client_event_id": event["event_id"]})
            for index, offset in enumerate(answers):
                provider.push(H.delegation(f"d-reask-{index}", offset))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def ask_and_barge():
        await _until(lambda: client.of("response_text"), label="greeting")
        box["p"].push(H.user_delta(QUESTION, 500, 900))
        await _until(lambda: any(f.get("text") == QUESTION for f in client.of("transcript")),
                     label="words")
        client._script.insert(0, {"type": "interrupt"})

    async def lost_reply():
        await _until(lambda: _finals(client), label="final")
        # Before the resume floor, inside the interrupt fence: dropped.
        box["p"].push(H.out_text("Okay, one moment.", 600, 850))
        box["p"].push(H.out_audio())
        await asyncio.sleep(0.6)

    def reask():
        client._script.insert(0, {
            "type": "inject_text", "text": QUESTION, "reason": "no_response",
            "reask_of_user_turn_id": _finals(client)[0]["turn_id"],
        })

    client = H.FakeClient([
        {**H.config(), "features": REASK_FEATURES},
        ask_and_barge, lost_reply, reask, 2.0, {"type": "stop"},
    ])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=12, db_session_id=db)
    return client, calls


@pytest.mark.asyncio
async def test_a_reask_answered_by_a_bare_delegation_runs_the_request(monkeypatch):
    client, calls = await _lost_reply_then_reask(monkeypatch, answers=[1800], db="db-f16")

    turn = _finals(client)[0]["turn_id"]
    assert [(f["user_turn_id"], f["outcome"]) for f in client.of("reask_result")] == [(turn, "accepted")]
    assert calls == [QUESTION]
    assert not [f for f in client.of("delegation") if f.get("phase") == "expired"]
    assert {f.get("turn_id") for f in client.of("delegation")} == {turn}
    assert {f["parent_user_turn_id"] for f in client.of("delegation")
            if "parent_user_turn_id" in f} == {turn}


@pytest.mark.asyncio
async def test_two_delegations_for_one_reask_still_run_it_once(monkeypatch):
    client, calls = await _lost_reply_then_reask(monkeypatch, answers=[1800, 1850], db="db-f16-twice")

    assert calls == [QUESTION]
    assert [f.get("reason") for f in client.of("delegation") if f.get("phase") == "expired"] == [
        "no_causal_turn",
    ]


# ── F44: a question stranded by a socket drop ─────────────────────────


def _drop():
    raise WebSocketDisconnect(code=1006)


async def _stranded_then_reconnect(monkeypatch, second_script, *, first_features=TF132,
                                   second_features=TF132, answer=None, db="db-f44"):
    H.fast_clocks(monkeypatch)
    calls: list[str] = []

    async def think(_user_id, task, _session_id, **kwargs):
        calls.append(kwargs.get("display_request") or task)
        return "Sunny, 21 degrees.", "test-model"

    H.patch_relay(monkeypatch, think=think)

    def first_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("what is the weather ", 0, 400))
            provider.push(H.user_delta("in toronto tomorrow", 400, 900))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    c1 = H.FakeClient([{**H.config(), "features": list(first_features)}, 0.6, _drop])
    await H.run_relay(c1, H.FakeProvider(on_send=first_send, auto_ack=True), timeout=6,
                      db_session_id=db)
    stranded = _finals(c1)[0]["turn_id"]

    def second_send(provider, event):
        if (
            answer is not None
            and event["type"] == "session.instructions.append"
            and str(event.get("event_id") or "").startswith("toup-live-reask-")
        ):
            answer(provider)
        elif event["type"] == "session.close":
            provider.push(H.closed())

    p2 = H.FakeProvider(on_send=second_send, auto_ack=True)
    c2 = H.FakeClient([{**H.config(), "features": list(second_features)}, *second_script(stranded)])
    await H.run_relay(c2, p2, timeout=8, db_session_id=db)
    return stranded, c1, c2, p2, calls


def _reask_appends(provider):
    return [e for e in provider.of("session.instructions.append")
            if str(e.get("event_id") or "").startswith("toup-live-reask-")]


@pytest.mark.asyncio
async def test_tf132s_replay_after_a_socket_drop_is_answered_once(monkeypatch):
    question = "what is the weather in toronto tomorrow"
    stranded, _c1, c2, p2, calls = await _stranded_then_reconnect(
        monkeypatch,
        lambda _turn: [0.15, {"type": "inject_text", "text": question}, 0.3,
                       {"type": "inject_text", "text": question}, 0.5, {"type": "stop"}],
        answer=lambda provider: provider.push(H.delegation("d-carried", 200)),
    )

    appends = _reask_appends(p2)
    assert [e["event_id"] for e in appends] == [f"toup-live-reask-{stranded}"]
    assert question in appends[0]["content"]
    # TF132 negotiated nothing new, so it is told nothing new.
    assert c2.of("reask_result") == []
    # The model answered by delegating: the stranded question runs, once, as
    # the same turn the phone is holding.
    assert calls == [question]
    assert {f.get("turn_id") for f in c2.of("delegation")} == {stranded}
    assert {f["parent_user_turn_id"] for f in c2.of("delegation")
            if "parent_user_turn_id" in f} == {stranded}


@pytest.mark.asyncio
async def test_a_reask_by_id_after_a_reconnect_is_accepted(monkeypatch):
    stranded, _c1, c2, p2, _calls = await _stranded_then_reconnect(
        monkeypatch,
        lambda turn: [0.15, {"type": "inject_text", "text": "x", "reason": "no_response",
                             "reask_of_user_turn_id": turn}, 0.4, {"type": "stop"}],
        first_features=REASK_FEATURES, second_features=REASK_FEATURES, db="db-f44-id",
    )

    assert [(f["user_turn_id"], f["outcome"]) for f in c2.of("reask_result")] == [
        (stranded, "accepted"),
    ]
    assert len(_reask_appends(p2)) == 1


@pytest.mark.asyncio
async def test_a_replay_after_the_caller_spoke_on_the_new_socket_is_stale(monkeypatch):
    question = "what is the weather in toronto tomorrow"

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)

    def first_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta(question, 0, 900))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    c1 = H.FakeClient([{**H.config(), "features": list(REASK_FEATURES)}, 0.6, _drop])
    await H.run_relay(c1, H.FakeProvider(on_send=first_send, auto_ack=True), timeout=6,
                      db_session_id="db-f44-stale")

    def second_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("never mind, play some jazz", 0, 900))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    p2 = H.FakeProvider(on_send=second_send, auto_ack=True)
    c2 = H.FakeClient([{**H.config(), "features": list(REASK_FEATURES)}, 0.5,
                       {"type": "inject_text", "text": question}, 0.3, {"type": "stop"}])
    await H.run_relay(c2, p2, timeout=6, db_session_id="db-f44-stale")

    assert _reask_appends(p2) == []
    assert [f["outcome"] for f in c2.of("reask_result")] == ["stale"]


async def _answered_then_dropped(monkeypatch, question, *, heard: bool, db: str):
    """TF132: the question is answered (text + audio), the phone sends a
    `playback_idle` receipt for that epoch only when `heard`, then the socket
    drops and the next socket replays the question as text."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)

    def first_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta(question, 0, 900))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def answer():
        await _until(lambda: _finals(c1), label="final")
        await asyncio.sleep(0.1)
        p1.push(H.out_text("Sunny tomorrow.", 1000, 1400))
        p1.push(H.out_audio())
        await _until(lambda: c1.of("audio_delta"), label="answer audio")
        if heard:
            eid = c1.of("audio_delta")[-1]["response_id"]
            c1._script[0:0] = [
                {"type": "playback_idle", "response_id": eid, "item_id": eid, "played_ms": 900},
                0.3,
            ]
        else:
            await asyncio.sleep(0.3)

    p1 = H.FakeProvider(on_send=first_send, auto_ack=True)
    c1 = H.FakeClient([{**H.config(), "features": list(TF132)}, answer, _drop])
    await H.run_relay(c1, p1, timeout=6, db_session_id=db)

    p2 = H.FakeProvider(on_send=lambda p, e: p.push(H.closed()) if e["type"] == "session.close" else None,
                        auto_ack=True)
    c2 = H.FakeClient([{**H.config(), "features": list(TF132)}, 0.15,
                       {"type": "inject_text", "text": question}, 0.3, {"type": "stop"}])
    await H.run_relay(c2, p2, timeout=6, db_session_id=db)
    return c2, p2


@pytest.mark.asyncio
async def test_an_answered_question_is_not_carried(monkeypatch):
    # Supervisor 2026-09-23 18:16: "answered" is a HEARD answer.  The old pin
    # sent NO receipt and still expected "not carried" — i.e. it pinned
    # receipt ABSENCE as proof of hearing.  The answer now carries the phone's
    # `playback_idle` receipt (HEARD), and is not carried; the no-receipt
    # shape is the next test (new verdict: carried, unconfirmed).
    question = "what is the weather in toronto tomorrow"
    _c2, p2 = await _answered_then_dropped(
        monkeypatch, question, heard=True, db="db-f44-answered",
    )
    assert _reask_appends(p2) == []


@pytest.mark.asyncio
async def test_a_question_answered_with_no_receipt_is_carried_unconfirmed(monkeypatch):
    # Supervisor 2026-09-23 18:16 (the old pin above, without a receipt): an
    # answer that went out with no receipt either way is UNKNOWN — neither
    # heard nor unheard.  New verdict: carried, and TF132's text replay on the
    # next socket is accepted ONCE with the instruction that claims neither
    # (no "No reply … reached them", no "did not reach"), asks for a recap or
    # a question, and forbids redoing work.  TF132 still gets no reask_result.
    question = "what is the weather in toronto tomorrow"
    c2, p2 = await _answered_then_dropped(
        monkeypatch, question, heard=False, db="db-f44-unconfirmed",
    )
    appends = _reask_appends(p2)
    assert len(appends) == 1, appends
    content = appends[0]["content"]
    assert question in content and "never confirmed playing it" in content
    assert "did not reach" not in content and "No reply" not in content
    assert "Don't redo any action" in content
    assert c2.of("reask_result") == []
