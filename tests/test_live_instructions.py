"""The prompt GPT-Live is started with, and what a legacy client receives.

The provider documents the prompt as the ONLY channel for client delegation:
"List the capabilities your backend supports. GPT-Live uses this list to decide
which requests to hand off."  Toup was sending the Realtime-era voice context
verbatim, which describes a `think` tool, a `navigate_to` tool, terminal access
and screen share — none of which exist on the Live wire.
"""

from datetime import datetime, timezone

import pytest

from app.agent.voice_context import render_voice_mode
from app.services.live_voice_protocol import (
    LIVE_BACKEND_SECTION,
    adapt_instructions_for_live,
    reply_language_directive,
)

import test_live_harness as H


def test_the_realtime_only_paragraphs_are_removed_from_the_live_prompt(caplog):
    source = render_voice_mode(datetime.now(timezone.utc))
    live = adapt_instructions_for_live(source)

    for probe in (
        "navigate_to",
        "FULL ACCESS to the user's computer terminal",
        "think tool",
        "[Screen context:",
        "terminal commands",
    ):
        assert probe not in live, f"{probe!r} survived the surgery"
    # The openings are matched literally against `render_voice_mode`. If that
    # renderer moves, the Live session is about to be told to call tools it does
    # not have — so a partial match is LOUD.
    assert "instructions surgery matched" not in caplog.text

    # What is true on both paths stays.
    assert "Do NOT use markdown" in live
    assert "native Tehrani accent" in live


def test_the_live_prompt_names_the_backend_its_rules_and_the_language_rule():
    live = adapt_instructions_for_live("# Voice Conversation Mode\n- Be brief.")
    assert LIVE_BACKEND_SECTION in live
    for probe in (
        "Music and audio",
        "Research: search the web",
        "Connected accounts",
        "Delegate to the backend when",
        "Do not delegate to the backend when",
        "Never say that an action has started",
        "Never say you cannot do something listed above",
        "progress notes",
        "Reply language",
        "the language the user JUST SPOKE",
    ):
        assert probe in live, probe
    assert live.startswith("# Voice Conversation Mode")


def test_a_surgery_that_matches_nothing_is_logged(caplog):
    import logging

    caplog.set_level(logging.INFO)
    # The instant-start stub legitimately has nothing to strip; warning on it
    # would train the reader to ignore the line that matters.
    adapt_instructions_for_live("nothing to strip here")
    stub = [r for r in caplog.records if "instructions surgery" in r.message]
    assert stub and stub[0].levelno == logging.INFO

    caplog.clear()
    # A prompt that IS the shared renderer's output but matches nothing means
    # that renderer moved: loud.
    adapt_instructions_for_live("# Voice Conversation Mode\n- renamed everything")
    drift = [r for r in caplog.records if "instructions surgery" in r.message]
    assert drift and drift[0].levelno == logging.WARNING


def test_the_reply_language_directive_names_the_language_without_pinning_it():
    assert reply_language_directive("fa").endswith("(Persian/Farsi).")
    assert reply_language_directive("en") == "Reply in the caller's language."


def test_an_explicit_language_request_outranks_turn_by_turn_mirroring():
    """V3 (§G). The rule said "answer in the language the user JUST SPOKE, turn
    by turn" and nothing else, so a caller who asked in Persian to be spoken to
    in English only was, by the letter of the prompt, to be answered in Persian
    on the very next Persian word. Mirroring stays the default; an explicit
    request outranks it until the caller changes it."""

    live = adapt_instructions_for_live("# Voice Conversation Mode\n- Be brief.")
    rule = live.split("# Reply language", 1)[1]
    assert "explicitly asked" in rule
    assert "outranks" in rule
    assert "until they ask for a different" in rule
    assert "the language the user JUST SPOKE" in rule          # still the default
    assert "every time" not in rule
    # The explicit clause comes first: the model reads the exception before
    # the rule it overrides, not as an afterthought to it.
    assert rule.index("explicitly asked") < rule.index("JUST SPOKE")


def test_a_preferred_language_directive_is_explicit_about_the_language():
    """Under a preference the delegated turn must not say "the caller's
    language": for a Persian-script request after "English only" that phrase
    reads as Persian."""

    from app.services.live_voice_protocol import preferred_language_directive

    en = preferred_language_directive("en")
    assert en.startswith("Reply in English")
    assert "Persian" not in en and "caller's language" not in en
    assert preferred_language_directive("fa").startswith("Reply in Persian/Farsi")


@pytest.mark.asyncio
async def test_a_reconnect_seeds_the_new_provider_session_with_recent_history(monkeypatch):
    """A8-4: `build_session_start(history=…)` had no production caller, so three
    provider sessions over one DB session each started from an empty
    conversation."""

    H.fast_clocks(monkeypatch, voice_live_history_rows=20)

    async def vps(_user_id):
        return ("http://agent.local", "key")

    calls = []

    async def vps_api(agent_url, key, method, path, params=None, json_body=None, timeout=15.0):
        calls.append((method, path, params))
        return [
            {"role": "user", "content": "what did we decide"},
            {"role": "assistant", "content": "we decided Thursday"},
            {"role": "system", "content": "ignored"},
        ]

    from app.api import ws_realtime as rt
    H.patch_relay(monkeypatch, vps=vps)
    monkeypatch.setattr(rt, "_vps_api", vps_api)

    def on_send(provider, event):
        if event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.05, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=6)

    assert calls and calls[0][0] == "GET"
    session = provider.of("session.start")[0]["session"]
    assert session["input"] == [
        {"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": "what did we decide"}]},
        {"type": "message", "role": "assistant",
         "content": [{"type": "output_text", "text": "we decided Thursday"}]},
    ]


@pytest.mark.asyncio
async def test_a_slow_tenant_never_holds_the_session_start(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_history_rows=20)

    async def vps(_user_id):
        import asyncio
        await asyncio.sleep(5)
        return ("http://agent.local", "key")

    H.patch_relay(monkeypatch, vps=vps)

    def on_send(provider, event):
        if event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.05, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=6)

    assert "input" not in provider.of("session.start")[0]["session"]


@pytest.mark.asyncio
async def test_a_legacy_client_receives_only_one_final_transcript_per_utterance(monkeypatch):
    """Build 129 and the web client announce no `features`.  They get the whole
    utterance once — strictly better than today's per-350 ms fragment — and no
    delegation frames, which they would ignore anyway."""

    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=300)

    async def think(*args, **kwargs):
        return "answered", "m"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("tell ", 0, 200))
            provider.push(H.user_delta("me ", 260, 400))
            provider.push(H.user_delta("something", 450, 700))
            provider.push(H.delegation("d1", 720))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.legacy_config(), 0.9, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    transcripts = client.of("transcript")
    assert len(transcripts) == 1
    assert transcripts[0]["text"] == "tell me something"
    assert transcripts[0]["final"] is True
    assert client.of("delegation") == []


@pytest.mark.asyncio
async def test_a_client_that_never_acks_still_gets_back_to_listening(monkeypatch):
    """A1-03: the synthetic output id was released only by the phone's native
    `SoundChunkPlayed(isFinal)`.  One missed event meant one output id for the
    whole call and no route back to Listening, with the half-duplex mic gate
    latched open behind the wrong label."""

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=120)
    H.patch_relay(monkeypatch)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.out_text("First reply.", 0, 400))
            provider.push(H.out_audio())
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.8, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    states = client.states()
    assert "speaking" in states
    assert states[-1] == "listening"
    segments = client.of("speech_segment_complete")
    assert len(segments) == 1
    assert segments[0]["text"] == "First reply."
    assert segments[0]["played_ms"] is None


@pytest.mark.asyncio
async def test_talking_over_the_agent_synthesizes_one_speech_started(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000,
                  voice_live_bargein_min_ms=200)
    H.patch_relay(monkeypatch)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.out_text("A long answer", 0, 2000))
            provider.push(H.out_audio())
            provider.push(H.user_delta("wait hold on", 500, 1200))
            provider.push(H.user_delta("stop", 1250, 1400))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.6, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    assert len(client.of("speech_started")) == 1
    assert client.of("speech_started")[0]["source"] == "transcript"


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_barge_in_can_be_switched_off_without_a_client_build(monkeypatch, enabled):
    """VOICE_LIVE_BARGEIN_ENABLED=false is the owner's documented no-build
    remediation. The utterance must be one that DOES fire with the switch on
    (the control) — a single delta never fires at all, so it would pass
    whether or not the switch was read."""

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=4000,
                  voice_live_bargein_enabled=enabled)
    H.patch_relay(monkeypatch)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.out_text("A long answer", 0, 2000))
            provider.push(H.out_audio())
            provider.push(H.user_delta("wait hold on", 500, 1200))
            provider.push(H.user_delta("stop", 1250, 1400))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.6, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    assert len(client.of("speech_started")) == (1 if enabled else 0)


@pytest.mark.asyncio
async def test_a_provider_close_reason_maps_to_its_own_client_code(monkeypatch):
    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.closed(reason="content"))

    client = H.FakeClient([H.config(), 3.0])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    codes = [f["code"] for f in client.of("error")]
    assert "live_closed_content" in codes


# ── Round 48 fix pass ─────────────────────────────────────────────────

def test_the_media_capability_line_is_word_identical_on_both_renderers():
    """C3. `adapt_instructions_for_live` is the guarantee until the agent image
    rolls and `render_live_delegation` takes over; the model must be told the
    same thing either way, and neither may claim more than `play_media` can do
    — the agent has no pause, skip or stop tool."""

    from app.agent.voice_context import render_live_delegation

    line = (
        "- Music and audio: start playback, or move to the next/previous track "
        "on the user's device."
    )
    assert line in LIVE_BACKEND_SECTION
    assert line in render_live_delegation()
    for overclaim in ("pause", "skip", "stop playback", "resume"):
        assert overclaim not in LIVE_BACKEND_SECTION.lower().split("# delegate")[0]


@pytest.mark.asyncio
async def test_the_whole_history_seed_shares_one_deadline(monkeypatch):
    """L1R-8 / §6. Two sequential 2 s waits could add FOUR seconds to
    tap→ready, and a sick tenant is exactly when both legs are slow. The
    lookup and the fetch now run under one 2 s budget."""

    import asyncio
    import time

    H.fast_clocks(monkeypatch, voice_live_history_rows=20)

    async def vps(_user_id):
        await asyncio.sleep(1.9)          # nearly the whole budget
        return ("http://agent.local", "key")

    async def vps_api(agent_url, key, method, path, params=None, json_body=None, timeout=15.0):
        await asyncio.sleep(5)            # …and the second leg hangs
        return []

    from app.api import ws_realtime as rt
    H.patch_relay(monkeypatch, vps=vps)
    monkeypatch.setattr(rt, "_vps_api", vps_api)

    def on_send(provider, event):
        if event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.05, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send)
    started = time.monotonic()
    await H.run_relay(client, provider, timeout=8)
    elapsed = time.monotonic() - started

    assert "input" not in provider.of("session.start")[0]["session"]
    assert elapsed < 3.5, f"the seed spent {elapsed:.1f}s of the caller's tap→ready"


@pytest.mark.asyncio
async def test_an_expired_provider_session_is_not_advertised_as_recoverable(monkeypatch):
    """C5.6. The relay sets `closed` and stops relaying in the same breath, so
    `recoverable: True` told the client to keep waiting on a socket that will
    never carry anything again."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.closed(reason="expired"))

    client = H.FakeClient([H.config(), 1.0])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    expired = [f for f in client.of("error") if f["code"] == "session_expired"]
    assert expired and expired[0]["recoverable"] is False


@pytest.mark.asyncio
async def test_state_seq_is_strictly_increasing_for_the_whole_session(monkeypatch):
    """C5.10. The client orders `state` frames by `seq`; a frozen counter makes
    every frame equally old and the ordering guard a no-op. Deleting the
    increment killed no test."""

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=60)

    async def think(*args, **kwargs):
        return "Here is the answer.", "m"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("look it up", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.commentary.append":
            provider.push(H.out_text("Here it is.", 900, 1400))
            provider.push(H.out_audio("SPEAKING"))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 1.0, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=8)

    seqs = [f["seq"] for f in client.of("state")]
    assert len(seqs) >= 3, "the session must actually change state a few times"
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs), "two state frames may never share a seq"


@pytest.mark.asyncio
async def test_a_partial_transcript_carries_the_whole_utterance_so_far(monkeypatch):
    """C5.11. `transcript.text` is the FULL utterance, never the delta — the
    client REPLACES the row's text with it. Truncating the partial frames' text
    killed no test, and on the phone it reads as the caption eating itself."""

    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=5000)
    H.patch_relay(monkeypatch)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("book ", 0, 200))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    provider = H.FakeProvider(on_send=on_send)

    async def more():
        import asyncio
        await asyncio.sleep(0.15)
        provider.push(H.user_delta("a table ", 250, 500))
        await asyncio.sleep(0.15)
        provider.push(H.user_delta("for four", 550, 800))

    import asyncio as _asyncio
    feeder = _asyncio.create_task(more())
    client = H.FakeClient([H.config(), 0.7, {"type": "stop"}])
    await H.run_relay(client, provider, timeout=6)
    await _asyncio.gather(feeder, return_exceptions=True)

    partials = [t for t in client.of("transcript") if t["partial"]]
    finals = [t for t in client.of("transcript") if t["final"]]
    assert len(partials) >= 2, "the caption must move more than once"
    assert finals[-1]["text"] == "book a table for four"
    for frame in partials:
        # A DELTA (or any truncation of the accumulated text) is not a prefix
        # of the finished utterance. The client REPLACES the row with this
        # string, so a delta here reads as the caption eating itself.
        assert finals[-1]["text"].startswith(frame["text"]), frame["text"]
    assert max(len(f["text"]) for f in partials) > len("book")


def test_no_voice_presence_claim_sits_above_a_bail_out_that_ends_no_card():
    """L1R-4 — a SOURCE-ORDER probe, because the two bail-outs it guards are
    inside a 2000-line endpoint with no test seam.

    `_voice_session_owner[user_id] = _session_nonce` was hoisted above the
    protocol fork so the Live branch could share it. That put it above two
    Realtime returns that call no `_defer_voice_la_end`: a session that
    connects and then fails session configuration (or whose client vanishes
    before `ready`) claims the user's voice presence and exits without ending
    anything — so a concurrently live earlier call finds itself no longer the
    owner and skips ending its own Lock-Screen card. That is the exact class
    A7-07 was hoisted to fix, moved onto the Realtime path.
    """

    import pathlib

    lines = pathlib.Path("app/api/ws_realtime.py").read_text().split("\n")

    def index(needle, start=0):
        for i in range(start, len(lines)):
            if needle in lines[i]:
                return i
        raise AssertionError(f"marker gone, rewrite this probe: {needle!r}")

    claim = "_voice_session_owner[user_id] = _session_nonce"
    claims = [i for i, line in enumerate(lines) if claim in line]
    assert len(claims) == 2, "one claim per protocol: Live, then Realtime"

    fork = index("    if use_live:")
    live_claim, realtime_claim = claims
    assert fork < live_claim, "the Live claim belongs inside the Live branch"
    # …and that branch ends the card unconditionally.
    assert index("_defer_voice_la_end(", live_claim) < index("        return", live_claim)

    # Both Realtime bail-outs end no card, so neither may be below the claim.
    config_failed = index('"code": "session_setup_failed"', live_claim)
    ready_failed = index('"type": "ready",', config_failed)
    assert realtime_claim > config_failed, (
        "a session that failed to configure must not own the user's voice presence"
    )
    assert realtime_claim > ready_failed, (
        "a session whose client vanished before `ready` must not own it either"
    )


def test_the_voice_context_request_says_which_wire_it_is_for():
    """C5.1 — also a source probe: `_agent_voice_context` is a closure inside
    the endpoint. The agent-side renderer does the Realtime-paragraph surgery
    properly once the image rolls, and it can only do it if it is told."""

    import pathlib

    src = pathlib.Path("app/api/ws_realtime.py").read_text()
    body = src.split("async def _agent_voice_context", 1)
    assert len(body) == 2, "marker gone, rewrite this probe"
    head = body[1].split("data = await _vps_api", 1)[0]
    assert '"live": use_live,' in head

    from app.api.api_v1 import VoiceContextRequest
    assert "live" in VoiceContextRequest.model_fields, (
        "the field the relay sends must exist on the route that receives it"
    )
