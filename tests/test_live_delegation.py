"""Delegation lifecycle on the GPT-Live relay, driven end to end.

Round 48 reversed three shipped behaviours here, each of which had a test
asserting it was correct.  The replacements say so in place.
"""

import asyncio

import pytest

import test_live_harness as H


def _tool_start(relay, call_id="i1", name="web_search", query="u of t"):
    return relay.on_event({
        "type": "tool.start", "call_id": call_id, "name": name,
        "args": {"query": query},
    })


def _tool_end(relay, call_id="i1", name="web_search"):
    return relay.on_event({
        "type": "tool.end", "call_id": call_id, "name": name,
        "ok": True, "preview": "found", "elapsed_ms": 40,
    })


@pytest.mark.asyncio
async def test_speech_during_a_delegation_neither_discards_nor_cancels_it(monkeypatch):
    """A1-01 / A6-1. A backchannel used to bump the tracker's revision, which
    made `accepts_result` false by the time the agent answered 25 s later:
    nothing spoken, nothing persisted, the orb stranded on thinking."""

    H.fast_clocks(monkeypatch)
    started = asyncio.Event()
    provider_box = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        started.set()
        provider = provider_box["p"]
        # The caller speaks WHILE the research runs — a status question, then a
        # bare acknowledgement.
        provider.push(H.user_delta("چی داری سرچ می‌کنی", 4000, 4600))
        provider.push(H.user_delta("آره", 6000, 6200))
        await asyncio.sleep(0.35)
        return "Admissions open in September.", "test-model"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        provider_box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("search u of t", 100, 900))
            provider.push(H.delegation("d1", 950))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 1.6, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=8)

    assert started.is_set()
    spoken = provider.of("session.commentary.append")
    assert spoken, "the answer must still be spoken after mid-flight speech"
    assert "Admissions open in September." in " ".join(e["content"] for e in spoken)
    assert any(
        s.get("assistant_text") == "Admissions open in September."
        for s in recorded["saves"]
    )
    assert "completed" in client.phases()


@pytest.mark.asyncio
async def test_a_delegation_dispatches_when_the_utterance_closes(monkeypatch):
    """A6-4. `offset_ms` after the last transcript used to park the task until
    the user repeated themselves, then run on the concatenation."""

    H.fast_clocks(monkeypatch)
    seen = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        seen["task"] = task
        seen["display"] = kwargs.get("display_request")
        return "ok", "m"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("play a song", 0, 3000))
            # The provider decided to delegate AFTER the utterance ended.
            provider.push(H.delegation("d1", 3400))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.9, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    assert seen.get("display") == "play a song"
    assert "play a songplay a song" not in seen["task"]


@pytest.mark.asyncio
async def test_a_delegation_with_nothing_to_consume_expires(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=0.2)
    thinks = []

    async def think(*args, **kwargs):
        thinks.append(args)
        return "ok", "m"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.delegation("orphan", 10))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.8, {"type": "stop"}])
    client_provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, client_provider, timeout=6)

    assert thinks == []
    assert "expired" in client.phases()
    expired = next(frame for frame in client.of("delegation") if frame["phase"] == "expired")
    assert expired["task_id"] == expired["delegation_id"] == "orphan"
    assert expired["state"] == "failed"
    assert expired["task_revision"] == 1
    assert expired["parent_user_turn_id"] == expired["turn_id"] == ""
    assert expired["clock"] == "provider"
    assert expired["reason"] == "no_causal_turn"


@pytest.mark.asyncio
async def test_interrupt_stops_speech_only_and_sends_one_non_elicitive_stop(monkeypatch):
    """REPLACES test_valid_interrupt_stops_backend_and_suppresses_late_provider_audio.

    That test asserted the shipped semantics: one tap to stop the agent talking
    cancelled every running delegation.  Per the provider's own contract a
    barge-in is a SPEECH event and "delegated work can continue" — and a 25 s
    research turn dying because the user stopped an unrelated sentence is the
    A6-2 defect.  The provider-close ordering assertions still hold and are kept.
    """

    # The epoch must still be open when the interrupt lands, so the gap
    # rotation is parked far away here; it has its own test.
    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=5000)
    started = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        started.set()
        try:
            await asyncio.wait_for(release.wait(), timeout=3)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "Still here.", "m"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("research this", 0, 400))
            provider.push(H.delegation("d1", 420))
            provider.push(H.out_text("Let me look that up", 500, 900))
            provider.push(H.out_audio("SPEAKING"))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([
        H.config(),
        0.5,
        {"type": "interrupt", "response_id": f"live:{H.PSID}:1",
         "item_id": f"live:{H.PSID}:1", "played_ms": 300},
        {"type": "interrupt", "response_id": "wrong", "item_id": "wrong"},
        0.1,
        lambda: release.set(),
        0.4,
        {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=8)

    assert started.is_set()
    assert not cancelled.is_set(), "an interrupt must never cancel delegated work"
    stops = provider.of("session.instructions.append")
    assert len(stops) == 1
    assert stops[0]["content"] == "Stop speaking now. Do not acknowledge this instruction."
    assert "acknowledge this instruction" in stops[0]["content"]
    assert stops[0]["delegation_id"] is None
    assert any(s.get("assistant_text") == "Still here." for s in recorded["saves"])
    # The interrupted segment is recorded as interrupted, with what was heard.
    spoken = [
        s for s in recorded["saves"]
        if (s.get("assistant_voice") or {}).get("source") == "live_spoken"
    ]
    assert spoken and spoken[0]["assistant_voice"]["interrupted"] is True


@pytest.mark.asyncio
async def test_explicit_cancel_task_stops_the_run_and_closes_its_tool_rows(monkeypatch):
    H.fast_clocks(monkeypatch)
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        await _tool_start(relay)
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("do the thing", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([
        H.config(), 0.6, {"type": "cancel_task", "delegation_id": "d1"},
        0.3, {"type": "stop"},
    ])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=8)

    assert started.is_set() and cancelled.is_set()
    completions = client.of("tool_call.completed")
    assert completions, "a cancelled turn must terminalise its open rows"
    # `cancelled` is not `ok:false`: the client renders that as "that one didn't
    # work", a failure claim about work the user deliberately stopped.
    assert completions[-1]["outcome"] == "cancelled"
    assert "cancelled" in client.phases()
    assert client.states()[-1] == "listening"


@pytest.mark.asyncio
async def test_a_correction_supersedes_and_carries_the_executed_ledger(monkeypatch):
    H.fast_clocks(monkeypatch)
    calls: list[str] = []
    first_started = asyncio.Event()
    box = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        calls.append(task)
        if len(calls) == 1:
            await _tool_start(relay, name="calendar__create_event", query="friday")
            await _tool_end(relay, name="calendar__create_event")
            first_started.set()
            await asyncio.Future()          # still running when the correction lands
        return "Booked Thursday.", "m"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        box["p"] = provider
        if event["type"] == "session.start":
            provider.push(H.user_delta("book friday with the dentist", 0, 900))
            provider.push(H.delegation("d1", 950))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def correct():
        await first_started.wait()
        provider = box["p"]
        provider.push(H.user_delta("no, book thursday with the dentist", 4000, 5000))
        provider.push(H.delegation("d2", 5100))

    client = H.FakeClient([H.config(), correct, 1.2, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=8)

    assert len(calls) == 2, "the correction must run, not merely cancel"
    assert "do NOT repeat" in calls[1]
    assert "calendar__create_event" in calls[1]
    assert "superseded" in client.phases()


@pytest.mark.asyncio
async def test_hanging_up_detaches_the_running_delegation_instead_of_killing_it(monkeypatch):
    """REPLACES test_stop_cancels_agent_work_before_provider_close_wait.

    That test asserted that `stop` cancelled every in-flight delegation.  It
    also threw away work the user had asked for and already paid for, while the
    tenant AgentRunner kept running regardless — the cancel was a local
    illusion with no result path back.  Tier 1 of the durable design (D4):
    detach, finish under the remaining budget, persist into the same
    conversation.  The provider-close ordering assertion is unchanged.
    """

    H.fast_clocks(monkeypatch)
    started = asyncio.Event()
    release = asyncio.Event()
    cancelled = asyncio.Event()

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "Finished after hang-up.", "m"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("long research", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.6, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=8)

    assert started.is_set()
    assert not cancelled.is_set()
    assert provider.of("session.close") == [
        {"type": "session.close", "event_id": "toup-live-stop"},
    ]
    release.set()
    for _ in range(60):
        await asyncio.sleep(0.05)
        if any(s.get("assistant_text") == "Finished after hang-up." for s in recorded["saves"]):
            break
    assert any(
        s.get("assistant_text") == "Finished after hang-up." for s in recorded["saves"]
    ), "detached work must persist its answer into the same conversation"


@pytest.mark.asyncio
async def test_every_delegation_exit_emits_one_terminal_state_and_frame(monkeypatch):
    H.fast_clocks(monkeypatch)

    async def boom(user_id, task, session_id, relay=None, out=None, **kwargs):
        await _tool_start(relay)
        raise RuntimeError("agent exploded")

    H.patch_relay(monkeypatch, think=boom)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("do it", 0, 300))
            provider.push(H.delegation("d1", 320))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.8, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    assert client.phases().count("failed") == 1
    assert client.states()[-1] == "listening"
    assert [f["outcome"] for f in client.of("tool_call.completed")] == ["error"]
    assert any(f.get("code") == "delegation_failed" for f in client.of("error"))
    # Codes only: the relay never authors a user-facing English sentence.
    assert all("message" not in f for f in client.of("error") if f["code"] == "delegation_failed")


@pytest.mark.asyncio
async def test_an_empty_answer_is_a_failure_with_its_own_code(monkeypatch):
    H.fast_clocks(monkeypatch)

    async def empty(*args, **kwargs):
        return "   ", "m"

    H.patch_relay(monkeypatch, think=empty)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("do it", 0, 300))
            provider.push(H.delegation("d1", 320))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.8, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    assert any(f.get("code") == "delegation_empty" for f in client.of("error"))


@pytest.mark.asyncio
async def test_progress_notes_reach_the_model_and_are_never_spoken(monkeypatch):
    """A8-3 / A6-5: for 25 s the model knew nothing, so a status question could
    only be answered by inventing or by delegating again."""

    H.fast_clocks(monkeypatch, voice_live_progress_max_per_delegation=2)

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        for i in range(6):
            await _tool_start(relay, call_id=f"i{i}")
            await _tool_end(relay, call_id=f"i{i}")
        return "Done.", "m"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("research this", 0, 300))
            provider.push(H.delegation("d1", 320))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.9, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=6)

    notes = [
        e for e in provider.of("session.thinking.append")
        if e.get("delegation_id") == "d1"
    ]
    assert 1 <= len(notes) <= 2, "rate-limited, and bounded per delegation"
    assert "Working on: research this" in notes[0]["content"]
    # EVERY note, not notes[0]. This assertion used to read only the START-phase
    # note, while the leak was in the END-phase one: `_InnerToolRelay` passed the
    # tool's own `preview` as the progress detail, so the tool's OUTPUT reached
    # the provider inside a note D9 says carries none (L1R-3).
    for note in notes:
        assert "found" not in note["content"], "no tool output in a progress note"
    # A progress note must never be spoken over the caller: the first
    # commentary on this delegation is the ANSWER.
    first_commentary = provider.of("session.commentary.append")[0]
    assert "Done." in first_commentary["content"]


@pytest.mark.asyncio
async def test_a_plain_play_skips_the_agent_turn_entirely(monkeypatch):
    H.fast_clocks(monkeypatch)
    thinks = []
    plays = []

    async def think(*args, **kwargs):
        thinks.append(args)
        return "agent answer", "m"

    async def play(_user_id, query, variety=False):
        plays.append((query, variety))
        return "Now playing: Halo.", {"type": "youtube", "video_id": "v1", "title": "Halo"}

    H.patch_relay(monkeypatch, think=think, play=play)

    class _Ask:
        query = "beyonce halo"
        variety = False
        mode = None

    import app.services.live_voice_protocol as live
    monkeypatch.setattr(live, "_media_request", lambda text: _Ask() if "halo" in text else None)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("play beyonce halo", 0, 500))
            provider.push(H.delegation("d1", 520))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.8, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send)
    recorded = H.patch_relay(monkeypatch, think=think, play=play)
    await H.run_relay(client, provider, timeout=6)

    assert plays == [("beyonce halo", False)]
    assert thinks == [], "the fast path must not also run an agent turn"
    assert [f["name"] for f in client.of("tool_call.started")] == ["play_media"]
    assert client.of("tool_call.completed")[0]["outcome"] == "ok"
    media_rows = [s for s in recorded["saves"] if s.get("media")]
    assert media_rows and media_rows[0]["media"]["video_id"] == "v1"
    voice = media_rows[0]["assistant_voice"]
    assert voice["source"] == "live_media"
    assert voice["request_turn_ids"] == [voice["parent_user_turn_id"]]
    from app.api.sessions import _clean_voice
    assert _clean_voice(dict(voice))["request_turn_ids"] == voice["request_turn_ids"]


@pytest.mark.asyncio
async def test_a_media_error_falls_through_to_the_ordinary_delegation(monkeypatch):
    H.fast_clocks(monkeypatch)
    thinks = []

    async def think(*args, **kwargs):
        thinks.append(args)
        return "I couldn't find that track, want another?", "m"

    async def play(_user_id, _query, _variety=False):
        return "ERROR: could not start that track.", None

    class _Ask:
        query = "nope"
        variety = False
        mode = None

    import app.services.live_voice_protocol as live
    monkeypatch.setattr(live, "_media_request", lambda text: _Ask())
    H.patch_relay(monkeypatch, think=think, play=play)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("play nope", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.8, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    assert len(thinks) == 1
    assert client.of("tool_call.completed")[0]["outcome"] == "error"


@pytest.mark.asyncio
async def test_a_delegation_never_closes_or_executes_an_open_turn(monkeypatch):
    """Provider delegation is model intent, not an utterance endpoint."""

    # The silence gap is far away: the delegation must remain queued.
    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=30000)
    seen = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        seen["display"] = kwargs.get("display_request")
        return "ok", "m"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("what is ", 0, 300))
            provider.push(H.user_delta("the weather", 320, 700))
            provider.push(H.delegation("d1", 750))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    def snapshot_before_stop():
        seen["before_stop"] = seen.get("display")

    client = H.FakeClient([
        H.config(), 0.4, snapshot_before_stop, {"type": "stop"},
    ])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    assert seen.get("before_stop") is None
    finals = [t for t in client.of("transcript") if t["final"]]
    assert len(finals) == 1
    assert finals[0]["text"] == "what is the weather"


@pytest.mark.asyncio
async def test_the_fast_path_uses_the_shared_media_predicate_when_it_exists(monkeypatch):
    """L1 codes against the interface, not against a local copy: if
    `app.agent.media_intent` is absent the fast path is simply off."""

    H.fast_clocks(monkeypatch)
    plays = []
    thinks = []

    async def think(*args, **kwargs):
        thinks.append(args)
        return "agent answer", "m"

    async def play(_user_id, query, variety=False):
        plays.append((query, variety))
        return "Now playing.", {"type": "youtube", "video_id": "v9", "title": "Halo"}

    import app.services.live_voice_protocol as live
    # Force a fresh resolution of the real module.
    monkeypatch.setattr(live, "_MEDIA_REQUEST", None, raising=False)
    monkeypatch.setattr(live, "_MEDIA_REQUEST_LOADED", False, raising=False)
    H.patch_relay(monkeypatch, think=think, play=play)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("play beyonce halo", 0, 500))
            provider.push(H.delegation("d1", 520))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.8, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    try:
        import app.agent.media_intent  # noqa: F401
    except ImportError:
        assert plays == [] and len(thinks) == 1
    else:
        assert plays == [("beyonce halo", False)]
        assert thinks == []


# ── Round 48 fix pass ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_an_identity_less_interrupt_still_tells_the_model_to_stop(monkeypatch):
    """L1R-1. The shipped client's barge-in frame carries no response_id and no
    item_id (`useRealtimeVoice.ts`, the A2-05 branch). The relay dropped it, so
    the commonest barge-in on Live produced no `instructions.append`, no fence
    and no `listening` — the agent talked straight through it."""

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=5000)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.out_text("Let me tell you all about it", 0, 400))
            provider.push(H.out_audio("SPEAKING"))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([
        H.config(), 0.3, {"type": "interrupt", "played_ms": 400}, 0.2, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=6)

    stops = [
        e for e in provider.of("session.instructions.append")
        if "Stop speaking" in e["content"]
    ]
    assert len(stops) == 1, "an identity-less interrupt must stop the model"
    assert "listening" in client.states()


@pytest.mark.asyncio
async def test_an_interrupt_after_the_gap_rotation_still_tells_the_model_to_stop(monkeypatch):
    """L1R-1. The gap rotation retires the epoch while the phone is still
    playing buffered audio, so the id the user barges in with is already in
    history. A barge-in on it used to be a complete no-op."""

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=60)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.out_text("A long reply", 0, 400))
            provider.push(H.out_audio("SPEAKING"))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    # 0.4 s is far past the 60 ms gap, so the epoch is retired by the tick
    # before the client — still playing what it buffered — interrupts it.
    client = H.FakeClient([
        H.config(), 0.4,
        {"type": "interrupt", "response_id": f"live:{H.PSID}:1",
         "item_id": f"live:{H.PSID}:1", "played_ms": 250},
        0.2, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send)
    recorded = H.patch_relay(monkeypatch)
    await H.run_relay(client, provider, timeout=6)

    stops = [
        e for e in provider.of("session.instructions.append")
        if "Stop speaking" in e["content"]
    ]
    assert len(stops) == 1, "a barge-in on an already-rotated epoch must stop the model"
    spoken = [
        s for s in recorded["saves"]
        if (s.get("assistant_voice") or {}).get("source") == "live_spoken"
    ]
    assert spoken, "the rotated epoch still owns a row"
    assert spoken[-1]["assistant_voice"]["interrupted"] is True
    assert spoken[-1]["assistant_voice"]["played_ms"] == 250


@pytest.mark.asyncio
async def test_cancel_task_speaks_the_honest_line_and_rejects_an_unknown_id(monkeypatch):
    """A real target is canceled and announced; an unknown id cannot mint a
    fake terminal transition that might overwrite a completed task."""

    H.fast_clocks(monkeypatch)
    started = asyncio.Event()

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        started.set()
        await asyncio.Future()

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("book the flight", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([
        H.config(), 0.6,
        {"type": "cancel_task", "delegation_id": "d1"},
        0.2,
        {"type": "cancel_task", "delegation_id": "never-ran"},
        0.2, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=8)

    assert started.is_set()
    spoken = " ".join(e["content"] for e in provider.of("session.commentary.append"))
    assert "Stopped." in spoken, "a cancel the user asked for is announced"
    unknown = [
        f for f in client.of("delegation")
        if f["delegation_id"] == "never-ran"
    ]
    assert unknown == []
    assert any(
        f.get("code") == "cancel_task_not_running"
        for f in client.of("error")
    )


@pytest.mark.asyncio
async def test_a_queued_delegation_is_terminalised_at_teardown(monkeypatch):
    """L1R-11. `supervisor.queued` was drained only by `finish_delegation`, so a
    third request that never got a slot left the client on the `pending` card it
    was told about — no terminal phase, no row, forever."""

    H.fast_clocks(monkeypatch, voice_live_max_concurrent_delegations=1)
    forever = asyncio.Event()

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        await forever.wait()
        return "never", "m"

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("book the flight to Paris", 0, 300))
            provider.push(H.delegation("d1", 320))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.5, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send)

    async def queue_a_second():
        # Nothing in common with the first request, so the follow-up classifier
        # reads it as NEW work rather than a correction — a correction would
        # supersede the first and bypass the concurrency cap this test needs.
        await asyncio.sleep(0.25)
        provider.push(H.user_delta("what temperature is it outside", 4000, 4400))
        provider.push(H.delegation("d2", 4420))

    task = asyncio.create_task(queue_a_second())
    await H.run_relay(client, provider, timeout=8)
    await asyncio.gather(task, return_exceptions=True)
    forever.set()

    queued = [f for f in client.of("delegation") if f["delegation_id"] == "d2"]
    assert any(f["phase"] == "pending" for f in queued), "the test must actually queue one"
    assert queued[-1]["phase"] == "cancelled"
    assert queued[-1].get("reason") == "session_ended"


@pytest.mark.asyncio
async def test_a_detached_failure_persists_but_says_nothing(monkeypatch):
    """L1R-12 / D4: detached work 'persists; no commentary'. The accepted branch
    was gated on a live client; the empty and failed branches were not, and were
    quiet only because `provider_send` swallows a write to a closed socket."""

    H.fast_clocks(monkeypatch)
    hung_up = asyncio.Event()

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        await asyncio.wait_for(hung_up.wait(), timeout=3)
        raise RuntimeError("the backend fell over after the caller left")

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("research this", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.6, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=8)
    before = len(provider.of("session.commentary.append"))
    hung_up.set()
    await asyncio.sleep(0.4)

    assert len(provider.of("session.commentary.append")) == before, (
        "a detached delegation must not push commentary after hang-up"
    )


@pytest.mark.asyncio
async def test_the_agent_turn_body_carries_the_delegation_id_the_row_is_stamped_with(monkeypatch):
    """C1. Delivery proof keys on the delegation, not on time: with two
    concurrent delegations 'an assistant row exists' is true for the OTHER
    one's answer too, so the id on the turn and the id on the row must match."""

    H.fast_clocks(monkeypatch)
    seen = {}

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        seen["delegation_id"] = kwargs.get("delegation_id")
        return "Booked.", "m"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("book it", 0, 300))
            provider.push(H.delegation("d-abc", 320))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.8, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    assert seen.get("delegation_id") == "d-abc"
    rows = [
        s for s in recorded["saves"]
        if (s.get("assistant_voice") or {}).get("source") == "delegated"
    ]
    assert rows and rows[-1]["assistant_voice"]["delegation_id"] == "d-abc"


@pytest.mark.asyncio
async def test_a_lowercased_media_error_still_falls_through_to_the_agent(monkeypatch):
    """C5.3. `startswith('ERROR')` on the raw string called ' error: no video'
    a SUCCESS and then spoke 'Playing .' over silence. `is_error_result` strips
    and upper-cases, and both fast paths must use the one predicate."""

    H.fast_clocks(monkeypatch)
    thinks = []

    async def think(*args, **kwargs):
        thinks.append(args)
        return "I could not find that track.", "m"

    async def play(_user_id, _query, _variety=False):
        return "  error: no video matched that query", None

    H.patch_relay(monkeypatch, think=think, play=play)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("play radiohead creep", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.9, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=6)

    assert thinks, "a failed direct play must fall through to the full delegation"
    spoken = " ".join(e["content"] for e in provider.of("session.commentary.append"))
    assert "Playing ." not in spoken


@pytest.mark.asyncio
async def test_a_play_with_no_title_speaks_the_result_not_an_empty_template(monkeypatch):
    """L1R-9. `relay_line` formats an empty title into 'Playing .', which is
    truthy — so the intended `or text` fallback was dead code."""

    H.fast_clocks(monkeypatch)

    async def play(_user_id, _query, _variety=False):
        return "Started the station.", {"video_id": "abc123"}

    H.patch_relay(monkeypatch, play=play)

    def on_send(provider, event):
        if event["type"] == "session.start":
            # PIN CHANGED (R2 addendum 6 R6-17): 'something' is in the closed
            # indefinite class and declines to the agent; a clean title keeps
            # this pin about the fast path's spoken result.
            provider.push(H.user_delta("play some jazz", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.9, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=6)

    spoken = " ".join(e["content"] for e in provider.of("session.commentary.append"))
    assert "Playing ." not in spoken
    assert "Started the station." in spoken


@pytest.mark.asyncio
async def test_an_interrupt_arriving_before_any_output_still_stops_the_model(monkeypatch):
    """L1R-1. The client sends its barge-in on ITS OWN playback state, which
    includes audio the relay has already retired or never tracked. Gating the
    stop instruction on the relay having something to close is the same defect
    in a smaller hole."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)

    def on_send(provider, event):
        if event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([
        H.config(), 0.2, {"type": "interrupt", "played_ms": 120}, 0.2, {"type": "stop"},
    ])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=6)

    stops = [
        e for e in provider.of("session.instructions.append")
        if "Stop speaking" in e["content"]
    ]
    assert len(stops) == 1


@pytest.mark.asyncio
async def test_the_agent_turn_http_body_carries_delegation_id(monkeypatch):
    """C1, at the wire. The relay-side pin above proves `_think` is CALLED with
    the id; this one proves the id reaches the agent route that reads it."""

    from app.api import ws_realtime as rt
    from app.config import settings

    monkeypatch.setattr(rt, "_agent_runner", None)
    monkeypatch.setattr(settings, "voice_realtime_v2", True, raising=False)
    monkeypatch.setattr(rt, "_v2_active", lambda *a, **k: True)
    monkeypatch.setattr(settings, "voice_realtime_tool_events", False, raising=False)

    class _Decision:
        model = "gpt-5.5"

    monkeypatch.setattr(
        "app.services.model_router.classify_request", lambda task: _Decision()
    )

    async def vps(_user_id):
        return ("https://u.agents.toup.ai", "agent-key")

    bodies = []

    async def vps_api(agent_url, key, method, path, params=None, json_body=None, timeout=15.0):
        bodies.append(json_body)
        return {"text": "Booked.", "model": "gpt-5.5"}

    monkeypatch.setattr(rt, "_get_vps_info", vps)
    monkeypatch.setattr(rt, "_vps_api", vps_api)

    await rt._think(
        "user-1", "book it", "sess-1",
        display_request="book it", reply_language="en", delegation_id="d-abc",
    )

    assert bodies and bodies[0]["delegation_id"] == "d-abc"

    from app.api.api_v1 import ChatRequest
    assert "delegation_id" in ChatRequest.model_fields, (
        "the field the relay sends must exist on the route that receives it"
    )


@pytest.mark.asyncio
async def test_a_detached_empty_answer_persists_but_says_nothing(monkeypatch):
    """L1R2-3. The `failed` half of the D4 gate dies under mutation; the
    `empty` half did not, because the only empty-answer test runs with a LIVE
    client, where `may_speak` is true either way. So the `empty` gate could be
    deleted and every check in this repo would stay green."""

    H.fast_clocks(monkeypatch)
    hung_up = asyncio.Event()

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        await asyncio.wait_for(hung_up.wait(), timeout=3)
        return "   ", "m"          # whitespace only ⇒ outcome `empty`

    H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("research this", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.6, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=8)
    before = len(provider.of("session.commentary.append"))
    hung_up.set()
    await asyncio.sleep(0.4)

    assert len(provider.of("session.commentary.append")) == before, (
        "a detached EMPTY delegation must not push commentary after hang-up"
    )


@pytest.mark.asyncio
async def test_a_detached_play_persists_its_card_but_says_nothing(monkeypatch):
    """L1R2-2. The same D4 rule on the third branch that speaks.
    `maybe_media_fast_path` is the FIRST await in `run_delegation` and
    `_play_media_direct` has its own timeout, so a caller who says 'play X' and
    hangs up immediately is inside it at teardown — `run()` detaches, it never
    cancels. Quiet only because `provider_send` swallows a write to a closing
    socket is not the rule, it is an accident of teardown order."""

    H.fast_clocks(monkeypatch)
    hung_up = asyncio.Event()
    import app.services.live_voice_protocol as live

    class _Req:
        query = "halo"
        variety = False
        mode = None

    monkeypatch.setattr(live, "_MEDIA_REQUEST", lambda _text: _Req(), raising=False)
    monkeypatch.setattr(live, "_MEDIA_REQUEST_LOADED", True, raising=False)

    async def play(_user_id, _query, _variety=False):
        await asyncio.wait_for(hung_up.wait(), timeout=3)
        return "Now playing.", {"type": "youtube", "video_id": "v9", "title": "Halo"}

    recorded = H.patch_relay(monkeypatch, play=play)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("play halo", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.6, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send)
    await H.run_relay(client, provider, timeout=8)
    before = len(provider.of("session.commentary.append"))
    hung_up.set()
    await asyncio.sleep(0.5)

    assert len(provider.of("session.commentary.append")) == before, (
        "a detached play must not push commentary after hang-up"
    )
    assert [s for s in recorded["saves"] if s.get("media")], (
        "the card itself is still persisted — detached work persists, it just "
        "says nothing"
    )


@pytest.mark.asyncio
async def test_the_fast_paths_log_says_the_mode_was_IGNORED_not_honoured(monkeypatch, caplog):
    """C5.12 / L1R2-5. `mode` is not plumbed: the phone's `surfaceForPlay()`
    answers 'song' during any live call, so a video ask mid-call is audio on
    purpose. The old `mode=%s` read as if the ask had been honoured."""

    H.fast_clocks(monkeypatch)
    import app.services.live_voice_protocol as live

    class _Req:
        query = "halo"
        variety = False
        mode = "video"

    monkeypatch.setattr(live, "_MEDIA_REQUEST", lambda _text: _Req(), raising=False)
    monkeypatch.setattr(live, "_MEDIA_REQUEST_LOADED", True, raising=False)

    async def play(_user_id, _query, _variety=False):
        return "Now playing.", {"type": "youtube", "video_id": "v9", "title": "Halo"}

    H.patch_relay(monkeypatch, play=play)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("watch the halo video", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    import logging

    client = H.FakeClient([H.config(), 0.8, {"type": "stop"}])
    with caplog.at_level(logging.INFO, logger=live.logger.name):
        await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    lines = [r.getMessage() for r in caplog.records if "media fast path" in r.getMessage()]
    assert lines, "the fast path must still say it ran"
    assert "mode_ignored=1" in lines[-1]
    assert "mode=" not in lines[-1].replace("mode_ignored=", "")


# ── L1R2-4: a step that never started is `not_run`, never a failure ───

@pytest.mark.asyncio
async def test_a_row_that_never_got_its_arguments_closes_as_not_run():
    """L1R2-4. `not_run` is the outcome D8/D14 tell the client to DROP the row
    for. Flattening it to `error` killed no test on either path, so a
    regression would put a phantom FAILED step on the phone for work that never
    started — the 'claim of failure about work that did not happen' class this
    round exists to remove. The Live path closes these rows from
    `finish_delegation`."""

    from app.api.ws_realtime import _InnerToolRelay

    class _WS:
        def __init__(self):
            self.sent: list[dict] = []

        async def send_json(self, frame):
            self.sent.append(frame)

    async def drive(close_outcome=None):
        ws = _WS()
        relay = _InnerToolRelay(ws, "outer1")
        # tool_use_start: the model named the step, the arguments have not
        # landed, so the row is provisional and the work never began.
        await relay.on_event({"type": "tool.intent", "name": "web_search"})
        if close_outcome is None:
            await relay.close_open()
        else:
            await relay.close_open(outcome=close_outcome)
        return [f for f in ws.sent if f.get("type") == "tool_call.completed"]

    ended = await drive()
    cancelled = await drive("cancelled")

    assert [f["outcome"] for f in ended] == ["not_run"]
    assert [f["outcome"] for f in cancelled] == ["cancelled"], (
        "a turn the user stopped is not a step that never started"
    )


@pytest.mark.asyncio
async def test_a_tool_only_delegated_row_carries_the_line_the_turn_spoke(monkeypatch):
    """L1V-2. A delegation that ran tools and came back with no words and no
    card submits a row whose whole content is the tool trail — and the tenant
    route answers 400 "Content is required" to an empty `content`. Because the
    Live persistence queue is serialized by design, those three doomed POSTs
    plus their backoff sit in front of every later transcript row for the
    session. The row carries the very line this turn spoke aloud, from the
    fixed en/fa table, so it lands instead of blocking the queue."""

    H.fast_clocks(monkeypatch)

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        await _tool_start(relay)
        await _tool_end(relay)
        return "", "m"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("check the weather", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.9, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    import app.services.live_voice_protocol as live

    rows = [s for s in recorded["saves"] if s.get("tool_events")]
    assert rows, "the tool trail needs an authorizing row"
    assert rows[-1]["assistant_text"] == live.relay_line("delegation_empty", "en"), (
        "a payload-bearing row with empty content is rejected by the receiving route"
    )


@pytest.mark.asyncio
async def test_an_empty_delegation_with_nothing_to_carry_still_writes_nothing(monkeypatch):
    """The other half of L1V-2's rule. The fixed-table line exists to make a
    PAYLOAD landable, never to give an empty turn a row of its own — otherwise
    every failed delegation narrates itself into the day chat."""

    H.fast_clocks(monkeypatch)

    async def think(*args, **kwargs):
        return "", "m"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("do the thing", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.9, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    # Filtered on what the row IS, not on how it was labelled (L1V3-R1): every
    # row this branch can produce is keyed `live-delegation:<psid>:<id>`, so a
    # future provenance value cannot make the escaped row invisible and leave
    # this assertion passing over a gate that is gone.
    delegated = [
        s for s in recorded["saves"]
        if str(s.get("assistant_ref") or "").startswith("live-delegation:")
    ]
    assert delegated == [], "nothing was produced, so there is no row to authorize"


@pytest.mark.asyncio
async def test_a_relay_authored_failure_line_is_not_stamped_as_a_delivered_answer(
        monkeypatch):
    """L1V2-R1. The fallback line that makes a crashed turn's tool trail
    landable must not also make the job card claim success.

    `voice_jobs.VoiceTurnJob._row_is_proof` accepts `source == 'delegated'`
    with neither flag set and a matching id, and `_close` polls ~6 s for a
    late answer row precisely to catch this POST — so the row whose whole
    content is "That didn't go through" closed the card COMPLETED for a turn
    that crashed. It carries a distinct source instead. The delivered answer's
    row is untouched: both halves are asserted here because they are one rule.
    """

    H.fast_clocks(monkeypatch)

    async def raising_think(user_id, task, session_id, relay=None, out=None, **kwargs):
        await _tool_start(relay)
        await _tool_end(relay)
        raise RuntimeError("agent route 500")

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("check the weather", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    import app.services.live_voice_protocol as live

    recorded = H.patch_relay(monkeypatch, think=raising_think)
    client = H.FakeClient([H.config(), 0.9, {"type": "stop"}])
    await H.run_relay(
        client, H.FakeProvider(on_send=on_send), timeout=6, db_session_id="db-fail",
    )

    rows = [s for s in recorded["saves"] if s.get("tool_events")]
    assert rows, "the tool trail still needs an authorizing row"
    assert rows[-1]["assistant_text"] == live.relay_line("delegation_failed", "en")
    assert rows[-1]["assistant_voice"]["source"] == "delegated_record", (
        "a row the RELAY wrote for a turn that never answered is a record, not "
        "the delivery proof that closes the job card as completed"
    )

    # …and the answer that WAS delivered keeps the provenance the proof reads.
    async def answering_think(*args, **kwargs):
        return "Fourteen degrees and clear.", "m"

    recorded2 = H.patch_relay(monkeypatch, think=answering_think)
    client2 = H.FakeClient([H.config(), 0.9, {"type": "stop"}])
    await H.run_relay(
        client2, H.FakeProvider(on_send=on_send), timeout=6, db_session_id="db-ok",
    )

    answered = [
        s for s in recorded2["saves"]
        if (s.get("assistant_voice") or {}).get("delegation_id") == "d1"
        and s.get("assistant_text")
    ]
    assert answered, recorded2["saves"]
    assert answered[-1]["assistant_voice"]["source"] == "delegated", (
        "demoting the delivered answer too would make every voice turn close "
        "cancelled — the same lie in the other direction"
    )
