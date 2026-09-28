"""What a Live call leaves behind in the day chat.

Before round 48 the user side wrote one durable row per 350 ms transcript
settle (29 rows of 2-20 characters for one request in the founder's recording)
and the assistant side wrote nothing at all unless the turn happened to produce
a tool event.  These tests pin the replacement contract.
"""

import asyncio

import pytest

import test_live_harness as H


def _user_rows(saves):
    return [s for s in saves if s.get("user_text")]


def _assistant_rows(saves):
    return [s for s in saves if s.get("assistant_text")]


@pytest.mark.asyncio
async def test_three_settles_are_one_turn_one_key_and_one_evolving_bubble(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=5000)
    recorded = H.patch_relay(monkeypatch)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("که ", 0, 300))
            provider.push(H.user_delta("ایونت ", 700, 1100))
            provider.push(H.user_delta("معروف", 1500, 1900))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.5, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    rows = _user_rows(recorded["saves"])
    assert rows, "the utterance must be durable well before the call ends"
    assert len({r["user_ref"] for r in rows}) == 1, "one key, revised in place"
    assert rows[-1]["user_text"] == "که ایونت معروف"
    revisions = [r["user_revision"] for r in rows]
    assert revisions == sorted(revisions)
    voice = rows[-1]["user_voice"]
    assert voice["source"] == "live_user"
    assert voice["turn_id"] == rows[-1]["user_ref"]
    assert voice["clock"] == "provider"
    assert voice["start_ms"] == 0
    assert voice["end_ms"] == 1900

    # The live caption still moves at the settle cadence: partial frames with a
    # STABLE turn_id, then exactly one final.
    transcripts = client.of("transcript")
    assert len({t["turn_id"] for t in transcripts}) == 1
    assert [t["final"] for t in transcripts].count(True) == 1
    assert transcripts[-1]["text"] == "که ایونت معروف"


@pytest.mark.asyncio
async def test_a_silence_gap_opens_a_second_turn_with_its_own_key(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=400)
    recorded = H.patch_relay(monkeypatch)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("first request", 0, 400))
            provider.push(H.user_delta("second request", 3000, 3400))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.6, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    refs = {r["user_ref"] for r in _user_rows(recorded["saves"])}
    assert len(refs) == 2
    assert {t["turn_id"] for t in client.of("transcript")} == refs


@pytest.mark.asyncio
async def test_occurred_at_is_anchored_to_the_provider_timeline_and_monotonic(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=400)

    async def think(*args, **kwargs):
        return "The answer.", "m"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("question one", 0, 400))
            provider.push(H.user_delta("question two", 3000, 3400))
            provider.push(H.delegation("d1", 3500))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 1.0, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    stamps = [
        s.get("user_occurred_at") or s.get("assistant_occurred_at")
        for s in recorded["saves"]
        if s.get("user_occurred_at") or s.get("assistant_occurred_at")
    ]
    assert stamps and all(s is not None for s in stamps)
    assert stamps == sorted(stamps), "the answer may never sort above its question"
    # The second utterance is ~3 s later on the provider timeline, and its
    # stamp says so even though both writes land within milliseconds.
    firsts = [s["user_occurred_at"] for s in _user_rows(recorded["saves"])]
    assert (max(firsts) - min(firsts)).total_seconds() >= 2.5


@pytest.mark.asyncio
async def test_persistence_is_serialized_one_write_per_key_in_flight(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=5000)
    inflight = {"now": 0, "max": 0}
    order: list[str] = []

    async def slow_save(user_id, session_id, **kwargs):
        inflight["now"] += 1
        inflight["max"] = max(inflight["max"], inflight["now"])
        await asyncio.sleep(0.02)
        order.append(kwargs.get("user_ref") or kwargs.get("assistant_ref") or "?")
        inflight["now"] -= 1
        return 0

    H.patch_relay(monkeypatch, save=slow_save)

    def on_send(provider, event):
        if event["type"] == "session.start":
            for i in range(6):
                provider.push(H.user_delta(f"w{i} ", i * 200, i * 200 + 150))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.6, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    assert inflight["max"] == 1, "two revisions of one key must never overlap"
    assert order == sorted(order, key=order.index)


@pytest.mark.asyncio
async def test_a_failed_write_is_retried_onto_the_same_row(monkeypatch):
    """The ladder itself, driven directly.

    This test used to run a whole session and assert `len(attempts) >= 2` with
    one distinct key — which ONE settle write plus ONE final write for the same
    turn already satisfies, with `_write` never looping. `MAX_ATTEMPTS = 1`
    killed no test. Now exactly one record is submitted, so every attempt after
    the first IS a retry.
    """

    import app.services.live_voice_protocol as live

    monkeypatch.setattr(live.PersistenceQueue, "BACKOFF_S", (0.0, 0.0, 0.0))
    attempts: list[str] = []

    async def flaky(user_id, session_id, **kwargs):
        ref = kwargs.get("user_ref") or kwargs.get("assistant_ref") or "?"
        attempts.append(ref)
        return 1 if len(attempts) <= 2 else 0     # two losses, then it lands

    H.patch_relay(monkeypatch, save=flaky)

    queue = live.PersistenceQueue("user-1", "db-session")
    queue.start()
    queue.submit(live.PersistenceRecord(
        kind="message",
        key="live-utt:live-psid:1",
        payload={"user_text": "only sentence", "assistant_text": "",
                 "user_ref": "live-utt:live-psid:1"},
    ))
    assert await queue.drain(3.0)
    await queue.stop()

    assert attempts == ["live-utt:live-psid:1"] * 3, (
        "three attempts on ONE key: a retry upserts, it never mints a new key"
    )
    assert queue.lost == 0 and queue.written == 1


@pytest.mark.asyncio
async def test_a_write_that_never_lands_is_counted_lost_not_pretended_written(monkeypatch):
    """The negative half. Pretending history was saved is the failure mode this
    whole module exists to remove, so the exhausted ladder has to be loud."""

    import app.services.live_voice_protocol as live

    monkeypatch.setattr(live.PersistenceQueue, "BACKOFF_S", (0.0, 0.0, 0.0))
    attempts: list[str] = []

    async def always_lost(user_id, session_id, **kwargs):
        attempts.append(kwargs.get("assistant_ref") or "?")
        return 1

    H.patch_relay(monkeypatch, save=always_lost)

    queue = live.PersistenceQueue("user-1", "db-session")
    queue.start()
    queue.submit(live.PersistenceRecord(
        kind="message",
        key="live-output:live-psid:1",
        payload={"user_text": "", "assistant_text": "an answer",
                 "assistant_ref": "live-output:live-psid:1"},
    ))
    assert await queue.drain(3.0)
    await queue.stop()

    # The literal 3, not `MAX_ATTEMPTS`: reading the constant back makes the
    # assertion follow whatever the constant becomes, which is no assertion.
    assert len(attempts) == 3
    assert queue.lost == 1 and queue.written == 0


@pytest.mark.asyncio
async def test_a_delegated_answer_persists_even_with_no_tool_evidence(monkeypatch):
    """A4-1 / A1-10: recording 1's 172-character answer had zero tool calls and
    was therefore never written — the thread kept every user fragment and none
    of the replies."""

    H.fast_clocks(monkeypatch)

    async def think(*args, **kwargs):
        return "Yes, the deadline is in January.", "deep-model"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("when is the deadline", 0, 600))
            provider.push(H.delegation("d1", 620))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.8, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    rows = _assistant_rows(recorded["saves"])
    answer = next(r for r in rows if r["assistant_text"].startswith("Yes,"))
    assert answer["tool_events"] is None
    assert {
        key: answer["assistant_voice"][key]
        for key in ("source", "delegation_id", "model", "superseded", "cancelled")
    } == {
        "source": "delegated",
        "delegation_id": "d1",
        "model": "deep-model",
        "superseded": False,
        "cancelled": False,
    }
    assert answer["assistant_voice"]["task_id"] == "d1"
    assert answer["assistant_voice"]["parent_user_turn_id"].startswith("live-utt:")
    assert answer["assistant_voice"]["clock"] == "provider"
    assert answer["assistant_ref"].endswith(":d1")


@pytest.mark.asyncio
async def test_the_last_words_before_hangup_are_flushed(monkeypatch):
    """A4-6: `settle` early-returned on `closed`, so whatever was said in the
    last window — usually the request they just made — was dropped."""

    H.fast_clocks(monkeypatch, voice_live_transcript_settle_ms=3000,
                  voice_live_utterance_gap_ms=9000)
    recorded = H.patch_relay(monkeypatch)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("one last thing", 0, 400))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.15, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    rows = _user_rows(recorded["saves"])
    assert [r["user_text"] for r in rows] == ["one last thing"]
    assert rows[-1]["user_voice"]["source"] == "live_user"
    assert rows[-1]["user_voice"]["turn_id"] == rows[-1]["user_ref"]
    assert rows[-1]["user_voice"]["clock"] == "provider"


@pytest.mark.asyncio
async def test_a_spoken_segment_records_what_was_actually_heard(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=3000)
    recorded = H.patch_relay(monkeypatch)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.out_text("Sure, here is the plan.", 100, 900))
            provider.push(H.out_audio())
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([
        H.config(), 0.3,
        {"type": "playback_idle", "response_id": f"live:{H.PSID}:1",
         "item_id": f"live:{H.PSID}:1", "played_ms": 640},
        0.1, {"type": "stop"},
    ])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    spoken = [
        s for s in recorded["saves"]
        if (s.get("assistant_voice") or {}).get("source") == "live_spoken"
    ]
    assert len(spoken) == 1
    voice = spoken[0]["assistant_voice"]
    assert voice["played_ms"] == 640
    assert voice["interrupted"] is False
    assert voice["generated_chars"] == len("Sure, here is the plan.")
    assert spoken[0]["assistant_ref"] == f"live-output:{H.PSID}:1"


@pytest.mark.asyncio
async def test_the_memory_curator_runs_once_per_closed_turn(monkeypatch):
    """A1-18: the Live path inherited `disable_post_processing=True` but not
    the compensating curator call, so a call wrote no memories at all."""

    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=300,
                  auto_extract_memories=True)
    recorded = H.patch_relay(monkeypatch)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("I prefer mornings", 0, 500))
            provider.push(H.user_delta("and I hate coriander", 3000, 3600))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.8, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    texts = [c[1] for c in recorded["curates"]]
    assert texts == ["I prefer mornings", "and I hate coriander"]
    # Never the wrapped scaffolding string — the 409A class of invented memory.
    assert all("Live-session context" not in t for t in texts)


@pytest.mark.asyncio
async def test_the_final_write_outranks_the_last_partial(monkeypatch):
    H.fast_clocks(monkeypatch, voice_live_utterance_gap_ms=250)
    recorded = H.patch_relay(monkeypatch)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("one ", 0, 200))
            provider.push(H.user_delta("two", 220, 400))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.8, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    rows = _user_rows(recorded["saves"])
    assert len(rows) >= 2
    assert rows[-1]["user_revision"] > rows[-2]["user_revision"]
    finals = [t for t in client.of("transcript") if t["final"]]
    assert finals[-1]["revision"] == rows[-1]["user_revision"]


def test_every_voice_key_the_relay_sends_survives_the_tenant_allowlist():
    """`sessions._clean_voice` drops any key it does not recognise. A relay
    that sends one believes it wrote a field nobody stored — the silent class
    this round exists to remove."""

    from app.api.sessions import _VOICE_KEYS

    sent = {
        "live_user": {"source"},
        "live_spoken": {"source", "epoch", "played_ms", "interrupted", "generated_chars"},
        "delegated": {"source", "delegation_id", "model", "superseded", "cancelled"},
        # Same keys, different provenance: the row the relay writes for a turn
        # that did NOT deliver, which `voice_jobs._row_is_proof` must reject
        # (L1V2-R1). It is listed because this is the relay's source inventory.
        "delegated_record": {"source", "delegation_id", "model", "superseded", "cancelled"},
        "live_media": {"source", "delegation_id"},
    }
    for source, keys in sent.items():
        assert keys <= _VOICE_KEYS, f"{source} sends keys the tenant drops: {keys - _VOICE_KEYS}"


# ── Round 48 fix pass ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_delegated_answer_is_written_once_not_twice_in_other_words(monkeypatch):
    """L1R-2 / D13. `persist_spoken`'s own docstring named the exception — the
    paraphrase of a delegated answer — and no such check existed, so one answer
    landed in the day chat twice: the verified text on the delegated row, and
    GPT-Live's rewording of it on a `live_spoken` row a second later."""

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=60)
    answer = "Admissions open in September."

    async def think(*args, **kwargs):
        return answer, "test-model"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("when do admissions open", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.commentary.append":
            # GPT-Live says the answer back in its own words.
            provider.push(H.out_text("Admissions are open from September.", 900, 1500))
            provider.push(H.out_audio("SPEAKING"))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 1.0, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=8)

    sources = [
        (s.get("assistant_voice") or {}).get("source")
        for s in _assistant_rows(recorded["saves"])
    ]
    assert "delegated" in sources
    assert "live_spoken" not in sources, (
        "the paraphrase of an accepted delegation is not a second assistant row"
    )
    # What the epoch DID add — how much of it was heard — lands on the
    # delegated row instead of being thrown away.
    delegated = [
        s for s in recorded["saves"]
        if (s.get("assistant_voice") or {}).get("source") == "delegated"
    ]
    assert "interrupted" in delegated[-1]["assistant_voice"]


@pytest.mark.asyncio
async def test_a_spoken_epoch_that_answers_nobody_still_gets_its_own_row(monkeypatch):
    """The other half of the same rule: ordinary Live speech, with no delegation
    behind it, must still be persisted. A blanket skip would delete the whole
    assistant side of a small-talk call."""

    H.fast_clocks(monkeypatch, voice_live_output_epoch_gap_ms=60)
    recorded = H.patch_relay(monkeypatch)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.out_text("Good morning.", 0, 400))
            provider.push(H.out_audio("SPEAKING"))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.5, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    sources = [
        (s.get("assistant_voice") or {}).get("source")
        for s in _assistant_rows(recorded["saves"])
    ]
    assert "live_spoken" in sources


@pytest.mark.asyncio
async def test_the_curator_is_given_the_answer_that_turn_produced(monkeypatch):
    """L1R-6. `curate()` ran from `close_turn`, which is always BEFORE any
    delegation for that turn can finish, so `utterance.answer` was empty at the
    one call site that had an answer to give and the field was dead."""

    H.fast_clocks(monkeypatch, auto_extract_memories=True)
    landed = asyncio.Event()
    seen: list[tuple] = []

    async def think(*args, **kwargs):
        return "Your dentist is on Thursday.", "m"

    async def curate(user_id, user_text, assistant_text):
        seen.append((user_text, assistant_text))
        landed.set()

    H.patch_relay(monkeypatch, think=think, curate=curate)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("when is my dentist", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def wait_for_curation():
        # DURING the call, not at hang-up: this blocks the `stop` frame, so a
        # relay that only curates in its teardown never gets one and this test
        # times out rather than passing on the wrong path.
        await asyncio.wait_for(landed.wait(), timeout=3.0)

    client = H.FakeClient([H.config(), wait_for_curation, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=8)

    assert seen, "the turn must still be curated"
    assert ("when is my dentist", "Your dentist is on Thursday.") in seen


@pytest.mark.asyncio
async def test_a_turn_flushed_at_hang_up_dispatches_pending_work_and_is_curated(monkeypatch):
    """Hang-up is a valid final boundary; accepted work survives the socket."""

    # A settle that never fires parks BOTH in-call close paths (the gap close
    # and `dispatch_pending`'s), so the turn is still open — with a delegation
    # pending — when `stop` arrives and flushes it.
    H.fast_clocks(monkeypatch, auto_extract_memories=True,
                  voice_live_transcript_settle_ms=30000,
                  voice_live_delegation_ttl_s=30.0)
    seen: list[str] = []

    async def curate(user_id, user_text, assistant_text):
        seen.append(user_text)

    thinks = []

    async def think(*args, **kwargs):
        thinks.append(args)
        return "never", "m"

    H.patch_relay(monkeypatch, curate=curate, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("i live in toronto now", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.3, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=8)

    assert len(thinks) == 1, "hang-up finalizes and detaches accepted work"
    assert seen == ["i live in toronto now"]


@pytest.mark.asyncio
async def test_a_delegated_play_persists_its_card_with_no_text_and_no_tools(monkeypatch):
    """C5.4. The persistence gate was `answer or tool_events`; a delegated play
    returns a media card with neither, so the card was never written."""

    H.fast_clocks(monkeypatch)

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        if out is not None:
            out["media"] = {"video_id": "abc123", "title": "Halo"}
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

    rows = [s for s in recorded["saves"] if s.get("media")]
    assert rows and rows[-1]["media"]["video_id"] == "abc123"


@pytest.mark.asyncio
async def test_the_session_id_frame_names_the_local_day_it_bound(monkeypatch):
    """C5.8. The client caches the session id PER LOCAL DAY and falls back to
    the DEVICE's today when the field is absent — the wrong day for a call that
    crosses midnight, and the reconnect then orphans the thread. The Realtime
    path has always sent it."""

    H.fast_clocks(monkeypatch)
    H.patch_relay(monkeypatch)

    from app.api import ws_realtime as rt

    async def tz(_user_id):
        return "America/Toronto"

    monkeypatch.setattr(rt, "_get_user_tz_name", tz)

    def on_send(provider, event):
        if event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.2, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    frames = client.of("session_id")
    assert frames, "the frame itself must still be sent"
    assert frames[0]["day_date"] == rt._local_today_str("America/Toronto")


# ── C9: a turn that answers by DOING still leaves a row ────────────────

@pytest.mark.asyncio
async def test_a_card_only_assistant_row_is_written_not_silently_dropped(monkeypatch):
    """C9 / L3C-R3. A delegated turn can answer with a media card and no words.
    `_save_voice_messages` tested the two TEXT strings and nothing else, so it
    returned 0 without writing anything — and `PersistenceQueue` reads 0 as
    success, so the card never reached the thread while the queue reported it
    written. C1's delivery proof then found no row and closed the job card
    `cancelled` for music the user heard start."""

    from app.api import ws_realtime as rt

    sent: list[dict] = []

    async def vps_api(agent_url, key, method, path, params=None, json_body=None, **kw):
        sent.append(dict(json_body or {}))
        return {"ok": True}

    async def vps_info(_user_id):
        return ("https://agent.example", "key")

    monkeypatch.setattr(rt, "_vps_api", vps_api)
    monkeypatch.setattr(rt, "_get_vps_info", vps_info)

    media = {"type": "youtube", "video_id": "abc123", "title": "Halo"}
    lost = await rt._save_voice_messages(
        "user-1", "sess-1", "", "", media=media,
    )

    assert lost == 0
    assert [m["role"] for m in sent] == ["assistant"], (
        "the card rides the assistant row; the empty user row is still skipped"
    )
    assert sent[0]["media"] == media


@pytest.mark.asyncio
async def test_an_entirely_empty_pair_still_writes_nothing(monkeypatch):
    """The other half of C9's rule: emptiness is about the ROW, and a record
    with neither text nor payload still has nothing to say."""

    from app.api import ws_realtime as rt

    calls: list[str] = []

    async def vps_info(_user_id):
        calls.append("vps")
        return ("https://agent.example", "key")

    monkeypatch.setattr(rt, "_get_vps_info", vps_info)

    assert await rt._save_voice_messages("user-1", "sess-1", "", "") == 0
    assert calls == [], "it must not even ask where the agent is"


@pytest.mark.asyncio
async def test_a_record_that_names_no_row_is_never_counted_written(monkeypatch):
    """C9's bookkeeping half. `_save_voice_messages` returns the number of rows
    that FAILED, so a record it declines to write and one it wrote perfectly
    both return 0 — and the queue counted both as `written`. Reporting a write
    that never happened is the exact failure this queue exists to remove."""

    import app.services.live_voice_protocol as live

    saves: list[dict] = []

    async def save(user_id, session_id, **kwargs):
        saves.append(kwargs)
        return 0

    H.patch_relay(monkeypatch, save=save)

    queue = live.PersistenceQueue("user-1", "db-session")
    queue.start()
    queue.submit(live.PersistenceRecord(
        kind="message",
        key="live-delegation:live-psid:d1",
        payload={"user_text": "", "assistant_text": "",
                 "assistant_ref": "live-delegation:live-psid:d1"},
    ))
    assert await queue.drain(3.0)
    await queue.stop()

    assert saves == [], "there was nothing to write, so nothing was attempted"
    assert queue.written == 0 and queue.lost == 1


@pytest.mark.asyncio
async def test_a_delegated_play_row_carries_text_the_receiving_route_accepts(monkeypatch):
    """The delivery half of C9. The tenant route answers 400 "Content is
    required" to an empty `content` — true of the DEPLOYED image too — so a
    card-only delegated answer needs a line on its row, and it is the same
    fixed-table line the media fast path already writes for this situation."""

    H.fast_clocks(monkeypatch)

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        if out is not None:
            out["media"] = {"video_id": "abc123", "title": "Halo"}
        return "", "m"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("play halo", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.9, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    import app.services.live_voice_protocol as live

    rows = [s for s in recorded["saves"] if s.get("media")]
    assert rows, "the card still has to be persisted"
    assert rows[-1]["assistant_text"] == live.relay_line(
        "media_playing", "en", title="Halo",
    )


# ── L1R2-1: two answers in one round trip claim two epochs ─────────────

def test_two_commentary_claims_bind_to_the_two_epochs_actually_minted():
    """L1R2-1. The claim used to be an absolute number (`epoch + 1`) computed
    before `speak()`. Two delegations finishing inside one provider round trip
    — a designed-for state, `max_concurrent` is 2 — computed the SAME number,
    so the first claim was overwritten: its paraphrase was then persisted as
    its own `live_spoken` row, i.e. the same answer twice in the day chat."""

    import app.services.live_voice_protocol as live

    adapter = live.LiveClientAdapter("db-session")
    adapter.claim_next_output({"ref": "live-delegation:p:d1",
                               "payload": {"which": "d1"}, "revision": 1})
    adapter.claim_next_output({"ref": "live-delegation:p:d2",
                               "payload": {"which": "d2"}, "revision": 1})

    adapter.provider_event(
        {"type": "session.output_transcript.delta", "delta": "first answer"}, now=0.0,
    )
    first = adapter.epoch
    adapter.close_for_new_output("commentary")
    adapter.provider_event(
        {"type": "session.output_transcript.delta", "delta": "second answer"}, now=1.0,
    )
    second = adapter.epoch

    assert first != second
    assert sorted(adapter.commentary_epochs) == [first, second]
    assert adapter.commentary_epochs[first]["payload"]["which"] == "d1"
    assert adapter.commentary_epochs[second]["payload"]["which"] == "d2"


# ── C9, the sad path: an unreachable agent must not report a phantom write ──

@pytest.mark.asyncio
async def test_an_unreachable_agent_reports_a_card_only_row_as_lost(monkeypatch):
    """C9's second clause, one branch lower than the queue. The bail-out taken
    when `_get_vps_info` answers None counted the assistant row by its TEXT, so
    a card-only row returned 0 — `_save_voice_messages`' word for "nothing
    failed". `PersistenceQueue` then incremented `written`, no retry ran and no
    counter fired for a row that was never written. `_get_vps_info` swallows
    every exception into None, so a pool timeout at persist time is enough to
    reach this; the card itself was produced moments earlier when the lookup
    still worked."""

    from app.api import ws_realtime as rt

    async def no_vps(_user_id):
        return None

    monkeypatch.setattr(rt, "_get_vps_info", no_vps)

    lost = await rt._save_voice_messages(
        "user-1", "sess-1", "", "",
        media={"type": "youtube", "video_id": "abc123", "title": "Halo"},
    )

    assert lost == 1, "a card-only row that did not land is one row lost"


@pytest.mark.asyncio
async def test_a_card_only_row_the_agent_never_took_is_counted_lost_not_written(monkeypatch):
    """The same defect through the real queue, which is where it does harm:
    `written += 1` for a row the thread never got, so C1's delivery proof finds
    nothing and the job card closes `cancelled` for music the user heard."""

    import app.services.live_voice_protocol as live
    from app.api import ws_realtime as rt

    async def no_vps(_user_id):
        return None

    # Deliberately NOT stubbing `_save_voice_messages`: the seam under test is
    # the number it returns to the queue.
    monkeypatch.setattr(rt, "_get_vps_info", no_vps)
    monkeypatch.setattr(live.PersistenceQueue, "BACKOFF_S", (0.0, 0.0, 0.0))

    queue = live.PersistenceQueue("user-1", "db-session")
    queue.start()
    queue.submit(live.PersistenceRecord(
        kind="message",
        key="live-delegation:live-psid:d1",
        payload={
            "user_text": "",
            "assistant_text": "",
            "media": {"type": "youtube", "video_id": "abc123", "title": "Halo"},
            "assistant_ref": "live-delegation:live-psid:d1",
        },
    ))
    assert await queue.drain(5.0)
    await queue.stop()

    assert queue.written == 0, "nothing was written, so nothing may be counted written"
    assert queue.lost == 1


@pytest.mark.asyncio
async def test_a_file_only_delegated_answer_is_submitted_at_all(monkeypatch):
    """L1V-3. The submit gate listed text, tool events and media; the payload it
    builds also carries `attachments` and `app_artifact`. A delegated turn whose
    whole answer is a file therefore reached NEITHER of the two emptiness rules
    written to agree with each other. The gate is now that same predicate."""

    H.fast_clocks(monkeypatch)

    async def think(user_id, task, session_id, relay=None, out=None, **kwargs):
        if out is not None:
            out["attachments"] = [{"name": "report.md", "url": "/f/report.md"}]
        return "", "m"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("write that up", 0, 400))
            provider.push(H.delegation("d1", 420))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    client = H.FakeClient([H.config(), 0.9, {"type": "stop"}])
    await H.run_relay(client, H.FakeProvider(on_send=on_send), timeout=6)

    import app.services.live_voice_protocol as live

    rows = [s for s in recorded["saves"] if s.get("attachments")]
    assert rows, "the file needs an authorizing row or it is not in the thread"
    # And it is landable: the receiving route rejects an empty `content`.
    assert rows[-1]["assistant_text"] == live.relay_line("delegation_empty", "en")
