"""The app's REAL close frames, replayed against the REAL relay.

This is the one test that closes the hangup-parity loop. Everything else in
this repair proves one side: the app guards execute the app's close-frame
helper and assert its output, and `test_live_hangup_parity.py` drives the relay
with frames a Python test authored. Neither proves the two AGREE — an app that
emits a correct-looking frame the relay ignores, or a relay that honours a frame
shape the app never sends, passes both halves and still loses the caller's
words.

So the fixture read here is not a restatement of the contract. It is generated
by EXECUTING the shipped `closeCallFrames` helper in
`src/shared/voice/liveProtocol.ts`, vendored byte-identically into both repos
and pinned to one sha256 — the same discipline the spoken-canonicalization
corpus uses. Editing either copy without the other fails on both sides.

The supervisor's `/private/tmp/toup-supervisor-hangup-repro.py` models a BARE
stop and must keep failing: it is the before picture, and the relay must never
invent a receipt nobody sent. This file is the after picture, on the same wire.
"""
import asyncio
import hashlib
import json
import pathlib

import pytest

import test_live_harness as H
from app.schemas import public_heard_text

CLOSE_FRAMES = pathlib.Path(__file__).parent / "fixtures" / "close-frames-fixtures.json"
CLOSE_FRAMES_SHA256 = "996eb27142deefa101d700d7683df307d3c2dda9f2777899ddfdde15a59505be"

GENERATED = "First sentence. Second sentence."


def _fixture():
    raw = CLOSE_FRAMES.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CLOSE_FRAMES_SHA256, (
        "the vendored close-frame fixture changed. Regenerate it by EXECUTING the "
        "app helper, copy it to both repos, and update CLOSE_FRAMES_SHA256 here and "
        "in scripts/check-live-voice-protocol.js — or the app and the relay are no "
        "longer pinned to one another."
    )
    return {case["name"]: case for case in json.loads(raw)}


def test_the_vendored_close_frames_are_the_ones_the_app_pins():
    cases = _fixture()
    assert len(cases) >= 10, "the close-frame matrix shrank"
    # The shapes this suite depends on, asserted rather than assumed.
    partial = cases["live-epoch-partial-prefix"]["frames"]
    assert [f["type"] for f in partial] == ["interrupt", "stop"]
    assert partial[0]["reason"] == "client_request"
    assert partial[0]["heard_text"] == "First sentence."
    assert "heard_text" not in cases["client-cannot-say"]["frames"][0], (
        "an absent receipt must stay absent — it is not the same as an empty one"
    )
    assert [f["type"] for f in cases["no-epoch-playing"]["frames"]] == ["stop"]


async def _hangup_with(frames, *, generated=GENERATED, monkeypatch):
    """Play one reply, then end the call with exactly these app frames."""

    H.fast_clocks(monkeypatch)
    ledger = H.patch_relay(monkeypatch)

    def on_send(provider, event):
        if event["type"] == "session.start":
            provider.push(H.user_delta("Explain this briefly.", 0, 200))
            provider.push(H.out_text(generated, 250, 650))
        elif event["type"] == "session.close":
            provider.push(H.closed())

    async def caller_ends_mid_reply():
        for _ in range(200):
            if client.of("response_text"):
                break
            await asyncio.sleep(0.01)
        assert client.of("response_text"), "the reply never reached the phone"
        # Let the relay's own idle timer retire the epoch, exactly as the
        # supervisor's reproducer does, so this is the same wire.
        await asyncio.sleep(0.2)

    script = [dict(H.config(), features=[*H.config()["features"], "heard_text"]),
              caller_ends_mid_reply]
    # The app's frames, verbatim, with the identity bound to this session.
    oid = f"live:{H.PSID}:1"
    for frame in frames:
        sent = dict(frame)
        if sent["type"] == "interrupt":
            sent["response_id"] = oid
            sent["item_id"] = oid
        script.append(sent)
    client = H.FakeClient(script)
    await H.run_relay(
        client, H.FakeProvider(on_send=on_send, auto_ack=True), timeout=6,
    )
    rows = [
        r for r in ledger["saves"]
        if str(r.get("assistant_ref", "")).startswith("live-output:")
    ]
    if not rows:
        return None
    last = rows[-1]
    return public_heard_text(last["assistant_text"], last.get("assistant_voice"))


@pytest.mark.asyncio
async def test_the_app_frames_make_the_saved_chat_match_what_was_displayed(monkeypatch):
    """THE BLOCKER, closed on the same wire the reproducer uses."""

    frames = _fixture()["live-epoch-partial-prefix"]["frames"]
    shown = await _hangup_with(frames, monkeypatch=monkeypatch)
    assert shown == "First sentence.", (
        f"the caller was shown {shown!r} after hanging up on "
        f"{GENERATED!r} having heard only 'First sentence.'"
    )


@pytest.mark.asyncio
async def test_an_empty_receipt_on_hangup_saves_nothing_unheard(monkeypatch):
    frames = _fixture()["live-epoch-empty-prefix"]["frames"]
    shown = await _hangup_with(frames, monkeypatch=monkeypatch)
    assert shown in (None, ""), (
        f"nothing was displayed before hangup, yet {shown!r} was saved"
    )


@pytest.mark.asyncio
async def test_a_fully_displayed_reply_is_saved_whole(monkeypatch):
    """Anti-vacuity: the fix must not blank every hangup."""

    frames = _fixture()["live-epoch-full-text"]["frames"]
    shown = await _hangup_with(frames, monkeypatch=monkeypatch)
    assert shown == GENERATED


@pytest.mark.asyncio
async def test_a_client_that_cannot_say_falls_back_honestly(monkeypatch):
    """Caption/playback identity mismatch: the key is ABSENT, not empty.

    The relay must then keep the generated text rather than invent a prefix —
    the degraded path, and it must stay distinguishable from 'heard nothing'.
    """

    frames = _fixture()["client-cannot-say"]["frames"]
    assert "heard_text" not in frames[0]
    shown = await _hangup_with(frames, monkeypatch=monkeypatch)
    assert shown == GENERATED


@pytest.mark.asyncio
async def test_the_same_wire_in_persian(monkeypatch):
    generated = "جمله اول. جمله دوم."
    frames = [dict(f) for f in _fixture()["live-epoch-partial-prefix"]["frames"]]
    frames[0]["heard_text"] = "جمله اول."
    shown = await _hangup_with(frames, generated=generated, monkeypatch=monkeypatch)
    assert shown == "جمله اول."
