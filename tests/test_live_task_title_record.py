"""The delegated row carries the task's NAME (R2 AF6, relay half).

The day chat headed a voice run with the nearest user row above it, and on the
recorded 2026-09-22 call that row was a fragment — cards titled «نیست» and
«بگو» for whole professor searches. The call itself heads the card with the
relay's `delegation.title`. So the relay now persists that SAME string as
`assistant_voice.task_title` on the delegated result row, and the app reads it
(`ChatScreen.foldVoiceRuns` → `activity.voiceRunTitle`, under the call's own
fragment floor).

Driven through the real relay (`test_live_harness`) with provider-shaped
events. The tenant half — the allowlist, the real write route and every read
surface — is pinned in tests/test_voice_record_metadata.py (RUN_MODE=agent);
this file hands the relay's actual row to the same `_clean_voice` so the seam
between the two is covered from this side too.
"""

import asyncio

import pytest

import test_live_harness as H

#: Longer than the 80-character card limit, so the title is CUT — the case a
#: reader would otherwise re-derive differently from the call — and Persian with
#: a ZWNJ, which must survive intact.
REQUEST = (
    "می‌خوام استاد رباتیک دانشگاه تورنتو رو پیدا کنی که ایرانی باشه و "
    "توی پردیس مرکزی کار می‌کنه، و بگو الان کجا درس میده"
)
ANSWER = "استاد علی رضایی در دانشکده علوم کامپیوتر پردیس مرکزی تدریس می‌کند."


def _rows_for(saves, did):
    return [s for s in saves if str(s.get("assistant_ref") or "").endswith(f":{did}")
            and str(s.get("assistant_ref") or "").startswith("live-delegation:")]


async def _wait(pred, timeout=4.0):
    for _ in range(int(timeout / 0.02)):
        if pred():
            return True
        await asyncio.sleep(0.02)
    return False


async def _run(monkeypatch, config):
    H.fast_clocks(monkeypatch, voice_live_delegation_ttl_s=2.0)

    async def think(user_id, task, session_id, relay=None, out=None, **kw):
        return ANSWER, "m"

    recorded = H.patch_relay(monkeypatch, think=think)

    def on_send(p, e):
        if e["type"] == "session.start":
            p.push(H.user_delta(REQUEST, 0, 2400))
            p.push(H.delegation("d1", 2450))
        elif e["type"] == "session.close":
            p.push(H.closed())

    async def settle():
        await _wait(lambda: any(
            (s.get("assistant_voice") or {}).get("spoken") is not None
            for s in _rows_for(recorded["saves"], "d1")
        ))

    client = H.FakeClient([config, settle, 0.2, {"type": "stop"}])
    provider = H.FakeProvider(on_send=on_send, auto_ack=True, speak_on_commentary=True)
    await H.run_relay(client, provider, timeout=10, db_session_id="db-af6-title")
    return client, recorded


@pytest.mark.asyncio
async def test_the_delegated_row_carries_the_title_the_call_showed(monkeypatch):
    from app.api.sessions import _clean_voice
    from app.services.live_voice_protocol import request_title

    client, recorded = await _run(monkeypatch, H.config())

    titles = {f.get("title") for f in client.of("delegation") if f.get("delegation_id") == "d1"}
    assert titles == {request_title(REQUEST)}, titles
    (title,) = titles
    assert title.endswith("…") and len(title) <= 80 and "‌" in title, title

    rows = _rows_for(recorded["saves"], "d1")
    assert rows, recorded["saves"]
    # EVERY write of the row — the first one and each revision (the spoken
    # verdict, the heard fold) — carries it: the tenant REPLACES `voice` on a
    # rewrite, so a revision without it would erase the name.
    assert len(rows) >= 2, [r.get("assistant_revision") for r in rows]
    for row in rows:
        voice = row["assistant_voice"]
        assert voice["task_title"] == title, voice
        assert voice["task_id"] == "d1"
        # …and it passes the tenant allowlist exactly as sent. (Only this key
        # is compared: a legacy heard fold sends `played_ms: None`, which the
        # allowlist drops — pre-existing, and not what this file pins.)
        assert _clean_voice(dict(voice))["task_title"] == title, voice
    assert rows[-1]["assistant_text"] == ANSWER, "the name is not the answer"


@pytest.mark.asyncio
async def test_the_row_is_named_whatever_the_client_negotiated(monkeypatch):
    """Persistence is not a negotiated feature: the day chat is read later, by
    whatever build the user has then, so a call from a client without the
    lifecycle family (or without delegation frames at all) still names its
    row. Additive — an older app reads `voice` and ignores the key."""

    client, recorded = await _run(monkeypatch, H.legacy_config())
    assert not client.of("delegation"), "a legacy client gets no task frames"
    rows = _rows_for(recorded["saves"], "d1")
    assert rows
    from app.services.live_voice_protocol import request_title

    for row in rows:
        assert row["assistant_voice"]["task_title"] == request_title(REQUEST)
