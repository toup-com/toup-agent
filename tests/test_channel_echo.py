"""A WhatsApp turn was invisible on the phone until the next refetch (E7).

Only two places in the product broadcast a `{type:'message'}` frame to a
user's OTHER sockets — the late-answer path at ws_chat.py:4531 and
sessions.py:808 — and a turn that arrives over WhatsApp, Telegram, Discord or
Slack produces neither. The reply exists, is persisted, and the open app shows
nothing until something makes it refetch.

D6 adds `channel_echo.broadcast_channel_turn`. Two rules decide its whole
shape:

  * It fires ONLY for off-app origins. A web or mobile turn already has a
    live socket receiving deltas; echoing it would double every message the
    user sends from the app.
  * Every frame carries `channel: <origin>`. The released mobile client
    exempts `type === 'message'` from its channel filter (api.ts:3189), so the
    key arrives; and it is that key which stops the frame from settling the
    phone's own in-flight turn (api.ts:2703 — the settle allowlist is
    ['mobile','app',null]). A channel-LESS assistant frame would look like
    the answer to whatever the user just typed.

Local run (from backend/):
    RUN_MODE=platform PYTHONPATH=. ENVIRONMENT=test JWT_SECRET=x ENCRYPTION_KEY=y \
      pytest tests/test_channel_echo.py -q --tb=short -p no:cacheprovider
"""
from __future__ import annotations

from datetime import datetime

import pytest


def mod():
    return pytest.importorskip(
        "app.agent.channel_echo",
        reason="channel_echo not landed yet — lane B2 (D6)",
    )


PERSISTED = {
    "user_message_id": "m-user-1",
    "user_created_at": datetime(2026, 9, 14, 1, 40, 21),
    "asst_message_id": "m-asst-1",
    "asst_created_at": datetime(2026, 9, 14, 1, 40, 29),
    "day_chat_id": "dc-1",
    "conversation_id": "conv-1",
    "channel": "whatsapp",
    "media": None,
    "tool_events": [],
    "attachments": [],
}


@pytest.fixture
def frames(monkeypatch):
    """Capture whatever reaches the socket fan-out."""
    import app.api.ws_chat as ws_chat

    got: list[dict] = []

    async def recorder(user_id, event, exclude=None):
        got.append(event)
        return 1

    monkeypatch.setattr(ws_chat, "broadcast_to_user", recorder)
    m = mod()
    if hasattr(m, "broadcast_to_user"):
        monkeypatch.setattr(m, "broadcast_to_user", recorder)
    return got


async def _echo(*, origin="whatsapp", persisted=None, user_text="play stronger",
                assistant_text="Putting it on."):
    return await mod().broadcast_channel_turn(
        "u-1", origin_channel=origin, session_id="conv-1",
        persisted=PERSISTED if persisted is None else persisted,
        user_text=user_text, assistant_text=assistant_text,
    )


@pytest.mark.asyncio
async def test_a_whatsapp_turn_emits_the_user_row_then_the_assistant_row(frames):
    sent = await _echo()
    assert sent >= 1  # the ASSISTANT send's socket count; both frames are asserted below
    assert [f["role"] for f in frames] == ["user", "assistant"], (
        "the assistant row arrived before the user's — the app appends in "
        "receive order and would show the answer above the question"
    )
    assert all(f["type"] == "message" for f in frames)
    assert [f["id"] for f in frames] == ["m-user-1", "m-asst-1"]


@pytest.mark.asyncio
async def test_every_frame_is_stamped_with_its_ORIGIN_channel(frames):
    """D6. Not the reply channel and not 'web': the badge the user sees says
    where the turn came FROM, and the stamp is what stops the assistant frame
    settling the phone's own in-flight turn."""
    await _echo(origin="whatsapp")
    assert all(f.get("channel") == "whatsapp" for f in frames), (
        "a channel-less frame here settles the phone's own pending turn "
        "(api.ts:2703 — the allowlist is ['mobile','app',null])"
    )


@pytest.mark.parametrize("origin", ["web", "mobile", "app", "voice", "extension"])
@pytest.mark.asyncio
async def test_an_in_app_origin_emits_nothing(frames, origin):
    """The gate. ws_chat turns already stream to the socket that owns them;
    echoing would show the user their own message twice."""
    assert await _echo(origin=origin) == 0
    assert frames == []


def test_the_off_app_set_is_exactly_the_four_channels():
    """Named rather than inferred: `voice` and `extension` reach AgentRunner
    the same way and must NOT be in here."""
    m = mod()
    assert m.OFF_APP_ORIGIN_CHANNELS == frozenset({"whatsapp", "telegram", "discord", "slack"})


@pytest.mark.asyncio
async def test_media_and_tool_events_ride_the_assistant_frame_only(frames):
    """A user row has no card and no tool pills; emitting empty keys on it
    makes a client that checks `'media' in frame` render an empty player."""
    # The record shape `_save_messages` persists (ToolPillRow): the frame
    # serves it through the day route's serializer, so it must be well
    # formed to travel at all — see the REST-shape case at the end of this file.
    p = dict(PERSISTED, media={"type": "youtube", "video_id": "abc"},
             tool_events=[{"tool": "play_media", "started_at_ms": 1,
                           "completed_at_ms": 2, "summary": "Stronger"}])
    await _echo(persisted=p)
    user_frame, asst_frame = frames
    assert "media" not in user_frame and "tool_events" not in user_frame
    assert asst_frame["media"] == {"type": "youtube", "video_id": "abc"}
    assert len(asst_frame["tool_events"]) == 1
    assert asst_frame["tool_events"][0]["tool"] == "play_media"
    assert asst_frame["tool_events"][0]["started_at_ms"] == 1


@pytest.mark.asyncio
async def test_falsy_media_and_tool_events_are_omitted_entirely(frames):
    """`media: null` on every frame is how a client learns to stop trusting
    the key. Truthy-only, the same rule `_save_messages` uses for
    metadata_json."""
    await _echo()
    assert "media" not in frames[1]
    assert "tool_events" not in frames[1]


@pytest.mark.asyncio
async def test_content_passes_through_public_text(frames):
    """The frame is a message body going to a client, so it obeys the same
    rule the REST serializers do: an internal marker is blanked, and D2's
    leaked channel tag is stripped from an assistant row."""
    from app.api.message_cards import public_text

    await _echo(assistant_text="[mobile 9:41pm] Putting on X",
                user_text="[mobile 9:41pm] play X")
    assert frames[1]["content"] == public_text("assistant", "[mobile 9:41pm] Putting on X")
    assert frames[0]["content"] == public_text("user", "[mobile 9:41pm] play X")


@pytest.mark.asyncio
async def test_an_empty_persisted_dict_emits_nothing_and_does_not_raise(frames):
    """An `AgentResponse` from before the D3 plumbing — or a turn whose save
    failed — has nothing to echo. Broadcasting a frame with no id makes the
    client's dedupe (which keys on id) append a duplicate on every refresh."""
    assert await _echo(persisted={}) == 0
    assert frames == []


@pytest.mark.asyncio
async def test_a_raising_broadcaster_is_swallowed_and_reports_zero(monkeypatch):
    """The echo runs after the turn is already persisted and already answered
    on WhatsApp. A failure here must never surface as a failed turn."""
    import app.api.ws_chat as ws_chat

    async def boom(*a, **k):
        raise RuntimeError("socket fan-out down")

    monkeypatch.setattr(ws_chat, "broadcast_to_user", boom)
    m = mod()
    if hasattr(m, "broadcast_to_user"):
        monkeypatch.setattr(m, "broadcast_to_user", boom)

    assert await _echo() == 0


@pytest.mark.asyncio
async def test_attachments_and_tool_events_ride_in_the_day_routes_shape(frames):
    """Review R2 P2: the frame used to copy the RAW persisted values — the
    agent's internal `storage_path` on the wire and no `download_url`, so the
    live card could not be opened while the refetched one could."""
    persisted = dict(PERSISTED)
    persisted["attachments"] = [{
        "id": "att-1", "filename": "report.pdf", "mime_type": "application/pdf",
        "size_bytes": 10, "storage_path": "/opt/toup-agent/workspace/report.pdf",
        "created_at": "2026-09-14T01:40:29Z",
    }]
    persisted["tool_events"] = [
        {"tool": "web_search", "started_at_ms": 1, "completed_at_ms": 5, "summary": "x"},
        {"summary": "malformed — no tool key"},
    ]
    await _echo(persisted=persisted)
    asst = [f for f in frames if f["role"] == "assistant"][0]

    atts = asst["attachments"]
    assert len(atts) == 1
    assert "storage_path" not in atts[0], "the agent's internal path is on the wire"
    assert atts[0]["download_url"].endswith("/files/m-asst-1/att-1"), atts[0]
    assert atts[0]["filename"] == "report.pdf"

    events = asst["tool_events"]
    assert len(events) == 1, "a malformed record must be dropped, as the day route drops it"
    assert events[0]["tool"] == "web_search" and events[0]["started_at_ms"] == 1
    # The user frame carries neither.
    user = [f for f in frames if f["role"] == "user"][0]
    assert "attachments" not in user and "tool_events" not in user
