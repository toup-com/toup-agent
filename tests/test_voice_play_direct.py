"""A voice "play me X" must be ONE tool call, not a nested agent turn.

Measured on prod for the founder, 2026-07-31 (llm_proxy_events + Railway logs):
routing a play through `think` cost 13.0s to the media_play frame. 79% of that
was inside the agent process and only 3.5s was inference — the two gpt-5.5 calls
produced 66 output tokens between them, to emit one tool call with one string
argument and then say one short sentence. The agent contributes exactly two
things to a play: query -> video_id, and pushing the frame.

So `play_media` is now a realtime tool of its own, served by a tool-less agent
route. These tests pin the three properties that make that safe:
  * the model can actually see it (the V2 filter drops everything it does not
    know about, so an un-listed tool is invisible rather than broken)
  * it does not touch the agent-turn endpoint
  * failure surfaces the real cause and never reads as "no such feature"
"""
from __future__ import annotations

import pytest

from app.config import settings
import app.api.ws_realtime as rt

pytestmark = pytest.mark.asyncio


@pytest.fixture
def v2_on(monkeypatch):
    monkeypatch.setattr(settings, "voice_realtime_v2", True)


def test_play_media_is_offered_to_the_voice_model():
    """V2 filters the tool array to a hard-coded set; a tool missing from it is
    silently invisible to the model, which is exactly how "I can't play music"
    becomes the honest answer."""
    assert "play_media" in rt._REALTIME_NATIVE
    names = {t["name"] for t in rt.REALTIME_TOOLS}
    assert "play_media" in names, f"not in REALTIME_TOOLS: {sorted(names)}"

    spec = next(t for t in rt.REALTIME_TOOLS if t["name"] == "play_media")
    assert spec["parameters"]["required"] == ["query"]
    # ONE definition — it comes from the agent's own tool list, with a
    # voice-specific description layered on. A second literal here would be a
    # duplicate function name on the wire.
    assert len([t for t in rt.REALTIME_TOOLS if t["name"] == "play_media"]) == 1
    # The description must actively steer away from `think`, or the model keeps
    # reaching for the general tool it already knows.
    assert "`think`" in spec["description"]
    assert "browser" not in spec["description"], "chat wording leaked into voice"


def test_play_media_step_has_a_human_label():
    title, detail = rt._tool_activity("play_media", {"query": "asap rocky"})
    assert title == "Starting the music"
    assert detail == "asap rocky"


async def test_a_play_never_runs_an_agent_turn(v2_on, monkeypatch):
    async def _vps(user_id):
        return ("https://u.agents.toup.ai", "agent-key")
    monkeypatch.setattr(rt, "_get_vps_info", _vps)

    calls = []

    async def _fake_api(agent_url, agent_api_key, method, path,
                        params=None, json_body=None, timeout=15.0):
        calls.append((method, path, json_body, timeout))
        return {"ok": True, "title": "A$AP Rocky - Praise The Lord", "video_id": "abc12345678"}
    monkeypatch.setattr(rt, "_vps_api", _fake_api)

    result, media = await rt._play_media_direct("user-1", "asap rocky")

    assert len(calls) == 1, f"expected exactly one hop, got {calls}"
    method, path, body, timeout = calls[0]
    assert (method, path) == ("POST", "/api/v1/internal/play-media")
    # The query is what must survive the hop; the body is additive over time
    # (`variety` rides along so an open-ended voice ask starts somewhere fresh).
    assert body["query"] == "asap rocky"
    assert timeout == rt._PLAY_MEDIA_TIMEOUT_S
    # The regression this whole change exists to prevent.
    assert all("agent-turn" not in c[1] for c in calls)
    # The title comes back with the result, so "what's playing?" needs no
    # second round trip — it is already in the model's own context.
    assert "A$AP Rocky - Praise The Lord" in result
    assert not result.upper().startswith("ERROR")
    # …and the same play leaves a renderable card in the chat thread.
    assert media["video_id"] == "abc12345678"
    assert media["thumbnail_url"], "a card without artwork is its own bug"


@pytest.mark.parametrize("failure", ["unreachable", "no_result"])
async def test_a_failed_play_states_the_real_cause_and_names_no_competitor(
    v2_on, monkeypatch, failure,
):
    """The founder's Recording 2: the agent told him it fundamentally could not
    play music and sent him to Spotify. Whatever else a failure does, it must
    never do that."""
    async def _vps(user_id):
        return ("https://u.agents.toup.ai", "agent-key")
    monkeypatch.setattr(rt, "_get_vps_info", _vps)

    async def _fake_api(*a, **k):
        if failure == "unreachable":
            raise ConnectionError("agent container restarting")
        return {"ok": False}
    monkeypatch.setattr(rt, "_vps_api", _fake_api)

    result, media = await rt._play_media_direct("user-1", "asap rocky")
    low = result.lower()
    # A failed play must not persist a card for a song that never started.
    assert media is None

    # Marked failed, so the phone shows the step as failed rather than done.
    assert result.upper().startswith("ERROR")
    # States a CAUSE the user can act on, rather than a bare refusal.
    assert ("temporary" in low) or ("may not be available" in low), result
    # Never a capability denial, never a competitor.
    for banned in ("spotify", "apple music", "youtube", "cannot play", "can't play",
                   "unable to play", "don't have"):
        assert banned not in low, f"failure text leaked {banned!r}: {result}"


async def test_an_empty_query_never_reaches_the_agent(v2_on, monkeypatch):
    called = []

    async def _vps(user_id):
        called.append(1)
        return ("https://u.agents.toup.ai", "k")
    monkeypatch.setattr(rt, "_get_vps_info", _vps)

    result, media = await rt._play_media_direct("user-1", "   ")
    assert result.upper().startswith("ERROR")
    assert media is None
    assert not called, "an empty query should not cost a network hop"
