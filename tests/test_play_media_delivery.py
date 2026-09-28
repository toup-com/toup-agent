"""`play_media` may only claim a play it actually delivered (R48, A5-08/A3-9).

Two defects with one shape — the tool's return string is the model's entire
view of what happened, and it was written from intent rather than from outcome:

* ``broadcast_to_user`` returns the number of sockets that received the event
  and the value was thrown away, so a broadcast that reached ZERO listeners
  produced "Now playing <title>" into silence. During a voice call the phone's
  chat WS is a different socket from the audio one, supervised on a 4 s timer
  and dropped by every ordinary rollout or network transition, so ``sent == 0``
  is routine rather than exotic. Worse, the station was then seeded from a
  track that never played.
* the two genuine failure paths returned prose, and the voice tool-event stream
  derives ``ok`` from an ``ERROR`` prefix (``api_v1.on_tool_event``) — so a
  failed play rendered as a green, completed step.

These tests drive the REAL ``_tool_play_media`` with the network legs stubbed,
because the contract under test is exactly what the function RETURNS.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


def _executor(tmp_path):
    from app.agent.tool_executor import ToolExecutor
    tools = ToolExecutor(workspace=str(tmp_path))
    tools.set_user_id("user-1")
    tools.set_channel("app")
    return tools


@pytest.fixture
def resolved():
    """Skip the scrape: the resolver is not what these tests are about.

    The three resolve tiers all reach the network; the query→(id,title) cache
    is the one seam that short-circuits every one of them.
    """
    import time as _t
    import app.agent.tool_executor as te

    te._MEDIA_RESOLVE_CACHE.clear()
    te._MEDIA_RESOLVE_CACHE["play me humble"] = (
        "vid123", "Kendrick Lamar - HUMBLE.", _t.time(),
    )
    yield
    te._MEDIA_RESOLVE_CACHE.clear()


async def _play(tools, monkeypatch, *, sent: int, query="play me humble"):
    """Run the tool with a broadcast that reports `sent` recipients."""
    import app.api.ws_chat as ws

    seen: list = []

    async def _broadcast(user_id, event, **kw):
        seen.append(dict(event))
        return sent

    async def _swap(video_id, user_id):
        return None

    monkeypatch.setattr(ws, "broadcast_to_user", _broadcast)
    monkeypatch.setattr(ws, "_check_age_and_swap", _swap)

    seeds: list = []

    class _Mgr:
        @staticmethod
        def is_channel_allowed(ch):
            return True

        def record_user_seed(self, **kw):
            seeds.append(kw)

        def get(self, *a, **kw):
            return None

        def set_display_mode(self, *a, **kw):
            return None

    import app.agent.radio as radio
    monkeypatch.setattr(radio, "get_radio_manager", lambda: _Mgr())
    monkeypatch.setattr(radio.RadioSessionManager, "is_channel_allowed",
                        staticmethod(lambda ch: True), raising=False)

    out = await tools._tool_play_media({"query": query})
    return out, seen, seeds


async def test_a_broadcast_that_reached_nobody_is_an_ERROR(tmp_path, monkeypatch, resolved):
    """The headline. Nothing can sound, so the model must not say it does."""
    tools = _executor(tmp_path)
    out, seen, seeds = await _play(tools, monkeypatch, sent=0)

    assert out.startswith("ERROR:"), out
    assert seen, "the broadcast is still attempted — delivery is the question"
    # And the station is NOT seeded off a track that never played.
    assert seeds == []
    assert tools._last_media is None
    # …and the LINK survives (review L3-R5). A channel with no player socket —
    # WhatsApp, or a web user whose tab is closed — has `sent == 0` as its
    # ORDINARY outcome, and the prose this replaced gave them a tappable URL.
    # The sibling failure path keeps it for the same reason.
    assert "vid123" in out, out


async def test_a_delivered_broadcast_is_unchanged(tmp_path, monkeypatch, resolved):
    """The success path is the one this round must not move."""
    tools = _executor(tmp_path)
    out, seen, seeds = await _play(tools, monkeypatch, sent=1)

    assert not out.upper().startswith("ERROR")
    assert "HUMBLE" in out
    assert tools._last_media == {
        "type": "youtube", "video_id": "vid123",
        "title": "Kendrick Lamar - HUMBLE.",
    }
    assert seeds and seeds[0]["seed_track"].video_id == "vid123"


async def test_an_unresolvable_query_is_an_ERROR(tmp_path, monkeypatch):
    """`ok` on the voice wire is derived from this prefix, so prose here is a
    green "Starting the music" step with no music."""
    import app.agent.tool_executor as te
    import httpx

    te._MEDIA_RESOLVE_CACHE.clear()

    class _Dead:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **kw):
            raise RuntimeError("no network in tests")

        async def post(self, *a, **kw):
            raise RuntimeError("no network in tests")

    monkeypatch.setattr(httpx, "AsyncClient", _Dead)

    async def _no_subprocess(*a, **kw):
        raise RuntimeError("no yt-dlp in tests")

    import asyncio as _aio
    monkeypatch.setattr(_aio, "create_subprocess_exec", _no_subprocess)

    tools = _executor(tmp_path)
    out = await tools._tool_play_media({"query": "a query that resolves to nothing"})
    assert out.startswith("ERROR:"), out
    assert "Could not find a video" in out


async def test_a_broadcast_that_RAISED_is_an_ERROR(tmp_path, monkeypatch, resolved):
    """The third failure path: the send itself blew up."""
    import app.api.ws_chat as ws

    async def _boom(user_id, event, **kw):
        raise RuntimeError("socket registry unavailable")

    monkeypatch.setattr(ws, "broadcast_to_user", _boom)
    tools = _executor(tmp_path)
    out = await tools._tool_play_media({"query": "play me humble"})
    assert out.startswith("ERROR:"), out


async def test_every_failure_return_in_the_tool_is_ERROR_prefixed(tmp_path):
    """A grep-style guard over the source, because the prefix IS the contract
    and a new early return is exactly how it gets broken again.

    The Telegram branch is exempt and named: it returns a playable LINK the
    user can tap, which is a degraded success, not a failure.
    """
    import ast
    import inspect
    import textwrap

    from app.agent.tool_executor import ToolExecutor

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(ToolExecutor._tool_play_media))
    )
    # What a failure sounds like in this tool. Matched on the RETURNED text,
    # because the contract is the string the model reads.
    _FAILURE_PHRASES = (
        "could not", "couldn't", "not reachable", "no user context",
        "provide a song",
    )
    fails = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        text = ast.unparse(node.value)
        low = text.lower()
        if any(p in low for p in _FAILURE_PHRASES):
            fails.append(text)
    # Anti-vacuity: if a rename made every phrase miss, this guard would pass
    # by finding nothing. Four is what the function has today.
    assert len(fails) >= 4, (
        f"the guard found only {len(fails)} failure returns — it has gone "
        f"blind to the paths it exists to watch: {fails}"
    )
    for text in fails:
        # The Telegram branch is a degraded SUCCESS: the audio upload failed
        # and the user gets a playable link instead.
        if "tap the link" in text:
            continue
        assert "ERROR" in text, text
