"""A song started by voice must leave the same card in the thread as one
started by typing.

In the founder's 2026-07-31 recording every voice play worked — the music
started, the agent said so out loud — and the chat thread showed nothing but
plain text. Reopening the app, there was no evidence any song had ever played:
no title, no artwork, no controls. A chat play, meanwhile, persists its card
because `AgentRunner._save_messages` writes `metadata_json={"media": ...}` onto
the assistant row, and that field is the only thing the mobile client renders a
media card from.

The voice path never ran an agent turn, so nothing captured the play, and the
endpoint the relay persists through had no field that could carry it anyway.

There is also a quieter bug here: `_last_media` is a one-slot mailbox that the
NEXT persisted chat turn captures-and-clears. A voice play that left a value in
it got its card stapled onto whatever unrelated message came next.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("ENVIRONMENT", "test")


# ── The payload carries media ───────────────────────────────────────────

def test_message_payload_carries_media_in_the_body():
    from app.api.ws_realtime import _message_payload

    media = {"type": "youtube", "video_id": "uelHwf8o7_U", "title": "Love The Way You Lie"}
    body, params = _message_payload("assistant", "Playing it now.", "gpt-realtime", media)
    assert body["media"] == media
    assert params is None, (
        "media must be body-only — a nested object cannot ride the query shim"
    )


def test_message_payload_without_media_keeps_the_query_shim():
    """The compatibility shim for older agent images must not regress."""
    from app.api.ws_realtime import _message_payload

    body, params = _message_payload("assistant", "Short reply.", "gpt-realtime")
    assert "media" not in body
    assert params is not None and params["content"] == "Short reply."


@pytest.mark.asyncio
async def test_save_voice_messages_attaches_media_to_the_assistant_row(monkeypatch):
    """The card belongs to the reply that started the song — same as chat."""
    from app.api import ws_realtime

    sent = []

    async def _fake_vps_api(agent_url, key, method, path, params=None, json_body=None, **kw):
        sent.append((json_body or {}).copy())
        return {"ok": True}

    async def _fake_vps_info(user_id):
        return ("https://agent.example", "key")

    monkeypatch.setattr(ws_realtime, "_vps_api", _fake_vps_api)
    monkeypatch.setattr(ws_realtime, "_get_vps_info", _fake_vps_info)

    media = {"type": "youtube", "video_id": "lWA2pjMjpBs", "title": "Rihanna - Diamonds"}
    await ws_realtime._save_voice_messages(
        "user-1", "sess-1", "play me rihanna", "Starting Rihanna now.", media=media,
    )

    by_role = {m["role"]: m for m in sent}
    assert by_role["assistant"].get("media") == media
    assert "media" not in by_role["user"], "the card rides the assistant row only"


# ── A failed play must not persist a card ───────────────────────────────

@pytest.mark.asyncio
async def test_a_failed_play_returns_no_media(monkeypatch):
    """Otherwise the thread shows a card for a song that never played."""
    from app.api import ws_realtime

    async def _no_vps(user_id):
        return None

    monkeypatch.setattr(ws_realtime, "_get_vps_info", _no_vps)
    text, media = await ws_realtime._play_media_direct("user-1", "shadmehr aghili")
    assert media is None
    assert text.startswith("ERROR")


@pytest.mark.asyncio
async def test_an_empty_query_returns_no_media():
    from app.api import ws_realtime

    text, media = await ws_realtime._play_media_direct("user-1", "   ")
    assert media is None and text.startswith("ERROR")


@pytest.mark.asyncio
async def test_an_ok_response_without_a_video_id_yields_no_card(monkeypatch):
    """A card with no video_id renders as a dead tile: no artwork, no
    playback, nothing to tap. Persist nothing rather than that."""
    from app.api import ws_realtime

    async def _fake_vps_info(user_id):
        return ("https://agent.example", "key")

    async def _fake_vps_api(agent_url, key, method, path, **kw):
        return {"ok": True, "title": "Something", "video_id": ""}

    monkeypatch.setattr(ws_realtime, "_get_vps_info", _fake_vps_info)
    monkeypatch.setattr(ws_realtime, "_vps_api", _fake_vps_api)

    text, media = await ws_realtime._play_media_direct("user-1", "something")
    assert media is None, "no video_id → no card"
    assert "Something" in text, "the model should still hear what started"


@pytest.mark.asyncio
async def test_a_successful_play_returns_a_renderable_card(monkeypatch):
    from app.api import ws_realtime

    async def _fake_vps_info(user_id):
        return ("https://agent.example", "key")

    async def _fake_vps_api(agent_url, key, method, path, **kw):
        return {"ok": True, "title": "Ebi - Gheseh Eshgh", "video_id": "8QuUxAfb7Sk"}

    monkeypatch.setattr(ws_realtime, "_get_vps_info", _fake_vps_info)
    monkeypatch.setattr(ws_realtime, "_vps_api", _fake_vps_api)

    text, media = await ws_realtime._play_media_direct("user-1", "ebi")
    assert media["video_id"] == "8QuUxAfb7Sk"
    assert media["title"] == "Ebi - Gheseh Eshgh"
    assert media["type"] == "youtube"
    # Artwork is never blank — the founder's Rihanna card had none.
    assert media["thumbnail_url"], "a card without artwork is the bug we fixed"
    assert "Ebi - Gheseh Eshgh" in text


# ── The one-slot mailbox is drained ─────────────────────────────────────

def test_internal_play_media_clears_last_media():
    """`_last_media` left set gets captured by the NEXT persisted chat turn,
    stapling a voice play's card onto an unrelated assistant message."""
    import inspect
    from app.api import api_v1

    src = inspect.getsource(api_v1.internal_play_media)
    assert "_last_media = None" in src, (
        "internal_play_media must consume _last_media — nothing else on this "
        "path ever drains it"
    )


def test_session_message_schema_accepts_media():
    from app.schemas import SessionMessageCreate

    m = SessionMessageCreate(role="assistant", content="hi", media={"type": "youtube"})
    assert m.media == {"type": "youtube"}
    # Still optional — every existing caller omits it.
    assert SessionMessageCreate(role="user", content="hi").media is None


def test_create_session_message_writes_metadata_json():
    """Media must land in metadata_json under "media" — the exact field chat
    writes and the only one the client renders a card from.

    Asserted through the writer rather than by grepping the route for a
    literal: round ten added a second key to the same blob (`tool_events`, so
    a voice RUN reaches the thread), and the two used to evict each other
    because media replaced the whole document. The property is the shape of
    what gets stored, not the spelling of the line that stores it.
    """
    import inspect
    import json as _json
    from app.api import sessions

    stored = _json.loads(sessions._build_metadata({"type": "youtube"}, None))
    assert stored == {"media": {"type": "youtube"}}

    # And the route must actually use that writer for the message it builds.
    src = inspect.getsource(sessions.create_session_message)
    assert "metadata_json=_build_metadata(media, tool_events)" in src


# ── Never guess an artist ───────────────────────────────────────────────

# ── Voice persona actually loads ────────────────────────────────────────

def test_identity_is_read_from_the_platform_db_not_the_agent():
    """Every voice session ran with NO persona for as long as this call existed.

    The relay fetched `GET /api/identity` from the user's AGENT container, but
    the identity router was then mounted only in platform_main — so it 404'd,
    `_vps_api` folded that into None, and an empty result was
    indistinguishable from a user who simply has no identity. The relay
    already runs inside the platform process; the rows are right there.
    (Verified against prod: the founder has an active `soul` row, 523 chars,
    priority 100, that had never once reached a voice prompt.)

    Since the closeout run the agent DOES serve /api/identity
    (test_agent_serves_identity) and `_finalize_onboarding` deliberately
    dual-writes the TENANT copy through it — the agent-side assembler and
    text chat read `identities` from the tenant DB (the W-6 1,525-char gap).
    So the pin is now targeted, not a module-wide literal ban: the persona
    READ path stays local to the platform DB; only the onboarding
    dual-WRITE may talk to the agent, and only non-fatally.
    """
    import inspect
    from app.api import ws_realtime

    src = inspect.getsource(ws_realtime)
    assert "_load_identities_local" in src
    # The doomed HTTP call must not come back on the READ path: outside
    # the onboarding dual-write, no identity fetch from the agent.
    finalize_src = inspect.getsource(ws_realtime._finalize_onboarding)
    src_without_finalize = src.replace(finalize_src, "")
    assert '"GET", "/api/identity"' not in src_without_finalize, (
        "identity must not be FETCHED from the agent outside the "
        "onboarding dual-write — the persona read path is "
        "_load_identities_local, against the platform DB"
    )
    # The dual-write itself must be non-fatal: a tenant write failure logs
    # a warning and never breaks onboarding finalize.
    assert '"GET", "/api/identity"' in finalize_src, (
        "the tenant dual-write left _finalize_onboarding — if it moved, "
        "move this pin with it; if it was deleted, the agent-side "
        "assembler is reading identities nothing writes anymore"
    )
    assert "Failed to save tenant Identity" in finalize_src, (
        "the tenant identity write must stay wrapped so its failure "
        "cannot break onboarding"
    )
    fn = inspect.getsource(ws_realtime._load_identities_local)
    assert "Identity" in fn and "is_active" in fn, (
        "must read active Identity rows from the platform DB"
    )
    # Call syntax, not the bare name — the docstring narrates the old
    # _vps_api failure mode and must stay allowed to.
    assert "_vps_api(" not in fn, (
        "the persona loader must never fall back to fetching identity "
        "over HTTP from the agent"
    )


@pytest.mark.asyncio
async def test_identity_loader_returns_the_shape_the_prompt_builder_expects():
    """The consumer reads {"identities": [...]} with identity_type/content/
    priority and sorts by priority. Changing the shape silently drops persona."""
    from app.api import ws_realtime

    class _Row:
        def __init__(self, t, c, p):
            self.identity_type, self.content, self.priority = t, c, p

    class _Result:
        def scalars(self):
            class _S:
                def all(_self):
                    return [_Row("soul", "Be warm.", 100)]
            return _S()

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def execute(self, *a, **k):
            return _Result()

    import app.db.database as dbmod
    orig = dbmod.async_session_maker
    dbmod.async_session_maker = lambda: _Session()
    try:
        out = await ws_realtime._load_identities_local("user-1")
    finally:
        dbmod.async_session_maker = orig

    assert out == {"identities": [
        {"identity_type": "soul", "content": "Be warm.", "priority": 100},
    ]}


# ── Now-playing awareness ───────────────────────────────────────────────

def test_now_playing_injects_context_without_starting_a_turn():
    """The relay must update the model's belief SILENTLY.

    Radio advances never reach the voice relay (they ride the chat socket), so
    the model's now-playing knowledge was frozen at the last play_media result
    and it named a song that had stopped playing minutes ago. The phone reports
    the change — but this must not become a spoken turn, or every station
    advance would interrupt the conversation to announce itself.
    """
    import inspect
    from app.api import ws_realtime

    src = inspect.getsource(ws_realtime.realtime_voice_ws) \
        if hasattr(ws_realtime, "realtime_voice_ws") else inspect.getsource(ws_realtime)
    i = src.find('msg_type == "now_playing"')
    assert i != -1, "the relay must handle a now_playing client message"
    # Everything up to the NEXT handler branch is this handler's body.
    j = src.find('elif msg_type ==', i + 10)
    body = src[i:j if j != -1 else len(src)]
    # Strip comments — the handler's own comment explains WHY it omits
    # response.create, and a naive substring check would trip on that prose.
    code = "\n".join(
        ln for ln in body.splitlines() if not ln.strip().startswith("#")
    )
    assert "conversation.item.create" in code, "it must inject the new title as context"
    assert "response.create" not in code, (
        "now_playing must NOT trigger a response — a station advance is a "
        "context correction, not something to announce"
    )


def test_voice_play_description_forbids_guessing_a_misheard_name():
    """The "Play me" → Drake case: the utterance was cut off before the artist
    and the model supplied one from earlier in the call."""
    from app.api.ws_realtime import _VOICE_TOOL_DESCRIPTIONS

    d = _VOICE_TOOL_DESCRIPTIONS["play_media"].lower()
    assert "ask" in d and ("garbled" in d or "cut off" in d)
    assert "do not substitute" in d or "not substitute" in d
    # …without undoing the speed instruction that made plays one tool call.
    assert "immediately" in d
    assert "if you did hear the name clearly, do not ask" in d


# ── The completed frame is the card, never the prose (2026-08-11) ────────
# play_media's result string is the MODEL's sentence ("Now playing: X. It is
# already audible on the user's device…") — it shipped verbatim as
# result_preview and the voice canvas rendered it: raw English, third person,
# truncated at the clamp, in a Farsi session. The user-facing shape is the
# structured card.


def test_completed_frame_ships_the_card_not_the_prose():
    from app.api.ws_realtime import _tool_completed_frame

    media = {"type": "youtube", "video_id": "vX", "title": "Suspicious Minds",
             "thumbnail_url": "https://i.ytimg.com/vi/vX/hqdefault.jpg"}
    f = _tool_completed_frame(
        "c1", "play_media",
        "Now playing: Elvis Presley - Suspicious Minds. It is already audible "
        "on the user's device, and more in the same style will follow automatically.",
        media=media,
    )
    assert f["ok"] is True
    assert f["result_preview"] == ""
    assert f["media"] == media


def test_completed_frame_failed_play_has_no_card_and_no_prose():
    from app.api.ws_realtime import _tool_completed_frame

    f = _tool_completed_frame("c1", "play_media",
                              "ERROR: could not start that track.", media=None)
    assert f["ok"] is False
    assert f["result_preview"] == ""
    assert "media" not in f


def test_completed_frame_other_tools_keep_their_previews():
    from app.api.ws_realtime import _tool_completed_frame

    f = _tool_completed_frame("c2", "web_search", "3 sources · example.com")
    assert f["result_preview"] == "3 sources · example.com"
    assert "media" not in f
