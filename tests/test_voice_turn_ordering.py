"""The two halves of one spoken turn, in the order they HAPPENED.

`Message.created_at` is insert time. A voice turn's question and answer are
persisted by a queue worker behind a session wait, so insert order is a property
of the database's mood rather than of the conversation — and a reply can land
above the question it answers, permanently, in the day thread.

`occurred_at` is stamped at the provider event and the assistant half is stamped
strictly after the user half, so a reader sorting on
`COALESCE(occurred_at, created_at), id` can never invert them.

`client_msg_id` is the other half of the same primitive: derived from the
provider's own item/response id, so a replayed persist upserts onto the row it
already wrote instead of speaking the user's sentence twice into their thread.

Pure/faked — platform sweep.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest


@pytest.mark.asyncio
async def test_each_half_carries_its_own_stamp_and_its_own_identity(monkeypatch):
    """What THIS file can prove: the request builder forwards the stamp it is
    handed to the right half, and the two halves get different keys.

    It deliberately no longer claims to prove the ORDERING. The previous
    version handed `_save_voice_messages` both `occurred_at` values itself and
    then asserted they were in order — i.e. it asserted that the test's own two
    literals were in the order the test had written them, and gutting the real
    stamping (`_enqueue_persist`'s speech-stopped queue and assistant floor,
    both closures inside `realtime_voice_ws`) left it green. The ordering is
    now driven through the real endpoint in
    `test_voice_turn_survives.py::test_the_question_is_stamped_before_the_answer…`
    and its two siblings.
    """
    from app.api import ws_realtime

    sent: list[dict] = []

    async def _fake_vps_api(agent_url, key, method, path, params=None, json_body=None, **kw):
        sent.append(json_body or {})
        return {"ok": True}

    async def _fake_info(_uid):
        return ("https://agent.test", "key")

    monkeypatch.setattr(ws_realtime, "_vps_api", _fake_vps_api)
    monkeypatch.setattr(ws_realtime, "_get_vps_info", _fake_info)

    base = datetime(2026, 9, 15, 10, 0, 0)
    await ws_realtime._save_voice_messages(
        "u1", "s1", "the question", "the answer",
        user_occurred_at=base,
        assistant_occurred_at=base + timedelta(milliseconds=1),
        user_ref="item:1", assistant_ref="response:1",
    )
    assert len(sent) == 2
    assert sent[0]["role"] == "user" and sent[1]["role"] == "assistant"
    assert sent[0]["occurred_at"].startswith("2026-09-15T10:00:00")
    assert sent[1]["occurred_at"].startswith("2026-09-15T10:00:00.001")
    assert sent[0]["client_msg_id"] != sent[1]["client_msg_id"]


@pytest.mark.asyncio
async def test_a_lost_transcript_is_counted_not_only_logged(monkeypatch):
    """The body-only cutover is unconditional: an agent image older than the
    body-aware messages route answers 400 and the spoken turn is gone. That was
    an ERROR line and nothing else, so a straggler container was detectable
    only by grepping Loki."""
    from app.api import ws_realtime

    async def _fake_vps_api(*a, **kw):
        return None  # the agent refused / did not answer

    async def _fake_info(_uid):
        return ("https://agent.test", "key")

    monkeypatch.setattr(ws_realtime, "_vps_api", _fake_vps_api)
    monkeypatch.setattr(ws_realtime, "_get_vps_info", _fake_info)

    before = ws_realtime.voice_counter_snapshot().get("voice_transcript_lost", 0)
    await ws_realtime._save_voice_messages("u1", "s1", "the question", "the answer")
    after = ws_realtime.voice_counter_snapshot().get("voice_transcript_lost", 0)
    assert after == before + 1, "a lost spoken turn left no counter behind"


@pytest.mark.asyncio
async def test_a_replayed_persist_carries_the_same_identity(monkeypatch):
    """The same provider events must produce the same keys, so the second write
    upserts rather than duplicating the turn."""
    from app.api import ws_realtime

    sent: list[dict] = []

    async def _fake_vps_api(agent_url, key, method, path, params=None, json_body=None, **kw):
        sent.append(json_body or {})
        return {"ok": True}

    async def _fake_info(_uid):
        return ("https://agent.test", "key")

    monkeypatch.setattr(ws_realtime, "_vps_api", _fake_vps_api)
    monkeypatch.setattr(ws_realtime, "_get_vps_info", _fake_info)

    for _ in range(2):
        await ws_realtime._save_voice_messages(
            "u1", "s1", "hello", "hi", user_ref="item:9", assistant_ref="response:9",
        )
    assert sent[0]["client_msg_id"] == sent[2]["client_msg_id"]
    assert sent[1]["client_msg_id"] == sent[3]["client_msg_id"]


@pytest.mark.asyncio
async def test_without_provider_refs_nothing_is_stamped(monkeypatch):
    """Mid-rollout a relay may have no ref to hand. Absent must mean 'unknown',
    never a synthesised key that two different turns could collide on."""
    from app.api import ws_realtime

    sent: list[dict] = []

    async def _fake_vps_api(agent_url, key, method, path, params=None, json_body=None, **kw):
        sent.append(json_body or {})
        return {"ok": True}

    async def _fake_info(_uid):
        return ("https://agent.test", "key")

    monkeypatch.setattr(ws_realtime, "_vps_api", _fake_vps_api)
    monkeypatch.setattr(ws_realtime, "_get_vps_info", _fake_info)

    await ws_realtime._save_voice_messages("u1", "s1", "hello", "hi")
    assert all("client_msg_id" not in b for b in sent)


@pytest.mark.asyncio
async def test_what_the_turn_produced_rides_the_assistant_row(monkeypatch):
    """Files and the presented app belong to the reply that made them — the
    same place a chat turn puts them."""
    from app.api import ws_realtime

    sent: list[dict] = []

    async def _fake_vps_api(agent_url, key, method, path, params=None, json_body=None, **kw):
        sent.append(json_body or {})
        return {"ok": True}

    async def _fake_info(_uid):
        return ("https://agent.test", "key")

    monkeypatch.setattr(ws_realtime, "_vps_api", _fake_vps_api)
    monkeypatch.setattr(ws_realtime, "_get_vps_info", _fake_info)

    await ws_realtime._save_voice_messages(
        "u1", "s1", "make me a pdf", "Here it is.",
        attachments=[{"id": "a1", "filename": "report.pdf"}],
        app_artifact={"slug": "budget"},
    )
    assert "attachments" not in sent[0] and "app_artifact" not in sent[0]
    assert sent[1]["attachments"][0]["id"] == "a1"
    assert sent[1]["app_artifact"]["slug"] == "budget"


# ── The save=False echo, which is where the files come from ────────────────

def test_agent_response_echoes_what_a_save_false_turn_produced():
    """`AgentResponse.persisted` is documented as the echo a caller that owns
    persistence builds its rows from. On a voice turn it was left EMPTY, so a
    generated PDF had no Message row for GET /api/files/{message_id}/{aid} to
    authorize against — the file existed and nothing in the product could reach
    it. Asserted on the source of the save gate rather than by running a full
    agent turn: the property is that the else-branch DRAINS, and the drain is a
    move (the list is cleared) so the next turn cannot re-deliver the files.

    The slice is NARROW on purpose. The previous version split to the end of
    `run()` — some seven hundred lines — and asserted four substrings appeared
    somewhere in all of it, and it never asserted the value reaches
    `AgentResponse` at all: deleting `persisted=_persisted` killed the whole
    feature with this test green. Both markers are checked before slicing, so a
    reworded log line fails readably instead of raising IndexError.
    """
    import inspect
    from app.agent.agent_runner import AgentRunner

    src = inspect.getsource(AgentRunner.run)
    for marker in ("save_assistant_message:", "phase3_save: SKIPPED",
                   "Phase 3b: Background tasks", "persisted=_persisted"):
        assert marker in src, f"probe anchor {marker!r} was renamed — update this test"

    else_branch = (
        src.split("phase3_save: SKIPPED", 1)[1].split("Phase 3b: Background tasks", 1)[0]
    )
    assert "pending_attachments" in else_branch
    assert "self.tools.pending_attachments = []" in else_branch, "the drain must be a MOVE"
    assert '_persisted["attachments"]' in else_branch
    assert '_persisted["app_artifact"]' in else_branch
    # …and the echo actually reaches the caller. This is the one line whose
    # deletion loses everything above it, and it lives outside the slice.
    assert "persisted=_persisted," in src


def test_chat_response_exposes_them():
    from app.api.api_v1 import ChatResponse

    r = ChatResponse(text="x", session_id="s")
    assert r.attachments == [] and r.app_artifact is None, (
        "both must default to absent so an older relay is unaffected"
    )
    r2 = ChatResponse(
        text="x", session_id="s",
        attachments=[{"id": "a"}], app_artifact={"slug": "s"},
    )
    assert r2.attachments[0]["id"] == "a"
