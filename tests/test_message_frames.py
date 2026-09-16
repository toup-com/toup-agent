"""ONE builder for the live `{"type":"message"}` frame, and the second writer
that never used to send one.

Two writers commit rows into a user's day thread outside `AgentRunner`:
`sessions.create_session_message` (the platform relay's voice transcript) and
`agent.voice_tasks._persist_message` (a durable voice task's result). Only the
first broadcast anything, and it built the frame inline — so a task the user
started by voice was written correctly and never announced, reaching an
already-open thread only on the next history fetch, and any field added to one
writer's frame was silently missing from the other's.

Pure — platform sweep.
"""
from __future__ import annotations

import inspect
from datetime import datetime
from types import SimpleNamespace


def _row(**kw):
    base = dict(
        id="m1", role="assistant", content="hello", created_at=datetime(2026, 9, 15, 9, 0, 0),
        channel="voice", day_chat_id="day-1", client_msg_id=None, occurred_at=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_the_frame_carries_the_client_contract():
    from app.api.message_frames import message_frame

    f = message_frame(_row())
    assert f["type"] == "message"
    assert f["id"] == "m1"
    assert f["role"] == "assistant"
    assert f["content"] == "hello"
    assert f["created_at"].endswith("Z")
    assert f["channel"] == "voice"
    assert f["day_chat_id"] == "day-1"


def test_absent_identity_fields_are_omitted_not_nulled():
    """Mid-rollout a row may have neither column. The client must be unable to
    tell that from today's behaviour — a null is a value it would have to
    interpret."""
    from app.api.message_frames import message_frame

    f = message_frame(_row())
    assert "client_msg_id" not in f
    assert "occurred_at" not in f


def test_identity_fields_are_carried_when_present():
    from app.api.message_frames import message_frame

    f = message_frame(_row(client_msg_id="cm-1", occurred_at=datetime(2026, 9, 15, 8, 59)))
    assert f["client_msg_id"] == "cm-1"
    assert f["occurred_at"] == "2026-09-15T08:59:00Z"


def test_renderable_parts_are_present_only_when_they_exist():
    """A row with no text and no renderable part is DROPPED by the client
    (ChatScreen.handleServerMessage), which is correct — so an empty key list
    must be indistinguishable from an absent one."""
    from app.api.message_frames import message_frame

    bare = message_frame(_row())
    for k in ("media", "tool_events", "attachments", "app_artifact"):
        assert k not in bare

    full = message_frame(
        _row(),
        media={"type": "youtube"},
        tool_events=[{"tool": "web_search"}],
        attachments=[{"id": "a1"}],
        app_artifact={"slug": "snake"},
    )
    assert full["media"]["type"] == "youtube"
    assert full["tool_events"][0]["tool"] == "web_search"
    assert full["attachments"][0]["id"] == "a1"
    assert full["app_artifact"]["slug"] == "snake"

    empty = message_frame(_row(), media=None, tool_events=[], attachments=[], app_artifact=None)
    for k in ("media", "tool_events", "attachments", "app_artifact"):
        assert k not in empty


def test_the_channel_argument_wins_over_the_row():
    """The Conversation's channel is the caller's fact; the row's is a
    denormalised copy that may be NULL on older rows."""
    from app.api.message_frames import message_frame

    assert message_frame(_row(channel=None), channel="voice")["channel"] == "voice"


def test_both_writers_use_the_one_builder():
    from app.api import sessions
    from app.agent import voice_tasks

    assert "message_frame(" in inspect.getsource(sessions.create_session_message), (
        "the relay's transcript frame must not be built inline again"
    )
    assert "message_frame(" in inspect.getsource(voice_tasks.VoiceTaskService._persist_message)


def test_the_frame_never_carries_the_internal_storage_key():
    """`storage_path` is "{user_id}/{att_id}_{filename}" — the object key
    `files.py` authorizes against, and the raw tenant id with it. Both REST
    serializers delete it explicitly; this builder is the one surface that
    bypasses both, and it used to copy caller dicts wholesale."""
    from app.api.message_frames import message_frame

    # Synthetic, never a real tenant prefix: this repo's own rule is that an
    # identifier that names a live account does not go into code, tests or
    # telemetry — and an object key's first segment IS the user id, which is
    # also the label of that user's public agent host.
    f = message_frame(_row(), attachments=[
        {"id": "a1", "filename": "report.pdf", "mime_type": "application/pdf",
         "storage_path": "u-test-0001/abc_report.pdf"},
    ])
    assert f["attachments"][0]["id"] == "a1"
    assert f["attachments"][0]["filename"] == "report.pdf"
    for att in f["attachments"]:
        assert "storage_path" not in att, "the internal object key reached the client"


# ── The durable task's wire form ───────────────────────────────────────────

def test_wire_artifact_uses_the_one_preview_policy():
    """`_wire_artifact` carried its own third copy of the previewable-MIME set
    and that copy omitted PPTX: the same deck was previewable in chat and not
    previewable when a durable voice task produced it."""
    from app.agent.voice_tasks import _wire_artifact

    deck = _wire_artifact({
        "id": "a1", "filename": "deck.pptx", "mime_type":
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }, "m1")
    assert deck["preview_url"], "PPTX is previewable — files.py has rendered it all along"
    assert deck["download_url"].endswith("/files/m1/a1")
    assert deck["kind"] == "pptx"

    opaque = _wire_artifact(
        {"id": "a2", "filename": "x.bin", "mime_type": "application/x-thing"}, "m1")
    assert "preview_url" not in opaque, "an unknown MIME must not advertise a preview"


def test_wire_artifact_advertises_a_thumb_only_when_one_exists():
    from app.agent.voice_tasks import _wire_artifact

    with_thumb = _wire_artifact(
        {"id": "a1", "filename": "p.png", "mime_type": "image/png", "has_thumb": True}, "m1")
    assert with_thumb["thumb_url"].endswith("variant=thumb"), (
        "without it a task's image loads the full-resolution original into a card"
    )
    without = _wire_artifact(
        {"id": "a2", "filename": "p.png", "mime_type": "image/png"}, "m1")
    assert "thumb_url" not in without


def test_wire_artifact_keeps_intrinsic_dimensions_without_a_message_id():
    """width/height belong to the FILE, not to the URL. Inside the message-id
    guard, a card with no row yet lays out at a guessed ratio until it decodes."""
    from app.agent.voice_tasks import _wire_artifact

    payload = _wire_artifact(
        {"id": "a1", "filename": "p.png", "mime_type": "image/png",
         "width": 1024, "height": 1536}, None)
    assert payload["width"] == 1024 and payload["height"] == 1536
    assert "download_url" not in payload, "no row, no message-scoped URL"


def test_a_durable_task_result_is_broadcast_as_a_message():
    """`_broadcast` has to send the row frame as well as the task snapshot.

    Whitespace-tolerant: the previous version paired a vacuous
    `assert "message" in src` (true of almost any Python source) with an
    exact-token match that a harmless reflow across two lines would break.
    """
    import re

    from app.agent import voice_tasks

    src = inspect.getsource(voice_tasks.VoiceTaskService._broadcast)
    assert re.search(r"broadcast_to_user\(\s*user_id\s*,\s*message\s*,?\s*\)", src), (
        "the row frame is accepted and then never sent"
    )
    assert re.search(r"\bmessage\b\s*(:|=|is not None)", src), (
        "`message` must be a real parameter, not an incidental word"
    )


def test_the_frame_carries_a_revision_so_a_correction_can_replace_it():
    """The durable-task row id is `uuid5(result:<job id>)` — stable across
    generations — and the voice transcript's is the UPSERT key, so the same id
    is legitimately broadcast twice with DIFFERENT content: a task corrected by
    `steer` finishes a second time. The client de-dupes by id with no update
    path, so without a monotonic marker the CORRECTED answer is the one that is
    discarded."""
    from app.api.message_frames import message_frame

    assert "revision" not in message_frame(_row()), "absent must stay absent"
    assert message_frame(_row(), revision=0)["revision"] == 0
    assert message_frame(_row(), revision=4)["revision"] == 4
    # Never raises on a value it cannot use — this runs on the persist path.
    assert "revision" not in message_frame(_row(), revision="four")


def test_the_durable_task_puts_its_state_revision_on_the_frame():
    from app.agent import voice_tasks

    src = inspect.getsource(voice_tasks.VoiceTaskService._persist_message)
    assert "revision=int(getattr(job, \"state_revision\", 0) or 0)" in src, (
        "a re-finished task broadcasts the same id with no way to tell the "
        "client which answer is newer"
    )


# Non-terminal transitions deliberately write the row and announce NOTHING.
# The row content at these points is the placeholder ("Working on: <title>") or
# a mid-run checkpoint, and `ChatScreen.handleServerMessage` de-dupes by id with
# no update path — so an intermediate broadcast PINS the placeholder and the
# real answer, sent later under the same deterministic id, is discarded.
_SILENT_PERSIST_SITES = {
    "_progress": "mid-run artifact checkpoint; the task is still RUNNING",
    "_merge_action": "an approval card changed; the task is still WAITING",
    "_queue_after_actions": "WAITING → queued, i.e. about to run again",
}
# Terminal, but the frame is returned to the caller that owns the broadcast.
_DELEGATED_PERSIST_SITES = {
    "_expire_running_in_session": "returns (user_id, snapshot, frame) to _expire_running",
}


def test_every_terminal_persist_site_announces_its_row():
    """Enumerated, not listed. The previous version of this test named two of
    the eight call sites and claimed 'every terminal path' — four still wrote
    the result row and broadcast nothing, so half the defect it closed was live."""
    import ast

    from app.agent import voice_tasks

    src = inspect.getsource(voice_tasks)
    tree = ast.parse(src)
    found: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = ast.get_source_segment(src, node) or ""
        if "self._persist_message(" in body:
            found[node.name] = body

    assert len(found) >= 8, (
        f"the enumeration lost sight of call sites: {sorted(found)}"
    )
    for name, body in found.items():
        if name in _SILENT_PERSIST_SITES:
            assert "message=_result_frame" not in body, (
                f"{name} is non-terminal ({_SILENT_PERSIST_SITES[name]}) and must "
                "not pin a placeholder row in an open thread"
            )
            continue
        assert "_result_frame = await self._persist_message" in body, (
            f"{name} writes the result row without capturing its frame"
        )
        if name in _DELEGATED_PERSIST_SITES:
            assert "_result_frame)" in body, name
            continue
        assert "message=_result_frame" in body, (
            f"{name} is terminal and must announce the row, not leave it for "
            "the next history fetch"
        )


def test_the_delegated_terminal_site_is_actually_broadcast_by_its_caller():
    from app.agent import voice_tasks

    caller = inspect.getsource(voice_tasks.VoiceTaskService._expire_running)
    assert "for user_id, snapshot, result_frame in rows" in caller
    assert "message=result_frame" in caller
