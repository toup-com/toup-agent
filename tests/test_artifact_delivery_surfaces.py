"""Where a generated file actually GOES — round 46, C9.

Three holes, all of them silent:

* `send_file` / `send_photo` refused unless there was an active Telegram chat
  (tool_executor.py:3274-3276, :3306-3308) even though every BaseChannel
  adapter implements both (channels/base.py:131,135) — so on WhatsApp, Slack,
  Discord, the app, the web and voice the tool stayed in the wire array and
  could only fail.
* A document generated during a WhatsApp / Telegram / Discord / Slack turn was
  never delivered on that channel: it was persisted on the Message row and
  echoed to the app/web sockets, so the user received a reply that CLAIMED a
  file which never arrived.
* `browser_screenshot` returned the raw base64 JPEG as TEXT inside the tool
  result, and nothing in the repo consumed it — never persisted, never an
  image block, never attached. The user saw nothing and the turn paid for the
  base64 in input tokens on every later iteration.

No network, no DB. Platform sweep.
"""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

import app.services.file_storage as _fs_module
from app.config import settings
from app.agent.tool_executor import ToolExecutor


class _Workspace:
    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = self._tmp.__enter__()
        self._orig_ws = settings.agent_workspace_dir
        self._orig_per_user = settings.workspace_per_user
        settings.agent_workspace_dir = self.path
        _fs_module._backend = None
        return self

    def __exit__(self, *a):
        settings.agent_workspace_dir = self._orig_ws
        settings.workspace_per_user = self._orig_per_user
        _fs_module._backend = None
        self._tmp.__exit__(*a)


def _te(ws: _Workspace) -> ToolExecutor:
    te = ToolExecutor(workspace=ws.path)
    te._user_id = "u1"
    te.set_user_id("u1")
    return te


# ── send_file / send_photo on every channel ─────────────────────────────

def test_send_file_attaches_with_no_telegram_anywhere():
    with _Workspace() as ws:
        te = _te(ws)
        os.makedirs(os.path.join(ws.path, "u1"), exist_ok=True)
        p = os.path.join(ws.path, "u1", "notes.md")
        with open(p, "w") as f:
            f.write("# Hello\n")
        assert te.telegram_bot is None
        out = asyncio.run(te.execute("send_file", {"path": "notes.md"}))
        assert not str(out).startswith("ERROR"), out
        assert len(te.pending_attachments) == 1
        att = te.pending_attachments[0]
        assert att["filename"] == "notes.md"
        assert att["mime_type"] == "text/markdown"
        assert att["kind"] == "markdown"
        assert att["role"] == "final"


def test_send_photo_refuses_a_file_that_is_not_a_picture():
    with _Workspace() as ws:
        te = _te(ws)
        os.makedirs(os.path.join(ws.path, "u1"), exist_ok=True)
        with open(os.path.join(ws.path, "u1", "notes.md"), "w") as f:
            f.write("x")
        out = asyncio.run(te.execute("send_photo", {"path": "notes.md"}))
        assert str(out).startswith("ERROR")
        assert "not an image" in str(out)
        assert te.pending_attachments == []


def test_send_file_on_a_missing_path_attaches_nothing():
    with _Workspace() as ws:
        te = _te(ws)
        out = asyncio.run(te.execute("send_file", {"path": "nope.pdf"}))
        assert str(out).startswith("ERROR")
        assert te.pending_attachments == []


def test_an_empty_file_is_never_attached():
    """A zero-byte file is a failure that happened to return, not a document."""
    with _Workspace() as ws:
        te = _te(ws)
        os.makedirs(os.path.join(ws.path, "u1"), exist_ok=True)
        open(os.path.join(ws.path, "u1", "empty.txt"), "w").close()
        out = asyncio.run(te.execute("send_file", {"path": "empty.txt"}))
        assert str(out).startswith("ERROR")
        assert te.pending_attachments == []


# ── The path the model is told (C13) ────────────────────────────────────

def test_register_attachment_names_a_path_that_actually_resolves():
    """It used to say `generated/{storage_path}`, which is relative to the
    workspace ROOT — but with `workspace_per_user` (the default) the model's
    own workspace is `<root>/<user_id>/`, so the string resolved to
    `<root>/<uid>/generated/<uid>/…` and named a file that is not there. A
    path the model cannot read back is the opposite of what it exists for."""
    with _Workspace() as ws:
        settings.workspace_per_user = True
        te = _te(ws)
        summary = asyncio.run(te.execute("generate_markdown", {
            "filename": "release-notes.md", "content": "# Notes\n\nShipped.\n",
        }))
        assert not str(summary).startswith("ERROR"), summary
        att = te.pending_attachments[0]
        named = [tok for tok in str(summary).split() if "generated/" in tok]
        assert named, f"no path in {summary!r}"
        path = named[0].rstrip(".")
        assert os.path.isfile(path), f"the model is told {path!r}, which is not there"
        # …and it is inside the jail the file tools enforce, so read_file and
        # send_file can actually open it.
        assert os.path.realpath(path).startswith(os.path.realpath(ws.path))
        assert att["storage_path"] in path


def test_the_tool_result_names_no_surface_the_client_may_not_have():
    """'the document pane' is a web-only surface: on iOS the attachment is an
    inline card, and on WhatsApp there is no pane at all. The result also used
    to leak the implementation ('agent_runner will emit the attachment
    event')."""
    with _Workspace() as ws:
        te = _te(ws)
        summary = str(asyncio.run(te.execute("generate_markdown", {
            "filename": "release-notes.md", "content": "# Notes\n",
        })))
        assert "document pane" not in summary
        assert "agent_runner" not in summary


# ── browser_screenshot ──────────────────────────────────────────────────

def test_browser_screenshot_becomes_an_attachment_and_never_base64():
    Image = pytest.importorskip("PIL.Image")
    import base64, io as _io
    from app.agent import extension_bridge

    buf = _io.BytesIO()
    Image.new("RGB", (8, 6), (10, 20, 30)).save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    with _Workspace() as ws:
        te = _te(ws)

        async def _dispatch(user_id, action, params, timeout_s=15):
            return {"width": 8, "height": 6, "quality": 80, "image_b64": b64}

        orig_dispatch = extension_bridge.dispatch
        orig_conn = extension_bridge.is_connected
        extension_bridge.dispatch = _dispatch            # type: ignore[assignment]
        extension_bridge.is_connected = lambda uid: True  # type: ignore[assignment]
        try:
            out = str(asyncio.run(te.execute("browser_screenshot", {"session_id": "s1"})))
        finally:
            extension_bridge.dispatch = orig_dispatch     # type: ignore[assignment]
            extension_bridge.is_connected = orig_conn     # type: ignore[assignment]

        assert "jpeg-b64" not in out, "the base64 fence is back in the model's context"
        assert b64[:40] not in out
        assert len(te.pending_attachments) == 1
        att = te.pending_attachments[0]
        assert att["mime_type"] == "image/jpeg"
        # A screenshot is EVIDENCE, not the deliverable: it belongs in the
        # thread but no client should treat it as the answer.
        assert att["role"] == "source"
        assert att["id"] in out, "the model needs the id it can reference"


# ── Channel delivery ────────────────────────────────────────────────────

class _FakeChannel:
    def __init__(self, kind="whatsapp"):
        from app.agent.channels.base import ChannelType
        self.channel_type = ChannelType(kind)
        self.photos: list = []
        self.files: list = []
        self.texts: list = []

    async def send_photo(self, chat_id, path, caption=None):
        self.photos.append(path)

    async def send_file(self, chat_id, path, caption=None):
        self.files.append(path)

    async def send_text(self, chat_id, text, parse_mode=None):
        self.texts.append(text)


def _att(name, mime, size=10, role="final", key="u1/abc_x"):
    return {"filename": name, "mime_type": mime, "size_bytes": size,
            "storage_path": key, "role": role, "id": "a1"}


def test_a_generated_document_reaches_the_channel_it_was_asked_on():
    from app.agent.channels.shared.message_handler import _deliver_attachments
    with _Workspace() as ws:
        key = "u1/abc_report.pdf"
        full = os.path.join(ws.path, "generated", key)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        open(full, "wb").write(b"%PDF-1.4\n")
        ch = _FakeChannel("whatsapp")
        undelivered = asyncio.run(_deliver_attachments(
            ch, "chat", [_att("report.pdf", "application/pdf", 9, key=key)], "chat"))
        assert undelivered == []
        assert ch.files == [full]
        assert ch.photos == []


def test_an_image_goes_through_send_photo_not_send_file():
    from app.agent.channels.shared.message_handler import _deliver_attachments
    with _Workspace() as ws:
        key = "u1/abc_pic.png"
        full = os.path.join(ws.path, "generated", key)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        open(full, "wb").write(b"\x89PNG\r\n")
        ch = _FakeChannel("telegram")
        asyncio.run(_deliver_attachments(ch, "c", [_att("pic.png", "image/png", 6, key=key)], "c"))
        assert ch.photos == [full] and ch.files == []


def test_a_derivative_is_never_delivered_twice():
    from app.agent.channels.shared.message_handler import _deliver_attachments
    with _Workspace() as ws:
        key = "u1/abc_pic.png"
        full = os.path.join(ws.path, "generated", key)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        open(full, "wb").write(b"\x89PNG\r\n")
        ch = _FakeChannel("telegram")
        asyncio.run(_deliver_attachments(
            ch, "c",
            [_att("pic.png", "image/png", 6, key=key),
             _att("pic.thumb.webp", "image/webp", 2, role="thumbnail", key=key)],
            "c"))
        assert len(ch.photos) == 1


def test_a_browser_screenshot_is_never_pushed_off_platform():
    """`role='source'` is EVIDENCE the model looked at, not the answer — and the
    agent's resident browser is logged into the user's own accounts. The
    delivery filter read `NON_CARD_ROLES` ({preview, thumbnail}), so every
    browsing turn started from WhatsApp/Telegram/Slack/Discord pushed one photo
    per screenshot of whatever was on screen into the user's messenger thread.
    "Do not draw a second card" and "do not ship this to a third party" are
    different questions and now read from different sets."""
    from app.agent.channels.shared.message_handler import _deliver_attachments
    with _Workspace() as ws:
        key = "u1/abc_shot.jpg"
        full = os.path.join(ws.path, "generated", key)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        open(full, "wb").write(b"\xff\xd8\xff")
        ch = _FakeChannel("whatsapp")
        undelivered = asyncio.run(_deliver_attachments(
            ch, "c",
            [_att("screenshot-1.jpg", "image/jpeg", 3, role="source", key=key),
             _att("answer.pdf", "application/pdf", 3, key=key)],
            "c"))
        assert ch.photos == [], "a browser screenshot reached the channel"
        assert ch.files == [full], "the actual answer must still be delivered"
        assert undelivered == []


def test_the_delivery_filter_and_the_card_filter_are_different_sets():
    from app.agent.artifact_kinds import (
        NON_CARD_ROLES, NON_DELIVERED_ROLES, ArtifactRole,
    )
    assert ArtifactRole.SOURCE in NON_DELIVERED_ROLES
    assert ArtifactRole.PROGRESS in NON_DELIVERED_ROLES
    assert ArtifactRole.FINAL not in NON_DELIVERED_ROLES
    # The client still DRAWS a source card; it just is not shipped anywhere.
    assert ArtifactRole.SOURCE not in NON_CARD_ROLES


def test_an_oversized_file_is_reported_rather_than_dropped():
    """A delivery failure must never fail a turn that already happened, and it
    must never be silent either: the reply has already claimed a file."""
    from app.agent.channels.shared.message_handler import _deliver_attachments
    with _Workspace():
        ch = _FakeChannel("discord")
        undelivered = asyncio.run(_deliver_attachments(
            ch, "c", [_att("huge.pdf", "application/pdf", 400 * 1024 * 1024)], "c"))
        assert undelivered == ["huge.pdf"]
        assert ch.files == [] and ch.photos == []


def test_a_channel_that_throws_does_not_take_the_turn_with_it():
    from app.agent.channels.shared.message_handler import _deliver_attachments
    with _Workspace() as ws:
        key = "u1/abc_report.pdf"
        full = os.path.join(ws.path, "generated", key)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        open(full, "wb").write(b"%PDF-1.4\n")

        class _Broken(_FakeChannel):
            async def send_file(self, chat_id, path, caption=None):
                raise RuntimeError("upload refused")

        undelivered = asyncio.run(_deliver_attachments(
            _Broken("slack"), "c", [_att("report.pdf", "application/pdf", 9, key=key)], "c"))
        assert undelivered == ["report.pdf"]


def test_the_handler_delivers_after_the_text_and_says_so_when_it_cannot():
    """Source probe: the ORDER matters (the text is the answer, the file is
    what the answer is about) and the honest fallback must exist."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "app" / "agent"
           / "channels" / "shared" / "message_handler.py").read_text()
    i_text = src.index("await channel.send_text(chat_id, reply_text)")
    i_atts = src.rindex("_deliver_attachments(")   # the CALL, not the def
    assert i_text < i_atts, "files must not overtake the answer"
    assert "in your Files" in src
    # A turn that produced ONLY a file must not be dropped as an empty reply.
    assert "if not reply_text and not _attachments:" in src


def test_telegram_still_CALLS_its_delivery_step():
    """The earlier version of this test asserted `"_deliver_turn_attachments"
    in src`, which the DEFINITION satisfies: deleting the only call site left
    the file with the function and every assertion green, restoring the exact
    reported defect — "the Telegram reply claimed a file that never arrived
    here". Pin the CALL, and pin that it happens after the reply."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "app" / "agent"
           / "telegram_bot.py").read_text()
    i_call = src.rindex("await self._deliver_turn_attachments(")
    i_text = src.rindex("await handler.finalize(final_text", 0, i_call)
    assert i_text < i_call, "files must not overtake the answer"


class _FakeBot:
    def __init__(self):
        self.photos, self.voices, self.docs, self.texts = [], [], [], []

    async def send_photo(self, chat_id, photo):
        self.photos.append(getattr(photo, "name", "?"))

    async def send_voice(self, chat_id, voice):
        self.voices.append(getattr(voice, "name", "?"))

    async def send_document(self, chat_id, document, filename=None):
        self.docs.append(filename)

    async def send_message(self, chat_id, text):
        self.texts.append(text)


class _StubTgSelf:
    def __init__(self, bot):
        self.app = type("_App", (), {"bot": bot})()


def _tg_deliver(bot, attachments):
    from app.agent.channels.shared.message_handler import deliver_bot_attachments
    return asyncio.run(deliver_bot_attachments(bot, 42, attachments))


def test_telegram_sends_each_kind_through_its_own_verb():
    """Audio as a VOICE note, not as a document: a .mp3 delivered as a file is
    a download, and the whole point of `generate_audio` on a phone is that it
    plays."""
    bot = _FakeBot()
    with _Workspace() as ws:
        for key, blob in (("u1/a_report.pdf", b"%PDF-1.4\n"),
                          ("u1/b_photo.png", b"\x89PNG\r\n"),
                          ("u1/c_note.mp3", b"ID3")):
            full = os.path.join(ws.path, "generated", key)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            open(full, "wb").write(blob)
        _tg_deliver(bot, [
            _att("report.pdf", "application/pdf", 9, key="u1/a_report.pdf"),
            _att("photo.png", "image/png", 8, key="u1/b_photo.png"),
            _att("note.mp3", "audio/mpeg", 3, key="u1/c_note.mp3"),
        ])
    assert bot.docs == ["report.pdf"]
    assert len(bot.photos) == 1
    assert len(bot.voices) == 1
    assert bot.texts == [], "nothing failed, so nothing to apologise for"


def test_telegram_never_pushes_a_browser_screenshot():
    """`role='source'` is EVIDENCE, not the answer. The agent's resident
    browser is logged into the user's accounts."""
    bot = _FakeBot()
    with _Workspace():
        _tg_deliver(bot, [_att("shot.png", "image/png", 8, role="source")])
    assert bot.photos == [] and bot.docs == [] and bot.texts == []


def test_telegram_says_so_when_a_file_is_too_big_for_the_channel():
    bot = _FakeBot()
    with _Workspace():
        _tg_deliver(bot, [_att("huge.pdf", "application/pdf", 400 * 1024 * 1024)])
    assert bot.docs == []
    assert bot.texts and "in your Files" in bot.texts[0]


def test_an_adapter_without_the_verb_is_a_FAILURE_not_a_success():
    """`BaseChannel.send_photo`/`send_file` LOG A WARNING AND RETURN — they do
    not raise — so an adapter that never overrode them (the live Baileys
    WhatsApp one implements neither) reported `sent=N failed=0` while nothing
    arrived, and the honest "it's in your Files" line never ran."""
    from app.agent.channels.base import BaseChannel, ChannelType
    from app.agent.channels.shared.message_handler import _deliver_attachments

    class _TextOnly(BaseChannel):
        channel_type = ChannelType("whatsapp")

        def __init__(self):
            self.texts: list = []

        async def send_text(self, chat_id, text, parse_mode=None):
            self.texts.append(text)

        async def send_typing(self, chat_id):
            pass

        async def start(self):
            pass

        async def stop(self):
            pass

    with _Workspace() as ws:
        key = "u1/abc_report.pdf"
        full = os.path.join(ws.path, "generated", key)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        open(full, "wb").write(b"%PDF-1.4\n")
        undelivered = asyncio.run(_deliver_attachments(
            _TextOnly(), "c", [_att("report.pdf", "application/pdf", 9, key=key)], "c"))
    assert undelivered == ["report.pdf"]
