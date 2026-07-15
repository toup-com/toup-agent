"""edit_image cross-turn source resolution.

Regression for the "I don't see the image attached on my side" bug: the
user sent a photo on an earlier turn, then asked to edit it on a LATER
turn without re-attaching. `_tool_edit_image` only looked at the current
turn's `_inbound_media`, so it errored out. The fix adds a history
fallback (`ToolExecutor._recent_uploaded_image`) that finds the most
recently uploaded image from persisted `Message.attachments`.

Uses a self-contained in-memory sqlite engine (the `messages` model has a
conditional pgvector column that `create_all` can't build on sqlite — the
same reason test_reply_history hand-rolls its DDL) and monkeypatches the
session maker the tool executor queries through.

Run: cd backend && env ENVIRONMENT=test STRIPE_SECRET_KEY=sk_test_x \
        pytest tests/test_edit_image_recent_source.py -q
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db.models.conversation import Conversation, Message
from app.agent.tool_executor import ToolExecutor


async def _make_engine():
    engine = create_async_engine(
        "sqlite+aiosqlite://", connect_args={"check_same_thread": False}
    )
    async with engine.begin() as conn:
        for stmt in [
            # No `users` table — the query only joins conversations→messages and
            # sqlite doesn't enforce the FK, so a bare user_id string suffices.
            """CREATE TABLE IF NOT EXISTS conversations (
                id VARCHAR(36) PRIMARY KEY, user_id VARCHAR(36) REFERENCES users(id),
                title VARCHAR(500), day_chat_id VARCHAR(36),
                channel VARCHAR(50) DEFAULT 'web', is_active BOOLEAN DEFAULT 1,
                started_at TIMESTAMP, ended_at TIMESTAMP, updated_at TIMESTAMP,
                message_count INTEGER DEFAULT 0, total_tokens INTEGER DEFAULT 0,
                builder_mode VARCHAR(10), metadata_json TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS messages (
                id VARCHAR(50) PRIMARY KEY, conversation_id VARCHAR(36) REFERENCES conversations(id),
                day_chat_id VARCHAR(36), role VARCHAR(20),
                content TEXT, created_at TIMESTAMP, tokens_prompt INTEGER,
                tokens_completion INTEGER, model_used VARCHAR(50),
                memories_retrieved_json TEXT, processing_time_ms INTEGER,
                metadata_json TEXT, embedding_json TEXT, embedding BLOB,
                channel VARCHAR(50), source VARCHAR(50),
                reply_to_message_id VARCHAR(50), attachments TEXT
            )""",
        ]:
            await conn.run_sync(lambda c, s=stmt: c.execute(text(s)))
    return engine


@pytest_asyncio.fixture
async def sm(monkeypatch):
    """Session maker on a self-contained sqlite engine, patched in as the
    one the tool executor's history lookup imports."""
    engine = await _make_engine()
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    import app.db.database as _dbmod
    monkeypatch.setattr(_dbmod, "async_session_maker", maker)
    yield maker
    await engine.dispose()


def _img(path: str, name: str = "photo.jpg", mime: str = "image/jpeg") -> dict:
    return {"id": uuid.uuid4().hex, "filename": name, "mime_type": mime,
            "size_bytes": 123, "storage_path": path}


def _doc(path: str) -> dict:
    return {"id": uuid.uuid4().hex, "filename": "report.pdf",
            "mime_type": "application/pdf", "size_bytes": 456, "storage_path": path}


async def _mk_conv(sm, uid: str) -> str:
    conv_id = str(uuid.uuid4())
    async with sm() as db:
        db.add(Conversation(id=conv_id, user_id=uid, channel="web"))
        await db.commit()
    return conv_id


async def _mk_msg(sm, conv_id: str, role: str, content: str, when: datetime, attachments=None) -> None:
    async with sm() as db:
        db.add(Message(
            id=str(uuid.uuid4()), conversation_id=conv_id, role=role,
            content=content, created_at=when, attachments=attachments,
        ))
        await db.commit()


def _exec(uid: str, tmp_path) -> ToolExecutor:
    ex = ToolExecutor(workspace=str(tmp_path))
    ex.set_user_id(uid)
    return ex


async def test_finds_newest_user_image_across_turns(sm, tmp_path):
    uid = "u-edit-1"
    conv = await _mk_conv(sm, uid)
    # turn 1: user sends image A
    await _mk_msg(sm, conv, "user", "here", datetime(2026, 7, 15, 12, 0), [_img("scope/a.jpg", "a.jpg")])
    # turn 2: assistant replies — its own generated image must NOT be picked
    await _mk_msg(sm, conv, "assistant", "nice", datetime(2026, 7, 15, 12, 1), [_img("scope/gen.png", "gen.png", "image/png")])
    # turn 3: user sends image B (newer) — this is the one to edit
    await _mk_msg(sm, conv, "user", "and this", datetime(2026, 7, 15, 12, 2), [_img("scope/b.jpg", "b.jpg")])
    # turn 4: user sends only a document — must be skipped
    await _mk_msg(sm, conv, "user", "a doc", datetime(2026, 7, 15, 12, 3), [_doc("scope/r.pdf")])

    att = await _exec(uid, tmp_path)._recent_uploaded_image()
    assert att is not None, "should find an earlier-uploaded image"
    assert att["storage_path"] == "scope/b.jpg", "must pick the NEWEST user image, not the assistant's or the older one"


async def test_returns_none_when_user_never_uploaded(sm, tmp_path):
    uid = "u-edit-2"
    conv = await _mk_conv(sm, uid)
    await _mk_msg(sm, conv, "user", "just text", datetime(2026, 7, 15, 12, 0), None)
    await _mk_msg(sm, conv, "user", "a doc only", datetime(2026, 7, 15, 12, 1), [_doc("scope/r.pdf")])

    assert await _exec(uid, tmp_path)._recent_uploaded_image() is None


async def test_scoped_to_current_user(sm, tmp_path):
    # Another user's image must not leak into this user's edit.
    other, me = "u-other", "u-me"
    conv_other = await _mk_conv(sm, other)
    await _mk_msg(sm, conv_other, "user", "mine", datetime(2026, 7, 15, 12, 0), [_img("scope/other.jpg")])

    assert await _exec(me, tmp_path)._recent_uploaded_image() is None


async def test_multiple_images_in_one_message_picks_last(sm, tmp_path):
    uid = "u-edit-3"
    conv = await _mk_conv(sm, uid)
    await _mk_msg(sm, conv, "user", "two pics", datetime(2026, 7, 15, 12, 0),
                  [_img("scope/first.jpg", "first.jpg"), _img("scope/last.jpg", "last.jpg")])

    att = await _exec(uid, tmp_path)._recent_uploaded_image()
    assert att is not None and att["storage_path"] == "scope/last.jpg"


# ── source-format normalization (so a found image also EDITS) ────────────

def _jpeg_bytes() -> bytes:
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, format="JPEG")
    return buf.getvalue()


def test_normalize_heic_labelled_jpeg_becomes_png():
    # The real data shape: JPEG bytes the old mobile build tagged image/heic.
    from PIL import Image
    import io
    out_bytes, out_name, out_mime = ToolExecutor._normalize_edit_source(
        _jpeg_bytes(), "IMG_1100.heic", "image/heic")
    assert out_mime == "image/png"
    assert out_name.endswith(".png")
    assert Image.open(io.BytesIO(out_bytes)).format == "PNG"


def test_normalize_passthrough_for_supported_types():
    raw = _jpeg_bytes()
    for mime in ("image/jpeg", "image/png", "image/webp"):
        b, n, m = ToolExecutor._normalize_edit_source(raw, "x.jpg", mime)
        assert (b, n, m) == (raw, "x.jpg", mime), f"{mime} must pass through untouched"


def test_normalize_genuine_heic_becomes_png():
    # A real .heic upload (web path) — decodable only with pillow-heif, which
    # tool_executor registers at import. Skip cleanly where the wheel is absent.
    import pytest
    pytest.importorskip("pillow_heif")
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (200, 50, 50)).save(buf, format="HEIF")
    out_bytes, out_name, out_mime = ToolExecutor._normalize_edit_source(
        buf.getvalue(), "IMG_2000.heic", "image/heic")
    assert out_mime == "image/png" and out_name.endswith(".png")
    assert Image.open(io.BytesIO(out_bytes)).format == "PNG"


def test_normalize_undecodable_falls_back_to_original():
    # Genuine non-image bytes under an unsupported label → keep original,
    # let OpenAI return a clean error rather than crashing the turn.
    junk = b"not-an-image"
    b, n, m = ToolExecutor._normalize_edit_source(junk, "weird.heic", "image/heic")
    assert (b, n, m) == (junk, "weird.heic", "image/heic")
