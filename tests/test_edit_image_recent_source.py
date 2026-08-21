"""Which image an edit resolves to.

Round 16, in production: the user generated a picture of two cartoon
characters, then said "Make morty playing with the portal machine". The edit
went to a selfie they had uploaded days earlier, in a different chat, and the
result was their own face composited into a science fiction laboratory.

The old resolver could not have done anything else. It read
``Message.attachments`` filtered to ``role == "user"`` — and a generated image
rides the ASSISTANT message — across EVERY conversation the user has. The
just-made picture was not outranked; it was never a candidate.

The tests below are written so the old implementation FAILS them:

* `test_generated_image_in_thread_wins_over_older_upload` — the Round 16 shape.
  Old code returns the upload; there is no ordering that saves it.
* `test_never_reaches_into_another_conversation` — the global scan. Old code
  returns the other chat's photo; the new one returns nothing and the tool asks.

The rest pins what must NOT regress: an earlier upload in the same thread is
still editable without re-attaching (the PR #246 fix), another user's image
never leaks, and a source that was FOUND still normalises into something the
edits endpoint accepts.

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

from app.agent.image_artifacts import (
    ORIGIN_EDITED,
    ORIGIN_GENERATED,
    ORIGIN_UPLOADED,
    choices_hint,
    resolve_by_id,
    resolve_implicit,
    thread_images,
    turn_artifacts,
)
from app.agent.tool_executor import ToolExecutor
from app.db.models.conversation import Conversation, Message


# No `users` table — the query only joins conversations→messages and sqlite
# doesn't enforce the FK, so a bare user_id string suffices.
_CONVERSATIONS_DDL = """CREATE TABLE IF NOT EXISTS conversations (
    id VARCHAR(36) PRIMARY KEY, user_id VARCHAR(36) REFERENCES users(id),
    title VARCHAR(500), day_chat_id VARCHAR(36),
    channel VARCHAR(50) DEFAULT 'web', is_active BOOLEAN DEFAULT 1,
    started_at TIMESTAMP, ended_at TIMESTAMP, updated_at TIMESTAMP,
    message_count INTEGER DEFAULT 0, total_tokens INTEGER DEFAULT 0,
    builder_mode VARCHAR(10), metadata_json TEXT
)"""

_MESSAGES_DDL = """CREATE TABLE IF NOT EXISTS messages (
    id VARCHAR(50) PRIMARY KEY, conversation_id VARCHAR(36) REFERENCES conversations(id),
    day_chat_id VARCHAR(36), role VARCHAR(20),
    content TEXT, created_at TIMESTAMP, tokens_prompt INTEGER,
    tokens_completion INTEGER, model_used VARCHAR(50),
    memories_retrieved_json TEXT, processing_time_ms INTEGER,
    metadata_json TEXT, embedding_json TEXT, embedding BLOB,
    channel VARCHAR(50), source VARCHAR(50), origin VARCHAR(50),
    reply_to_message_id VARCHAR(50), attachments TEXT
)"""


async def _make_engine():
    engine = create_async_engine(
        "sqlite+aiosqlite://", connect_args={"check_same_thread": False}
    )
    async with engine.begin() as conn:
        for stmt in (_CONVERSATIONS_DDL, _MESSAGES_DDL):
            await conn.run_sync(lambda c, s=stmt: c.execute(text(s)))
    return engine


def test_ddl_covers_every_message_column():
    """The hand-rolled DDL above is a copy of the model, and a copy drifts.

    It already had: `messages.origin` was added to the model, the fixture was
    not updated, and every DB-backed test in this file has been failing on
    `no such column: origin` — silently, because this file is not in the
    default lane. A test that cannot insert a row proves nothing about which
    image an edit picks, which is exactly the assertion this file exists to
    make. So the DDL is checked against the model rather than trusted.
    """
    import re

    from app.db.models.conversation import Message

    ddl = _MESSAGES_DDL.lower()
    declared = {
        c.name for c in Message.__table__.columns
        if not re.search(rf"\b{re.escape(c.name)}\b", ddl)
    }
    assert not declared, (
        f"tests/test_edit_image_recent_source.py's messages DDL is missing "
        f"{sorted(declared)} — add them, or the fixture cannot insert a row"
    )


@pytest_asyncio.fixture
async def sm(monkeypatch):
    """Session maker on a self-contained sqlite engine, patched in as the
    one the artifact lookup imports."""
    engine = await _make_engine()
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    import app.db.database as _dbmod
    monkeypatch.setattr(_dbmod, "async_session_maker", maker)
    yield maker
    await engine.dispose()


def _img(path: str, name: str = "photo.jpg", mime: str = "image/jpeg", att_id: str = "") -> dict:
    return {"id": att_id or uuid.uuid4().hex, "filename": name, "mime_type": mime,
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


async def _resolve(conv, uid, pending=(), inbound=()):
    return await resolve_implicit(
        conversation_id=conv, user_id=uid,
        pending_attachments=pending, inbound_media=inbound,
    )


# ── The Round 16 defect ──────────────────────────────────────────────────

async def test_generated_image_in_thread_wins_over_older_upload(sm):
    """THE bug. The agent generated a picture this conversation; the user has
    an older selfie in the same conversation. "Make morty playing with the
    portal machine" must resolve to the generated picture.

    The old resolver filtered to role == "user", so the generated image was not
    in the candidate set at all and the selfie was the only answer."""
    uid = "u-r16"
    conv = await _mk_conv(sm, uid)
    await _mk_msg(sm, conv, "user", "here's me", datetime(2026, 8, 1, 9, 0),
                  [_img("s/selfie.jpg", "selfie.jpg")])
    await _mk_msg(sm, conv, "user", "draw rick and morty", datetime(2026, 8, 20, 12, 0))
    await _mk_msg(sm, conv, "assistant", "here you go", datetime(2026, 8, 20, 12, 1),
                  [_img("s/gen.png", "image_ab12cd34.png", "image/png")])

    art = await _resolve(conv, uid)
    assert art is not None
    assert art.storage_path == "s/gen.png", (
        "the just-generated picture is what 'it' means — the selfie is what "
        "Round 16 shipped"
    )
    assert art.origin == ORIGIN_GENERATED
    assert art.role == "assistant"


async def test_never_reaches_into_another_conversation(sm):
    """The pointer was global. A photo in a DIFFERENT chat must not be
    reachable — the tool asks instead, which is what the user would have
    wanted at 'which picture do you mean?'."""
    uid = "u-scope"
    other = await _mk_conv(sm, uid)
    await _mk_msg(sm, other, "user", "old selfie", datetime(2026, 8, 1, 9, 0),
                  [_img("s/other-chat.jpg")])
    here = await _mk_conv(sm, uid)
    await _mk_msg(sm, here, "user", "hello", datetime(2026, 8, 20, 12, 0))

    assert await _resolve(here, uid) is None, "an image from another chat is not a candidate"
    # ...and it really is there, so the None above is scoping, not an empty DB.
    assert len(await thread_images(conversation_id=other, user_id=uid)) == 1


# ── What must not regress ────────────────────────────────────────────────

async def test_earlier_upload_in_same_thread_still_resolves(sm):
    """PR #246's fix: "edit the photo I sent earlier" works without a
    re-attach, as long as 'earlier' is this conversation."""
    uid = "u-earlier"
    conv = await _mk_conv(sm, uid)
    await _mk_msg(sm, conv, "user", "here", datetime(2026, 7, 15, 12, 0),
                  [_img("s/a.jpg", "a.jpg")])
    await _mk_msg(sm, conv, "user", "and this", datetime(2026, 7, 15, 12, 2),
                  [_img("s/b.jpg", "b.jpg")])
    await _mk_msg(sm, conv, "user", "a doc", datetime(2026, 7, 15, 12, 3), [_doc("s/r.pdf")])

    art = await _resolve(conv, uid)
    assert art is not None and art.storage_path == "s/b.jpg"
    assert art.origin == ORIGIN_UPLOADED


async def test_scoped_to_current_user(sm):
    """Another user's image must not leak into this user's edit, even with a
    conversation id in hand."""
    conv_other = await _mk_conv(sm, "u-other")
    await _mk_msg(sm, conv_other, "user", "mine", datetime(2026, 7, 15, 12, 0),
                  [_img("s/other.jpg")])

    assert await _resolve(conv_other, "u-me") is None


async def test_multiple_images_in_one_message_picks_last(sm):
    uid = "u-multi"
    conv = await _mk_conv(sm, uid)
    await _mk_msg(sm, conv, "user", "two pics", datetime(2026, 7, 15, 12, 0),
                  [_img("s/first.jpg", "first.jpg"), _img("s/last.jpg", "last.jpg")])

    art = await _resolve(conv, uid)
    assert art is not None and art.storage_path == "s/last.jpg"


async def test_no_conversation_id_resolves_to_nothing(sm):
    """A turn with no conversation (a subagent sentinel) has no thread to
    scope to. It must ask rather than fall back to a global scan."""
    assert await _resolve(None, "u-x") is None


async def test_document_only_history_is_not_an_image(sm):
    uid = "u-doc"
    conv = await _mk_conv(sm, uid)
    await _mk_msg(sm, conv, "user", "just text", datetime(2026, 7, 15, 12, 0), None)
    await _mk_msg(sm, conv, "user", "a doc only", datetime(2026, 7, 15, 12, 1), [_doc("s/r.pdf")])

    assert await _resolve(conv, uid) is None


# ── This turn, which is not in the database yet ──────────────────────────

def test_turn_upload_outranks_this_turns_generated_output():
    """A file attached to THIS message is the least ambiguous signal there is
    — "here's my photo, put me on a beach" — so it wins over a picture we
    produced a moment ago while answering."""
    arts = turn_artifacts(
        pending_attachments=[_img("s/gen.png", "image_1.png", "image/png")],
        inbound_media=[_img("s/up.jpg", "up.jpg")],
    )
    assert arts[0].storage_path == "s/up.jpg"
    assert arts[0].origin == ORIGIN_UPLOADED
    assert arts[1].origin == ORIGIN_GENERATED


def test_this_turns_output_beats_thread_history_shape():
    """generate → edit inside ONE turn. Nothing of the turn is persisted until
    it ends, so without this the edit would reach into history for a picture
    the user is not talking about."""
    arts = turn_artifacts(
        pending_attachments=[_img("s/one.png", "image_1.png", "image/png"),
                             _img("s/two.png", "edited_2.png", "image/png")],
        inbound_media=[],
    )
    assert arts[0].storage_path == "s/two.png", "the newest output of this turn"
    assert arts[0].origin == ORIGIN_EDITED
    assert arts[0].turn_scope == "this_turn"


async def test_this_turn_beats_history(sm):
    uid = "u-turn"
    conv = await _mk_conv(sm, uid)
    await _mk_msg(sm, conv, "assistant", "older", datetime(2026, 8, 1, 9, 0),
                  [_img("s/old.png", "image_old.png", "image/png")])

    art = await _resolve(conv, uid,
                         pending=[_img("s/new.png", "image_new.png", "image/png")])
    assert art is not None and art.storage_path == "s/new.png"


# ── Addressing one by id ─────────────────────────────────────────────────

async def test_resolve_by_id_finds_a_history_image(sm):
    uid = "u-id"
    conv = await _mk_conv(sm, uid)
    wanted = uuid.uuid4().hex
    await _mk_msg(sm, conv, "assistant", "a", datetime(2026, 8, 1, 9, 0),
                  [_img("s/a.png", "image_a.png", "image/png", att_id=wanted)])
    await _mk_msg(sm, conv, "assistant", "b", datetime(2026, 8, 1, 9, 5),
                  [_img("s/b.png", "image_b.png", "image/png")])

    art = await resolve_by_id(wanted.upper(), conversation_id=conv, user_id=uid,
                              pending_attachments=(), inbound_media=())
    assert art is not None and art.storage_path == "s/a.png", (
        "an explicit id must beat recency, case-insensitively"
    )


async def test_resolve_by_id_returns_none_for_another_conversation(sm):
    """An id is a handle for something in THIS thread. Resolving one from
    another chat would reintroduce the global pointer through a side door."""
    uid = "u-id2"
    other = await _mk_conv(sm, uid)
    wanted = uuid.uuid4().hex
    await _mk_msg(sm, other, "assistant", "a", datetime(2026, 8, 1, 9, 0),
                  [_img("s/a.png", "image_a.png", "image/png", att_id=wanted)])
    here = await _mk_conv(sm, uid)

    assert await resolve_by_id(wanted, conversation_id=here, user_id=uid,
                               pending_attachments=(), inbound_media=()) is None


async def test_choices_hint_names_the_options(sm):
    uid = "u-hint"
    conv = await _mk_conv(sm, uid)
    await _mk_msg(sm, conv, "user", "me", datetime(2026, 8, 1, 9, 0), [_img("s/u.jpg", "me.jpg")])
    await _mk_msg(sm, conv, "assistant", "art", datetime(2026, 8, 1, 9, 1),
                  [_img("s/g.png", "image_g.png", "image/png")])

    hint = choices_hint(await thread_images(conversation_id=conv, user_id=uid))
    assert "image_g.png" in hint and "me.jpg" in hint
    assert "image_id" in hint


def test_describe_says_where_it_came_from():
    from app.agent.image_artifacts import ImageArtifact
    gen = ImageArtifact(attachment=_img("s/g.png", "g.png"), origin=ORIGIN_GENERATED)
    up = ImageArtifact(attachment=_img("s/u.jpg", "u.jpg"), origin=ORIGIN_UPLOADED,
                       turn_scope="this_turn")
    assert "generated" in gen.describe() and "g.png" in gen.describe()
    assert "attached to this message" in up.describe()


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
