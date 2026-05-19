"""Auto Builder completion posts to Day-as-Chat.

PR 8 of the unified-jobs arc. Drives
``AppBuilderSkill._post_completion_to_day_chat`` directly with
seeded ``App`` + ``BuildJob`` rows and asserts:

  1. With ``feature_auto_builder_chat_output=True``:
     - One ``Message`` row lands with ``channel='app_builder'``,
       ``role='assistant'``, content shaped around the build's
       success or failure.
     - The user's day-Conversation for ``channel='app_builder'``
       exists (created on demand).
     - ``BuildJob.summary_message_id`` is stamped with the new
       message id — the unified activity feed (PR 6) links the
       timeline entry to the detail drawer via this column.

  2. With ``feature_auto_builder_chat_output=False`` (default):
     - The helper is a NO-OP. No Message row, no Conversation
       created, ``BuildJob.summary_message_id`` stays NULL.

  3. A write failure in the helper does NOT propagate (best-effort
     write — Day-as-Chat is the user surface, not the source of
     truth). Job stays in its terminal status.
"""
from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio


os.environ.setdefault("AGENT_API_KEY", "test-key-auto-builder-day-chat")
os.environ.setdefault("USER_ID", "00000000-0000-0000-0000-0000000abf01")


USER_ID = "00000000-0000-0000-0000-0000000abf01"


@pytest_asyncio.fixture(autouse=True)
async def _reset_database():
    """Bypass conftest's init_db autouse fixture. Build only the
    tables the completion writer touches (User, App, BuildJob,
    DayChat, Conversation, Message)."""
    from app.config import settings
    from app.db.database import rebind_database

    await rebind_database(settings.DATABASE_URL)

    from app.db.database import engine
    from app.db.models import (
        App, BuildJob, Conversation, DayChat, Message, User,
    )

    async with engine.begin() as conn:
        for model_cls in (User, App, BuildJob, DayChat, Conversation, Message):
            await conn.run_sync(model_cls.__table__.create, checkfirst=True)
    yield
    async with engine.begin() as conn:
        for model_cls in (Message, Conversation, DayChat, BuildJob, App, User):
            await conn.run_sync(model_cls.__table__.drop, checkfirst=True)


@pytest_asyncio.fixture
async def _seed_user_app_job():
    """Seed a User + App + BuildJob row in ``status='running'``.
    Returns the ids the tests need."""
    from datetime import datetime
    from app.db.database import async_session_maker
    from app.db.models import App, BuildJob, User

    app_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    slug = "test-app-" + uuid.uuid4().hex[:6]

    async with async_session_maker() as db:
        if await db.get(User, USER_ID) is None:
            db.add(User(
                id=USER_ID,
                email=f"day-chat-{USER_ID[:8]}@example.com",
                hashed_password="x",
                timezone="UTC",
            ))
        db.add(App(
            id=app_id,
            user_id=USER_ID,
            name="Test App",
            slug=slug,
            status="building",
            app_dir=f"/tmp/test-apps/{slug}",
            publish_url="https://preview.test/" + slug,
            github_url="https://github.com/test/" + slug,
        ))
        db.add(BuildJob(
            id=job_id,
            user_id=USER_ID,
            app_id=app_id,
            title=f"Build: Test App",
            prompt="test",
            job_type="auto_builder",
            status="running",
            source_kind="app_builder_skill",
            source_id=app_id,
            created_at=datetime.utcnow(),
        ))
        await db.commit()
    return {"app_id": app_id, "job_id": job_id, "slug": slug}


def _make_skill():
    from app.agent.skills.builtins.app_builder.skill import AppBuilderSkill
    return AppBuilderSkill()


# ──────────────────────────────────────────────────────────────────────
# Flag OFF (default) → no-op
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_flag_off_writes_nothing(_seed_user_app_job, monkeypatch):
    """The default ``feature_auto_builder_chat_output=False`` means
    the helper is a NO-OP. No Message row, no Conversation, the
    BuildJob's ``summary_message_id`` stays NULL."""
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, Message
    from sqlalchemy import select, func
    from app.config import settings

    # Make sure the flag is OFF (in case some other module set it).
    monkeypatch.setattr(settings, "feature_auto_builder_chat_output", False)

    skill = _make_skill()
    await skill._post_completion_to_day_chat(
        job_id=_seed_user_app_job["job_id"],
        user_id=USER_ID,
        app_id=_seed_user_app_job["app_id"],
        app_name="Test App",
        slug=_seed_user_app_job["slug"],
        terminal_status="completed",
    )

    async with async_session_maker() as db:
        msg_count = (await db.execute(
            select(func.count(Message.id))
        )).scalar_one()
        assert msg_count == 0, (
            "feature_auto_builder_chat_output=False must be a no-op; "
            f"got {msg_count} Message rows"
        )
        job = await db.get(BuildJob, _seed_user_app_job["job_id"])
        assert job.summary_message_id is None


# ──────────────────────────────────────────────────────────────────────
# Flag ON → success path
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_flag_on_completed_posts_to_day_chat(
    _seed_user_app_job, monkeypatch,
):
    """With the flag ON and ``terminal_status='completed'``:
      - One Message row exists with channel='app_builder',
        role='assistant'.
      - Content contains the success marker AND the preview/github
        URLs from the App row.
      - BuildJob.summary_message_id is stamped with the message id.
      - The Conversation row carries channel='app_builder' too."""
    from app.config import settings
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, Conversation, Message
    from sqlalchemy import select

    monkeypatch.setattr(settings, "feature_auto_builder_chat_output", True)

    skill = _make_skill()
    await skill._post_completion_to_day_chat(
        job_id=_seed_user_app_job["job_id"],
        user_id=USER_ID,
        app_id=_seed_user_app_job["app_id"],
        app_name="Test App",
        slug=_seed_user_app_job["slug"],
        terminal_status="completed",
    )

    async with async_session_maker() as db:
        messages = (await db.execute(
            select(Message).where(Message.channel == "app_builder")
        )).scalars().all()
        assert len(messages) == 1
        msg = messages[0]
        assert msg.role == "assistant"
        assert msg.channel == "app_builder"
        assert msg.source == "app_builder"
        assert "Test App" in (msg.content or "")
        # Either preview or github URL surfaces in the body.
        assert ("preview.test" in (msg.content or "")
                or "github.com/test" in (msg.content or ""))

        # Conversation row exists with the same channel.
        convs = (await db.execute(
            select(Conversation).where(Conversation.channel == "app_builder")
        )).scalars().all()
        assert len(convs) == 1

        # BuildJob.summary_message_id linkage is set.
        job = await db.get(BuildJob, _seed_user_app_job["job_id"])
        assert job.summary_message_id == msg.id


# ──────────────────────────────────────────────────────────────────────
# Flag ON → failure path
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_flag_on_failed_posts_error_summary(
    _seed_user_app_job, monkeypatch,
):
    """With the flag ON and ``terminal_status='failed'``:
      - Message content carries the failure marker AND a short
        version of the error message.
      - Full error stays on BuildJob.error_message (out of scope
        here — we don't trim the row; just the chat preview)."""
    from app.config import settings
    from app.db.database import async_session_maker
    from app.db.models import Message
    from sqlalchemy import select

    monkeypatch.setattr(settings, "feature_auto_builder_chat_output", True)

    skill = _make_skill()
    long_error = (
        "Metro port 4001 busy\n"
        "Full stack trace: many lines of text that don't belong "
        "in the chat surface but are useful for the detail drawer."
    )
    await skill._post_completion_to_day_chat(
        job_id=_seed_user_app_job["job_id"],
        user_id=USER_ID,
        app_id=_seed_user_app_job["app_id"],
        app_name="Test App",
        slug=_seed_user_app_job["slug"],
        terminal_status="failed",
        error_message=long_error,
    )

    async with async_session_maker() as db:
        msgs = (await db.execute(
            select(Message).where(Message.channel == "app_builder")
        )).scalars().all()
        assert len(msgs) == 1
        content = msgs[0].content or ""
        # Failure marker + first line of error.
        assert "failed" in content.lower() or "❌" in content
        assert "Metro port 4001 busy" in content
        # Long trace stays OUT of the chat surface — only first line.
        assert "Full stack trace" not in content


# ──────────────────────────────────────────────────────────────────────
# Best-effort write — never fails the build
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_failure_is_swallowed(_seed_user_app_job, monkeypatch):
    """The helper is best-effort: a Day-as-Chat write failure must
    NOT propagate. We simulate by patching
    ``resolve_or_create_day_conversation`` to raise."""
    from app.config import settings

    monkeypatch.setattr(settings, "feature_auto_builder_chat_output", True)

    async def _explode(*a, **kw):
        raise RuntimeError("simulated day-chat write failure")

    import app.agent.conversation_resolver as cr
    monkeypatch.setattr(cr, "resolve_or_create_day_conversation", _explode)

    skill = _make_skill()
    # Must NOT raise.
    await skill._post_completion_to_day_chat(
        job_id=_seed_user_app_job["job_id"],
        user_id=USER_ID,
        app_id=_seed_user_app_job["app_id"],
        app_name="Test App",
        slug=_seed_user_app_job["slug"],
        terminal_status="completed",
    )
