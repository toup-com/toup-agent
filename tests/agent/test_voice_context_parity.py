"""G-19a PR-A — one context assembler, and the byte pin that proves it.

Voice's instructions and text chat's system prompt rendered the SAME four
identity documents through TWO hand-copied renderers living in two
processes (`agent_runner._build_system_prompt` on the tenant,
`ws_realtime.build_realtime_instructions` on platform-api). Nine live
divergences were catalogued; this PR ends the two that matter most —
identity ORDER and the identity ANCHOR — by extracting both into module
level pure functions that BOTH callers use.

The load-bearing constraint is that the chat path must not move by one
byte. `IDENTITY_GOLDEN_*` below were captured from the PRE-refactor
`_build_system_prompt` and are asserted against its assembled output;
they were written and run green BEFORE the extraction existed.

Also pinned here:
  * the anchor goldens for all four (name × format) combinations;
  * the UNIFIED ordering rule — soul first, even when a higher-priority
    non-soul row exists (the runner's rule wins; voice's plain
    priority-sorted append is the one that changes);
  * #488 by construction — `build_voice_context` resolves TODAY's day
    chat via `resolve_day_chat_id_for_now`, so a user whose newest day
    chat is YESTERDAY gets today's (empty) history, never yesterday's
    transcript relabelled.

Tables: this file is swept under RUN_MODE=platform, where init_db does
not build `identities` (AGENT_ONLY, base.py). It therefore creates every
table it needs from Base.metadata itself — the same trick
tests/test_shared_day_context_invariants.py uses — rather than parking
another file in COVERAGE_DEBT.
"""

from __future__ import annotations

import uuid
from datetime import date as Date, datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio


TORONTO = "America/Toronto"

# 2026-08-04 02:30 UTC == 2026-08-03 22:30 in Toronto. The UTC date and
# the user's local date are DIFFERENT here, which is the only instant at
# which a tz-blind day resolver is distinguishable from a correct one.
FROZEN_UTC = datetime(2026, 8, 4, 2, 30, 0, tzinfo=timezone.utc)
LOCAL_TODAY = "2026-08-03"
LOCAL_YESTERDAY = "2026-08-02"


# ── Table fixture ─────────────────────────────────────────────────────

_NEEDED_TABLES = [
    "users", "identities", "agent_configs", "memories",
    "day_chats", "conversations", "messages",
]


@pytest_asyncio.fixture
async def voice_tables():
    """Create every table this file touches, whatever the RUN_MODE.

    `identities` is AGENT_ONLY and `day_chats`/`conversations`/`messages`
    are too, so the platform sweep's init_db skips them. Creating them
    here keeps these pins in the default sweep instead of COVERAGE_DEBT.
    """
    from sqlalchemy import inspect as sa_inspect
    from app.db.database import engine
    from app.db.models.base import Base
    import app.db.models  # noqa: F401 — registers every mapper

    async with engine.begin() as conn:
        for name in _NEEDED_TABLES:
            table = Base.metadata.tables.get(name)
            if table is None:
                continue
            try:
                await conn.run_sync(table.create, checkfirst=True)
            except Exception:
                # e.g. a pgvector column on a backend without the type.
                pass

    async with engine.connect() as conn:
        existing = await conn.run_sync(lambda c: set(sa_inspect(c).get_table_names()))
    missing = [n for n in _NEEDED_TABLES if n not in existing]
    if missing:
        pytest.skip(f"cannot create {missing} on this backend")
    yield


# ── Fixture data ──────────────────────────────────────────────────────

SOUL_TEXT = "Warm, direct, allergic to filler."
PROFILE_TEXT = "Nariman. Toronto. Ships at night."
INSTRUCTIONS_TEXT = "Never guess a number you did not measure."
TOOLS_TEXT = "Prefer exec over describing what exec would do."
AGENT_NAME = "Aria"


async def _seed(db, user_id: str, *, rows, agent_name: str | None):
    """rows: list of (identity_type, content, priority)."""
    from app.db.models import Identity, AgentConfig
    from app.db.models.user import User

    db.add(User(
        id=user_id, email=f"g19a-{uuid.uuid4().hex[:10]}@example.com",
        hashed_password="x", name="G19a", timezone=TORONTO,
    ))
    for itype, content, priority in rows:
        db.add(Identity(
            id=str(uuid.uuid4()), user_id=user_id, identity_type=itype,
            name=itype, content=content, priority=priority, is_active=True,
        ))
    if agent_name is not None:
        db.add(AgentConfig(user_id=user_id, agent_name=agent_name))
    await db.commit()


FOUR_DOCS = [
    ("soul", SOUL_TEXT, 100),
    ("user_profile", PROFILE_TEXT, 90),
    ("agent_instructions", INSTRUCTIONS_TEXT, 50),
    ("tools", TOOLS_TEXT, 10),
]


# ── The goldens — captured from the PRE-refactor builder ──────────────

IDENTITY_GOLDEN_FOUR_DOCS = (
    f"# Core Identity\n{SOUL_TEXT}\n\n"
    f"# About the User\n{PROFILE_TEXT}\n\n"
    f"# Behavioral Guidelines\n{INSTRUCTIONS_TEXT}\n\n"
    f"# Tool Guidelines\n{TOOLS_TEXT}"
)

IDENTITY_GOLDEN_NO_SOUL = (
    "# Core Identity\n"
    "You are the user's personal agent — present in their life as a "
    "real person who happens to be exceptionally capable. Not a "
    "chatbot, not an FAQ, not a help desk. Be warm, direct, curious. "
    "A trusted friend who can also actually get things done.\n\n"
    f"# About the User\n{PROFILE_TEXT}"
)

ANCHOR_GOLDEN_CHAT_NAMED = (
    "# Identity\n"
    f"Your name is **{AGENT_NAME}**. That is your name — use it "
    "when self-identifying.\n\n"
    "Toup is the **platform** you run on (toup.ai), NOT your name. "
    "Never introduce yourself as 'Toup' or call yourself 'Toup'. "
    f"When asked your name, say **{AGENT_NAME}**, not Toup.\n\n"
    "You are NOT Claude, NOT GPT, NOT Sonnet, NOT Opus, NOT any "
    "specific provider model. When the user asks what you are, "
    "who built you, or what model is powering you, answer as "
    f"**{AGENT_NAME}** — never name the underlying LLM provider "
    "or version, and don't disclose Toup's underlying tech stack "
    "or how it's built (that's proprietary). The provider may "
    "change without notice; your identity to the user is stable."
)

ANCHOR_GOLDEN_CHAT_UNNAMED = (
    "# Identity\n"
    "You don't have a name yet — the user hasn't picked one. "
    "Don't introduce yourself with a made-up name, and especially "
    "do NOT call yourself 'Toup'. Toup is the platform you run on "
    "(toup.ai), not your name. If naming comes up naturally, ask "
    "what they'd like to call you.\n\n"
    "You are NOT Claude, NOT GPT, NOT Sonnet, NOT Opus, NOT any "
    "specific provider model. When the user asks what model is "
    "powering you, answer as the agent — never name the underlying "
    "LLM provider or version, and don't disclose Toup's underlying "
    "tech stack or how it's built (that's proprietary)."
)

ANCHOR_GOLDEN_VOICE_NAMED = (
    "# Who you are (identity)\n"
    f"Your name is {AGENT_NAME}. That is your name — use it when you "
    f"introduce or refer to yourself, and when the user asks your name, "
    f"answer {AGENT_NAME} (never 'Toup', which is only the platform).\n"
    "You are the user's own personal agent on Toup. Toup is the "
    "platform you run on (toup.ai), not your name.\n"
    "You are NOT Claude, NOT GPT, NOT Sonnet, NOT Opus, NOT any "
    "specific provider model. If the user asks what model you are, "
    "who built you, what powers you, or what technology/stack Toup "
    "is built with, answer as their agent — never name the "
    "underlying LLM provider or version, and don't disclose the "
    "underlying tech stack (it's proprietary). The provider may "
    "change without notice; your identity to the user is stable.\n"
    "(If the user asks who FOUNDED or owns Toup — the company — that "
    "is a separate, allowed question; answer it if you know. This "
    "guard is only about the underlying model/technology.)"
)

ANCHOR_GOLDEN_VOICE_UNNAMED = (
    "# Who you are (identity)\n"
    "You are the user's own personal agent on Toup. Toup is the "
    "platform you run on (toup.ai), not your name.\n"
    "You are NOT Claude, NOT GPT, NOT Sonnet, NOT Opus, NOT any "
    "specific provider model. If the user asks what model you are, "
    "who built you, what powers you, or what technology/stack Toup "
    "is built with, answer as their agent — never name the "
    "underlying LLM provider or version, and don't disclose the "
    "underlying tech stack (it's proprietary). The provider may "
    "change without notice; your identity to the user is stable.\n"
    "(If the user asks who FOUNDED or owns Toup — the company — that "
    "is a separate, allowed question; answer it if you know. This "
    "guard is only about the underlying model/technology.)"
)


# ── Helper: run the real _build_system_prompt against sqlite ──────────

async def _build_prompt(user_id: str, channel: str | None = None) -> str:
    from app.db import async_session_maker
    from app.agent.agent_runner import AgentRunner

    runner = AgentRunner(llm_service=AsyncMock(), tool_executor=AsyncMock())
    with patch(
        "app.services.memory_service.MemoryService.hybrid_search",
        AsyncMock(return_value=[]),
    ), patch(
        "app.services.user_portrait_service.UserPortraitService.get_or_build_portrait",
        AsyncMock(return_value=""),
    ):
        async with async_session_maker() as db:
            return await runner._build_system_prompt(
                db=db, user_id=user_id, user_message="hello", channel=channel,
            )


# ──────────────────────────────────────────────────────────────────────
# 1. THE BYTE PIN — written and run green BEFORE the extraction existed
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_identity_and_anchor_are_byte_identical(voice_tables):
    """`identity` is _FULL_SECTIONS[0] and `identity_anchor` is [1], and
    the assembler joins with "\\n\\n" — so the prompt's first bytes ARE
    those two sections. Any drift in wording, ordering, spacing or the
    join changes this prefix.
    """
    from app.db import async_session_maker

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed(db, user_id, rows=FOUR_DOCS, agent_name=AGENT_NAME)

    prompt = await _build_prompt(user_id)
    expected = IDENTITY_GOLDEN_FOUR_DOCS + "\n\n" + ANCHOR_GOLDEN_CHAT_NAMED

    assert prompt.startswith(expected), (
        "chat identity/anchor prefix moved.\n"
        f"--- expected ---\n{expected!r}\n"
        f"--- got ---\n{prompt[:len(expected)]!r}"
    )


@pytest.mark.asyncio
async def test_chat_no_soul_fallback_is_byte_identical(voice_tables):
    """The no-soul default fires whenever the `soul` row is absent, even
    with other identities present — and it renders FIRST."""
    from app.db import async_session_maker

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed(
            db, user_id,
            rows=[("user_profile", PROFILE_TEXT, 90)], agent_name=None,
        )

    prompt = await _build_prompt(user_id)
    expected = IDENTITY_GOLDEN_NO_SOUL + "\n\n" + ANCHOR_GOLDEN_CHAT_UNNAMED
    assert prompt.startswith(expected), (
        f"--- expected ---\n{expected!r}\n--- got ---\n{prompt[:len(expected)]!r}"
    )


@pytest.mark.asyncio
async def test_soul_is_hoisted_above_a_higher_priority_row(voice_tables):
    """UNIFIED ORDERING RULE, stated explicitly: soul renders FIRST even
    when a non-soul document carries a HIGHER priority.

    The two copies disagreed here (D1): the runner hoists souls with
    `insert(0)` on top of the priority sort; voice did a plain
    priority-descending append, so a `user_profile` at 90 rendered ABOVE
    a `soul` at 10. The runner's rule wins — the soul IS the persona and
    an imported/legacy low-priority soul row must not fall behind a
    profile blob.
    """
    from app.db import async_session_maker

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed(db, user_id, rows=[
            ("user_profile", PROFILE_TEXT, 90),
            ("soul", SOUL_TEXT, 10),
        ], agent_name=None)

    prompt = await _build_prompt(user_id)
    expected = (
        f"# Core Identity\n{SOUL_TEXT}\n\n"
        f"# About the User\n{PROFILE_TEXT}"
    )
    assert prompt.startswith(expected), (
        "soul must be hoisted above a higher-priority non-soul row.\n"
        f"--- got ---\n{prompt[:len(expected)]!r}"
    )


# ──────────────────────────────────────────────────────────────────────
# 2. The renderers themselves — pure, no DB
# ──────────────────────────────────────────────────────────────────────

def test_identity_sections_renderer_byte_pin():
    """Same goldens as the builder pin above, straight off the renderer.

    Rows are handed in UNSORTED so the renderer's own ordering rule is
    what produces the golden, not the caller's ORDER BY.
    """
    from app.agent.agent_runner import render_identity_sections

    rows = [
        {"identity_type": "tools", "content": TOOLS_TEXT, "priority": 10},
        {"identity_type": "soul", "content": SOUL_TEXT, "priority": 100},
        {"identity_type": "agent_instructions", "content": INSTRUCTIONS_TEXT,
         "priority": 50},
        {"identity_type": "user_profile", "content": PROFILE_TEXT, "priority": 90},
    ]
    text, has_soul = render_identity_sections(rows)
    assert has_soul is True
    assert text == IDENTITY_GOLDEN_FOUR_DOCS


def test_identity_sections_renderer_soul_hoist_and_default():
    from app.agent.agent_runner import render_identity_sections

    # soul BELOW a user_profile in priority → still first.
    text, has_soul = render_identity_sections([
        {"identity_type": "user_profile", "content": PROFILE_TEXT, "priority": 90},
        {"identity_type": "soul", "content": SOUL_TEXT, "priority": 10},
    ])
    assert has_soul is True
    assert text == (
        f"# Core Identity\n{SOUL_TEXT}\n\n# About the User\n{PROFILE_TEXT}"
    )

    # no soul, but other documents present → the default fires anyway,
    # and reports has_soul=False so the agent_soul de-dup stays correct.
    text, has_soul = render_identity_sections([
        {"identity_type": "user_profile", "content": PROFILE_TEXT, "priority": 90},
    ])
    assert has_soul is False
    assert text == IDENTITY_GOLDEN_NO_SOUL

    # nothing at all → the default persona alone, never an empty string.
    text, has_soul = render_identity_sections([])
    assert has_soul is False
    assert text.startswith("# Core Identity\nYou are the user's personal agent")

    # `system` / `context` documents are dropped by BOTH historical
    # copies. Anti-vacuity: this is what keeps the renderer from becoming
    # a generic "render every row" function.
    text, _ = render_identity_sections([
        {"identity_type": "system", "content": "SHOULD NOT RENDER", "priority": 999},
        {"identity_type": "context", "content": "NOR THIS", "priority": 998},
        {"identity_type": "soul", "content": SOUL_TEXT, "priority": 1},
    ])
    assert "SHOULD NOT RENDER" not in text and "NOR THIS" not in text
    assert text == f"# Core Identity\n{SOUL_TEXT}"


@pytest.mark.parametrize("name,fmt,golden", [
    (AGENT_NAME, "chat", ANCHOR_GOLDEN_CHAT_NAMED),
    (AGENT_NAME, "voice", ANCHOR_GOLDEN_VOICE_NAMED),
    (None, "chat", ANCHOR_GOLDEN_CHAT_UNNAMED),
    (None, "voice", ANCHOR_GOLDEN_VOICE_UNNAMED),
])
def test_identity_anchor_goldens(name, fmt, golden):
    """All four (name × format) combinations, byte-exact.

    The voice goldens are the strings ws_realtime emitted before the
    extraction; the chat goldens are agent_runner's. One function, two
    surfaces, zero drift.
    """
    from app.agent.agent_runner import render_identity_anchor

    assert render_identity_anchor(name, fmt=fmt) == golden


def test_voice_anchor_carries_no_markdown_and_keeps_the_founder_carve_out():
    """The two things that make the voice wording different, asserted as
    behaviour rather than as a diff: a Realtime model reads `**` aloud,
    and "who founded Toup" must stay an ANSWERABLE question."""
    from app.agent.agent_runner import render_identity_anchor

    voice = render_identity_anchor(AGENT_NAME, fmt="voice")
    assert "**" not in voice
    assert "who FOUNDED or owns Toup" in voice

    chat = render_identity_anchor(AGENT_NAME, fmt="chat")
    assert f"**{AGENT_NAME}**" in chat
    assert "who FOUNDED or owns Toup" not in chat

    # The guard itself is on BOTH, whatever the wording.
    for text in (voice, chat):
        assert "NOT Claude, NOT GPT" in text
        assert "underlying LLM provider" in text


# ──────────────────────────────────────────────────────────────────────
# 3. build_voice_context — the assembler
# ──────────────────────────────────────────────────────────────────────

async def _seed_day(db, user_id: str, local_date: Date, msgs):
    """msgs: list of (role, content, channel, hour)."""
    import uuid as _uuid
    from app.db.models import Conversation, Message
    from app.db.models.day_chat import DayChat

    dc_id = str(_uuid.uuid4())
    db.add(DayChat(
        id=dc_id, user_id=user_id, local_date=local_date, timezone=TORONTO,
        summary_status="up_to_date",
    ))
    await db.flush()
    conv_id = str(_uuid.uuid4())
    db.add(Conversation(
        id=conv_id, user_id=user_id, channel="web", day_chat_id=dc_id,
        started_at=datetime(local_date.year, local_date.month, local_date.day, 9, 0, 0),
    ))
    for i, (role, content, channel, hour) in enumerate(msgs):
        db.add(Message(
            id=str(_uuid.uuid4()), conversation_id=conv_id, day_chat_id=dc_id,
            channel=channel, role=role, content=content,
            created_at=datetime(local_date.year, local_date.month,
                                local_date.day, hour, i, 0),
        ))
    await db.commit()
    return dc_id


@pytest.mark.asyncio
async def test_voice_context_identity_matches_the_chat_bytes(voice_tables):
    """The whole point: voice's persona block IS chat's persona block.

    Same rows, same renderer, same bytes — the anchor differs only in the
    wording `fmt="voice"` selects.
    """
    from app.db import async_session_maker
    from app.agent.voice_context import build_voice_context

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed(db, user_id, rows=FOUR_DOCS, agent_name=AGENT_NAME)

    async with async_session_maker() as db:
        ctx = await build_voice_context(
            db, user_id, tz_name=TORONTO, now_utc=FROZEN_UTC,
        )

    assert ctx.sections["identity"] == IDENTITY_GOLDEN_FOUR_DOCS
    assert ctx.sections["identity_anchor"] == ANCHOR_GOLDEN_VOICE_NAMED

    # …and they lead the instructions, in that order (the runner's
    # position for the anchor, not the relay's after-the-transcript one).
    assert ctx.instructions.startswith(
        IDENTITY_GOLDEN_FOUR_DOCS + "\n\n" + ANCHOR_GOLDEN_VOICE_NAMED
    )

    chat_prompt = await _build_prompt(user_id)
    assert chat_prompt.startswith(ctx.sections["identity"]), (
        "voice and chat must render the SAME identity documents"
    )


@pytest.mark.asyncio
async def test_voice_context_soul_hoist_and_no_soul_default(voice_tables):
    """D1 + D3 on the voice side: the relay would have rendered the
    profile FIRST here, and would have emitted no persona at all in the
    second case."""
    from app.db import async_session_maker
    from app.agent.voice_context import build_voice_context

    hoist_id = str(uuid.uuid4())
    plain_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed(db, hoist_id, rows=[
            ("user_profile", PROFILE_TEXT, 90),
            ("soul", SOUL_TEXT, 10),
        ], agent_name=None)
        await _seed(db, plain_id, rows=[
            ("user_profile", PROFILE_TEXT, 90),
        ], agent_name=None)

    async with async_session_maker() as db:
        hoisted = await build_voice_context(
            db, hoist_id, tz_name=TORONTO, now_utc=FROZEN_UTC)
        soulless = await build_voice_context(
            db, plain_id, tz_name=TORONTO, now_utc=FROZEN_UTC)

    assert hoisted.sections["identity"] == (
        f"# Core Identity\n{SOUL_TEXT}\n\n# About the User\n{PROFILE_TEXT}"
    )
    assert soulless.sections["identity"] == IDENTITY_GOLDEN_NO_SOUL
    assert soulless.sections["identity_anchor"] == ANCHOR_GOLDEN_VOICE_UNNAMED


@pytest.mark.asyncio
async def test_488_day_is_today_by_construction_not_the_newest_row(voice_tables):
    """#488, pinned.

    The user has a day chat for YESTERDAY and has said nothing today. The
    relay asked `/api/day-chats?limit=1` (ordered local_date DESC) and
    printed yesterday's transcript under "Today's Full Conversation
    History". This assembler resolves the day from the user's local NOW,
    so it gets today's row — created on the spot — and an empty history.

    The frozen instant matters: 2026-08-04 02:30 UTC is still
    2026-08-03 in Toronto, so a UTC-based resolver would ALSO get the
    wrong day (2026-08-04). Both failure modes are excluded at once.
    """
    from app.db import async_session_maker
    from app.agent.voice_context import build_voice_context

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed(db, user_id, rows=FOUR_DOCS, agent_name=AGENT_NAME)
        await _seed_day(db, user_id, Date(2026, 8, 2), [
            ("user", "book the dentist for me", "web", 14),
            ("assistant", "Booked for Thursday.", "web", 14),
        ])

    async with async_session_maker() as db:
        ctx = await build_voice_context(
            db, user_id, tz_name=TORONTO, now_utc=FROZEN_UTC,
        )

    assert ctx.day_date == LOCAL_TODAY, (
        f"expected the user's LOCAL today {LOCAL_TODAY}; "
        f"{LOCAL_YESTERDAY} is #488 (newest existing row), "
        f"2026-08-04 is the UTC date"
    )
    assert "day_history" not in ctx.sections
    assert "book the dentist" not in ctx.instructions, (
        "yesterday's transcript must not reach the voice prompt at all"
    )
    assert "Today's Full Conversation History" not in ctx.instructions
    # The leg SUCCEEDED and today is genuinely blank — that is `empty`,
    # not `degraded`. Reporting it as degraded would fire the alarm on the
    # first voice call of every morning for every user.
    assert "day" in ctx.empty and not ctx.degraded


@pytest.mark.asyncio
async def test_488_control_todays_messages_do_load(voice_tables):
    """ANTI-VACUITY CONTROL for the pin above: the same code path with
    messages stamped on TODAY renders them. Without this, a resolver that
    simply returned nothing would pass the #488 test."""
    from app.db import async_session_maker
    from app.agent.voice_context import build_voice_context

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed(db, user_id, rows=FOUR_DOCS, agent_name=AGENT_NAME)
        await _seed_day(db, user_id, Date(2026, 8, 3), [
            ("user", "remind me about the flight", "telegram", 9),
            ("assistant", "It leaves at six.", "telegram", 9),
        ])

    async with async_session_maker() as db:
        ctx = await build_voice_context(
            db, user_id, tz_name=TORONTO, now_utc=FROZEN_UTC,
        )

    assert ctx.day_date == LOCAL_TODAY
    day = ctx.sections["day_history"]
    assert day.startswith(
        "# Today's Full Conversation History (2 messages across all channels)"
    )
    assert "User: [telegram" in day and "remind me about the flight" in day
    assert "You: [telegram" in day and "It leaves at six." in day
    assert "day" not in ctx.degraded


@pytest.mark.asyncio
async def test_voice_context_memories_keep_the_dump_shape(voice_tables):
    """D7 deliberately NOT changed: both brains, unranked, voice's
    headers and row format — but read from the TENANT db instead of over
    HTTP from the platform."""
    from app.db import async_session_maker
    from app.db.models.memory import Memory
    from app.agent.voice_context import build_voice_context

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed(db, user_id, rows=FOUR_DOCS, agent_name=AGENT_NAME)
        for brain, cat, content in (
            ("agent", "agent_soul", "Speak plainly."),
            ("user", "preferences", "Flat white, no sugar."),
        ):
            db.add(Memory(
                id=str(uuid.uuid4()), user_id=user_id, brain_type=brain,
                content=content, category=cat, memory_type="semantic",
                importance=0.8, confidence=1.0, strength=1.0,
                memory_level="semantic", decay_rate=0.05,
                is_active=True, is_deleted=False, source_type="test",
                created_at=datetime(2026, 8, 3, 12, 0, 0),
                updated_at=datetime(2026, 8, 3, 12, 0, 0),
            ))
        await db.commit()

    async with async_session_maker() as db:
        ctx = await build_voice_context(
            db, user_id, tz_name=TORONTO, now_utc=FROZEN_UTC,
        )

    assert ctx.sections["agent_brain"] == (
        "# Agent Brain (Permanent Knowledge)\n- [agent_soul] Speak plainly."
    )
    assert ctx.sections["user_brain"] == (
        "# User Brain (What You Know About the User)\n"
        "- [preferences] Flat white, no sugar."
    )
    assert "memories" not in ctx.degraded


@pytest.mark.asyncio
async def test_voice_context_degraded_names_every_empty_leg(voice_tables):
    """A total context failure must never look like a new user's prompt
    (the 2026-07-31 incident). A user with NOTHING gets all three legs
    named — and still gets a persona, because the default fires."""
    from app.db import async_session_maker
    from app.agent.voice_context import build_voice_context
    from app.db.models.user import User

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id, email=f"g19a-bare-{uuid.uuid4().hex[:8]}@example.com",
            hashed_password="x", name="Bare", timezone=TORONTO,
        ))
        await db.commit()

    async with async_session_maker() as db:
        ctx = await build_voice_context(
            db, user_id, tz_name=TORONTO, now_utc=FROZEN_UTC,
        )

    # A brand-new user: every leg queried fine and every leg is blank.
    # `degraded` is for FAILURES, so it must stay clean here.
    assert sorted(ctx.empty) == ["day", "identity", "memories"]
    assert ctx.degraded == [], (
        "a user with no data must not look like a context outage — that "
        "conflation is what let the 2026-07-31 incident hide"
    )
    assert ctx.sections["identity"].startswith(
        "# Core Identity\nYou are the user's personal agent"
    )
    assert ctx.sections["identity_anchor"] == ANCHOR_GOLDEN_VOICE_UNNAMED
    # The channel document is unconditional — never degraded away.
    assert ctx.sections["voice_mode"].startswith("# Voice Conversation Mode")
    assert "onboarding" not in ctx.sections


@pytest.mark.asyncio
async def test_voice_context_honours_the_frozen_clock_and_onboarding(voice_tables):
    """`now_utc` reaches BOTH the day resolver and the clock line — a
    module that read the wall clock for either would fail here."""
    from app.db import async_session_maker
    from app.agent.voice_context import build_voice_context

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed(db, user_id, rows=FOUR_DOCS, agent_name=AGENT_NAME)

    async with async_session_maker() as db:
        ctx = await build_voice_context(
            db, user_id, tz_name=TORONTO, now_utc=FROZEN_UTC, onboarding=True,
        )

    assert "The current date and time is 2026-08-04 02:30 UTC." in ctx.instructions
    assert ctx.day_date == LOCAL_TODAY
    assert ctx.sections["onboarding"].startswith("# ONBOARDING MODE")
    assert ctx.instructions.rstrip().endswith(
        "Do NOT call finalize_onboarding until you have gathered enough info."
    )


@pytest.mark.asyncio
async def test_voice_context_budget_trims_only_the_trimmable_blocks(voice_tables):
    """Identity is never trimmed (it IS the persona); the brains and the
    day transcript are. Same split the relay used: 20/30/50."""
    from app.db import async_session_maker
    from app.db.models.memory import Memory
    from app.agent.voice_context import build_voice_context

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed(db, user_id, rows=FOUR_DOCS, agent_name=AGENT_NAME)
        for i in range(40):
            db.add(Memory(
                id=str(uuid.uuid4()), user_id=user_id, brain_type="user",
                content=f"fact number {i} " + ("x" * 60), category="preferences",
                memory_type="semantic", importance=0.5, confidence=1.0,
                strength=1.0, memory_level="semantic", decay_rate=0.05,
                is_active=True, is_deleted=False, source_type="test",
                created_at=datetime(2026, 8, 3, 12, 0, i),
                updated_at=datetime(2026, 8, 3, 12, 0, i),
            ))
        await db.commit()

    async with async_session_maker() as db:
        big = await build_voice_context(
            db, user_id, tz_name=TORONTO, now_utc=FROZEN_UTC, budget_chars=0)
        small = await build_voice_context(
            db, user_id, tz_name=TORONTO, now_utc=FROZEN_UTC, budget_chars=1000)

    assert "- [context trimmed to budget]" not in big.sections["user_brain"]
    assert "- [context trimmed to budget]" in small.sections["user_brain"]
    assert len(small.sections["user_brain"]) < len(big.sections["user_brain"])
    # Persona survives the budget untouched.
    assert small.sections["identity"] == big.sections["identity"]
    assert small.sections["identity"] == IDENTITY_GOLDEN_FOUR_DOCS


# ──────────────────────────────────────────────────────────────────────
# 4. The internal route's gate — copied from /internal/agent-turn
# ──────────────────────────────────────────────────────────────────────

async def test_voice_context_route_is_gated_exactly_like_agent_turn():
    """Drive the real route. 404 off an agent container, 401 without the
    tenant key — the gate is the only thing between a prober and a user's
    whole persona.

    This used to assert on `inspect.getsource` substrings and never call
    the endpoint, which made it unable to detect the very thing it names.
    An adversarial review proved it by mutation: inserting

        if True:
            return VoiceContextResponse(instructions="LEAKED PERSONA", ...)

    at the TOP of the handler, above every check, left all five gate lines
    intact below it — and the test still passed. A source-shape assertion
    cannot see a bypass, a reordering, or an unmounted router. So it now
    exercises the route.
    """
    import httpx
    from fastapi import FastAPI
    import app.api.api_v1 as api_v1
    from app.config import settings

    # ASGITransport rather than TestClient: starlette's TestClient passes
    # `app=` into httpx.Client, which newer httpx rejects — the failure is
    # a version skew between CI and a dev machine, not a routing problem,
    # and it would fail this test for a reason unrelated to the gate.
    fastapi_app = FastAPI()
    fastapi_app.include_router(api_v1.router)
    transport = httpx.ASGITransport(app=fastapi_app, raise_app_exceptions=False)
    body = {"onboarding": False, "budget_chars": 0, "tz_name": TORONTO}

    async def post(headers=None):
        async with httpx.AsyncClient(
            transport=transport, base_url="http://t"
        ) as c:
            return await c.post("/v1/internal/voice-context", json=body,
                                headers=headers or {})

    orig_mode, orig_key = settings.run_mode, settings.agent_api_key
    try:
        # Off an agent container the route must not exist at all.
        settings.run_mode = "platform"
        settings.agent_api_key = "k"
        assert (await post()).status_code == 404

        settings.run_mode = "agent"
        # No key, wrong key, and a blank configured key all 401 — the last
        # one matters: an empty server-side key must not make every caller
        # match.
        assert (await post()).status_code == 401
        assert (await post({"X-Agent-Key": "wrong"})).status_code == 401
        settings.agent_api_key = ""
        assert (await post({"X-Agent-Key": ""})).status_code == 401
    finally:
        settings.run_mode, settings.agent_api_key = orig_mode, orig_key

    paths = {r.path for r in api_v1.router.routes if hasattr(r, "path")}
    assert "/v1/internal/voice-context" in paths



# ── The timezone the fix depends on, when we do not have it ─────────────
#
# `resolve_local_date` falls back to UTC on an unparseable zone, and
# `resolve_day_chat_id_for_now` only consults `User.timezone` when the
# override is FALSY — never when it is present-but-invalid. At 22:30 in
# Toronto the UTC day is already tomorrow, and tomorrow has no messages, so
# "today by construction" quietly resolved to a session that had forgotten
# the entire day. That is #488 inverted and worse: it looks exactly like a
# user who has not spoken.
#
# Every other test in this file passes an explicit valid tz, which is why
# none of them saw it. These three pin the inputs that broke it.


async def _seed_user_and_today(db, user_id, *, timezone_value):
    from app.db.models import User
    db.add(User(
        id=user_id, email=f"g19a-tz-{uuid.uuid4().hex[:8]}@example.com",
        hashed_password="x", name="TZ", timezone=timezone_value,
    ))
    await db.commit()


async def test_unparseable_timezone_does_not_hide_todays_transcript():
    """An invalid zone must fall back to the newest real day, not to an
    empty UTC tomorrow."""
    from app.db import async_session_maker
    from app.agent.voice_context import build_voice_context

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed(db, user_id, rows=FOUR_DOCS, agent_name=AGENT_NAME)
        # No zone ANYWHERE — clear the one _seed persists, so this exercises
        # the real "we do not know where this user is" path rather than the
        # user-timezone fallback (which the next test covers).
        from sqlalchemy import update as _upd
        from app.db.models import User as _U
        await db.execute(_upd(_U).where(_U.id == user_id).values(timezone=None))
        await db.commit()
        await _seed_day(db, user_id, Date(2026, 8, 3), [
            ("user", "cancel my flight tomorrow", "web", 21),
        ])

    async with async_session_maker() as db:
        ctx = await build_voice_context(
            db, user_id, tz_name="Not/AZone", now_utc=FROZEN_UTC,
        )

    assert "cancel my flight tomorrow" in ctx.instructions, (
        "an unresolvable timezone dropped the day the user actually had — "
        "the session opened having forgotten everything they said"
    )
    assert "day_timezone" in ctx.degraded, (
        "falling back because we could not resolve a zone is a real "
        "degradation and must be visible, not silent"
    )


async def test_no_timezone_anywhere_still_serves_the_day():
    """tz_name=None AND User.timezone NULL — reachable with no client
    misbehaviour at all, since the request field defaults to None."""
    from app.db import async_session_maker
    from app.agent.voice_context import build_voice_context

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed(db, user_id, rows=FOUR_DOCS, agent_name=AGENT_NAME)
        # No zone ANYWHERE — clear the one _seed persists, so this exercises
        # the real "we do not know where this user is" path rather than the
        # user-timezone fallback (which the next test covers).
        from sqlalchemy import update as _upd
        from app.db.models import User as _U
        await db.execute(_upd(_U).where(_U.id == user_id).values(timezone=None))
        await db.commit()
        await _seed_day(db, user_id, Date(2026, 8, 3), [
            ("user", "remind me about the dentist", "web", 21),
        ])

    async with async_session_maker() as db:
        ctx = await build_voice_context(
            db, user_id, tz_name=None, now_utc=FROZEN_UTC,
        )

    assert "remind me about the dentist" in ctx.instructions
    assert "day_timezone" in ctx.degraded


async def test_users_persisted_timezone_is_used_when_the_caller_sends_none():
    """The normal case for a client that does not know the zone: fall back
    to `User.timezone` and keep the today-by-construction guarantee."""
    from app.db import async_session_maker
    from app.agent.voice_context import build_voice_context

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed(db, user_id, rows=FOUR_DOCS, agent_name=AGENT_NAME)
        from sqlalchemy import update as _upd
        from app.db.models import User
        await db.execute(_upd(User).where(User.id == user_id).values(timezone=TORONTO))
        await db.commit()

    async with async_session_maker() as db:
        ctx = await build_voice_context(
            db, user_id, tz_name=None, now_utc=FROZEN_UTC,
        )

    assert ctx.day_date == LOCAL_TODAY, (
        "with a persisted zone the day must still be the user's LOCAL today"
    )
    assert "day_timezone" not in ctx.degraded
