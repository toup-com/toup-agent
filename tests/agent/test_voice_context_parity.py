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
  * #488 the way the live relay fixed it — the newest day chat is
    served, and when it is not the user's local today its header names
    the real date and says so. The earlier draft resolved-and-created
    TODAY instead; that was byte-divergent from the relay, and the W-6
    flip criterion is `ctx_shadow match=True` on real sessions, so the
    day leg now mirrors the relay exactly (header, line format, raw-row
    count, 500-row cap). A previous day narrated as today stays
    impossible in both designs.

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

# The default persona's operating model, composed into the no-soul golden below
# rather than copied into it.
from app.services.soul_compiler import OPERATING_MODEL as _OPERATING_MODEL


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
    # v3: voice's memory section reads FILES, through the same loader and
    # the same renderer text chat uses.
    "memory_files", "memory_file_changes",
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
    "A trusted friend who can also actually get things done.\n"
    # The default persona's operating model, which agent_runner appends to
    # DEFAULT_SOUL_CONTENT from the ONE place it is written. COMPOSED, not
    # re-pinned: a second copy of a 1,130-character prompt block in a test is a
    # second copy to drift, and what this file exists to pin is that voice and
    # chat render the SAME bytes — which composition proves just as well.
    + _OPERATING_MODEL
    + f"\n\n# About the User\n{PROFILE_TEXT}"
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
    # v3: nothing to patch. `hybrid_search` and the user portrait are both
    # off the prompt path — the memory block is `load_brain` over
    # `memory_files`, which these tests leave empty and which costs one
    # query. `user_portrait_service` no longer exists to patch.
    if True:
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

async def _seed_day(db, user_id: str, local_date: Date, msgs, conv_channel: str = "web"):
    """msgs: list of (role, content, channel, hour).

    `conv_channel` is what the day feed actually renders — the endpoint
    (and therefore the assembler) reads the CONVERSATION's channel via the
    join, not the per-message column.
    """
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
        id=conv_id, user_id=user_id, channel=conv_channel, day_chat_id=dc_id,
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
    # No tenant name → the platform default name, NOT the nameless anchor
    # (W-6 fleet run: legacy reads the platform row, which says "Agent").
    assert soulless.sections["identity_anchor"] == ANCHOR_GOLDEN_VOICE_UNNAMED


@pytest.mark.asyncio
async def test_488_previous_day_is_served_with_its_real_date_label(voice_tables):
    """#488, pinned the way the live relay pins it.

    The user has a day chat for YESTERDAY and has said nothing today. The
    pre-guard relay printed yesterday's transcript under "Today's Full
    Conversation History" — that mislabelling IS #488. The shipped guard
    (and now this assembler, byte-identically) serves the newest day but
    labels it with its real date and says nothing has been said today.

    The frozen instant matters: 2026-08-04 02:30 UTC is still
    2026-08-03 in Toronto, so a UTC-based labeller would call the block
    "not today" with the WRONG today (2026-08-04) in the sentence.
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

    assert ctx.day_date == LOCAL_YESTERDAY, (
        "the newest existing day chat is what the relay serves — parity "
        "means serving the same one"
    )
    day = ctx.sections["day_history"]
    assert day.startswith(
        f"# Conversation from {LOCAL_YESTERDAY} — the last day you and the "
        "user spoke (2 messages across all channels). This is NOT today: "
        f"today is {LOCAL_TODAY} and nothing has been said today yet."
    ), "the #488 protection is the truthful label, not an empty prompt"
    assert "book the dentist" in ctx.instructions, (
        "the transcript itself is served — context is kept, only the "
        "mislabelling is gone"
    )
    assert "Today's Full Conversation History" not in ctx.instructions, (
        "a previous day must never carry the today header — that IS #488"
    )
    assert not ctx.degraded


@pytest.mark.asyncio
async def test_488_control_todays_messages_do_load(voice_tables):
    """ANTI-VACUITY CONTROL for the pin above: the same code path with
    messages stamped on TODAY renders them under the today header, in the
    relay's exact line format (`{speaker} [{channel}]: {content}`, channel
    from the CONVERSATION row)."""
    from app.db import async_session_maker
    from app.agent.voice_context import build_voice_context

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed(db, user_id, rows=FOUR_DOCS, agent_name=AGENT_NAME)
        await _seed_day(db, user_id, Date(2026, 8, 3), [
            ("user", "remind me about the flight", "telegram", 9),
            ("assistant", "It leaves at six.", "telegram", 9),
        ], conv_channel="telegram")

    async with async_session_maker() as db:
        ctx = await build_voice_context(
            db, user_id, tz_name=TORONTO, now_utc=FROZEN_UTC,
        )

    assert ctx.day_date == LOCAL_TODAY
    day = ctx.sections["day_history"]
    assert day == (
        "# Today's Full Conversation History (2 messages across all channels)\n"
        "User [telegram]: remind me about the flight\n"
        "You [telegram]: It leaves at six."
    )
    assert "day" not in ctx.degraded


@pytest.mark.asyncio
async def test_voice_context_renders_the_same_memory_files_as_text_chat(voice_tables):
    """D7 CLOSED (memory v3 §3.3). Voice used to dump 200 rows per brain in
    a `- [category] content` shape text chat had not used for a year —
    forgetting this second assembler is a documented prod-incident class,
    and "parity by remembering" is what kept failing. Parity is structural
    now: one loader (`memory_file_ops.load_brain`) and one renderer
    (`memory_files.render_user_brain`), so the two assemblers cannot
    describe the user differently. Only the budget differs."""
    from app.db import async_session_maker
    from app.db.models.memory import MemoryFile
    from app.agent.voice_context import VOICE_SECTION_ORDER, build_voice_context
    from app.memory_files import LEARNED_SLUG, PROFILE_SLUG

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed(db, user_id, rows=FOUR_DOCS, agent_name=AGENT_NAME)
        db.add(MemoryFile(
            user_id=user_id, slug=PROFILE_SLUG, section="you", title="Profile",
            description="Who this person is — setup; read when it matters.",
            body_md="- drinks a flat white, no sugar", is_system=True,
        ))
        db.add(MemoryFile(
            user_id=user_id, slug=LEARNED_SLUG, section="learned", title="Learned",
            description="Working rules — how they want things done; read when acting.",
            body_md="- speak plainly", is_system=True,
        ))
        db.add(MemoryFile(
            user_id=user_id, slug="topics/music", section="topics", title="Music",
            description="Music taste — artists; read when music comes up.",
            body_md="- likes Googoosh and Ebi",
        ))
        await db.commit()

    async with async_session_maker() as db:
        ctx = await build_voice_context(
            db, user_id, tz_name=TORONTO, now_utc=FROZEN_UTC,
        )

    body = ctx.sections["user_brain"]
    assert body.startswith("# User Brain (this user's memory files")
    assert "## Profile\n- drinks a flat white, no sugar" in body
    assert "## Learned (how to work with this user)\n- speak plainly" in body
    # Round 33: the index line carries the slug, on BOTH assemblers — which
    # is why the renderer is shared in the first place.
    assert "## Memory files\n- Music (topics/music) — Music taste — artists" in body
    # The agent brain is the `learned` FILE now — no second section.
    assert "agent_brain" not in ctx.sections
    assert "agent_brain" not in VOICE_SECTION_ORDER
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
    # A bare tenant still wears the platform default name (W-6 fleet run:
    # a NULL tenant agent_name IS the platform's "Agent").
    assert ctx.sections["identity_anchor"] == ANCHOR_GOLDEN_VOICE_UNNAMED
    # The channel document is unconditional — never degraded away.
    assert ctx.sections["voice_mode"].startswith("# Voice Conversation Mode")
    assert "onboarding" not in ctx.sections


@pytest.mark.asyncio
async def test_voice_context_honours_the_frozen_clock_and_onboarding(voice_tables):
    """`now_utc` reaches BOTH the clock line and the day labelling — a
    module that read the wall clock for either would fail here. With no
    day chats at all, `day_date` is None: the relay serves nothing here
    and creates nothing, so parity means we do too."""
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
    assert ctx.day_date is None and "day_history" not in ctx.sections
    assert ctx.sections["onboarding"].startswith("# ONBOARDING MODE")
    assert ctx.instructions.rstrip().endswith(
        "Do NOT call finalize_onboarding until you have gathered enough info."
    )


@pytest.mark.asyncio
async def test_voice_context_budget_trims_only_the_trimmable_blocks(voice_tables):
    """Identity is never trimmed (it IS the persona); the memory block and
    the day transcript are. v3 folds the agent brain into `learned`, so the
    relay's 20/30/50 split becomes 50/50 across one memory section."""
    from app.db import async_session_maker
    from app.db.models.memory import MemoryFile
    from app.agent.voice_context import build_voice_context
    from app.memory_files import PROFILE_SLUG

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed(db, user_id, rows=FOUR_DOCS, agent_name=AGENT_NAME)
        db.add(MemoryFile(
            user_id=user_id, slug=PROFILE_SLUG, section="you", title="Profile",
            description="Who this person is — setup; read when it matters.",
            body_md="\n".join(
                f"- fact number {i} " + ("x" * 60) for i in range(40)
            ),
            is_system=True,
        ))
        await db.commit()

    async with async_session_maker() as db:
        big = await build_voice_context(
            db, user_id, tz_name=TORONTO, now_utc=FROZEN_UTC, budget_chars=0)
        small = await build_voice_context(
            db, user_id, tz_name=TORONTO, now_utc=FROZEN_UTC, budget_chars=1000)

    assert "trimmed to budget" not in big.sections["user_brain"]
    assert "trimmed" in small.sections["user_brain"]
    assert len(small.sections["user_brain"]) < len(big.sections["user_brain"])
    # Persona survives the budget untouched.
    assert small.sections["identity"] == big.sections["identity"]
    assert small.sections["identity"] == IDENTITY_GOLDEN_FOUR_DOCS


# ──────────────────────────────────────────────────────────────────────
# 3b. W-6 flip criterion — byte parity with the relay's day block, and
#     the documented-benign divergence classes
# ──────────────────────────────────────────────────────────────────────


def test_day_header_matches_the_relay_byte_for_byte():
    """Cross-pin: `voice_context.day_history_header` IS the relay's
    `_day_history_header`. Two copies exist only until the legacy builder
    is deleted; until then this test is what holds them together.
    """
    import app.api.ws_realtime as rt
    from app.agent.voice_context import day_history_header

    grid = [
        (0, None, None),
        (2, "2026-08-03", None),
        (2, None, "2026-08-03"),
        (2, "2026-08-03", "2026-08-03"),
        (82, "2026-08-02", "2026-08-03"),
        (500, "2025-12-31", "2026-01-01"),
    ]
    for total, day_date, local_today in grid:
        assert day_history_header(total, day_date, local_today) == \
            rt._day_history_header(total, day_date, local_today), (
                f"header diverged for {(total, day_date, local_today)} — "
                "the shadow will report every session as differs="
            )


@pytest.mark.asyncio
async def test_day_header_counts_raw_rows_like_the_relay(voice_tables):
    """The relay's header counts the RAW endpoint rows; the lines below
    filter to user/assistant turns with non-empty content. A day holding
    a tool row and a blank assistant row therefore says "4 messages" over
    2 rendered lines — on both sides identically, because the comparator
    hashes bytes, not intentions."""
    from app.db import async_session_maker
    from app.agent.voice_context import build_voice_context

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed(db, user_id, rows=FOUR_DOCS, agent_name=AGENT_NAME)
        await _seed_day(db, user_id, Date(2026, 8, 3), [
            ("user", "what's on today?", "web", 9),
            ("tool", '{"result": "noise"}', "web", 9),
            ("assistant", "", "web", 9),
            ("assistant", "Your day is clear.", "web", 9),
        ])

    async with async_session_maker() as db:
        ctx = await build_voice_context(
            db, user_id, tz_name=TORONTO, now_utc=FROZEN_UTC,
        )

    day = ctx.sections["day_history"]
    assert day == (
        "# Today's Full Conversation History (4 messages across all channels)\n"
        "User [web]: what's on today?\n"
        "You [web]: Your day is clear."
    )


@pytest.mark.asyncio
async def test_divergence_class_empty_day_emits_nothing(voice_tables):
    """DOCUMENTED-BENIGN divergence class 1 (W-6 flip justification).

    A day chat exists but holds zero messages. The relay falls back to a
    legacy sessions-based block ("# Today's Conversation History (most
    recent)" — up to 20 stale, 300-char-truncated lines); this assembler
    serves nothing. The shadow reports differs= for such sessions, and
    that is accepted: the stale block is noise the deletion PR removes,
    the persona and brains are unaffected, and reproducing a
    session-table scan just to match dead code would keep the dead code's
    behaviour alive past its deletion. Pinned so the class is a decision,
    not an accident."""
    from app.db import async_session_maker
    from app.agent.voice_context import build_voice_context

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed(db, user_id, rows=FOUR_DOCS, agent_name=AGENT_NAME)
        await _seed_day(db, user_id, Date(2026, 8, 3), [])

    async with async_session_maker() as db:
        ctx = await build_voice_context(
            db, user_id, tz_name=TORONTO, now_utc=FROZEN_UTC,
        )

    assert "day_history" not in ctx.sections
    assert "Today's Conversation History (most recent)" not in ctx.instructions
    assert "day" in ctx.empty and "day" not in ctx.degraded


@pytest.mark.asyncio
async def test_divergence_class_tenant_tz_labels_when_relay_cannot(voice_tables):
    """DOCUMENTED-BENIGN divergence class 2 (W-6 flip justification).

    The relay's date guard reads the PLATFORM `users.timezone`, NULL for
    most accounts, and with no zone it puts the today-header on whatever
    day it serves — #488 alive for exactly those users. This assembler
    falls back to the TENANT copy (the one chat and mobile actually
    write) and labels truthfully. For a platform-NULL/tenant-set user
    whose newest day is not today, the shadow reports the day section as
    differs= — and the agent side is the RIGHT one, so the class is
    accepted rather than "fixed" by reproducing the relay's mislabel.
    The class shrinks as clients send their zone (#566 self-heal fills
    the platform copy)."""
    from app.db import async_session_maker
    from app.agent.voice_context import build_voice_context

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        # _seed writes tenant User.timezone=TORONTO; the relay would pass
        # tz_name=None for a platform-NULL account, which is what we do.
        await _seed(db, user_id, rows=FOUR_DOCS, agent_name=AGENT_NAME)
        await _seed_day(db, user_id, Date(2026, 8, 2), [
            ("user", "wrap up my day", "web", 22),
            ("assistant", "Done — three tasks closed.", "web", 22),
        ])

    async with async_session_maker() as db:
        ctx = await build_voice_context(
            db, user_id, tz_name=None, now_utc=FROZEN_UTC,
        )

    day = ctx.sections["day_history"]
    assert day.startswith(f"# Conversation from {LOCAL_YESTERDAY}"), (
        "the tenant zone must label the stale day truthfully — the "
        "mislabelled alternative is #488"
    )
    assert "day_timezone" not in ctx.degraded, (
        "the tenant zone resolved; degraded would be a false alarm"
    )


@pytest.mark.asyncio
async def test_divergence_class_soulless_default_persona_is_pinned(voice_tables):
    """DOCUMENTED-BENIGN divergence class 3 (W-6 flip justification).

    A user with no identity documents in the tenant DB: this assembler
    emits the runner's default persona (D3 — voice matches text chat);
    the relay emits no identity block at all when other sections exist.
    The shadow reports Core Identity as differs= for such sessions. The
    class is accepted because the default IS the product decision text
    chat already ships, and it EMPTIES after the identity backfill: every
    assigned tenant then carries real rows on both sides. Byte-pinned so
    a drift in the default text shows up here, not in the shadow."""
    from app.db import async_session_maker
    from app.agent.agent_runner import DEFAULT_SOUL_CONTENT
    from app.agent.voice_context import build_voice_context
    from app.db.models.user import User

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id, email=f"g19a-nosoul-{uuid.uuid4().hex[:8]}@example.com",
            hashed_password="x", name="NoSoul", timezone=TORONTO,
        ))
        await db.commit()

    async with async_session_maker() as db:
        ctx = await build_voice_context(
            db, user_id, tz_name=TORONTO, now_utc=FROZEN_UTC,
        )

    assert ctx.sections["identity"] == f"# Core Identity\n{DEFAULT_SOUL_CONTENT}"


# ──────────────────────────────────────────────────────────────────────
# 3c. W-6 fleet parity run (2026-08-12) — the fallback classes. Four
#     divergence classes came out of the 45-tenant field-faithful run;
#     the live evidence (chars/digest) in each docstring is quoted
#     through the shadow's `_section_fingerprints` lens: sections are
#     split on "\n\n# ", so every section EXCEPT the first loses its
#     leading "# " before (len, sha256[:8]) is taken.
# ──────────────────────────────────────────────────────────────────────


def _shadow_fingerprint(block: str) -> tuple:
    """One section through the relay shadow's exact math (ws_realtime
    `_section_fingerprints`): strip, then (len, sha256 hex[:8])."""
    import hashlib

    b = block.strip()
    return len(b), hashlib.sha256(b.encode("utf-8")).hexdigest()[:8]


@pytest.mark.asyncio
async def test_a_nameless_tenant_is_rendered_nameless_not_defaulted(voice_tables):
    """The anchor renders what the TENANT row says, including nothing.

    An earlier version of this fix defaulted an empty tenant name to
    "Agent" so the assembler would match what legacy renders, on the
    premise that the platform holds "Agent" for every never-renamed
    agent. Measured against production that premise is false for 5 of
    45 bound tenants, whose PLATFORM `agent_name` is itself NULL. Four
    of those five already AGREE with legacy today — both sides render
    the nameless anchor — so the default would have broken four
    matching tenants and, after the flip, made their agents announce
    "Your name is Agent" to users who never named one. That is the same
    outcome the agent_name backfill was explicitly forbidden from
    producing in data.

    MUTATION: restore `_tenant_agent_name or DEFAULT_AGENT_NAME` → every
    empty shape renders the NAMED anchor → red.
    """
    from app.db import async_session_maker
    from app.agent.voice_context import build_voice_context
    from app.db.models import AgentConfig
    from app.db.models.user import User

    shapes = {
        "no-agent-config-row": None,
        "empty-string": "",
        "whitespace-only": "   ",
        "real-name-control": AGENT_NAME,
    }
    ids = {shape: str(uuid.uuid4()) for shape in shapes}
    async with async_session_maker() as db:
        for shape, name in shapes.items():
            uid = ids[shape]
            db.add(User(
                id=uid, email=f"g19a-name-{uuid.uuid4().hex[:8]}@example.com",
                hashed_password="x", name="W6", timezone=TORONTO,
            ))
            if name is not None:
                db.add(AgentConfig(user_id=uid, agent_name=name))
        await db.commit()

    anchors = {}
    async with async_session_maker() as db:
        for shape, uid in ids.items():
            ctx = await build_voice_context(
                db, uid, tz_name=TORONTO, now_utc=FROZEN_UTC,
            )
            anchors[shape] = ctx.sections["identity_anchor"]

    for shape in ("no-agent-config-row", "empty-string", "whitespace-only"):
        assert anchors[shape] == ANCHOR_GOLDEN_VOICE_UNNAMED, (
            f"a tenant with {shape} must render the NAMELESS anchor — "
            "claiming a name the user never chose is worse than "
            "diverging from a legacy builder that does"
        )
    assert anchors["real-name-control"] == ANCHOR_GOLDEN_VOICE_NAMED


@pytest.mark.asyncio
async def test_divergence_class_duplicate_soul_rows_render_core_identity_once(voice_tables):
    """W-6 live divergence, duplicate-soul class (tenant 03cbc72f): a
    double-write race (two concurrent soul syncs, each finding no
    existing row) left TWO identical active soul rows 15ms apart in the
    tenant DB, and the assembler emitted "# Core Identity" once per row.
    The legacy builder reads the PLATFORM copy, which the upsert keeps
    single (save_soul / sync_soul: scalar_one_or_none + update-in-place),
    so it renders the section ONCE. The assembler keeps the row that
    upsert would own — highest priority, then OLDEST created_at — and
    drops the clones; with drifted contents the oldest row (the one
    later platform upserts update in place) still wins,
    deterministically."""
    from app.db import async_session_maker
    from app.agent.voice_context import build_voice_context
    from app.db.models import Identity
    from app.db.models.user import User

    def _soul_row(uid, content, ts, name="Agent Soul"):
        return Identity(
            id=str(uuid.uuid4()), user_id=uid, identity_type="soul",
            name=name, content=content, priority=100, is_active=True,
            created_at=ts, updated_at=ts,
        )

    twin_id = str(uuid.uuid4())
    drift_id = str(uuid.uuid4())
    t0 = datetime(2026, 8, 3, 12, 0, 0)
    t1 = datetime(2026, 8, 3, 12, 0, 0, 15000)  # 15ms later — the race
    async with async_session_maker() as db:
        for uid in (twin_id, drift_id):
            db.add(User(
                id=uid, email=f"g19a-dup-{uuid.uuid4().hex[:8]}@example.com",
                hashed_password="x", name="Dup", timezone=TORONTO,
            ))
            db.add(Identity(
                id=str(uuid.uuid4()), user_id=uid, identity_type="user_profile",
                name="user_profile", content=PROFILE_TEXT, priority=90,
                is_active=True, created_at=t0, updated_at=t0,
            ))
        db.add(_soul_row(twin_id, SOUL_TEXT, t0))
        db.add(_soul_row(twin_id, SOUL_TEXT, t1))
        db.add(_soul_row(drift_id, SOUL_TEXT, t0))
        db.add(_soul_row(drift_id, "A drifted clone.", t1))
        await db.commit()

    async with async_session_maker() as db:
        twins = await build_voice_context(
            db, twin_id, tz_name=TORONTO, now_utc=FROZEN_UTC)
        drifted = await build_voice_context(
            db, drift_id, tz_name=TORONTO, now_utc=FROZEN_UTC)

    for ctx in (twins, drifted):
        assert ctx.sections["identity"].count("# Core Identity") == 1, (
            "duplicate active soul rows must render ONE Core Identity "
            "section — legacy's platform copy is upsert-single"
        )
    assert twins.sections["identity"] == (
        f"# Core Identity\n{SOUL_TEXT}\n\n# About the User\n{PROFILE_TEXT}"
    )
    assert drifted.sections["identity"] == (
        f"# Core Identity\n{SOUL_TEXT}\n\n# About the User\n{PROFILE_TEXT}"
    ), "the OLDEST soul row is the one the platform upsert owns"


@pytest.mark.asyncio
async def test_seeded_platform_defaults_render_byte_identical_to_legacy(voice_tables):
    """W-6 live divergences "Behavioral Guidelines" (tenants a8176b1d,
    fc1c53c3 — legacy 1242 chars / 847d31db) and "Core Identity"
    (a8176b1d — legacy 1158 / c93d5d3b): DIAGNOSIS pin. Legacy has no
    template fallback for either section — ws_realtime renders whatever
    active Identity rows its DB holds; those "defaults" are the
    `_seed_default_identities` rows every signup puts in the PLATFORM DB,
    never delivered to the tenant (the identities split;
    scripts/backfill_tenant_identities.py copies them verbatim). Fed the
    SAME rows — the real seeder — the assembler's bytes ARE the live
    legacy fingerprints, so the verbatim backfill restores parity by
    construction and no code fallback is needed (or safe: see the
    founder test below)."""
    from sqlalchemy import select

    from app.db import async_session_maker
    from app.agent.voice_context import build_voice_context
    from app.db.models import Identity
    from app.db.models.user import User
    from app.services.auth_service import _seed_default_identities

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id, email=f"g19a-seed-{uuid.uuid4().hex[:8]}@example.com",
            hashed_password="x", name="Seeded", timezone=TORONTO,
        ))
        await db.flush()
        await _seed_default_identities(db, user_id)
        await db.commit()

    async with async_session_maker() as db:
        rows = (await db.execute(
            select(Identity).where(Identity.user_id == user_id)
        )).scalars().all()
        by_type = {r.identity_type: r.content for r in rows}
        ctx = await build_voice_context(
            db, user_id, tz_name=TORONTO, now_utc=FROZEN_UTC,
        )

    soul, instr = by_type["soul"], by_type["agent_instructions"]
    assert ctx.sections["identity"] == (
        f"# Core Identity\n{soul}\n\n# Behavioral Guidelines\n{instr}"
    ), "same rows in, legacy's bytes out — the divergence is the DATA"

    # The live run's evidence digests: Core Identity leads the prompt
    # (first section keeps its "# "); Behavioral Guidelines is mid-text.
    assert _shadow_fingerprint(f"# Core Identity\n{soul}") == (1158, "c93d5d3b")
    assert _shadow_fingerprint(f"Behavioral Guidelines\n{instr}") == (1242, "847d31db")


@pytest.mark.asyncio
async def test_no_tenant_instructions_row_emits_no_behavioral_guidelines(voice_tables):
    """The founder's condition, mirrored exactly: legacy emits Behavioral
    Guidelines IFF an active agent_instructions row exists in the DB it
    reads — the founder has none platform-side and matched at the
    closeout with NO section on either side. So: no tenant row → no
    section. A code-level default (the tempting "fix" for the class
    above) would emit one here and flip that matching session to a
    mismatch."""
    from app.db import async_session_maker
    from app.agent.voice_context import build_voice_context

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed(db, user_id, rows=[("soul", SOUL_TEXT, 100)],
                    agent_name=AGENT_NAME)

    async with async_session_maker() as db:
        ctx = await build_voice_context(
            db, user_id, tz_name=TORONTO, now_utc=FROZEN_UTC,
        )

    assert "# Behavioral Guidelines" not in ctx.instructions
    assert ctx.sections["identity"] == f"# Core Identity\n{SOUL_TEXT}"


def test_flip_allowlist_serves_agent_context_per_user():
    """`voice_context_from_agent` is a platform-process GLOBAL; the
    allowlist is what makes canary→founder→fleet a real sequence instead
    of a fleet flip with a reassuring name."""
    from unittest.mock import patch as _patch

    from app.api.ws_realtime import _agent_ctx_enabled_for
    from app.config import settings

    canary = "533354ce-0000-0000-0000-000000000000"
    other = "871bac24-0000-0000-0000-000000000000"

    with _patch.object(settings, "voice_context_from_agent", False), \
         _patch.object(settings, "voice_context_from_agent_user_ids", f" {canary} , "):
        assert _agent_ctx_enabled_for(canary) is True
        assert _agent_ctx_enabled_for(other) is False

    with _patch.object(settings, "voice_context_from_agent", False), \
         _patch.object(settings, "voice_context_from_agent_user_ids", ""):
        assert _agent_ctx_enabled_for(canary) is False

    with _patch.object(settings, "voice_context_from_agent", True), \
         _patch.object(settings, "voice_context_from_agent_user_ids", ""):
        assert _agent_ctx_enabled_for(other) is True


def test_internal_route_request_accepts_the_relay_clock():
    """The `now` field parses an ISO instant (the relay sends
    `.isoformat()` of an aware UTC datetime). A schema that rejected it
    would silently un-align the two builders' clock lines again."""
    from app.api.api_v1 import VoiceContextRequest

    req = VoiceContextRequest(
        onboarding=False, budget_chars=0, tz_name=TORONTO,
        now="2026-08-04T02:30:00+00:00",
    )
    assert req.now == FROZEN_UTC
    assert VoiceContextRequest().now is None


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
    to `User.timezone`, and let it drive the date labelling — today's day
    chat renders under the today header with no false degradation."""
    from app.db import async_session_maker
    from app.agent.voice_context import build_voice_context

    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        await _seed(db, user_id, rows=FOUR_DOCS, agent_name=AGENT_NAME)
        from sqlalchemy import update as _upd
        from app.db.models import User
        await db.execute(_upd(User).where(User.id == user_id).values(timezone=TORONTO))
        await db.commit()
        await _seed_day(db, user_id, Date(2026, 8, 3), [
            ("user", "morning — what's first?", "web", 8),
        ])

    async with async_session_maker() as db:
        ctx = await build_voice_context(
            db, user_id, tz_name=None, now_utc=FROZEN_UTC,
        )

    assert ctx.day_date == LOCAL_TODAY
    assert ctx.sections["day_history"].startswith(
        "# Today's Full Conversation History (1 messages across all channels)"
    ), "the persisted zone recognises today's chat AS today"
    assert "day_timezone" not in ctx.degraded
