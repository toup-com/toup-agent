"""The job card the WebSocket has never once managed to write.

``ws_chat._broadcast_reader`` persisted every ``job_update`` frame's card as
``Message(id="job-<jid>", conversation_id="build-<jid[:8]>")``. Nothing in
the repo has ever inserted a ``build-%`` row into ``conversations``, and
``messages.conversation_id`` is NOT NULL onto ``conversations.id`` — so the
INSERT violated ``messages_conversation_id_fkey`` on every attempt from
2026-04-13 onwards. Confirmed live on pool-82 at 2026-09-16 13:36:49Z:
``Key (conversation_id)=(build-bd272753) is not present in table
"conversations"``. Because ``_persisted_job_ids.add`` ran BEFORE the try,
the socket never re-tried: the card was lost for the life of the
connection. The cards actually lost are chat-originated ``agent_task``
jobs — app_builder cards are re-written at turn end against a real
``response.session_id``, which is why nobody noticed for five months.

Nothing here is mocked: the defect IS the value that reaches the database,
so the tests drive the real ``_persist_job_card`` and the real
``_write_job_card`` against the suite's own DB. ``conversations`` /
``messages`` / ``day_chats`` / ``build_jobs`` are all AGENT_ONLY, so this
runs in the agent lane (COVERAGE_DEBT.txt) and the DB-driven tests SKIP
rather than error if they are ever collected under RUN_MODE=platform —
the same `requires_*_tables` shape test_message_client_msg_id.py uses.

Local run (from backend/):
    RUN_MODE=agent PYTHONPATH=. pytest tests/test_job_card_persistence.py \
      -q -p no:cacheprovider

Control (verified 2026-09-18): restore the pre-change writer (the inline
block with ``conversation_id=f"build-{_jid[:8]}"``) and
``test_card_lands_on_the_turns_real_conversation`` fails with
``conversation_id == 'build-<8 hex>'``, and
``test_a_failed_write_is_retried_on_a_later_frame`` fails because the id is
latched before the attempt. SQLite does not enforce the FK, so the wrong
value LANDS instead of raising — the assertion is on the value, which is
what Postgres rejects.
"""
from __future__ import annotations

import logging
import sys
import uuid
from datetime import date as Date, datetime, timezone as _tzutc
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# asyncio_mode = auto (pytest.ini) — the async tests below need no mark.

WS_LOGGER = "app.api.ws_chat"

#: 25 hours apart, so their local calendar dates ALWAYS differ — which is
#: what makes the midnight-boundary test deterministic on any clock.
TZ_AHEAD = "Pacific/Kiritimati"   # UTC+14
TZ_BEHIND = "Pacific/Niue"        # UTC-11


@pytest_asyncio.fixture
async def requires_job_card_tables():
    """Skip unless the AGENT_ONLY tables this file writes to exist."""
    from sqlalchemy import inspect as _inspect

    from app.db.database import engine

    async with engine.connect() as conn:
        present = await conn.run_sync(
            lambda sc: {n: _inspect(sc).has_table(n) for n in
                        ("conversations", "messages", "day_chats", "build_jobs")}
        )
    missing = [n for n, ok in present.items() if not ok]
    if missing:
        pytest.skip(
            f"requires AGENT_ONLY table(s) {missing} — run with RUN_MODE=agent"
        )


async def _seed(
    *,
    with_conversation: bool = True,
    job_conv: bool = False,
    job_type: str = "agent_task",
    user_tz: str | None = None,
):
    """A user, a day, a chat-originated job row, and (optionally) the
    conversation the job's turn belongs to."""
    from app.db.database import async_session_maker
    from app.db.models import BuildJob, Conversation, User
    from app.db.models.day_chat import DayChat

    user_id = str(uuid.uuid4())
    conv_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    day = Date.today()
    base = datetime(day.year, day.month, day.day, 10, 0, 0)

    async with async_session_maker() as db:
        db.add(User(id=user_id, email=f"{user_id}@t.local",
                    hashed_password="x", name="T", timezone=user_tz))
        db.add(DayChat(id=conv_id[:8] + "-dc", user_id=user_id, local_date=day,
                       timezone=user_tz or "UTC", started_at=base,
                       last_message_at=base))
        if with_conversation:
            db.add(Conversation(id=conv_id, user_id=user_id, channel="app",
                                started_at=base, updated_at=base))
        db.add(BuildJob(
            id=job_id, user_id=user_id, title="Check the flight prices",
            prompt="Check the flight prices", status="running", layer=0,
            job_type=job_type, source_kind="manual",
            conversation_id=conv_id if job_conv else None,
        ))
        await db.commit()
    return user_id, conv_id, job_id


async def _add_conversation(user_id: str, conv_id: str):
    from app.db.database import async_session_maker
    from app.db.models import Conversation
    async with async_session_maker() as db:
        db.add(Conversation(id=conv_id, user_id=user_id, channel="app",
                            started_at=datetime.utcnow(),
                            updated_at=datetime.utcnow()))
        await db.commit()


def _frame(job_id: str, chat_id: str | None = None, status: str = "running") -> dict:
    """A ``job_update`` frame exactly as ``_tool_create_job`` broadcasts it."""
    ev = {
        "type": "job_update",
        "job_id": job_id,
        "job_type": "research",
        "name": "Check the flight prices",
        "status": status,
        "step": "Working...",
        "total_steps": 3,
        "completed_steps": 0,
        "message_id": None,
    }
    if chat_id is not None:
        ev["chat_id"] = chat_id
    return ev


async def _card(job_id: str):
    from app.db.database import async_session_maker
    from app.db.models import Message
    async with async_session_maker() as db:
        return await db.get(Message, f"job-{job_id}")


async def _day_of(card) -> Date | None:
    from app.db.database import async_session_maker
    from app.db.models.day_chat import DayChat
    async with async_session_maker() as db:
        row = await db.get(DayChat, card.day_chat_id)
    return row.local_date if row else None


# ══════════════════════════════════════════════════════════════════════
# The write itself
# ══════════════════════════════════════════════════════════════════════

async def test_card_lands_on_the_turns_real_conversation(requires_job_card_tables):
    """THE regression. The frame carries the turn's conversation id; the
    row must point at it, not at a synthesised `build-*` value."""
    from app.api.ws_chat import _JOB_CARD_WRITTEN, _persist_job_card

    user_id, conv_id, job_id = await _seed()

    assert await _persist_job_card(user_id, _frame(job_id, conv_id)) == _JOB_CARD_WRITTEN

    row = await _card(job_id)
    assert row is not None, "the card was not written at all"
    assert row.conversation_id == conv_id
    assert not row.conversation_id.startswith("build-")
    assert row.role == "job"
    assert row.day_chat_id, "a card with no day is invisible to the day reader"


async def test_card_is_readable_through_the_day_chat_join(requires_job_card_tables):
    """The FK is not the only reason the id has to be real: every day-chat
    reader INNER JOINs `messages` -> `conversations` (api/day_chats.py), so
    a row pointing at a non-existent conversation is unreadable even where
    the database accepts it."""
    from sqlalchemy import select

    from app.api.ws_chat import _persist_job_card
    from app.db.database import async_session_maker
    from app.db.models import Conversation, Message

    user_id, conv_id, job_id = await _seed()
    await _persist_job_card(user_id, _frame(job_id, conv_id))

    async with async_session_maker() as db:
        found = (await db.execute(
            select(Message.id)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(Message.id == f"job-{job_id}")
        )).scalar_one_or_none()
    assert found == f"job-{job_id}"


async def test_a_frame_without_chat_id_falls_back_to_the_job_row(requires_job_card_tables):
    """The reaper, the reconciler's cancel arm and the sub-agent
    orchestrator broadcast `job_update` with no `chat_id`. The job's own
    back-link is a real conversation id and is the right answer."""
    from app.api.ws_chat import _JOB_CARD_WRITTEN, _persist_job_card

    user_id, conv_id, job_id = await _seed(job_conv=True)

    assert await _persist_job_card(user_id, _frame(job_id)) == _JOB_CARD_WRITTEN
    row = await _card(job_id)
    assert row is not None and row.conversation_id == conv_id


async def test_an_unusable_chat_id_still_falls_back_to_the_job_row(requires_job_card_tables):
    """The fallback was gated on `if not _conv_id`, so a `chat_id` that was
    PRESENT but pointed at no conversation returned "nothing to point at"
    without ever consulting the job row — dead for exactly the frames that
    could need it. agent_runner mints `subagent:<hex>` session ids with no
    Conversation row, so that shape is real."""
    from app.api.ws_chat import _JOB_CARD_WRITTEN, _persist_job_card

    user_id, conv_id, job_id = await _seed(job_conv=True)

    sentinel = f"subagent:{uuid.uuid4().hex[:20]}"
    assert await _persist_job_card(user_id, _frame(job_id, sentinel)) == _JOB_CARD_WRITTEN
    row = await _card(job_id)
    assert row is not None and row.conversation_id == conv_id


async def test_no_conversation_anywhere_writes_nothing(requires_job_card_tables):
    """With neither a frame `chat_id` nor a job back-link there is no real
    conversation to point at, and inventing one is the defect. Skip."""
    from app.api.ws_chat import _JOB_CARD_NO_TARGET, _persist_job_card

    user_id, _conv_id, job_id = await _seed(job_conv=False)

    assert await _persist_job_card(user_id, _frame(job_id)) == _JOB_CARD_NO_TARGET
    assert await _card(job_id) is None


async def test_another_users_conversation_is_refused(requires_job_card_tables):
    """The conversation id arrives on a broadcast frame. Stamping a card
    into someone else's thread must be impossible even if a producer ever
    passes the wrong id — the same (id, user_id) pair the pre-save
    checks."""
    from app.api.ws_chat import _JOB_CARD_NO_TARGET, _persist_job_card

    _other_id, other_conv, _other_job = await _seed()
    user_id, _conv_id, job_id = await _seed(with_conversation=False)

    assert await _persist_job_card(
        user_id, _frame(job_id, other_conv)
    ) == _JOB_CARD_NO_TARGET
    assert await _card(job_id) is None


async def test_an_existing_card_is_left_alone(requires_job_card_tables):
    """Idempotent: `job-<id>` is a fixed row id shared with the turn-end
    `_pending_job_cards` writer, and every frame of a job re-enters here,
    so whichever writer arrives second must not double-write."""
    from app.api.ws_chat import _JOB_CARD_WRITTEN, _persist_job_card

    user_id, conv_id, job_id = await _seed()

    assert await _persist_job_card(user_id, _frame(job_id, conv_id)) == _JOB_CARD_WRITTEN
    first = await _card(job_id)
    assert await _persist_job_card(user_id, _frame(job_id, conv_id)) == _JOB_CARD_WRITTEN
    second = await _card(job_id)
    assert first.created_at == second.created_at
    assert second.conversation_id == conv_id


async def test_the_marker_is_the_only_body_and_carries_the_job_type(requires_job_card_tables):
    """The row is a POINTER (api/message_cards.py). Keep the marker shape
    the three producers share, job_type included."""
    from app.api.message_cards import parse_job_marker
    from app.api.ws_chat import _persist_job_card

    user_id, conv_id, job_id = await _seed()
    await _persist_job_card(user_id, _frame(job_id, conv_id))

    marker = parse_job_marker((await _card(job_id)).content)
    assert marker.get("job_id") == job_id
    assert marker.get("job_name") == "Check the flight prices"
    assert marker.get("job_type") == "research"


# ══════════════════════════════════════════════════════════════════════
# The local day (R2)
# ══════════════════════════════════════════════════════════════════════

async def test_the_card_lands_on_the_clients_local_day(requires_job_card_tables):
    """The card belongs to the client's local day, like every other row of
    the turn (the sibling `_pending_job_cards` writer passes the same
    `tz_override=client_tz`). Without the override the helper falls back to
    `users.timezone`, so the user row here carries the OTHER zone — 25
    hours away, so the two dates can never coincide on any clock."""
    from app.api.ws_chat import _persist_job_card

    user_id, conv_id, job_id = await _seed(user_tz=TZ_BEHIND)

    await _persist_job_card(user_id, _frame(job_id, conv_id), tz=TZ_AHEAD)

    now = datetime.now(_tzutc.utc)
    expected = now.astimezone(ZoneInfo(TZ_AHEAD)).date()
    fallback = now.astimezone(ZoneInfo(TZ_BEHIND)).date()
    assert expected != fallback, "the two fixture zones must straddle midnight"

    got = await _day_of(await _card(job_id))
    assert got == expected, f"card filed under {got}, client's day is {expected}"


async def test_the_reader_hands_the_client_zone_to_the_writer():
    """The caller's half of the same rule, and the only half no behavioural
    test can reach: `_broadcast_reader` is a closure inside the WebSocket
    handler. It must read the message loop's `client_tz` and pass it."""
    reader = _reader_source()
    # The exact assignment, not the word: a version that reads the zone and
    # throws it away (`_tz = None  # client_tz`) satisfies a substring
    # check for "client_tz" while filing every card under `users.timezone`,
    # which is the defect this rule exists to prevent.
    assert "_tz = client_tz" in reader, "the reader no longer reads the client's zone"
    assert "tz=_tz" in reader, "the zone is read but not passed to the writer"


# ══════════════════════════════════════════════════════════════════════
# The retry budget and the log levels (R3, R5)
#
# These drive the REAL `_write_job_card`, which is why it is at module
# level: the bookkeeping used to live in `_broadcast_reader`'s closure,
# where its ORDER — the whole defect — was reachable only by grepping.
# ══════════════════════════════════════════════════════════════════════

async def test_a_failed_write_is_retried_on_a_later_frame(requires_job_card_tables):
    """`_persisted_job_ids.add` ran before the try, so one unwritable frame
    burned the job for the whole socket. Nothing may be latched for a frame
    that could not be written, and the next frame that CAN be written must
    still land the card."""
    from app.api.ws_chat import (
        _JOB_CARD_NO_TARGET, _JOB_CARD_WRITTEN, _JobCardBudget, _write_job_card,
    )

    # No conversation row yet: the frame has outrun its own turn's commit.
    user_id, conv_id, job_id = await _seed(with_conversation=False)
    budget = _JobCardBudget()

    assert await _write_job_card(user_id, _frame(job_id, conv_id), budget) == _JOB_CARD_NO_TARGET
    assert await _card(job_id) is None, "a card was written against a missing conversation"
    assert job_id not in budget.closed, "the job was retired before it was ever written"

    await _add_conversation(user_id, conv_id)

    assert await _write_job_card(user_id, _frame(job_id, conv_id), budget) == _JOB_CARD_WRITTEN
    row = await _card(job_id)
    assert row is not None and row.conversation_id == conv_id


async def test_frames_with_nothing_to_point_at_do_not_burn_the_budget(requires_job_card_tables):
    """The old budget was 3 tries charged for every outcome, so the in-turn
    frames spent it while the conversation was genuinely not committed yet
    and the card was dropped although a later frame would have landed it."""
    from app.api.ws_chat import (
        _JOB_CARD_MAX_TRIES, _JOB_CARD_NO_TARGET, _JOB_CARD_WRITTEN,
        _JobCardBudget, _write_job_card,
    )

    user_id, conv_id, job_id = await _seed(with_conversation=False)
    budget = _JobCardBudget()

    for _ in range(_JOB_CARD_MAX_TRIES - 1):
        assert await _write_job_card(
            user_id, _frame(job_id, conv_id), budget
        ) == _JOB_CARD_NO_TARGET
    assert await _card(job_id) is None

    await _add_conversation(user_id, conv_id)
    assert await _write_job_card(
        user_id, _frame(job_id, conv_id), budget
    ) == _JOB_CARD_WRITTEN
    assert (await _card(job_id)).conversation_id == conv_id
    # Persisted ONCE: nothing further is even attempted.
    assert await _write_job_card(user_id, _frame(job_id, conv_id), budget) is None


async def test_a_terminal_frame_always_gets_an_attempt(requires_job_card_tables):
    """A job emits one frame per step, so the ordinary budget can be gone
    before the conversation commits — and the LAST frame of a job is the
    one most likely to find it there. `completed` / `failed` / `cancelled`
    come from app.agent.job_status, never a literal list here."""
    from app.agent.job_status import TERMINAL_STATUSES
    from app.api.ws_chat import (
        _JOB_CARD_MAX_TRIES, _JOB_CARD_WRITTEN, _JobCardBudget, _write_job_card,
    )

    assert {"completed", "failed", "cancelled"} <= set(TERMINAL_STATUSES)

    user_id, conv_id, job_id = await _seed(with_conversation=False)
    budget = _JobCardBudget()

    for _ in range(_JOB_CARD_MAX_TRIES):
        await _write_job_card(user_id, _frame(job_id, conv_id), budget)

    # Ordinary budget gone: a running frame is not even attempted.
    assert await _write_job_card(user_id, _frame(job_id, conv_id), budget) is None

    await _add_conversation(user_id, conv_id)
    assert await _write_job_card(
        user_id, _frame(job_id, conv_id, status="completed"), budget
    ) == _JOB_CARD_WRITTEN
    assert (await _card(job_id)).conversation_id == conv_id


async def test_a_conversation_that_never_appears_gives_up_once(
    requires_job_card_tables, caplog,
):
    """Bounded: a long-lived socket must not re-SELECT on every frame of a
    job whose conversation never shows up. And exactly one line when it is
    abandoned — a card dropped in silence is how this writer stayed broken
    for five months."""
    from app.api.ws_chat import _JOB_CARD_MAX_TRIES, _JobCardBudget, _write_job_card

    user_id, _conv_id, job_id = await _seed(with_conversation=False)
    budget = _JobCardBudget()

    with caplog.at_level(logging.DEBUG, logger=WS_LOGGER):
        outcomes = [
            await _write_job_card(
                user_id, _frame(job_id, status="completed"), budget,
            )
            for _ in range(_JOB_CARD_MAX_TRIES * 4)
        ]

    attempted = [o for o in outcomes if o is not None]
    assert len(attempted) == _JOB_CARD_MAX_TRIES + 1, (
        f"{len(attempted)} attempts — the terminal frame's one guaranteed "
        f"attempt beyond the budget, and no more"
    )
    warnings = [r for r in caplog.records
                if r.name == WS_LOGGER and r.levelno >= logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]
    assert "dropped" in warnings[0].getMessage()
    assert job_id[:8] in warnings[0].getMessage()
    assert job_id not in warnings[0].getMessage(), "full job id in a log line"


async def test_a_card_written_elsewhere_is_debug_not_warning(
    requires_job_card_tables, caplog,
):
    """An `auto_builder` row that names NO conversation has nothing to
    point at, ever: that is app_builder's build/modify class, whose cards
    are written at turn end. Warning about them made every healthy app
    build announce a card it had not lost — in a round whose subject is log
    noise."""
    from app.api.ws_chat import _JOB_CARD_NOT_OURS, _JobCardBudget, _write_job_card

    user_id, _conv_id, job_id = await _seed(
        with_conversation=False, job_type="auto_builder",
    )
    budget = _JobCardBudget()

    with caplog.at_level(logging.DEBUG, logger=WS_LOGGER):
        assert await _write_job_card(
            user_id, _frame(job_id), budget,
        ) == _JOB_CARD_NOT_OURS
        # Permanent answer: never asked again, so never logged again.
        assert await _write_job_card(user_id, _frame(job_id), budget) is None

    mine = [r for r in caplog.records if r.name == WS_LOGGER]
    assert [r.levelno for r in mine] == [logging.DEBUG], [
        (r.levelname, r.getMessage()) for r in mine
    ]
    assert "turn end" in mine[0].getMessage()


async def test_an_auto_builder_with_a_back_link_is_retried_not_closed(
    requires_job_card_tables,
):
    """The other shape of the same row. `not_ours` was decided on
    `job_type == "auto_builder"` alone, so an auto_builder job that DOES
    name a conversation was closed permanently on its first frame whenever
    that conversation was merely not visible yet — the exact retryable case
    the budget exists for, and no other writer covers it (the turn-end
    `_pending_job_cards` block is fed only by `app_builder__build_app`
    summaries). The class is real: app_html's `ensure_job` creates
    `job_type="auto_builder"` with `conversation_id=turn_deep_link()[0]`,
    and an adopted job keeps the `create_job` tool's id while its type
    flips to auto_builder."""
    from app.api.ws_chat import (
        _JOB_CARD_NO_TARGET, _JOB_CARD_WRITTEN, _JobCardBudget, _write_job_card,
    )

    user_id, conv_id, job_id = await _seed(
        with_conversation=False, job_conv=True, job_type="auto_builder",
    )
    budget = _JobCardBudget()

    assert await _write_job_card(
        user_id, _frame(job_id), budget,
    ) == _JOB_CARD_NO_TARGET
    assert job_id not in budget.closed, "a back-linked job was retired on frame 1"
    assert await _card(job_id) is None

    # The conversation commits later in the same turn, as the deferred
    # pre-save paths do, and the next frame still lands the card.
    await _add_conversation(user_id, conv_id)
    assert await _write_job_card(
        user_id, _frame(job_id), budget,
    ) == _JOB_CARD_WRITTEN
    assert (await _card(job_id)).conversation_id == conv_id


async def test_an_auto_builder_naming_no_conversation_is_closed_at_once(
    requires_job_card_tables,
):
    """And the shape that IS permanent: an `auto_builder` row with no
    conversation_id at all has nothing to point at, ever — app_builder's
    build (TaskSpec sets none) and modify (bare insert) jobs, whose cards
    the turn-end writer owns. One answer, and the budget is never spent on
    it."""
    from app.api.ws_chat import (
        _JOB_CARD_NOT_OURS, _JobCardBudget, _persist_job_card, _write_job_card,
    )

    user_id, _conv_id, job_id = await _seed(
        with_conversation=False, job_conv=False, job_type="auto_builder",
    )

    assert await _persist_job_card(user_id, _frame(job_id)) == _JOB_CARD_NOT_OURS

    budget = _JobCardBudget()
    assert await _write_job_card(
        user_id, _frame(job_id), budget,
    ) == _JOB_CARD_NOT_OURS
    assert job_id in budget.closed
    assert budget.tries.get(job_id, 0) == 0, "a permanent answer cost a try"


async def test_a_write_that_raises_warns_with_the_type_only(
    requires_job_card_tables, caplog, monkeypatch,
):
    """WARNING is reserved for this branch. And the exception BODY never
    reaches the log: a SQLAlchemy error stringifies its bound parameters,
    and this row's parameters include the job marker — the user's own
    words (the leak api/message_cards.py documents)."""
    from app.api import ws_chat as _ws
    from app.api.ws_chat import (
        _JOB_CARD_ERROR, _JOB_CARD_MAX_ERRORS, _JobCardBudget, _write_job_card,
    )

    user_id, conv_id, job_id = await _seed()
    budget = _JobCardBudget()
    secret = "book me a flight to Tehran"

    async def _boom(*_a, **_kw):
        raise ValueError(f"(psycopg) INSERT ... parameters: {secret}")

    monkeypatch.setattr(_ws, "_persist_job_card", _boom)

    with caplog.at_level(logging.DEBUG, logger=WS_LOGGER):
        for _ in range(_JOB_CARD_MAX_ERRORS):
            assert await _write_job_card(
                user_id, _frame(job_id, conv_id), budget,
            ) == _JOB_CARD_ERROR
        # Bounded: the error budget is its own, and smaller.
        assert await _write_job_card(user_id, _frame(job_id, conv_id), budget) is None

    warnings = [r for r in caplog.records
                if r.name == WS_LOGGER and r.levelno >= logging.WARNING]
    assert len(warnings) == _JOB_CARD_MAX_ERRORS
    for rec in warnings:
        msg = rec.getMessage()
        assert "ValueError" in msg
        assert secret not in msg
        assert "parameters" not in msg


async def test_the_writer_never_raises_into_the_broadcast_reader(
    requires_job_card_tables, monkeypatch,
):
    """`_broadcast_reader` forwards EVERY event to this socket. An
    exception escaping the job-card write would kill that task, so a lost
    card would cost the user their live chat."""
    from app.api import ws_chat as _ws
    from app.api.ws_chat import _JOB_CARD_ERROR, _JobCardBudget, _write_job_card

    user_id, conv_id, job_id = await _seed()

    class _Hostile:
        """Every seam the writer touches before the try used to be
        unguarded — a budget that raises must be survivable too."""

        closed: set = set()
        tries: dict = {}
        final_try: set = set()
        errors: dict = {}

        def may_try(self, *_a, **_kw):
            raise RuntimeError("bookkeeping exploded")

    assert await _write_job_card(
        user_id, _frame(job_id, conv_id), _Hostile(),
    ) == _JOB_CARD_ERROR

    async def _boom(*_a, **_kw):
        raise RuntimeError("the write exploded")

    monkeypatch.setattr(_ws, "_persist_job_card", _boom)
    assert await _write_job_card(
        user_id, _frame(job_id, conv_id), _JobCardBudget(),
    ) == _JOB_CARD_ERROR


# ══════════════════════════════════════════════════════════════════════
# Source probes — the shipped defect's exact shapes
#
# CLAUDE.md: read new guards for ORDER, not just for logic. The latch
# order is now covered behaviourally as well (above), but the literal
# below is the one-line form of the production failure and is worth
# pinning on its own.
# ══════════════════════════════════════════════════════════════════════

def _ws_chat_source() -> str:
    return (Path(__file__).resolve().parents[1]
            / "app" / "api" / "ws_chat.py").read_text()


def _reader_source() -> str:
    src = _ws_chat_source()
    reader = src[src.index("async def _broadcast_reader()"):]
    return reader[:reader.index("broadcast_task = asyncio.create_task")]


def test_no_writer_synthesises_a_conversation_id():
    """`conversation_id=f"build-…"` is the defect in one line."""
    src = _ws_chat_source()
    assert 'conversation_id=f"build-' not in src
    assert "conversation_id=f'build-" not in src


def test_the_latch_comes_after_the_attempt():
    """`_persisted_job_ids.add(_jid)` ran BEFORE the try, so the first
    failure retired the job for the life of the socket. The success latch
    must sit after the call that earns it."""
    src = _ws_chat_source()
    writer = src[src.index("async def _write_job_card("):]
    writer = writer[:writer.index('@router.websocket("/ws/chat")')]
    attempt = writer.index("await _persist_job_card(")
    assert writer.index("budget.closed.add(") > attempt
