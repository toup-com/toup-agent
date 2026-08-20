"""Current context — the six-layer replacement for "Working on" (v3 §6).

Self-contained sqlite (its own engine), a frozen clock and a stubbed model,
so every assertion here is about MECHANISM: the serialisation both clients
parse, the per-layer budgets, the injected render's drop order, the
ten-minute debounce and its bypasses, and a rollover that is idempotent per
user-local day and monotonic across a DST fall-back.

Nothing in this file calls a real model. A clock-dependent test with a live
model is not a test, and the QUALITY of a rewritten Today paragraph is not
something a unit test can observe — that needs a real key and a real eye.
"""

import json
import uuid
from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models.app import BuildJob
from app.db.models.base import Base
from app.db.models.day_chat import DayChat
from app.db.models.memory import MemoryFile, MemoryFileChange
from app.db.models.routine import Routine
from app.db.models.user import User
from app.memory_files import (
    CAP_CURRENT_CONTEXT,
    CURRENT_CONTEXT_LAYERS,
    CURRENT_CONTEXT_SLUG,
    LAYER_BUDGETS,
    LAYER_TODAY,
    MAX_MONTH_PARAGRAPHS,
    MONTH_PARAGRAPH_MAX,
    PROSE_LAYERS,
    CurrentContext,
    clamp_prose,
    month_key,
    parse_current_context,
    parse_month_key,
    render_current_context,
    render_user_brain,
    trim_current_context,
)
from app.services import current_context as cc
from app.services import memory_file_ops as ops
from app.services.user_identity import forget_cached_identity

UTC = timezone.utc


# ── Harness ───────────────────────────────────────────────────────────

async def _session(tz: str | None = "America/Toronto"):
    engine = create_async_engine(
        "sqlite+aiosqlite://", connect_args={"check_same_thread": False}
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all, tables=[
            User.__table__, MemoryFile.__table__, MemoryFileChange.__table__,
            DayChat.__table__, Routine.__table__, BuildJob.__table__,
        ])
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    db = maker()
    user_id = str(uuid.uuid4())
    user = User(
        id=user_id, email=f"{user_id[:8]}@test.local", hashed_password="x",
        name="Nariman Hosseini",
    )
    if tz is not None:
        user.timezone = tz
    db.add(user)
    await db.commit()
    forget_cached_identity()
    return db, user_id, maker


class _StubLLM:
    """One model, a scripted reply queue, and a record of every prompt."""

    def __init__(self, *replies, raises: BaseException | None = None):
        self.replies = list(replies)
        self.raises = raises
        self.prompts: list[str] = []

    async def complete(self, messages, model=None, temperature=None, **kwargs):
        self.prompts.append(messages[0]["content"])
        if self.raises is not None:
            raise self.raises
        text = self.replies.pop(0) if self.replies else ""

        class _Response:
            content = text

        return _Response()

    @property
    def calls(self) -> int:
        return len(self.prompts)


@pytest.fixture
def stub_llm(monkeypatch):
    def _install(*replies, raises=None):
        stub = _StubLLM(*replies, raises=raises)
        import app.services.memory_curator as curator

        monkeypatch.setattr(curator, "_llm", lambda api_key=None: stub)
        return stub

    return _install


async def _body(db, user_id: str) -> str:
    row = (await db.execute(select(MemoryFile).where(
        MemoryFile.user_id == user_id, MemoryFile.slug == CURRENT_CONTEXT_SLUG,
    ))).scalar_one()
    return row.body_md or ""


async def _meta(db, user_id: str) -> dict:
    row = (await db.execute(select(MemoryFile).where(
        MemoryFile.user_id == user_id, MemoryFile.slug == CURRENT_CONTEXT_SLUG,
    ))).scalar_one()
    return json.loads(row.pinned_meta_json) if row.pinned_meta_json else {}


async def _seed_body(db, user_id: str, ctx: CurrentContext, meta: dict | None = None):
    await ops.ensure_system_files(db, user_id)
    await db.commit()
    row = (await db.execute(select(MemoryFile).where(
        MemoryFile.user_id == user_id, MemoryFile.slug == CURRENT_CONTEXT_SLUG,
    ))).scalar_one()
    row.body_md = render_current_context(ctx)
    if meta is not None:
        row.pinned_meta_json = json.dumps(meta)
    await db.commit()


def _fill(label: str, cap: int) -> str:
    """Sentences up to (and no further than) a budget."""
    out = ""
    n = 1
    while True:
        nxt = f"{out}{label} fact number {n} sits here. "
        if len(nxt.strip()) > cap:
            return out.strip()
        out, n = nxt, n + 1


def _full_context() -> CurrentContext:
    """A file at its stated maximum: five full layers plus twelve full month
    paragraphs — ~5,800 characters, which is what makes the injected render
    have to shed something."""
    ctx = CurrentContext(today_note="Thu, Aug 20 — America/Toronto")
    for layer in PROSE_LAYERS:
        ctx.set(layer, _fill(layer, LAYER_BUDGETS[layer]))
    months = []
    year, month = 2026, 8
    for _ in range(MAX_MONTH_PARAGRAPHS):
        label = month_key(date(year, month, 1))
        months.append((label, _fill(label, MONTH_PARAGRAPH_MAX)))
        year, month = (year - 1, 12) if month == 1 else (year, month - 1)
    ctx.months = months
    return ctx


# ══ 1. The content model ══════════════════════════════════════════════

def test_an_empty_file_round_trips_as_an_empty_body():
    """A brand-new user's Current context is EMPTY, not a stack of bare
    headings. `body_is_empty` reads `.strip()`, so headings-only would make
    the prompt health line, the injected block and the client's empty state
    all claim the file has something in it."""
    assert render_current_context(CurrentContext()) == ""
    assert parse_current_context("").is_empty()
    assert parse_current_context(None).is_empty()


def test_a_full_twelve_month_file_round_trips_byte_for_byte():
    ctx = _full_context()
    body = render_current_context(ctx)
    back = parse_current_context(body)
    assert render_current_context(back) == body
    assert list(back.layers) == list(PROSE_LAYERS)
    assert len(back.months) == MAX_MONTH_PARAGRAPHS
    assert back.today_note == "Thu, Aug 20 — America/Toronto"


def test_the_serialisation_is_the_one_the_clients_parse():
    """Mobile `memoryModel.ts::parseContextLayers` (vendored byte-exact to
    web) reads `^\\s*(#{1,6})\\s+(.*)$` and splits a trailing `(…)` off the
    heading as the layer's note. A `###` line is a SUB-heading of the layer
    above it, which is what makes `Past 12 months` hold its months."""
    ctx = CurrentContext(today_note="Thu, Aug 20 — America/Toronto")
    ctx.set(LAYER_TODAY, "Has soccer at 5:20 PM.")
    ctx.months = [("Aug 2026", "Sat the exam.")]
    body = render_current_context(ctx)
    assert body.splitlines()[0] == "## Today (Thu, Aug 20 — America/Toronto)"
    assert "## Past 12 months" in body
    assert "### Aug 2026" in body
    # The note may never contain parentheses — the client's splitter is
    # `\\(([^()]*)\\)` and a nested pair costs the layer its title.
    note = cc.today_note(date(2026, 8, 20), "America/Toronto")
    assert "(" not in note and ")" not in note


def test_every_canonical_layer_name_is_recognised_case_insensitively():
    body = "\n\n".join(f"## {name.lower()}\ntext for {name}"
                       for name in CURRENT_CONTEXT_LAYERS[:5])
    ctx = parse_current_context(body)
    assert list(ctx.layers) == list(PROSE_LAYERS)


def test_a_stray_bullet_is_folded_into_the_prose():
    """This file is prose on every surface, so a `- ` that survived into a
    body would be rendered as a literal dash by the client and read as a
    bullet by nobody."""
    ctx = parse_current_context("## Today\n- has soccer at 5:20 PM\n- drafting the essay")
    assert ctx.get(LAYER_TODAY) == "has soccer at 5:20 PM drafting the essay"


def test_prose_before_the_first_heading_is_kept_not_dropped():
    ctx = parse_current_context("loose text\n\n## Today\nreal text")
    assert ctx.get(LAYER_TODAY) == "real text"
    assert any(v == "loose text" for v in ctx.layers.values())


# ══ 2. Budgets ════════════════════════════════════════════════════════

@pytest.mark.parametrize("layer,cap", sorted(LAYER_BUDGETS.items()))
def test_every_layer_is_clamped_to_its_budget_on_write(layer, cap):
    ctx = CurrentContext()
    ctx.set(layer, ("Sentence number one is here. " * 60))
    assert len(ctx.get(layer)) <= cap
    assert ctx.get(layer)


def test_a_clamped_layer_never_ends_mid_sentence():
    """Half a sentence is a false statement, not a short one — the same
    reasoning `truncate_body` applies to a bullet, at the granularity a
    paragraph actually has."""
    text = "First sentence here. Second sentence here. Third sentence here."
    assert clamp_prose(text, 45) == "First sentence here. Second sentence here."
    # No sentence end to find inside the budget: cut at a word and SAY so.
    assert clamp_prose("a" * 20 + " " + "b" * 40, 30).endswith("…")


def test_clamp_prose_handles_the_persian_question_mark():
    assert clamp_prose("سلام چطوری؟ بعدی اینجاست.", 12).endswith("؟")


def test_a_month_paragraph_is_clamped_and_months_cap_at_twelve():
    ctx = _full_context()
    cc.merge_month(ctx, "Sep 2026", "x" * 900)
    assert len(dict(ctx.months)["Sep 2026"]) <= MONTH_PARAGRAPH_MAX
    assert len(ctx.months) <= MAX_MONTH_PARAGRAPHS


def test_merging_a_month_that_already_exists_joins_rather_than_duplicates():
    ctx = CurrentContext(months=[("Aug 2026", "Sat the exam.")])
    cc.merge_month(ctx, "Aug 2026", "Moved apartments.")
    assert len(ctx.months) == 1
    assert "Sat the exam." in ctx.months[0][1]
    assert "Moved apartments." in ctx.months[0][1]


# ══ 3. The injected render (§5) ═══════════════════════════════════════

def test_the_render_sheds_the_OLDEST_months_first_and_keeps_the_layers():
    ctx = _full_context()
    body = render_current_context(ctx)
    assert len(body) > CAP_CURRENT_CONTEXT, "the fixture must exceed the cap"

    trimmed = trim_current_context(body, CAP_CURRENT_CONTEXT)
    assert len(trimmed) <= CAP_CURRENT_CONTEXT
    kept = parse_current_context(trimmed)
    # Today through This month are NEVER dropped.
    assert list(kept.layers) == list(PROSE_LAYERS)
    # The months that survived are the NEWEST ones, in order.
    assert [m for m, _ in kept.months] == [m for m, _ in ctx.months][: len(kept.months)]
    assert len(kept.months) < MAX_MONTH_PARAGRAPHS


@pytest.mark.parametrize("cap", [400, 900, 1800, 3200, 5000])
def test_the_render_never_leaves_a_heading_with_no_body(cap):
    """`## Past 12 months` is the one heading whose body is other headings.
    Every other heading must be followed by prose — an empty layer is
    dropped whole, heading and all."""
    trimmed = trim_current_context(render_current_context(_full_context()), cap)
    lines = [ln for ln in trimmed.splitlines() if ln.strip()]
    for index, line in enumerate(lines):
        if not line.startswith("#"):
            continue
        nxt = lines[index + 1] if index + 1 < len(lines) else ""
        if line == "## Past 12 months":
            assert nxt.startswith("### "), line
        else:
            assert nxt and not nxt.startswith("#"), line


def test_today_survives_even_a_budget_that_cannot_hold_it():
    ctx = CurrentContext()
    ctx.set(LAYER_TODAY, "First sentence here. Second sentence here.")
    trimmed = trim_current_context(render_current_context(ctx), 40)
    assert trimmed.startswith("## Today")
    assert "First sentence here." in trimmed


def test_a_body_in_no_recognised_shape_falls_back_to_the_bullet_cut():
    """A hand-written or migrated body with no layer heading anywhere is not
    reformatted — re-rendering it here would flatten a bullet list into one
    paragraph, which is a worse lie than a short body."""
    bullets = "\n".join(f"- fact number {n} about the day" for n in range(60))
    trimmed = trim_current_context(bullets, 300)
    assert trimmed.startswith("- fact number 0")
    assert "memory_read_file" in trimmed
    assert len(trimmed) <= 300


def test_render_user_brain_uses_the_layer_aware_trim():
    """`truncate_body` cuts between LINES and appends "open with
    memory_read_file". There is no longer version of Current context to
    open, and a line cut can leave `### Aug 2026` heading nothing."""
    body = render_current_context(_full_context())
    rendered = render_user_brain(current_context_body=body)
    assert "memory_read_file" not in rendered
    assert "## Current context" in rendered
    assert len(rendered) <= CAP_CURRENT_CONTEXT + len("## Current context\n") + 1


# ══ 4. The post-turn updater ══════════════════════════════════════════

@pytest.mark.asyncio
async def test_two_turns_a_minute_apart_cost_ONE_model_call(stub_llm):
    db, user_id, _ = await _session()
    llm = stub_llm("Working through the IELTS drills.", "Second paragraph.")
    await _seed_body(db, user_id, CurrentContext())
    db.add(DayChat(user_id=user_id, local_date=date(2026, 8, 20),
                   rolling_summary="Talked about the exam."))
    await db.commit()

    t0 = datetime(2026, 8, 20, 18, 0, tzinfo=UTC)
    first = await cc.refresh_today(db, user_id, now=t0)
    second = await cc.refresh_today(db, user_id, now=t0 + timedelta(minutes=1))

    assert first["written"] is True
    assert second == {"skipped": "debounced", "written": False}
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_the_debounce_cursor_survives_a_restart(stub_llm):
    """An in-memory cursor is reset by every container restart, and this
    fleet's median redeploy gap was 0.3 h — so the cursor is a column."""
    db, user_id, _ = await _session()
    stub_llm("A paragraph.")
    await _seed_body(db, user_id, CurrentContext())
    db.add(DayChat(user_id=user_id, local_date=date(2026, 8, 20),
                   rolling_summary="Something happened."))
    await db.commit()
    await cc.refresh_today(db, user_id, now=datetime(2026, 8, 20, 18, 0, tzinfo=UTC))
    assert cc._META_LAST_REFRESH in await _meta(db, user_id)


@pytest.mark.asyncio
async def test_a_material_change_bypasses_the_debounce(stub_llm):
    db, user_id, _ = await _session()
    llm = stub_llm("One.", "Two.")
    await _seed_body(db, user_id, CurrentContext())
    db.add(DayChat(user_id=user_id, local_date=date(2026, 8, 20),
                   rolling_summary="Talked about the exam."))
    await db.commit()

    t0 = datetime(2026, 8, 20, 18, 0, tzinfo=UTC)
    await cc.refresh_today(db, user_id, now=t0)
    again = await cc.refresh_today(
        db, user_id, now=t0 + timedelta(minutes=1), material=True,
    )
    assert again["written"] is True
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_the_first_turn_of_a_new_local_day_is_material(stub_llm):
    db, user_id, _ = await _session()
    llm = stub_llm("Day one.", "Day two.")
    await _seed_body(db, user_id, CurrentContext())
    for day in (date(2026, 8, 20), date(2026, 8, 21)):
        db.add(DayChat(user_id=user_id, local_date=day, rolling_summary=f"On {day}."))
    await db.commit()

    # 23:30 Toronto on Aug 20, then 00:10 Toronto on Aug 21 — forty minutes
    # apart, so the wall-clock debounce would have refused the second.
    await cc.refresh_today(db, user_id, now=datetime(2026, 8, 21, 3, 30, tzinfo=UTC))
    second = await cc.refresh_today(db, user_id, now=datetime(2026, 8, 21, 4, 10, tzinfo=UTC))
    assert second["written"] is True
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_a_reminder_that_fired_since_the_last_refresh_is_material(stub_llm):
    """The fire itself runs with post-processing disabled, so it produces no
    turn this updater may run on. The next real turn notices it here."""
    db, user_id, _ = await _session()
    llm = stub_llm("One.", "Two.")
    await _seed_body(db, user_id, CurrentContext())
    db.add(DayChat(user_id=user_id, local_date=date(2026, 8, 20),
                   rolling_summary="Talked about the exam."))
    await db.commit()

    t0 = datetime(2026, 8, 20, 18, 0, tzinfo=UTC)
    await cc.refresh_today(db, user_id, now=t0)
    db.add(Routine(
        user_id=user_id, kind="reminder", name="Soccer",
        reminder_text="soccer", schedule_cron_local="20 17 * * *",
        last_run_at=datetime(2026, 8, 20, 18, 0, 30),
    ))
    await db.commit()
    again = await cc.refresh_today(db, user_id, now=t0 + timedelta(minutes=2))
    assert again["written"] is True
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_a_refresh_writes_NO_memory_file_changes_row(stub_llm):
    """The Memory log is "what the writer changed about what it knows about
    you". A file that rewrites itself every ten minutes would drown every
    real line in it, so a context refresh is not a memory change."""
    db, user_id, _ = await _session()
    stub_llm("Has soccer at 5:20 PM.")
    await _seed_body(db, user_id, CurrentContext())
    db.add(DayChat(user_id=user_id, local_date=date(2026, 8, 20),
                   rolling_summary="Talked about soccer."))
    await db.commit()

    before = len((await db.execute(select(MemoryFileChange))).scalars().all())
    result = await cc.refresh_today(
        db, user_id, now=datetime(2026, 8, 20, 18, 0, tzinfo=UTC),
    )
    assert result["written"] is True
    after = (await db.execute(select(MemoryFileChange))).scalars().all()
    assert len(after) == before, [c.summary for c in after]


@pytest.mark.asyncio
async def test_a_routine_is_REFERENCED_once_and_never_owned(stub_llm):
    """The founder's acceptance criterion: their 5:06 PM quote routine
    appears at most once, as a one-line reference. So what reaches the
    prompt is a name and a local time — never the cron expression, never
    the prompt text, never the id."""
    db, user_id, _ = await _session()
    llm = stub_llm("Has a daily quote at 5:06 PM and soccer at 5:20 PM.")
    await _seed_body(db, user_id, CurrentContext())
    quote = Routine(
        user_id=user_id, kind="agent_task", name="Daily quote",
        prompt_text="Send me an inspiring quote about persistence, 2 lines max",
        schedule_cron_local="6 17 * * *",
        next_run_at=datetime(2026, 8, 20, 21, 6),      # 5:06 PM Toronto
    )
    db.add(quote)
    db.add(Routine(
        user_id=user_id, kind="reminder", name="Soccer", reminder_text="soccer",
        schedule_cron_local="20 17 * * *",
        next_run_at=datetime(2026, 8, 20, 21, 20),     # 5:20 PM Toronto
    ))
    await db.commit()

    await cc.refresh_today(db, user_id, now=datetime(2026, 8, 20, 18, 0, tzinfo=UTC))
    prompt = llm.prompts[0]
    assert prompt.count("Daily quote") == 1
    assert "5:06 PM" in prompt and "5:20 PM" in prompt
    for owned in ("6 17 * * *", "20 17 * * *", quote.prompt_text, quote.id):
        assert owned not in prompt, owned


@pytest.mark.asyncio
async def test_the_prompt_asks_for_STATE_not_a_transcript(stub_llm):
    db, user_id, _ = await _session()
    llm = stub_llm("A paragraph.")
    await _seed_body(db, user_id, CurrentContext())
    db.add(DayChat(user_id=user_id, local_date=date(2026, 8, 20),
                   rolling_summary="Discussed IELTS."))
    await db.commit()
    await cc.refresh_today(db, user_id, now=datetime(2026, 8, 20, 18, 0, tzinfo=UTC))
    prompt = llm.prompts[0]
    assert "Never recap the conversation" in prompt
    assert "STATE" in prompt


@pytest.mark.asyncio
async def test_a_refusal_leaves_the_previous_paragraph_standing(stub_llm):
    """A reply carrying an internal id or a tool parameter is not written.
    Refusing costs one stale paragraph; writing puts engine metadata in the
    one place the product presents as fact."""
    db, user_id, _ = await _session()
    ctx = CurrentContext()
    ctx.set(LAYER_TODAY, "Drafting the IELTS essay.")
    await _seed_body(db, user_id, ctx)
    stub_llm("Ran the briefing with max_results=1.")
    db.add(DayChat(user_id=user_id, local_date=date(2026, 8, 20), rolling_summary="x y z"))
    await db.commit()

    result = await cc.refresh_today(
        db, user_id, now=datetime(2026, 8, 20, 18, 0, tzinfo=UTC),
    )
    assert result["written"] is False
    assert "Drafting the IELTS essay." in await _body(db, user_id)


@pytest.mark.asyncio
async def test_a_model_that_says_NOTHING_writes_nothing(stub_llm):
    db, user_id, _ = await _session()
    stub_llm("NOTHING")
    await _seed_body(db, user_id, CurrentContext())
    db.add(DayChat(user_id=user_id, local_date=date(2026, 8, 20), rolling_summary="hello"))
    await db.commit()
    result = await cc.refresh_today(
        db, user_id, now=datetime(2026, 8, 20, 18, 0, tzinfo=UTC),
    )
    assert result == {"skipped": "model_said_nothing", "written": False}
    assert await _body(db, user_id) == ""


@pytest.mark.asyncio
async def test_an_empty_day_costs_no_model_call(stub_llm):
    db, user_id, _ = await _session()
    llm = stub_llm("never used")
    await _seed_body(db, user_id, CurrentContext())
    result = await cc.refresh_today(
        db, user_id, now=datetime(2026, 8, 20, 18, 0, tzinfo=UTC),
    )
    assert result == {"skipped": "nothing_to_say", "written": False}
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_the_updater_raising_cannot_break_the_turn(monkeypatch):
    """`refresh_after_turn` is what makes this unable to fail a reply: it
    opens its own session, and it swallows."""
    db, user_id, maker = await _session()

    async def _boom(*args, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(cc, "refresh_today", _boom)
    assert await cc.refresh_after_turn(maker, user_id) is None


@pytest.mark.asyncio
async def test_markdown_from_the_model_is_folded_back_into_prose(stub_llm):
    db, user_id, _ = await _session()
    stub_llm("## Today\n- has soccer at 5:20 PM\n- drafting the essay")
    await _seed_body(db, user_id, CurrentContext())
    db.add(DayChat(user_id=user_id, local_date=date(2026, 8, 20), rolling_summary="x y z"))
    await db.commit()
    await cc.refresh_today(db, user_id, now=datetime(2026, 8, 20, 18, 0, tzinfo=UTC))
    ctx = parse_current_context(await _body(db, user_id))
    assert ctx.get(LAYER_TODAY) == "has soccer at 5:20 PM drafting the essay"
    assert list(ctx.layers) == [LAYER_TODAY]


# ══ 5. Timezones ══════════════════════════════════════════════════════

def test_a_plus_0330_user_crosses_midnight_before_utc_does():
    """Asia/Tehran is +03:30 all year (Iran abolished DST in 2022), so
    20:45 UTC on Aug 19 is already 00:15 on Aug 20 in Tehran."""
    when = datetime(2026, 8, 19, 20, 45, tzinfo=UTC)
    assert cc._local_date(when, "Asia/Tehran") == date(2026, 8, 20)
    assert cc._local_date(when, "America/Toronto") == date(2026, 8, 19)
    assert cc._local_date(when, None) == date(2026, 8, 19)


def test_a_missing_or_invalid_timezone_falls_back_to_utc():
    when = datetime(2026, 8, 19, 20, 45, tzinfo=UTC)
    assert cc._local_date(when, "Mars/Olympus_Mons") == date(2026, 8, 19)
    assert cc._local_date(when, "") == date(2026, 8, 19)


@pytest.mark.asyncio
async def test_an_unparseable_User_timezone_resolves_to_utc():
    db, user_id, _ = await _session(tz="Not/AZone")
    assert await ops.resolve_user_tz(db, user_id) is None


def test_the_today_note_names_the_day_and_the_zone():
    assert cc.today_note(date(2026, 8, 20), "America/Toronto") == (
        "Thu, Aug 20 — America/Toronto"
    )
    assert cc.today_note(date(2026, 8, 20), None).endswith("UTC")


# ══ 6. Rollover ═══════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_the_first_pass_plants_the_cursor_and_ages_nothing():
    db, user_id, _ = await _session()
    ctx = CurrentContext()
    ctx.set(LAYER_TODAY, "Drafting the essay.")
    await _seed_body(db, user_id, ctx)

    result = await cc.roll_over_user(
        db, user_id, now=datetime(2026, 8, 20, 18, 0, tzinfo=UTC),
    )
    assert result["rolled"] is False and result["reason"] == "cursor_planted"
    assert (await _meta(db, user_id))[cc._META_LAST_ROLLOVER] == "2026-08-20"
    assert "Drafting the essay." in await _body(db, user_id)


@pytest.mark.asyncio
async def test_three_runs_in_one_local_day_roll_once():
    db, user_id, _ = await _session()
    ctx = CurrentContext()
    ctx.set(LAYER_TODAY, "Drafting the essay.")
    await _seed_body(db, user_id, ctx, {cc._META_LAST_ROLLOVER: "2026-08-19"})

    first = await cc.roll_over_user(
        db, user_id, now=datetime(2026, 8, 20, 14, 0, tzinfo=UTC),
    )
    assert first["rolled"] is True and first["days"] == 1
    body = await _body(db, user_id)
    for hour in (15, 16, 17):
        again = await cc.roll_over_user(
            db, user_id, now=datetime(2026, 8, 20, hour, 0, tzinfo=UTC),
        )
        assert again["rolled"] is False and again["reason"] == "not_a_new_day"
    assert await _body(db, user_id) == body


@pytest.mark.asyncio
async def test_one_day_moves_today_into_yesterday_and_clears_today():
    db, user_id, _ = await _session()
    ctx = CurrentContext()
    ctx.set(LAYER_TODAY, "Drafting the essay.")
    ctx.set("Yesterday", "Booked the exam.")
    await _seed_body(db, user_id, ctx, {cc._META_LAST_ROLLOVER: "2026-08-19"})

    await cc.roll_over_user(db, user_id, now=datetime(2026, 8, 20, 14, 0, tzinfo=UTC))
    rolled = parse_current_context(await _body(db, user_id))
    assert rolled.get(LAYER_TODAY) == ""
    assert rolled.get("Yesterday") == "Drafting the essay."
    assert rolled.get("Last 2 days") == "Booked the exam."


@pytest.mark.asyncio
async def test_a_six_day_gap_rolls_forward_in_ONE_pass():
    """A user away for a week is ordinary. Rolling forward N days must not
    take N passes, and Yesterday must never end up back in Today."""
    db, user_id, _ = await _session()
    ctx = CurrentContext()
    ctx.set(LAYER_TODAY, "Drafting the essay.")
    ctx.set("Yesterday", "Booked the exam.")
    ctx.set("This month", "Moved apartments.")
    await _seed_body(db, user_id, ctx, {cc._META_LAST_ROLLOVER: "2026-08-14"})

    result = await cc.roll_over_user(
        db, user_id, now=datetime(2026, 8, 20, 14, 0, tzinfo=UTC),
    )
    assert result["rolled"] is True and result["days"] == 6
    rolled = parse_current_context(await _body(db, user_id))
    assert rolled.get(LAYER_TODAY) == ""
    assert rolled.get("Yesterday") == ""
    assert "Drafting the essay." in rolled.get("This month")
    assert "Moved apartments." in rolled.get("This month")


def test_the_rollover_is_monotonic_content_can_only_get_older():
    """`This week`'s representative is its Monday, so a Monday→Tuesday step
    classifies it as "yesterday" by date alone. Clamping the destination to
    the source's own slot is what stops the file rolling BACKWARDS."""
    ctx = CurrentContext()
    ctx.set("This week", "week text.")
    ctx.set(LAYER_TODAY, "today text.")
    rolled, todo = cc.plan_rollover(ctx, date(2026, 8, 17), date(2026, 8, 18))
    assert rolled.get("This week") == "week text."
    assert rolled.get("Yesterday") == "today text."
    assert todo == []


@pytest.mark.asyncio
async def test_a_clock_that_goes_backwards_never_rolls():
    """DST fall-back, a corrected timezone, a replica with a skewed clock."""
    db, user_id, _ = await _session()
    ctx = CurrentContext()
    ctx.set(LAYER_TODAY, "Drafting the essay.")
    await _seed_body(db, user_id, ctx, {cc._META_LAST_ROLLOVER: "2026-11-02"})

    result = await cc.roll_over_user(
        db, user_id, now=datetime(2026, 11, 1, 14, 0, tzinfo=UTC),
    )
    assert result["rolled"] is False and result["reason"] == "not_a_new_day"
    assert parse_current_context(await _body(db, user_id)).get(LAYER_TODAY) == (
        "Drafting the essay."
    )
    assert (await _meta(db, user_id))[cc._META_LAST_ROLLOVER] == "2026-11-02"


@pytest.mark.asyncio
@pytest.mark.parametrize("label,prev,now_utc", [
    # Spring forward: Sun Mar 8 2026, 02:00 → 03:00 in America/Toronto.
    ("spring-forward", "2026-03-07", datetime(2026, 3, 8, 16, 0, tzinfo=UTC)),
    # Fall back: Sun Nov 1 2026, 02:00 → 01:00.
    ("fall-back", "2026-10-31", datetime(2026, 11, 1, 16, 0, tzinfo=UTC)),
])
async def test_a_dst_transition_is_exactly_one_day(label, prev, now_utc):
    db, user_id, _ = await _session()
    ctx = CurrentContext()
    ctx.set(LAYER_TODAY, "Drafting the essay.")
    await _seed_body(db, user_id, ctx, {cc._META_LAST_ROLLOVER: prev})

    result = await cc.roll_over_user(db, user_id, now=now_utc)
    assert result["rolled"] is True, label
    assert result["days"] == 1, label
    rolled = parse_current_context(await _body(db, user_id))
    assert rolled.get("Yesterday") == "Drafting the essay.", label


@pytest.mark.asyncio
async def test_a_month_boundary_compresses_with_ONE_model_call(stub_llm):
    db, user_id, _ = await _session()
    llm = stub_llm("Moved apartments and started the IELTS course.")
    ctx = CurrentContext()
    ctx.set(LAYER_TODAY, "Drafting the essay.")
    ctx.set("This month", "Moved apartments. Started the course.")
    await _seed_body(db, user_id, ctx, {cc._META_LAST_ROLLOVER: "2026-07-31"})
    db.add(DayChat(user_id=user_id, local_date=date(2026, 7, 15),
                   archival_summary="Signed the lease."))
    await db.commit()

    result = await cc.roll_over_user(
        db, user_id, now=datetime(2026, 8, 1, 14, 0, tzinfo=UTC),
    )
    assert result["rolled"] is True and result["months_written"] == 1
    assert llm.calls == 1
    assert "Signed the lease." in llm.prompts[0], "archival day summaries feed it"
    rolled = parse_current_context(await _body(db, user_id))
    assert dict(rolled.months)["Jul 2026"].startswith("Moved apartments and started")
    assert rolled.get("This month") == ""


@pytest.mark.asyncio
async def test_one_model_failure_leaves_the_file_and_the_cursor_UNTOUCHED(stub_llm):
    """Never a half-rolled body: everything is computed before anything is
    written, so the next hourly pass simply tries again."""
    db, user_id, _ = await _session()
    stub_llm(raises=RuntimeError("provider down"))
    ctx = CurrentContext()
    ctx.set(LAYER_TODAY, "Drafting the essay.")
    ctx.set("This month", "Moved apartments.")
    await _seed_body(db, user_id, ctx, {cc._META_LAST_ROLLOVER: "2026-07-31"})
    before = await _body(db, user_id)

    with pytest.raises(RuntimeError):
        await cc.roll_over_user(db, user_id, now=datetime(2026, 8, 1, 14, 0, tzinfo=UTC))
    await db.rollback()
    assert await _body(db, user_id) == before
    assert (await _meta(db, user_id))[cc._META_LAST_ROLLOVER] == "2026-07-31"


@pytest.mark.asyncio
async def test_the_two_writers_do_not_clobber_each_other_s_cursor(stub_llm):
    """Both cursors live in the same `pinned_meta_json` blob, and the hourly
    rollover and a post-turn refresh can overlap. Neither may put back the
    copy it read before its own model call."""
    db, user_id, _ = await _session()
    stub_llm("Has soccer at 5:20 PM.")
    await _seed_body(db, user_id, CurrentContext(), {cc._META_LAST_ROLLOVER: "2026-08-20"})
    db.add(DayChat(user_id=user_id, local_date=date(2026, 8, 20), rolling_summary="x y z"))
    await db.commit()

    await cc.refresh_today(db, user_id, now=datetime(2026, 8, 20, 18, 0, tzinfo=UTC))
    meta = await _meta(db, user_id)
    assert meta[cc._META_LAST_ROLLOVER] == "2026-08-20"   # refresh kept it
    assert cc._META_LAST_REFRESH in meta

    await cc.roll_over_user(db, user_id, now=datetime(2026, 8, 21, 18, 0, tzinfo=UTC))
    meta = await _meta(db, user_id)
    assert meta[cc._META_LAST_ROLLOVER] == "2026-08-21"
    assert cc._META_LAST_REFRESH in meta                  # rollover kept it


def test_month_keys_round_trip():
    assert month_key(date(2026, 8, 20)) == "Aug 2026"
    assert parse_month_key("Aug 2026") == (2026, 8)
    assert parse_month_key("not a month") is None


# ══ 7. The boundary with the curator ══════════════════════════════════

@pytest.mark.asyncio
async def test_the_curator_cannot_write_bullets_into_current_context():
    """`FileState.has_prose` is one half of the boundary — the curator must
    not be able to flatten the layers on its way past."""
    db, user_id, _ = await _session()
    ctx = CurrentContext()
    ctx.set(LAYER_TODAY, "Has soccer at 5:20 PM.")
    await _seed_body(db, user_id, ctx)

    rows = await ops._all_files(db, user_id)
    plan = ops.validate_ops([
        {"op": "add", "slug": CURRENT_CONTEXT_SLUG,
         "bullet": "likes Googoosh", "change": "Added a fact."},
    ], rows)
    assert plan.accepted == []
    assert any("not a bullet list" in c for c in plan.complaints)
    assert "Has soccer at 5:20 PM." in await _body(db, user_id)


@pytest.mark.asyncio
async def test_the_updater_writes_current_context_and_NOTHING_else(stub_llm):
    db, user_id, _ = await _session()
    stub_llm("Has soccer at 5:20 PM.")
    await _seed_body(db, user_id, CurrentContext())
    rows = {r.slug: (r.body_md or "") for r in await ops._all_files(db, user_id)}
    db.add(DayChat(user_id=user_id, local_date=date(2026, 8, 20), rolling_summary="x y z"))
    await db.commit()

    await cc.refresh_today(db, user_id, now=datetime(2026, 8, 20, 18, 0, tzinfo=UTC))
    after = {r.slug: (r.body_md or "") for r in await ops._all_files(db, user_id)}
    assert after.pop(CURRENT_CONTEXT_SLUG) != rows.pop(CURRENT_CONTEXT_SLUG)
    assert after == rows


@pytest.mark.asyncio
async def test_a_FULL_file_survives_the_loader_and_the_renderer():
    """WS-1 wired both injection seams against a file that was always empty.
    This is the same path with the file at its stated maximum: one loader
    (`load_brain`), one renderer (`render_user_brain`), and the block that
    comes out is inside the cap with every layer still whole."""
    db, user_id, _ = await _session()
    await _seed_body(db, user_id, _full_context())

    brain = await ops.load_brain(db, user_id, query="soccer")
    assert brain.current_context.startswith("## Today (")
    block = render_user_brain(
        profile_body=brain.profile,
        current_context_body=brain.current_context,
        learned_body=brain.learned,
        index=brain.index,
        relevant=brain.relevant,
    )
    assert "## Current context" in block
    body = block.split("## Current context\n", 1)[1]
    assert len(body) <= CAP_CURRENT_CONTEXT
    kept = parse_current_context(body)
    assert list(kept.layers) == list(PROSE_LAYERS)
    assert 0 < len(kept.months) < MAX_MONTH_PARAGRAPHS
    # Whole sentences only — nothing ends mid-word.
    for text in list(kept.layers.values()) + [t for _, t in kept.months]:
        assert not text.endswith("…"), text


# ══ 8. The seams (source probes) ══════════════════════════════════════
#
# The behaviour above is all in one module; these read the CALL SITES,
# because a guard whose precondition something above it destroys is
# invisible to every other check in this repo.

def _src(path: str) -> str:
    import pathlib

    return (pathlib.Path(__file__).resolve().parents[1] / path).read_text()


def test_the_runner_spawns_the_refresh_INSIDE_the_post_processing_gate():
    """Never for a SUBAGENT turn, never for a routine / autopilot / email
    handler — all of which pass `disable_post_processing=True`."""
    src = _src("app/agent/agent_runner.py")
    gate = src.index("if not disable_post_processing:\n            _spawn_background")
    spawn = src.index("spawn_refresh(", gate)
    closing = src.index("        else:\n            logger.debug(\n"
                        "                \"[AGENT] background post-processing SKIPPED", gate)
    assert gate < spawn < closing


def test_the_runner_derives_the_material_signal_from_tool_names():
    src = _src("app/agent/agent_runner.py")
    assert "_ctx_material" in src
    assert '"create_job", "update_job"' in src
    assert '__remind' in src


def test_the_voice_route_spawns_the_refresh_off_the_response_path():
    src = _src("app/api/api_v1.py")
    curate = src.index("memory_curator.curate_turn(")
    spawn = src.index("spawn_refresh(async_session_maker, user_id)", curate)
    respond = src.index("return CurateTurnResponse(", curate)
    assert curate < spawn < respond


def test_the_rollover_is_registered_hourly_and_on_a_CRON_trigger():
    """RC3.1: an interval trigger's first fire is measured from scheduler
    start, and this fleet is recreated more often than an hour."""
    src = _src("agent_main.py")
    assert '"current_context_rollover"' in src
    block = src[src.index('"current_context_rollover"'):]
    assert "_MMCron(minute=5)" in block[:400]
    assert "IntervalTrigger" not in block[:400]
    assert "_IT(" not in block[:400]


def test_the_boot_slot_also_runs_the_rollover():
    src = _src("app/services/memory_file_ops.py")
    assert "run_context_rollover" in src


def test_current_context_is_never_written_through_apply_ops():
    """One writer for this file, and it is not the ops engine. `_save` sets
    the row directly, which is also what keeps the change log clean."""
    src = _src("app/services/current_context.py")
    assert "apply_ops" not in src
    assert "MemoryFileChange" not in src
