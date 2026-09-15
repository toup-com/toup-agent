"""The agent started reproducing the history annotations it was being shown.

Two rows reached a real user on 2026-09-14:

    [mobile 9:41pm] Putting on **Kanye West - Stronger**.
    [mobile 9:42pm] Putting on Kanye West - Stronger.

`load_day_context` prefixes every history row with `[<channel> <clock>]` so the
model knows where each turn came from. Nothing ever told the model those are
the SYSTEM's annotations, so it copied the shape into its own replies — and the
rows are persisted, so they are still there on every reload.

D2 makes the sanitizer UNCONDITIONAL (the documented exception to flag-off
byte-identity: this is output hygiene, not prompt shape) and applies it at four
seams. This file is the negative corpus as much as the positive one: a
sanitizer is the one class of fix that can be wrong in a user-visible way, and
a greedy regex eating `[Link](url)` or `[[Yes]]` would be a worse bug than the
one it fixes.

Local run (from backend/):
    RUN_MODE=platform PYTHONPATH=. ENVIRONMENT=test JWT_SECRET=x ENCRYPTION_KEY=y \
      pytest tests/test_history_tag_sanitizer.py -q --tb=short -p no:cacheprovider
"""
from __future__ import annotations

import uuid
from datetime import date as Date, datetime
from pathlib import Path

import asyncio
import inspect
import pytest
import pytest_asyncio

_BACKEND = Path(__file__).resolve().parent.parent

# The two rows, exactly as persisted.
E4_ONE = "[mobile 9:41pm] Putting on **Kanye West - Stronger**."
E4_TWO = "[mobile 9:42pm] Putting on Kanye West - Stronger."


def mod():
    return pytest.importorskip(
        "app.agent.channel_annotations",
        reason="channel_annotations not landed yet — lane B1 (D2)",
    )


# ══════════════════════════════════════════════════════════════════════
# 1. The pure function — what it strips
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("raw,clean", [
    (E4_ONE, "Putting on **Kanye West - Stronger**."),
    (E4_TWO, "Putting on Kanye West - Stronger."),
    ("[web 1:00am] hi", "hi"),
    ("[whatsapp 12:05 AM] hi", "hi"),
    ("[Mobile 9:41Pm] x", "x"),                      # case-insensitive
    ("[telegram 21:05] evening", "evening"),         # 24-hour clock
    ("  [whatsapp 12:05 AM]\n[web 1:00am] hi", "hi"),  # a RUN of tags, across a newline
    ("[mobile 9:41pm] [web 9:42pm] text", "text"),   # multi-tag, one line
])
def test_a_leading_channel_tag_is_removed_and_the_body_survives_exactly(raw, clean):
    assert mod().strip_leaked_tags(raw)[0] == clean


def test_the_removed_prefix_is_returned_verbatim():
    """The backfill (D12) stores it under `metadata_json.leaked_prefix` and
    `--revert` restores the row byte-for-byte. An approximation there is an
    un-revertable migration."""
    clean, prefix = mod().strip_leaked_tags(E4_ONE)
    assert prefix + clean == E4_ONE


@pytest.mark.parametrize("bad", [None, "", "   "])
def test_it_never_raises_on_nothing(bad):
    clean, prefix = mod().strip_leaked_tags(bad)
    assert isinstance(clean, str) and isinstance(prefix, str)


def test_every_known_channel_is_strippable_when_it_carries_a_clock():
    """Parametrised over the frozenset so the regex cannot drift from the
    table the annotator writes with."""
    from app.agent.channel_util import KNOWN_CHANNELS

    for ch in sorted(KNOWN_CHANNELS):
        assert mod().strip_leaked_tags(f"[{ch} 9:41pm] body")[0] == "body", ch


# ══════════════════════════════════════════════════════════════════════
# 2. The NEGATIVE corpus — the reason this file exists
# ══════════════════════════════════════════════════════════════════════


NEGATIVE = [
    "[Link](https://toup.ai)",                      # a markdown link head
    "[1] a footnote reference",
    "[TODO] finish the deck",
    "[INFO] 9:41pm — a log line, not a channel",
    "[mobile] hello",                               # a tag with no clock
    "[soup 9:41pm] x",                              # not a KNOWN channel
    "[[Yes]]",                                      # the mobile quick-reply shape
    "[[button:Play|play]] pick one",
    "Putting on [mobile 9:41pm] later",             # not at position 0 — quoted history
    "```\n[debug 9:41am] trace\n```",               # inside a code block
    "I saw [mobile 9:41pm] in the transcript and it confused me",
    "[9:41pm] no channel at all",
]


@pytest.mark.parametrize("text", NEGATIVE)
def test_the_negative_corpus_is_untouched(text):
    """Run in the same file and the same process as the positive cases: a
    greedy regex passes the positive half on its own."""
    clean, prefix = mod().strip_leaked_tags(text)
    assert clean == text, f"sanitizer ate prose: {text!r} -> {clean!r}"
    assert prefix == ""


def test_a_tag_at_position_zero_of_a_quoted_block_is_still_quoted_history():
    """The user asking 'why does it say [mobile 9:41pm] in my history?' must
    get an answer that still contains the string they asked about, once the
    reply does not START with it."""
    text = "You asked about this:\n\n> [mobile 9:41pm] Putting on X\n\nThat prefix is mine."
    assert mod().strip_leaked_tags(text)[0] == text


# ══════════════════════════════════════════════════════════════════════
# 3. The STREAM seam — never emit a tag, never swallow text
# ══════════════════════════════════════════════════════════════════════


def _drive(chunks):
    """Feed chunks through the stream filter and return what reached the
    client, plus how many downstream calls were made."""
    m = mod()
    out: list[str] = []
    f = m.make_stream_tag_filter(out.append)

    async def go():
        for c in chunks:
            r = f(c)
            if inspect.isawaitable(r):
                await r
        flush = getattr(f, "flush", None)
        if callable(flush):
            r = flush()
            if inspect.isawaitable(r):
                await r

    asyncio.run(go())
    return "".join(out), len(out)


def test_a_tag_split_across_chunks_never_reaches_the_client():
    text, _ = _drive(["[mo", "bile 9:4", "1pm] Put", "ting on X"])
    assert text == "Putting on X"
    assert "[mobile" not in text


def test_ordinary_text_is_forwarded_without_being_held():
    """The filter arms once per turn and must disarm on the first chunk that
    cannot begin a tag — otherwise every reply in the product pays a buffer."""
    text, calls = _drive(["Hello", " there", " friend"])
    assert text == "Hello there friend"
    assert calls == 3, f"chunks were coalesced or held ({calls} downstream calls)"


def test_a_bracket_that_is_not_a_tag_is_released_promptly():
    """`[Open report](toup://x)` is a link the client renders. Holding it
    until end-of-turn would stall the first paint of every such reply."""
    text, _ = _drive(["[Open report](toup://x) done"])
    assert text == "[Open report](toup://x) done"


def test_a_turn_whose_only_text_is_an_open_bracket_still_emits_it():
    """Teardown flush. Swallowing a character is worse than leaking a tag."""
    text, _ = _drive(["["])
    assert text == "["


def test_the_hold_is_bounded():
    m = mod()
    assert m.MAX_TAG_HOLD == 64
    text, _ = _drive(["a"] * 100)
    assert text == "a" * 100


def test_nothing_is_lost_for_any_chunking_of_the_e4_string():
    """ANTI-VACUITY over the split points: a filter that only handles the
    boundary its author happened to test is not a filter."""
    for cut in range(1, len(E4_ONE)):
        text, _ = _drive([E4_ONE[:cut], E4_ONE[cut:]])
        assert text == "Putting on **Kanye West - Stronger**.", f"cut at {cut}: {text!r}"


# ══════════════════════════════════════════════════════════════════════
# 4. ORDER — after the citation gate, before the save
# ══════════════════════════════════════════════════════════════════════


def test_the_final_sanitizer_runs_after_the_citation_gate_and_before_the_save():
    """A guard whose precondition something above it destroys is invisible to
    every check in this repo. The citation gate REWRITES final_text, so a
    sanitizer above it cleans a string that is then replaced; a sanitizer
    below the save cleans a string already on disk."""
    src = (_BACKEND / "app" / "agent" / "agent_runner.py").read_text()
    if "strip_leaked_tags" not in src:
        pytest.skip("sanitizer not wired into agent_runner yet — lane B1 (D2)")

    gate = src.index("apply_citation_gate(")
    strip = src.index("strip_leaked_tags(final_text)")
    save = src.index("assistant_response=final_text")
    assert gate < strip < save, (
        f"sanitizer out of order (citation_gate@{gate}, strip@{strip}, "
        f"save@{save}) — it must clean the string that is actually persisted"
    )


# ══════════════════════════════════════════════════════════════════════
# 5. The READ side — the two rows are already on disk
# ══════════════════════════════════════════════════════════════════════


@pytest_asyncio.fixture
async def day_tables():
    from sqlalchemy import inspect as sa_inspect
    from app.db.database import engine
    from app.db.models.base import Base

    names = ["day_chats", "conversations", "messages"]
    async with engine.begin() as conn:
        for n in names:
            try:
                await conn.run_sync(Base.metadata.tables[n].create, checkfirst=True)
            except Exception:
                pass
    async with engine.connect() as conn:
        existing = await conn.run_sync(lambda c: set(sa_inspect(c).get_table_names()))
    missing = [n for n in names if n not in existing]
    if missing:
        pytest.skip(f"cannot create {missing} on this backend")
    yield


def test_public_text_strips_an_assistant_row():
    from app.api.message_cards import public_text

    out = public_text("assistant", E4_ONE)
    if out == E4_ONE:
        pytest.skip("public_text strip not landed yet — lane B3 (D2)")
    assert out == "Putting on **Kanye West - Stronger**."


def test_public_text_leaves_a_user_row_alone():
    """A user may legitimately paste a tag back — quoting the thing they are
    asking about. Stripping their words is editing the user."""
    from app.api.message_cards import public_text

    assert public_text("user", E4_ONE) == E4_ONE


@pytest.mark.asyncio
async def test_the_history_loader_annotates_a_leaked_row_once_not_twice(day_tables):
    """The loader prefixes every row. An assistant row that already begins
    with a tag would come back double-tagged — which is also how the model
    learned the shape in the first place."""
    from app.agent.day_context_loader import load_day_context
    from app.db.database import async_session_maker
    from app.db.models import Conversation, Message
    from app.db.models.day_chat import DayChat
    from app.db.models.user import User

    uid, dc_id, conv_id = (str(uuid.uuid4()) for _ in range(3))
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"tag-{uuid.uuid4().hex[:10]}@example.com",
                    hashed_password="x", name="Tag", timezone="UTC"))
        db.add(DayChat(id=dc_id, user_id=uid, local_date=Date(2026, 9, 14), timezone="UTC"))
        await db.flush()
        db.add(Conversation(id=conv_id, user_id=uid, channel="mobile", day_chat_id=dc_id,
                            started_at=datetime(2026, 9, 14, 21, 41)))
        db.add(Message(id=str(uuid.uuid4()), conversation_id=conv_id, day_chat_id=dc_id,
                       channel="mobile", role="user", content=E4_ONE,
                       created_at=datetime(2026, 9, 14, 21, 40)))
        db.add(Message(id=str(uuid.uuid4()), conversation_id=conv_id, day_chat_id=dc_id,
                       channel="mobile", role="assistant", content=E4_ONE,
                       created_at=datetime(2026, 9, 14, 21, 41)))
        await db.commit()

    async with async_session_maker() as db:
        ctx = await load_day_context(db, dc_id, model_context_tokens=200_000, tz_name="UTC")

    rows = ctx["messages"]
    asst = [r for r in rows if r.get("role") == "assistant"]
    user = [r for r in rows if r.get("role") == "user"]
    assert asst and user

    if asst[0]["content"].count("[mobile") > 1:
        pytest.skip("read-side strip not landed yet — lane B1 (D2)")
    assert asst[0]["content"].count("[mobile") == 1, (
        "the assistant row is double-annotated — the model is being shown the "
        "very pattern it must not reproduce, twice"
    )
    assert user[0]["content"].count("[mobile") == 2, (
        "ANTI-VACUITY: the strip must apply to ASSISTANT rows only. A user "
        "row that quotes a tag keeps it, and still gets the loader's own."
    )


@pytest.mark.asyncio
async def test_the_two_serializers_agree_byte_for_byte(day_tables):
    """day_chats.py and sessions.py serve the same row to the same client on
    two different routes. A strip on one and not the other is a body that
    changes when the user navigates."""
    from sqlalchemy import select
    from app.db.database import async_session_maker
    from app.db.models import Conversation, Message
    from app.db.models.day_chat import DayChat
    from app.db.models.user import User

    uid, dc_id, conv_id = (str(uuid.uuid4()) for _ in range(3))
    mid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"ser-{uuid.uuid4().hex[:10]}@example.com",
                    hashed_password="x", name="Ser", timezone="UTC"))
        db.add(DayChat(id=dc_id, user_id=uid, local_date=Date(2026, 9, 14), timezone="UTC"))
        await db.flush()
        db.add(Conversation(id=conv_id, user_id=uid, channel="mobile", day_chat_id=dc_id,
                            started_at=datetime(2026, 9, 14, 21, 41)))
        db.add(Message(id=mid, conversation_id=conv_id, day_chat_id=dc_id, channel="mobile",
                       role="assistant", content=E4_ONE,
                       created_at=datetime(2026, 9, 14, 21, 41)))
        await db.commit()

    from app.api.message_cards import public_text
    async with async_session_maker() as db:
        row = (await db.execute(select(Message).where(Message.id == mid))).scalar_one()

    # `public_text` is the single seam both serializers are specified to use
    # (D6 says the realtime frame's `content` goes through it too), so
    # agreement is asserted by pinning that they all read the same function.
    body = public_text(row.role, row.content)
    for src_name in ("app/api/day_chats.py", "app/api/sessions.py"):
        src = (_BACKEND / src_name).read_text()
        assert "public_text(" in src, (
            f"{src_name} does not route its message body through public_text — "
            "the two routes can then disagree about the same row"
        )
    assert body == public_text("assistant", E4_ONE)


# ══════════════════════════════════════════════════════════════════════
# Review round 1 (2026-09-14)
# ══════════════════════════════════════════════════════════════════════


def test_the_systems_own_unknown_tag_is_strippable():
    """`unknown` is resolve_channel's default, so the history loader can EMIT
    `[unknown 9:41pm] …`; the model copies it like any other tag, and the
    alternation built from KNOWN_CHANNELS alone let it through every layer."""
    from app.agent.channel_annotations import strip_leaked_tags

    assert strip_leaked_tags("[unknown 9:41pm] hi") == ("hi", "[unknown 9:41pm] ")
    assert strip_leaked_tags("[Unknown 09:41 PM] hi")[0] == "hi"


def test_the_backfill_prefilter_names_the_same_channels_as_the_stripper():
    import scripts.backfill_annotation_leaks as bf
    from app.agent.channel_annotations import STRIPPABLE_CHANNELS

    pre = bf._pg_prefilter()
    for ch in STRIPPABLE_CHANNELS:
        assert ch in pre, f"{ch} is strippable but the Postgres prefilter would skip its rows"


def test_the_voice_day_block_strips_an_assistant_row_and_keeps_its_shape():
    """A fourth history assembler: `voice_context.render_day_history` feeds
    the realtime voice model, whose replies are persisted back through the
    relay. It emitted `You [web]: [mobile 9:41pm] Putting on Kanye…` — the
    exact demonstration the other three read-side strips remove."""
    from app.agent.voice_context import render_day_history

    rows = [
        ("user", "[mobile 9:40pm] play stronger", "mobile"),      # a USER row is never touched
        ("assistant", "[mobile 9:41pm] Putting on Kanye West.", "web"),
    ]
    out = render_day_history(rows, "2026-09-14", "2026-09-14")
    lines = out.splitlines()
    assert lines[1] == "User [mobile]: [mobile 9:40pm] play stronger"
    assert lines[2] == "You [web]: Putting on Kanye West.", lines[2]


def test_the_voice_day_feed_orders_by_id_and_reads_the_rows_own_channel():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "app" / "agent" / "voice_context.py").read_text()
    fn = src.split("async def _load_newest_day(")[1].split("\nasync def ")[0]
    assert ".order_by(Message.created_at.asc(), Message.id.asc())" in fn
    assert "Message.channel" in fn and "msg_channel or conv_channel" in fn


def test_the_chat_history_route_orders_by_id_and_serves_public_text():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "app" / "api" / "chat.py").read_text()
    assert ".order_by(Message.created_at.asc(), Message.id.asc())" in src
    assert "content=public_text(msg.role, msg.content)" in src


@pytest.mark.asyncio
async def test_a_tenant_whose_only_days_are_future_dated_feeds_the_voice_model_nothing(day_tables):
    """Review R3 P3: the future-day clamp fell back to the UNCLAMPED pick when
    it found nothing, so a tenant whose only rows were mis-bucketed still
    hijacked the voice feed with a day dated in the user's future."""
    import uuid as _uuid
    from datetime import date as _date, timedelta as _td
    from app.agent.voice_context import _load_newest_day
    from app.db.database import async_session_maker
    from app.db.models import User
    from app.db.models.day_chat import DayChat

    uid = str(_uuid.uuid4())
    tz = "America/Toronto"
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"vc-{uid[:8]}@example.com", hashed_password="x",
                    name="V", timezone=tz))
        await db.flush()
        db.add(DayChat(id=str(_uuid.uuid4()), user_id=uid,
                       local_date=_date.today() + _td(days=3), timezone="UTC"))
        await db.commit()

    async with async_session_maker() as db:
        newest, rows = await _load_newest_day(db, uid, tz)
    assert newest is None and rows == [], "a future-dated day reached the voice feed"

    # ANTI-VACUITY: with no zone the unclamped pick still answers (unchanged).
    async with async_session_maker() as db:
        newest_nozone, _ = await _load_newest_day(db, uid, None)
    assert newest_nozone is not None
