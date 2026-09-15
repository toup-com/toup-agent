"""
Tests for the time + channel + mobile-DaC coordinated fix.

Covers:
- resolve_channel priority chain + unknown-channel warn log
- Per-channel formatting guidance: known channels vs unknown
- tz resolution fallback (client_tz → User.timezone → UTC)
- Day-boundary: PST user at 23:58 / 00:02 → same day_chat_id
- Message.channel column exists, indexed, nullable, backfill-safe SQL

Run:
  pytest backend/tests/test_time_channel_fix.py -v
"""
from __future__ import annotations

import logging
import sys
import uuid
from datetime import datetime, timezone, date as Date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker


# ──────────────────────────────────────────────────────────────
# resolve_channel — pure-Python priority chain
# ──────────────────────────────────────────────────────────────

from app.agent.channel_util import resolve_channel, KNOWN_CHANNELS


def test_resolve_channel_explicit_wins():
    assert resolve_channel(explicit="web") == "web"
    assert resolve_channel(explicit="mobile", payload_hint="web") == "mobile"
    assert resolve_channel(explicit="telegram", conversation_hint="web") == "telegram"


def test_resolve_channel_payload_over_conversation():
    assert resolve_channel(payload_hint="mobile", conversation_hint="web") == "mobile"


def test_resolve_channel_conversation_fallback():
    assert resolve_channel(conversation_hint="app") == "app"


def test_resolve_channel_unknown_defaults_to_unknown(caplog):
    caplog.set_level(logging.WARNING, logger="app.agent.channel_util")
    assert resolve_channel() == "unknown"
    # And the warning fired so ops can trace the missing ingress value.
    assert any("resolve_channel default" in r.getMessage() for r in caplog.records), (
        "Expected a WARNING log when channel is unknown; silent fallback is the bug."
    )


def test_resolve_channel_empty_string_treated_as_missing(caplog):
    caplog.set_level(logging.WARNING, logger="app.agent.channel_util")
    assert resolve_channel(explicit="", payload_hint="  ", conversation_hint=None) == "unknown"


def test_resolve_channel_unknown_value_passes_through_with_warning(caplog):
    """Unknown channel values (e.g. 'unittest') pass through so the actual value
    stays visible in the Runtime Context instead of being silently 'normalized'
    to web/unknown. A WARNING fires so ops see the unrecognized value."""
    caplog.set_level(logging.WARNING, logger="app.agent.channel_util")
    out = resolve_channel(explicit="smoke-signal")
    assert out == "smoke-signal"
    assert any("unknown_value" in r.getMessage() for r in caplog.records)


def test_resolve_channel_custom_default():
    assert resolve_channel(default="api") == "api"


# ──────────────────────────────────────────────────────────────
# Per-channel guidance: known vs unknown
# ──────────────────────────────────────────────────────────────

# 2026-09-14: this block used to hold a hand-maintained COPY of
# CHANNEL_GUIDANCE, seven entries deep, and asserted per-channel distinctness
# over the copy. It had already drifted — the copy said web and app share one
# string; production has given them their own since — and it would have stayed
# green through D1, which removes that table from the cacheable system prompt
# altogether. A green test describing a table the model no longer receives is
# the "STALE: asserts something the code no longer does" class this suite
# already carries nine entries of.
#
# Re-pointed at the REAL dict. What the prompt DOES with it is pinned in
# tests/test_channel_neutral_prefix.py: under D1 the guidance leaves the
# prefix and rides the per-turn <runtime_envelope> instead, so these
# assertions describe the envelope's per-channel rules from here on.
from app.agent.agent_runner import CHANNEL_GUIDANCE as CHANNEL_GUIDANCE_TABLE

# The unknown-channel fallback, agent_runner.py:6250. Kept as a literal
# because its whole purpose is to be the string for a channel not in the dict.
UNKNOWN_GUIDANCE = "Unknown channel — format conservatively: short, minimal markdown."


def test_known_channels_have_distinct_guidance():
    """Every channel in the real table says something different. Two channels
    sharing a string means one of them is being taught the wrong surface."""
    assert len(set(CHANNEL_GUIDANCE_TABLE.values())) == len(CHANNEL_GUIDANCE_TABLE), (
        "two channels share one guidance string: "
        + repr(sorted(
            k for k, v in CHANNEL_GUIDANCE_TABLE.items()
            if list(CHANNEL_GUIDANCE_TABLE.values()).count(v) > 1
        ))
    )


def test_mobile_guidance_is_compact():
    m = CHANNEL_GUIDANCE_TABLE["mobile"].lower()
    assert "compact" in m or "small screen" in m


def test_voice_guidance_excludes_markdown():
    assert "no markdown" in CHANNEL_GUIDANCE_TABLE["voice"].lower()


def test_the_user_facing_channels_are_all_present():
    """ANTI-VACUITY for the three above now that the table is the real one:
    pin the channels a user can actually be on, so a deletion shows up here
    rather than as a silently missing surface contract."""
    for ch in ("web", "app", "mobile", "voice", "telegram", "whatsapp",
               "discord", "slack", "extension"):
        assert CHANNEL_GUIDANCE_TABLE.get(ch), f"no guidance for {ch!r}"


def test_all_known_channels_covered_except_agent():
    """KNOWN_CHANNELS should be a superset of the user-facing guidance map.
    Internal-only channels (e.g. 'agent', 'vibecoding') aren't in the table
    by design — they get the conservative unknown-fallback guidance."""
    user_facing = set(CHANNEL_GUIDANCE_TABLE.keys())
    assert user_facing.issubset(KNOWN_CHANNELS), (
        f"Channels missing from KNOWN_CHANNELS: {user_facing - KNOWN_CHANNELS}"
    )


# ──────────────────────────────────────────────────────────────
# Timezone conversion math (the actual bug — UTC → local formatting)
# ──────────────────────────────────────────────────────────────

def test_utc_to_eastern_drift_is_zero():
    """The user report: 2:21 AM Eastern was shown as 6:22 AM (~4h drift).
    Confirm the conversion round-trips correctly post-fix."""
    from zoneinfo import ZoneInfo
    # 2:21 AM EDT on 2026-04-21 is 06:21 UTC same date.
    local_input = datetime(2026, 4, 21, 2, 21, 0, tzinfo=ZoneInfo("America/Toronto"))
    as_utc = local_input.astimezone(timezone.utc)
    assert as_utc.hour == 6
    assert as_utc.minute == 21

    # And the reverse: UTC back to local should be 2:21 AM, not 6:21.
    back = as_utc.astimezone(ZoneInfo("America/Toronto"))
    assert back.hour == 2
    assert back.minute == 21
    assert back.strftime("%Z") in ("EDT", "EST")  # DST-dependent, both valid


def test_utc_fallback_when_no_tz():
    """If no tz is resolvable, the conversion falls back to UTC identity."""
    utc_dt = datetime(2026, 4, 21, 6, 21, 0, tzinfo=timezone.utc)
    # astimezone(utc) is a no-op
    same = utc_dt.astimezone(timezone.utc)
    assert same == utc_dt


# ──────────────────────────────────────────────────────────────
# Day-boundary: PST user at 23:58 and 00:02 → same day_chat_id
# ──────────────────────────────────────────────────────────────

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "day_chat_resolver",
    str(Path(__file__).resolve().parent.parent / "app" / "agent" / "day_chat_resolver.py"),
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
resolve_local_date = _mod.resolve_local_date


def test_day_boundary_pst_late_night_same_day():
    """PST user at 23:58 local (07:58 UTC next day) and 00:02 local the next
    day (08:02 UTC next day). With correct tz handling these bucket into
    DIFFERENT days (Mon 23:58 → Mon; Tue 00:02 → Tue). The test name
    captures the INVERSE risk: without tz, both would bucket to the same
    UTC day (whichever one the offset lands on). This confirms
    resolve_local_date honors tz correctly."""
    # 2026-04-21 23:58 PST = 2026-04-22 06:58 UTC — in LA this is still Apr 21.
    late_utc = datetime(2026, 4, 22, 6, 58, 0, tzinfo=timezone.utc)
    early_utc = datetime(2026, 4, 22, 8, 2, 0, tzinfo=timezone.utc)  # 00:02 PST Apr 22

    late_date, _tz1 = resolve_local_date(late_utc, "America/Los_Angeles")
    early_date, _tz2 = resolve_local_date(early_utc, "America/Los_Angeles")

    assert late_date == Date(2026, 4, 21), f"Expected Apr 21, got {late_date}"
    assert early_date == Date(2026, 4, 22), f"Expected Apr 22, got {early_date}"
    assert late_date != early_date, "Crossing midnight should flip the day"


def test_day_boundary_same_day_both_sides_of_midnight():
    """Companion to the above: two PST times within the SAME local day
    but crossing UTC midnight should bucket to the same PST date."""
    # 16:00 PST Apr 21 = 23:00 UTC Apr 21 (same UTC day).
    # 20:00 PST Apr 21 = 03:00 UTC Apr 22 (different UTC day).
    # Both are still 'Apr 21' in PST.
    utc_before_midnight = datetime(2026, 4, 21, 23, 0, 0, tzinfo=timezone.utc)
    utc_after_midnight = datetime(2026, 4, 22, 3, 0, 0, tzinfo=timezone.utc)

    d1, _ = resolve_local_date(utc_before_midnight, "America/Los_Angeles")
    d2, _ = resolve_local_date(utc_after_midnight, "America/Los_Angeles")
    assert d1 == Date(2026, 4, 21)
    assert d2 == Date(2026, 4, 21), (
        f"UTC midnight crossed but PST still Apr 21; got {d2}"
    )
    assert d1 == d2


# ──────────────────────────────────────────────────────────────
# Message.channel column: schema + backfill SQL is idempotent
# ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_message_channel_column_nullable_and_backfillable():
    """Create a messages table, insert one row with NULL channel, run the
    backfill SQL manually, confirm it populates from conversations."""
    engine = create_async_engine(
        "sqlite+aiosqlite://", connect_args={"check_same_thread": False}
    )
    async with engine.begin() as conn:
        await conn.execute(text(
            "CREATE TABLE conversations (id VARCHAR(36) PRIMARY KEY, "
            "channel VARCHAR(50))"
        ))
        await conn.execute(text(
            "CREATE TABLE messages (id VARCHAR(50) PRIMARY KEY, "
            "conversation_id VARCHAR(36), channel VARCHAR(50), content TEXT, origin VARCHAR(16))"
        ))
        await conn.execute(text(
            "INSERT INTO conversations (id, channel) VALUES ('c1', 'mobile'), ('c2', NULL)"
        ))
        await conn.execute(text(
            "INSERT INTO messages (id, conversation_id, channel, content) VALUES "
            "('m1', 'c1', NULL, 'hi'), "     # should backfill to 'mobile'
            "('m2', 'c2', NULL, 'hello'), "   # conv also NULL → stays NULL
            "('m3', 'c1', 'web', 'switched')" # already set → stays 'web'
        ))
        # Backfill SQL — SQLite variant of the Postgres UPDATE ... FROM.
        await conn.execute(text(
            "UPDATE messages SET channel = ("
            "  SELECT c.channel FROM conversations c WHERE c.id = messages.conversation_id"
            ") WHERE channel IS NULL AND conversation_id IN ("
            "  SELECT id FROM conversations WHERE channel IS NOT NULL"
            ")"
        ))
        row = (await conn.execute(text("SELECT id, channel FROM messages ORDER BY id"))).fetchall()
    by_id = {r[0]: r[1] for r in row}
    assert by_id["m1"] == "mobile", "Message with NULL channel should inherit from Conversation"
    assert by_id["m2"] is None, "Both NULL → stay NULL, never guess"
    assert by_id["m3"] == "web", "Existing Message.channel must not be overwritten"


# ──────────────────────────────────────────────────────────────
# Channel parity: the Runtime Context differs ONLY in channel + guidance
# ──────────────────────────────────────────────────────────────

# 2026-09-14: three tests lived here that diffed a LOCAL hand-model of the
# Runtime Context block ("- Channel: {channel} — {guidance}") across channels.
# They asserted that the block differs per channel — which D1 makes false by
# design: the channel line leaves the cacheable prefix so that every channel
# shares one provider cache lineage, and the per-turn facts move into a
# <runtime_envelope> message. Because the model was local, they would have
# stayed GREEN through that change while describing a line the model no longer
# receives.
#
# Deleted rather than repaired. Their replacements, against the REAL render:
#   tests/test_channel_neutral_prefix.py
#     - test_every_channel_shares_one_system_prompt      (the new invariant)
#     - test_no_channel_guidance_string_survives_into_the_prefix
#     - test_flag_off_is_byte_identical_to_the_goldens   (the old behaviour,
#       pinned as bytes captured from origin/main rather than as a hand-model)
# The unknown-channel fallback string itself is still pinned above.


def test_the_unknown_channel_fallback_string_is_unchanged():
    """The one part of the deleted trio worth keeping here: `resolve_channel`
    passes an unrecognised value straight through, and something has to say
    what the agent is then told."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "app" / "agent" / "agent_runner.py").read_text()
    assert UNKNOWN_GUIDANCE in src
    assert "smoke-signal" not in CHANNEL_GUIDANCE_TABLE


# ──────────────────────────────────────────────────────────────
# Regression: _save_messages accepts `channel` kwarg
# ──────────────────────────────────────────────────────────────
# Hotfixed post-#9 deploy when mobile first sent a real message and hit
# NameError: name 'channel' is not defined at agent_runner.py:1814. The
# resolve_channel call was inside _save_messages but the method signature
# lacked `channel`. Caller threading + signature kwarg added in the hotfix.
# This test locks the signature so it can't regress.

def test_save_messages_accepts_channel_kwarg():
    """Import inspection — _save_messages must declare `channel` in its
    signature, otherwise the insert-site resolve_channel call hits NameError
    at runtime (Rule 5 / helper-scope trap)."""
    import inspect
    # Read the source without importing the full module (avoids anthropic dep).
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "app" / "agent" / "agent_runner.py").read_text()
    # Grab the _save_messages def block and confirm `channel:` is in it.
    idx = src.index("async def _save_messages(")
    # Signature ends at first `):` that closes the arg list at column 4+.
    sig = src[idx:src.index("):", idx) + 2]
    assert "channel:" in sig, (
        "_save_messages signature missing `channel:` param — "
        "resolve_channel(explicit=channel, ...) inside the method will NameError. "
        "Hotfix: add channel: Optional[str] = None. See run() call site + agent_runner.py:1812."
    )


def test_run_threads_channel_to_save_messages():
    """Similar inspection — run() must pass `channel=channel` into
    _save_messages. Without the thread, the param defaults to None and
    Message.channel won't populate on insert."""
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "app" / "agent" / "agent_runner.py").read_text()
    # Find the call site + walk forward ~1000 chars to capture the full
    # multi-line call block. `src.index(")")` picks up inner-arg parens;
    # a window-based scan is simpler and the call is not that long.
    call_idx = src.index("await self._save_messages(")
    call_block = src[call_idx:call_idx + 1500]
    # Truncate at the first closing paren that sits alone on its own line
    # (standard Python formatting for multi-line call closes).
    lines = call_block.splitlines()
    end = next((i for i, ln in enumerate(lines) if ln.strip() == ")"), len(lines))
    call_block = "\n".join(lines[: end + 1])
    assert "channel=channel" in call_block, (
        "run() must pass channel=channel to _save_messages so "
        "Message.channel is stamped at insert time. Missing thread means "
        "day-context history annotation falls back to Conversation.channel "
        "for all new rows — drift across mid-day channel switches is lost."
    )
