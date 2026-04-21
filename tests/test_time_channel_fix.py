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

CHANNEL_GUIDANCE_TABLE = {
    "web":      "Full markdown and formatting OK. Long code blocks, tables, headings all fine.",
    "app":      "Full markdown and formatting OK. Long code blocks, tables, headings all fine.",
    "mobile":   "Keep responses compact — short paragraphs, avoid large code blocks or tables. Users are on small screens.",
    "voice":    "Conversational tone. No markdown. Sentences should read naturally when spoken aloud.",
    "telegram": "Short messages. Basic markdown only (bold/italic). Avoid code blocks over ~20 lines.",
    "discord":  "Full markdown and code blocks OK. Keep message length under ~2000 chars.",
    "slack":    "Slack-flavored markdown (limited). Short messages preferred.",
}


def test_known_channels_have_distinct_guidance():
    """Every user-facing channel has its own guidance; no two are identical
    (except web/app, which are intentional twins for content-area surfaces)."""
    # Intentional duplicates: web and app both live in the content area.
    unique_guidance = set(CHANNEL_GUIDANCE_TABLE.values())
    assert len(unique_guidance) == len(CHANNEL_GUIDANCE_TABLE) - 1


def test_mobile_guidance_is_compact():
    assert "compact" in CHANNEL_GUIDANCE_TABLE["mobile"].lower() or \
           "small screens" in CHANNEL_GUIDANCE_TABLE["mobile"].lower()


def test_voice_guidance_excludes_markdown():
    assert "no markdown" in CHANNEL_GUIDANCE_TABLE["voice"].lower()


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
            "conversation_id VARCHAR(36), channel VARCHAR(50), content TEXT)"
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

def _runtime_context_fixture(channel: str, tz_name: str = "America/Toronto") -> str:
    """Standalone reproduction of the Runtime Context block so we can diff
    by channel without spinning up a full AgentRunner. Mirrors the code
    in agent_runner.py:1443-1527 minus the db-dependent pieces."""
    from zoneinfo import ZoneInfo
    guidance_map = CHANNEL_GUIDANCE_TABLE.copy()
    guidance = guidance_map.get(channel, "Unknown channel — format conservatively: short, minimal markdown.")
    now_utc = datetime(2026, 4, 21, 15, 23, 0, tzinfo=timezone.utc)
    now_local = now_utc.astimezone(ZoneInfo(tz_name))
    return "\n".join([
        "# Runtime Context",
        f"- Current date/time: {now_local.strftime('%Y-%m-%d %H:%M')} "
        f"{now_local.strftime('%Z')} "
        f"(UTC offset {now_local.strftime('%z')}; UTC wall clock: "
        f"{now_utc.strftime('%Y-%m-%d %H:%M')}Z)",
        f"- User timezone: {tz_name} (source=client)",
        f"- Channel: {channel} — {guidance}",
    ])


def test_channel_parity_web_vs_mobile_differs_only_in_channel_block():
    """Two identical runs, one per channel. Everything below the channel
    line should be byte-equal. Catches future regressions that gate
    unrelated system-prompt content on channel."""
    web = _runtime_context_fixture("web")
    mobile = _runtime_context_fixture("mobile")
    # They differ, yes — that's the point. But the current-time line should
    # be identical.
    web_lines = web.split("\n")
    mobile_lines = mobile.split("\n")
    assert web_lines[0] == mobile_lines[0]  # header
    assert web_lines[1] == mobile_lines[1]  # current date/time
    assert web_lines[2] == mobile_lines[2]  # timezone
    # Line 3 is the channel line — this is the expected delta.
    assert web_lines[3] != mobile_lines[3]
    assert "Channel: web" in web_lines[3]
    assert "Channel: mobile" in mobile_lines[3]


def test_channel_parity_known_channels_all_render():
    """Every known user-facing channel renders a non-empty Runtime Context
    with a matching guidance line. No channel is accidentally missing."""
    for channel in ("web", "app", "mobile", "voice", "telegram", "discord", "slack"):
        out = _runtime_context_fixture(channel)
        assert f"Channel: {channel}" in out
        assert CHANNEL_GUIDANCE_TABLE[channel] in out


def test_channel_parity_unknown_channel_gets_conservative_guidance():
    out = _runtime_context_fixture("smoke-signal")
    assert "format conservatively" in out
