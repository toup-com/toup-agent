"""Regression tests for PR-2 — cross-channel session continuity.

Audit context (docs/audits/2026-07-token-efficiency.md, findings F-4/F-5):

  * F-5 / A2-1: since commit ae6b218d (2026-04-13),
    ``_get_or_create_session`` referenced an out-of-scope ``client_tz`` —
    the swallowed NameError wrote ``day_chat_id=NULL`` on every
    runner-created Conversation, killing app-per-day reuse and letting
    system-channel rows evade the one-per-day partial unique index.
  * A2-2: session rollover compared UTC dates while DayChat rolls on the
    user's LOCAL date, so sessions (and the prompt_cache_key) re-minted
    mid-local-day.
  * A2-7: the Chrome extension sends its timezone as ``client_tz`` but
    the WS server only read ``tz``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.agent.agent_runner import same_local_day


def _utc(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


class TestSameLocalDay:
    def test_utc_midnight_crossing_is_same_local_day(self):
        # 7:58 PM and 8:02 PM in Toronto (EDT, UTC-4) straddle UTC
        # midnight — the old UTC comparison re-minted the session here.
        a = _utc(2026, 7, 23, 23, 58)  # 7:58 PM EDT July 23
        b = _utc(2026, 7, 24, 0, 2)    # 8:02 PM EDT July 23
        assert same_local_day(a, b, "America/Toronto") is True

    def test_local_midnight_rolls_even_when_utc_date_is_same(self):
        # 11:59 PM → 12:01 AM in Toronto is one UTC date (03:59→04:01)
        # but a new local day — the session must roll with DayChat.
        a = _utc(2026, 7, 24, 3, 59)
        b = _utc(2026, 7, 24, 4, 1)
        assert same_local_day(a, b, "America/Toronto") is False

    def test_no_tz_falls_back_to_utc(self):
        a = _utc(2026, 7, 23, 23, 58)
        b = _utc(2026, 7, 24, 0, 2)
        assert same_local_day(a, b, None) is False
        assert same_local_day(a, a, None) is True

    def test_invalid_tz_falls_back_to_utc(self):
        a = _utc(2026, 7, 23, 12, 0)
        b = _utc(2026, 7, 23, 13, 0)
        assert same_local_day(a, b, "Not/AZone") is True


# ---------------------------------------------------------------------------
# Wiring pins (source-grep style — the runner needs a full boot to run).
# ---------------------------------------------------------------------------

_RUNNER = (Path(__file__).resolve().parent.parent / "app" / "agent" / "agent_runner.py").read_text()
_WS = (Path(__file__).resolve().parent.parent / "app" / "api" / "ws_chat.py").read_text()


class TestSessionTzWiring:
    def test_get_or_create_session_declares_client_tz(self):
        """F-5: the body reads client_tz for DayChat resolution — without
        the parameter it NameErrors inside a swallowed try/except and
        every runner-created Conversation gets day_chat_id=NULL."""
        idx = _RUNNER.index("async def _get_or_create_session(")
        sig = _RUNNER[idx:_RUNNER.index("):", idx) + 2]
        assert "client_tz" in sig

    def test_run_passes_client_tz_to_session_resolution(self):
        call = _RUNNER.index("await self._get_or_create_session(")
        block = _RUNNER[call:call + 300]
        assert "client_tz=client_tz" in block

    def test_tz_resolved_before_session_resolution(self):
        """PR-2 moved the tz seed ahead of session resolution so the
        rollover comparison and day_chat stamping both see the user's
        real timezone (channels like Telegram never send one)."""
        resolve_idx = _RUNNER.index("await self._resolve_effective_tz(")
        session_idx = _RUNNER.index("await self._get_or_create_session(")
        assert resolve_idx < session_idx

    def test_rollover_uses_local_day_helper(self):
        assert "if not same_local_day(started, now_utc, client_tz):" in _RUNNER

    def test_ws_chat_accepts_both_tz_keys(self):
        """A2-7: extension sidepanel sends client_tz; web/mobile send tz."""
        assert 'msg.get("tz") or msg.get("client_tz")' in _WS


class TestSystemChannelUniqueIndex:
    """Regression for the verification-phase STOP finding: PR-2's
    day_chat_id stamping re-armed the partial unique index
    ix_conversations_system_channel_per_day, which the runner's blind
    insert would then violate on a user's 2nd same-day routine.
    """

    def test_runner_routes_system_channels_through_resolver(self):
        """The blind Conversation insert must NOT be reachable for
        indexed system channels — they route through the canonical
        resolver (reuse + IntegrityError recovery) instead."""
        seg = _RUNNER[_RUNNER.index("async def _get_or_create_session("):]
        seg = seg[:seg.index("async def _build_system_prompt(")]
        assert "resolve_or_create_day_conversation" in seg
        assert '_INDEXED_SYSTEM_CHANNELS = ("routine", "trigger", "api", "digest")' in seg
        # the guarded return must appear before the blind insert
        assert seg.index("return conv, False") < seg.index("session = Conversation(")

    def test_partial_index_collides_but_select_first_avoids_it(self):
        """DB-level proof (raw sqlite, the same partial index init_db
        installs): a blind 2nd insert of a same-day (user, day, 'routine')
        active row COLLIDES — this is what the runner did before the fix
        once PR-2 stamped a non-NULL day_chat_id. A SELECT-first-then-reuse
        (what resolve_or_create_day_conversation does, and what the runner
        now routes system channels through) avoids the collision.
        Control: NULL day_chat_id (the pre-PR2 NameError state) evades the
        index — which is why main never crashed."""
        import sqlite3
        c = sqlite3.connect(":memory:")
        c.execute("CREATE TABLE conversations (id TEXT PRIMARY KEY, user_id TEXT, "
                  "day_chat_id TEXT, channel TEXT, is_active INTEGER)")
        c.execute("CREATE UNIQUE INDEX ix_conversations_system_channel_per_day "
                  "ON conversations (user_id, day_chat_id, channel) "
                  "WHERE channel IN ('routine','trigger','api','digest') AND is_active = 1")
        c.execute("INSERT INTO conversations VALUES ('c1','u1','d1','routine',1)")
        # blind 2nd insert (pre-fix runner path) -> collision
        import pytest
        with pytest.raises(sqlite3.IntegrityError):
            c.execute("INSERT INTO conversations VALUES ('c2','u1','d1','routine',1)")
        # SELECT-first (resolver / fixed runner path) -> reuse, no 2nd row
        row = c.execute("SELECT id FROM conversations WHERE user_id='u1' AND "
                        "day_chat_id='d1' AND channel='routine' AND is_active=1").fetchone()
        assert row[0] == "c1"
        # control: NULL day_chat_id evades the index (pre-PR2 NameError state)
        c.execute("INSERT INTO conversations VALUES ('c3','u1',NULL,'routine',1)")
        c.execute("INSERT INTO conversations VALUES ('c4','u1',NULL,'routine',1)")  # no raise
