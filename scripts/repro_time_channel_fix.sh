#!/bin/bash
# ============================================================================
# Manual repro for the time + channel + mobile-DaC coordinated fix.
#
# Exercises:
#   - Bug #1: Runtime Context renders LOCAL time, not UTC
#   - Bug #2: channel=<explicit> with guidance; unknown → warn log + "unknown"
#   - Bug #3: day_context_load failure is ERROR not WARNING; entry log fires
#             Mobile tz reaches the backend; Message.channel populated
#
# Usage:
#   # Against a running agent container (mrhx's container shown; adapt as needed):
#   AGENT_CONTAINER=toup-agent-c47c5b4b \
#   USER_ID=c47c5b4b-16d2-498c-b231-15d53d7454b9 \
#   bash backend/scripts/repro_time_channel_fix.sh
#
#   # Each section prints what it's checking. Grep the log tail shown after.
# ============================================================================
set -euo pipefail

AGENT_CONTAINER="${AGENT_CONTAINER:-toup-agent-c47c5b4b}"
USER_ID="${USER_ID:-c47c5b4b-16d2-498c-b231-15d53d7454b9}"

say() { printf '\n\033[1;35m══ %s ══\033[0m\n' "$*"; }
run() { echo "$ $*"; eval "$*"; }


# ─── 1. Migration applied: messages.channel column exists + indexed ──────
say "Migration: messages.channel column present + indexed"
run "docker exec $AGENT_CONTAINER python -c \"
import asyncio
from app.db.database import async_session_maker
from sqlalchemy import text
async def _():
    async with async_session_maker() as s:
        r = await s.execute(text('SELECT column_name, is_nullable FROM information_schema.columns WHERE table_name=\\'messages\\' AND column_name=\\'channel\\''))
        print('column:', r.fetchone())
        r = await s.execute(text('SELECT indexname FROM pg_indexes WHERE tablename=\\'messages\\' AND indexname=\\'ix_messages_channel\\''))
        print('index:', r.fetchone())
asyncio.run(_())
\""


# ─── 2. Runtime Context renders local time ────────────────────────────────
say "Runtime Context: local time + tz name + channel guidance"
echo "Send a message from the web UI, then:"
run "docker logs --since 2m $AGENT_CONTAINER 2>&1 | grep '\[agent\] tz_resolved' | tail -3"
echo
echo "Expected: lines like"
echo "  [agent] tz_resolved user=<uid8> channel=web tz=America/Toronto source=client local=2026-04-21 11:30 utc=2026-04-21 15:30"
echo
echo "If you see source=utc_default or source=user_profile, the client didn't send tz"
echo "(mobile pre-fix, Telegram/voice always) — user's profile tz is the fallback."


# ─── 3. Unknown channel warn path ─────────────────────────────────────────
say "Unknown-channel warn path (synthetic call into the runner)"
run "docker exec $AGENT_CONTAINER python -c \"
from app.agent.channel_util import resolve_channel
import logging
logging.basicConfig(level=logging.WARNING)
# Explicit empty, no hints — should return 'unknown' + warn
out = resolve_channel(site='repro_script')
print('resolved:', out)
assert out == 'unknown'
\""


# ─── 4. Day-context load: entry log fires ─────────────────────────────────
say "Day-context: entry log present on next message"
run "docker logs --since 2m $AGENT_CONTAINER 2>&1 | grep '\[day_ctx\] load_day_context entry' | tail -3"
echo
echo "Expected at least one line after the next user message. If absent,"
echo "either USE_DAY_CHAT_CONTEXT is off OR the day-chat loader short-circuited"
echo "before entry (no day_chat_id resolved)."


# ─── 5. enable_day_recall flipped on ──────────────────────────────────────
say "settings.enable_day_recall is True (post deploy-note)"
run "docker exec $AGENT_CONTAINER python -c \"from app.config import settings; print('enable_day_recall =', settings.enable_day_recall)\""
echo
echo "Must read 'True'. If False, set ENABLE_DAY_RECALL=true in"
echo "/data/agents/<user>/.env and restart the container."


# ─── 6. Message.channel populated on new inserts ──────────────────────────
say "Messages inserted this minute have channel populated"
run "docker exec $AGENT_CONTAINER python -c \"
import asyncio
from app.db.database import async_session_maker
from sqlalchemy import text
async def _():
    async with async_session_maker() as s:
        r = await s.execute(text('''
          SELECT COALESCE(channel, '<null>') AS channel, COUNT(*) AS n
          FROM messages
          WHERE created_at > NOW() - INTERVAL '10 minutes'
          GROUP BY 1 ORDER BY 2 DESC
        '''))
        for row in r.fetchall():
            print(row)
asyncio.run(_())
\""
echo
echo "Expected: last 10 minutes' rows have a non-null channel from all active channels."
echo "If <null> dominates, Message.channel stamping isn't reaching the insert path —"
echo "check agent_runner.py:1796/1837."


# ─── 7. Mobile tz arrives in payload ──────────────────────────────────────
say "Mobile WS payload carries tz (look for tz= in [WS] RECV lines)"
run "docker logs --since 5m $AGENT_CONTAINER 2>&1 | grep 'tz=' | head -5"
echo
echo "If mobile sends no tz, look for: tz=None or tz_fallback source=user_profile"
echo "(Mobile fix: app/src/shared/api.ts sendChatMessage now includes 'tz' field.)"


say "Done."
echo "Send a fresh web message + a fresh mobile message + a Telegram reply, then"
echo "re-run. All seven sections should produce the expected output. Any section"
echo "that fails points at the broken hop."
