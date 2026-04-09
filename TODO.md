# Day-as-Chat follow-ups (post-ship)

## Mobile port
Port the Day-as-Chat UI changes to the React Native app at
/Users/nariman/toup-platform-app. Same changes as the web frontend:
- Sidebar shows DayChats with channel badges (shared/api.ts + drawer)
- Chat screen loads messages via /api/day-chats/{date}/messages with fallback
- Channel badges on every message bubble
- Session dividers where conversation_id changes
- New Thread button creates session inside day without resetting context
- tz field on every WS message via Intl.DateTimeFormat().resolvedOptions().timeZone
- Feature flag fallback: try new endpoint, fall back to old if empty/error
Do after web has been in use for at least a few days to confirm UX is right.

## /browser chat toolbar bug
The chat panel inside /browser is missing toolbar controls that the main
chat view has (scroll-to-bottom, copy, regenerate). Likely either a prop
that hides the toolbar or a separate BrowserChatPanel component that was
never updated. Small frontend-only fix.

## Fixed 2026-04-08: Environment variable persistence

The `USE_DAY_CHAT_CONTEXT` env var was being dropped whenever containers were
recreated via multi-line SSH `docker run -e` commands. pydantic-settings defaulted
to False, so the day-chat context path never fired in production despite the
architectural plumbing being correct.

Fix: env vars are now stored in `/data/agents/{user_id}/.env` on the VPS and
loaded via `docker run --env-file`. This persists across recreations and doesn't
rely on shell escaping of multi-line command arguments.

Lesson: never trust env vars to survive container recreation through a multi-line
SSH command. Always use --env-file for anything that must persist.

## Fixed 2026-04-08: ContextBudgetLog datetime mismatch

The telemetry INSERT used `datetime.now(timezone.utc)` (tz-aware) but the DB
column stores naive UTC. asyncpg crashed with "can't subtract offset-naive and
offset-aware". Fixed by using `datetime.utcnow()`. Same class of bug as Bug 4
(naive/tz-aware mismatch). Total: 5 instances of this bug discovered during ship.

## Automate container recreation

Manually reconstructing `docker run` flags from `docker inspect` output is
error-prone and lost the feature flag twice during the Day-as-Chat ship.
Write a `deploy.sh` script that reads `/data/agents/{id}/.env`, stops, removes,
and recreates the container with `--env-file`. Idempotent, safe to run repeatedly.
Put it in `scripts/` in the toup-agent repo.

## Real Postgres integration tests
Three production bugs shipped because SQLite tests don't exercise paths
that fire on real Postgres with production-like data. Build an integration
test suite that runs against a real Postgres instance with a snapshot of
production data, exercising init_db() end-to-end, ALTER TABLE idempotency,
runtime assertions, migration backfill, and cross-channel context loading.
See commits 5d9aea9 and surrounding hotfixes for the incident that motivated
this. Should run in CI before every deploy.

## Auto-deploy pipeline for VPS agent containers
Currently, deploying code to agent containers requires manual SSH + git pull +
docker build + docker run with explicit env vars. The USE_DAY_CHAT_CONTEXT flag
is passed via docker run -e and gets lost on every manual rebuild that forgets it.
Build a GitHub Actions workflow that on push to toup-agent main:
1. SSHs to the VPS
2. Pulls latest code
3. Rebuilds the Docker image
4. Recreates containers preserving all env vars (store them in a docker-compose.yml
   or .env file on the VPS, not in the docker run command)
5. Verifies health check passes

## Remove old fallback paths — after 2026-04-15
After one week of USE_DAY_CHAT_CONTEXT=true being stable in production with
no rollbacks, delete the legacy session-based code paths:
- agent_runner.py: old _load_history path (keep only day-scoped)
- HubPageV2.tsx: fallback to fetchSessions (keep only day-chat mode)
- ChatPage.tsx: fallback to /api/sessions/by-date (keep only day-chats endpoint)
- day_chat_resolver.py: should_use_day_chat_context() feature flag gate
- config.py: use_day_chat_context setting
Removes ~400 lines of fallback code and eliminates the need for the flag.
