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

## ContextBudgetLog telemetry not writing
The context_budget_logs table exists but has 0 rows despite the day-chat
context path being active. The write in agent_runner.py is wrapped in
try/except and silently failing. Debug and fix — the telemetry data is
needed to tune token budgets and monitor summary staleness.

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
