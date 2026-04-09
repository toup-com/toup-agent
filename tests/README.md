# Tests

## Current Test Suite

All tests use in-memory SQLite for speed and portability. Run with:

```bash
cd backend
python -m pytest tests/ -v
# or individually:
python tests/test_day_chat_resolver.py
python tests/test_tool_result_elision.py
python tests/test_backfill_day_chats.py
python tests/test_day_context.py
python tests/test_ws_tz_persistence.py
python tests/test_day_chats_api.py
python tests/test_active_task.py
python tests/test_table_partitioning.py
```

## TODO: Integration Test Suite Against Real Postgres

Add an integration test suite that runs against real Postgres with a snapshot
of production-like data, exercising `init_db()` end-to-end.

Three production bugs shipped because SQLite tests don't exercise these paths:

1. **Runtime assertion on monolith DBs** — `init_db()` crashed on boot because
   agent DBs contain platform-only tables (legacy monolith). The runtime assertion
   (`raise RuntimeError`) fired in production but never in SQLite tests.
   (See commit `0204c5a`, hotfix `5d9aea9` in toup-agent)

2. **Undefined `logger` in `agent_main.py`** — backfill scheduling code used
   `logger.info()` but `agent_main.py` uses `print()`. NameError crashed startup.
   (See commit `9fea30d`)

3. **`stripe_customer_id` / `setup_type` in wrong ALTER partition** — columns
   existed in the ORM model but the ALTER was gated to platform-only mode.
   Agent DB queries crashed with `UndefinedColumnError`.
   (See commits `4e55ae3`, `b898286`)

4. **Naive vs tz-aware datetime mismatch** — `day_chat_resolver` used
   `datetime.now(timezone.utc)` (tz-aware) but DB columns store naive UTC.
   asyncpg crashed on insert.
   (See commit `5bda52f`)

All four bugs would have been caught by running `init_db()` + a sample query
against a real Postgres with all tables present.

**Proposed approach:** Docker Compose with a Postgres service, seed script that
creates the same tables a production agent DB has, then run `init_db()` and
verify no crashes. Add to CI.

---

# Test retrospective — Day-as-Chat ship (2026-04-08)

The Day-as-Chat refactor shipped with 57+ passing unit tests across 8 test
files, all running against SQLite. Despite this, four production bugs made
it to deploy:

1. Runtime assertion in init_db() crashed monolith-style agent DBs that
   contain both platform and agent tables. SQLite tests used clean partitioned
   schemas and never triggered it. Fixed in 5d9aea9 (downgraded to warning).

2. agent_main.py backfill code used logger.info() but no logger was defined
   at module level. NameError crashed startup. SQLite tests don't import
   agent_main.py. Fixed by using print().

3. users.stripe_customer_id ALTER TABLE was in the platform-only block, but
   the User ORM model references it and users is a shared table. Agent
   queries crashed with UndefinedColumnError. SQLite tests don't use the
   platform/agent partition system the same way. Fixed by moving ALTER to
   the shared block.

4. Telegram long-lived Conversation sessions span multiple days, but the
   original implementation read day_chat_id from the Conversation row.
   SQLite tests used short-lived sessions that didn't cross day boundaries.
   Fixed via Option C refactor: Message.day_chat_id resolved at save time
   from current local date, Conversation.day_chat_id demoted to a hint.

## Testing gap to close

SQLite unit tests are necessary but not sufficient. They don't catch:
- Real Postgres-only syntax (AT TIME ZONE, INSERT ON CONFLICT nuances)
- Schema ALTER idempotency on half-migrated DBs
- Long-lived sessions that span day/month/year boundaries
- Runtime assertions on production-shaped data
- Module-level import errors that tests don't import

## TODO: Integration test suite

Build a test harness that:
- Spins up a real Postgres container
- Loads a snapshot of production-like data (anonymized)
- Runs init_db() from a fresh state AND from a half-migrated state
- Runs the full backfill idempotently
- Exercises Telegram/cross-day scenarios specifically
- Runs a synthetic cross-channel end-to-end turn
- Runs in CI before any backend deploy

Until this exists, manual verification on production (like the one that
caught these four bugs) is the only safety net.
