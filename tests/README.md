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
