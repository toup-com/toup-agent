"""083 — persist cache_write_tokens: the last unstored cache number (G-11).

The value was extracted on both wires, priced into cost_cents and the
credit charge — and then dropped at the persistence boundary, recoverable
only from [CACHE] log lines (which rotate). Part of the gpt-5.6 family
bills cache WRITES at a premium, so this is a money column.

Same test shape as test_migration_080/test_channel_attribution_telemetry:
model column pins + platform-gate pins on the migration file + the writer
actually stores the value + the dashboard reports it.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
MIGRATION = (
    BACKEND / "alembic" / "versions"
    / "20260809_0083_083_llm_proxy_events_cache_write_tokens.py"
)


# ── model ────────────────────────────────────────────────────────────────

def test_the_model_carries_a_nullable_cache_write_column():
    from app.db.models.platform import LLMProxyEvent

    col = LLMProxyEvent.__table__.columns["cache_write_tokens"]
    assert col.nullable is True, (
        "pre-083 rows never measured this; NULL is honest, a default would "
        "invent a measurement"
    )


def test_an_event_can_be_built_without_cache_write_tokens():
    from app.db.models.platform import LLMProxyEvent

    ev = LLMProxyEvent(
        id="x", user_id="u", provider="openai", model="m", endpoint="chat",
        input_tokens=1, output_tokens=1, cost_cents=0, latency_ms=1,
    )
    assert getattr(ev, "cache_write_tokens", None) is None


# ── migration file ───────────────────────────────────────────────────────

def test_migration_083_exists_and_chains_082():
    src = MIGRATION.read_text()
    assert 'revision = "083"' in src
    assert 'down_revision = "082"' in src


def test_migration_083_is_platform_gated_and_idempotent():
    """llm_proxy_events is PLATFORM_ONLY and alembic runs on boot in BOTH
    images; an ungated ADD COLUMN would die on every tenant container."""
    src = MIGRATION.read_text()
    assert "_is_platform_db()" in src
    assert "get_table_names()" in src, "must no-op when the table is absent"
    assert re.search(r"if _COLUMN not in cols", src), "must be re-runnable"
    assert "nullable=True" in src
    assert "server_default" not in src, "no default — NULL means unmeasured"


# ── the writer stores it ─────────────────────────────────────────────────

def test_log_event_persists_the_value_it_already_receives():
    """The defect was precisely here: _log_event took cache_write_tokens,
    priced it, and did not put it on the row."""
    from app.api import llm_proxy

    src = inspect.getsource(llm_proxy._log_event)
    # The constructor spans lines and contains nested parens
    # (str(uuid.uuid4())); bound the slice by the db.add that follows it.
    ctor = src.split("LLMProxyEvent(")[1].split("db.add(event)")[0]
    assert "cache_write_tokens=cache_write_tokens" in ctor, (
        "_log_event prices cache_write_tokens but drops it at the "
        "persistence boundary again"
    )


# ── the dashboard reports it ─────────────────────────────────────────────

def test_cache_daily_row_carries_write_volume():
    from app.api.llm_proxy import CacheDailyRow

    assert "cache_write_tokens" in CacheDailyRow.model_fields


def test_cache_daily_query_sums_the_column_with_null_as_zero():
    from app.api import llm_proxy

    src = inspect.getsource(llm_proxy.get_admin_cache_daily)
    assert re.search(
        r"coalesce\(func\.sum\(LLMProxyEvent\.cache_write_tokens\), 0\)", src
    ), "pre-083 NULLs must aggregate as 0, same convention as cached_tokens"
