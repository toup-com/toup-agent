"""
T0a — Platform <-> bridge schema contract test.

Locks three surfaces:

  1. The top-level keys of the POST /v1/tenants request body the platform
     sends to the bridge (`{prefix, user_id, image_tag, agent_config}`).
  2. The keys of the inner `agent_config` dict produced by
     `_agent_config_to_bridge_body` when every AgentConfig column is
     populated (so adding a new field to the mapper *and* a new column to
     AgentConfig is detected).
  3. The declared field set of `AgentEnvContract` (uppercased — locks the
     declaration, not the serialization, so adding a new Optional field
     drifts even if the fixture doesn't populate it).

Any drift in any surface fails CI. Adding/removing a field intentionally
requires:
  a. Bumping `BRIDGE_SCHEMA_VERSION` in `app/types/agent_env.py`.
  b. Updating `app/types/bridge_schema_v<N>.json` to match.
  c. Updating the bridge's Pydantic AgentConfig + `_build_env` on the VPS
     (this test cannot enforce that — it can only force the operator to
     stop and think).

Why: two production incidents (matin 2026-04-27 on `llm_mode`,
whatsapp_mode 2026-04-30) were caused by adding fields to the platform
without updating the bridge. The bridge silently drops unknown fields.

Pure-unit test — no DB, no fixtures from conftest, no FastAPI app.
The conftest's autouse `_reset_database` fixture still runs against
whatever DATABASE_URL is set; on SQLite locally that emits a wall of
"ALTER TABLE ... IF NOT EXISTS — syntax error" warnings. **Those warnings
are unrelated to this test.** This test only inspects in-memory model
declarations and pure functions; if it fails in CI, it's real drift,
not noise from the autouse fixture.

Two design lessons baked in (caught while writing this test, not in code review):

  - **Lock model_fields, not serialization.** First version compared
    `to_env_lines()` of a fixture. Adding `Optional[str] = None` fields
    slips past — they don't appear in serialized output unless populated.
    Locking `AgentEnvContract.model_fields.keys()` directly catches every
    new field at declaration time, regardless of fixture coverage.

  - **Auto-populate every column, including nullable=False.** SQLAlchemy's
    Python-side `default=` only fires on flush, not on
    `AgentConfig(user_id=...)` construction. So `llm_mode` (declared
    `Mapped[str]` default "manual") is None on a freshly-built in-memory
    instance, and the mapper's `if val:` truthiness check drops it.
    `_populate_every_column` ignores nullability for that reason.
"""

from __future__ import annotations

import json
from datetime import datetime as _dt
from pathlib import Path

import pytest
from sqlalchemy import Boolean, DateTime, Integer, String, Text
from sqlalchemy.inspection import inspect as sa_inspect

from app.db.models.agent import AgentConfig
from app.services.docker_host_service import _agent_config_to_bridge_body
from app.types.agent_env import BRIDGE_SCHEMA_VERSION, AgentEnvContract


SNAPSHOT_PATH = (
    Path(__file__).resolve().parent.parent
    / "app"
    / "types"
    / f"bridge_schema_v{BRIDGE_SCHEMA_VERSION}.json"
)


# ─── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def snapshot() -> dict:
    if not SNAPSHOT_PATH.exists():
        pytest.fail(
            f"Snapshot file missing: {SNAPSHOT_PATH}\n"
            f"BRIDGE_SCHEMA_VERSION={BRIDGE_SCHEMA_VERSION} but no matching "
            f"snapshot exists. Either create the snapshot, or revert the "
            f"version bump."
        )
    return json.loads(SNAPSHOT_PATH.read_text())


def _populate_every_column(cfg: AgentConfig) -> AgentConfig:
    """Set a non-null sentinel on every column of AgentConfig (nullable or
    not), except `id`/`user_id` which the caller already set.

    Crucial for the mapper drift test: if a new column is added to
    AgentConfig AND the mapper grows a `getattr(cfg, "<new_column>", None)`
    branch, the test must see the new key in the mapper output.

    Why nullable=False columns must also be populated: SQLAlchemy's
    Python-side `default=` only fires on flush/insert, not on plain
    `AgentConfig(user_id=...)` construction. So `llm_mode` (declared
    `Mapped[str]` with default="manual") is None on a freshly-constructed
    in-memory instance, and the mapper's `if val:` truthiness check
    drops it. Populating regardless of nullability keeps the fixture
    in lockstep with the model declaration.
    """
    mapper = sa_inspect(AgentConfig)
    skip_keys = {"id", "user_id"}
    for col in mapper.columns:
        if col.key in skip_keys:
            continue
        if getattr(cfg, col.key, None) is not None:
            continue
        col_type = col.type
        if isinstance(col_type, (String, Text)):
            setattr(cfg, col.key, f"sentinel_{col.key}")
        elif isinstance(col_type, Boolean):
            setattr(cfg, col.key, True)
        elif isinstance(col_type, Integer):
            setattr(cfg, col.key, 1)
        elif isinstance(col_type, DateTime):
            setattr(cfg, col.key, _dt(2026, 1, 1))
    return cfg


@pytest.fixture(scope="module")
def fully_populated_agent_config() -> AgentConfig:
    """An AgentConfig with every nullable column populated to a non-null
    sentinel. Required-non-null columns are set explicitly; nullable
    columns are auto-populated by `_populate_every_column` so this fixture
    stays in lockstep with the model."""
    cfg = AgentConfig(
        user_id="3134fece-c3de-411b-add8-3f2a58882794",
    )
    return _populate_every_column(cfg)


# ─── Snapshot integrity ───────────────────────────────────────────────


def test_snapshot_version_matches_constant(snapshot):
    """Choke-point that forces operators to bump constant + snapshot together."""
    assert snapshot["version"] == BRIDGE_SCHEMA_VERSION, (
        f"Snapshot version ({snapshot['version']}) does not match "
        f"BRIDGE_SCHEMA_VERSION ({BRIDGE_SCHEMA_VERSION}). Bump both "
        f"together, or revert one of them."
    )


# ─── Bridge request body — top level ──────────────────────────────────


def test_bridge_request_top_level_keys(snapshot, fully_populated_agent_config):
    """The four top-level keys of POST /v1/tenants are locked:
    prefix, user_id, image_tag, agent_config."""
    body = {
        "prefix": "3134fece",
        "user_id": fully_populated_agent_config.user_id,
        "image_tag": "ghcr.io/toup-com/toup-agent:test",
        "agent_config": _agent_config_to_bridge_body(fully_populated_agent_config),
    }
    actual = sorted(body.keys())
    expected = sorted(snapshot["bridge_request_body_top_level"])
    _assert_set_equal("bridge_request_body_top_level", actual, expected)


# ─── Bridge agent_config inner dict ───────────────────────────────────


def test_bridge_agent_config_fields(snapshot, fully_populated_agent_config):
    """Most failure-prone surface: the bridge's own AgentConfig Pydantic
    model lives on the VPS and silently drops unknown fields. Any field
    added to the mapper without a matching update to the bridge model is
    a latent incident.

    Fixture populates EVERY column on AgentConfig, so adding a new
    `getattr(cfg, "<new_column>", None)` branch to the mapper is detected.
    """
    actual = sorted(_agent_config_to_bridge_body(fully_populated_agent_config).keys())
    expected = sorted(snapshot["bridge_request_agent_config_fields"])
    _assert_set_equal("bridge_request_agent_config_fields", actual, expected)


def test_bridge_agent_config_empty_when_config_missing():
    """Sanity: passing None returns {} (no keys), not crashing."""
    assert _agent_config_to_bridge_body(None) == {}


# ─── AgentEnvContract — declared fields (THE drift detector) ─────────


def test_agent_env_contract_declared_fields(snapshot):
    """Locks the *declared* field set of AgentEnvContract. Adding a new
    field — even an Optional one with default None — drifts immediately,
    independent of any fixture's coverage.

    This is what catches "developer adds OPENAI_ORG_ID to the contract
    but forgets to add it to the bridge." Without locking the declaration
    itself, an Optional[str] field unset in the fixture would emit
    nothing and slip past serialization-based tests.
    """
    actual = sorted(name.upper() for name in AgentEnvContract.model_fields.keys())
    expected = sorted(snapshot["agent_env_contract_keys"])
    _assert_set_equal("agent_env_contract_keys", actual, expected)


def test_agent_env_contract_serialization_emits_all_when_populated(snapshot):
    """Sanity check on `to_env_lines`: when every Optional is populated,
    the emitted env keys equal the declared field set. Catches accidental
    breakage of the serialization layer (e.g. someone hard-codes an
    `if name == "FOO": continue` skip that would silently strip a field).
    """
    contract = AgentEnvContract(
        user_id="3134fece-c3de-411b-add8-3f2a58882794",
        agent_api_key="x" * 40,
        database_url=(
            "postgresql+asyncpg://toup_agent_3134fece:pw@"
            "host.docker.internal:6432/toup_agent_3134fece"
        ),
        platform_api_url="https://toup.ai/api",
        toup_token="tt",
        openai_api_key="sk",
        anthropic_api_key="ant",
        google_api_key="g",
        mistral_api_key="m",
        xai_api_key="x",
        deepseek_api_key="d",
        telegram_bot_token="tg",
        discord_bot_token="ds",
        agent_model="claude-sonnet-4-6",
        agent_color="#ffffff",
    )
    emitted = sorted(line.split("=", 1)[0] for line in contract.to_env_lines())
    expected = sorted(snapshot["agent_env_contract_keys"])
    _assert_set_equal("to_env_lines output", emitted, expected)


def test_agent_env_contract_omits_optionals_when_none():
    """A minimal AgentEnvContract (only required fields) emits only
    required + path/port defaults + boolean flags — not the optional
    keys. Catches accidental defaults that would leak empty values."""
    minimal = AgentEnvContract(
        user_id="3134fece-c3de-411b-add8-3f2a58882794",
        agent_api_key="x" * 40,
        database_url=(
            "postgresql+asyncpg://toup_agent_3134fece:pw@"
            "host.docker.internal:6432/toup_agent_3134fece"
        ),
        platform_api_url="https://toup.ai/api",
    )
    keys = {line.split("=", 1)[0] for line in minimal.to_env_lines()}
    assert {
        "RUN_MODE", "USER_ID", "AGENT_API_KEY", "DATABASE_URL",
        "AGENT_WORKSPACE_DIR", "TOUP_APPS_DIR", "SKILLS_DIR",
        "PLATFORM_API_URL", "AGENT_PORT",
        "USE_DAY_CHAT_CONTEXT", "ENABLE_DAY_RECALL",
    }.issubset(keys)
    for absent in (
        "TOUP_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY", "MISTRAL_API_KEY", "XAI_API_KEY",
        "DEEPSEEK_API_KEY", "TELEGRAM_BOT_TOKEN", "DISCORD_BOT_TOKEN",
        "AGENT_MODEL", "AGENT_COLOR",
    ):
        assert absent not in keys, (
            f"{absent} leaked into env output despite being None — "
            f"check exclude_none in to_env_lines()"
        )


# ─── Helpers ──────────────────────────────────────────────────────────


def _assert_set_equal(surface: str, actual: list[str], expected: list[str]) -> None:
    """Diff-friendly set equality with a guidance message on failure."""
    if actual == expected:
        return
    only_actual = sorted(set(actual) - set(expected))
    only_expected = sorted(set(expected) - set(actual))
    parts = [f"\n{surface} drift detected:"]
    if only_actual:
        parts.append(f"  + Added (in code, not in snapshot): {only_actual}")
    if only_expected:
        parts.append(f"  - Removed (in snapshot, not in code): {only_expected}")
    parts.append(
        "\nIf this drift is INTENTIONAL:\n"
        "  1. Update the bridge's Pydantic AgentConfig + _build_env on the VPS.\n"
        "  2. Bump BRIDGE_SCHEMA_VERSION in app/types/agent_env.py.\n"
        "  3. Rename app/types/bridge_schema_v<N>.json and update its keys.\n"
        "If UNINTENTIONAL: revert the change."
    )
    pytest.fail("\n".join(parts))
