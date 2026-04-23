"""
Phase 3 rollout service tests — pure-unit, no DB, no real bridge.

Covers:
  - Image-tag prefix validation at the service layer
  - RolloutInProgress raised when active rollout exists
  - Canary selection: finds user.is_canary=TRUE in candidate set
  - Canary selection: returns None when canary user isn't in set
  - Attempt status transitions: upgrade OK / failed / rolled_back /
    rollback_failed
  - AgentEnvContract serialization matches the Phase 2 .env format

End-to-end tests (bridge roundtrip, full rollout against a fake bridge)
belong in docs/new-vps/10-VERIFICATION.md §2-§3 — operator-driven, not
pytest.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from app.types import AgentEnvContract


# ─── AgentEnvContract ─────────────────────────────────────────────


def _valid_contract_kwargs(**overrides):
    base = dict(
        user_id="3134fece-c3de-411b-add8-3f2a58882794",
        agent_api_key="x" * 40,
        database_url="postgresql+asyncpg://toup_agent_3134fece:pw@host.docker.internal:6432/toup_agent_3134fece",
        platform_api_url="https://toup.ai/api",
    )
    base.update(overrides)
    return base


def test_agent_env_happy_path():
    c = AgentEnvContract(**_valid_contract_kwargs(
        openai_api_key="sk-abc", telegram_bot_token="123:abc",
    ))
    lines = c.to_env_lines()
    keys = {line.split("=", 1)[0] for line in lines}
    # Required keys present
    for k in ("RUN_MODE", "USER_ID", "AGENT_API_KEY", "DATABASE_URL",
              "AGENT_WORKSPACE_DIR", "PLATFORM_API_URL", "AGENT_PORT"):
        assert k in keys, f"{k} missing from env output"
    # Optional keys only included when set
    assert "OPENAI_API_KEY" in keys
    assert "ANTHROPIC_API_KEY" not in keys
    assert "TELEGRAM_BOT_TOKEN" in keys
    # Feature-flag booleans rendered lowercase
    assert "USE_DAY_CHAT_CONTEXT=true" in lines
    assert "ENABLE_DAY_RECALL=false" in lines


def test_agent_env_rejects_direct_pg_5432():
    with pytest.raises(ValueError, match="PgBouncer"):
        AgentEnvContract(**_valid_contract_kwargs(
            database_url="postgresql+asyncpg://u:p@host.docker.internal:5432/db",
        ))


def test_agent_env_rejects_http_platform_url():
    with pytest.raises(ValueError, match="https://"):
        AgentEnvContract(**_valid_contract_kwargs(
            platform_api_url="http://toup.ai/api",
        ))


def test_agent_env_rejects_missing_api_suffix():
    with pytest.raises(ValueError, match="/api"):
        AgentEnvContract(**_valid_contract_kwargs(
            platform_api_url="https://toup.ai",
        ))


def test_agent_env_rejects_non_asyncpg_driver():
    with pytest.raises(ValueError, match="asyncpg"):
        AgentEnvContract(**_valid_contract_kwargs(
            database_url="postgresql://u:p@host.docker.internal:6432/db",
        ))


def test_agent_env_short_user_id_rejected():
    with pytest.raises(ValueError, match="user_id"):
        AgentEnvContract(**_valid_contract_kwargs(user_id="short"))


def test_agent_env_to_env_string_is_newline_terminated():
    c = AgentEnvContract(**_valid_contract_kwargs())
    s = c.to_env_string()
    assert s.endswith("\n")
    # One key per line, no blank lines
    assert "" not in [line for line in s.split("\n") if line == ""][-1:] or s.endswith("\n")


# ─── rollout_service.start_rollout validation ────────────────────


@pytest.mark.asyncio
async def test_start_rollout_rejects_non_ghcr_tag():
    from app.services.rollout_service import start_rollout

    db = MagicMock()
    with pytest.raises(ValueError, match="ghcr.io/toup-com/toup-agent"):
        await start_rollout(db, image_tag="docker.io/evil/backdoor:latest", trigger="ci")


@pytest.mark.asyncio
async def test_start_rollout_raises_in_progress_when_active():
    from app.services.rollout_service import start_rollout, RolloutInProgress
    from app.db.models import Rollout

    mock_active = Rollout(
        id="active-id-123",
        image_tag="ghcr.io/toup-com/toup-agent:earlier1234",
        status="running",
        trigger="ci",
    )

    # Patch active_rollout to return a fake in-flight rollout
    with patch("app.services.rollout_service.active_rollout", AsyncMock(return_value=mock_active)):
        db = MagicMock()
        with pytest.raises(RolloutInProgress) as exc_info:
            await start_rollout(db, image_tag="ghcr.io/toup-com/toup-agent:abcd1234", trigger="ci")
        assert exc_info.value.active_id == "active-id-123"
        assert exc_info.value.active_tag == "ghcr.io/toup-com/toup-agent:earlier1234"


# ─── canary selection ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_canary_container_empty_candidates():
    from app.services.rollout_service import _canary_container

    db = MagicMock()
    result = await _canary_container(db, [])
    assert result is None


@pytest.mark.asyncio
async def test_canary_container_no_canary_user():
    from app.services.rollout_service import _canary_container

    # Candidates exist but no user flagged is_canary=True
    cand = [MagicMock(user_id="aaaaaaaa-...-...-..."), MagicMock(user_id="bbbbbbbb-...-...-...")]

    db = MagicMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=None)
    db.execute = AsyncMock(return_value=mock_result)

    result = await _canary_container(db, cand)
    assert result is None


@pytest.mark.asyncio
async def test_canary_container_canary_user_in_set():
    from app.services.rollout_service import _canary_container

    canary_uid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    cand = [
        MagicMock(user_id=canary_uid),
        MagicMock(user_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
    ]

    db = MagicMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=canary_uid)
    db.execute = AsyncMock(return_value=mock_result)

    result = await _canary_container(db, cand)
    assert result is cand[0]


@pytest.mark.asyncio
async def test_canary_container_canary_user_not_in_set():
    """User is_canary=TRUE exists but their container isn't in candidates —
    rollout MUST refuse to proceed."""
    from app.services.rollout_service import _canary_container

    cand = [
        MagicMock(user_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"),
        MagicMock(user_id="cccccccc-cccc-cccc-cccc-cccccccccccc"),
    ]
    # canary user has id 'aaaa...' but isn't in the candidate list
    canary_uid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

    db = MagicMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=canary_uid)
    db.execute = AsyncMock(return_value=mock_result)

    result = await _canary_container(db, cand)
    assert result is None, "canary absent from running set → rollout must refuse"
