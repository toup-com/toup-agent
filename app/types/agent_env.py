"""
AgentEnvContract — the typed set of env vars written into a tenant's
`/data/agents/<prefix>/.env` file.

Before Phase 3, `_build_env` in `docker_host_service.py` assembled this
content by string concatenation. That approach drifted: `USE_DAY_CHAT_CONTEXT`
and `ENABLE_DAY_RECALL` were set on some tenants and not others, with no
single source of truth. Phase 3 replaces ad-hoc concatenation with this
pydantic model:

  - Every key the agent container reads is a named field here.
  - Optional fields default to None; their `.env` line is omitted.
  - `.to_env_lines()` produces the canonical serialization; it's the ONLY
    way tenant env files should be produced.
  - Adding a new env key requires adding a field here (and a maintenance
    task to rewrite every existing tenant's `.env`).

This contract is imported by:
  - `app/services/docker_host_service.py` at tenant provisioning / upgrade
  - `app/services/rollout_service.py` via the bridge request body
  - Any future "rewrite all tenant .envs" maintenance script

The AGENT does not import this — it reads env vars via pydantic-settings
in `app/config.py`. The contract exists on the PLATFORM side only; keeping
them aligned is a process concern (migration task when a new key is added).
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# Bumped whenever the platform↔bridge contract changes shape — i.e. when
# either `_agent_config_to_bridge_body` or `AgentEnvContract` adds/removes
# a field. The bridge's own Pydantic AgentConfig model on the VPS must be
# updated in lockstep, plus `bridge_schema_v<N>.json` regenerated.
#
# `tests/test_bridge_contract.py` enforces this — bumping the constant
# without updating the snapshot (or vice versa) fails CI.
BRIDGE_SCHEMA_VERSION = 1


class AgentEnvContract(BaseModel):
    """Canonical env-var payload for a tenant agent container.

    Shape-stable: adding a field is a schema change that triggers the
    "rewrite every tenant's .env" maintenance flow.
    """

    # ── Core identity ──────────────────────────────────────────────
    # These four are REQUIRED; no defaults. Missing any = invalid contract.
    run_mode: str = Field(default="agent", description="Always 'agent' for tenant containers")
    user_id: str = Field(..., description="Full UUID of the tenant user")
    agent_api_key: str = Field(..., min_length=32, description="Bearer token platform uses to call this agent's API")
    database_url: str = Field(..., description="postgresql+asyncpg://<role>:<pw>@host.docker.internal:6432/<db>")

    # ── Paths (match Dockerfile.agent + bridge bind-mounts) ───────
    agent_workspace_dir: str = Field(default="/app/workspace")
    toup_apps_dir: str = Field(default="/app/workspace/apps")
    skills_dir: str = Field(default="/app/skills")

    # ── Platform callback ──────────────────────────────────────────
    platform_api_url: str = Field(..., description="https://toup.ai/api — how the agent reaches the platform")
    toup_token: Optional[str] = Field(default=None, description="Connect token from AgentConfig.connect_token")

    # ── Self-knowledge ─────────────────────────────────────────────
    agent_port: int = Field(default=8001, description="Internal container port (bridge binds host:8001)")

    # ── LLM provider keys (optional; only set keys the tenant has) ─
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None
    xai_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None

    # ── Ingress channel tokens (optional) ──────────────────────────
    telegram_bot_token: Optional[str] = None
    discord_bot_token: Optional[str] = None

    # ── Soul / runtime config ─────────────────────────────────────
    agent_model: Optional[str] = None
    agent_color: Optional[str] = None

    # ── Feature flags (explicit; previously drifted) ──────────────
    use_day_chat_context: bool = Field(
        default=True,
        description="Day-as-Chat context loader. Shipped ON by default (production).",
    )
    enable_day_recall: bool = Field(
        default=False,
        description="recall_day tool + archival summaries. Opt-in per tenant.",
    )

    # ── Validators ─────────────────────────────────────────────────

    @field_validator("user_id")
    @classmethod
    def user_id_must_look_like_uuid(cls, v: str) -> str:
        # Not a strict UUID check — agent code only uses the 8-char prefix,
        # but we do want to catch glaring errors like truncated IDs.
        if len(v) < 8:
            raise ValueError(f"user_id too short: {v!r}")
        return v

    @field_validator("database_url")
    @classmethod
    def database_url_uses_pgbouncer(cls, v: str) -> str:
        # Defense in depth: Phase 2 PgBouncer is on :6432. A :5432 in the
        # URL means we're bypassing the pooler, which breaks at scale.
        if ":5432/" in v:
            raise ValueError(
                "database_url hits Postgres on :5432 directly; "
                "Phase 2+ agents must go through PgBouncer on :6432"
            )
        if "+asyncpg" not in v:
            raise ValueError(
                f"database_url must use the asyncpg driver, got: {v}"
            )
        return v

    @field_validator("platform_api_url")
    @classmethod
    def platform_api_url_is_https_with_api_suffix(cls, v: str) -> str:
        if not v.startswith("https://"):
            raise ValueError(f"platform_api_url must be https://, got: {v}")
        if not v.rstrip("/").endswith("/api"):
            raise ValueError(f"platform_api_url must end in /api, got: {v}")
        return v.rstrip("/")

    # ── Serialization ─────────────────────────────────────────────

    def to_env_lines(self) -> list[str]:
        """Render as KEY=VALUE lines for `/data/agents/<prefix>/.env`.

        - Omits Optional fields that are None (don't write empty keys —
          pydantic-settings handles missing-env as default, but writing
          `KEY=` would set it to empty string).
        - Keys emitted in UPPERCASE (env convention).
        - Order matches field declaration (stable for diffs).
        - Booleans render as lowercase strings (`true`/`false`) for
          pydantic-settings parsing compatibility.
        """
        lines: list[str] = []
        for name, value in self.model_dump(exclude_none=True).items():
            key = name.upper()
            if isinstance(value, bool):
                rendered = "true" if value else "false"
            else:
                rendered = str(value)
            lines.append(f"{key}={rendered}")
        return lines

    def to_env_string(self) -> str:
        """Newline-terminated env-file content."""
        return "\n".join(self.to_env_lines()) + "\n"
