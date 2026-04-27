"""
OpenAI Admin API — per-user project + service-account-key provisioning.

Flow on bundle activation:
  1. POST /v1/organization/projects   → create a project tagged to the user
  2. POST /v1/organization/projects/{id}/service_accounts → returns api_key
  3. Persist project_id + api_key on AgentConfig
  4. LLM proxy reads bundle_openai_api_key on every OpenAI outbound

On admin user-delete:
  1. POST /v1/organization/projects/{id}/archive → revokes all keys, stops billing

Defensive design:
  - Idempotent. Calling create twice is safe — returns the existing project
    if the AgentConfig already has a project_id (caller's responsibility to
    check). Helpers don't write to DB; orchestrator (vps.py / billing.py /
    admin/users.py) owns persistence.
  - Graceful degradation. If `OPENAI_ADMIN_API_KEY` is not configured, all
    helpers raise `OpenAIAdminUnavailable` — caller logs + falls back to the
    proxy's master-key path. This is the correct behavior on a fresh
    platform install before the operator has wired the admin key.
  - Failures don't block bundle activation. The webhook handler that calls
    these helpers wraps them in try/except and continues — better to ship
    the user a working agent on the master-key fallback than fail the whole
    activation chain.

Security:
  - The admin key is a powerful credential (org-level scope). It lives ONLY
    in `settings.openai_admin_api_key` (Railway env), never in DB or logs.
  - Per-user api_keys ARE persisted (AgentConfig.bundle_openai_api_key).
    Plaintext for now; encrypting via the streaming-credential Fernet pattern
    is queued as a hardening follow-up.

Reference: https://platform.openai.com/docs/api-reference/admin
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


_ORG_BASE = "https://api.openai.com/v1/organization"
_TIMEOUT_S = 15


class OpenAIAdminUnavailable(RuntimeError):
    """Raised when settings.openai_admin_api_key is missing. Callers should
    log + fall back to platform master-key flow without failing user
    activation."""


def _admin_headers() -> dict[str, str]:
    if not settings.openai_admin_api_key:
        raise OpenAIAdminUnavailable(
            "OPENAI_ADMIN_API_KEY is not configured — per-user OpenAI project "
            "auto-provisioning is unavailable; bundle agents fall back to the "
            "platform master OpenAI key. Set the env var to enable."
        )
    return {
        "Authorization": f"Bearer {settings.openai_admin_api_key}",
        "Content-Type": "application/json",
    }


def create_project(name: str) -> str:
    """Create a new OpenAI organization project. Returns the project ID.

    `name` is what shows up in the operator's OpenAI dashboard — keep it
    deterministic + identifiable (e.g. `toup-tenant-<8hex>`).
    """
    headers = _admin_headers()
    with httpx.Client(timeout=_TIMEOUT_S) as client:
        resp = client.post(
            f"{_ORG_BASE}/projects",
            headers=headers,
            json={"name": name},
        )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"OpenAI create_project failed: HTTP {resp.status_code} {resp.text[:300]}"
        )
    data = resp.json()
    project_id = data.get("id")
    if not project_id:
        raise RuntimeError(f"OpenAI create_project returned no id: {data}")
    logger.info("[openai-admin] created project %s name=%r", project_id, name)
    return project_id


def create_project_api_key(project_id: str, key_name: str = "agent") -> str:
    """Create a service account in the project + return its API key value.

    Service accounts are project-scoped (better security model than user-
    scoped admin keys for programmatic agent access). The returned api_key
    starts with `sk-proj-...` and is the value to inject into the agent's
    outbound flow.

    The api_key is only returned ONCE on creation — store it immediately
    in AgentConfig.bundle_openai_api_key. There is no way to retrieve a
    forgotten service-account key short of recreating it.
    """
    headers = _admin_headers()
    with httpx.Client(timeout=_TIMEOUT_S) as client:
        resp = client.post(
            f"{_ORG_BASE}/projects/{project_id}/service_accounts",
            headers=headers,
            json={"name": key_name},
        )
    if resp.status_code >= 400:
        raise RuntimeError(
            f"OpenAI create_project_api_key failed: HTTP {resp.status_code} {resp.text[:300]}"
        )
    data = resp.json()
    # Response shape: { id: "svc_acct_...", api_key: { value: "sk-proj-...", ... }, ... }
    api_key_obj = data.get("api_key") or {}
    api_key_value = api_key_obj.get("value")
    if not api_key_value:
        raise RuntimeError(
            f"OpenAI create_project_api_key returned no api_key.value: {data}"
        )
    logger.info(
        "[openai-admin] created service-account key for project %s (key prefix %s...)",
        project_id, api_key_value[:12],
    )
    return api_key_value


def archive_project(project_id: str) -> bool:
    """Archive an OpenAI project. Cascade-revokes all of its service-account
    keys and stops billing for that project. Idempotent — archiving an
    already-archived project returns True.

    Used by admin user-delete to wipe per-user OpenAI state alongside the
    platform DB and Stripe customer cascade.
    """
    try:
        headers = _admin_headers()
    except OpenAIAdminUnavailable:
        logger.warning(
            "[openai-admin] archive_project skipped (admin key not configured): %s",
            project_id,
        )
        return False

    with httpx.Client(timeout=_TIMEOUT_S) as client:
        resp = client.post(
            f"{_ORG_BASE}/projects/{project_id}/archive",
            headers=headers,
        )
    # Treat 404 (already archived / never existed) as idempotent success.
    if resp.status_code == 404:
        logger.info(
            "[openai-admin] archive_project %s: already gone (HTTP 404, idempotent OK)",
            project_id,
        )
        return True
    if resp.status_code >= 400:
        logger.warning(
            "[openai-admin] archive_project %s failed: HTTP %d %s",
            project_id, resp.status_code, resp.text[:300],
        )
        return False
    logger.info("[openai-admin] archived project %s", project_id)
    return True


def provision_tenant(prefix: str) -> tuple[str, str]:
    """Create project + service-account key for a tenant. Convenience wrapper
    used by bundle-activation paths.

    Returns (project_id, api_key_value). On any failure raises RuntimeError
    (or OpenAIAdminUnavailable if the admin key isn't configured).
    """
    project_name = f"toup-tenant-{prefix}"
    project_id = create_project(project_name)
    try:
        api_key = create_project_api_key(project_id, key_name="agent")
    except Exception:
        # Best-effort: if api-key creation fails, archive the project so we
        # don't leave an orphan project pile up in the operator's dashboard.
        try:
            archive_project(project_id)
        except Exception:
            pass
        raise
    return project_id, api_key
