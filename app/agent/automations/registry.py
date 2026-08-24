"""Capability-registry client — the agent's view of what is automatable.

The truth lives platform-side (`ConnectorRegistry.automation_registry()`,
built from the manifests' `automation:` blocks). The agent fetches it
over the existing agent→platform channel (`platform_api_url` +
`X-Agent-Key`, same as credit_reporter / trigger provisioning) and
caches it for a short TTL — the registry changes on deploy, not at
runtime, but a stale cache must not outlive a rollout by much.

Also fetches per-user connector connection state so the setup agent can
answer "is Jira connected, and with which scopes?" without a second
tool round-trip.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_TIMEOUT_S = 10.0
_CACHE_TTL_S = 300.0

_cache: dict[str, Any] = {"at": 0.0, "data": None}


def _platform_url(path: str) -> Optional[str]:
    base = (settings.platform_api_url or "").strip().rstrip("/")
    if not base:
        return None
    return f"{base}{path}"


def _headers(user_id: str) -> dict[str, str]:
    return {
        "X-Agent-Key": settings.agent_api_key or "",
        "X-Agent-User-Id": user_id,
    }


async def fetch_registry(user_id: str, *, force: bool = False) -> dict[str, dict]:
    """connector_id → capability entry. Cached for _CACHE_TTL_S.

    Returns {} when the platform is unreachable — callers treat an
    empty registry as "nothing is automatable right now" and say so,
    rather than guessing (fail closed, no fabricated capabilities).
    """
    now = time.monotonic()
    if not force and _cache["data"] is not None and now - _cache["at"] < _CACHE_TTL_S:
        return _cache["data"]

    url = _platform_url("/v1/automations/registry")
    key = settings.agent_api_key
    if url is None or not key:
        logger.warning("[automations] registry fetch skipped: no platform url/key")
        return {}

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            resp = await client.get(url, headers=_headers(user_id))
        if resp.status_code != 200:
            logger.warning(
                "[automations] registry fetch non-200: %s %s",
                resp.status_code, resp.text[:200],
            )
            return _cache["data"] or {}
        entries = (resp.json() or {}).get("connectors") or []
    except Exception as e:  # noqa: BLE001 — network fetch, fail soft to cache
        logger.warning("[automations] registry fetch failed: %s", e)
        return _cache["data"] or {}

    data = {e["connector_id"]: e for e in entries if e.get("connector_id")}
    _cache["data"] = data
    _cache["at"] = now
    return data


async def fetch_connection_state(user_id: str) -> dict[str, dict]:
    """connector_id → {connected, status, scopes} for this user."""
    url = _platform_url("/v1/automations/connections")
    if url is None or not settings.agent_api_key:
        return {}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            resp = await client.get(url, headers=_headers(user_id))
        if resp.status_code != 200:
            return {}
        entries = (resp.json() or {}).get("connections") or []
    except Exception as e:  # noqa: BLE001
        logger.warning("[automations] connections fetch failed: %s", e)
        return {}
    return {e["connector_id"]: e for e in entries if e.get("connector_id")}


async def fetch_templates(user_id: str) -> list[dict]:
    """The server-curated template catalog (Round 28) — slug, name,
    category, connectors, declared variables, and the spec skeleton.
    Returns [] when the platform is unreachable (the setup agent says
    so rather than inventing templates)."""
    url = _platform_url("/v1/automations/templates")
    if url is None or not settings.agent_api_key:
        return []
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            resp = await client.get(url, headers=_headers(user_id))
        if resp.status_code != 200:
            return []
        return (resp.json() or {}).get("templates") or []
    except Exception as e:  # noqa: BLE001
        logger.warning("[automations] templates fetch failed: %s", e)
        return []


async def fetch_grant(user_id: str, grant_id: str) -> Optional[dict]:
    """The platform's authoritative view of one grant, or None when it
    does not exist / is not this user's. Used by the compiler at arm
    time; the dispatcher re-verifies independently at call time."""
    url = _platform_url("/v1/automations/grant-status")
    if url is None or not settings.agent_api_key:
        return None
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            resp = await client.get(
                url, params={"grant_id": grant_id}, headers=_headers(user_id),
            )
        if resp.status_code != 200:
            return None
        return (resp.json() or {}).get("grant")
    except Exception as e:  # noqa: BLE001
        logger.warning("[automations] grant fetch failed: %s", e)
        return None


async def dispatch_via_platform(
    user_id: str,
    *,
    connector_id: str,
    tool_name: str,
    tool_input: dict,
    grant_id: Optional[str] = None,
    automation_id: Optional[str] = None,
    request_id: Optional[str] = None,
    timeout_s: float = 60.0,
) -> dict:
    """One connector call on the automation channel, via the platform's
    grant-gated dispatch RPC. Returns the serialized ConnectorResult
    dict ({kind: ok|tool_error|...}). Network failures come back as a
    retryable tool_error — the executor treats them like any provider
    failure (health streak, sweep, auto-pause)."""
    url = _platform_url("/v1/automations/dispatch")
    if url is None or not settings.agent_api_key:
        return {"kind": "tool_error", "retryable": False,
                "message": "platform dispatch unavailable (no url/key)"}
    body = {
        "connector_id": connector_id,
        "tool_name": tool_name,
        "tool_input": tool_input or {},
        "grant_id": grant_id,
        "automation_id": automation_id,
        "request_id": request_id,
    }
    # e2e harness marker (scripts/e2e_automations.py): excluded from
    # metering by the platform — which additionally refuses the marker
    # in production, so a stray env var cannot dodge billing.
    import os as _os
    if _os.environ.get("AUTOMATIONS_E2E") == "1":
        body["mode"] = "e2e"
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(url, json=body, headers=_headers(user_id))
        if resp.status_code != 200:
            return {"kind": "tool_error", "retryable": resp.status_code >= 500,
                    "message": f"dispatch HTTP {resp.status_code}: "
                               f"{resp.text[:200]}"}
        return resp.json() or {"kind": "tool_error", "retryable": False,
                               "message": "empty dispatch response"}
    except Exception as e:  # noqa: BLE001 — network edge, fail soft+retryable
        logger.warning("[automations] dispatch RPC failed: %s", e)
        return {"kind": "tool_error", "retryable": True,
                "message": f"dispatch unreachable: {e}"}


async def create_grant_request(
    user_id: str,
    *,
    connector_id: str,
    tool_name: str,
    target: dict,
    cadence: Optional[dict] = None,
    mode: str = "confirm",
    summary: str,
    preview: Optional[dict] = None,
    automation_id: Optional[str] = None,
) -> Optional[dict]:
    """Stage a pending grant request on the platform. Returns the grant
    payload dict, or None on any failure (the caller tells the user the
    permission could not be prepared — never pretends)."""
    url = _platform_url("/v1/automations/grant-requests")
    if url is None or not settings.agent_api_key:
        return None
    body = {
        "connector_id": connector_id,
        "tool_name": tool_name,
        "target": target,
        "cadence": cadence,
        "mode": mode,
        "summary": summary,
        "preview": preview,
        "automation_id": automation_id,
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            resp = await client.post(url, json=body, headers=_headers(user_id))
        if resp.status_code != 200:
            logger.warning("[automations] grant-request non-200: %s %s",
                           resp.status_code, resp.text[:200])
            return None
        return (resp.json() or {}).get("grant")
    except Exception as e:  # noqa: BLE001
        logger.warning("[automations] grant-request failed: %s", e)
        return None


def invalidate_cache() -> None:
    _cache["data"] = None
    _cache["at"] = 0.0
