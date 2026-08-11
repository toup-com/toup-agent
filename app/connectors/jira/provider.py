"""Jira Cloud connector provider.

Atlassian OAuth 2.0 (3LO) + Jira Cloud REST API v3. Standalone (no
`_atlassian_base.py`) for the same reason LinkedIn is standalone: there
is exactly one Atlassian connector today. When Confluence lands, lift
the token/site/ADF helpers out of here verbatim — they are all
product-agnostic, only `_api_base`'s `/ex/jira/` segment is not.

Everything below was verified against developer.atlassian.com and the
published v3 OpenAPI document (swagger-v3.v3.json) on 2026-08-08.

────────────────────────────────────────────────────────────────────────
BLOCKER — this connector cannot complete a connect until api/oauth.py
grows two Atlassian-specific authorize parameters.
────────────────────────────────────────────────────────────────────────
Atlassian documents BOTH of these as *required* on
`https://auth.atlassian.com/authorize`:

  audience=api.atlassian.com   "(required) Set this to api.atlassian.com"
  prompt=consent               "(required) Set to consent"

`_build_authorize_url` emits a fixed parameter set plus a
`provider_name == "google"` branch, so neither is reachable from a
manifest. Without `audience` the authorize call does not produce a token
usable against api.atlassian.com; without `prompt=consent` Atlassian may
skip the consent screen on re-auth and — exactly as with Google — decline
to re-issue a refresh token, which is the "the connector disconnected
itself overnight" symptom already documented in that file. An `elif
provider_name == "jira"` branch setting both is the whole fix.

Second, coupled item for whoever writes that branch: this manifest
declares `pkce: false` (Atlassian staff state 3LO supports the plain
authorization-code flow only; PKCE exists behind an unexposed internal
flag), but `_exchange_code_for_tokens` puts `code_verifier` in the token
body UNCONDITIONALLY. GitHub and LinkedIn ignore the stray field, which
is why nobody has had to care. auth.atlassian.com is Auth0-hosted and
Auth0 *validates* a verifier when one is presented, so the stray field is
a live risk here rather than dead weight. Gate it on `use_pkce` while you
are in there. Flagged, not diagnosed — it has not been run against the
live endpoint.

────────────────────────────────────────────────────────────────────────
Provider quirks that will bite anyone who writes this from memory
────────────────────────────────────────────────────────────────────────
1. THE CLOUD ID. A 3LO token is not scoped to a site. Every call goes to
   `https://api.atlassian.com/ex/jira/{cloudid}/rest/api/3/...` and the
   cloudid must be looked up from `/oauth/token/accessible-resources`
   first. See `_accessible_sites` for the caching decision.

2. `GET /rest/api/3/search` IS GONE. Atlassian shut the legacy JQL search
   endpoints down progressively from 2025-08-01 and blocked all traffic
   by the end of 2025-10; they answer 410 Gone. The replacement is
   `/rest/api/3/search/jql`, which is not a drop-in: pagination is an
   opaque `nextPageToken` instead of `startAt`, and it returns NO fields
   unless you ask for them explicitly.

3. v3 SPEAKS ADF, NOT TEXT. Comment bodies and issue descriptions are
   Atlassian Document Format — a JSON document tree. v3 rejects a plain
   string outright ("Comment body is not valid!"); v2 rejects ADF. We
   take plain text from the model and convert both ways, because handing
   an LLM a raw ADF tree wastes context and handing Jira a raw string
   fails.

4. A SCOPE SHORTFALL IS A 401, NOT A 403. The api.atlassian.com gateway
   answers `401 Unauthorized; scope does not match` when the token lacks
   a scope. Jira's own 403 means a *project permission* refusal. See
   `_handle_jira_error` — this is the one place we can honestly report
   ConnectorScopeMissing, and the one place we must not.

5. ROTATING REFRESH TOKENS. Every refresh returns a NEW refresh_token and
   disables the one used. Failing to persist it bricks the identity at
   the next expiry. Reuse outside the 10-minute leeway answers 403
   invalid_grant — note 403, which is why `jira_refresh` treats it as
   permanent alongside 400/401.

6. NO REVOKE ENDPOINT. See `JiraProvider.revoke`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, ClassVar, Optional

import httpx

from app.connectors.base import (
    BaseConnectorProvider,
    ConnectorContext,
    ConnectorOk,
    ConnectorProviderDown,
    ConnectorRateLimited,
    ConnectorReauthRequired,
    ConnectorResult,
    ConnectorScopeMissing,
    ConnectorToolError,
    HealthResult,
    RefreshFailed,
    RefreshResult,
)
from app.db.database import async_session_maker
from app.services import connector_vault as _vault
from app.services.provider_apps import get_provider_app_async

logger = logging.getLogger(__name__)


ATLASSIAN_API = "https://api.atlassian.com"
ACCESSIBLE_RESOURCES_URL = f"{ATLASSIAN_API}/oauth/token/accessible-resources"
ATLASSIAN_STATUS_URL = "https://status.atlassian.com"
_HTTP_TIMEOUT_S = 15.0

_REAUTH_URL = "/agent/integrations/jira"


# Pooled AsyncClient — same rationale as `_google_base._get_google_client`:
# without it every Jira call eats a fresh TLS handshake (~200-400 ms), and
# a single tool call here is often two round-trips (site lookup + the API
# call itself), so the saving doubles.
_JIRA_CLIENT: Optional[httpx.AsyncClient] = None
_JIRA_CLIENT_LOCK = asyncio.Lock()


async def _get_jira_client() -> httpx.AsyncClient:
    """Process-singleton AsyncClient, with the same defensive-fallback
    contract as the Google/Microsoft/LinkedIn bases: a construction
    failure degrades to per-call clients and shouts, rather than taking
    the connector down."""
    global _JIRA_CLIENT
    if _JIRA_CLIENT is not None:
        return _JIRA_CLIENT
    async with _JIRA_CLIENT_LOCK:
        if _JIRA_CLIENT is None:
            try:
                # HTTP/1.1 keepalive only — http2=True needs the `h2`
                # extra and hard-crashes every call on first use without
                # it. Same trap documented in `_microsoft_base`.
                _JIRA_CLIENT = httpx.AsyncClient(
                    timeout=_HTTP_TIMEOUT_S,
                    limits=httpx.Limits(
                        max_connections=50,
                        max_keepalive_connections=50,
                        keepalive_expiry=300.0,
                    ),
                )
            except Exception as e:
                logger.error(
                    "[jira] pooled AsyncClient construction failed "
                    "(%s: %s) — falling back to per-call clients. "
                    "Investigate immediately.",
                    type(e).__name__, e,
                )
                _JIRA_CLIENT = httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S)
    return _JIRA_CLIENT


async def shutdown_jira_client() -> None:
    global _JIRA_CLIENT
    if _JIRA_CLIENT is not None:
        await _JIRA_CLIENT.aclose()
        _JIRA_CLIENT = None


class _JiraError(Exception):
    """Wraps a `ConnectorResult` so any depth of the call chain can exit
    via `raise` and translate once, at `execute`'s outer except. Same
    pattern as `_GoogleConnectorError` / `_LinkedInError`."""

    def __init__(self, result: ConnectorResult):
        super().__init__(repr(result))
        self.result = result


async def _resolve_token(user_id: str) -> str:
    async with async_session_maker() as db:
        ident = await _vault.get(db, user_id, "jira")
    if ident is None or not ident.access_token:
        raise _JiraError(
            ConnectorToolError(message="No active Jira identity", retryable=False),
        )
    return ident.access_token


def _retarget_reauth(result: ConnectorResult) -> ConnectorResult:
    if isinstance(result, ConnectorReauthRequired):
        return ConnectorReauthRequired(reauth_url=_REAUTH_URL)
    return result


# ─── Error mapping ───────────────────────────────────────────────────


def _jira_message(resp: httpx.Response) -> str:
    """Jira's own words, when it has any.

    Jira answers errors as `{"errorMessages": [...], "errors": {field: msg}}`
    and those strings are genuinely good — "Project keys must start with an
    uppercase letter…", or an issue-type rejection that lists the valid
    types for that project. Inventing a cause on top of that is strictly
    worse than repeating it, so this is what every non-scope branch below
    surfaces.

    The gateway (as opposed to Jira itself) answers `{"message": ...}`, so
    that key is checked too. Falls back to a truncated body.
    """
    try:
        body = resp.json()
    except ValueError:
        return (resp.text or "")[:300]
    if not isinstance(body, dict):
        return (resp.text or "")[:300]

    parts: list[str] = []
    for m in body.get("errorMessages") or []:
        if isinstance(m, str):
            parts.append(m)
    errors = body.get("errors")
    if isinstance(errors, dict):
        for field, msg in errors.items():
            parts.append(f"{field}: {msg}")
    for key in ("message", "error_description", "error"):
        val = body.get(key)
        if isinstance(val, str) and val:
            parts.append(val)
            break
    return "; ".join(parts)[:300] if parts else (resp.text or "")[:300]


def _handle_jira_error(resp: httpx.Response, *, scope_hint: str = "") -> None:
    """Map an Atlassian HTTP response onto the ConnectorResult sum type.

    The 401/403 split is inverted relative to every other connector here,
    and it is inverted on purpose:

      401 — two distinct meanings sharing one status. The api.atlassian.com
            gateway rejects a token whose grant lacks a scope with the
            literal string "scope does not match" (widely reproduced on
            Atlassian's own community, and the only scope signal Atlassian
            emits). That string IS evidence, so it is the one case that
            earns ConnectorScopeMissing. Any other 401 is an expired,
            revoked, or wrong-site token → reauth.

      403 — NOT a scope problem, and reporting one here would be the
            confident-wrong-diagnosis failure. Jira returns 403 when the
            *user* lacks a Jira permission: no Browse Projects on that
            project, no Create Issues, issue-level security, or an
            Atlassian admin app-access policy blocking the app. Re-authing
            with more OAuth scopes cannot fix any of those, so sending the
            user around the consent loop is a dead end. Surface Jira's own
            sentence and let the human read it.
    """
    if 200 <= resp.status_code < 300:
        return

    if resp.status_code == 401:
        detail = _jira_message(resp)
        if "scope does not match" in (detail or resp.text or "").lower():
            raise _JiraError(
                ConnectorScopeMissing(required_scope=scope_hint or "read:jira-work"),
            )
        raise _JiraError(ConnectorReauthRequired(reauth_url=_REAUTH_URL))

    if resp.status_code == 403:
        raise _JiraError(ConnectorToolError(
            message=(
                "Jira refused this action (403). This is a Jira permission, "
                "not a missing connection — the account needs the relevant "
                "project permission, or an Atlassian admin has restricted "
                f"this app. Jira said: {_jira_message(resp) or '(no detail)'}"
            ),
            retryable=False,
        ))

    if resp.status_code == 404:
        # Jira deliberately conflates "does not exist" with "you may not
        # see it" so a 404 cannot leak the existence of a restricted
        # issue. Say both rather than picking one.
        raise _JiraError(ConnectorToolError(
            message=(
                "Not found — the issue or project does not exist, or this "
                "account cannot see it (Jira does not distinguish the two). "
                f"Jira said: {_jira_message(resp) or '(no detail)'}"
            ),
            retryable=False,
        ))

    if resp.status_code == 410:
        # Reachable only if someone reintroduces a legacy endpoint; the
        # shut-down JQL search paths answer exactly this. Named so the
        # next reader does not spend an afternoon on it.
        raise _JiraError(ConnectorToolError(
            message=(
                "Jira answered 410 Gone — this endpoint has been removed by "
                "Atlassian. The legacy /rest/api/3/search JQL endpoints were "
                "shut down in 2025; use /rest/api/3/search/jql. "
                f"Jira said: {_jira_message(resp) or '(no detail)'}"
            ),
            retryable=False,
        ))

    if resp.status_code == 429:
        # Atlassian sends Retry-After in SECONDS on 429 (X-RateLimit-Reset
        # is the ISO-8601 sibling; we want the integer).
        retry_after = 60
        try:
            retry_after = int(resp.headers.get("Retry-After", "60"))
        except (TypeError, ValueError):
            pass
        raise _JiraError(ConnectorRateLimited(retry_after_s=max(retry_after, 1)))

    if resp.status_code >= 500:
        # Atlassian attaches Retry-After to some transient 5xx (503) too,
        # but ConnectorProviderDown carries no budget field and the health
        # sweep is what drives the card — leave the retry to it.
        raise _JiraError(ConnectorProviderDown(
            provider_status_url=ATLASSIAN_STATUS_URL,
        ))

    raise _JiraError(ConnectorToolError(
        message=f"{resp.status_code} from Jira: {_jira_message(resp) or '(no detail)'}",
        retryable=False,
    ))


async def _jira_request(
    method: str,
    url: str,
    *,
    access_token: str,
    json_body: Optional[dict] = None,
    params: Optional[dict] = None,
    scope_hint: str = "",
) -> Any:
    """Authenticated Atlassian request → parsed JSON (dict or list).

    Returns a LIST unchanged: `/oauth/token/accessible-resources` answers
    a bare JSON array, so callers must not assume a dict.
    """
    client = await _get_jira_client()
    try:
        resp = await client.request(
            method,
            url,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            },
            json=json_body,
            params=params,
        )
    except httpx.HTTPError as e:
        # Transport-level: DNS, TLS, timeout. Indistinguishable from an
        # outage from here, and the health sweep will confirm.
        raise _JiraError(ConnectorProviderDown(
            provider_status_url=ATLASSIAN_STATUS_URL,
        )) from e

    _handle_jira_error(resp, scope_hint=scope_hint)

    if not resp.content:
        return {"raw": ""}
    if resp.headers.get("content-type", "").startswith("application/json"):
        try:
            return resp.json()
        except ValueError:
            return {"raw": resp.text[:2000]}
    return {"raw": resp.text[:2000]}


# ─── The cloud-id problem ────────────────────────────────────────────


@dataclass(frozen=True)
class _CloudSite:
    id: str        # the cloudid
    name: str
    url: str       # e.g. https://acme.atlassian.net — used to build browse links


# The site list is a property of the GRANT, so it is cached under the
# ACCESS TOKEN rather than under the user id. That choice is the whole
# design:
#
#   - It self-invalidates. A refresh or a re-auth mints a new token, which
#     misses the cache. So a user who adds a site, removes one, or
#     re-consents with a different site selection can be served a stale
#     answer for at most one access-token lifetime (~1 h) — and there is
#     no invalidation hook anywhere in the connector layer that we would
#     otherwise have to remember to call.
#   - It cannot cross tenants. Two identities never share a key, even for
#     the same user id.
#   - The TTL is a MEMORY bound, not a correctness mechanism. Correctness
#     comes from the key.
#
# Keyed on a hash of the token, not the token, so a heap dump or an
# accidental repr of this dict cannot hand out credentials.
#
# Per-process, and platform-api runs 2 replicas — a cold replica just pays
# one extra round-trip. Not worth a shared cache.
_SITE_TTL_S = 900.0
_SITE_CACHE_MAX = 512
_SITE_CACHE: dict[str, tuple[float, list[_CloudSite]]] = {}
_SITE_CACHE_LOCK = asyncio.Lock()


def _token_key(access_token: str) -> str:
    return hashlib.sha256(access_token.encode("utf-8")).hexdigest()[:32]


async def _accessible_sites(access_token: str) -> list[_CloudSite]:
    """Resolve (and cache) the Jira sites this token can address."""
    key = _token_key(access_token)
    now = time.monotonic()

    async with _SITE_CACHE_LOCK:
        hit = _SITE_CACHE.get(key)
        if hit is not None and now - hit[0] < _SITE_TTL_S:
            return hit[1]

    raw = await _jira_request(
        "GET",
        ACCESSIBLE_RESOURCES_URL,
        access_token=access_token,
        scope_hint="read:jira-work",
    )
    sites = [
        _CloudSite(
            id=str(r.get("id") or ""),
            name=str(r.get("name") or ""),
            url=str(r.get("url") or "").rstrip("/"),
        )
        for r in (raw if isinstance(raw, list) else [])
        if r.get("id")
    ]

    async with _SITE_CACHE_LOCK:
        if len(_SITE_CACHE) >= _SITE_CACHE_MAX:
            # Drop expired entries first; if that frees nothing (every
            # entry is live), drop the oldest so the dict stays bounded.
            for k, (ts, _) in list(_SITE_CACHE.items()):
                if now - ts >= _SITE_TTL_S:
                    _SITE_CACHE.pop(k, None)
            if len(_SITE_CACHE) >= _SITE_CACHE_MAX:
                oldest = min(_SITE_CACHE.items(), key=lambda kv: kv[1][0])[0]
                _SITE_CACHE.pop(oldest, None)
        _SITE_CACHE[key] = (now, sites)

    return sites


async def _resolve_site(access_token: str, requested: Optional[str]) -> _CloudSite:
    """Pick the site for this call, or fail with something actionable."""
    sites = await _accessible_sites(access_token)
    if not sites:
        # A real and confusing state: the OAuth dance succeeded but the
        # user granted no Jira site (or granted Confluence only). Every
        # subsequent call would 404 against an empty cloudid, so say what
        # actually happened.
        raise _JiraError(ConnectorToolError(
            message=(
                "This Atlassian account has no Jira site available to the "
                "connection. Reconnect Jira and pick a site on Atlassian's "
                "consent screen."
            ),
            retryable=False,
        ))

    if not requested:
        return sites[0]

    q = requested.strip().lower().rstrip("/")
    for s in sites:
        if q in (s.id.lower(), s.name.lower(), s.url.lower()):
            return s
    for s in sites:
        if q in s.name.lower() or q in s.url.lower():
            return s

    raise _JiraError(ConnectorToolError(
        message=(
            f"No Jira site matching {requested!r}. Available: "
            + ", ".join(f"{s.name} ({s.url})" for s in sites)
        ),
        retryable=False,
    ))


def _api_base(site: _CloudSite) -> str:
    return f"{ATLASSIAN_API}/ex/jira/{site.id}/rest/api/3"


# ─── Atlassian Document Format ───────────────────────────────────────


def _adf(text: str) -> dict:
    """Plain text → a minimal ADF document.

    v3 rejects a plain string for `body` / `description`, so this is not
    a nicety. Blank lines separate paragraphs; single newlines inside a
    paragraph become hardBreak nodes. Empty paragraphs are dropped —
    a `paragraph` with an empty `content` array is a shape Jira rejects.
    """
    blocks: list[dict] = []
    for block in (text or "").replace("\r\n", "\n").split("\n\n"):
        lines = [ln for ln in block.split("\n")]
        content: list[dict] = []
        for i, line in enumerate(lines):
            if i:
                content.append({"type": "hardBreak"})
            if line:
                content.append({"type": "text", "text": line})
        # Strip a block that turned out to be nothing but breaks.
        if not any(n.get("type") == "text" for n in content):
            continue
        blocks.append({"type": "paragraph", "content": content})

    if not blocks:
        blocks = [{"type": "paragraph", "content": [{"type": "text", "text": text}]}]
    return {"type": "doc", "version": 1, "content": blocks}


def _adf_to_text(node: Any) -> str:
    """ADF → readable plain text.

    Handing an LLM the raw document tree burns context on structural JSON
    and reads badly; this flattens it. Deliberately lossy — marks,
    tables and panels collapse to their text. Tolerates a plain string,
    because a few Jira fields answer one.
    """
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(_adf_to_text(n) for n in node)
    if not isinstance(node, dict):
        return ""

    ntype = node.get("type")
    if ntype == "text":
        return str(node.get("text") or "")
    if ntype == "hardBreak":
        return "\n"
    if ntype == "mention":
        return "@" + str((node.get("attrs") or {}).get("text") or "").lstrip("@")
    if ntype == "emoji":
        attrs = node.get("attrs") or {}
        return str(attrs.get("text") or attrs.get("shortName") or "")
    if ntype == "inlineCard":
        return str((node.get("attrs") or {}).get("url") or "")
    if ntype == "rule":
        return "\n---\n"

    inner = _adf_to_text(node.get("content"))
    if ntype in ("paragraph", "heading", "codeBlock", "blockquote", "panel"):
        return inner + "\n\n"
    if ntype == "listItem":
        return "- " + inner.strip() + "\n"
    return inner


def _text_field(value: Any, *, limit: int = 20_000) -> str:
    return _adf_to_text(value).strip()[:limit]


# ─── OAuth refresh ───────────────────────────────────────────────────


async def jira_refresh(refresh_token: str) -> RefreshResult:
    """Exchange a ROTATING refresh token for a fresh access token.

    Two things here are Atlassian-specific and both are load-bearing:

    1. The body goes as JSON. Atlassian's own docs are inconsistent
       (the 3LO guide shows `application/json`, the service-account guide
       shows form-encoded) and Auth0 accepts both — but the flow we are
       implementing is the 3LO one, so we send what the 3LO guide
       documents rather than betting on the gateway's tolerance. This is
       our own request end to end; there is no shared helper to fight.

    2. Every refresh returns a NEW refresh_token and DISABLES the one
       just used. Returning it in `RefreshResult.refresh_token` is what
       keeps the chain alive — drop it and the identity bricks at the
       next expiry, roughly an hour later, looking like a random
       disconnect.

    Reusing a retired token answers **403** `invalid_grant` (outside the
    documented 10-minute leeway), so 403 is a permanent failure here even
    though it is emphatically not one on the API surface above.
    """
    app_cfg = await get_provider_app_async("jira")
    if app_cfg is None:
        raise RefreshFailed(
            "jira provider not registered — set credentials via "
            "/admin/oauth-apps or JIRA_OAUTH_CLIENT_ID env var"
        )
    client = await _get_jira_client()
    resp = await client.post(
        app_cfg.token_url,
        json={
            "grant_type": "refresh_token",
            "client_id": app_cfg.client_id,
            "client_secret": app_cfg.client_secret,
            "refresh_token": refresh_token,
        },
        headers={"Accept": "application/json"},
    )
    if resp.status_code in (400, 401, 403):
        raise RefreshFailed(
            f"jira refresh returned {resp.status_code}: {resp.text[:200]}"
        )
    if resp.status_code >= 500:
        resp.raise_for_status()
    if not resp.is_success:
        raise RefreshFailed(f"jira refresh unexpected status {resp.status_code}")

    body = resp.json()
    if not body.get("access_token"):
        raise RefreshFailed(f"jira refresh returned no access_token: {resp.text[:200]}")
    expires_in = int(body.get("expires_in", 3600))
    return RefreshResult(
        access_token=body["access_token"],
        refresh_token=body.get("refresh_token"),
        expires_at=datetime.utcnow() + timedelta(seconds=expires_in),
    )


# ─── Response trimming ───────────────────────────────────────────────


_SEARCH_FIELDS = "summary,status,assignee,issuetype,priority,updated,created,project"
_ISSUE_FIELDS = (
    "summary,description,status,assignee,reporter,issuetype,priority,"
    "labels,created,updated,project,resolution,comment"
)


def _person(value: Any) -> Optional[str]:
    """`displayName` when Jira gives one, accountId otherwise.

    We do not request `read:jira-user`, so display names are whatever
    rides along inside the issue payload. Falling back to the accountId
    keeps the field useful instead of silently empty.
    """
    if not isinstance(value, dict):
        return None
    return value.get("displayName") or value.get("accountId")


def _issue_row(issue: dict, site: _CloudSite) -> dict:
    f = issue.get("fields") or {}
    key = issue.get("key")
    return {
        "key": key,
        "summary": f.get("summary"),
        "status": (f.get("status") or {}).get("name"),
        "type": (f.get("issuetype") or {}).get("name"),
        "priority": (f.get("priority") or {}).get("name"),
        "assignee": _person(f.get("assignee")),
        "project": (f.get("project") or {}).get("key"),
        "updated": f.get("updated"),
        "url": f"{site.url}/browse/{key}" if key and site.url else None,
    }


# ─── Provider ────────────────────────────────────────────────────────


_TOOLS = frozenset({
    "jira__search_issues",
    "jira__get_issue",
    "jira__list_projects",
    "jira__create_issue",
    "jira__add_comment",
})


class JiraProvider(BaseConnectorProvider):
    manifest_id: ClassVar[str] = "jira"

    async def execute(
        self,
        tool_name: str,
        tool_input: dict,
        ctx: ConnectorContext,
    ) -> ConnectorResult:
        # Checked BEFORE the token and the site lookup: every tool here
        # needs a cloudid, so resolving first would answer an unknown tool
        # name with a site or auth error and hide the actual mistake.
        if tool_name not in _TOOLS:
            return ConnectorToolError(
                message=f"unknown jira tool {tool_name!r}", retryable=False,
            )

        # Prefer the dispatcher's pre-decrypted token — skips a duplicate
        # vault.get + Fernet decrypt (~100-300 ms on Railway+pgbouncer).
        try:
            access_token = ctx.access_token or await _resolve_token(ctx.user_id)
        except _JiraError as e:
            return _retarget_reauth(e.result)

        try:
            site = await _resolve_site(access_token, tool_input.get("site"))
            base = _api_base(site)

            if tool_name == "jira__search_issues":
                jql = (tool_input.get("jql") or "").strip()
                if not jql:
                    return ConnectorToolError(message="jql required", retryable=False)
                params: dict[str, Any] = {
                    "jql": jql,
                    "maxResults": max(1, min(int(tool_input.get("max_results", 25)), 100)),
                    # The new endpoint returns NOTHING without an explicit
                    # field list — omitting this yields bare ids and keys,
                    # which reads as "search is broken".
                    "fields": tool_input.get("fields") or _SEARCH_FIELDS,
                }
                if tool_input.get("next_page_token"):
                    params["nextPageToken"] = tool_input["next_page_token"]
                result = await _jira_request(
                    "GET",
                    f"{base}/search/jql",
                    access_token=access_token,
                    params=params,
                    scope_hint="read:jira-work",
                )
                issues = [
                    _issue_row(i, site)
                    for i in (result.get("issues") or [])
                    if isinstance(i, dict)
                ]
                return ConnectorOk(content=json.dumps({
                    "site": site.name,
                    "issues": issues,
                    # Opaque and 7-day-lived; the agent hands it straight
                    # back rather than computing an offset.
                    "next_page_token": result.get("nextPageToken"),
                    "is_last": result.get("isLast"),
                }))

            if tool_name == "jira__get_issue":
                key = (tool_input.get("issue_key") or "").strip()
                if not key:
                    return ConnectorToolError(
                        message="issue_key required", retryable=False,
                    )
                max_comments = max(0, min(int(tool_input.get("max_comments", 20)), 100))
                issue = await _jira_request(
                    "GET",
                    f"{base}/issue/{key}",
                    access_token=access_token,
                    # `comment` in the field list is what makes this ONE
                    # call instead of two — same lesson as the Outlook
                    # read path.
                    params={"fields": _ISSUE_FIELDS},
                    scope_hint="read:jira-work",
                )
                f = issue.get("fields") or {}
                out = _issue_row(issue, site)
                out.update({
                    "id": issue.get("id"),
                    "reporter": _person(f.get("reporter")),
                    "labels": f.get("labels") or [],
                    "resolution": (f.get("resolution") or {}).get("name"),
                    "created": f.get("created"),
                    "description": _text_field(f.get("description")),
                })
                comments = (f.get("comment") or {}).get("comments") or []
                if max_comments:
                    out["comments"] = [
                        {
                            "author": _person(c.get("author")),
                            "created": c.get("created"),
                            "body": _text_field(c.get("body"), limit=5_000),
                        }
                        # Jira returns comments oldest-first; the tail is
                        # the part that matters when we are truncating.
                        for c in comments[-max_comments:]
                        if isinstance(c, dict)
                    ]
                out["comment_count"] = (f.get("comment") or {}).get("total", len(comments))
                return ConnectorOk(content=json.dumps(out))

            if tool_name == "jira__list_projects":
                params = {
                    "maxResults": max(1, min(int(tool_input.get("max_results", 50)), 100)),
                    "orderBy": "key",
                }
                if tool_input.get("query"):
                    params["query"] = tool_input["query"]
                result = await _jira_request(
                    "GET",
                    f"{base}/project/search",
                    access_token=access_token,
                    params=params,
                    scope_hint="read:jira-work",
                )
                projects = [
                    {
                        "key": p.get("key"),
                        "name": p.get("name"),
                        "id": p.get("id"),
                        "type": p.get("projectTypeKey"),
                        "url": (
                            f"{site.url}/browse/{p.get('key')}"
                            if p.get("key") and site.url else None
                        ),
                    }
                    for p in (result.get("values") or [])
                    if isinstance(p, dict)
                ]
                return ConnectorOk(content=json.dumps({
                    "site": site.name,
                    "site_url": site.url,
                    "projects": projects,
                    "total": result.get("total"),
                }))

            if tool_name == "jira__create_issue":
                project_key = (tool_input.get("project_key") or "").strip()
                summary = (tool_input.get("summary") or "").strip()
                if not (project_key and summary):
                    return ConnectorToolError(
                        message="project_key and summary are required",
                        retryable=False,
                    )
                fields: dict[str, Any] = {
                    "project": {"key": project_key},
                    "summary": summary,
                    "issuetype": {"name": tool_input.get("issue_type") or "Task"},
                }
                if (tool_input.get("description") or "").strip():
                    fields["description"] = _adf(tool_input["description"])
                labels = tool_input.get("labels")
                if isinstance(labels, list) and labels:
                    # Jira rejects the whole create if any label contains
                    # whitespace, so normalise rather than fail the call.
                    fields["labels"] = [
                        str(l).strip().replace(" ", "-") for l in labels if str(l).strip()
                    ]
                if tool_input.get("priority"):
                    fields["priority"] = {"name": tool_input["priority"]}
                if tool_input.get("assignee_account_id"):
                    fields["assignee"] = {"id": tool_input["assignee_account_id"]}

                created = await _jira_request(
                    "POST",
                    f"{base}/issue",
                    access_token=access_token,
                    json_body={"fields": fields},
                    scope_hint="write:jira-work",
                )
                key = created.get("key")
                return ConnectorOk(content=json.dumps({
                    "key": key,
                    "id": created.get("id"),
                    "site": site.name,
                    "url": f"{site.url}/browse/{key}" if key and site.url else None,
                }))

            if tool_name == "jira__add_comment":
                key = (tool_input.get("issue_key") or "").strip()
                body = tool_input.get("body") or ""
                if not key or not body.strip():
                    return ConnectorToolError(
                        message="issue_key and a non-empty body are required",
                        retryable=False,
                    )
                comment = await _jira_request(
                    "POST",
                    f"{base}/issue/{key}/comment",
                    access_token=access_token,
                    # v3 takes ADF only; a plain string here comes back
                    # as "Comment body is not valid!".
                    json_body={"body": _adf(body)},
                    scope_hint="write:jira-work",
                )
                return ConnectorOk(content=json.dumps({
                    "id": comment.get("id"),
                    "issue_key": key,
                    "author": _person(comment.get("author")),
                    "created": comment.get("created"),
                    "site": site.name,
                    "url": (
                        f"{site.url}/browse/{key}?focusedCommentId={comment.get('id')}"
                        if site.url and comment.get("id") else None
                    ),
                }))

            # Unreachable — `_TOOLS` is checked on entry. Kept so a tool
            # added to that set but not to this chain fails loudly rather
            # than returning None to the dispatcher.
            return ConnectorToolError(
                message=f"jira tool {tool_name!r} is declared but not implemented",
                retryable=False,
            )
        except _JiraError as e:
            return _retarget_reauth(e.result)
        except (TypeError, ValueError) as e:
            # Bad shape from the model (max_results="lots") or from an
            # unexpected payload. Caller error, not a connector fault.
            return ConnectorToolError(
                message=f"invalid argument for {tool_name}: {e}", retryable=False,
            )

    async def revoke(self, user_id, access_token, refresh_token=None):
        """No-op, documented.

        Atlassian exposes no token-revoke endpoint for 3LO — repeatedly
        confirmed on their developer community, never shipped. Consent is
        withdrawn by the user at id.atlassian.com → Connected apps, and an
        org admin cannot do it on their behalf either.

        Disconnecting on our side zeroes the ciphertext, which is what
        actually stops us calling Jira; the grant then ages out (90-day
        refresh-token inactivity expiry). Present as a no-op rather than
        omitted so the dispatcher's best-effort revoke step does not have
        to branch per provider — same shape as Microsoft and LinkedIn.
        """
        return None

    async def refresh(
        self,
        refresh_token: str,
        *,
        scopes: Optional[list[str]] = None,
    ) -> RefreshResult:
        return await jira_refresh(refresh_token)

    async def health_probe(self, ctx: ConnectorContext) -> HealthResult:
        """Backs `health.probe: jira__list_projects`.

        Two round-trips, and both earn their place: the site lookup proves
        the token is live and still has a Jira site attached (the state
        that otherwise fails every tool with a confusing 404), and
        `project/search?maxResults=1` proves read:jira-work actually
        reaches Jira. Nothing cheaper covers both.
        """
        try:
            access_token = ctx.access_token or await _resolve_token(ctx.user_id)
            site = await _resolve_site(access_token, None)
            await _jira_request(
                "GET",
                f"{_api_base(site)}/project/search",
                access_token=access_token,
                params={"maxResults": 1},
                scope_hint="read:jira-work",
            )
            return HealthResult(ok=True)
        except _JiraError as e:
            return HealthResult(ok=False, detail=repr(e.result))
        except Exception as e:
            return HealthResult(ok=False, detail=f"{type(e).__name__}: {e}")
