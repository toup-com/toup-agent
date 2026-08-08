"""T4a — GitHub connector provider.

REST v3 (api.github.com). PAT-style bearer auth, no token refresh
(GitHub OAuth tokens don't expire — `refresh: false` in the manifest
means the dispatcher never enters the refresh path; revocation is the
only way to invalidate).

Rate limits: GitHub returns `X-RateLimit-Remaining` / `X-RateLimit-
Reset` headers. We surface a 429 ConnectorRateLimited when remaining
hits 0 or the API explicitly 429s. T5a will scrape the headers into
a Prometheus gauge.
"""

from __future__ import annotations

import json
import re
from typing import ClassVar, Optional

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

GH_API = "https://api.github.com"
_HTTP_TIMEOUT = 15.0


class _GHError(Exception):
    def __init__(self, result: ConnectorResult):
        self.result = result


async def _resolve_token(user_id: str) -> str:
    async with async_session_maker() as db:
        ident = await _vault.get(db, user_id, "github")
    if ident is None or not ident.access_token:
        raise _GHError(ConnectorToolError(
            message="No active GitHub identity", retryable=False,
        ))
    return ident.access_token


def _gh_message(body: str) -> str:
    """GitHub's own `message` field, if the body is the usual JSON error."""
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return ""
    msg = parsed.get("message") if isinstance(parsed, dict) else None
    return msg if isinstance(msg, str) else ""


def _org_from_restriction(body: str) -> str:
    """Pull the org login out of GitHub's OAuth-App-restriction message.

    The org name is the whole value of the error: without it we can only
    say "some organization", and the user has to guess which of theirs is
    blocking us. GitHub backticks it:

        …credentials, the `toup-com` organization has enabled OAuth App
        access restrictions…

    Matched against the `message` field rather than the raw body so a
    backticked string elsewhere in the payload can't be mistaken for it.
    """
    msg = _gh_message(body) or body
    m = re.search(r"`([^`]{1,100})`\s+organization has enabled OAuth App", msg)
    if not m:
        return ""
    org = m.group(1).strip()
    # Only ever interpolated into a URL path, so refuse anything that
    # isn't a plausible GitHub login rather than building a broken link.
    return org if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}", org) else ""


def _handle_response(resp: httpx.Response, *, scope_hint: str = "") -> dict:
    if 200 <= resp.status_code < 300:
        if resp.headers.get("content-type", "").startswith("application/json"):
            return resp.json()
        return {"raw": resp.text}
    if resp.status_code == 401:
        raise _GHError(ConnectorReauthRequired(
            reauth_url="/agent/integrations/github",
        ))
    if resp.status_code == 403:
        # A 403 from GitHub is four different problems wearing one status
        # code, and three of them are NOT the user's permissions. Order
        # matters: rate-limit is header-detectable, the two org policies
        # are body/header-detectable, and only what is left is a scope
        # problem — which this used to assume unconditionally.
        if resp.headers.get("X-RateLimit-Remaining") == "0":
            reset = int(resp.headers.get("X-RateLimit-Reset", "0") or 0)
            import time
            wait = max(reset - int(time.time()), 30)
            raise _GHError(ConnectorRateLimited(retry_after_s=wait))

        body = resp.text or ""

        # ── Org has OAuth App access restrictions and hasn't approved us ──
        # GitHub says so explicitly, and names the org:
        #   "Although you appear to have the correct authorization
        #    credentials, the `acme` organization has enabled OAuth App
        #    access restrictions…"
        # Reporting this as a missing scope was actively harmful: the
        # scope IS granted (X-Accepted-OAuth-Scopes: repo against
        # X-OAuth-Scopes: read:user, repo), so the "re-authorize to grant
        # it" advice that follows a scope error sends the user round a
        # loop that cannot succeed — re-auth issues the same scopes and
        # the org blocks them again. Only an org owner changing a setting
        # fixes it, so say that, and say where.
        if "OAuth App access restrictions" in body:
            org = _org_from_restriction(body)
            where = (
                f"https://github.com/organizations/{org}/settings/oauth_application_policy"
                if org else
                "the organization's Settings → Third-party Access → OAuth app policy"
            )
            named = f"The '{org}' organization" if org else "That repository's organization"
            raise _GHError(ConnectorToolError(
                message=(
                    f"{named} restricts third-party OAuth apps, and Toup has not been "
                    f"approved for it yet. This is an organization setting on GitHub's "
                    f"side — your account and permissions are fine, and reconnecting "
                    f"will not change it. An organization owner needs to grant Toup "
                    f"access at {where} (open it, find Toup, choose Grant). "
                    f"Repositories in your personal account and in unrestricted "
                    f"organizations keep working meanwhile."
                ),
                retryable=False,
            ))

        # ── Org enforces SAML SSO and this token isn't authorized for it ──
        # Same shape of problem, different remedy: the user authorizes
        # their existing token rather than an owner approving the app.
        sso = resp.headers.get("X-GitHub-SSO") or ""
        if sso or "SAML enforcement" in body or "single sign-on" in body.lower():
            raise _GHError(ConnectorToolError(
                message=(
                    "That organization enforces SAML single sign-on, and this GitHub "
                    "connection has not been authorized for it. Open "
                    "https://github.com/settings/connections/applications and authorize "
                    "the Toup app for that organization, then try again. Your "
                    "permissions are fine — SSO authorization is a separate step."
                ),
                retryable=False,
            ))

        # ── Genuinely a scope problem — but only claim it if it's true ──
        # `X-Accepted-OAuth-Scopes` is what the endpoint requires and
        # `X-OAuth-Scopes` is what we hold. If we already hold one of the
        # accepted scopes then the missing-scope story is false, and a
        # false diagnosis is worse than an honest unknown.
        accepted = {s.strip() for s in (resp.headers.get("X-Accepted-OAuth-Scopes") or "").split(",") if s.strip()}
        granted = {s.strip() for s in (resp.headers.get("X-OAuth-Scopes") or "").split(",") if s.strip()}
        if accepted and not (accepted & granted):
            raise _GHError(ConnectorScopeMissing(
                required_scope=", ".join(sorted(accepted)) or scope_hint or "unknown",
            ))
        if not accepted and scope_hint:
            raise _GHError(ConnectorScopeMissing(required_scope=scope_hint))

        raise _GHError(ConnectorToolError(
            message=(
                "GitHub refused this request (403) even though the connection holds "
                f"the scopes it asks for ({', '.join(sorted(granted)) or 'none reported'}). "
                "This is usually an organization policy or a repository you no longer "
                f"have access to. GitHub said: {_gh_message(body) or body[:200]}"
            ),
            retryable=False,
        ))
    if resp.status_code == 429:
        wait = 30
        try:
            wait = int(resp.headers.get("Retry-After", "30"))
        except (TypeError, ValueError):
            pass
        raise _GHError(ConnectorRateLimited(retry_after_s=wait))
    if resp.status_code >= 500:
        raise _GHError(ConnectorProviderDown(
            provider_status_url="https://www.githubstatus.com",
        ))
    raise _GHError(ConnectorToolError(
        message=f"{resp.status_code}: {resp.text[:200]}",
        retryable=False,
    ))


async def _gh_request(
    method: str,
    path: str,
    *,
    access_token: str,
    json_body: Optional[dict] = None,
    params: Optional[dict] = None,
    scope_hint: str = "",
) -> dict:
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        resp = await client.request(
            method,
            f"{GH_API}{path}",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json=json_body,
            params=params,
        )
    return _handle_response(resp, scope_hint=scope_hint)


class GitHubProvider(BaseConnectorProvider):
    manifest_id: ClassVar[str] = "github"

    async def execute(
        self,
        tool_name: str,
        tool_input: dict,
        ctx: ConnectorContext,
    ) -> ConnectorResult:
        try:
            access_token = ctx.access_token or await _resolve_token(ctx.user_id)
        except _GHError as e:
            return e.result

        try:
            if tool_name == "github__list_repos":
                params = {
                    "visibility": tool_input.get("visibility", "all"),
                    "sort": tool_input.get("sort", "pushed"),
                    "per_page": int(tool_input.get("per_page", 30)),
                }
                result = await _gh_request(
                    "GET", "/user/repos",
                    access_token=access_token, params=params, scope_hint="repo",
                )
                trimmed = [
                    {
                        "full_name": r.get("full_name"),
                        "private": r.get("private"),
                        "description": r.get("description"),
                        "html_url": r.get("html_url"),
                        "default_branch": r.get("default_branch"),
                    }
                    for r in (result if isinstance(result, list) else [])
                ]
                return ConnectorOk(content=json.dumps({"repos": trimmed}))

            if tool_name == "github__get_issue":
                owner = tool_input.get("owner")
                repo = tool_input.get("repo")
                number = tool_input.get("number")
                if not (owner and repo and number):
                    return ConnectorToolError(message="owner/repo/number required", retryable=False)
                result = await _gh_request(
                    "GET", f"/repos/{owner}/{repo}/issues/{number}",
                    access_token=access_token, scope_hint="repo",
                )
                return ConnectorOk(content=json.dumps({
                    "number": result.get("number"),
                    "title": result.get("title"),
                    "state": result.get("state"),
                    "user": (result.get("user") or {}).get("login"),
                    "body": result.get("body"),
                    "html_url": result.get("html_url"),
                    "is_pull_request": "pull_request" in result,
                }))

            if tool_name == "github__list_issues":
                owner = tool_input.get("owner")
                repo = tool_input.get("repo")
                if not (owner and repo):
                    return ConnectorToolError(message="owner/repo required", retryable=False)
                params = {
                    "state": tool_input.get("state", "open"),
                    "per_page": int(tool_input.get("per_page", 30)),
                }
                result = await _gh_request(
                    "GET", f"/repos/{owner}/{repo}/issues",
                    access_token=access_token, params=params, scope_hint="repo",
                )
                trimmed = [
                    {
                        "number": i.get("number"),
                        "title": i.get("title"),
                        "state": i.get("state"),
                        "user": (i.get("user") or {}).get("login"),
                        "html_url": i.get("html_url"),
                        "is_pull_request": "pull_request" in i,
                    }
                    for i in (result if isinstance(result, list) else [])
                ]
                return ConnectorOk(content=json.dumps({"issues": trimmed}))

            if tool_name == "github__create_comment":
                owner = tool_input.get("owner")
                repo = tool_input.get("repo")
                number = tool_input.get("number")
                body = tool_input.get("body")
                if not (owner and repo and number and body):
                    return ConnectorToolError(message="owner/repo/number/body required", retryable=False)
                result = await _gh_request(
                    "POST", f"/repos/{owner}/{repo}/issues/{number}/comments",
                    access_token=access_token, json_body={"body": body},
                    scope_hint="repo",
                )
                return ConnectorOk(content=json.dumps({
                    "id": result.get("id"),
                    "html_url": result.get("html_url"),
                }))

            if tool_name == "github__search_code":
                q = tool_input.get("q")
                if not q:
                    return ConnectorToolError(message="q required", retryable=False)
                result = await _gh_request(
                    "GET", "/search/code",
                    access_token=access_token,
                    params={"q": q, "per_page": int(tool_input.get("per_page", 20))},
                    scope_hint="repo",
                )
                items = [
                    {
                        "name": i.get("name"),
                        "path": i.get("path"),
                        "html_url": i.get("html_url"),
                        "repository": (i.get("repository") or {}).get("full_name"),
                    }
                    for i in (result.get("items") or [])
                ]
                return ConnectorOk(content=json.dumps({
                    "total_count": result.get("total_count"),
                    "items": items,
                }))

            return ConnectorToolError(
                message=f"unknown github tool {tool_name!r}",
                retryable=False,
            )
        except _GHError as e:
            return e.result

    async def revoke(self, user_id, access_token, refresh_token=None):
        # GitHub's app revocation is a DELETE to
        # /applications/{client_id}/grant — needs the OAuth app's
        # basic-auth credentials, not the user's bearer. Best-effort:
        # if we can't reach it, the vault.disconnect will still zero
        # ciphertext and the user can revoke from github.com/settings.
        from app.services.provider_apps import get_provider_app_async
        cfg = await get_provider_app_async("github")
        if cfg is None or not access_token:
            return
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            try:
                await client.delete(
                    f"{GH_API}/applications/{cfg.client_id}/grant",
                    auth=(cfg.client_id, cfg.client_secret),
                    json={"access_token": access_token},
                )
            except Exception:
                # Logged at the dispatcher level via the vault.
                pass

    async def refresh(self, refresh_token: str) -> RefreshResult:
        # Per manifest `refresh: false` — should never be called.
        # Defensive raise so a misconfiguration is loud.
        raise RefreshFailed(
            "GitHub OAuth tokens don't expire; refresh path should be unreachable. "
            "Manifest declares refresh: false."
        )

    async def health_probe(self, ctx: ConnectorContext) -> HealthResult:
        try:
            access_token = ctx.access_token or await _resolve_token(ctx.user_id)
            await _gh_request(
                "GET", "/user",
                access_token=access_token, scope_hint="read:user",
            )
            return HealthResult(ok=True)
        except _GHError as e:
            return HealthResult(ok=False, detail=repr(e.result))
        except Exception as e:
            return HealthResult(ok=False, detail=f"{type(e).__name__}: {e}")
