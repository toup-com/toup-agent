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


_REPO_URL_RE = re.compile(r"/repos/([^/]+/[^/]+)/?$")

#: GitHub's check-run conclusions that a human reads as "red". `failure`
#: alone is not the set: a check that timed out or that demands manual
#: approval blocks a merge exactly as hard, and a build_red trigger that
#: ignored them would be silent on the two failures people escalate
#: fastest. `neutral`/`skipped`/`cancelled`/`stale` are deliberately NOT
#: here — none of them means the build broke.
_FAILING_CONCLUSIONS = frozenset({"failure", "timed_out", "action_required"})

_CHECK_CONCLUSIONS = (
    "failing", "failure", "timed_out", "action_required",
    "success", "neutral", "cancelled", "skipped", "stale",
)


def _repo_full_name(item: dict) -> str:
    """`repository_url` → "owner/repo".

    `search/issues` items carry no repository object, only the API URL
    of one, so every caller that wants to say WHERE a pull request is
    would otherwise parse this by hand.
    """
    m = _REPO_URL_RE.search(str(item.get("repository_url") or ""))
    return m.group(1) if m else ""


def _search_item(i: dict) -> dict:
    """One `search/issues` hit, trimmed.

    `comments_key` is derived, not GitHub's: it is `<id>:<comments>`,
    which changes exactly when the comment count changes. It exists
    because the `pr_commented` event has to dedupe on "a comment
    arrived" and GitHub's search index exposes the COUNT and nothing
    else — keying on `id` would fire once in the automation's life and
    keying on `updated_at` would fire on a label change. The known miss
    is a comment deleted and another posted between two polls: same
    count, same key, no event.
    """
    comments = i.get("comments")
    ident = i.get("id")
    pull = i.get("pull_request") if isinstance(i.get("pull_request"), dict) else None
    row: dict = {
        # The GLOBAL issue id, not the per-repo `number`: a dedupe space
        # spanning repositories cannot key on a number two repos share.
        "id": ident,
        "number": i.get("number"),
        "title": i.get("title"),
        "state": i.get("state"),
        "repository": _repo_full_name(i),
        "user": (i.get("user") or {}).get("login"),
        "html_url": i.get("html_url"),
        "is_pull_request": pull is not None,
        "draft": i.get("draft"),
        "comments": comments,
        "comments_key": (
            f"{ident}:{comments}" if ident is not None and comments is not None
            else None
        ),
        "created_at": i.get("created_at"),
        "updated_at": i.get("updated_at"),
        "labels": [
            str(l.get("name")) for l in (i.get("labels") or [])
            if isinstance(l, dict) and l.get("name")
        ][:10],
    }
    if pull is not None and pull.get("merged_at"):
        row["merged_at"] = pull.get("merged_at")
    return row


_GH_DATE_RE = re.compile(
    r"\A\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})?)?\Z")


def _with_updated_since(q: str, raw) -> str:
    """`q` narrowed to what changed since `raw`, or `q` unchanged.

    A PARAM rather than a query term because a chip cannot carry a
    clock: R43 §6's "Changed since yesterday" compiles as a
    `time_window` into `updated_since`, the executor writes an ISO
    timestamp there, and turning that into GitHub's `updated:>=` is this
    file's job — which is what keeps the compile vocabulary at five
    kinds instead of six.

    Fails OPEN on an unparseable bound (the read is wider, never empty)
    and leaves a query that already names `updated:` alone, so a step
    with its own window keeps it.
    """
    text = str(raw or "").strip()
    if not text or not _GH_DATE_RE.match(text):
        return q
    if "updated:" in q.lower():
        return q
    return f"{q} updated:>={text}".strip()


def _check_row(r: dict) -> dict:
    return {
        "id": r.get("id"),
        "name": r.get("name"),
        "status": r.get("status"),
        "conclusion": r.get("conclusion"),
        "head_sha": r.get("head_sha"),
        "html_url": r.get("html_url"),
        "started_at": r.get("started_at"),
        "completed_at": r.get("completed_at"),
        "app": (r.get("app") or {}).get("name"),
    }


def _check_rollup(rows: list[dict]) -> str:
    """One word for the whole ref, computed over EVERY run — never over
    the filtered subset, or a `conclusion=failing` read would always
    report the ref as failing even when it holds one stale red among
    twenty greens."""
    if not rows:
        return "none"
    if any(str(r.get("conclusion") or "") in _FAILING_CONCLUSIONS for r in rows):
        return "failure"
    if any(str(r.get("status") or "") != "completed" for r in rows):
        return "pending"
    return "success"


def _clamp(raw, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(raw)))
    except (TypeError, ValueError):
        return default


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
                        # GitHub's own meaning: issues AND pull requests
                        # together. Callers that phrase it as "N PRs"
                        # are wrong about this number, not about the
                        # field — the automations source list says
                        # "open" for exactly that reason.
                        "open_issues_count": r.get("open_issues_count"),
                        "pushed_at": r.get("pushed_at"),
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
                        # Present on pull requests only, and absent (not
                        # False) on plain issues — which is what lets a
                        # caller tell "this PR is ready for a human"
                        # from "this row is not a PR at all".
                        "draft": i.get("draft"),
                        "updated_at": i.get("updated_at"),
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

            if tool_name == "github__search_issues":
                q = str(tool_input.get("q") or "").strip()
                if not q:
                    return ConnectorToolError(message="q required", retryable=False)
                q = _with_updated_since(q, tool_input.get("updated_since"))
                params = {
                    "q": q,
                    "per_page": _clamp(tool_input.get("per_page"), 30, 1, 100),
                    # GitHub's new issue-search engine. Sent explicitly
                    # rather than left to the default so the qualifiers
                    # the automation events depend on — `review:approved`,
                    # `review-requested:@me`, `status:failure` — are always
                    # evaluated by the engine that documents them.
                    "advanced_search": "true",
                }
                sort = str(tool_input.get("sort") or "").strip()
                if sort in ("comments", "created", "updated", "reactions"):
                    params["sort"] = sort
                    params["order"] = (
                        "asc" if str(tool_input.get("order") or "") == "asc"
                        else "desc"
                    )
                result = await _gh_request(
                    "GET", "/search/issues",
                    access_token=access_token, params=params, scope_hint="repo",
                )
                return ConnectorOk(content=json.dumps({
                    "total_count": result.get("total_count"),
                    # GitHub caps the search index at 1000 results and
                    # says so here; a caller reading `total_count` as a
                    # promise it can page to all of them is wrong.
                    "incomplete_results": result.get("incomplete_results"),
                    "items": [
                        _search_item(i) for i in (result.get("items") or [])
                        if isinstance(i, dict)
                    ],
                }))

            if tool_name == "github__list_check_runs":
                owner = tool_input.get("owner")
                repo = tool_input.get("repo")
                if not (owner and repo):
                    return ConnectorToolError(message="owner/repo required", retryable=False)
                ref = str(tool_input.get("ref") or "").strip()
                if not ref:
                    # The pin vocabulary reaches a repository and stops —
                    # there is no `ref` focus kind — so a repo-only caller
                    # has to mean the branch the repo itself calls default.
                    meta = await _gh_request(
                        "GET", f"/repos/{owner}/{repo}",
                        access_token=access_token, scope_hint="repo",
                    )
                    ref = str(meta.get("default_branch") or "").strip()
                    if not ref:
                        return ConnectorToolError(
                            message=(
                                f"{owner}/{repo} reports no default branch, so there "
                                f"is no commit to read checks from. Pass `ref`."
                            ),
                            retryable=False,
                        )
                params = {
                    "per_page": _clamp(tool_input.get("per_page"), 30, 1, 100),
                    # One row per check NAME, the newest run of each.
                    # Without this a re-run repository answers with every
                    # historical attempt and a red that was fixed an hour
                    # ago still reads as red.
                    "filter": "latest",
                }
                status = str(tool_input.get("status") or "").strip()
                if status in ("queued", "in_progress", "completed"):
                    params["status"] = status
                result = await _gh_request(
                    "GET", f"/repos/{owner}/{repo}/commits/{ref}/check-runs",
                    access_token=access_token, params=params, scope_hint="repo",
                )
                rows = [
                    _check_row(r) for r in (result.get("check_runs") or [])
                    if isinstance(r, dict)
                ]
                rollup = _check_rollup(rows)
                want = str(tool_input.get("conclusion") or "").strip().lower()
                if want == "failing":
                    rows = [
                        r for r in rows
                        if str(r.get("conclusion") or "") in _FAILING_CONCLUSIONS
                    ]
                elif want:
                    rows = [
                        r for r in rows
                        if str(r.get("conclusion") or "").lower() == want
                    ]
                return ConnectorOk(content=json.dumps({
                    "owner": owner,
                    "repo": repo,
                    "ref": ref,
                    "conclusion": rollup,
                    "total_count": result.get("total_count"),
                    "check_runs": rows,
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

    async def refresh(
        self,
        refresh_token: str,
        *,
        scopes: Optional[list[str]] = None,
    ) -> RefreshResult:
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
