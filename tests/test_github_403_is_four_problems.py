"""A GitHub 403 is four different problems wearing one status code.

The handler used to answer "scope missing" for all of them. For an org
with OAuth App access restrictions that is not merely unhelpful, it is
FALSE and it loops: the scope IS granted, so the "re-authorize to grant
it" remedy that follows a scope error issues the same scopes and the org
blocks them again. Only an org owner changing a setting fixes it.

Found in prod on 2026-08-08. The founder connected GitHub successfully,
then asked the agent about a repo in his own organization and was told
it did not exist — while `X-OAuth-Scopes: read:user, repo` satisfied
`X-Accepted-OAuth-Scopes: repo` on the very same response.

The bodies below are copied verbatim from live api.github.com responses,
not composed for the test.
"""

from __future__ import annotations

import httpx
import pytest

from app.connectors.base import (
    ConnectorRateLimited,
    ConnectorScopeMissing,
    ConnectorToolError,
)
from app.connectors.github.provider import (
    _GHError,
    _handle_response,
    _org_from_restriction,
)

# Verbatim from api.github.com, GET /repos/toup-com/toup-platform.
RESTRICTED_BODY = (
    '{"message":"Although you appear to have the correct authorization credentials, '
    "the `toup-com` organization has enabled OAuth App access restrictions, meaning "
    "that data access to third-parties is limited. For more information on these "
    "restrictions, including how to enable this app, visit "
    'https://docs.github.com/articles/restricting-access-to-your-organization-s-data/",'
    '"documentation_url":"https://docs.github.com/rest/repos/repos#get-a-repository",'
    '"status":"403"}'
)


def _resp(status: int, body: str = "", **headers) -> httpx.Response:
    return httpx.Response(
        status,
        headers={"content-type": "application/json", **headers},
        text=body,
        request=httpx.Request("GET", "https://api.github.com/x"),
    )


def _raised(resp: httpx.Response, scope_hint: str = ""):
    with pytest.raises(_GHError) as ei:
        _handle_response(resp, scope_hint=scope_hint)
    return ei.value.result


# ── the incident ─────────────────────────────────────────────────────


def test_org_restriction_is_not_reported_as_a_missing_scope():
    """The whole point. The scopes on this response are sufficient."""
    result = _raised(_resp(
        403, RESTRICTED_BODY,
        **{"X-Accepted-OAuth-Scopes": "repo", "X-OAuth-Scopes": "read:user, repo"},
    ))
    assert not isinstance(result, ConnectorScopeMissing)
    assert isinstance(result, ConnectorToolError)


def test_org_restriction_names_the_org_and_links_the_exact_setting():
    result = _raised(_resp(403, RESTRICTED_BODY, **{"X-OAuth-Scopes": "read:user, repo"}))
    msg = result.message
    assert "toup-com" in msg
    assert "github.com/organizations/toup-com/settings/oauth_application_policy" in msg


def test_org_restriction_tells_the_user_reconnecting_will_not_help():
    """The instinct on any connector error is to disconnect and reconnect.
    Here that burns a round trip and cannot work, so say so."""
    msg = _raised(_resp(403, RESTRICTED_BODY)).message.lower()
    assert "reconnect" in msg
    assert "owner" in msg


def test_org_restriction_absolves_the_user_of_a_permissions_problem():
    msg = _raised(_resp(403, RESTRICTED_BODY)).message.lower()
    assert "organization setting" in msg or "organization owner" in msg


# ── the org name is the load-bearing detail ──────────────────────────


def test_org_is_extracted_from_the_real_message():
    assert _org_from_restriction(RESTRICTED_BODY) == "toup-com"


def test_a_bogus_org_name_never_reaches_the_url():
    """The org is interpolated into a link. A junk value should degrade
    to generic prose, not build a broken or attacker-shaped URL."""
    evil = (
        '{"message":"the `../../evil?x=1` organization has enabled OAuth App '
        'access restrictions"}'
    )
    assert _org_from_restriction(evil) == ""
    msg = _raised(_resp(403, evil)).message
    assert "../../evil" not in msg
    assert "Third-party Access" in msg  # the generic path instead


def test_unparseable_body_still_produces_a_useful_answer():
    result = _raised(_resp(403, "the `acme` organization has enabled OAuth App access restrictions"))
    assert "acme" in result.message


# ── the other three problems still map correctly ─────────────────────


def test_rate_limit_still_wins_over_everything():
    result = _raised(_resp(403, RESTRICTED_BODY, **{
        "X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "0",
    }))
    assert isinstance(result, ConnectorRateLimited)


def test_saml_sso_gets_its_own_remedy_not_the_org_owner_one():
    """Different fix: the user authorizes their own token, no owner needed."""
    body = '{"message":"Resource protected by organization SAML enforcement."}'
    result = _raised(_resp(403, body, **{"X-GitHub-SSO": "required; organizations=ABC"}))
    assert isinstance(result, ConnectorToolError)
    assert "single sign-on" in result.message.lower()
    assert "settings/connections/applications" in result.message


def test_a_genuinely_missing_scope_is_still_a_missing_scope():
    result = _raised(_resp(
        403, '{"message":"Must have admin rights to Repository."}',
        **{"X-Accepted-OAuth-Scopes": "admin:org", "X-OAuth-Scopes": "read:user, repo"},
    ))
    assert isinstance(result, ConnectorScopeMissing)
    assert result.required_scope == "admin:org"


def test_a_held_scope_is_never_reported_missing():
    """`X-Accepted-OAuth-Scopes` lists alternatives; holding ANY one of
    them means the scope story is false."""
    result = _raised(_resp(
        403, '{"message":"Forbidden."}',
        **{"X-Accepted-OAuth-Scopes": "repo, public_repo", "X-OAuth-Scopes": "read:user, repo"},
    ))
    assert not isinstance(result, ConnectorScopeMissing)
    assert "403" in result.message


def test_success_and_other_statuses_are_untouched():
    assert _handle_response(_resp(200, '{"ok":true}')) == {"ok": True}
    assert isinstance(_raised(_resp(500)), type(_raised(_resp(500))))
