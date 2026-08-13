"""A named credential must outrank whatever cookie the domain happens to hold.

Found 2026-08-13. A user with two accounts on toup.ai — a personal one
and a review one, which is the ordinary case, not an exotic one — opened
a document and got "Preview unavailable". The platform answered 404 in
579 ms while the correct agent served that exact file as a valid PDF in
4.68 s, i.e. the platform never asked it.

Cause: `_get_user_for_file` ranked the `hex_sso_token` COOKIE above the
`?token=` query param. A cookie is per-DOMAIN, so toup.ai holds exactly
one of them — whichever account signed in last — while the bearer token
and the query param are per-REQUEST and name the account the client
means. The sidebar reads the localStorage identity, so the UI can say
one account while the cookie says another and nothing looks wrong.

The request therefore resolved to the *other* account, the platform
proxied to *that* user's agent, and the file was genuinely not there.
A confident 404 on a file that exists reads as data loss.

Rule: explicit beats ambient. Bearer, then `?token=`, then the cookie.
"""

from __future__ import annotations

import pathlib
import re

FILES = pathlib.Path("app/api/files.py")


def _resolution_block() -> str:
    """The body of `_get_user_for_file` up to the agent-key fallback."""
    src = FILES.read_text()
    start = src.index("async def _get_user_for_file")
    end = src.index("# Agent-mode fallback", start)
    return src[start:end]


def test_query_token_is_consulted_before_the_cookie():
    block = _resolution_block()
    q = block.index("if not user_id and token:")
    c = block.index("SSO_COOKIE_NAME")
    assert q < c, (
        "the ambient SSO cookie is consulted before the explicit ?token= "
        "param. A user signed into two accounts on this domain will have "
        "the wrong one resolved, and the platform will proxy their file "
        "request to an agent that does not hold the file — a 404 on a "
        "file that exists."
    )


def test_bearer_still_wins_over_everything():
    """The header is the most explicit credential; nothing may outrank it."""
    block = _resolution_block()
    b = block.index('request.headers.get("authorization"')
    q = block.index("if not user_id and token:")
    c = block.index("SSO_COOKIE_NAME")
    assert b < q < c


def test_each_credential_is_guarded_on_the_previous_failing():
    """Order only means anything if each step is skipped once one wins.

    A missing `if not user_id` turns the sequence into last-writer-wins,
    which silently restores the exact bug — the cookie would overwrite a
    perfectly good token.
    """
    block = _resolution_block()
    # The first (bearer) is unguarded by construction; every later one
    # must be guarded.
    for needle in ("if not user_id and token:", "if not user_id and request is not None:"):
        assert needle in block, f"missing guard: {needle}"
    # ...and no bare re-assignment of user_id outside a guard after the
    # first block.
    tail = block[block.index("if not user_id and token:"):]
    unguarded = [
        ln for ln in tail.splitlines()
        if re.match(r"^\s{4}user_id = ", ln)  # 4 spaces == function level
    ]
    assert not unguarded, (
        f"user_id reassigned at function level after the first credential: "
        f"{unguarded}. That is last-writer-wins, not precedence."
    )


def test_the_docstring_states_the_order_it_implements():
    """The order is a security-relevant contract; a stale docstring here
    is how the next person 'tidies' the cookie back to the top."""
    doc = _resolution_block()
    lowered = doc.lower()
    assert "explicit" in lowered and "ambient" in lowered, (
        "the docstring must say WHY the cookie is last, or the ordering "
        "reads as arbitrary and will be reshuffled"
    )
