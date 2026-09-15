"""No tenant proxy forwards an agent-origin 401/403 to the client (D1).

The 2026-09-12 incident's worst user-visible consequence. `sessions.py:275` and
friends forward a tenant 4xx VERBATIM — a deliberate rule ("the agent is the
authority on refusals") that is right for 404/409/413 and catastrophic for 401,
because an agent 401 means the PLATFORM's own `X-Agent-Key` is stale, not that
the user's JWT is bad. On the phone `api.ts:233` treats ANY 401 as a rejected
JWT: `clearToken()` + `_onUnauthorized()` → `setUser(null)` → the login screen.
At 19:14:50.874 the agent 401'd `/api/routines` and
`/api/day-chats/2026-09-12/messages`, and `dial_client_gone` fired 0.4 s later.

Wave 1 fixed `day_chats`, `sessions` and `tenant_proxy`. That was incomplete
for the very incident it cites: **`/api/routines` is served by none of them**.
It is `routines_proxy._proxy`, a generic verbatim pass-through with no
`AgentSaidNo` gate and no 4xx discrimination at all — and so are
`triggers_proxy`, `automations_proxy`, `apps_proxy`, `autopilot_proxy`,
`workspace_proxy` and the WhatsApp-QR proxy in `agent_setup`.

This file is the grep-guard the repo has no eslint to provide: it FAILS on a
pass-through of `resp.status_code` in a platform→agent proxy that does not
first ask `is_agent_auth_failure`. A rule that lives only in the modules that
happened to be fixed is a rule the next proxy will not inherit.

Run:
    cd backend && PYTHONPATH=. python -m pytest -q \
        tests/test_agent_401_never_reaches_client.py
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

API = Path(__file__).resolve().parents[1] / "app" / "api"

# Platform→agent proxies: they authenticate to the tenant with X-Agent-Key, so
# a 4xx they receive is the AGENT's answer about the PLATFORM's credential.
PROXY_FILES = [
    "routines_proxy.py",
    "triggers_proxy.py",
    "automations_proxy.py",
    "apps_proxy.py",
    "autopilot_proxy.py",
    "workspace_proxy.py",
    "tenant_proxy.py",
]

# `apps.py`'s preview route is NOT in the list, and that is a decision, not an
# omission: it reverse-proxies `http://127.0.0.1:<web_port>` — the Expo dev
# server inside the agent container — with no `X-Agent-Key`. A 401 from there
# is the dev server's, not an agent-key failure, and remapping it would hide a
# real refusal. Same reason `netflix_proxy` and `llm_proxy` are absent: neither
# forwards a tenant-agent identity answer.

_PASSTHROUGH = re.compile(r"status_code\s*=\s*resp\.status_code")


@pytest.mark.parametrize("name", PROXY_FILES)
def test_every_agent_proxy_guards_its_status_passthrough(name):
    src = (API / name).read_text()
    if not _PASSTHROUGH.search(src):
        pytest.skip(f"{name} has no raw status pass-through")
    assert (
        "is_agent_auth_failure" in src
        or "agent_passthrough_response" in src
        or "agent_key_stale" in src
    ), (
        f"{name} forwards the agent's status verbatim without remapping "
        "401/403. An agent 401 signs the mobile user out."
    )


@pytest.mark.parametrize("name", PROXY_FILES)
def test_the_guard_precedes_the_passthrough(name):
    """Presence is not enough. A remap BELOW the return it guards is
    unreachable — the exact class of defect this repo's CLAUDE.md records as
    invisible to every other check here."""
    src = (API / name).read_text()
    m = list(_PASSTHROUGH.finditer(src))
    if not m:
        pytest.skip(f"{name} has no raw status pass-through")
    for hit in m:
        before = src[:hit.start()]
        assert (
            "is_agent_auth_failure" in before
            or "agent_key_stale" in before
            or "agent_passthrough_response" in before
        ), (
            f"{name}: a pass-through at offset {hit.start()} has no 401 guard "
            "above it"
        )


def test_routines_proxy_specifically_no_longer_forwards_a_401():
    """The endpoint in the incident record, named on purpose."""
    src = (API / "routines_proxy.py").read_text()
    assert "agent_passthrough_response" in src
    assert "status_code=resp.status_code" not in src


def test_one_predicate_not_seven_copies():
    """`in (401, 403)` scattered across seven files is seven places to forget
    one. The proxies must ask the shared predicate."""
    tp = (API / "tenant_proxy.py").read_text()
    assert "def is_agent_auth_failure" in tp
    assert "def agent_passthrough_response" in tp
    assert "def agent_key_stale_json_response" in tp


def test_the_503_answer_carries_what_the_client_classifies_on():
    """`Retry-After` makes it retryable; `X-Toup-Reason: agent_key_stale` is
    what lets the mobile interceptor tell an agent-origin refusal from a
    genuinely rejected JWT."""
    from app.api.tenant_proxy import (
        agent_key_stale_response, agent_key_stale_json_response,
    )

    exc = agent_key_stale_response()
    assert exc.status_code == 503
    assert exc.headers["X-Toup-Reason"] == "agent_key_stale"
    assert exc.headers["Retry-After"] == "3"

    resp = agent_key_stale_json_response()
    assert resp.status_code == 503
    assert resp.headers["x-toup-reason"] == "agent_key_stale"
    assert resp.headers["retry-after"] == "3"


class _FakeResp:
    def __init__(self, status_code, content=b"{}"):
        self.status_code = status_code
        self.content = content


def test_passthrough_remaps_401_and_403_only():
    from app.api.tenant_proxy import agent_passthrough_response

    for bad in (401, 403):
        assert agent_passthrough_response(_FakeResp(bad)).status_code == 503
    # Every other agent answer is still the agent's own — it is the authority
    # on duplicates, sizes, gate refusals and genuine 404s.
    for ok in (200, 204, 404, 409, 413, 422, 500):
        assert agent_passthrough_response(_FakeResp(ok)).status_code == ok


def test_no_new_proxy_escapes_the_list():
    """A proxy added later must be triaged into (or explicitly out of)
    PROXY_FILES. This fails on any `*_proxy.py` nobody has classified."""
    on_disk = {p.name for p in API.glob("*_proxy.py")}
    known = set(PROXY_FILES) | {
        # Not tenant-agent identity proxies — see the note above.
        "netflix_proxy.py", "llm_proxy.py", "media_proxy.py",
        "analyze_image_proxy.py",
        # Upstream is Brave, not a tenant agent.
        "search_proxy.py",
        # Collapses EVERY agent 4xx (401 included) into 502/404 — no status is
        # ever mirrored, so there is nothing for the client to misread.
        "artifact_proxy.py",
        # Answers HTTP 200 with `{"error": true, "status": N}` in the body, so
        # an agent 401 never becomes an HTTP 401 on the wire.
        "dashboard_proxy.py",
        # WebSockets, not HTTP statuses. `ws_chat_proxy` owns the agent's 4001
        # /"Authentication required" identity rejection (W3); `ws_browser_proxy`
        # relays browser frames and never mirrors an agent HTTP status.
        "ws_chat_proxy.py", "ws_browser_proxy.py",
    }
    unclassified = on_disk - known
    assert not unclassified, (
        f"unclassified proxy module(s): {sorted(unclassified)} — decide "
        "whether each forwards a tenant-agent 401 and add it to PROXY_FILES "
        "or to the exclusion set with a reason"
    )


def test_the_guard_would_fail_on_the_pre_fix_code():
    """The falsifier. A green guard proves nothing until it is shown to go red
    on the code it was written against."""
    pre_fix = (
        "    return Response(\n"
        "        content=resp.content,\n"
        "        status_code=resp.status_code,\n"
        "        headers=out_headers,\n"
        "    )\n"
    )
    assert _PASSTHROUGH.search(pre_fix), "the pattern must match the old code"
    assert "is_agent_auth_failure" not in pre_fix
    assert "agent_key_stale" not in pre_fix
    # …and it must NOT match the replacement.
    post_fix = (
        "    return agent_passthrough_response(\n"
        "        resp, headers=out_headers,\n"
        "    )\n"
    )
    assert not _PASSTHROUGH.search(post_fix)


def test_every_patched_proxy_still_parses():
    for name in PROXY_FILES:
        ast.parse((API / name).read_text())
