"""R43 W3-A — the GitHub and Slack surfaces the round's pickers need.

Every option in the workflow canvas has to compile to a real provider
call (CONTRACT-R43 §0.2: "a picker that writes nowhere is forbidden"),
and before this package three of §5's sources, four of §6's chips and
eight of §7's events had no tool to bind to. These tests hold the two
halves of that together:

  1. **Declared ⇔ dispatchable.** Every tool in each manifest reaches a
     branch of `execute`, and every branch reaches a declared tool. A
     tool that is one without the other is worse than none — it is a
     picker that 500s, or a capability nobody can see.
  2. **The event bindings are real.** Every §7 event's `source_tool`,
     `items_path`, `dedupe_field` and `fields` are checked against the
     tool's ACTUAL output, produced by the provider from a recorded
     provider payload — not against a hand-written shape that can drift
     away from the code the moment either side is edited.
  3. **The derived keys behave.** `comments_key` moves on a new comment
     and stands still otherwise; `_check_rollup` reads every run, not
     the filtered subset; a permalink yields its thread.

No network: the two transports (`_gh_request`, `_call`) are the seam.
"""

from __future__ import annotations

import json

import pytest
import yaml

from app.connectors.base import ConnectorContext, ConnectorOk
from app.connectors.github import provider as ghp
from app.connectors.slack import provider as slp

CTX = ConnectorContext(user_id="u1", access_token="xoxp-test")


def _manifest(connector_id: str) -> dict:
    import app.connectors as pkg
    import os
    root = os.path.dirname(pkg.__file__)
    with open(os.path.join(root, connector_id, "manifest.yaml")) as fh:
        return yaml.safe_load(fh)


def _ok(result) -> dict:
    assert isinstance(result, ConnectorOk), result
    return json.loads(result.content)


def _resolve(item: dict, path: str):
    cur = item
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


# ── Recorded provider payloads ───────────────────────────────────────

GH_SEARCH = {
    "total_count": 2,
    "incomplete_results": False,
    "items": [
        {
            "id": 90001, "number": 12, "title": "Retry the flag write",
            "state": "open", "draft": False, "comments": 3,
            "repository_url": "https://api.github.com/repos/toup-ai/platform",
            "user": {"login": "dana"},
            "html_url": "https://github.com/toup-ai/platform/pull/12",
            "created_at": "2026-08-30T09:00:00Z",
            "updated_at": "2026-08-31T09:00:00Z",
            "labels": [{"name": "bug"}, {"name": "p1"}],
            "pull_request": {"merged_at": None},
        },
        {
            "id": 90002, "number": 7, "title": "Docs typo", "state": "open",
            "comments": 0,
            "repository_url": "https://api.github.com/repos/toup-ai/web",
            "user": {"login": "sam"},
            "html_url": "https://github.com/toup-ai/web/issues/7",
            "updated_at": "2026-08-31T10:00:00Z", "labels": [],
        },
    ],
}

GH_CHECKS = {
    "total_count": 3,
    "check_runs": [
        {"id": 501, "name": "build", "status": "completed",
         "conclusion": "failure", "head_sha": "abc123",
         "html_url": "https://github.com/x/y/runs/501",
         "app": {"name": "GitHub Actions"}},
        {"id": 502, "name": "lint", "status": "completed",
         "conclusion": "success", "head_sha": "abc123",
         "html_url": "https://github.com/x/y/runs/502",
         "app": {"name": "GitHub Actions"}},
        {"id": 503, "name": "deploy", "status": "completed",
         "conclusion": "timed_out", "head_sha": "abc123",
         "html_url": "https://github.com/x/y/runs/503",
         "app": {"name": "GitHub Actions"}},
    ],
}


@pytest.fixture
def gh(monkeypatch):
    """Stub GitHub's transport, recording every request."""
    calls: list[dict] = []

    async def fake(method, path, *, access_token, json_body=None,
                   params=None, scope_hint=""):
        calls.append({"method": method, "path": path, "params": params or {}})
        if path == "/search/issues":
            return dict(GH_SEARCH)
        if path.endswith("/check-runs"):
            return dict(GH_CHECKS)
        if path.startswith("/repos/") and path.count("/") == 3:
            return {"default_branch": "main"}
        if path == "/user/repos":
            return []
        if path.endswith("/issues"):
            return []
        if path == "/search/code":
            return {"total_count": 0, "items": []}
        if path == "/user":
            return {"login": "me"}
        return {"number": 1, "title": "x", "id": 1, "html_url": "u"}

    monkeypatch.setattr(ghp, "_gh_request", fake)
    return calls


# ── GitHub: declared ⇔ dispatchable ──────────────────────────────────


MIN_GH_ARGS = {
    "github__get_issue": {"owner": "o", "repo": "r", "number": 1},
    "github__list_issues": {"owner": "o", "repo": "r"},
    "github__create_comment": {"owner": "o", "repo": "r", "number": 1,
                               "body": "hi"},
    "github__search_code": {"q": "x"},
    "github__search_issues": {"q": "is:open is:pr review-requested:@me"},
    "github__list_check_runs": {"owner": "o", "repo": "r"},
}


@pytest.mark.asyncio
async def test_every_declared_github_tool_dispatches(gh):
    for tool in _manifest("github")["tools"]:
        name = tool["name"]
        result = await ghp.GitHubProvider().execute(
            name, dict(MIN_GH_ARGS.get(name, {})), CTX,
        )
        assert "unknown github tool" not in repr(result), name


@pytest.mark.asyncio
async def test_no_github_branch_is_undeclared(gh):
    """The other direction: a branch nobody declared is dead code that
    the agent can never call and the automation registry cannot see."""
    declared = {t["name"] for t in _manifest("github")["tools"]}
    import re
    src = open(ghp.__file__).read()
    branches = set(re.findall(r'tool_name == "(github__[a-z_]+)"', src))
    assert branches == declared, branches ^ declared


# ── GitHub: search ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_issues_hits_the_search_endpoint_with_the_query(gh):
    out = _ok(await ghp.GitHubProvider().execute(
        "github__search_issues",
        {"q": "is:open is:pr review-requested:@me", "per_page": 50},
        CTX,
    ))
    assert gh[0]["path"] == "/search/issues"
    assert gh[0]["params"]["q"] == "is:open is:pr review-requested:@me"
    assert gh[0]["params"]["per_page"] == 50
    # The engine that documents `review:approved` / `status:failure`.
    assert gh[0]["params"]["advanced_search"] == "true"
    assert out["total_count"] == 2
    first = out["items"][0]
    # The repository, which GitHub sends only as an API URL.
    assert first["repository"] == "toup-ai/platform"
    assert first["is_pull_request"] is True
    assert out["items"][1]["is_pull_request"] is False
    assert first["labels"] == ["bug", "p1"]


@pytest.mark.asyncio
async def test_search_issues_refuses_an_empty_query(gh):
    result = await ghp.GitHubProvider().execute(
        "github__search_issues", {"q": "  "}, CTX,
    )
    assert not isinstance(result, ConnectorOk)
    assert gh == []


@pytest.mark.asyncio
async def test_sort_is_only_forwarded_when_github_accepts_it(gh):
    await ghp.GitHubProvider().execute(
        "github__search_issues", {"q": "x", "sort": "updated"}, CTX)
    assert gh[-1]["params"]["sort"] == "updated"
    assert gh[-1]["params"]["order"] == "desc"
    await ghp.GitHubProvider().execute(
        "github__search_issues", {"q": "x", "sort": "haunted"}, CTX)
    assert "sort" not in gh[-1]["params"]


def test_comments_key_moves_only_when_a_comment_arrives():
    """The `pr_commented` dedupe key. `id` alone fires once in the
    automation's life; `updated_at` fires on a label change."""
    base = {"id": 5, "comments": 2,
            "repository_url": "https://api.github.com/repos/a/b"}
    same_pr_new_comment = dict(base, comments=3)
    unrelated_edit = dict(base, updated_at="2026-09-01T00:00:00Z")
    assert ghp._search_item(base)["comments_key"] == "5:2"
    assert (ghp._search_item(same_pr_new_comment)["comments_key"]
            != ghp._search_item(base)["comments_key"])
    assert (ghp._search_item(unrelated_edit)["comments_key"]
            == ghp._search_item(base)["comments_key"])


def test_comments_key_is_none_when_either_half_is_missing():
    """A null dedupe key is skipped by the event pipeline, which is the
    correct failure — an empty-string key would collapse every hit of
    the window onto one event."""
    assert ghp._search_item({"id": 5})["comments_key"] is None
    assert ghp._search_item({"comments": 2})["comments_key"] is None


# ── GitHub: check runs ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_check_runs_default_to_the_repositorys_own_branch(gh):
    out = _ok(await ghp.GitHubProvider().execute(
        "github__list_check_runs", {"owner": "o", "repo": "r"}, CTX))
    assert gh[0]["path"] == "/repos/o/r"          # asked for the default
    assert gh[1]["path"] == "/repos/o/r/commits/main/check-runs"
    assert out["ref"] == "main"
    # `filter=latest`: without it a re-run repository answers with every
    # historical attempt and a fixed red still reads as red.
    assert gh[1]["params"]["filter"] == "latest"


@pytest.mark.asyncio
async def test_an_explicit_ref_costs_no_lookup(gh):
    await ghp.GitHubProvider().execute(
        "github__list_check_runs",
        {"owner": "o", "repo": "r", "ref": "fix/retries"}, CTX)
    assert len(gh) == 1
    assert gh[0]["path"] == "/repos/o/r/commits/fix/retries/check-runs"


@pytest.mark.asyncio
async def test_failing_means_the_whole_red_family(gh):
    out = _ok(await ghp.GitHubProvider().execute(
        "github__list_check_runs",
        {"owner": "o", "repo": "r", "ref": "main", "conclusion": "failing"},
        CTX))
    got = {r["name"] for r in out["check_runs"]}
    # timed_out blocks a merge exactly as hard as failure.
    assert got == {"build", "deploy"}


@pytest.mark.asyncio
async def test_an_exact_conclusion_is_still_exact(gh):
    out = _ok(await ghp.GitHubProvider().execute(
        "github__list_check_runs",
        {"owner": "o", "repo": "r", "ref": "main", "conclusion": "failure"},
        CTX))
    assert {r["name"] for r in out["check_runs"]} == {"build"}


@pytest.mark.asyncio
async def test_the_rollup_reads_every_run_not_the_filtered_slice(gh):
    """A `conclusion=success` read of a red ref must still say the ref
    is failing — the envelope describes the COMMIT, not the query."""
    out = _ok(await ghp.GitHubProvider().execute(
        "github__list_check_runs",
        {"owner": "o", "repo": "r", "ref": "main", "conclusion": "success"},
        CTX))
    assert out["check_runs"] and out["conclusion"] == "failure"


def test_check_rollup_words():
    assert ghp._check_rollup([]) == "none"
    assert ghp._check_rollup([{"status": "completed",
                               "conclusion": "success"}]) == "success"
    assert ghp._check_rollup([{"status": "in_progress",
                               "conclusion": None}]) == "pending"
    assert ghp._check_rollup([
        {"status": "completed", "conclusion": "success"},
        {"status": "completed", "conclusion": "action_required"},
    ]) == "failure"
    # Not red: a skipped or cancelled check is not a broken build.
    assert ghp._check_rollup([{"status": "completed",
                               "conclusion": "skipped"}]) == "success"


# ── Slack transport stub ─────────────────────────────────────────────


SL_DM_LIST = {"channels": [
    {"id": "D01ABCDEF", "is_im": True, "user": "U_SARA"},
    {"id": "D02ABCDEF", "is_im": True, "user": "U_ME"},
]}

SL_HISTORY = {"messages": [
    {"ts": "1756400100.000100", "user": "U_SARA",
     "text": "Anyone own the retry flag? <@U_ME>", "thread_ts": "1756400000.000100"},
]}

SL_PARENT = {"messages": [
    {"ts": "1756400000.000100", "user": "U_DANA", "text": "Retry flag",
     "reply_count": 4, "latest_reply": "1756400900.000700",
     "reply_users": ["U_ME", "U_DANA"], "thread_ts": "1756400000.000100"},
]}

SL_SEARCH_MINE = {"messages": {"total": 1, "matches": [
    {"ts": "1756400500.000200", "user": "U_ME", "text": "on it",
     "channel": {"id": "C01ABCDEF", "name": "platform"},
     "permalink": ("https://acme.slack.com/archives/C01ABCDEF/p1756400500000200"
                   "?thread_ts=1756400000.000100&cid=C01ABCDEF")},
]}}

SL_SEARCH_MENTIONS = {"messages": {"total": 2, "matches": [
    {"ts": "1756400100.000100", "user": "U_SARA",
     "text": "can <@U_ME> look at this",
     "channel": {"id": "C01ABCDEF", "name": "platform"},
     "permalink": "https://acme.slack.com/archives/C01ABCDEF/p1756400100000100"},
    {"ts": "1756400200.000100", "user": "U_DANA",
     "text": "asked me about it in standup",
     "channel": {"id": "C01ABCDEF", "name": "platform"},
     "permalink": "https://acme.slack.com/archives/C01ABCDEF/p1756400200000100"},
    {"ts": "1756400300.000100", "user": "U_ME",
     "text": "note to <@U_ME>",
     "channel": {"id": "C01ABCDEF", "name": "platform"},
     "permalink": "https://acme.slack.com/archives/C01ABCDEF/p1756400300000100"},
]}}


@pytest.fixture
def sl(monkeypatch):
    calls: list[dict] = []

    async def fake(method, *, access_token, params=None, json_body=None):
        calls.append({"method": method, "params": dict(params or {}),
                      "body": json_body})
        if method == "auth.test":
            return {"user_id": "U_ME", "user": "me", "team": "Acme",
                    "team_id": "T1", "url": "https://acme.slack.com/"}
        if method == "conversations.list":
            types = str((params or {}).get("types") or "")
            # `_channel_index` asks for all four types at once, so the
            # DM branch has to be the one that asks for ONLY DMs.
            if "public_channel" not in types:
                return dict(SL_DM_LIST)
            return {"channels": [
                {"id": "C01ABCDEF", "name": "platform", "is_member": True},
                {"id": "C02ONCALL", "name": "oncall", "is_member": True},
            ]}
        if method == "conversations.info":
            return {"channel": {"id": "C01ABCDEF", "name": "platform",
                                "last_read": "1756300000.000000",
                                "unread_count": 52,
                                "unread_count_display": 52,
                                "topic": {"value": "the platform"},
                                "purpose": {"value": ""}}}
        if method == "conversations.history":
            return dict(SL_HISTORY)
        if method == "conversations.replies":
            return dict(SL_PARENT)
        if method == "search.messages":
            q = str((params or {}).get("query") or "")
            return dict(SL_SEARCH_MINE if q.startswith("from:")
                        else SL_SEARCH_MENTIONS)
        if method == "users.info":
            return {"user": {"id": "U_X", "name": "someone",
                             "profile": {"display_name": "Someone"}}}
        if method == "users.list":
            return {"members": []}
        if method == "chat.postMessage":
            return {"channel": "C01ABCDEF", "ts": "1"}
        return {}

    monkeypatch.setattr(slp, "_call", fake)
    slp._SELF_IDS.clear()
    yield calls
    slp._SELF_IDS.clear()


MIN_SL_ARGS = {
    "slack__read_messages": {"channel": "C01ABCDEF"},
    "slack__search_messages": {"query": "x"},
    "slack__send_message": {"channel": "C01ABCDEF", "text": "hi"},
    "slack__conversation_info": {"channel": "C01ABCDEF"},
}


@pytest.mark.asyncio
async def test_every_declared_slack_tool_dispatches(sl):
    for tool in _manifest("slack")["tools"]:
        name = tool["name"]
        result = await slp.SlackProvider().execute(
            name, dict(MIN_SL_ARGS.get(name, {})), CTX,
        )
        assert "unknown slack tool" not in repr(result), name


@pytest.mark.asyncio
async def test_no_slack_branch_is_undeclared(sl):
    declared = {t["name"] for t in _manifest("slack")["tools"]}
    import re
    src = open(slp.__file__).read()
    branches = set(re.findall(r'tool_name == "(slack__[a-z_]+)"', src))
    assert branches == declared, branches ^ declared


# ── Slack: identity ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_whoami_answers_the_token_owner_and_caches(sl):
    out = _ok(await slp.SlackProvider().execute("slack__whoami", {}, CTX))
    assert out["user_id"] == "U_ME" and out["handle"] == "me"
    await slp.SlackProvider().execute("slack__whoami", {}, CTX)
    # `auth.test` cannot change for the life of a token.
    assert [c["method"] for c in sl].count("auth.test") == 1


@pytest.mark.asyncio
async def test_self_identity_for_user_is_reusable_by_delivery(sl, monkeypatch):
    """Slack-DM delivery has a user id and no token, and
    `ConnectorIdentity.provider_account_id` is empty for Slack — so this
    is the seam that lets it prove a `U…` target is the owner."""
    async def token(_uid):
        return "xoxp-test"
    monkeypatch.setattr(slp, "_resolve_token", token)
    assert (await slp.self_identity_for_user("u1"))["user_id"] == "U_ME"


@pytest.mark.asyncio
async def test_self_identity_declines_rather_than_raising(monkeypatch):
    """A delivery that cannot prove ownership must decline, not crash."""
    async def boom(_uid):
        raise slp._SlackError(object())
    monkeypatch.setattr(slp, "_resolve_token", boom)
    assert (await slp.self_identity_for_user("u1"))["user_id"] == ""


# ── Slack: read state, DMs, mentions, threads ────────────────────────


@pytest.mark.asyncio
async def test_conversation_info_carries_the_read_state(sl):
    out = _ok(await slp.SlackProvider().execute(
        "slack__conversation_info", {"channel": "C01ABCDEF"}, CTX))
    assert out["unread_count_display"] == 52
    assert out["last_read"] == "1756300000.000000"
    # Absent ≠ zero: without this flag a source row cannot tell "you
    # have read everything" from "Slack declined to say".
    assert out["unread_supported"] is True


@pytest.mark.asyncio
async def test_unread_supported_is_false_when_slack_stays_silent(sl, monkeypatch):
    async def quiet(method, *, access_token, params=None, json_body=None):
        if method == "conversations.info":
            return {"channel": {"id": "C01ABCDEF", "name": "platform"}}
        if method == "auth.test":
            return {"user_id": "U_ME", "user": "me"}
        return {}
    monkeypatch.setattr(slp, "_call", quiet)
    out = _ok(await slp.SlackProvider().execute(
        "slack__conversation_info", {"channel": "C01ABCDEF"}, CTX))
    assert out["unread_supported"] is False
    assert "unread_count_display" not in out


@pytest.mark.asyncio
async def test_list_dms_finds_the_owners_own_conversation(sl):
    """`slack_dm` delivery's target: the one DM that involves nobody
    else."""
    out = _ok(await slp.SlackProvider().execute("slack__list_dms", {}, CTX))
    selves = [d for d in out["dms"] if d["is_self"]]
    assert [d["channel"] for d in selves] == ["D02ABCDEF"]
    assert out["total"] == 2


@pytest.mark.asyncio
async def test_list_dms_reports_a_latest_message_to_dedupe_on(sl):
    out = _ok(await slp.SlackProvider().execute(
        "slack__list_dms", {"with_unread": False}, CTX))
    assert all(d["latest_ts"] == "1756400100.000100" for d in out["dms"])
    # `with_unread: false` is what the trigger polls with — no second
    # call per conversation.
    assert not [c for c in sl if c["method"] == "conversations.info"]


@pytest.mark.asyncio
async def test_list_mentions_keeps_real_mentions_and_drops_the_rest(sl):
    out = _ok(await slp.SlackProvider().execute(
        "slack__list_mentions", {}, CTX))
    texts = [m["text"] for m in out["mentions"]]
    # "asked me about it" matched the HANDLE in Slack's index and is not
    # a mention; "note to <@U_ME>" is the owner naming themselves.
    assert len(texts) == 1 and "look at this" in texts[0]
    assert out["mentions"][0]["permalink"].endswith("p1756400100000100")


@pytest.mark.asyncio
async def test_list_mentions_can_be_asked_for_the_loose_reading(sl):
    out = _ok(await slp.SlackProvider().execute(
        "slack__list_mentions", {"strict": False}, CTX))
    # Still never the owner's own message.
    assert len(out["mentions"]) == 2


@pytest.mark.asyncio
async def test_list_mentions_declines_without_an_identity(sl, monkeypatch):
    async def anon(method, *, access_token, params=None, json_body=None):
        if method == "auth.test":
            return {}
        return {}
    monkeypatch.setattr(slp, "_call", anon)
    slp._SELF_IDS.clear()
    result = await slp.SlackProvider().execute("slack__list_mentions", {}, CTX)
    assert not isinstance(result, ConnectorOk)


@pytest.mark.asyncio
async def test_list_threads_finds_my_threads_from_one_search(sl):
    out = _ok(await slp.SlackProvider().execute("slack__list_threads", {}, CTX))
    assert len(out["threads"]) == 1
    t = out["threads"][0]
    assert t["thread_ts"] == "1756400000.000100"   # from the permalink
    assert t["channel"] == "C01ABCDEF"
    assert t["latest_reply"] == "1756400900.000700"
    assert t["i_replied"] is True
    methods = [c["method"] for c in sl]
    # One search names every thread; one replies call fills each in.
    assert methods.count("search.messages") == 1


@pytest.mark.asyncio
async def test_list_threads_in_one_channel_uses_reply_users(sl):
    out = _ok(await slp.SlackProvider().execute(
        "slack__list_threads", {"channel": "C01ABCDEF"}, CTX))
    # The recorded history's message is a REPLY (thread_ts != ts) with
    # no reply_count, so it is not a thread parent and nothing matches.
    assert out["threads"] == []
    assert out["scope"] == "C01ABCDEF"


def test_permalink_yields_its_thread():
    assert slp._permalink_thread(
        "https://acme.slack.com/archives/C01ABCDEF/p1756400500000200"
        "?thread_ts=1756400000.000100&cid=C01ABCDEF") == "1756400000.000100"
    # A top-level message's permalink has no thread, and a junk value is
    # not a timestamp — both must answer "", never a bad dedupe key.
    assert slp._permalink_thread(
        "https://acme.slack.com/archives/C01ABCDEF/p1756400500000200") == ""
    assert slp._permalink_thread(
        "https://x/y?thread_ts=DROP TABLE") == ""
    assert slp._permalink_thread(None) == ""


# ── Slack: message row fields the §6 chips read ──────────────────────


def test_a_bot_message_is_identifiable():
    """"Skip bots" drops on `bot_id`: `subtype` answers a different
    question and is absent on a plain bot post."""
    row = slp._message_row(
        {"ts": "1", "bot_id": "B01", "username": "Deploybot",
         "text": "shipped"}, {}, "U_ME")
    assert row["bot_id"] == "B01"
    assert slp._message_row({"ts": "1", "user": "U_A", "text": "hi"},
                            {}, "U_ME").get("bot_id") is None


def test_thread_ts_is_present_on_both_halves_of_a_thread():
    """"Threads I am in" needs ONE field that a parent and a reply both
    carry. `in_thread_of` stays for the callers already reading it."""
    parent = slp._message_row(
        {"ts": "100.1", "user": "U_A", "text": "q", "reply_count": 3,
         "thread_ts": "100.1", "latest_reply": "900.7"}, {}, "")
    reply = slp._message_row(
        {"ts": "500.2", "user": "U_B", "text": "a", "thread_ts": "100.1"},
        {}, "")
    assert parent["thread_ts"] == reply["thread_ts"] == "100.1"
    assert parent["latest_reply"] == "900.7"
    assert reply["in_thread_of"] == "100.1"
    assert "in_thread_of" not in parent


def test_a_plain_message_claims_no_thread():
    row = slp._message_row({"ts": "1", "user": "U_A", "text": "hi"}, {}, "")
    assert "thread_ts" not in row and "in_thread_of" not in row


# ── The event bindings, checked against real tool output ─────────────


@pytest.mark.asyncio
async def test_github_event_bindings_resolve_against_real_output(gh):
    """Each §7 GitHub event's items_path / dedupe_field / fields, run
    against what the provider actually returns."""
    args = {
        "github__search_issues": {"q": "x"},
        "github__list_check_runs": {"owner": "o", "repo": "r"},
    }
    for ev in _manifest("github")["automation"]["events"]:
        tool = ev["source_tool"]
        if tool not in args:
            continue                       # issue_opened: R26, already pinned
        payload = dict(args[tool])
        payload.update(ev.get("poll_args") or {})
        out = _ok(await ghp.GitHubProvider().execute(tool, payload, CTX))
        items = out[ev["items_path"]]
        assert items, f"{ev['key']}: {ev['items_path']} came back empty"
        for item in items:
            assert item.get(ev["dedupe_field"]) is not None, (
                f"{ev['key']} would dedupe on a null "
                f"{ev['dedupe_field']!r} and drop every event"
            )
            for name, path in (ev.get("fields") or {}).items():
                assert _resolve(item, path) is not None or path in (
                    "conclusion",), f"{ev['key']}: field {name}→{path} is null"


@pytest.mark.asyncio
async def test_slack_event_bindings_resolve_against_real_output(sl):
    for ev in _manifest("slack")["automation"]["events"]:
        tool = ev["source_tool"]
        payload = dict(ev.get("poll_args") or {})
        for p in ev.get("params_required") or []:
            payload.setdefault(p, "C01ABCDEF")
        out = _ok(await slp.SlackProvider().execute(tool, payload, CTX))
        items = out[ev["items_path"]]
        assert items, f"{ev['key']}: {ev['items_path']} came back empty"
        for item in items:
            assert item.get(ev["dedupe_field"]) is not None, (
                f"{ev['key']} would dedupe on a null "
                f"{ev['dedupe_field']!r} and drop every event"
            )


def test_the_rounds_events_are_all_declared():
    """CONTRACT-R43 §7's rows for these two connectors."""
    gh = {e["key"] for e in _manifest("github")["automation"]["events"]}
    assert {"review_requested", "pr_commented", "build_red",
            "pr_approved"} <= gh
    sl = {e["key"] for e in _manifest("slack")["automation"]["events"]}
    assert {"mentioned", "dm_arrived", "thread_moved",
            "oncall_message"} <= sl


def test_no_new_oauth_scope_was_taken():
    """A new scope invalidates every existing grant. Everything this
    package added reads under scopes both connectors already request."""
    gh = _manifest("github")["oauth"]
    assert gh["scopes"] == ["read:user", "repo"]
    assert gh["scopes_optional"] == ["read:org"]
    sl = _manifest("slack")["oauth"]
    assert "search:read" in sl["scopes"]
    assert sl["scopes_optional"] == []
    # The automation read set may only name scopes the connector asks for.
    for cid in ("github", "slack"):
        m = _manifest(cid)
        assert set(m["automation"]["scopes_read"]) <= set(m["oauth"]["scopes"])


# ── The §6 chips, as the provider params they compile into ───────────


@pytest.mark.asyncio
async def test_updated_since_becomes_githubs_own_qualifier(gh):
    """"Changed since yesterday" compiles as a `time_window` into a
    PARAM, because a chip carries no clock. Turning the ISO the executor
    writes into `updated:>=` is the provider's job — which is what keeps
    the compile vocabulary at five kinds."""
    await ghp.GitHubProvider().execute(
        "github__search_issues",
        {"q": "is:open is:pr author:@me",
         "updated_since": "2026-08-30T09:00:00+00:00"}, CTX)
    assert gh[-1]["params"]["q"] == (
        "is:open is:pr author:@me updated:>=2026-08-30T09:00:00+00:00")


@pytest.mark.asyncio
async def test_a_step_that_already_has_a_window_keeps_it(gh):
    await ghp.GitHubProvider().execute(
        "github__search_issues",
        {"q": "is:pr updated:>=2026-01-01", "updated_since": "2026-08-30"},
        CTX)
    assert gh[-1]["params"]["q"] == "is:pr updated:>=2026-01-01"


@pytest.mark.asyncio
async def test_an_unparseable_bound_widens_rather_than_empties(gh):
    """Fails OPEN. A narrowing that cannot be read must not silently
    return nothing — that is indistinguishable from a quiet week."""
    await ghp.GitHubProvider().execute(
        "github__search_issues",
        {"q": "is:pr", "updated_since": "yesterday-ish"}, CTX)
    assert gh[-1]["params"]["q"] == "is:pr"


@pytest.mark.asyncio
async def test_mentions_only_keeps_the_messages_that_name_you(sl):
    out = _ok(await slp.SlackProvider().execute(
        "slack__read_messages",
        {"channel": "C01ABCDEF", "mentions_only": True}, CTX))
    assert len(out["messages"]) == 1
    assert out["applied"] == ["mentions_only"]


@pytest.mark.asyncio
async def test_mentions_only_refuses_rather_than_widens(sl, monkeypatch):
    """Handing back the whole channel under a lit "Mentions me" is the
    chip that lies."""
    async def anon(method, *, access_token, params=None, json_body=None):
        if method == "auth.test":
            return {}
        if method == "conversations.list":
            return {"channels": [{"id": "C01ABCDEF", "name": "platform"}]}
        if method == "conversations.history":
            return dict(SL_HISTORY)
        return {}
    monkeypatch.setattr(slp, "_call", anon)
    slp._SELF_IDS.clear()
    result = await slp.SlackProvider().execute(
        "slack__read_messages",
        {"channel": "C01ABCDEF", "mentions_only": True}, CTX)
    assert not isinstance(result, ConnectorOk)


@pytest.mark.asyncio
async def test_since_last_read_uses_slacks_own_cursor(sl):
    out = _ok(await slp.SlackProvider().execute(
        "slack__read_messages",
        {"channel": "C01ABCDEF", "since_last_read": True}, CTX))
    hist = [c for c in sl if c["method"] == "conversations.history"][0]
    assert hist["params"]["oldest"] == "1756300000.000000"
    assert out["applied"] == ["since_last_read"]


@pytest.mark.asyncio
async def test_an_explicit_oldest_beats_the_read_cursor(sl):
    """A bound the spec already set is narrower than "since I last
    looked", and a filter may never widen a read."""
    await slp.SlackProvider().execute(
        "slack__read_messages",
        {"channel": "C01ABCDEF", "since_last_read": True,
         "oldest": "1756400000.000000"}, CTX)
    hist = [c for c in sl if c["method"] == "conversations.history"][0]
    assert hist["params"]["oldest"] == "1756400000.000000"
    assert not [c for c in sl if c["method"] == "conversations.info"]


@pytest.mark.asyncio
async def test_since_last_read_widens_when_slack_will_not_say(sl, monkeypatch):
    async def quiet(method, *, access_token, params=None, json_body=None):
        if method == "conversations.info":
            return {"channel": {"id": "C01ABCDEF"}}
        if method == "auth.test":
            return {"user_id": "U_ME", "user": "me"}
        if method == "conversations.history":
            return dict(SL_HISTORY)
        return {}
    monkeypatch.setattr(slp, "_call", quiet)
    slp._SELF_IDS.clear()
    out = _ok(await slp.SlackProvider().execute(
        "slack__read_messages",
        {"channel": "C01ABCDEF", "since_last_read": True}, CTX))
    assert out["messages"] and out["applied"] == []


@pytest.mark.asyncio
async def test_threads_only_reads_reply_users_on_the_parent(sl, monkeypatch):
    async def with_parent(method, *, access_token, params=None, json_body=None):
        if method == "auth.test":
            return {"user_id": "U_ME", "user": "me"}
        if method == "conversations.history":
            return {"messages": [
                # A thread the owner is in.
                {"ts": "100.1", "user": "U_DANA", "text": "mine",
                 "reply_count": 2, "reply_users": ["U_ME"],
                 "thread_ts": "100.1"},
                # A thread they are not.
                {"ts": "200.1", "user": "U_SAM", "text": "theirs",
                 "reply_count": 5, "reply_users": ["U_SAM", "U_KIM"],
                 "thread_ts": "200.1"},
                # Not a thread at all.
                {"ts": "300.1", "user": "U_SAM", "text": "aside"},
            ]}
        return {}
    monkeypatch.setattr(slp, "_call", with_parent)
    slp._SELF_IDS.clear()
    out = _ok(await slp.SlackProvider().execute(
        "slack__read_messages",
        {"channel": "C01ABCDEF", "threads_only": True}, CTX))
    assert [m["text"] for m in out["messages"]] == ["mine"]


@pytest.mark.asyncio
async def test_no_bots_needs_no_provider_change(sl):
    """R43 §6's "Skip bots" is a `drop` on `bot_id` — the compile kind
    that already exists — so this asserts only that the field the drop
    reads is on the row."""
    out = _ok(await slp.SlackProvider().execute(
        "slack__read_messages", {"channel": "C01ABCDEF"}, CTX))
    assert all("bot_id" not in m for m in out["messages"])
    row = slp._message_row({"ts": "1", "bot_id": "B01", "text": "x"}, {}, "U_ME")
    assert row["bot_id"] == "B01"
