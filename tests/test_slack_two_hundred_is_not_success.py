"""Slack answers `200 OK` when it fails, and that changes everything.

Every other connector in this package signals failure in the status
line, so the house pattern — look at `resp.status_code`, branch on 401 /
403 / 429 — is correct for them and *blind* for Slack. A revoked Slack
token comes back as:

    HTTP/1.1 200 OK
    {"ok": false, "error": "invalid_auth"}

Read the status first and that is a success with no `channels` key,
which the shaping code turns into `{"channels": []}` and the agent
reports to the user as "your workspace has no channels". Silent, wrong,
and indistinguishable from an empty account — the worst shape a
connector bug can take, because there is nothing to notice.

So these tests assert the ERROR MAPPING, not the happy path, and they
assert it through `_check` — the one function every Slack call funnels
through.

The second half covers id rendering. Slack sends `<@U04J1F2>` and
`<https://x|label>` on the wire; an LLM handed that verbatim quotes raw
ids back at the user and cannot tell it was mentioned by name.
"""

from __future__ import annotations

import httpx
import pytest

from app.connectors.base import (
    ConnectorProviderDown,
    ConnectorRateLimited,
    ConnectorReauthRequired,
    ConnectorScopeMissing,
    ConnectorToolError,
)
from app.connectors.slack.provider import (
    _channel_row,
    _check,
    _clamp,
    _message_row,
    _render_text,
    _SlackError,
    _speaker_ids,
)


def _resp(payload: dict, status: int = 200, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status,
        headers={"content-type": "application/json", **(headers or {})},
        json=payload,
    )


def _verdict(payload: dict, status: int = 200, headers: dict | None = None):
    with pytest.raises(_SlackError) as ei:
        _check(_resp(payload, status, headers), method="conversations.history")
    return ei.value.result


# ── The headline trap ────────────────────────────────────────────────


def test_ok_false_on_a_200_is_an_error_not_an_empty_result():
    """The whole reason this file exists."""
    verdict = _verdict({"ok": False, "error": "invalid_auth"})
    assert isinstance(verdict, ConnectorReauthRequired)


def test_ok_true_passes_the_body_through():
    body = _check(_resp({"ok": True, "channels": [{"id": "C1"}]}),
                  method="conversations.list")
    assert body["channels"] == [{"id": "C1"}]


@pytest.mark.parametrize("err", [
    "invalid_auth", "not_authed", "token_revoked",
    "token_expired", "account_inactive",
])
def test_every_dead_credential_becomes_reauth(err):
    assert isinstance(_verdict({"ok": False, "error": err}), ConnectorReauthRequired)


def test_missing_scope_names_the_scope_slack_named():
    """Slack is the only provider here that says WHICH scope. Passing it
    through is the difference between "reconnect and grant everything
    again" and "you need channels:history"."""
    verdict = _verdict({
        "ok": False, "error": "missing_scope",
        "needed": "channels:history", "provided": "channels:read",
    })
    assert isinstance(verdict, ConnectorScopeMissing)
    assert verdict.required_scope == "channels:history"


def test_missing_scope_without_a_needed_field_still_maps_cleanly():
    verdict = _verdict({"ok": False, "error": "missing_scope"})
    assert isinstance(verdict, ConnectorScopeMissing)
    assert verdict.required_scope  # never an empty string


def test_rate_limit_reads_the_header_not_the_default():
    verdict = _verdict({}, status=429, headers={"Retry-After": "17"})
    assert isinstance(verdict, ConnectorRateLimited)
    assert verdict.retry_after_s == 17


def test_rate_limit_survives_a_junk_retry_after():
    verdict = _verdict({}, status=429, headers={"Retry-After": "soon"})
    assert isinstance(verdict, ConnectorRateLimited)
    assert verdict.retry_after_s > 0


def test_body_level_ratelimited_is_also_a_rate_limit():
    """Slack signals it both ways depending on which edge answers."""
    assert isinstance(_verdict({"ok": False, "error": "ratelimited"}),
                      ConnectorRateLimited)


def test_a_429_is_read_from_the_status_line_before_the_body():
    """429 and 5xx come from Slack's proxy, which can answer with an
    HTML body — so those two must NOT depend on parsing JSON."""
    resp = httpx.Response(429, headers={"Retry-After": "5",
                                        "content-type": "text/html"},
                          text="<html>rate limited</html>")
    with pytest.raises(_SlackError) as ei:
        _check(resp, method="x")
    assert isinstance(ei.value.result, ConnectorRateLimited)


def test_a_500_with_an_html_body_is_provider_down():
    resp = httpx.Response(503, headers={"content-type": "text/html"}, text="<html/>")
    with pytest.raises(_SlackError) as ei:
        _check(resp, method="x")
    assert isinstance(ei.value.result, ConnectorProviderDown)


@pytest.mark.parametrize("err", ["fatal_error", "internal_error", "service_unavailable"])
def test_slack_side_failures_are_provider_down_not_tool_errors(err):
    assert isinstance(_verdict({"ok": False, "error": err}), ConnectorProviderDown)


def test_not_in_channel_explains_the_remedy():
    """The slug alone ("not_in_channel") gives the agent nothing to tell
    the user. The fix is a human action in Slack, not a retry."""
    verdict = _verdict({"ok": False, "error": "not_in_channel"})
    assert isinstance(verdict, ConnectorToolError)
    assert verdict.retryable is False
    assert "join" in verdict.message.lower()


def test_an_unknown_error_still_names_the_method_and_the_slug():
    verdict = _verdict({"ok": False, "error": "some_new_slack_error"})
    assert isinstance(verdict, ConnectorToolError)
    assert "some_new_slack_error" in verdict.message
    assert "conversations.history" in verdict.message


def test_a_missing_error_field_does_not_read_as_success():
    """`{"ok": false}` with nothing else. Falling through to a generic
    error is right; falling through to `return body` is the empty-result
    bug wearing a different hat."""
    verdict = _verdict({"ok": False})
    assert isinstance(verdict, ConnectorToolError)


# ── Rendering ────────────────────────────────────────────────────────


def test_mentions_become_names():
    out = _render_text("hey <@U04J1F2> can you look?", {"U04J1F2": "Sara"})
    assert out == "hey @Sara can you look?"


def test_an_unresolved_mention_degrades_to_the_id_not_to_nothing():
    out = _render_text("ping <@U04J1F2>", {})
    assert "U04J1F2" in out


def test_links_keep_their_label_and_their_url():
    out = _render_text("see <https://toup.ai/docs|the docs>", {})
    assert out == "see the docs (https://toup.ai/docs)"
    assert _render_text("see <https://toup.ai>", {}) == "see https://toup.ai"


def test_channel_refs_and_broadcasts_render_readably():
    assert _render_text("moved to <#C01ABCDEF|eng>", {}) == "moved to #eng"
    assert _render_text("<!here> standup", {}) == "@here standup"


def test_slack_entity_escapes_are_undone():
    assert _render_text("a &lt;b&gt; &amp; c", {}) == "a <b> & c"


def test_speaker_ids_collects_authors_and_mentions():
    """Both, because a message whose author is cached but whose mentions
    are not still renders half in ids."""
    ids = _speaker_ids([
        {"user": "U1", "text": "hi <@U2>"},
        {"user": "U3", "text": "yo"},
    ])
    assert set(ids) == {"U1", "U2", "U3"}


def test_a_bot_message_has_no_user_and_must_not_render_as_blank():
    row = _message_row({"ts": "1.0", "username": "GitHub", "text": "deployed"}, {})
    assert row["from"] == "GitHub"
    row = _message_row({"ts": "1.0", "text": "x"}, {})
    assert row["from"] == "(app)"


def test_a_threaded_parent_advertises_its_thread():
    """The channel view returns only the parent, so without `thread_ts`
    on the row the agent cannot ask for the 40 replies underneath."""
    row = _message_row({"ts": "170.5", "user": "U1", "text": "ship it?",
                        "reply_count": 40}, {"U1": "Sara"})
    assert row["reply_count"] == 40
    assert row["thread_ts"] == "170.5"


def test_a_dm_row_carries_the_person_not_a_missing_name():
    """A 1:1 conversation has no `name` field at all. Keyed on `name`
    alone every DM renders as null — which is most of a person's Slack."""
    row = _channel_row({"id": "D1", "is_im": True, "user": "U9"}, {"U9": "Sara"})
    assert row["type"] == "im"
    assert row["user_name"] == "Sara"
    assert "name" not in row


def test_private_and_public_channels_are_distinguishable():
    pub = _channel_row({"id": "C1", "name": "general", "is_member": True}, {})
    priv = _channel_row({"id": "G1", "name": "founders", "is_private": True}, {})
    assert pub["type"] == "public_channel"
    assert priv["type"] == "private_channel"


def test_clamp_holds_slacks_ceilings_against_junk():
    assert _clamp(5000, 30, 1, 200) == 200
    assert _clamp(0, 30, 1, 200) == 1
    assert _clamp("many", 30, 1, 200) == 30
    assert _clamp(None, 30, 1, 200) == 30
