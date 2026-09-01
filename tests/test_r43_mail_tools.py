"""R43 W3-C — the mail tool surface the pickers and the delivery need.

Every test here exists because an option in the design had nothing to
write to. Four gaps, and each one is a REQUEST that changes, not a
label:

  1. THE LIST NEVER SAID UNREAD — `gmail__list_messages` inlines headers
     and the snippet with `include_body` and dropped `labelIds`, the one
     field that answers "is this unread". The contents reader paid a
     SECOND `is:unread in:inbox` read on every popup open to get it back.
     The per-message GET was already carrying them.

  2. A USER'S OWN LABELS / FOLDERS COULD NOT BE NAMED — no
     `gmail__list_labels`, no `outlook__list_folders`, so §5's "one
     source per label / per folder" could only ever offer the three
     well-known names. Worse on Outlook: `query` is a KQL `$search`,
     which cannot name a folder at all, so a folder pin produced the
     same account-wide read as no pin — a picker that wrote nowhere.

  3. "ADDRESSED TO ME" HAD NO ME — Outlook stores no mailbox address
     (`provider_account_id` is NULL for this connector by construction),
     so §6's `me` chip had nothing to search for.

  4. A FORMAT WITH A FILE IN ITS NAME ARRIVED AS TEXT — `brief_render`
     produces real bytes for the PDF, the CSV and the markdown file, and
     neither draft tool could carry a file, so all three reached mail as
     a body paste. "One-page PDF" was the name of something that did not
     exist.

The refusal cases are load-bearing in the same way: a draft that quietly
arrives WITHOUT the file the user picked is worse than one that refuses,
because the run reports success either way.
"""

from __future__ import annotations

import base64
import json

import pytest

from app.connectors import base as cbase
from app.connectors.base import ConnectorContext, ConnectorOk, ConnectorToolError
from app.connectors.gmail import provider as gm
from app.connectors.outlook import provider as ol


def _ctx() -> ConnectorContext:
    return ConnectorContext(user_id="u1", access_token="tok")


def _content(result) -> dict:
    assert isinstance(result, ConnectorOk), result
    return json.loads(result.content)


# ── 1. Gmail: the list finally says unread ───────────────────────────


@pytest.mark.asyncio
async def test_gmail_list_with_bodies_returns_labelids(monkeypatch):
    async def _req(method, url, **kw):
        if url.endswith("/messages"):
            return {"messages": [{"id": "m1", "threadId": "t1"}],
                    "resultSizeEstimate": 1}
        return {
            "id": "m1", "threadId": "t1",
            "labelIds": ["INBOX", "UNREAD"],
            "payload": {"headers": [{"name": "Subject", "value": "Hi"}],
                        "mimeType": "text/plain", "body": {}},
            "snippet": "hello",
        }

    monkeypatch.setattr(gm, "google_request", _req)
    out = _content(await gm.GmailProvider().execute(
        "gmail__list_messages", {"max_results": 1}, _ctx()))
    # The whole point: no second read is needed to know this is unread.
    assert out["messages"][0]["labelIds"] == ["INBOX", "UNREAD"]


@pytest.mark.asyncio
async def test_a_message_with_no_labels_answers_an_empty_list(monkeypatch):
    """Absent ≠ empty is the round's rule, and it cuts the other way for
    a row: `[]` is "Gmail said none", never a missing key the caller has
    to `.get()` around."""
    async def _req(method, url, **kw):
        if url.endswith("/messages"):
            return {"messages": [{"id": "m1"}]}
        return {"id": "m1", "payload": {"headers": []}}

    monkeypatch.setattr(gm, "google_request", _req)
    out = _content(await gm.GmailProvider().execute(
        "gmail__list_messages", {}, _ctx()))
    assert out["messages"][0]["labelIds"] == []


# ── 2a. Gmail: the user's own labels ─────────────────────────────────


_LABELS = {"labels": [
    {"id": "Label_9", "name": "clients", "type": "user"},
    {"id": "INBOX", "name": "INBOX", "type": "system"},
    {"id": "Label_2", "name": "Alpha", "type": "user"},
    {"name": "orphan with no id"},
]}


@pytest.mark.asyncio
async def test_gmail_labels_are_system_first_then_alphabetical(monkeypatch):
    seen = []

    async def _req(method, url, **kw):
        seen.append(url)
        if url.endswith("/labels"):
            return dict(_LABELS)
        lid = url.rsplit("/", 1)[-1]
        return {"id": lid, "messagesTotal": 7, "messagesUnread": 2}

    monkeypatch.setattr(gm, "google_request", _req)
    out = _content(await gm.GmailProvider().execute(
        "gmail__list_labels", {}, _ctx()))
    assert [lab["id"] for lab in out["labels"]] == [
        "INBOX", "Label_2", "Label_9"]
    # Counts are a `users.labels.get` field — the list call carries
    # none, whatever the contract table says — so one GET per row.
    assert sum(1 for u in seen if "/labels/" in u) == 3
    assert out["labels"][0]["messages_unread"] == 2


@pytest.mark.asyncio
async def test_gmail_labels_without_counts_makes_no_extra_reads(monkeypatch):
    seen = []

    async def _req(method, url, **kw):
        seen.append(url)
        return dict(_LABELS)

    monkeypatch.setattr(gm, "google_request", _req)
    out = _content(await gm.GmailProvider().execute(
        "gmail__list_labels", {"include_counts": False}, _ctx()))
    assert len(seen) == 1
    assert out["labels"][0]["messages_total"] is None


@pytest.mark.asyncio
async def test_a_label_whose_count_fails_says_none_not_zero(monkeypatch):
    """"0 unread" and "we could not count" read identically on screen
    and mean opposite things."""
    async def _req(method, url, **kw):
        if url.endswith("/labels"):
            return {"labels": [{"id": "INBOX", "name": "INBOX",
                                "type": "system"}]}
        raise gm._GoogleConnectorError(
            ConnectorToolError(message="boom", retryable=True))

    monkeypatch.setattr(gm, "google_request", _req)
    out = _content(await gm.GmailProvider().execute(
        "gmail__list_labels", {}, _ctx()))
    assert out["labels"][0]["messages_unread"] is None


# ── 2b. Outlook: folders, and a folder pin that actually scopes ──────


def test_a_folder_scopes_the_collection_not_the_search():
    assert ol._messages_url({}) == f"{ol.GRAPH_API}/me/messages"
    assert ol._messages_url({"folder": "inbox"}) == (
        f"{ol.GRAPH_API}/me/mailFolders/inbox/messages")


def test_a_graph_folder_id_survives_the_path_segment():
    # Graph folder ids are base64url with `=` padding; unencoded they
    # would change meaning in a path.
    fid = "AAMkAD_x-y=="
    assert ol._messages_url({"folder": fid}).endswith(
        "/me/mailFolders/AAMkAD_x-y%3D%3D/messages")


_FOLDER_LISTING = {"value": [
    {"id": "f1", "displayName": "Vendors", "unreadItemCount": 2,
     "totalItemCount": 9, "childFolderCount": 0},
    {"id": "fi", "displayName": "Boîte de réception", "unreadItemCount": 18,
     "totalItemCount": 400, "childFolderCount": 0},
    {"displayName": "no id at all"},
]}


@pytest.mark.asyncio
async def test_outlook_folders_are_named_by_id_not_by_display_name(monkeypatch):
    """`displayName` is localised — "Inbox" is "Boîte de réception" on a
    French mailbox — and `wellKnownName` is beta-only, so the v1.0 way
    to know which folder is the inbox is to ask for it by the well-known
    segment and compare ids. Matching the English string would list the
    inbox twice and find the well-known row nowhere."""
    async def _req(method, url, **kw):
        if url.endswith("/me/mailFolders"):
            return dict(_FOLDER_LISTING)
        if url.endswith("/inbox"):
            return {"id": "fi"}
        raise ol._MicrosoftConnectorError(
            ConnectorToolError(message="no such folder", retryable=False))

    monkeypatch.setattr(ol, "microsoft_graph_request", _req)
    out = _content(await ol.OutlookProvider().execute(
        "outlook__list_folders", {}, _ctx()))
    # The inbox leads, tagged by id; the user's own folder follows; the
    # row with no id is dropped (it cannot be picked or pinned).
    assert [f["id"] for f in out["folders"]] == ["fi", "f1"]
    assert out["folders"][0]["well_known"] == "inbox"
    assert out["folders"][0]["unread_count"] == 18
    assert out["folders"][1] == {
        "id": "f1", "name": "Vendors", "unread_count": 2,
        "total_count": 9, "child_count": 0, "well_known": None}


@pytest.mark.asyncio
async def test_a_mailbox_with_no_archive_still_lists(monkeypatch):
    """A missing well-known folder is a real shape (older Exchange),
    not a failure."""
    async def _req(method, url, **kw):
        if url.endswith("/me/mailFolders"):
            return {"value": [{"id": "f1", "displayName": "Vendors"}]}
        raise ol._MicrosoftConnectorError(
            ConnectorToolError(message="not found", retryable=False))

    monkeypatch.setattr(ol, "microsoft_graph_request", _req)
    out = _content(await ol.OutlookProvider().execute(
        "outlook__list_folders", {}, _ctx()))
    assert [f["id"] for f in out["folders"]] == ["f1"]
    assert out["folders"][0]["well_known"] is None


@pytest.mark.asyncio
async def test_a_read_with_a_folder_hits_that_folders_collection(monkeypatch):
    seen = {}

    async def _req(method, url, **kw):
        seen["url"] = url
        return {"value": []}

    monkeypatch.setattr(ol, "microsoft_graph_request", _req)
    await ol.OutlookProvider().execute(
        "outlook__list_messages", {"folder": "archive"}, _ctx())
    assert seen["url"].endswith("/me/mailFolders/archive/messages")


# ── 3. Outlook: "addressed to me" has a me ───────────────────────────


def test_to_me_becomes_a_kql_recipient_restriction():
    params, _scan, _limit = ol._list_messages_params(
        {"to_me": True}, "sam@contoso.com")
    assert params["$search"] == '"to:sam@contoso.com"'
    # Graph refuses $filter/$orderby beside a search — the existing
    # contract, unchanged by the new term.
    assert "$orderby" not in params and "$filter" not in params


def test_to_me_composes_with_the_users_own_query_and_since():
    params, _scan, _limit = ol._list_messages_params(
        {"to_me": True, "query": "subject:invoice",
         "since": "2026-08-30T00:00:00Z"}, "sam@contoso.com")
    assert params["$search"] == (
        '"to:sam@contoso.com subject:invoice received>=2026-08-30"')


def test_an_unresolvable_mailbox_narrows_nothing_rather_than_guessing():
    """Searching for the literal word "me" would return an arbitrary
    slice of the mailbox under a lit "Addressed to me" chip."""
    params, _scan, _limit = ol._list_messages_params({"to_me": True}, "")
    assert "$search" not in params
    assert params["$orderby"] == "receivedDateTime desc"


@pytest.mark.asyncio
async def test_the_mailbox_address_is_asked_for_once_per_process(monkeypatch):
    ol._MAILBOX_CACHE.clear()
    calls = []

    async def _req(method, url, **kw):
        calls.append(url)
        if url.endswith("/me"):
            return {"mail": "sam@contoso.com"}
        return {"value": []}

    monkeypatch.setattr(ol, "microsoft_graph_request", _req)
    for _ in range(2):
        await ol.OutlookProvider().execute(
            "outlook__list_messages", {"to_me": True}, _ctx())
    assert sum(1 for u in calls if u.endswith("/me")) == 1
    ol._MAILBOX_CACHE.clear()


@pytest.mark.asyncio
async def test_a_read_without_to_me_never_asks_who_you_are(monkeypatch):
    ol._MAILBOX_CACHE.clear()
    calls = []

    async def _req(method, url, **kw):
        calls.append(url)
        return {"value": []}

    monkeypatch.setattr(ol, "microsoft_graph_request", _req)
    await ol.OutlookProvider().execute("outlook__list_messages", {}, _ctx())
    assert not [u for u in calls if u.endswith("/me")]


# ── 4. Carrying a file ───────────────────────────────────────────────


_PDF = b"%PDF-1.4 tiny"


def _att(name="brief.pdf", mime="application/pdf", blob=_PDF):
    return {"filename": name, "content_type": mime,
            "content_base64": base64.b64encode(blob).decode("ascii")}


def test_a_plain_draft_is_still_a_single_part_message():
    """The shape every existing draft and send has produced. A mail that
    gained a MIME envelope it did not need would be a change nobody
    asked for."""
    raw = gm._build_rfc822(to="a@b.c", subject="Hi", body="text")
    assert "multipart/mixed" not in raw
    assert "MIME-Version" not in raw
    assert raw.endswith("\r\n\r\ntext")


def test_an_attached_brief_rides_as_a_base64_mime_part():
    raw = gm._build_rfc822(
        to="a@b.c", subject="Hi", body="text",
        attachments=[("brief.pdf", "application/pdf", _PDF)])
    assert "MIME-Version: 1.0" in raw
    assert "multipart/mixed; boundary=" in raw
    assert 'Content-Disposition: attachment; filename="brief.pdf"' in raw
    assert "Content-Transfer-Encoding: base64" in raw
    # The body is still the first part, unchanged.
    assert "\r\nContent-Type: text/plain; charset=utf-8\r\n\r\ntext\r\n" in raw
    payload = base64.b64encode(_PDF).decode("ascii")
    assert payload in raw.replace("\r\n", "")
    # RFC 2045 caps an encoded line at 76 characters.
    assert all(len(line) <= 998 for line in raw.split("\r\n"))


def test_a_long_attachment_is_wrapped_at_76_columns():
    raw = gm._build_rfc822(
        to="a@b.c", subject="Hi", body="t",
        attachments=[("big.bin", "application/octet-stream", b"x" * 900)])
    b64_lines = [ln for ln in raw.split("\r\n")
                 if ln and set(ln) <= set(
                     "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                     "abcdefghijklmnopqrstuvwxyz0123456789+/=")]
    assert b64_lines and max(len(ln) for ln in b64_lines) == 76


def test_a_filename_cannot_inject_a_header():
    assert gm._safe_filename('a\r\nBcc: x@y.z"') == "aBcc: x@y.z"
    assert gm._safe_filename("../../etc/passwd") == "....etcpasswd"
    assert gm._safe_filename("   ") == "attachment"
    assert ol._safe_filename("a\r\nb") == "ab"


@pytest.mark.parametrize("bad,why", [
    ("nope", "attachments must be a list"),
    ([{"filename": "a"}], "each attachment needs content_base64"),
    ([{"filename": "a", "content_base64": "!!!!"}],
     "content_base64 is not valid base64"),
    ([_att(), _att(), _att(), _att()], "at most 3 attachments"),
    (["a string"], "each attachment must be an object"),
])
def test_a_malformed_attachment_is_a_message_in_both_providers(bad, why):
    for parse in (gm._parse_attachments, ol._graph_attachments):
        files, err = parse(bad)
        assert files == [] and err == why, parse


def test_the_cap_is_the_same_number_on_both_sides():
    """Graph's inline ceiling. A delivery that succeeds on one mail
    channel and refuses on the other is a channel picker that lies."""
    assert gm.MAX_ATTACHMENT_BYTES == ol.MAX_ATTACHMENT_BYTES == 3 * 1024 * 1024
    assert gm.MAX_ATTACHMENTS == ol.MAX_ATTACHMENTS == 3
    big = _att(blob=b"z" * (3 * 1024 * 1024 + 1))
    for parse in (gm._parse_attachments, ol._graph_attachments):
        assert parse([big])[1] == "attachments exceed 3 MB"


def test_nothing_to_carry_is_not_an_error():
    for parse in (gm._parse_attachments, ol._graph_attachments):
        assert parse(None) == ([], "")
        assert parse([]) == ([], "")


@pytest.mark.asyncio
async def test_a_gmail_draft_carries_the_file_into_the_raw_message(monkeypatch):
    seen = {}

    async def _req(method, url, *, json_body=None, **kw):
        seen["body"] = json_body
        return {"id": "d1", "message": {"id": "m1", "threadId": "t1"}}

    monkeypatch.setattr(gm, "google_request", _req)
    out = _content(await gm.GmailProvider().execute(
        "gmail__create_draft",
        {"to": "me@x.io", "subject": "Brief", "body": "see attached",
         "attachments": [_att()]},
        _ctx()))
    assert out["attachments"] == ["brief.pdf"]
    raw = seen["body"]["message"]["raw"]
    decoded = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode()
    assert "multipart/mixed" in decoded
    assert base64.b64encode(_PDF).decode("ascii") in decoded.replace("\r\n", "")


@pytest.mark.asyncio
async def test_a_gmail_draft_refuses_rather_than_dropping_the_file(monkeypatch):
    called = []

    async def _req(*a, **kw):
        called.append(a)
        return {}

    monkeypatch.setattr(gm, "google_request", _req)
    result = await gm.GmailProvider().execute(
        "gmail__create_draft",
        {"to": "me@x.io", "subject": "Brief", "body": "b",
         "attachments": [{"filename": "x", "content_base64": "!!!"}]},
        _ctx())
    assert isinstance(result, ConnectorToolError)
    # Nothing was written: a draft missing its file would still report
    # success to the run.
    assert not called


@pytest.mark.asyncio
async def test_an_outlook_draft_carries_a_fileattachment_and_is_read(monkeypatch):
    seen = {}

    async def _req(method, url, *, json_body=None, **kw):
        seen["body"] = json_body
        return {"id": "d1", "webLink": "https://o", "isRead": True}

    monkeypatch.setattr(ol, "microsoft_graph_request", _req)
    out = _content(await ol.OutlookProvider().execute(
        "outlook__create_draft",
        {"to": "me@x.io", "subject": "Brief", "body": "see attached",
         "is_read": True, "attachments": [_att()]},
        _ctx()))
    body = seen["body"]
    assert body["isRead"] is True
    assert body["attachments"] == [{
        "@odata.type": "#microsoft.graph.fileAttachment",
        "name": "brief.pdf",
        "contentType": "application/pdf",
        "contentBytes": base64.b64encode(_PDF).decode("ascii"),
    }]
    assert out["attachments"] == ["brief.pdf"]


@pytest.mark.asyncio
async def test_an_outlook_draft_left_alone_sets_no_read_state(monkeypatch):
    seen = {}

    async def _req(method, url, *, json_body=None, **kw):
        seen["body"] = json_body
        return {"id": "d1"}

    monkeypatch.setattr(ol, "microsoft_graph_request", _req)
    await ol.OutlookProvider().execute(
        "outlook__create_draft",
        {"to": "me@x.io", "subject": "S", "body": "b"}, _ctx())
    assert "isRead" not in seen["body"]
    assert "attachments" not in seen["body"]


@pytest.mark.asyncio
async def test_an_outlook_draft_refuses_rather_than_dropping_the_file(monkeypatch):
    called = []

    async def _req(*a, **kw):
        called.append(a)
        return {}

    monkeypatch.setattr(ol, "microsoft_graph_request", _req)
    result = await ol.OutlookProvider().execute(
        "outlook__create_draft",
        {"to": "me@x.io", "subject": "S", "body": "b",
         "attachments": [{"filename": "x", "content_base64": "??"}]},
        _ctx())
    assert isinstance(result, ConnectorToolError)
    assert not called


# ── the manifests declare exactly what the providers dispatch ────────


def test_every_new_tool_is_both_declared_and_dispatchable():
    """A tool declared but not dispatchable, or dispatchable but not
    declared, is worse than none."""
    import yaml
    from pathlib import Path

    root = Path(gm.__file__).parent.parent
    for cid, new_tools in (
        ("gmail", {"gmail__list_labels"}),
        ("outlook", {"outlook__list_folders"}),
    ):
        man = yaml.safe_load((root / cid / "manifest.yaml").read_text())
        names = {t["name"] for t in man["tools"]}
        assert new_tools <= names, cid
        src = (root / cid / "provider.py").read_text()
        for tool in new_tools:
            assert f'tool_name == "{tool}"' in src, tool


def test_the_new_inputs_are_declared_where_the_provider_reads_them():
    import yaml
    from pathlib import Path

    root = Path(gm.__file__).parent.parent
    gmail = yaml.safe_load((root / "gmail" / "manifest.yaml").read_text())
    outlook = yaml.safe_load((root / "outlook" / "manifest.yaml").read_text())

    def props(man, tool):
        for t in man["tools"]:
            if t["name"] == tool:
                return (t["input_schema"].get("properties") or {})
        raise AssertionError(tool)

    assert "attachments" in props(gmail, "gmail__create_draft")
    assert "attachments" in props(outlook, "outlook__create_draft")
    assert "is_read" in props(outlook, "outlook__create_draft")
    for p in ("folder", "to_me"):
        assert p in props(outlook, "outlook__list_messages"), p
    # Nothing is sent on the user's behalf: the send tools stay
    # file-free, so this round widened no exfiltration surface.
    assert "attachments" not in props(gmail, "gmail__send_message")
    assert "attachments" not in props(outlook, "outlook__send_message")


def test_no_new_oauth_scope_was_requested():
    """A new scope invalidates every existing grant and makes every user
    reconnect. Labels ride `gmail.readonly`, folders and `to_me` ride
    `Mail.Read` + `User.Read`, attachments ride the draft scopes already
    asked for."""
    import yaml
    from pathlib import Path

    root = Path(gm.__file__).parent.parent
    gmail = yaml.safe_load((root / "gmail" / "manifest.yaml").read_text())
    outlook = yaml.safe_load((root / "outlook" / "manifest.yaml").read_text())
    assert set(gmail["oauth"]["scopes"]) == {
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.send",
    }
    assert set(gmail["oauth"]["scopes_optional"]) == {
        "https://www.googleapis.com/auth/gmail.compose"}
    assert set(outlook["oauth"]["scopes"]) == {
        "https://graph.microsoft.com/Mail.Read",
        "https://graph.microsoft.com/Mail.ReadWrite",
        "https://graph.microsoft.com/Mail.Send",
        "https://graph.microsoft.com/User.Read",
        "offline_access",
    }

