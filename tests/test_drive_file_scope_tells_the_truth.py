"""A per-file scope must never let the agent deny a file's existence.

Found 2026-08-12, from the same founder session as the Google-Doc
routing bug. The user asked the agent to add a row to their expenses
sheet. The agent called `drive__list_files` twice, got `{"files": []}`
both times, and replied:

    "I couldn't find an Expenses spreadsheet in the connected Drive,
     so I didn't add anything."

The spreadsheet existed. `drive.file` is Google's PER-FILE scope —
"access to files created or opened by the app" — so `files.list`
returns only what this app made, and `files.get` 404s on everything
else. The user's own files are structurally invisible.

That makes this different from the two tools deleted on 2026-08-11.
Those raised 403 and failed loudly. This one returns HTTP 200 with an
empty array, which is a well-formed, successful, WRONG answer — and
the model has no way to tell it apart from a real "no such file". It
then states the false negative to the user as fact.

The rule: where a scope bounds what a read can SEE, the boundary ships
with the response, not just in a comment. A caller that cannot
distinguish "absent" from "invisible" will pick the wrong one, and it
picks it in the direction that destroys trust.

`test_list_files_ships_the_scope_boundary_in_its_payload` is the
behavioural check; the rest pin the copy that the model and the user
read before the call is ever made.
"""

from __future__ import annotations

import json
import pathlib

import pytest
import yaml

DRIVE = pathlib.Path("app/connectors/drive")


def _manifest() -> dict:
    return yaml.safe_load((DRIVE / "manifest.yaml").read_text())


def _tool(name: str) -> dict:
    return next(t for t in _manifest()["tools"] if t["name"] == name)


def _provider_src() -> str:
    return (DRIVE / "provider.py").read_text()


@pytest.mark.asyncio
async def test_list_files_ships_the_scope_boundary_in_its_payload(monkeypatch):
    """An empty list must arrive WITH the reason it might be empty."""
    from app.connectors.base import ConnectorContext, ConnectorOk
    from app.connectors.drive import provider as drive_provider

    async def fake_request(method, url, **kw):
        return {"files": []}

    monkeypatch.setattr(drive_provider, "google_request", fake_request)

    result = await drive_provider.DriveProvider().execute(
        "drive__list_files",
        {},
        ConnectorContext(user_id="u1", access_token="tok"),
    )
    assert isinstance(result, ConnectorOk)
    payload = json.loads(result.content)

    assert payload["files"] == []
    assert payload.get("scope") == "drive.file"
    note = (payload.get("note") or "").lower()
    assert note, (
        "an empty drive__list_files result carries no explanation — the agent "
        "cannot tell 'you have no files' from 'you cannot see their files', "
        "and it guessed wrong in production"
    )
    assert "does not exist" in note or "never tell them" in note, (
        "the note must forbid the specific wrong answer ('I couldn't find it'), "
        "not merely describe the scope"
    )


@pytest.mark.asyncio
async def test_non_empty_list_carries_the_boundary_too(monkeypatch):
    """A partial list is the same lie, quieter.

    Two app-created files presented as 'your Drive' is as wrong as zero
    presented as 'you have none' — so the note is unconditional.
    """
    from app.connectors.base import ConnectorContext, ConnectorOk
    from app.connectors.drive import provider as drive_provider

    async def fake_request(method, url, **kw):
        return {"files": [{"id": "1", "name": "Notes"}]}

    monkeypatch.setattr(drive_provider, "google_request", fake_request)

    result = await drive_provider.DriveProvider().execute(
        "drive__list_files", {}, ConnectorContext(user_id="u1", access_token="tok"),
    )
    payload = json.loads(ConnectorOk and result.content)
    assert payload.get("note"), "the boundary is dropped as soon as anything matches"


def test_get_file_text_reads_404_as_no_access():
    """Google's 404 body says 'File not found'. That phrasing, passed
    through, is what makes the agent announce a deletion."""
    src = _provider_src()
    assert 'msg.startswith("404")' in src, (
        "drive__get_file_text no longer special-cases 404 — the raw "
        "'File not found' reaches the model and reads as deletion"
    )
    assert "NO ACCESS" in src


def test_list_files_description_does_not_promise_search():
    """The model picks tools from descriptions. 'List Drive files
    matching a query' reads as Drive search, which this cannot do."""
    desc = _tool("drive__list_files")["description"].lower()
    assert "not a search" in desc or "not a search of" in desc, (
        "the description must deny being a Drive search outright"
    )
    assert "does not exist" in desc, (
        "it must name the wrong conclusion an empty result invites"
    )


def test_connector_card_does_not_promise_to_find_the_users_files():
    """`short_description` is the card copy on /connectors — the promise
    the USER reads before granting anything. It said 'Find files'."""
    card = _manifest()["short_description"].lower()
    assert not card.startswith("find files"), (
        "the Drive card promises file-finding that drive.file cannot do"
    )


def test_drive_still_requests_only_the_per_file_scope():
    """The fix must not become 'ask for drive.readonly'.

    That scope is RESTRICTED: it would put the project on an annual paid
    CASA assessment and invalidate the in-flight verification. The whole
    point of the change above is to be honest within drive.file.
    """
    oauth = _manifest()["oauth"]
    assert oauth["scopes"] == ["https://www.googleapis.com/auth/drive.file"]
    optional = oauth.get("scopes_optional") or []
    assert "https://www.googleapis.com/auth/drive.readonly" in optional, (
        "drive.readonly must stay OPTIONAL — never promoted into scopes"
    )
