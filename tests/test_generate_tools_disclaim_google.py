"""A tool's own description outranks a system-prompt bullet 3,700 lines away.

Found 2026-08-13, on a Google OAuth verification recording. The user asked
for "a new Google Doc". The agent did the right thing AND the wrong thing in
one turn: it created a real Google Doc via `drive__create_doc` (returning
only `{id, title, url}` — that tool produces no attachment), and it ALSO
called `generate_docx`, stapling a stray `x.docx` beside it. The Word file
then failed to preview, so the visible result was a broken duplicate next to
a correct answer.

PR #603 had already added routing guidance to `_build_system_prompt`. It was
present, un-gated, and on the running image. It lost anyway — because
`generate_docx` described itself as:

    "Use when the user wants an editable document (vs. a read-only PDF)."

A Google Doc IS an editable document. The tool's own description matched the
request almost word for word and never mentioned Google, and a tool
description is read at the moment of choosing, while a system-prompt bullet
is thousands of tokens of context away.

So the exclusion has to live on the tool. Guidance in the prompt is the
backstop, not the mechanism.
"""

from __future__ import annotations

import os

os.environ.setdefault("ENVIRONMENT", "test")


def _tool(name: str) -> dict:
    from app.agent.tool_definitions import get_doc_generation_tools

    for t in get_doc_generation_tools():
        if t.get("name") == name:
            return t
    raise AssertionError(f"tool {name!r} not found")


def _desc(name: str) -> str:
    return _tool(name)["description"].lower()


def test_generate_docx_names_the_google_alternative():
    """Naming the CORRECT tool is what redirects the call. 'Do not do X'
    without 'do Y instead' leaves the model with the same best option."""
    d = _desc("generate_docx")
    assert "google doc" in d, "generate_docx never mentions Google Docs"
    assert "docs__create" in d or "drive__create_doc" in d, (
        "generate_docx warns about Google Docs but does not name the tool to "
        "use instead — the model is left with this tool as its best match"
    )


def test_generate_docx_forbids_calling_BOTH():
    """The bug was not choosing wrongly — it was doing both in one turn.
    A description that only says 'prefer X' permits the duplicate."""
    d = _desc("generate_docx")
    assert "not also call" in d or "do not also" in d, (
        "nothing forbids calling generate_docx ALONGSIDE the Google tool, "
        "which is the exact failure: a stray .docx beside a real Doc"
    )


def test_generate_xlsx_names_the_google_alternative():
    d = _desc("generate_xlsx")
    assert "google sheet" in d, "generate_xlsx never mentions Google Sheets"
    assert "sheets__create_spreadsheet" in d or "sheets__append_rows" in d, (
        "generate_xlsx warns about Google Sheets but names no replacement"
    )


def test_generate_xlsx_forbids_calling_BOTH():
    d = _desc("generate_xlsx")
    assert "not also call" in d or "do not also" in d


def test_the_descriptions_say_where_the_file_LANDS():
    """The user's actual confusion is location: a generated file is in the
    chat, a Google Doc is in their Drive. If the description does not say
    where the output goes, 'editable document' matches both."""
    for name in ("generate_docx", "generate_xlsx"):
        d = _desc(name)
        assert "in this chat" in d, (
            f"{name} does not say the file lands IN THE CHAT, so it still "
            f"reads as a valid way to satisfy 'make me a doc in my Drive'"
        )
        assert "drive" in d, f"{name} never contrasts itself with Drive"
