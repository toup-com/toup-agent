"""A real Google Doc and a stray .docx must never ship together.

Found 2026-08-13, on a Google OAuth verification recording. "Create a new
Google Doc" produced a correct Doc AND a `x.docx` beside it — a file titled
"X", containing nothing but that heading, because the model passed a
throwaway filename for a file it never meant to make.

Two earlier attempts were PERSUASION and both failed:

  #603 — routing guidance in `_build_system_prompt`. Present, un-gated, on
         the running image. Ignored.
  #631 — the exclusion moved onto the tool descriptions, which is much more
         proximate. Better, but still advisory.

The same request produced three different answers across three attempts
(`x.docx`, `Toup Verification Notes.docx`, `x.docx`), which is the signature
of a probabilistic fix. So the last line is MECHANICAL: once a Google doc
creation tool has succeeded this turn, the matching generate_* tool refuses.

The model cannot opt out of this one.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

os.environ.setdefault("ENVIRONMENT", "test")


@contextmanager
def _no_disk():
    """`gen_*` writes through `_persist` to the agent workspace, which does not
    exist off-container. Stub the write so these tests measure the GUARD, not
    the filesystem."""
    att = {
        "id": "a1",
        "filename": "x.docx",
        "mime_type": "application/octet-stream",
        "size_bytes": 10,
        "storage_path": "u1/a1_x.docx",
        "created_at": "2026-08-13T00:00:00Z",
    }
    with patch("app.agent.doc_generators._persist", new=AsyncMock(return_value=att)):
        yield


def _ex():
    """A ToolExecutor with only the state these handlers touch."""
    from app.agent.tool_executor import ToolExecutor

    ex = ToolExecutor.__new__(ToolExecutor)
    ex.pending_attachments = []
    ex.google_docs_created_this_run = set()
    ex._user_id = "u1"
    return ex


def _docx(ex, **inp):
    with _no_disk():
        return asyncio.run(
            ex._tool_generate_docx(inp or {"content": "hi", "filename": "x"})
        )


def _xlsx(ex, **inp):
    with _no_disk():
        return asyncio.run(
            ex._tool_generate_xlsx(inp or {"sheets": [], "filename": "x"})
        )


def test_docx_is_refused_after_a_google_doc_was_created():
    ex = _ex()
    ex.google_docs_created_this_run.add("drive__create_doc")
    out = _docx(ex)
    assert out.startswith("SKIPPED:"), out
    assert not ex.pending_attachments, (
        "a .docx was still attached beside the real Doc — this is the exact "
        "broken duplicate the user saw"
    )


def test_docs__create_also_blocks_it():
    """Two tools create a Doc. Covering only one leaves the bug reachable."""
    ex = _ex()
    ex.google_docs_created_this_run.add("docs__create")
    assert _docx(ex).startswith("SKIPPED:")
    assert not ex.pending_attachments


def test_xlsx_is_refused_after_a_google_sheet_was_created():
    ex = _ex()
    ex.google_docs_created_this_run.add("sheets__create_spreadsheet")
    out = _xlsx(ex)
    assert out.startswith("SKIPPED:"), out
    assert not ex.pending_attachments


def test_a_sheet_does_not_block_a_docx():
    """Cross-type blocking would refuse a legitimate ask: 'make the Sheet,
    and also give me a Word summary'. Only the MATCHING type is refused."""
    ex = _ex()
    ex.google_docs_created_this_run.add("sheets__create_spreadsheet")
    out = _docx(ex)
    assert not out.startswith("SKIPPED:"), (
        "creating a Google Sheet blocked an unrelated .docx"
    )


def test_a_doc_does_not_block_an_xlsx():
    ex = _ex()
    ex.google_docs_created_this_run.add("docs__create")
    out = _xlsx(ex)
    assert not out.startswith("SKIPPED:")


def test_appending_to_an_existing_doc_does_not_block_anything():
    """`docs__append_text` writes into a doc the user already has. That is not
    a reason to refuse a download they may genuinely have asked for."""
    from app.agent.tool_executor import ToolExecutor

    assert "docs__append_text" not in ToolExecutor._GOOGLE_DOC_CREATORS
    assert "sheets__append_rows" not in ToolExecutor._GOOGLE_DOC_CREATORS


def test_with_no_google_doc_the_docx_is_generated_normally():
    """The guard must be inert in the ordinary case — this tool is how every
    non-Google document in the product gets made."""
    ex = _ex()
    out = _docx(ex)
    assert not out.startswith("SKIPPED:"), out
    assert len(ex.pending_attachments) == 1


def test_the_refusal_is_not_an_ERROR_prefix():
    """`ERROR:` is a contract: `_meter_flat_tool` reads it to decide the call
    did no billable work, and the model treats it as retryable. A refusal is
    neither — mislabelling it invites the model to try again in a loop."""
    ex = _ex()
    ex.google_docs_created_this_run.add("docs__create")
    assert not _docx(ex).startswith("ERROR:")


def test_the_refusal_tells_the_model_what_to_do_instead():
    """A bare 'no' leaves the model to improvise, usually by apologising to
    the user about a file they never asked for."""
    ex = _ex()
    ex.google_docs_created_this_run.add("docs__create")
    out = _docx(ex).lower()
    assert "link" in out and "google doc" in out


def test_the_dispatch_point_actually_RECORDS_a_google_creation():
    """The tests above set the flag by hand, so every one of them passes with
    the recording line deleted — and the guard would then never fire in
    production. Mutation D5 proved exactly that. This reads `execute` and
    asserts the write exists, is keyed off `_GOOGLE_DOC_CREATORS`, and is
    gated on the call having SUCCEEDED (a failed Drive call must not suppress
    the local file the user then has nothing else to fall back on).
    """
    import inspect
    from app.agent.tool_executor import ToolExecutor

    src = inspect.getsource(ToolExecutor.execute)
    assert "google_docs_created_this_run.add(tool_name)" in src, (
        "the dispatch point never records a successful Google doc creation, "
        "so the refusal in _tool_generate_docx can never trigger"
    )
    assert "_GOOGLE_DOC_CREATORS" in src, (
        "the recording is not keyed off the creator set"
    )
    add_at = src.index("google_docs_created_this_run.add(tool_name)")
    guard = src.rindex('startswith("ERROR:")', 0, add_at)
    assert add_at - guard < 400, (
        "the record is not gated on success — a FAILED Google call would "
        "suppress the .docx and leave the user with nothing at all"
    )


def test_the_flag_is_cleared_between_runs():
    """Left set, a follow-up 'now give me a Word version' is refused as a
    duplicate of a doc created in a PREVIOUS turn — a bug with a much worse
    shape than the one being fixed, because the user asked for that file."""
    import inspect
    from app.agent.agent_runner import AgentRunner

    src = inspect.getsource(AgentRunner.run)
    assert "google_docs_created_this_run = set()" in src, (
        "agent_runner never resets google_docs_created_this_run; the refusal "
        "leaks into every later turn of the same session"
    )
