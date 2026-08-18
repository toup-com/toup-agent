"""The unrequested x.pdf (founder, iOS, 2026-08-18) — layer (b): the pipeline.

Prod log, toup-agent-871bac24, 2026-08-18 04:46Z:

    [AGENT] Tool called: generate_pdf({"filename": "x", "content": []})
    [credits] generate_pdf metered user=871bac24 … credits=1.00 … success=True
    [AGENT] Tool result: Generated x.pdf (952 bytes, application/pdf) at
            workspace path generated/shared/…_x.pdf. File will appear in
            the document pane; …

952 bytes is EXACTLY what reportlab emits for an empty story with the
"Page 1" footer (measured locally with reportlab 4.1.0). So the pipeline
did three wrong things with a stub call: it built an empty document, it
persisted + attached it under a placeholder name, and it billed a credit
for it. Pinned here:

  * every generator refuses empty content with EmptyDocumentError BEFORE
    persisting — no file on disk, no attachment, tool result starts with
    ERROR: (which is what _meter_flat_tool keys off to skip billing);
  * a placeholder filename ("x", "document", "file1", "untitled") is
    replaced from the title / first heading / first prose / dated
    fallback; a descriptive filename the model chose is kept verbatim;
  * _register_attachment refuses a zero-byte artifact from any path.

Only gen_markdown / gen_pdf are exercised end-to-end through the tool
executor (reportlab is in the platform + agent requirement sets); the
docx/xlsx/pptx checks stay on the pure-Python guard functions so this
file runs wherever test_docgen_path_normalize.py runs.
"""
import asyncio
import os
import tempfile
from datetime import datetime, timezone

import pytest

from app.config import settings
from app.agent import doc_generators as dg
from app.agent.doc_generators import (
    EmptyDocumentError, _blocks_have_content, _derive_filename,
    _is_placeholder_stem, _slugify,
)
from app.agent.tool_executor import ToolExecutor
import app.services.file_storage as _fs_module


def _with_tmp_workspace():
    class _Ctx:
        def __enter__(self):
            self.tmp = tempfile.TemporaryDirectory()
            self.path = self.tmp.__enter__()
            self._orig = settings.agent_workspace_dir
            settings.agent_workspace_dir = self.path
            _fs_module._backend = None
            return self.path

        def __exit__(self, *a):
            settings.agent_workspace_dir = self._orig
            _fs_module._backend = None
            self.tmp.__exit__(*a)
    return _Ctx()


def _executor(tmp: str) -> ToolExecutor:
    te = ToolExecutor(workspace=tmp)
    te._user_id = "u1"
    return te


def _files_under(root: str) -> list:
    out = []
    for dirpath, _, files in os.walk(root):
        out.extend(os.path.join(dirpath, f) for f in files)
    return out


def _has_reportlab() -> bool:
    try:
        import reportlab  # noqa: F401
        return True
    except ImportError:
        return False


# ── The incident, byte-for-byte, through the tool executor ─────────


@pytest.mark.skipif(not _has_reportlab(), reason="reportlab not installed")
def test_incident_stub_call_creates_nothing_and_attaches_nothing():
    """`generate_pdf({"filename": "x", "content": []})` — the exact prod
    payload. Before: 952-byte x.pdf persisted + attached + billed. After:
    an ERROR: result naming what was NOT done, no file, no attachment."""
    async def run():
        with _with_tmp_workspace() as tmp:
            te = _executor(tmp)
            r = await te.execute("generate_pdf", {"filename": "x", "content": []})
            assert r.startswith("ERROR:"), r           # billing keys off this prefix
            assert "EmptyDocumentError" in r
            assert "nothing was created or attached" in r
            assert "answer in chat" in r
            assert te.pending_attachments == []
            assert _files_under(os.path.join(tmp, "generated")) == []
    asyncio.run(run())


@pytest.mark.skipif(not _has_reportlab(), reason="reportlab not installed")
def test_pdf_with_only_page_breaks_or_blank_paragraphs_is_empty():
    """The kinds of "content" a stub can carry that still render nothing."""
    async def run():
        with _with_tmp_workspace() as tmp:
            te = _executor(tmp)
            for content in (
                [],
                [{"type": "page_break"}],
                [{"type": "paragraph", "text": ""}, {"type": "heading", "text": "   "}],
                [{"type": "table", "headers": [], "rows": []}],
                [{"type": "bullet_list", "items": []}],
                "",
                "   \n",
            ):
                r = await te.execute("generate_pdf", {
                    "filename": "empty.pdf", "content": content, "title": "Whatever",
                    "cover_page": True,
                })
                assert r.startswith("ERROR:"), (content, r)
            assert te.pending_attachments == []
            assert _files_under(os.path.join(tmp, "generated")) == []
    asyncio.run(run())


@pytest.mark.skipif(not _has_reportlab(), reason="reportlab not installed")
def test_real_pdf_still_generates_and_attaches():
    """The guard must not eat a genuine document."""
    async def run():
        with _with_tmp_workspace() as tmp:
            te = _executor(tmp)
            r = await te.execute("generate_pdf", {
                "filename": "august-ai-news.pdf",
                "title": "August AI News",
                "content": [
                    {"type": "heading", "level": 1, "text": "Top 5"},
                    {"type": "paragraph", "text": "1. Something happened."},
                    {"type": "table", "headers": ["Story", "Source"], "rows": [["A", "Reuters"]]},
                ],
            })
            assert not r.startswith("ERROR:"), r
            assert len(te.pending_attachments) == 1
            att = te.pending_attachments[0]
            assert att["filename"] == "august-ai-news.pdf"
            assert att["size_bytes"] > 952
            assert os.path.isfile(os.path.join(tmp, "generated", att["storage_path"]))
    asyncio.run(run())


@pytest.mark.skipif(not _has_reportlab(), reason="reportlab not installed")
def test_placeholder_pdf_filename_is_derived_from_title_then_heading():
    async def run():
        with _with_tmp_workspace() as tmp:
            te = _executor(tmp)
            # title wins
            r = await te.execute("generate_pdf", {
                "filename": "x", "title": "Q3 Expense Summary — Draft",
                "content": [{"type": "paragraph", "text": "hi"}],
            })
            assert not r.startswith("ERROR:"), r
            assert te.pending_attachments[-1]["filename"] == "q3-expense-summary-draft.pdf"
            # no title → first heading
            r = await te.execute("generate_pdf", {
                "filename": "document.pdf",
                "content": [
                    {"type": "heading", "level": 1, "text": "App Rebuild: Timeline"},
                    {"type": "paragraph", "text": "hi"},
                ],
            })
            assert not r.startswith("ERROR:"), r
            assert te.pending_attachments[-1]["filename"] == "app-rebuild-timeline.pdf"
            # no title, no heading → first prose words
            r = await te.execute("generate_pdf", {
                "filename": "file1.pdf",
                "content": [{"type": "paragraph", "text": "Meeting notes from the Monday standup, covering roadmap items."}],
            })
            assert not r.startswith("ERROR:"), r
            assert te.pending_attachments[-1]["filename"] == "meeting-notes-from-the-monday-standup.pdf"
    asyncio.run(run())


# ── gen_markdown end-to-end (no heavy deps) ────────────────────────


def test_markdown_empty_content_refused_and_nothing_persisted():
    async def run():
        with _with_tmp_workspace() as tmp:
            te = _executor(tmp)
            for content in ("", "   ", "\n\n", None):
                r = await te.execute("generate_markdown", {"content": content, "filename": "x"})
                assert r.startswith("ERROR:"), (content, r)
                assert "EmptyDocumentError" in r
            assert te.pending_attachments == []
            assert _files_under(os.path.join(tmp, "generated")) == []
    asyncio.run(run())


def test_markdown_placeholder_filename_derived_from_heading_then_line_then_date():
    async def run():
        with _with_tmp_workspace() as tmp:
            te = _executor(tmp)
            r = await te.execute("generate_markdown", {
                "content": "# Weekly Plan\n\n- a\n- b", "filename": "x.md",
            })
            assert not r.startswith("ERROR:"), r
            assert te.pending_attachments[-1]["filename"] == "weekly-plan.md"

            r = await te.execute("generate_markdown", {
                "content": "Groceries for the week ahead and more\nmilk\neggs", "filename": "untitled",
            })
            assert not r.startswith("ERROR:"), r
            assert te.pending_attachments[-1]["filename"] == "groceries-for-the-week-ahead-and.md"

            # Nothing slug-able (CJK only) → dated fallback, still .md
            r = await te.execute("generate_markdown", {"content": "你好世界", "filename": "x"})
            assert not r.startswith("ERROR:"), r
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            assert te.pending_attachments[-1]["filename"] == f"document-{today}.md"
    asyncio.run(run())


def test_descriptive_filename_kept_verbatim():
    async def run():
        with _with_tmp_workspace() as tmp:
            te = _executor(tmp)
            for name in ("march-expenses.md", "cv.md", "q3.md", "AI_News_Aug.md", "notes"):
                r = await te.execute("generate_markdown", {"content": "# x\nbody", "filename": name})
                assert not r.startswith("ERROR:"), r
                expect = name if name.endswith(".md") else name + ".md"
                assert te.pending_attachments[-1]["filename"] == expect, name
    asyncio.run(run())


def test_error_result_is_not_billed():
    """`_meter_flat_tool` skips a result that starts with ERROR: — the
    incident charged 1 credit for a stub. Pin the contract at the seam:
    an EmptyDocumentError result never reaches report_flat_charge."""
    async def run():
        with _with_tmp_workspace() as tmp:
            te = _executor(tmp)
            calls = []
            import app.services.credit_reporter as cr
            orig = getattr(cr, "report_flat_charge", None)
            async def _spy(*a, **k):
                calls.append((a, k))
            cr.report_flat_charge = _spy
            orig_flag = settings.flat_tool_metering_enabled
            settings.flat_tool_metering_enabled = True
            te.set_user_id("u1")
            try:
                r = await te.execute("generate_pdf", {"filename": "x", "content": []})
                assert r.startswith("ERROR:"), r
                assert calls == []
            finally:
                settings.flat_tool_metering_enabled = orig_flag
                if orig is not None:
                    cr.report_flat_charge = orig
    asyncio.run(run())


# ── _register_attachment: zero-byte guard for every other path ─────


def test_register_attachment_refuses_zero_byte_artifact():
    async def run():
        with _with_tmp_workspace() as tmp:
            te = _executor(tmp)
            att = dg.Attachment(
                id="abc", filename="conv.pdf", mime_type=dg.MIME_PDF,
                size_bytes=0, storage_path="u1/abc_conv.pdf", created_at="now",
            )
            r = await te._register_attachment(att)
            assert r.startswith("ERROR:"), r
            assert te.pending_attachments == []
            ok = await te._register_attachment(dg.Attachment(
                id="def", filename="conv.pdf", mime_type=dg.MIME_PDF,
                size_bytes=10, storage_path="u1/def_conv.pdf", created_at="now",
            ))
            assert not ok.startswith("ERROR:")
            assert len(te.pending_attachments) == 1
    asyncio.run(run())


# ── Pure guards for docx / xlsx / pptx (no heavy deps) ────────────


def test_blocks_have_content_semantics():
    assert not _blocks_have_content([])
    assert not _blocks_have_content(None)
    assert not _blocks_have_content("")
    assert not _blocks_have_content([{"type": "page_break"}])
    assert not _blocks_have_content([{"type": "paragraph", "text": " "}])
    assert not _blocks_have_content([{"type": "table", "headers": ["", None], "rows": [[], None]}])
    assert not _blocks_have_content([{"type": "image", "path": "/definitely/not/here.png"}])
    assert _blocks_have_content("plain string paragraph")
    assert _blocks_have_content([{"type": "heading", "text": "H"}])
    assert _blocks_have_content([{"type": "table", "headers": ["A"], "rows": []}])
    assert _blocks_have_content([{"type": "table", "headers": [], "rows": [["1"]]}])
    assert _blocks_have_content([{"type": "numbered_list", "items": ["one"]}])
    assert _blocks_have_content([{"type": "page_break"}, {"type": "paragraph", "text": "x"}])


def test_xlsx_and_pptx_empty_guards():
    """Guard semantics for the generators that need openpyxl/python-pptx —
    asserted on the raise-before-import path, which needs neither."""
    async def run():
        with _with_tmp_workspace():
            for sheets in ([], None, [{"name": "S", "headers": [], "rows": []}], [{"headers": ["", ""], "rows": [[]]}]):
                with pytest.raises(EmptyDocumentError):
                    await dg.gen_xlsx(sheets=sheets, filename="x", user_scope="u1")
            for slides in ([], None, [{"type": "content", "title": "", "bullets": []}], [{"type": "image", "path": "/nope.png"}]):
                with pytest.raises(EmptyDocumentError):
                    await dg.gen_pptx(slides=slides, filename="x", user_scope="u1")
            for content in ([], None, "", [{"type": "page_break"}]):
                with pytest.raises(EmptyDocumentError):
                    await dg.gen_docx(content=content, filename="x", user_scope="u1", title="T")
            for html in ("", None, "<html><body>   </body></html>", "<p></p>"):
                with pytest.raises(EmptyDocumentError):
                    await dg.gen_html_to_pdf(html=html, filename="x", user_scope="u1")
    asyncio.run(run())


def test_empty_document_error_message_is_actionable():
    e = EmptyDocumentError("generate_pdf", "content blocks")
    s = str(e)
    assert "generate_pdf" in s
    assert "nothing was created or attached" in s
    assert "answer in chat" in s
    assert isinstance(e, ValueError)


# ── Filename derivation unit pins ─────────────────────────────────


def test_placeholder_stems():
    for s in ("x", "X", "a", "1", "007", "test", "test2", "file", "file (3)", "document",
              "untitled-1", "output", "tmp", "doc", "pdf", "sheet1", "Untitled"):
        assert _is_placeholder_stem(s), s
    for s in ("cv", "q3", "ai", "march-expenses", "invoice-1042", "report-q3", "notes", "plan"):
        assert not _is_placeholder_stem(s), s


def test_slugify():
    assert _slugify("March 2026 — Expenses & Totals") == "march-2026-expenses-totals"
    assert _slugify("  Résumé: José  ") == "resume-jose"
    assert _slugify("你好") == ""
    long = "word " * 40
    s = _slugify(long)
    assert len(s) <= 60 and not s.endswith("-")


def test_derive_filename_contract():
    assert _derive_filename("march-expenses", ".xlsx") == "march-expenses.xlsx"
    assert _derive_filename("x", ".pdf", title="AI News") == "ai-news.pdf"
    assert _derive_filename("x", ".pdf", title="", hints=("", None, "First Heading")) == "first-heading.pdf"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert _derive_filename("x", ".pdf") == f"document-{today}.pdf"
    assert _derive_filename(None, ".docx") == f"document-{today}.docx"
    # traversal is still stripped before the placeholder check
    assert _derive_filename("../../etc/passwd", ".pdf") == "passwd.pdf"
    assert _derive_filename("../../etc/x", ".pdf", title="Brief") == "brief.pdf"
    # a placeholder TITLE does not rescue a placeholder filename
    assert _derive_filename("x", ".pdf", title="document", hints=("Real Title",)) == "real-title.pdf"
