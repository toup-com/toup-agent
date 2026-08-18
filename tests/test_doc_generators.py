"""Smoke tests for the doc-generation feature (Phase 1+2).

Verifies:
- Each generator (pdf, docx, xlsx, pptx, markdown) produces a valid
  file with the correct MIME type and non-trivial size.
- ToolExecutor dispatches generate_* tool names correctly.
- pending_attachments accumulates per-call and the returned Attachment
  has stable id + storage_path.
- The file_storage LocalDiskBackend rejects path traversal.
- The feature-flag gate means the 6 tool schemas are absent by
  default (when settings.feature_doc_generation=False).
"""
import asyncio
import os
import tempfile

import pytest

from app.config import settings
from app.agent import doc_generators
from app.agent.tool_executor import ToolExecutor
from app.agent.tool_definitions import get_doc_generation_tools
from app.services.file_storage import LocalDiskBackend, get_storage_backend
import app.services.file_storage as _fs_module


def _with_tmp_workspace():
    """Context manager that points storage + config at a fresh temp dir."""
    class _Ctx:
        def __enter__(self):
            self.tmp = tempfile.TemporaryDirectory()
            self.path = self.tmp.__enter__()
            self._orig = settings.agent_workspace_dir
            settings.agent_workspace_dir = self.path
            _fs_module._backend = None  # Bust cache
            return self.path

        def __exit__(self, *a):
            settings.agent_workspace_dir = self._orig
            _fs_module._backend = None
            self.tmp.__exit__(*a)
    return _Ctx()


def test_local_disk_backend_rejects_traversal():
    with _with_tmp_workspace() as _:
        b = LocalDiskBackend(root="/tmp/test_root")
        with pytest.raises(ValueError):
            b._full("../escape")
        with pytest.raises(ValueError):
            b._full("/absolute/path")
        # Nested subdir is fine:
        assert b._full("user1/file.pdf").endswith("generated/user1/file.pdf")


def test_storage_backend_selection():
    with _with_tmp_workspace() as tmp:
        b = get_storage_backend()
        assert isinstance(b, LocalDiskBackend)
        assert b.root == tmp


def test_generate_pdf_produces_valid_file():
    async def run():
        with _with_tmp_workspace() as tmp:
            att = await doc_generators.gen_pdf(
                content=[
                    {"type": "heading", "level": 1, "text": "Test"},
                    {"type": "paragraph", "text": "Body."},
                    {"type": "table", "headers": ["A", "B"], "rows": [["1", "2"]]},
                ],
                filename="out.pdf",
                user_scope="u1",
                title="Title",
                cover_page=True,
            )
            assert att.mime_type == "application/pdf"
            assert att.size_bytes > 1000
            full = os.path.join(tmp, "generated", att.storage_path)
            assert os.path.isfile(full)
            with open(full, "rb") as f:
                assert f.read(4) == b"%PDF"
    asyncio.run(run())


def test_generate_xlsx_multi_sheet():
    async def run():
        with _with_tmp_workspace() as tmp:
            att = await doc_generators.gen_xlsx(
                sheets=[
                    {"name": "March", "headers": ["Date", "Amount"], "rows": [["2026-03-01", 42.5]]},
                    {"name": "April", "headers": ["Date", "Amount"], "rows": [["2026-04-01", 9.0]]},
                ],
                filename="expenses.xlsx",
                user_scope="u1",
            )
            assert att.mime_type.endswith("sheet")
            full = os.path.join(tmp, "generated", att.storage_path)
            assert os.path.isfile(full)
            # XLSX is a ZIP — first 2 bytes are "PK"
            with open(full, "rb") as f:
                assert f.read(2) == b"PK"
    asyncio.run(run())


def test_generate_docx_docx_and_pptx():
    async def run():
        with _with_tmp_workspace():
            att = await doc_generators.gen_docx(
                content=[{"type": "heading", "level": 1, "text": "Hi"},
                         {"type": "paragraph", "text": "Body"},
                         {"type": "bullet_list", "items": ["a", "b"]}],
                filename="d.docx",
                user_scope="u1",
            )
            assert att.mime_type.endswith("document")

            att = await doc_generators.gen_pptx(
                slides=[{"type": "title", "title": "T", "subtitle": "S"},
                        {"type": "content", "title": "X", "bullets": ["a", "b"]}],
                filename="d.pptx",
                user_scope="u1",
            )
            assert att.mime_type.endswith("presentation")
    asyncio.run(run())


def test_generate_markdown_plain_write():
    async def run():
        with _with_tmp_workspace() as tmp:
            att = await doc_generators.gen_markdown(
                content="# hi\n\nSome text.",
                filename="n.md",
                user_scope="u1",
            )
            assert att.mime_type == "text/markdown"
            full = os.path.join(tmp, "generated", att.storage_path)
            with open(full, "rb") as f:
                assert b"# hi" in f.read()
    asyncio.run(run())


def test_html_to_pdf_graceful_error_without_weasyprint():
    """If weasyprint is absent (e.g. platform image without Pango/Cairo),
    the generator raises RuntimeError with an actionable message — and
    _tool_generate_html_to_pdf converts that into an ERROR string."""
    async def run():
        with _with_tmp_workspace():
            te = ToolExecutor()
            te._user_id = "u1"
            try:
                import weasyprint  # noqa: F401
                has_wp = True
            except ImportError:
                has_wp = False

            result = await te.execute("generate_html_to_pdf",
                                      {"html": "<b>hi</b>", "filename": "h.pdf"})
            if has_wp:
                # Available — the tool should succeed.
                assert not result.startswith("ERROR:"), result
                assert len(te.pending_attachments) == 1
            else:
                assert result.startswith("ERROR:"), result
                assert "weasyprint" in result.lower()
    asyncio.run(run())


def test_tool_dispatch_and_pending_attachments():
    """ToolExecutor.execute('generate_pdf', ...) registers the attachment
    on pending_attachments with stable id + storage_path."""
    async def run():
        with _with_tmp_workspace() as tmp:
            te = ToolExecutor(workspace=tmp)
            te._user_id = "u1"
            # A descriptive filename is kept verbatim (a one-letter stem
            # like "r.pdf" is a placeholder and gets renamed — see
            # test_docgen_empty_and_placeholder.py).
            r = await te.execute("generate_pdf", {
                "filename": "release-notes.pdf",
                "content": [{"type": "paragraph", "text": "hi"}],
            })
            assert not r.startswith("ERROR:"), r
            assert len(te.pending_attachments) == 1
            att = te.pending_attachments[0]
            assert att["mime_type"] == "application/pdf"
            assert att["id"]
            assert att["storage_path"].startswith("u1/")
            assert att["storage_path"].endswith("_release-notes.pdf")
    asyncio.run(run())


def test_doc_generation_tools_schema_count_and_names():
    schemas = get_doc_generation_tools()
    names = {s["name"] for s in schemas}
    # convert_document (LibreOffice DOCX/PPTX→PDF) ships in the same gated
    # block — this assertion went stale when it landed (the file wasn't
    # CI-wired, so nothing caught it).
    #
    # `navigate_to` USED to be in this list. It is page navigation, not
    # document generation, so gating the group took "take me to my brain"
    # down with the exporters; it now lives in get_navigation_tools() and
    # ships unconditionally. Pinned in
    # tests/test_tool_family_entitlements.py.
    assert names == {
        "generate_pdf", "generate_docx", "generate_xlsx",
        "generate_pptx", "generate_markdown", "generate_html_to_pdf",
        "convert_document",
    }
    assert "navigate_to" not in names
    # Each tool has required input schema
    for s in schemas:
        assert "input_schema" in s
        assert s["input_schema"]["type"] == "object"


def test_safe_filename_strips_traversal_and_forces_extension():
    assert doc_generators._safe_filename("../etc/passwd", ".pdf") == "passwd.pdf"
    assert doc_generators._safe_filename("report", ".pdf") == "report.pdf"
    assert doc_generators._safe_filename("", ".pdf") == "document.pdf"
    # Case-insensitive extension match — already ends with .pdf (case-insensitive), keep as-is.
    assert doc_generators._safe_filename("a.PDF", ".pdf") == "a.PDF"
