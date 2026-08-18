"""Document-output placement pins (canary 533354ce, 2026-07-28).

The behavioral-suite doc_generation scenario asked for a PDF and the file
landed at the workspace ROOT instead of generated/, where the harness and
the document pane look. Two-pronged root cause, both pinned here:

1. EXPOSURE — "Make me a one-page PDF summarizing the water cycle"
   classified as code intent (make…page) and the generate_* tools were
   filtered out, so the model satisfied the ask with write_file/exec.
   Pins: TOOLS_DOCGEN exposed whenever the message names a document
   (has_document_intent — originally "merged into every work intent",
   narrowed 2026-08-18 after the unrequested x.pdf incident), format-word
   classification for short asks.
2. PLACEMENT — write_file resolved relative paths against the workspace
   root with no document routing. Pins: doc-extension write_file targets
   are normalized into {workspace}/generated/ (workspace-relative
   subpath preserved so distinct model paths never collide, generated/
   auto-created, traversal rejected), the tool result names the
   normalized generated/ path, exec/pty_exec runs sweep freshly-written
   root documents into generated/, and the generate_* path
   (gen_markdown — no heavy deps) keeps its existing basename/traversal
   normalization with a real relative path in the _register_attachment
   summary.

Deliberately dependency-free: only gen_markdown (plain bytes write) is
exercised on the generate_* side so this file runs under CI's targeted
pip list (no reportlab/python-docx/openpyxl/python-pptx).
"""
import asyncio
import os
import tempfile

from app.config import settings
from app.agent import doc_generators
from app.agent.query_intent import (
    TOOLS_DOCGEN,
    INTENT_AGENT, INTENT_CODE, INTENT_MEDIA, INTENT_MEMORY,
    INTENT_SCHEDULING, INTENT_WEB,
    classify_query_intent, filter_tools_by_intent,
)
from app.agent.tool_executor import ToolExecutor
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


def _tool_stubs():
    """A representative tool-definition list for filter_tools_by_intent."""
    return [{"name": n} for n in (
        "write_file", "exec", "pty_exec", "read_file", "web_search",
        "generate_pdf", "generate_docx", "generate_xlsx", "generate_pptx",
        "generate_markdown", "generate_html_to_pdf", "convert_document",
        "generate_image", "edit_image", "memory_search", "recall_day",
    )]


def _executor(tmp: str) -> ToolExecutor:
    te = ToolExecutor(workspace=tmp)
    te._user_id = "u1"
    return te


# ── Prong 1: exposure ─────────────────────────────────────────────


def test_eval_prompt_exposes_generate_pdf():
    """The EXACT behavioral-suite sentence must expose the docgen tools on
    turn 1 — on canary 533354ce it classified as code (make…page) and
    generate_pdf was filtered out."""
    intent = classify_query_intent(
        "Make me a one-page PDF summarizing the water cycle. "
        "Keep it brief - a few short sections is fine."
    )
    exposed = {t["name"] for t in filter_tools_by_intent(_tool_stubs(), intent)}
    assert "generate_pdf" in exposed
    assert "convert_document" in exposed


def test_short_format_asks_expose_docgen_tools():
    """Short format-word asks must not fall through to the tool-less
    question intent ('Make me a PDF please' scored zero everywhere)."""
    for msg, tool in [
        ("Make me a PDF please", "generate_pdf"),
        ("Create a spreadsheet of my expenses this month", "generate_xlsx"),
        ("Can you make me a slide deck about Mars?", "generate_pptx"),
    ]:
        intent = classify_query_intent(msg)
        exposed = {t["name"] for t in filter_tools_by_intent(_tool_stubs(), intent)}
        assert tool in exposed, f"{msg!r} → {intent.category} hides {tool}"


def test_compound_doc_image_asks_keep_image_tools():
    """Regression pin: the document nouns added to _CODE_PATTERNS_RE
    (presentation|slides|deck|document…) pull compound doc+image asks
    from media intent into code intent — on main all three sentences
    classified media and exposed generate_image. Whatever intent wins,
    the image-generation tools must stay exposed on turn 1 (turn-1 tool
    choice is final for single-call turns — same failure shape as the
    incident this PR fixes)."""
    for msg in (
        "make me a presentation with images",
        "make a slide deck with photos from my trip",
        "Make me a document with a picture of a sunset in it",
    ):
        intent = classify_query_intent(msg)
        exposed = {t["name"] for t in filter_tools_by_intent(_tool_stubs(), intent)}
        assert "generate_image" in exposed, f"{msg!r} → {intent.category} hides generate_image"
        assert "edit_image" in exposed, f"{msg!r} → {intent.category} hides edit_image"


def test_docgen_tools_ride_on_document_intent_not_category():
    """A doc ask classifies by its SUBJECT vocabulary, not the export
    format — so the docgen set used to be merged into every work intent.
    Since 2026-08-18 (the unrequested x.pdf incident, see
    test_query_intent_docgen_gate.py) it rides on has_document_intent
    instead: the BASE intents no longer carry it, and any category the
    subject vocabulary picks gets it merged when a format/artifact word
    is present. Both halves pinned here."""
    for intent in (INTENT_CODE, INTENT_WEB, INTENT_MEDIA, INTENT_MEMORY,
                   INTENT_SCHEDULING, INTENT_AGENT):
        assert not (TOOLS_DOCGEN & intent.tool_names), intent.category
    # Subject vocabulary → memory / web / media / scheduling; the export
    # word still exposes the tools on top of whatever category wins.
    for msg in (
        "make a PDF of what we discussed yesterday",
        "search the web for the top 5 AI stories and export them as a spreadsheet",
        "make me a slide deck with photos from my trip",
        "every morning send me a PDF briefing",
    ):
        intent = classify_query_intent(msg)
        assert TOOLS_DOCGEN <= intent.tool_names, f"{msg!r} → {intent.category}"


# ── Prong 2a: write_file doc-extension normalization ──────────────


def test_write_file_bare_doc_filename_lands_in_generated():
    """Bare filename + generated/ missing: the file is normalized into
    {workspace}/generated/ (auto-created) and the result names that path."""
    async def run():
        with _with_tmp_workspace() as tmp:
            te = _executor(tmp)
            ws = te._get_user_workspace()
            assert not os.path.isdir(os.path.join(ws, "generated"))
            r = await te.execute("write_file", {
                "path": "water-cycle-summary.pdf", "content": "x",
            })
            assert not r.startswith("ERROR:"), r
            assert os.path.isfile(os.path.join(ws, "generated", "water-cycle-summary.pdf"))
            assert not os.path.exists(os.path.join(ws, "water-cycle-summary.pdf"))
            assert "generated/water-cycle-summary.pdf" in r
    asyncio.run(run())


def test_write_file_absolute_doc_path_redirected():
    async def run():
        with _with_tmp_workspace() as tmp:
            te = _executor(tmp)
            ws = te._get_user_workspace()
            os.makedirs(ws, exist_ok=True)
            r = await te.execute("write_file", {
                "path": os.path.join(ws, "summary.pdf"), "content": "x",
            })
            assert not r.startswith("ERROR:"), r
            assert os.path.isfile(os.path.join(ws, "generated", "summary.pdf"))
            assert not os.path.exists(os.path.join(ws, "summary.pdf"))
    asyncio.run(run())


def test_write_file_nested_doc_path_preserves_subpath():
    """Model-supplied directories are preserved UNDER generated/ — the
    subpath moves wholesale (reports/2026/summary.docx →
    generated/reports/2026/summary.docx), no phantom dirs appear at the
    workspace root, and the tool result cites the real relative path."""
    async def run():
        with _with_tmp_workspace() as tmp:
            te = _executor(tmp)
            ws = te._get_user_workspace()
            r = await te.execute("write_file", {
                "path": "reports/2026/summary.docx", "content": "x",
            })
            assert not r.startswith("ERROR:"), r
            assert os.path.isfile(os.path.join(ws, "generated", "reports", "2026", "summary.docx"))
            assert not os.path.exists(os.path.join(ws, "reports"))
            assert "generated/reports/2026/summary.docx" in r
    asyncio.run(run())


def test_write_file_distinct_doc_dirs_never_collide():
    """Two writes to the same basename in different model-supplied
    directories must remain two distinct files — basename flattening
    would silently clobber drafts/report.pdf with final/report.pdf and
    leave the tool results claiming two files exist when only one does."""
    async def run():
        with _with_tmp_workspace() as tmp:
            te = _executor(tmp)
            ws = te._get_user_workspace()
            r1 = await te.execute("write_file", {
                "path": "drafts/report.pdf", "content": "draft",
            })
            r2 = await te.execute("write_file", {
                "path": "final/report.pdf", "content": "final",
            })
            assert not r1.startswith("ERROR:"), r1
            assert not r2.startswith("ERROR:"), r2
            p1 = os.path.join(ws, "generated", "drafts", "report.pdf")
            p2 = os.path.join(ws, "generated", "final", "report.pdf")
            assert os.path.isfile(p1) and os.path.isfile(p2)
            with open(p1) as f:
                assert f.read() == "draft"
            with open(p2) as f:
                assert f.read() == "final"
    asyncio.run(run())


def test_write_file_doc_traversal_rejected():
    """Traversal that escapes the allowed roots is refused by the path
    jail (PermissionError → ERROR: string), never silently relocated."""
    async def run():
        with _with_tmp_workspace() as tmp:
            te = _executor(tmp)
            r = await te.execute("write_file", {
                "path": "../" * 10 + "etc/x.pdf", "content": "x",
            })
            assert r.startswith("ERROR:"), r
            assert not os.path.exists("/etc/x.pdf")
    asyncio.run(run())


def test_write_file_inside_generated_untouched():
    """Already-correct placement (incl. nested under generated/) stays put."""
    async def run():
        with _with_tmp_workspace() as tmp:
            te = _executor(tmp)
            ws = te._get_user_workspace()
            r = await te.execute("write_file", {
                "path": "generated/sub/deep.pdf", "content": "x",
            })
            assert not r.startswith("ERROR:"), r
            assert os.path.isfile(os.path.join(ws, "generated", "sub", "deep.pdf"))
    asyncio.run(run())


def test_write_file_non_doc_and_session_workspace_exempt():
    """Source/text writes keep their exact path, and a session-workspace
    override (vibecoding app builds) is never redirected."""
    async def run():
        with _with_tmp_workspace() as tmp:
            te = _executor(tmp)
            ws = te._get_user_workspace()
            r = await te.execute("write_file", {"path": "notes/plan.md", "content": "x"})
            assert not r.startswith("ERROR:"), r
            assert os.path.isfile(os.path.join(ws, "notes", "plan.md"))

            app_dir = os.path.join(tmp, "vibecoding", "myapp")
            os.makedirs(app_dir, exist_ok=True)
            te.set_session_workspace(app_dir)
            try:
                r = await te.execute("write_file", {"path": "brochure.pdf", "content": "x"})
                assert not r.startswith("ERROR:"), r
                assert os.path.isfile(os.path.join(app_dir, "brochure.pdf"))
            finally:
                te.set_session_workspace(None)
    asyncio.run(run())


# ── Prong 2c: exec/pty_exec root-document sweep ───────────────────


def test_exec_root_doc_write_swept_into_generated():
    """The incident toolset offered exec alongside write_file, and a shell
    write into the workspace root reproduces the incident byte-for-byte
    with no write_file involved. After exec, freshly-written root
    documents must be relocated into generated/ and the tool result must
    say so (the model cites the real location)."""
    async def run():
        with _with_tmp_workspace() as tmp:
            te = _executor(tmp)
            ws = te._ensure_workspace()
            r = await te.execute("exec", {
                "command": "printf x > water-cycle-summary.pdf",
            })
            assert not r.startswith("ERROR:"), r
            assert os.path.isfile(os.path.join(ws, "generated", "water-cycle-summary.pdf"))
            assert not os.path.exists(os.path.join(ws, "water-cycle-summary.pdf"))
            assert "generated/water-cycle-summary.pdf" in r
    asyncio.run(run())


def test_pty_exec_root_doc_write_swept_into_generated():
    async def run():
        with _with_tmp_workspace() as tmp:
            te = _executor(tmp)
            ws = te._ensure_workspace()
            r = await te.execute("pty_exec", {
                "command": "printf x > summary.docx",
            })
            assert not r.startswith("ERROR:"), r
            assert os.path.isfile(os.path.join(ws, "generated", "summary.docx"))
            assert not os.path.exists(os.path.join(ws, "summary.docx"))
            assert "generated/summary.docx" in r
    asyncio.run(run())


def test_exec_sweep_leaves_old_files_and_uniquifies_collisions():
    """Pre-existing root documents (older than the tool call) stay put;
    a sweep collision with an existing generated/ file uniquifies
    (report-1.pdf) instead of clobbering."""
    async def run():
        with _with_tmp_workspace() as tmp:
            te = _executor(tmp)
            ws = te._ensure_workspace()
            # Pre-existing root doc, mtime well before the exec call.
            old = os.path.join(ws, "old.pdf")
            with open(old, "w") as f:
                f.write("old")
            past = os.path.getmtime(old) - 3600
            os.utime(old, (past, past))
            # Existing generated/report.pdf the sweep must not clobber.
            gen = os.path.join(ws, "generated")
            os.makedirs(gen, exist_ok=True)
            with open(os.path.join(gen, "report.pdf"), "w") as f:
                f.write("keep")
            r = await te.execute("exec", {"command": "printf new > report.pdf"})
            assert not r.startswith("ERROR:"), r
            assert os.path.isfile(old)  # untouched
            with open(os.path.join(gen, "report.pdf")) as f:
                assert f.read() == "keep"
            with open(os.path.join(gen, "report-1.pdf")) as f:
                assert f.read() == "new"
            assert "generated/report-1.pdf" in r
    asyncio.run(run())


def test_exec_sweep_exempts_session_workspace():
    """Vibecoding builds own their layout — exec inside a session
    workspace never gets swept."""
    async def run():
        with _with_tmp_workspace() as tmp:
            te = _executor(tmp)
            app_dir = os.path.join(tmp, "vibecoding", "myapp")
            os.makedirs(app_dir, exist_ok=True)
            te.set_session_workspace(app_dir)
            try:
                r = await te.execute("exec", {
                    "command": "printf x > brochure.pdf", "workdir": app_dir,
                })
                assert not r.startswith("ERROR:"), r
                assert os.path.isfile(os.path.join(app_dir, "brochure.pdf"))
            finally:
                te.set_session_workspace(None)
    asyncio.run(run())


# ── Prong 2b: generate_* placement + honest tool result ───────────


def test_generate_markdown_normalizes_every_model_path_shape():
    """Bare, absolute, nested, and traversal filenames all land as a flat
    basename under {root}/generated/{scope}/ — with generated/ absent at
    the start of each call. (Descriptive stems throughout — a one-letter
    stem is a placeholder and would be renamed from the content; see
    test_docgen_empty_and_placeholder.py.)"""
    async def run():
        for supplied, expect in [
            ("notes.md", "notes.md"),
            ("/etc/plan.md", "plan.md"),
            ("a/b/agenda.md", "agenda.md"),
            ("../../etc/plan", "plan.md"),
            ("..\\..\\etc\\plan", "plan.md"),
        ]:
            with _with_tmp_workspace() as tmp:
                assert not os.path.isdir(os.path.join(tmp, "generated"))
                att = await doc_generators.gen_markdown(
                    content="# hi", filename=supplied, user_scope="u1",
                )
                assert att.filename == expect, supplied
                assert att.storage_path == f"u1/{att.id}_{expect}"
                full = os.path.join(tmp, "generated", att.storage_path)
                assert os.path.isfile(full), supplied
                # The normalized location stays inside the jail.
                assert os.path.realpath(full).startswith(os.path.realpath(tmp) + os.sep)
    asyncio.run(run())


def test_generate_tool_result_names_real_relative_path():
    """The tool result the model reads must cite the real workspace path
    (generated/{storage_path}) so its confirmation matches reality."""
    async def run():
        with _with_tmp_workspace() as tmp:
            te = _executor(tmp)
            r = await te.execute("generate_markdown", {
                "content": "# hi", "filename": "reports/summary.md",
            })
            assert not r.startswith("ERROR:"), r
            assert len(te.pending_attachments) == 1
            att = te.pending_attachments[0]
            assert f"generated/{att['storage_path']}" in r
            assert os.path.isfile(os.path.join(tmp, "generated", att["storage_path"]))
    asyncio.run(run())


def test_safe_filename_strips_backslash_traversal():
    assert doc_generators._safe_filename("..\\..\\etc\\x", ".pdf") == "x.pdf"
    assert doc_generators._safe_filename("a\\b\\report.pdf", ".pdf") == "report.pdf"
