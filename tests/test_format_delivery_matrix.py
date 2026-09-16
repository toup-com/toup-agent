"""Every deliverable kind, end to end, with STRUCTURAL validation — round 46.

Before this round the deliverable set was whatever happened to call
`doc_generators._persist`: PDF, DOCX, XLSX, PPTX, Markdown and images. There
was no path at all for text/plain, text/csv, application/json, code or audio,
while the turn-1 gate that unlocks the export tools had advertised `csv` since
it was written. Asking to "export this as CSV" opened the tool set and the
honest answer was not in it.

"Non-trivial size" is not validation: an empty 952-byte PDF containing only a
page footer once shipped to a user (2026-08-18). Each generator here is opened
with the library that reads that format and asked for its CONTENT back.

No network, no DB. Platform sweep.
"""

from __future__ import annotations

import asyncio
import io
import json
import tempfile

import pytest

import app.services.file_storage as _fs_module
from app.agent import doc_generators as dg
from app.agent.artifact_kinds import ArtifactKind, kind_for_mime
from app.config import settings


class _Workspace:
    """Storage + settings pointed at a fresh temp dir (the pattern
    tests/test_doc_generators.py established)."""

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = self._tmp.__enter__()
        self._orig = settings.agent_workspace_dir
        settings.agent_workspace_dir = self.path
        _fs_module._backend = None
        return self

    def __exit__(self, *a):
        settings.agent_workspace_dir = self._orig
        _fs_module._backend = None
        self._tmp.__exit__(*a)

    def read(self, att) -> bytes:
        return _fs_module.get_storage_backend().open(att.storage_path).read()


def _run(coro):
    return asyncio.run(coro)


# ── The four new generators ──────────────────────────────────────────────

def test_gen_csv_from_row_dicts_is_real_csv():
    import csv as _csv
    with _Workspace() as ws:
        att = _run(dg.gen_csv(
            [{"date": "2026-03-01", "category": "Groceries", "amount": 87.4},
             {"date": "2026-03-02", "category": "Transit", "amount": 3.25}],
            "march-expenses.csv", user_scope="u1"))
        assert att.filename == "march-expenses.csv"
        assert att.mime_type == "text/csv"
        rows = list(_csv.DictReader(io.StringIO(ws.read(att).decode("utf-8"))))
        assert [r["category"] for r in rows] == ["Groceries", "Transit"]
        assert rows[0]["amount"] == "87.4"
        assert kind_for_mime(att.mime_type, att.filename) == ArtifactKind.DATA_CSV


def test_gen_csv_keeps_columns_only_later_rows_carry():
    import csv as _csv
    with _Workspace() as ws:
        att = _run(dg.gen_csv([{"a": 1}, {"a": 2, "b": 3}], "x.csv", user_scope="u1"))
        rows = list(_csv.DictReader(io.StringIO(ws.read(att).decode("utf-8"))))
        assert rows[1]["b"] == "3", "a ragged list must not silently lose a column"


def test_gen_json_parses_a_string_before_writing_it():
    """The model hands back JSON-in-a-string constantly. Shipping that
    unparsed as application/json is how a client gets a file it cannot read."""
    with _Workspace() as ws:
        att = _run(dg.gen_json('{"total": 3, "items": ["a", "b"]}',
                               "summary.json", user_scope="u1"))
        assert att.mime_type == "application/json"
        payload = json.loads(ws.read(att).decode("utf-8"))
        assert payload == {"total": 3, "items": ["a", "b"]}


def test_gen_json_refuses_content_that_is_not_json():
    with _Workspace():
        with pytest.raises(dg.EmptyDocumentError):
            _run(dg.gen_json("not json at all", "x.json", user_scope="u1"))


def test_gen_text_and_gen_code():
    with _Workspace() as ws:
        txt = _run(dg.gen_text("line one\nline two", "notes.txt", user_scope="u1"))
        assert ws.read(txt).decode("utf-8") == "line one\nline two"
        assert kind_for_mime(txt.mime_type, txt.filename) == ArtifactKind.TEXT

        code = _run(dg.gen_code("print('hi')\n", "hello", user_scope="u1",
                                language="python"))
        # The EXTENSION is what makes the mobile reader syntax-colour it, since
        # every storage backend serves source as text/plain.
        assert code.filename == "hello.py"
        assert kind_for_mime(code.mime_type, code.filename) == ArtifactKind.CODE
        assert ws.read(code).decode("utf-8") == "print('hi')\n"


def test_gen_code_honours_an_extension_the_caller_already_chose():
    with _Workspace():
        att = _run(dg.gen_code("select 1;", "report.sql", user_scope="u1",
                               language="python"))
        assert att.filename == "report.sql"


def test_gen_audio_persists_instead_of_unlinking():
    """The `tts` tool used to upload to Telegram and then `os.unlink` the file
    in a `finally`, so audio output was impossible on the app, the web,
    WhatsApp and voice while the tool stayed in the wire array on all of them."""
    with _Workspace() as ws:
        att = _run(dg.gen_audio(b"ID3\x03\x00" + b"\x00" * 64, "greeting",
                                user_scope="u1"))
        assert att.filename == "greeting.mp3"
        assert att.mime_type == "audio/mpeg"
        assert len(ws.read(att)) == 69
        assert kind_for_mime(att.mime_type, att.filename) == ArtifactKind.AUDIO


@pytest.mark.parametrize("gen,args", [
    (dg.gen_csv, ([],)), (dg.gen_csv, ("",)),
    (dg.gen_json, ("",)), (dg.gen_json, ({},)),
    (dg.gen_text, ("   ",)), (dg.gen_code, ("",)),
])
def test_empty_content_is_refused_before_anything_is_persisted(gen, args):
    with _Workspace() as ws:
        with pytest.raises(dg.EmptyDocumentError):
            _run(gen(*args, "x", user_scope="u1"))
        import os
        gen_dir = os.path.join(ws.path, "generated")
        assert not os.path.isdir(gen_dir) or not os.listdir(gen_dir)


# ── The existing generators still produce readable documents ─────────────

def test_pdf_contains_its_heading():
    pypdf = pytest.importorskip("pypdf", reason="pypdf ships in requirements*.txt; absent from the local dev venv")
    with _Workspace() as ws:
        att = _run(dg.gen_pdf(
            [{"type": "heading", "level": 1, "text": "Quarterly Review"},
             {"type": "paragraph", "text": "Revenue grew."}],
            "q.pdf", user_scope="u1", title="Quarterly Review"))
        reader = pypdf.PdfReader(io.BytesIO(ws.read(att)))
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
        assert "Quarterly Review" in text
        assert "Revenue grew" in text


def test_docx_contains_its_paragraphs():
    docx = pytest.importorskip("docx")
    with _Workspace() as ws:
        att = _run(dg.gen_docx(
            [{"type": "heading", "level": 1, "text": "Offer letter"},
             {"type": "paragraph", "text": "We are pleased to offer you the role."}],
            "offer.docx", user_scope="u1"))
        doc = docx.Document(io.BytesIO(ws.read(att)))
        body = "\n".join(p.text for p in doc.paragraphs)
        assert "Offer letter" in body and "pleased to offer" in body


def test_xlsx_contains_its_cells():
    openpyxl = pytest.importorskip("openpyxl")
    with _Workspace() as ws:
        att = _run(dg.gen_xlsx(
            [{"name": "March", "headers": ["Date", "Amount"],
              "rows": [["2026-03-01", 87.4]]}],
            "march.xlsx", user_scope="u1"))
        wb = openpyxl.load_workbook(io.BytesIO(ws.read(att)))
        sheet = wb["March"]
        assert [c.value for c in sheet[1]] == ["Date", "Amount"]
        assert sheet.cell(row=2, column=2).value == 87.4


def test_pptx_contains_its_slide_titles_and_is_previewable():
    pptx = pytest.importorskip("pptx")
    from app.agent.artifact_kinds import preview_policy
    with _Workspace() as ws:
        att = _run(dg.gen_pptx(
            [{"title": "Plan", "bullets": ["Ship it"]}],
            "plan.pptx", user_scope="u1"))
        deck = pptx.Presentation(io.BytesIO(ws.read(att)))
        texts = [sh.text_frame.text for sl in deck.slides for sh in sl.shapes
                 if sh.has_text_frame]
        assert any("Plan" in x for x in texts)
        # C9: every one of these was pre-converted for a preview no client was
        # ever offered, because all three server copies of the set omitted it.
        assert preview_policy(att.mime_type) == "converted"


def test_markdown_is_unchanged_bytes():
    with _Workspace() as ws:
        body = "# Title\n\nSome *body* text.\n"
        att = _run(dg.gen_markdown(body, "notes.md", user_scope="u1"))
        assert ws.read(att).decode("utf-8") == body


# ── Filename / MIME correctness (C7) ─────────────────────────────────────

def test_safe_filename_rejects_a_dotless_extension():
    """Both image call sites passed `"png"`, so `sunset` became
    `sunsetpng.png` and a requested `.jpg` was forced to `.png`."""
    assert dg._safe_filename("sunset", ".png") == "sunset.png"
    assert dg._safe_filename("portrait.png", ".png") == "portrait.png"
    with pytest.raises(ValueError):
        dg._safe_filename("sunset", "png")


def test_sniff_image_follows_the_bytes_not_the_caller():
    Image = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (255, 0, 0)).save(buf, format="JPEG")
    mime, name = dg.sniff_image(buf.getvalue(), "portrait.png", "image/png")
    assert mime == "image/jpeg"
    assert name == "portrait.jpg", "the extension has to follow the real format"

    buf2 = io.BytesIO()
    Image.new("RGB", (4, 4), (0, 255, 0)).save(buf2, format="PNG")
    mime2, name2 = dg.sniff_image(buf2.getvalue(), "sunset.png", "image/png")
    assert (mime2, name2) == ("image/png", "sunset.png")


def test_sniff_image_falls_back_rather_than_raising():
    mime, name = dg.sniff_image(b"not an image", "x.png", "image/png")
    assert mime == "image/png" and name == "x.png"


def test_attachment_carries_kind_role_intent_and_old_rows_answer_none():
    att = dg.Attachment(id="a", filename="x.pdf", mime_type="application/pdf",
                        size_bytes=1, storage_path="u/x.pdf", created_at="now")
    d = att.to_dict()
    # Additive on the existing Message.attachments JSON column — no migration,
    # and a row written before round 46 simply answers None here.
    assert d["kind"] is None and d["role"] == "final" and d["intent"] is None
