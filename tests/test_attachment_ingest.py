"""attachment_ingest — every branch ends in a STATUS, and no exception text
ever reaches the model.

Round 46, incident 4. Before this module a document's fate was one of two
things: text, or a line of prose composed from whatever the parser raised. Zero
`[AGENT] Document extracted` lines exist fleet-wide across the whole Loki
retention window, so nothing about the old path was observable either.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fixtures"))
import attachments as fx  # noqa: E402

from app.agent.attachment_ingest import (  # noqa: E402
    STATUS_EMPTY,
    STATUS_OK,
    STATUS_PASSWORD_PROTECTED,
    STATUS_TRUNCATED,
    STATUS_UNSUPPORTED,
    downscale_image,
    ingest_one,
    sniff_mime,
)
from app.agent.attachment_limits import (  # noqa: E402
    MAX_EXTRACTED_CHARS_PER_DOCUMENT,
    MODEL_IMAGE_LONG_EDGE,
)

pypdf_only = pytest.mark.skipif(not fx.have("pypdf"), reason="pypdf not installed here")
pdfium_only = pytest.mark.skipif(not fx.have("pypdfium2"), reason="pypdfium2 not installed here")


# ── the type is decided by the BYTES, not by the name ────────────────────

def test_a_lying_extension_does_not_decide_the_type():
    """expo's image picker hands base64 JPEG bytes under a `.heic` name, and
    the old path wrote a `.heic` file of JPEG bytes that a vision model may
    reject. The sniff is what makes the stored mime honest."""
    png = fx.png_image()
    assert sniff_mime(png, "photo.heic", "image/heic") == "image/png"
    ing = ingest_one(png, "photo.heic", "image/heic")
    assert ing.kind == "image" and ing.mime == "image/png"


def test_an_unsupported_type_is_named_not_swallowed():
    ing = ingest_one(b"MZ\x90\x00binary", "tool.exe", "application/x-msdownload")
    assert ing.status == STATUS_UNSUPPORTED
    assert ing.reason == "attachment_unsupported"
    assert ing.text is None


def test_an_empty_file_is_its_own_status():
    ing = ingest_one(b"", "empty.txt", "text/plain")
    assert ing.status == STATUS_EMPTY


# ── documents ────────────────────────────────────────────────────────────

@pypdf_only
def test_a_text_pdf_round_trips_its_token():
    ing = ingest_one(fx.text_pdf("ZEBRAFISH"), "a.pdf", "application/pdf")
    assert ing.status == STATUS_OK
    assert "ZEBRAFISH" in (ing.text or "")
    assert ing.page_count == 1


@pypdf_only
def test_a_password_protected_pdf_is_a_status_not_a_stack_string():
    """The old path did `f"[Attached document: {fname} — extraction failed: {e}]"`,
    which put pypdf's own exception text into the model's context."""
    ing = ingest_one(fx.encrypted_pdf(), "locked.pdf", "application/pdf")
    assert ing.status == STATUS_PASSWORD_PROTECTED
    assert ing.reason == STATUS_PASSWORD_PROTECTED
    assert not (ing.text or "")


@pypdf_only
def test_no_exception_text_reaches_the_prompt_for_a_corrupt_file():
    ing = ingest_one(fx.corrupt_pdf(), "broken.pdf", "application/pdf")
    assert ing.status in ("unreadable", STATUS_EMPTY)
    body = (ing.text or "") + (ing.reason or "")
    for leak in ("Traceback", "Error", "Exception", "pypdf"):
        assert leak not in body


@pypdf_only
@pdfium_only
def test_a_scanned_pdf_becomes_page_images_for_the_vision_model():
    """There is no OCR in the agent image and none is wanted: the same turn
    already hands every photo to a model that reads text out of pictures."""
    ing = ingest_one(fx.scanned_pdf(), "scan.pdf", "application/pdf")
    assert ing.status == STATUS_OK
    assert len(ing.page_images) >= 1
    assert all(len(p) > 0 for p in ing.page_images)


@pypdf_only
def test_extracted_text_is_capped_and_says_so():
    data = fx.long_text_pdf("LOREM", MAX_EXTRACTED_CHARS_PER_DOCUMENT + 50_000)
    ing = ingest_one(data, "big.pdf", "application/pdf")
    assert len(ing.text or "") <= MAX_EXTRACTED_CHARS_PER_DOCUMENT
    if len(ing.text or "") == MAX_EXTRACTED_CHARS_PER_DOCUMENT:
        assert ing.truncated is True
        assert ing.status == STATUS_TRUNCATED


def test_docx_pptx_xlsx_and_zip_each_carry_their_token():
    for data, name, mime in (
        (fx.docx_doc("ALPHATOKEN"), "a.docx",
         "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        (fx.pptx_deck("BETATOKEN"), "b.pptx",
         "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
        (fx.xlsx_sheet("GAMMATOKEN"), "c.xlsx",
         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    ):
        ing = ingest_one(data, name, mime)
        assert ing.status == STATUS_OK, f"{name} -> {ing.status}"
        assert ing.kind == "document"
        assert ing.text

    z = ingest_one(fx.zip_of("notes.txt", b"DELTATOKEN"), "z.zip", "application/zip")
    assert z.status == STATUS_OK and "DELTATOKEN" in (z.text or "")


_PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


def test_a_deck_with_nothing_but_pictures_is_empty_not_ok(monkeypatch):
    """A picture-only deck came back as nothing but "--- Slide N ---" markers,
    which is non-empty text — so it ingested `ok` and the model was handed six
    slide numbers to "read". It is `empty` now, whose guidance says the file
    held no readable text. A deck with words on one slide still reads.

    python-pptx is stubbed so this runs on every box (the real-deck twin
    below needs the library); the stub has exactly the attributes
    `_extract_pptx` reads."""
    import io
    import types
    import zipfile

    class _Frame:
        def __init__(self, text):
            self.text = text

    class _Shape:
        def __init__(self, text=None):
            self.has_text_frame = text is not None
            self.text_frame = _Frame(text or "")
            self.has_table = False

    class _Slide:
        def __init__(self, *shapes):
            self.shapes = list(shapes)
            self.has_notes_slide = False

    decks = {
        "pictures.pptx": [_Slide(_Shape()), _Slide(_Shape(), _Shape("   "))],
        "words.pptx": [_Slide(_Shape()), _Slide(_Shape("EPSILONTOKEN"))],
    }
    current = {}
    stub = types.ModuleType("pptx")
    stub.Presentation = lambda _stream: types.SimpleNamespace(slides=decks[current["name"]])
    monkeypatch.setitem(sys.modules, "pptx", stub)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:   # enough of an OOXML container to route
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("ppt/presentation.xml", "<p:presentation/>")

    current["name"] = "pictures.pptx"
    empty = ingest_one(buf.getvalue(), "pictures.pptx", _PPTX_MIME)
    assert empty.status == STATUS_EMPTY and empty.text is None

    current["name"] = "words.pptx"
    words = ingest_one(buf.getvalue(), "words.pptx", _PPTX_MIME)
    assert words.status == STATUS_OK
    assert "EPSILONTOKEN" in words.text and "--- Slide 2 ---" in words.text


@pytest.mark.skipif(not fx.have("pptx"), reason="python-pptx not installed here")
def test_a_real_picture_only_deck_is_empty():
    import io

    from pptx import Presentation

    prs = Presentation()
    for _ in range(3):
        slide = prs.slides.add_slide(prs.slide_layouts[6])   # the blank layout
        slide.shapes.add_picture(io.BytesIO(fx.png_image()), 0, 0)
    buf = io.BytesIO()
    prs.save(buf)
    ing = ingest_one(buf.getvalue(), "pictures.pptx", _PPTX_MIME)
    assert ing.status == STATUS_EMPTY and ing.text is None


# ── images ───────────────────────────────────────────────────────────────

def test_a_large_photo_is_downscaled_to_the_model_edge():
    """A modern iPhone photo is ~4000 px; sent whole it costs several thousand
    tokens and four full copies inside a 768 MiB container."""
    from PIL import Image
    import io as _io

    big = fx.jpeg_image(2000, 1500)
    out, mime = downscale_image(big, "image/jpeg")
    with Image.open(_io.BytesIO(out)) as im:
        assert max(im.size) <= MODEL_IMAGE_LONG_EDGE
    assert len(out) < len(big)
    assert mime == "image/webp"


def test_a_small_image_is_left_exactly_as_it_is():
    small = fx.png_image(64, 48)
    out, mime = downscale_image(small, "image/png")
    assert out == small and mime == "image/png"


def test_an_image_that_will_not_decode_is_still_delivered():
    junk = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
    out, mime = downscale_image(junk, "image/png")
    assert out == junk and mime == "image/png"


def test_the_record_carries_no_contents():
    """`to_record()` is what gets persisted onto Message.attachments and logged
    around. It may carry counts; it may never carry the file's text."""
    ing = ingest_one(b"SENSITIVE BODY TEXT", "n.txt", "text/plain")
    rec = ing.to_record()
    assert "SENSITIVE" not in str(rec)
    assert rec["status"] == STATUS_OK
    assert rec["chars"] == len("SENSITIVE BODY TEXT")
