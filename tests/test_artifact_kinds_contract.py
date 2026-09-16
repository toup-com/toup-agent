"""One artifact taxonomy, one preview policy — round 46, contract C9.

Before this round "format" was not a field anywhere. The MIME was guessed at
each call site, and what a client may DO with a file was decided independently
in ws_chat's `on_attachment` closure, `day_chats._attachment_urls`,
`voice_tasks._wire_artifact` and a fourth table inside the mobile client. All
three server copies had drifted, and all three omitted PPTX — which
`files.py::render_preview` has always been able to render and
`doc_generators._prewarm_preview` pre-converts on every generated deck, so the
conversion was paid for and never offered.

Pure: no DB, no network, no AGENT_ONLY tables. Platform sweep.
"""

from __future__ import annotations

import pathlib

import pytest

from app.agent.artifact_kinds import (
    ArtifactKind,
    ArtifactRole,
    NON_CARD_ROLES,
    canonical_filename,
    channel_size_limit,
    ext_for_mime,
    has_preview,
    kind_for_mime,
    mime_for_filename,
    mime_for_kind,
    mime_for_pil_format,
    preview_policy,
)

MIME_PDF = "application/pdf"
MIME_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
MIME_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
MIME_PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


@pytest.mark.parametrize(
    "mime,filename,expected",
    [
        (MIME_PDF, "report.pdf", ArtifactKind.PDF),
        (MIME_DOCX, "cv.docx", ArtifactKind.DOCX),
        (MIME_XLSX, "q3.xlsx", ArtifactKind.XLSX),
        (MIME_PPTX, "deck.pptx", ArtifactKind.PPTX),
        ("text/markdown", "notes.md", ArtifactKind.MARKDOWN),
        ("text/csv", "rows.csv", ArtifactKind.DATA_CSV),
        ("application/json", "payload.json", ArtifactKind.DATA_JSON),
        ("text/plain", "notes.txt", ArtifactKind.TEXT),
        ("image/png", "a.png", ArtifactKind.IMAGE),
        ("image/jpeg", "a.jpg", ArtifactKind.IMAGE),
        ("image/webp", "a.webp", ArtifactKind.IMAGE),
        ("audio/mpeg", "a.mp3", ArtifactKind.AUDIO),
        ("video/mp4", "a.mp4", ArtifactKind.VIDEO),
        ("application/zip", "a.zip", ArtifactKind.ARCHIVE),
        ("application/octet-stream", "a.bin", ArtifactKind.OTHER),
    ],
)
def test_kind_for_mime_matrix(mime, filename, expected):
    assert kind_for_mime(mime, filename) == expected


def test_a_generic_mime_loses_to_a_specific_name():
    """Every storage backend serves source and data as text/plain or
    octet-stream. A reader that can syntax-colour a .py, or a client deciding
    whether a .csv is a table, has to key off the extension."""
    assert kind_for_mime("text/plain", "script.py") == ArtifactKind.CODE
    assert kind_for_mime("text/plain", "rows.csv") == ArtifactKind.DATA_CSV
    assert kind_for_mime("text/plain", "payload.json") == ArtifactKind.DATA_JSON
    assert kind_for_mime("text/plain", "notes.md") == ArtifactKind.MARKDOWN
    assert kind_for_mime("application/octet-stream", "deck.pptx") == ArtifactKind.PPTX
    # …but a SPECIFIC mime is not overridden by a bare name.
    assert kind_for_mime(MIME_PDF, "scan") == ArtifactKind.PDF


def test_kind_for_mime_never_raises_and_never_returns_none():
    for bad in (None, "", "   ", ";;;", "application/"):
        assert kind_for_mime(bad, None) in (ArtifactKind.OTHER, ArtifactKind.TEXT)


def test_preview_policy_is_one_function_and_pptx_is_in_it():
    assert preview_policy(MIME_PDF) == "native"
    assert preview_policy("image/png") == "native"
    assert preview_policy(MIME_DOCX) == "converted"
    assert preview_policy(MIME_XLSX) == "converted"
    # The whole point of C9's first half: three drifted copies, all missing it.
    assert preview_policy(MIME_PPTX) == "converted"
    for none in ("text/markdown", "text/csv", "application/json", "audio/mpeg",
                 "video/mp4", "application/zip", "application/octet-stream", "", None):
        assert preview_policy(none) == "none", none
    assert has_preview(MIME_PPTX) and not has_preview("audio/mpeg")


def test_pptx_preview_is_actually_renderable_by_the_route_that_serves_it():
    """`preview_policy` may only promise what files.py can deliver. It said
    'converted' for PPTX above; this is the other half of that claim."""
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "app" / "api" / "files.py").read_text()
    assert "presentationml.presentation" in src
    assert "render_to_pdf" in src


def test_every_serializer_asks_the_same_function():
    """The three server-side copies are gone. Source probes, because two of
    them are closures inside request handlers and cannot be imported."""
    root = pathlib.Path(__file__).resolve().parents[1] / "app"
    for rel in ("api/ws_chat.py", "api/day_chats.py"):
        src = (root / rel).read_text()
        assert "preview_policy" in src, rel
    # …and no local preview-mime set survives in either.
    assert "_PREVIEW_MIMES" not in (root / "api" / "day_chats.py").read_text()
    ws = (root / "api" / "ws_chat.py").read_text()
    assert "preview_mimes = {" not in ws


def test_canonical_filename_keeps_the_stem_and_fixes_the_extension():
    # C7: `_safe_filename(name, "png")` (dotless) turned this into sunsetpng.png
    assert canonical_filename("sunset", "image/png") == "sunset.png"
    # A correct extension is left exactly alone.
    assert canonical_filename("portrait.jpg", "image/jpeg") == "portrait.jpg"
    # A WRONG one follows the bytes, not the caller's guess.
    assert canonical_filename("portrait.jpg", "image/png") == "portrait.png"
    assert canonical_filename("draft.docx", MIME_PDF) == "draft.pdf"
    # Traversal never survives into a storage key.
    assert "/" not in canonical_filename("../../etc/passwd", "text/plain")
    assert "\\" not in canonical_filename("..\\..\\etc\\x", "text/plain")


def test_mime_and_ext_round_trip_for_every_kind_we_produce():
    for kind in (ArtifactKind.PDF, ArtifactKind.DOCX, ArtifactKind.XLSX,
                 ArtifactKind.PPTX, ArtifactKind.MARKDOWN, ArtifactKind.DATA_CSV,
                 ArtifactKind.DATA_JSON, ArtifactKind.TEXT, ArtifactKind.IMAGE,
                 ArtifactKind.AUDIO):
        mime = mime_for_kind(kind)
        assert mime != "application/octet-stream", kind
        ext = ext_for_mime(mime)
        assert ext.startswith("."), (kind, mime, ext)
        assert kind_for_mime(mime, f"x{ext}") == kind, (kind, mime, ext)


def test_pil_format_is_the_only_honest_source_for_an_image_mime():
    assert mime_for_pil_format("JPEG") == "image/jpeg"
    assert mime_for_pil_format("PNG") == "image/png"
    assert mime_for_pil_format("WEBP") == "image/webp"
    assert mime_for_pil_format("NOT-A-FORMAT") is None
    assert mime_for_pil_format(None) is None


def test_mime_for_filename_prefers_our_table_over_the_stdlib():
    assert mime_for_filename("a.md") == "text/markdown"
    assert mime_for_filename("a.csv") == "text/csv"
    assert mime_for_filename("a.unknownext") == "application/octet-stream"


def test_roles_that_must_never_get_a_card():
    assert ArtifactRole.PREVIEW in NON_CARD_ROLES
    assert ArtifactRole.THUMBNAIL in NON_CARD_ROLES
    # A screenshot the model looked at IS in the thread — it is simply not the
    # answer. Filtering it out would hide the evidence for the reply.
    assert ArtifactRole.SOURCE not in NON_CARD_ROLES
    assert ArtifactRole.FINAL not in NON_CARD_ROLES


def test_channel_size_limits_are_smaller_for_images_than_documents():
    for ch in ("telegram", "whatsapp"):
        assert channel_size_limit(ch, "image/png", "a.png") < channel_size_limit(ch, MIME_PDF, "a.pdf")
    # An unknown channel still answers a usable number rather than 0, or every
    # file on it would be reported undeliverable.
    assert channel_size_limit("mattermost", MIME_PDF, "a.pdf") > 0
    assert channel_size_limit(None, "image/png", "a.png") > 0
