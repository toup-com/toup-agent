"""The limit table is the backstop, and the three copies must not drift.

Round 46, incident 4: there was no size gate anywhere on the attachment path.
The only ceiling was uvicorn's default 16 MiB websocket frame, and hitting it
closed the socket — which the app rendered as the USER'S INTERNET being at
fault, after re-sending the doomed frame twice more.
"""

from __future__ import annotations

import json
import os
import re

import pytest

from app.agent import attachment_limits as L

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_the_pinned_numbers_are_the_pinned_numbers():
    """OWNERSHIP A13. Changing one of these changes what a phone will accept
    and what a 768 MiB / 1.00 CPU container is asked to hold."""
    assert L.MAX_ATTACHMENTS_PER_TURN == 8
    assert L.MAX_BYTES_PER_IMAGE == 15 * 1024 * 1024
    assert L.MAX_BYTES_PER_DOCUMENT == 25 * 1024 * 1024
    assert L.MAX_TOTAL_BYTES_PER_TURN == 60 * 1024 * 1024
    assert L.LEGACY_MAX_FRAME_MEDIA_BYTES == 8 * 1024 * 1024
    assert L.MODEL_IMAGE_LONG_EDGE == 1280
    assert L.MAX_EXTRACTED_CHARS_PER_DOCUMENT == 200_000
    assert L.MAX_EXTRACTED_CHARS_PER_TURN == 400_000
    assert L.MAX_PDF_PAGES_RASTERIZED == 20
    assert L.SCAN_MIN_CHARS == 200


def test_every_rejection_has_a_distinct_stable_code():
    codes = {
        L.REASON_TOO_LARGE,
        L.REASON_TOTAL_TOO_LARGE,
        L.REASON_UNSUPPORTED,
        L.REASON_COUNT_EXCEEDED,
    }
    assert len(codes) == 4
    assert all(c.startswith("attachment_") for c in codes)


@pytest.mark.parametrize(
    "mime,name,kind",
    [
        ("image/png", "a.png", "image"),
        ("image/heic", "a.heic", "image"),
        ("application/pdf", "a.pdf", "document"),
        ("", "a.docx", "document"),
        ("application/octet-stream", "a.xlsx", "document"),
        ("text/markdown", "a.md", "document"),
        ("application/x-zip-compressed", "a.zip", "document"),
        ("application/x-msdownload", "a.exe", None),
        ("video/mp4", "a.mp4", None),
    ],
)
def test_kind_is_decided_by_mime_then_extension(mime, name, kind):
    assert L.kind_for(mime, name) == kind


def test_an_oversized_file_is_refused_by_kind():
    kind, reason = L.check_one(L.MAX_BYTES_PER_IMAGE + 1, "image/png", "big.png")
    assert kind is None and reason == L.REASON_TOO_LARGE
    # the same byte count is fine for a document
    kind, reason = L.check_one(L.MAX_BYTES_PER_IMAGE + 1, "application/pdf", "big.pdf")
    assert kind == "document" and reason is None


def test_an_unsupported_type_is_refused_with_its_own_code():
    kind, reason = L.check_one(10, "video/quicktime", "clip.mov")
    assert kind is None and reason == L.REASON_UNSUPPORTED


def _read(path):
    with open(os.path.join(REPO, path), "r", encoding="utf-8") as fh:
        return fh.read()


def _num(src, name):
    m = re.search(rf"{name}\s*=\s*([0-9_ */]+);", src)
    assert m, f"{name} not found"
    return eval(m.group(1).replace("_", ""))  # noqa: S307 — arithmetic literal only


#: The limits both clients must carry. The app's copy lives in ANOTHER REPO, so
#: this test can only reach the WEB one: the app half is pinned on the app side
#: by `scripts/check-attachments.js`, against a literal copy of these numbers.
#: An `os.path.exists("../toup-r46-app/…")` branch here was true on exactly one
#: developer machine and dead in CI and in every clone, i.e. an assertion that
#: read as coverage and ran nowhere.
_SHARED_LIMIT_NAMES = (
    "MAX_ATTACHMENTS_PER_TURN",
    "MAX_BYTES_PER_IMAGE",
    "MAX_BYTES_PER_DOCUMENT",
    "MAX_TOTAL_BYTES_PER_TURN",
    "LEGACY_MAX_FRAME_MEDIA_BYTES",
)


def test_the_web_copy_of_the_table_agrees_with_the_server():
    """A limit duplicated across three files drifts. The web keeps a copy so a
    refusal can name the file before anything is uploaded; this is the test that
    says the two IN THIS REPO are the same table."""
    web_src = _read("frontend/src/modules/chat/attachmentLimits.ts")
    for name in _SHARED_LIMIT_NAMES:
        assert _num(web_src, name) == getattr(L, name), f"web {name} drifted"


def test_the_webs_legacy_fallback_is_bounded_by_the_in_frame_cap():
    """`frontend/` has no test runner, so its one behavioural guard is here.

    The fallback (used when an upload produced no id — an agent image older than
    round 46 has no /chat/attachments route) mapped every picked file straight
    into `payload.media` with no size test, while the client table allows 25 MB
    per document. A single ordinary attachment therefore put the frame past
    uvicorn's 16 MiB ceiling, the socket died, and the app rendered a
    server-side limit as the user's connection."""
    atts = _read("frontend/src/modules/chat/attachments.ts")
    page = _read("frontend/src/modules/chat/ChatPage.tsx")
    assert "export function legacyFrameMedia" in atts
    body = atts[atts.index("export function legacyFrameMedia"):]
    body = body[: body.index("\n}\n") + 3]
    assert "LEGACY_MAX_FRAME_MEDIA_BYTES" in body, "the fallback is unbounded"
    assert "dropped" in body, "a file that cannot ride the frame must be named"
    # …and the send path uses it rather than mapping the raw list.
    assert "legacyFrameMedia(attachments)" in page
    assert "wsPayload.media = attachments.map" not in page


def test_the_webs_legacy_fallback_is_bounded_by_COUNT_as_well_as_bytes():
    """Bytes were bounded; the item count was not, and that is a silent drop.

    The image the fleet runs until the rollout walk finishes reads
    `msg["media"][:5]` — it TRUNCATES, with no log, no `attachments_ingested`
    entry and nothing under the bubble. With `MAX_ATTACHMENTS_PER_TURN` raised
    to 8, a web user attaching six to eight files against that image had the
    surplus vanish. The app enforces the same cap on the same path
    (`src/shared/attachmentLimits.ts`, LEGACY_MAX_FRAME_MEDIA_ITEMS = 5).
    """
    limits = _read("frontend/src/modules/chat/attachmentLimits.ts")
    assert _num(limits, "LEGACY_MAX_FRAME_MEDIA_ITEMS") == 5

    atts = _read("frontend/src/modules/chat/attachments.ts")
    body = atts[atts.index("export function legacyFrameMedia"):]
    body = body[: body.index("\n}\n") + 3]
    assert "LEGACY_MAX_FRAME_MEDIA_ITEMS" in body, "the fallback is unbounded by count"
    # The surplus is REFUSED BY NAME with its own reason, not folded into the
    # byte sentence: "the frame is full" and "this agent reads only the first
    # five" are different facts and the user can only act on the right one.
    assert "legacy_frame_too_many" in body
    assert "legacy_frame_too_many" in atts[atts.index("export function uploadFailureText"):]


def test_the_web_never_draws_a_card_for_a_derivative_and_never_auto_opens_evidence():
    """`role` is on the frame (ws_chat `on_attachment`) and the web dropped it.

    A browser screenshot is registered `role='source'` — evidence, not the
    answer — and the web's `case 'attachment'` force-opened the DocumentSplit
    pane on every one of them, mid-turn, on every browsing turn. The role table
    itself is the third copy of one set (server `artifact_kinds.NON_CARD_ROLES`,
    app `filesModel.NON_CARD_ROLES`); this is where the two in THIS repo are
    pinned together.
    """
    from app.agent.artifact_kinds import NON_CARD_ROLES

    atts = _read("frontend/src/modules/chat/attachments.ts")
    m = re.search(r"NON_CARD_ROLES[^=]*=\s*new Set\(\[([^\]]*)\]\)", atts)
    assert m, "the web has no role table"
    assert {s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()} == set(NON_CARD_ROLES)

    page = _read("frontend/src/modules/chat/ChatPage.tsx")
    case = page[page.index("case 'attachment': {"):]
    case = case[: case.index("\n      case ")]
    assert "isCardRole(role)" in case, "a preview/thumbnail would get its own card"
    assert "role: 'final'" not in case
    # An absent role is 'final' — an agent older than round 46 stamps none, and
    # its frames must keep behaving exactly as they do today.
    assert "'final'" in case
    # …and only a deliverable may take over the screen.
    open_call = case[case.index("openDocument(att)") - 300: case.index("openDocument(att)")]
    assert "role === 'final'" in open_call, "evidence still force-opens the pane"
    # The frame's own taxonomy reaches the card rather than being dropped.
    for field in ("kind:", "width:", "height:", "thumb_url:"):
        assert field in case, f"the frame's {field} is dropped on the floor"


def test_the_legacy_in_frame_cap_is_smaller_than_the_upload_caps():
    """The legacy path holds base64 in memory on a 768 MiB container; the
    upload path streams to storage. They are not the same budget and the
    in-frame one must stay the tighter of the two."""
    assert L.LEGACY_MAX_FRAME_MEDIA_BYTES < L.MAX_BYTES_PER_IMAGE
    assert L.LEGACY_MAX_FRAME_MEDIA_BYTES < L.MAX_BYTES_PER_DOCUMENT


# ── The number a refusal states (incident 2026-09-28) ─────────────────────
#
# The 413 detail said "That file is larger than this chat accepts." — no number,
# no way to act on it — and both clients' sentences carry the limit through
# their own `formatBytes`. The server now states the same number the clients
# do, formatted the same way, so the three can never disagree about "25 MB".


@pytest.mark.parametrize(
    "n,shown",
    [
        (25 * 1024 * 1024, "25 MB"),
        (15 * 1024 * 1024, "15 MB"),
        (60 * 1024 * 1024, "60 MB"),
        (6 * 1024 * 1024, "6 MB"),
        (50 * 1024 * 1024, "50 MB"),
        (int(9.5 * 1024 * 1024), "9.5 MB"),
        (1024 * 1024, "1 MB"),
        (1536, "2 KB"),
        (1, "1 KB"),
        (1023 * 1024, "1023 KB"),
        # JavaScript's Math.round rounds half UP; Python's round() would give
        # "10 MB" and "2.2 MB" here and drift from the clients.
        (int(10.5 * 1024 * 1024), "11 MB"),
        (int(2.25 * 1024 * 1024), "2.3 MB"),
        (0, "0 MB"),
        (-5, "0 MB"),
        (float("nan"), "0 MB"),
        (float("inf"), "0 MB"),
        (None, "0 MB"),
    ],
)
def test_format_limit_bytes_is_the_clients_formatBytes(n, shown):
    assert L.format_limit_bytes(n) == shown


def test_the_too_large_detail_states_the_limit_of_its_kind():
    assert L.too_large_detail(L.KIND_IMAGE) == (
        "That image is too large — 15 MB is the most one image can be.")
    document = "That file is too large — 25 MB is the most one file can be."
    assert L.too_large_detail(L.KIND_DOCUMENT) == document
    assert L.too_large_detail(None) == document


class _Upload:
    """An `UploadFile` stand-in that yields `total` zero bytes."""

    def __init__(self, total, filename, content_type):
        self.filename = filename
        self.content_type = content_type
        self._left = total

    async def read(self, n):
        take = max(0, min(n, self._left))
        self._left -= take
        return b"\0" * take


class _User:
    id = "user-limits-413"


async def _refusal_for(monkeypatch, upload):
    from fastapi import HTTPException

    from app.api import chat_attachments as CA

    async def _must_not_run(*a, **kw):
        raise AssertionError("ingest ran on a refused upload")

    monkeypatch.setattr(CA, "ingest_and_store", _must_not_run)
    with pytest.raises(HTTPException) as exc:
        await CA.upload_chat_attachment(
            file=upload, sha256=None, client_attachment_id=None,
            current_user=_User(), db=None,
        )
    return exc.value


@pytest.mark.asyncio
async def test_an_image_over_its_cap_is_told_the_image_number(monkeypatch):
    """`check_one` answers kind=None on the too-large path (pinned above), so
    the route must look the kind up again or an image is told 25 MB."""
    err = await _refusal_for(monkeypatch, _Upload(
        L.MAX_BYTES_PER_IMAGE + 1, "photo.png", "image/png"))
    assert err.status_code == 413
    assert err.headers.get("X-Toup-Reason") == L.REASON_TOO_LARGE
    assert err.detail == L.too_large_detail(L.KIND_IMAGE)
    assert "15 MB" in err.detail


@pytest.mark.asyncio
async def test_a_part_over_the_read_ceiling_is_told_the_document_number(monkeypatch):
    from app.api import chat_attachments as CA

    err = await _refusal_for(monkeypatch, _Upload(
        CA._ABSOLUTE_MAX + 1, "deck.pptx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation"))
    assert err.status_code == 413
    assert err.headers.get("X-Toup-Reason") == L.REASON_TOO_LARGE
    assert err.detail == L.too_large_detail(None)
    assert "25 MB" in err.detail


@pytest.mark.asyncio
async def test_the_body_ceiling_states_the_number_too():
    from app.api import chat_attachments as CA

    sent = []

    async def _send(msg):
        sent.append(msg)

    async def _inner(_scope, _receive, _send):  # pragma: no cover - must not run
        raise AssertionError("an oversized body reached the app")

    scope = {
        "type": "http", "method": "POST", "path": "/api/chat/attachments",
        "headers": [(b"content-length", str(CA.BODY_HARD_MAX + 1).encode())],
    }
    await CA.AttachmentBodyLimitMiddleware(_inner)(scope, None, _send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = next(m for m in sent if m["type"] == "http.response.body")
    assert start["status"] == 413
    assert dict(start["headers"])[b"x-toup-reason"] == L.REASON_TOO_LARGE.encode()
    assert json.loads(body["body"].decode("utf-8")) == {"detail": L.too_large_detail(None)}


def test_the_webs_too_large_copy_says_what_to_do_and_keeps_the_kind():
    """`frontend/` has no test runner. The refusal a user reads is composed on
    the client from the reason code: it must state the limit, offer a remedy
    that fits the file (saving a PDF as a PDF is not one), and — on the
    server-413 path, where only the code comes back — keep the item's kind so
    an image is not told the document number. The app pins the same table in
    its `scripts/check-attachments.js`."""
    limits = _read("frontend/src/modules/chat/attachmentLimits.ts")
    assert "export function tooLargeRemedy(name: string)" in limits
    remedy = limits[limits.index("export function tooLargeRemedy"):]
    remedy = remedy[: remedy.index("\n}\n") + 3]
    office = remedy[remedy.index("case 'pptx':"):remedy.index("case 'pdf':")]
    assert "case 'docx':" in office
    assert "saving it as a PDF" in office
    pdf = remedy[remedy.index("case 'pdf':"):remedy.index("default:")]
    assert "as a PDF" not in pdf
    assert "Compress" not in remedy and "→" not in remedy, "no menu paths"
    refusal = limits[limits.index("export function refusalText"):]
    refusal = refusal[: refusal.index("\n}\n") + 3]
    assert "tooLargeRemedy(name)" in refusal
    assert "MAX_BYTES_PER_IMAGE" in refusal and "MAX_BYTES_PER_DOCUMENT" in refusal

    atts = _read("frontend/src/modules/chat/attachments.ts")
    failure = atts[atts.index("export function uploadFailureText"):]
    assert re.search(r"uploadFailureText\(\s*name: string,\s*reason: string,\s*kind\?", failure)
    page = _read("frontend/src/modules/chat/ChatPage.tsx")
    assert "uploadFailureText(item.name, reason, item.kind)" in page
