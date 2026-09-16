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
