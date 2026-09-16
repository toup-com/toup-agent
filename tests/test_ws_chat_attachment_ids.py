"""The ws_chat attachment block — what it must no longer do.

Source-read rather than socket-driven: the block sits inside a 1500-line
handler with an authenticated websocket, a ledger claim and a live agent task
around it, and the properties worth pinning are ABSENCES — no `[:5]`, no
`mkdtemp`, no bare `continue`, no b64decode on the event loop. A guard whose
subject is an absence has to read the source; this repo's own copy/behaviour
guards work the same way.

Round 46, incident 4. Five distinct silent-drop sites lived in this block:
the `[:5]` truncation, the empty-payload `continue`, the decode failure, the
persist failure (ASYMMETRIC — the model saw an image history then lost), and
the back-fill failure.
"""

from __future__ import annotations

import inspect
import os
import re

import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "app", "api", "ws_chat.py")


def block() -> str:
    src = open(SRC, "r", encoding="utf-8").read()
    start = src.index("# ── Inbound attachments ──")
    end = src.index("# ── Fast-path: detect play/music requests")
    return src[start:end]


def code() -> str:
    """The block with comment lines stripped — an absence check must not be
    satisfied or defeated by the comment that explains the absence."""
    out = []
    for line in block().splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        out.append(line.split("  # ")[0] if "  # " in line else line)
    return "\n".join(out)


def test_the_five_attachment_cap_is_gone():
    b = code()
    assert "[:5]" not in b
    assert "Max 5 attachments" not in b


def test_nothing_creates_a_temp_directory_any_more():
    """`tempfile.mkdtemp(prefix="toup_media_")` ran once per attachment and was
    never removed — for the life of the container. Loki confirmed the leak on a
    live container (pool-87, 2026-09-14 11:06:07: two dirs from one turn)."""
    b = code()
    assert "mkdtemp" not in b
    assert "toup_media_" not in b


def test_the_whole_file_no_longer_creates_toup_media_dirs():
    src = open(SRC, "r", encoding="utf-8").read()
    live = [l for l in src.splitlines() if not l.strip().startswith("#")]
    assert "toup_media_" not in "\n".join(live)


def test_the_decode_leaves_the_event_loop():
    """A 1.00-CPU container decoding base64 inline is the same starvation
    signature as the incident-2 WS accept delay."""
    b = code()
    assert "asyncio.to_thread(_b64.b64decode" in b
    assert re.search(r"(?<!to_thread\()\b_b64\.b64decode\(_mdata\)", b) is None


def test_the_ingest_also_leaves_the_event_loop():
    """Whitespace-insensitive on purpose: the property is "ingest_one is only
    ever called through to_thread", and pinning one of three exact indentations
    made a `try:` around the block a red guard with no behaviour change."""
    b = code()
    assert re.search(r"asyncio\.to_thread\(\s*ingest_one", b)
    assert re.search(r"(?<!to_thread\()\bingest_one\(", b) is None


def test_an_attachment_id_branch_exists_and_is_preferred():
    b = block()
    assert 'msg.get("attachment_ids")' in b
    # `attachment_ids` is checked FIRST; `media` is the elif.
    assert b.index("if _att_ids and isinstance(_att_ids, list):") < b.index("elif _media_items")


def _upload_branch() -> str:
    return code().split("elif _media_items")[0]


def _legacy_branch() -> str:
    return code().split("elif _media_items")[1]


def test_every_rejection_carries_the_index_the_client_used():
    """The property this file used to claim and never checked.

    The earlier version grepped for three reason STRINGS, which were all
    present while two of them shipped `{"name": "", "index": -1}` — and the app
    maps a status onto `sent[item.index]`, so `sent[-1]` is undefined and the
    refusal reached nobody. The reasons themselves are now driven behaviourally
    in test_attachment_turn_plan.py; what has to be true HERE is that no
    rejection is ever reported at a position the client cannot resolve.
    """
    b = code()
    assert '"index": -1' not in b, "a rejection reported at -1 is a silent drop"
    # Every `_rejected_atts.append(...)` in the legacy branch carries an index
    # (the upload branch's rejections come from `plan_uploaded_attachments`,
    # which is tested directly).
    for call in re.findall(r"_rejected_atts\.append\(\s*(\{[^}]*\})", _legacy_branch()):
        assert '"index"' in call, call
    assert '"type": "attachments_ingested"' in b


def test_the_report_is_numbered_from_the_wire_not_from_the_accepted_list():
    """One rejected file used to shift every later file's status onto the wrong
    row, because accepted items were numbered 0..n-1 by their position in the
    ACCEPTED list rather than in what the client sent."""
    b = code()
    assert "_record_indices" in b
    assert re.search(r'"index":\s*_record_indices\[', b)


def test_the_upload_branch_decides_through_the_tested_helper():
    """MAX_TOTAL_BYTES_PER_TURN was pinned in the table, asserted by
    test_attachment_limits, enforced by the two CLIENTS — and by nothing on the
    server. The count cap, the byte budget and the wire indices are now one
    pure function with behavioural coverage (test_attachment_turn_plan.py);
    this branch must route through it rather than re-deciding inline."""
    up = _upload_branch()
    assert "plan_uploaded_attachments" in up
    assert "asyncio.to_thread(" in up
    # Re-deciding the budget here is how the two copies drift apart.
    assert "MAX_TOTAL_BYTES_PER_TURN" not in up


def test_the_legacy_frame_budget_is_enforced_in_the_legacy_path():
    lg = _legacy_branch()
    assert "LEGACY_MAX_FRAME_MEDIA_BYTES" in lg
    assert "attachment_total_too_large" in lg


def test_a_partial_rejection_is_not_spoken_in_the_FAULT_vocabulary():
    """`fault_frame()` builds `{type:'error', code}`, and both clients treat an
    error frame whose code they do not know as a TERMINAL turn failure — they
    write an error bubble, settle the turn and tear the socket down while this
    server deliberately keeps running the turn. On the installed base (document
    picker `type:'*/*'`) that fired for every file type outside the allowlist.
    A non-fatal event gets its own type; 4508 stays for a refused turn."""
    b = code()
    assert "ATTACHMENT_REJECTED" not in b
    assert "fault_frame" not in b
    assert '"type": "attachments_rejected"' in b


def test_a_refusal_does_not_close_the_socket():
    """The durable ledger has already claimed this client_msg_id, so a turn
    refused here could never be retried — the resend would be dropped as a
    duplicate and the user would watch nothing happen."""
    b = code()
    assert "websocket.close" not in b


def test_the_records_list_is_the_one_source_of_the_ids():
    """A persist failure must not shift every later id by one: a MISLABELLED
    image is worse than an unlabelled one (C7, L6's spec)."""
    b = code()
    assert "_attachment_records.append(_rec)" in b
    # Every append is `_rec`, the record built in that same loop iteration —
    # the property, rather than an exact count a legitimate third ingest path
    # would break.
    assert re.findall(r"_attachment_records\.append\(([^)]*)\)", b) not in ([], None)
    assert set(re.findall(r"_attachment_records\.append\(([^)]*)\)", b)) == {"_rec"}


def test_the_durable_ingest_status_rides_message_attachments():
    b = block()
    assert '_att_dict["ingest"]' in b
    assert "_urow.attachments = _inbound_attachments" in b


def test_the_pre_turn_line_distinguishes_ok_failed_and_rejected():
    src = open(SRC, "r", encoding="utf-8").read()
    emit = src[src.index("att=len(_inbound_attachments)"): src.index("agent_task = asyncio.create_task")]
    for key in ("att_ok=", "att_fail=", "att_rej=", "att_up="):
        assert key in emit


def test_the_runner_is_given_the_records():
    src = open(SRC, "r", encoding="utf-8").read()
    assert "attachment_records=_attachment_records if _attachment_records else None" in src


def test_no_filename_reaches_a_log_line_in_this_block():
    """Users' document and photo filenames were written into the fleet log
    (`[AGENT] Image loaded: /tmp/toup_media_…/IMG_4893.jpg`) — the same privacy
    class as incident 3's transcript logging (A15)."""
    b = block()
    for line in b.splitlines():
        if "logger." in line:
            assert "_mname" not in line and "_fname" not in line and "_ing.name" not in line, line
