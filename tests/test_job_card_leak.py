"""Round 16 — a job card's MARKER must never be a chat message.

The bug these tests exist for, verbatim from the founder's thread: a voice
turn's card row arrived in chat as an agent bubble reading

    {"job_id": "d89c6987-17ff-420b-bf1b-419d29ce8217", "job_name":
     "\\u06a9\\u0627\\u0631\\u0628\\u0631 ..."}

Three of the four history readers returned a ``role='job'`` row's
``content`` verbatim, and that content is a machine marker, not prose.
``/api/day-chats/{date}/messages`` — the reader every client asks FIRST —
was one of the three.

Every test below drives the REAL route or the REAL serializer. None of
them re-implements the projection: the leak survived years of tests that
simulated the endpoint's query and wrote their own response dict, because
a test that re-implements the serializer cannot see the serializer's bug.

Mutation proofs (each restores the exact pre-fix behaviour):
  * ``day_chats.get_day_chat_messages``: put ``"content": msg.content``
    back → ``test_day_chats_never_returns_the_marker_as_text`` fails on
    the assert that the body is empty.
  * ``message_cards.public_text``: ``return content or ""`` unguarded →
    ``test_a_marker_on_a_renderable_role_is_blanked`` fails.
  * ``message_cards.job_card_fields``: drop ``job_steps`` →
    ``test_the_card_carries_the_run_s_steps`` fails.
  * ``message_cards.attach_run_to_cards``: ``return rows`` immediately →
    ``test_one_job_one_card_the_run_rides_the_card`` fails.
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timedelta, date as Date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api.message_cards import (  # noqa: E402
    attach_run_to_cards,
    is_internal_marker,
    job_card_fields,
    job_marker_content,
    public_text,
)

# asyncio_mode = auto (pytest.ini) — the async tests below need no mark.

# The exact body the founder saw, minus the length. `ensure_ascii=True` is
# what produced the escapes, so this string is built the way the old writer
# built it — not with the new helper, which would hide the escaping.
LEAKED_JOB_ID = "d89c6987-17ff-420b-bf1b-419d29ce8217"
LEAKED_TITLE = "کاربر درباره‌ی بهترین مدل تصویرسازی پرسید"
LEAKED_MARKER = json.dumps({"job_id": LEAKED_JOB_ID, "job_name": LEAKED_TITLE})


# ── The guard, on its own ────────────────────────────────────────────

def test_a_job_row_body_is_always_blank():
    assert public_text("job", LEAKED_MARKER) == ""
    assert public_text("job", "anything at all") == ""


def test_a_marker_on_a_renderable_role_is_blanked():
    """The second lock: even if a writer puts a marker on an assistant row,
    no reader may render it."""
    assert public_text("assistant", LEAKED_MARKER) == ""
    assert public_text("assistant", json.dumps({"job_id": "x"})) == ""


def test_a_bare_identifier_is_not_an_answer():
    assert public_text("assistant", LEAKED_JOB_ID) == ""
    assert public_text("assistant", f"  job-{LEAKED_JOB_ID}  ") == ""


def test_the_user_s_own_words_survive_the_guard():
    """The guard must not eat real messages. A user row is the user's own
    text — echoing a UUID they typed is harmless; deleting it is not."""
    assert public_text("user", LEAKED_JOB_ID) == LEAKED_JOB_ID
    # An object that carries a marker key AND anything else is somebody's
    # real content, not a marker.
    real = json.dumps({"job_id": "x", "note": "here is my config"})
    assert public_text("assistant", real) == real
    assert public_text("assistant", f"the id is {LEAKED_JOB_ID}") != ""
    assert not is_internal_marker("[]")
    assert not is_internal_marker("{}")


def test_the_marker_writer_does_not_ascii_escape():
    """`json.dumps` defaults to `ensure_ascii=True`; that default is why a
    Persian title reached the founder as a run of backslash-u escapes."""
    marker = job_marker_content(LEAKED_JOB_ID, LEAKED_TITLE, "agent_task")
    assert "\\u" not in marker
    assert LEAKED_TITLE in marker
    assert json.loads(marker)["job_type"] == "agent_task"


# ── The projection ───────────────────────────────────────────────────

class _FakeJob:
    id = LEAKED_JOB_ID
    status = "completed"
    app_id = None
    title = "Build: something"
    config_json = json.dumps({"job_type": "agent_task"})
    steps_json = json.dumps([
        {"id": "1", "type": "step_0", "label": "جست‌وجوی وب", "status": "done",
         "duration_ms": 3500},
        {"id": "2", "type": "step_1", "label": "خواندن منابع", "status": "done"},
        {"id": "3", "type": "step_answer", "label": "پاسخ", "status": "running"},
    ])


class _Row:
    """The minimum of a Message row the projection reads."""
    def __init__(self, role="job", content=LEAKED_MARKER):
        self.role = role
        self.content = content


def test_the_card_carries_the_run_s_steps():
    out = job_card_fields(_Row(), {LEAKED_JOB_ID: _FakeJob()})
    assert out["job_id"] == LEAKED_JOB_ID
    # The title is the user's own sentence, in their own script — not an
    # id, not "App Build", not escaped.
    assert out["job_name"] == LEAKED_TITLE
    assert out["job_type"] == "agent_task"
    assert out["job_total_steps"] == 3
    assert out["job_completed_steps"] == 2
    assert [s["label"] for s in out["job_steps"]] == [
        "جست‌وجوی وب", "خواندن منابع", "پاسخ",
    ]
    assert out["job_step"] == "پاسخ"


def test_a_card_with_no_row_behind_it_is_not_left_spinning():
    out = job_card_fields(_Row(), {})
    assert out["job_status"] == "completed"
    assert out["job_name"] == LEAKED_TITLE


def test_the_projection_ignores_every_other_role():
    assert job_card_fields(_Row(role="assistant", content="hi"), {}) == {}


# ── One job, one card ────────────────────────────────────────────────

def _web_record(job_id):
    return {
        "tool": "web_search", "started_at_ms": 1, "completed_at_ms": 4,
        "call_id": "c1", "job_id": job_id, "summary": "…",
        "domains": ["arxiv.org"], "urls": ["https://arxiv.org/abs/1"],
        "sources": [{"title": "A paper", "url": "https://arxiv.org/abs/1",
                     "domain": "arxiv.org"}],
    }


def test_one_job_one_card_the_run_rides_the_card():
    rows = [
        {"id": "job-1", "role": "job", "job_id": LEAKED_JOB_ID, "tool_events": None},
        {"id": "m2", "role": "assistant", "tool_events": [_web_record(LEAKED_JOB_ID)]},
    ]
    attach_run_to_cards(rows)
    card = rows[0]
    assert card["tool_events"] and len(card["tool_events"]) == 1
    # Sources parity: the favicon (`domains`), the link and the TITLE all
    # ride the card, so the persisted card can draw the source list the
    # voice overlay drew — not a plain text line.
    src = card["tool_events"][0]["sources"][0]
    assert src["title"] == "A paper" and src["domain"] == "arxiv.org"
    assert card["tool_events"][0]["domains"] == ["arxiv.org"]
    # Idempotent — a second pass must not double the list.
    attach_run_to_cards(rows)
    assert len(card["tool_events"]) == 1


def test_a_record_with_no_job_stays_with_its_answer():
    rows = [
        {"id": "job-1", "role": "job", "job_id": LEAKED_JOB_ID},
        {"id": "m2", "role": "assistant", "tool_events": [_web_record(None)]},
    ]
    attach_run_to_cards(rows)
    assert not rows[0].get("tool_events")
    assert len(rows[1]["tool_events"]) == 1


# ── The invariant, as source ─────────────────────────────────────────

def test_no_message_serializer_emits_a_raw_body():
    """A reader added next quarter must not be able to reintroduce this.

    The leak was one line — ``"content": msg.content`` — repeated in three
    files. Nothing else in the codebase legitimately serializes a message
    body straight out of the row, so its ABSENCE is a checkable invariant,
    and it is the only check that covers a reader nobody has written yet.
    """
    import re
    root = Path(__file__).resolve().parent.parent / "app" / "api"
    readers = ["day_chats.py", "sessions.py", "messages_recover.py"]
    raw = re.compile(r"""["']content["']\s*:\s*\w+\.content\b""")
    offenders = []
    for name in readers:
        src = (root / name).read_text()
        assert "public_text" in src, f"{name} does not use the guard"
        for i, line in enumerate(src.splitlines(), 1):
            if raw.search(line):
                offenders.append(f"{name}:{i}: {line.strip()}")
    assert not offenders, (
        "a message body is being serialized without public_text():\n"
        + "\n".join(offenders)
    )
