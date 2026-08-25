"""R30 D-01 / D-03 / D-17 — one-off agent task cards speak the dictionary.

The founder's recordings (R30-recordings-notes, 10:08) showed a main-chat
job sheet built from raw machinery: step rows reading "List events",
"Search issues", "List repos" (connector wire ids with the underscores
swapped out), a detail line "Site: Toup · Is last: true" (a vendor JSON
payload prettified key-by-key), and terminal-dialect status lines
("Overall: ✅ OK", "⚠ That step didn't finish."). CONTRACTS-R30 §9/§10:
the one-off agent-task surface passes the same verb-dictionary discipline
the automation ledger already does — no raw tool ids, no raw arguments,
no emoji, no unformatted internal errors in any SERVED string.

Everything here is pure: the label dictionary, the summary redactor, and
the two serve paths (`day_chats._serialize_tool_events`,
`message_cards.job_card_fields`) run on plain dicts and stub rows — no DB,
no HTTP.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.tool_display import (  # noqa: E402
    _EMOJI,
    ToolResult,
    client_summary,
    public_step_label,
    sanitize_for_client,
    strip_emoji,
)

_BACKEND = Path(__file__).resolve().parent.parent

#: The tools the recording actually showed as raw ids on the job sheet.
_RECORDED_TOOLS = (
    "calendar__list_events",
    "teams__list_chats",
    "jira__search_issues",
    "github__list_repos",
    "sessions_list",
)

#: The bare humanised verbs the sheet showed. A served label may CONTAIN
#: the word ("Listing sessions" was the defect, "Checking Teams" is not) —
#: what it may never BE is the bare verb or the humanised id.
_BANNED_LABELS = {"", "List", "Create", "List events", "List chats",
                  "Search issues", "List repos", "Listing sessions"}


def _no_emoji(text: str) -> bool:
    return not _EMOJI.search(text)


# ── 1. The label dictionary is total ─────────────────────────────────────

def test_every_recorded_tool_gets_a_sentence_not_its_id():
    for tool in _RECORDED_TOOLS:
        label = public_step_label(tool)
        assert label not in _BANNED_LABELS, (tool, label)
        assert "__" not in label and "_" not in label, (tool, label)
        assert label[0].isupper() and " " in label, (tool, label)


def test_the_fallback_is_total_never_the_wire_id():
    # An unmapped single-word connector derives its BRAND, never the action.
    assert public_step_label("asana__list_tasks") == "Checking Asana"
    assert public_step_label("asana__create_task") == "Updating Asana"
    # A multi-word or internal prefix is not a brand — the honest generic.
    assert public_step_label("some_future__unmapped_tool") == "Working"
    assert public_step_label("app_html__future_tool") == "Working"
    # A tool literally named after a verb is not a label.
    for degenerate in ("list", "create", "", None):
        assert public_step_label(degenerate) == "Working"


def test_the_dictionary_agrees_with_the_live_status_line():
    """The Dynamic Island subtitle and the persisted row label are one
    vocabulary — the R18 pin, restated over the total form."""
    from app.agent.turn_progress import _subtitle_for
    for tool in _RECORDED_TOOLS + ("app_html__create_app_file",):
        line = _subtitle_for(tool)
        assert "__" not in line, (tool, line)
        assert line.rstrip("…") == public_step_label(tool), (tool, line)


# ── 2. Connector JSON never serves as a detail line (D-17) ───────────────

def test_a_connector_json_result_serves_as_nothing():
    """`{"site": "Toup", "is_last": true}` is where "Site: Toup · Is last:
    true" came from — a vendor payload prettified by the client because the
    server served it. It serves empty now; the model still reads it whole."""
    raw = json.dumps({"site": "Toup", "is_last": True,
                      "issues": [{"key": "SCRUM-1"}]})
    assert client_summary(raw, tool_name="jira__search_issues") == ""
    assert client_summary("[1, 2, 3]", tool_name="github__list_repos") == ""


def test_a_connector_display_sentence_still_serves():
    r = ToolResult('{"site": "Toup"}', display="Found 3 Jira issues")
    assert client_summary(r, tool_name="jira__search_issues") == \
        "Found 3 Jira issues"


def test_first_party_json_still_binds_the_job_card():
    """The pass-through this rule must NOT break: `create_job` answers JSON
    and the client reads job_id out of it to bind the card to its turn."""
    payload = '{"job_id": "abc", "title": "T", "steps": 3}'
    assert client_summary(payload, tool_name="create_job") == payload
    assert client_summary(payload) == payload


# ── 3. Emoji are the terminal's dialect, not the product's (D-03) ────────

def test_the_doctor_report_line_loses_its_glyphs():
    assert sanitize_for_client("Overall: ✅ OK") == "Overall: OK"
    assert sanitize_for_client("⚠ That step didn't finish.") == \
        "That step didn't finish."
    assert sanitize_for_client(
        "⚠️ The build looked for a file that wasn't there."
    ) == "The build looked for a file that wasn't there."


def test_strip_emoji_leaves_prose_alone_and_glyph_only_labels_empty():
    assert strip_emoji("Review the results") == "Review the results"
    assert strip_emoji("✅ Review the results") == "Review the results"
    assert strip_emoji("✅") == ""  # the mint falls back to "Working"
    # Untouched text is byte-identical — the reader's changed-check depends
    # on it.
    assert strip_emoji("two  spaces  stay") == "two  spaces  stay"


def test_no_card_title_seam_authors_an_emoji():
    """The `title=` lines of every job-card push producer, scanned as
    source. `subagent_orchestrator` authored "✅ Done:" / "⚠️ Didn't
    finish:" / "🛠 Working on:" straight into APNs titles; the words stay,
    the glyphs are gone. (Telegram fan-out bodies are a chat channel, not a
    card, and are out of this scan on purpose.)"""
    producers = (
        "app/agent/subagent_orchestrator.py",
        "app/agent/tool_executor.py",
        "app/agent/voice_jobs.py",
        "app/agent/job_reconciler.py",
        "app/agent/job_reaper.py",
        "app/agent/job_recovery.py",
    )
    for rel in producers:
        src = (_BACKEND / rel).read_text()
        for i, line in enumerate(src.splitlines(), 1):
            code = line.split("#", 1)[0]
            if "title=" not in code:
                continue
            assert not _EMOJI.search(code), (rel, i, line.strip())


# ── 4. Mint → steps_json → served card, end to end (D-01) ────────────────

def test_steps_minted_from_tools_serve_clean_through_the_card():
    """A voice turn's agent_task job mints its steps FROM tool names
    (`step_label_for_tool`); the card serves them through
    `message_cards.job_card_fields`. The whole served payload: no wire ids,
    no bare verbs, no emoji."""
    from datetime import datetime

    from app.agent.job_steps import dump_steps, open_first_step
    from app.agent.voice_jobs import step_label_for_tool
    from app.api.message_cards import job_card_fields

    steps = []
    for i, tool in enumerate(_RECORDED_TOOLS):
        steps.append({
            "id": f"s{i}", "type": f"step_{i}",
            "label": step_label_for_tool(tool), "status": "pending",
        })
    open_first_step(steps, datetime(2026, 8, 25, 12, 0, 0))

    row = SimpleNamespace(
        role="job",
        content=json.dumps({"job_id": "j1", "job_name": "Morning check"}),
    )
    bj = SimpleNamespace(
        id="j1", status="completed", app_id=None, title="Morning check",
        config_json={"job_type": "research"},
        steps_json=dump_steps(steps),
        outcome="success", error_class=None, user_message=None,
    )
    card = job_card_fields(row, {"j1": bj})

    served = json.dumps(card, ensure_ascii=False)
    assert "__" not in served, served
    assert _no_emoji(served), served
    assert card["job_steps"] and len(card["job_steps"]) == len(_RECORDED_TOOLS)
    for step in card["job_steps"]:
        assert step["label"] not in _BANNED_LABELS, step


def test_create_job_mint_strips_a_model_authored_glyph():
    """The chat mint (`tool_executor._tool_create_job`) routes every
    model-authored step label through `strip_emoji(...) or "Working"` —
    asserted at the seam it uses, plus a source pin that the seam is still
    wired."""
    assert (strip_emoji("✅ Review results") or "Working") == "Review results"
    assert (strip_emoji("✅") or "Working") == "Working"
    src = (_BACKEND / "app/agent/tool_executor.py").read_text()
    mint = src.split("async def _tool_create_job", 1)[1].split("async def ", 1)[0]
    assert "strip_emoji" in mint, (
        "create_job's step-label mint no longer strips emoji — a model-"
        "authored glyph will persist into steps_json and serve verbatim"
    )


# ── 5. The read path serves history clean too (D-01/D-03/D-17) ───────────

def _msg(events):
    return SimpleNamespace(metadata_json=json.dumps({"tool_events": events}))


def test_persisted_records_serve_labels_and_clean_summaries():
    """Rows written BEFORE this round: no label, a vendor-JSON summary on a
    connector record, doctor glyphs in a prose summary. The reader is where
    the rollout boundary is fixed — these rows are already in the founder's
    history."""
    from app.api.day_chats import _serialize_tool_events

    legacy = [
        {"tool": "jira__search_issues", "started_at_ms": 1,
         "completed_at_ms": 2,
         "summary": '{"site": "Toup", "is_last": true}'},
        {"tool": "calendar__list_events", "started_at_ms": 3,
         "completed_at_ms": 4, "summary": ""},
        {"tool": "doctor", "started_at_ms": 5, "completed_at_ms": 6,
         "summary": "Summary: 9 ok, 1 warnings, 0 errors\nOverall: ✅ OK"},
    ]
    out = _serialize_tool_events(_msg(legacy))
    assert out is not None and len(out) == 3

    jira, cal, doc = out
    assert jira["label"] == "Searching Jira"
    assert jira["summary"] == ""  # the vendor payload is not a detail line
    assert cal["label"] == "Checking your calendar"
    assert doc["label"] == "Running a health check"
    assert "✅" not in doc["summary"] and "Overall: OK" in doc["summary"]

    for rec in out:
        assert rec["label"] not in _BANNED_LABELS
        assert "__" not in rec["label"]
    assert _no_emoji(json.dumps(out, ensure_ascii=False))


def test_the_read_path_never_rewrites_what_is_already_clean():
    from app.api.day_chats import _serialize_tool_events

    clean = {
        "tool": "web_search", "started_at_ms": 1, "completed_at_ms": 2,
        "summary": "Web results for \"gemini\" — 3 results.",
        "label": "Searching the web",
        "domains": ["blog.google"], "urls": ["https://blog.google/x"],
    }
    out = _serialize_tool_events(_msg([clean]))
    assert out == [clean]
