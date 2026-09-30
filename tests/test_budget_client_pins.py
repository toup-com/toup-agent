"""Source pins for the web's budget-refusal surfaces (incident 2026-09-28).

The frontend has no unit-test runner, so the gates that keep a budget-refused
user from being told "Something went wrong", shown the raw exception, or
offered a retry that the proxy will refuse until the reset are pinned here,
next to the backend vocabulary they mirror (app/services/budget_refusal.py):

* ChatPage renders an error frame with ``code: monthly_model_budget_exceeded``
  as the server's sentence, without the "⚠️ Error:" dressing;
* ChatPage does not start a job card from a ``job_update`` frame that names
  neither the job nor its type (it could only be drawn as "Couldn't build
  App Build");
* BuildJobCard and JobsPage hide "Try again" for ``error_class ==
  "model_budget"`` and never fall back to the raw exception text;
* the dashboard's kanban card shows the row's ``user_message`` and never its
  raw ``error_message``;
* the web's fallback job sentence is byte-identical to
  ``budget_refusal.job_sentence(None)``, so the copy cannot drift apart;
* the 25 MB refusal names a remedy chosen by the file's extension.

The pins compare tokens, not layout: whitespace between tokens is ignored, so
a reformat or a re-wrap cannot break them, while string literals still have
to match exactly.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.services import budget_refusal

SRC = Path(__file__).resolve().parents[2] / "frontend" / "src"

# One token: a whole string literal (kept exact), a word, or one other character.
_TOKENS = re.compile(
    r"""'(?:[^'\\\n]|\\.)*'|"(?:[^"\\\n]|\\.)*"|`(?:[^`\\]|\\.)*`|\w+|[^\w\s]"""
)


def _read(rel: str) -> str:
    return (SRC / rel).read_text(encoding="utf-8")


def _loose(snippet: str) -> re.Pattern:
    """``snippet`` as a pattern that allows any whitespace between its tokens."""
    return re.compile(r"\s*".join(re.escape(tok) for tok in _TOKENS.findall(snippet)))


def _find(src: str, snippet: str, start: int = 0) -> int:
    match = _loose(snippet).search(src, start)
    assert match, f"not found: {snippet!r}"
    return match.start()


def _has(src: str, snippet: str) -> bool:
    return _loose(snippet).search(src) is not None


def _count(src: str, snippet: str) -> int:
    return len(_loose(snippet).findall(src))


def _between(src: str, start: str, end: str) -> str:
    i = _find(src, start)
    return src[i:_find(src, end, i + 1)]


def _code(src: str) -> str:
    """``src`` without its comments, so a note about a field is not a use of it."""
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    return re.sub(r"(^|[^:])//[^\n]*", r"\1", src)


def test_chat_error_frame_with_budget_code_is_shown_without_error_dressing():
    src = _read("modules/chat/ChatPage.tsx")
    assert _has(src, f"data.code === '{budget_refusal.FRAME_CODE}'")
    branch = _between(src, "const budgetRefusal = data.code", "setMessages(")
    # The budget branch picks the undressed message; only the other branch says "⚠️ Error:".
    assert _has(branch, "budgetRefusal ? (msg.trim() ||")
    assert _has(branch, "`⚠️ Error: ${data.message}`")
    assert _has(branch, "budgetRefusal ? '' : '⚠️ '")


def test_a_job_update_naming_neither_the_job_nor_its_type_starts_no_card():
    src = _read("modules/chat/ChatPage.tsx")
    handler = _between(src, "case 'job_update': {", "case 'app_ready': {")
    belt = _find(handler, "if (existingIdx < 0 && !data.name && !data.job_type) return prev;")
    # It reads the existing card first and returns before a card is built.
    assert _find(handler, "const existingIdx =") < belt < _find(handler, "const marker: Message = {")


def test_build_card_hides_try_again_for_a_budget_stop():
    src = _read("modules/chat/BuildJobCard.tsx")
    assert _has(src, f"const budgetStopped = job.errorClass === '{budget_refusal.ERROR_CLASS}';")
    assert _has(src, "const canRetry = !budgetStopped;")
    # Both retry affordances (collapsed chip and expanded card) are gated.
    assert _count(src, "&& canRetry && (") >= 2
    # The class name itself is never rendered.
    assert not re.search(r">\s*model_budget\s*<", src) and not _has(src, "{job.errorClass}")


def test_a_finished_chat_task_is_not_labelled_as_a_built_app():
    # Chat-intent jobs ride the same card (job_type "agent_task"); a PDF
    # summary that finished must not read "Built Summarise my handout".
    src = _read("modules/chat/BuildJobCard.tsx")
    assert _has(src, "return !job.jobType || job.jobType === 'auto_builder';")
    collapsed = _between(src, "const label = ok", "const chip =")
    assert _find(collapsed, "? appBuild") < _find(collapsed, "`Built ${job.name}`") < _find(collapsed, "`Finished ${job.name}`")
    assert _has(src, "const TASK_LABELS: Record<string, string> = { running: 'Working', completed: 'Done' };")
    assert _has(src, "appBuild={appBuild} />")


def test_jobs_page_hides_try_again_and_never_shows_the_raw_exception_for_a_budget_stop():
    src = _read("modules/workspace/JobsPage.tsx")
    assert _has(src, f"const budgetStopped = errorClass === '{budget_refusal.ERROR_CLASS}';")
    assert _has(src, "{!budgetStopped && (")
    failure = _between(src, "const failureText =", ";")
    # The class-specific sentence is preferred over error_message, which for a
    # budget stop is the raw exception.
    assert _find(failure, "userMessage") < _find(failure, "job.error_message")
    assert _find(failure, "budgetStopped ?") < _find(failure, "job.error_message")


def test_dashboard_kanban_card_shows_the_row_sentence_never_the_raw_exception():
    src = _read("pages/DashboardPage.tsx")
    card = _between(src, "function KanbanCard(", "const KANBAN_COLS")
    assert _has(card, "const failureText = ((job as BuildJobInfo & JobTaxonomy).user_message || '').trim();")
    assert _has(card, "{failureText && noteTone && (")
    # Red only for a stop; a job waiting on the user's approval reads neutral,
    # and a queued, running or finished job shows no sentence.
    assert _has(card, "const noteTone = stoppedShort ? 'text-danger' : waitingOnUser ? 'text-ink-secondary' : '';")
    assert _has(card, "const waitingOnUser = job.status === 'waiting_on_user' || job.status === 'paused';")
    tone = _between(card, "const stoppedShort =", ";")
    for status in ("'queued'", "'running'", "'completed'"):
        assert _has(tone, f"job.status !== {status}")
    assert not _has(card, 'className="text-danger truncate')
    # No code on the page reads the raw exception text at all.
    assert "error_message" not in _code(src)


def test_web_fallback_job_sentence_matches_the_backend_sentence():
    expected = budget_refusal.job_sentence(None)
    literal = re.compile(r"(['\"`])" + re.escape(expected) + r"\1")
    for rel in ("modules/chat/BuildJobCard.tsx", "modules/workspace/JobsPage.tsx"):
        assert literal.search(_read(rel)), rel


def test_too_large_refusal_names_an_extension_specific_remedy():
    src = _read("modules/chat/attachmentLimits.ts")
    start = _find(src, "export function tooLargeRemedy")
    end = re.compile(r"^\}", re.M).search(src, start)
    assert end, "tooLargeRemedy has no closing brace at column 0"
    remedy = src[start:end.start()]
    assert re.search(r"case 'pptx':\s*case 'docx':\s*return 'Try saving it as a PDF", remedy)
    assert _has(remedy, "case 'pdf': return 'Try compressing it or splitting it into smaller PDFs.';")
    assert _has(remedy, "return 'Try splitting it into smaller files.';")
    # No menu paths: they differ between Office for Mac and Windows.
    assert "→" not in remedy and "File >" not in remedy
