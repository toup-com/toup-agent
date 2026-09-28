"""A voice turn's work is a JOB, and the runner is the one that opens it.

Round 13. Until now a voice run landed in the day chat as an action summary —
"Searched the web · 4× · 3.5s" — while the same request typed into chat landed
as a card titled "Find the strongest image-generation model · Done in 3 steps".
Same agent, same tools, same thread, two different products.

The gap is not in the persistence (Round 12 already carries ``tool_events``
through ``_save_voice_messages``); it is that **nothing opens a job on a voice
turn**. Chat's card exists because the model calls ``create_job`` — and voice
does not have that tool, on purpose:

    ``prompt_profile.VOICE_DISABLED_TOOLS`` removes ``create_job`` /
    ``update_job`` / ``spawn`` from the voice loadout. On 2026-08-01 the
    founder asked, aloud, for U of T professors working on LLMs; in 145
    seconds the agent produced three background jobs and zero spoken answers,
    promising a report that never came. Prompts are advisory, tool-list
    omission is hard, so voice lost the deferral tools instead.

That decision stands and this module does not touch it. What it removes is the
false inference that came with it — "voice has no ``create_job``" was read as
"voice has no jobs". A job card is not a deferral: it is the RECORD of work
this turn actually did. The model still cannot promise the user a report for
later; the runner writes the card for work already in hand.

So the producer moves from the model to the server:

* one step per DISTINCT tool kind the turn used, in call order, collapsed
  ("web_search ×4" is one step, not four), plus a closing step for the answer;
* the title comes from the request the user made, through the same normaliser
  the ``create_job`` tool now runs its model-supplied title through
  (``job_titles.py``), so the two cards are one design;
* the row is a normal ``agent_task`` ``BuildJob``, opened through
  ``JobRunner.create_job`` and closed through
  ``job_reconciler.close_job_completed`` — the SAME two functions chat uses,
  which is what makes the ``job_update`` frames, the Live Activity card, the
  step timing and the terminal push identical rather than merely similar.

**Nothing here is allowed on the critical path.** The user is holding a live
audio session open and TTS starts when the turn returns; ``create_job`` alone
was measured at 0.6–1 s of DB + notify work. Every method below is
synchronous, returns immediately, and never raises — the DB and push work runs
in ``background_tasks.spawn``-ed tasks serialised behind one lock, so the
turn's own coroutine never awaits any of it.

Safety nets, in order: this module's own ``seal`` (the normal close), the
context sweep from ``AgentRunner.run``'s ``finally`` (a cancelled turn — how a
voice call normally ends when the caller hangs up), and, if the process dies
between them, ``job_reconciler.reconcile_delivered_turn_jobs`` (the row is an
ordinary ``agent_task`` with a ``conversation_id``, so the watchdog closes it
once the answer message proves delivery). Every one of the three writes through
the same guarded ``status == 'running'`` UPDATE, so exactly one wins and the
phone card is pushed exactly once.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from app.services.background_tasks import spawn as _spawn_bg

logger = logging.getLogger(__name__)

__all__ = [
    "SEAL_CANCELLED",
    "SEAL_COMPLETED",
    "SEAL_INTERRUPTED_AFTER_ANSWER",
    "VoiceTurnJob",
    "clean_request_text",
    "current_voice_job",
    "is_work_tool",
    "set_current_voice_job",
    "step_label_for_tool",
    "sweep_current_voice_job",
]

# ── How a card ENDED ──────────────────────────────────────────────────────
#: The turn produced its answer. Today's normal close.
SEAL_COMPLETED = "completed"
#: The turn produced its answer and was then cut off — the caller hanging up
#: after hearing the reply, which is how a voice call normally ends. Still a
#: COMPLETED card: "Didn't finish" over work the agent already spoke aloud is
#: the exact lie `job_status.turn_interrupted` was written to stop telling.
SEAL_INTERRUPTED_AFTER_ANSWER = "interrupted_after_answer"
#: The turn was stopped before anything was delivered — a barge-in, a
#: supersession, a hang-up mid-research. The work genuinely happened and its
#: steps and sources stay readable, but nothing was ever received, so the card
#: says `cancelled`. It used to say "Done in 3 steps", over an answer the user
#: never heard, and tapping it showed real steps and real sources, which made
#: the claim more convincing rather than less.
SEAL_CANCELLED = "cancelled"

# ── Scaffolding must never become a title ─────────────────────────────────
# A Live delegation's `user_message` is a CONSTRUCTED PROMPT: the utterance
# wrapped in English framing plus verbatim prior-turn content
# (`live_voice_protocol.delegated_agent_input`). The title path fans whatever
# it is handed out to four surfaces — the card, the chat marker row, the APNs
# body and the Live Activity content-state — and the last two escape nothing.
# Production shipped `Working on: Live-session context from earlier accepted…`
# to a lock screen.
#
# `display_request` (R48) is the real fix: the caller passes the clean
# utterance. This list is the belt-and-braces half, and it is independent of
# that field on purpose — a caller that forgets to send it must not be able to
# put prompt scaffolding on a notification surface.
_SCAFFOLD_MARKERS = (
    "live-session context",
    "prior caller request",
    "accepted backend result",
    "current caller request",
    "treat it as prior conversation",
    "context from earlier in this conversation",
)

#: Below this a derived title is a caption fragment, not a request. A 350 ms
#: transcript settle produces exactly that — "ایونتش", "چک کن" — and a card
#: titled with a clipped word names nothing. Two words is the floor; the
#: fallback ("Voice request") is a truthful label and a better one.
_TITLE_MIN_WORDS = 2

#: …but a WORD floor is a claim about the script, not about the request.
#: Chinese, Japanese and Thai do not separate words with spaces, so a whole
#: sentence is one `split()` token: `'播放周杰伦的歌'` and `'ニュースを教えて'`
#: are complete requests and were both titled "Voice request", unconditionally,
#: for every speaker of those languages. For those scripts the honest floor is
#: a LENGTH: a settle fragment is a character or two, a request is not.
_SPACELESS_SCRIPT_RE = re.compile(
    "["
    "\\u3040-\\u309f"   # hiragana
    "\\u30a0-\\u30ff"   # katakana
    "\\u3400-\\u4dbf"   # CJK extension A
    "\\u4e00-\\u9fff"   # CJK unified ideographs
    "\\uf900-\\ufaff"   # CJK compatibility ideographs
    "\\u0e00-\\u0e7f"   # thai
    "]"
)
#: Characters, for the scripts above. Deliberately small: two ideographs is a
#: word ("天气"), four is a request.
_TITLE_MIN_CHARS = 4


def clean_request_text(request: Optional[str]) -> Optional[str]:
    """The part of ``request`` that may become a user-visible title.

    Returns None when the text is prompt scaffolding rather than a request, so
    the caller falls back to its generic label. Matched on the marker's own
    line: a delegation prompt puts the framing FIRST, and an utterance that
    merely quotes one of these phrases mid-sentence is still the user's.
    """
    if not request or not isinstance(request, str):
        return None
    head = request.strip()[:200].lower()
    if any(head.startswith(m) or f"\n{m}" in head for m in _SCAFFOLD_MARKERS):
        return None
    return request


def _title_is_fragment(title: str) -> bool:
    """True when ``title`` is a settle fragment rather than a request.

    Two rules, one per script family, because the question "is this a whole
    request" is asked of text: where words are separated by spaces, count
    words; where they are not, count characters. Applying the word rule to a
    space-free script answers "fragment" for every sentence in it.
    """
    stripped = (title or "").strip()
    if not stripped:
        return True
    if _SPACELESS_SCRIPT_RE.search(stripped):
        return len(stripped) < _TITLE_MIN_CHARS
    return len(stripped.split()) < _TITLE_MIN_WORDS


# ── What counts as work ───────────────────────────────────────────────────
# A job card for "what time is it" is noise, and noise on this surface is the
# failure this module inherits from — the 2026-08-01 session minted a card per
# re-ask. So a turn earns a card by DOING something, and these are the tools
# that do not count as doing something:
#
#   * chrome the user already sees another way — navigation, playback (the
#     media card IS the surface), TTS preferences;
#   * reads that serve the conversation rather than a task — memory lookup,
#     day recall, session introspection;
#   * the two hand-offs that open their own tracked work — ``start_mission``
#     (voice keeps it: it is the one deferral users ask for in words) and
#     ``spawn``. A second card for the same ask is the duplicate this module
#     exists to avoid.
#
# Everything else — web, browser, files, exec, generation, every connector —
# is work. Deny-list rather than allow-list on purpose: a newly registered
# connector tool should count as work on the day it ships, not on the day
# someone remembers to add it here.
NON_WORK_TOOLS: frozenset = frozenset({
    "navigate_to", "play_media", "play_netflix", "radio",
    "tts", "tts_prefs", "talk_mode",
    "memory_search", "recall_day",
    "session_status", "sessions_list", "sessions_history", "thread",
    "start_mission", "spawn",
    # Voice cannot call these (VOICE_DISABLED_TOOLS) — listed so that a
    # future loadout change cannot make the runner open a second card
    # alongside the model's own.
    "create_job", "update_job",
})


def is_work_tool(name: Optional[str]) -> bool:
    """True when a call to ``name`` is work a card should record."""
    return bool(name) and str(name) not in NON_WORK_TOOLS


# ── Step labels ───────────────────────────────────────────────────────────
# A chat job's steps are written by the model, in the user's language. A voice
# job's steps are written here, so they are localised the only way a server
# can: English and Persian, picked from the script of the request. Persian
# because that is the founder's primary language and voice is where he uses
# it; anything else falls back to English, exactly as the app's own verb
# vocabulary does (`VoiceModeOverlay.VERBS_EN` / `VERBS_FA`).
#
# Phrased as work, not as narration — these sit beside model-written steps in
# the same card, and the model writes "Compare the benchmarks", not
# "Comparing the benchmarks".
_LABELS_EN: Dict[str, str] = {
    "web": "Search the web",
    "fetch": "Read the sources",
    "browser": "Work the page",
    "file_read": "Read the files",
    "file_write": "Write the files",
    "exec": "Run the command",
    "image": "Make the image",
    "doc": "Build the document",
    "memory": "Save what matters",
    "send": "Send it over",
    "look": "Look at the image",
    "schedule": "Set the schedule",
    "connector_read": "Check {brand}",
    "connector_write": "Update {brand}",
    "other": "Work on it",
    "answer": "Put the answer together",
}

_LABELS_FA: Dict[str, str] = {
    "web": "جست‌وجو در وب",
    "fetch": "خواندن منابع",
    "browser": "کار با صفحه",
    "file_read": "خواندن فایل‌ها",
    "file_write": "نوشتن فایل‌ها",
    "exec": "اجرای دستور",
    "image": "ساخت تصویر",
    "doc": "ساخت سند",
    "memory": "ذخیره در حافظه",
    "send": "ارسال",
    "look": "بررسی تصویر",
    "schedule": "تنظیم زمان‌بندی",
    "connector_read": "بررسی {brand}",
    "connector_write": "به‌روزرسانی {brand}",
    "other": "انجام کار",
    "answer": "جمع‌بندی پاسخ",
}

#: tool name -> label key. Anything unmatched falls through to the connector
#: rule (``<brand>__<action>``) and then to "other".
_TOOL_KEYS: Dict[str, str] = {
    "web_search": "web", "extension_search": "web", "extension_research": "web",
    "web_fetch": "fetch", "extension_read": "fetch", "smart_fetch": "fetch",
    "browser": "browser",
    "read_file": "file_read", "ls": "file_read", "grep": "file_read",
    "find": "file_read",
    "write_file": "file_write", "edit_file": "file_write",
    "apply_patch": "file_write",
    "exec": "exec", "pty_exec": "exec", "process": "exec",
    "generate_image": "image", "edit_image": "image", "canvas": "image",
    "analyze_image": "look",
    "generate_pdf": "doc", "generate_docx": "doc", "generate_xlsx": "doc",
    "generate_pptx": "doc", "generate_markdown": "doc",
    "generate_html_to_pdf": "doc", "convert_document": "doc",
    "memory_store": "memory", "memory_delete": "memory",
    "send_file": "send", "send_photo": "send", "message": "send",
    "routines__create": "schedule", "routines__remind": "schedule",
    "routines__update": "schedule", "routines__delete": "schedule",
    "routines__run_now": "schedule", "routines__list": "schedule",
    "triggers__create": "schedule", "triggers__update": "schedule",
    "triggers__delete": "schedule", "cron": "schedule",
}

#: Connector actions that only READ. Everything else on a connector is treated
#: as a write, which is the safe direction for a label: "Update Gmail" beside
#: a message that was only listed is a smaller lie than "Check Gmail" beside
#: one that was sent.
_CONNECTOR_READ_VERBS = (
    "list", "get", "search", "read", "find", "fetch", "query", "lookup",
    "describe", "download", "export", "check",
)

#: Brand casing for the connector prefix. Unlisted prefixes are title-cased.
_BRANDS: Dict[str, str] = {
    "gmail": "Gmail", "gcal": "Calendar", "calendar": "Calendar",
    "gdrive": "Drive", "drive": "Drive", "docs": "Docs", "sheets": "Sheets",
    "slides": "Slides", "outlook": "Outlook", "github": "GitHub",
    "linkedin": "LinkedIn", "notion": "Notion", "linear": "Linear",
    "slack": "Slack", "stripe": "Stripe", "figma": "Figma",
    "whatsapp": "WhatsApp", "telegram": "Telegram",
}


def _brand(prefix: str) -> str:
    return _BRANDS.get(prefix.lower(), prefix.replace("_", " ").title())


def step_label_for_tool(
    tool_name: Optional[str], _tool_input: Any = None, *, rtl: bool = False,
) -> str:
    """The card label for one tool call, in the request's language.

    Deliberately keyed on the tool NAME only: an argument (a search query, a
    file path, a recipient) is user or model text and this string is a
    notification-surface string — it rides Live Activity content-state and
    push bodies, where nothing is escaped. Keeping the closed vocabulary
    closed is the same rule the app applies to its own verb ledger.
    """
    labels = _LABELS_FA if rtl else _LABELS_EN
    name = str(tool_name or "")
    key = _TOOL_KEYS.get(name)
    if key:
        return labels[key]
    if "__" in name:
        prefix, _, action = name.partition("__")
        read = any(action.lower().startswith(v) for v in _CONNECTOR_READ_VERBS)
        tpl = labels["connector_read" if read else "connector_write"]
        return tpl.format(brand=_brand(prefix))
    return labels["other"]


#: The stopped card's own words, in the same two languages the step labels
#: use and picked by the same `rtl` flag. A card that says `cancelled` has to
#: say it in the language the request was made in, or the honest word is one
#: the user cannot read.
_STOPPED_STEP_EN = "Stopped"
_STOPPED_STEP_FA = "متوقف شد"
_STOPPED_BODY_EN = "This stopped before it had an answer. Ask me again and I'll pick it up."
_STOPPED_BODY_FA = "پیش از رسیدن به پاسخ متوقف شد. دوباره بپرس تا ادامه بدهم."

#: The delegation id's ONE bound, for every side of this file's comparison.
#: The id is provider-supplied and unbounded (`live_voice_protocol` reads it
#: off the provider's delegation object), and it reaches the card and the
#: persisted row down different paths — relay body `[:64]`, `ChatRequest`
#: max_length=64, `sessions._clean_voice` (which bounds this key to the same
#: number for exactly this reason). All three must agree or the delivery proof
#: fails closed and a delivered answer is announced as cancelled.
_DELEGATION_ID_MAX = 64

#: The cancelled close asks the thread whether an answer landed, and POLLS
#: rather than guessing one interval. The relay's assistant write and this
#: coroutine race: the row that counts is the DELEGATED one, which the relay
#: submits after the delegation completes, through a persistence queue and an
#: HTTP POST to the tenant — an interval nothing in either process measures. A
#: single sleep therefore had to be either too short (a delivered answer
#: announced as cancelled: silent, and safe-looking) or a flat cost on every
#: real cancellation. This runs off the turn's critical path — the caller is
#: already gone — so polling costs the call nothing, answers a row that is
#: ALREADY there immediately, and bounds the wrong answer by a ceiling instead
#: of by a guess.
_CANCEL_PROOF_POLL_S = 0.5
_CANCEL_PROOF_MAX_WAIT_S = 6.0

#: How many of the newest assistant rows the delivery proof reads. The relay
#: writes one spoken row per output epoch, so a talkative turn can leave
#: several between the job opening and this question; the answer row this card
#: is looking for is among the newest of them or it does not exist.
_PROOF_ROW_SCAN = 20


def _stop_open_steps(
    steps: List[Dict[str, Any]], now: datetime,
) -> List[Dict[str, Any]]:
    """Close a cancelled card's open steps WITHOUT greening them.

    A terminal row whose step list still says a step is ``running`` is a
    stopped job that renders a live spinner forever — the Round-27 "archived
    frozen as running" class, on the durable surface. ``finish_all_steps`` is
    not the answer either: it marks every remaining step done, which is the
    overclaim this whole close exists to correct.

    So the step that was running gets its REAL window (it really ran, and for
    that long) and becomes ``skipped``; steps that never started become
    ``skipped`` with no window, because a duration nobody measured is unknown.
    Both words are already in the clients' step vocabulary.
    """
    for s in steps:
        status = s.get("status")
        if status == "running":
            started = s.get("started_at")
            s["status"] = "skipped"
            if not s.get("completed_at"):
                s["completed_at"] = _iso_ms(now)
                start_dt = _parse_iso_ms(started)
                if start_dt is not None:
                    dur = int(max(0.0, (now - start_dt).total_seconds() * 1000))
                    s["duration_ms"] = dur
                    s["durationMs"] = dur
        elif status in (None, "pending"):
            s["status"] = "skipped"
    return steps


def _iso_ms(dt: datetime) -> str:
    """The same millisecond-truncated ISO stamp ``job_steps`` writes, so one
    card's step windows are not written in two formats."""
    return dt.replace(microsecond=(dt.microsecond // 1000) * 1000).isoformat()


def _parse_iso_ms(raw: Any) -> Optional[datetime]:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def answer_step_label(*, rtl: bool = False) -> str:
    """The closing step every voice job carries — the round with no tools, in
    which the agent actually says the answer. Declared at open time and kept
    last so ``StepTracker.final_step_index`` has somewhere honest to point."""
    return (_LABELS_FA if rtl else _LABELS_EN)["answer"]


# ── Per-turn handle ───────────────────────────────────────────────────────
# Set by ``AgentRunner._run_inner`` and read by ``AgentRunner.run``'s finally,
# which is the only place a CANCELLED voice turn can still be caught. There is
# no Task boundary between those two frames, so the set is visible — unlike a
# ``.set()`` inside a ``gather``/``create_task``, which lands in a context copy
# and evaporates (the 2026-08-19 registry regression).
_VOICE_JOB_CTX: contextvars.ContextVar[Optional["VoiceTurnJob"]] = (
    contextvars.ContextVar("voice_turn_job", default=None)
)


def set_current_voice_job(job: Optional["VoiceTurnJob"]) -> None:
    _VOICE_JOB_CTX.set(job)


def current_voice_job() -> Optional["VoiceTurnJob"]:
    try:
        return _VOICE_JOB_CTX.get()
    except LookupError:  # pragma: no cover — default=None makes this dead
        return None


def sweep_current_voice_job() -> None:
    """Close this turn's voice job if the turn ended without sealing it.

    Called from ``AgentRunner.run``'s ``finally``, which is usually reached
    *because the task is being cancelled* — the caller hung up mid-turn. Never
    awaits, for exactly the reason ``_sweep_unclosed_created_jobs`` does not:
    an ``await`` inside that ``finally`` raises ``CancelledError`` immediately
    and the cleanup is skipped by the very condition that makes it necessary.

    Reaching here means ``finalize`` did not run, so THIS turn produced no
    answer — the happy path seals inside the turn and this is then a no-op. So
    the reason is :data:`SEAL_CANCELLED`, and the close checks the thread for a
    delivered answer before it settles on that word.
    """
    try:
        job = current_voice_job()
        set_current_voice_job(None)
        if job is not None:
            job.seal(reason=SEAL_CANCELLED)
    except Exception:  # noqa: BLE001 — cleanup must never mask the real error
        logger.exception("[voice-job] sweep failed")


class VoiceTurnJob:
    """The job card for one voice turn. Every public method is synchronous,
    returns immediately, and never raises."""

    __slots__ = (
        "_active", "_answer_label", "_conversation_id", "_delegation_id",
        "_job_id", "_lock", "_opened", "_opened_at", "_row_exists", "_rtl",
        "_sealed", "_steps", "_title", "_user_id",
    )

    def __init__(
        self,
        *,
        user_id: str,
        conversation_id: Optional[str],
        request_text: Optional[str],
        display_request: Optional[str] = None,
        delegation_id: Optional[str] = None,
    ) -> None:
        """``display_request`` is what the caller SAID, when the caller knows
        it separately from what the model was given. On the Live path they are
        different strings — `request_text` is a constructed prompt — and only
        the first of them may reach a screen.

        ``delegation_id`` names WHICH piece of work this card is, so the
        cancelled close can tell this turn's answer from a different one that
        landed in the same conversation at the same time — two concurrent
        delegations are an ordinary Live state, not an edge case.
        """
        from app.agent.job_titles import derive_job_title, is_rtl_text

        self._user_id = user_id
        self._conversation_id = conversation_id
        self._delegation_id = (
            str(delegation_id)[:_DELEGATION_ID_MAX] if delegation_id else None
        )
        _display = clean_request_text(display_request) or clean_request_text(request_text)
        self._rtl = is_rtl_text(_display if _display else request_text)
        _fallback = "جمع‌بندی درخواست صوتی" if self._rtl else "Voice request"
        _derived = derive_job_title(_display, fallback="")
        # A one-word title is a caption fragment. `derive_job_title` normalises
        # a REQUEST; it has no notion of "this is only part of one", so the
        # floor lives here, where the input is known to be a transcript. Which
        # floor depends on the SCRIPT: a space-free one measures the fragment
        # in characters, because there every request is one `split()` token.
        if _title_is_fragment(_derived):
            _derived = ""
        self._title = _derived or derive_job_title(None, fallback=_fallback)
        self._answer_label = answer_step_label(rtl=self._rtl)
        # Minted here, not by JobRunner: the id has to exist synchronously so
        # the StepTracker can stamp `job_id` on the tool frames of the very
        # round that opens the job. The row is inserted with this id off-turn.
        self._job_id: str = str(uuid.uuid4())
        # When this turn started, in the same naive-UTC clock the Message rows
        # use. The cancelled close asks whether an answer landed in the thread
        # AFTER this instant; without it the proof would match the question the
        # turn was answering.
        self._opened_at: datetime = datetime.utcnow()
        self._steps: List[Dict[str, Any]] = []
        self._active: int = 0
        self._opened = False
        self._row_exists = False
        self._sealed = False
        self._lock = asyncio.Lock()

    # ── read-only state ───────────────────────────────────────────────
    @property
    def job_id(self) -> Optional[str]:
        return self._job_id if self._opened else None

    @property
    def opened(self) -> bool:
        return self._opened

    @property
    def title(self) -> str:
        return self._title

    @property
    def delegation_id(self) -> Optional[str]:
        return self._delegation_id

    def step_labels(self) -> List[str]:
        return [str(s.get("label") or "") for s in self._steps]

    # ── the turn feeds these ──────────────────────────────────────────
    def plan(self, tool_calls: Iterable[Dict[str, Any]], tracker: Any = None) -> None:
        """Declare the steps for one round of tool calls, BEFORE they run.

        Called with the whole round so the card opens (and the phone's timer
        card starts) before the first search goes out, rather than after the
        batch comes back. Repeated calls of the same tool kind collapse onto
        the step already declared for it.

        ``tracker`` is adopted here and not left to :meth:`attribute`, because
        the runner's parallel batch captures ``event_fields()`` ONCE before it
        runs and emits every frame in the round from that snapshot. A tracker
        first synced in the per-call loop is synced after those frames have
        already gone out, so a batched round — which is what a voice research
        turn is — would ship its actions with no step on them.
        """
        try:
            fresh = False
            first: Optional[str] = None
            for tc in tool_calls or ():
                name = (tc or {}).get("name")
                if not is_work_tool(name):
                    continue
                label = step_label_for_tool(name, rtl=self._rtl)
                if first is None:
                    first = label
                if self._ensure_step(label):
                    fresh = True
            if first is None:
                return
            # The step this ROUND opens on. The batch's frames are all stamped
            # from one snapshot taken right after this call, so binding to the
            # round's first work tool is what a batched round needs; the
            # per-call `attribute` below refines each frame the loop emits
            # itself.
            was = self._active
            self._active = max(self._active, self._index_of(first))
            self.bind(tracker)
            if not fresh and self._active == was:
                return
            if not self._opened:
                self._opened = True
                self.bind(tracker)
                self._spawn(self._open())
            else:
                self._spawn(self._advance())
        except Exception:  # noqa: BLE001 — a turn must never fail on a card
            logger.exception("[voice-job] plan failed")

    def _index_of(self, label: str) -> int:
        for i, s in enumerate(self._steps):
            if s.get("label") == label:
                return i
        return self._active

    def attribute(self, tool_name: Optional[str], tracker: Any) -> None:
        """Point ``tracker`` at the step this tool call belongs to, so the
        tool frames and the persisted ``tool_events`` carry the same
        ``job_id`` / ``step_index`` / ``step_name`` a chat turn's do."""
        try:
            if not self._opened or tracker is None:
                return
            if not is_work_tool(tool_name):
                return
            label = step_label_for_tool(tool_name, rtl=self._rtl)
            idx = next(
                (i for i, s in enumerate(self._steps) if s.get("label") == label),
                None,
            )
            if idx is None:
                return
            # The TRACKER gets the true index — this action belongs to this
            # step whether or not the turn has been here before.
            self._sync_tracker(tracker, active=idx)
            # The CARD's progress only ever moves forward. A turn that goes
            # search → fetch → search revisits step 0, and a bar that walked
            # backwards would read as the job losing work it had already done.
            if idx > self._active:
                self._active = idx
                self._spawn(self._advance())
        except Exception:  # noqa: BLE001
            logger.debug("[voice-job] attribute failed", exc_info=True)

    def bind(self, tracker: Any) -> None:
        """Adopt this job into the turn's :class:`StepTracker`."""
        try:
            if self._opened and tracker is not None:
                self._sync_tracker(tracker, active=self._active)
        except Exception:  # noqa: BLE001
            logger.debug("[voice-job] bind failed", exc_info=True)

    def seal(
        self,
        *,
        final_text: str = "",
        total_tokens: Optional[int] = None,
        model: Optional[str] = None,
        reason: str = SEAL_COMPLETED,
    ) -> None:
        """Terminalise the card. Idempotent; safe from a ``finally``.

        ``reason`` is one of the three SEAL_* constants. It replaced an
        ``interrupted: bool``, which could only ever express two of the three
        outcomes — and the pair it collapsed ("stopped after the answer" and
        "stopped before it") are the two this round exists to tell apart.
        """
        try:
            if not self._opened or self._sealed:
                return
            self._sealed = True
            self._spawn(self._close(final_text, total_tokens, model, reason))
        except Exception:  # noqa: BLE001
            logger.exception("[voice-job] seal failed")

    # ── internals ─────────────────────────────────────────────────────
    def _spawn(self, coro) -> None:
        """Fire-and-forget through the shared helper — a bare create_task can
        be collected mid-await and abandon its DB session."""
        try:
            _spawn_bg(coro)
        except Exception:  # noqa: BLE001
            coro.close()
            logger.exception("[voice-job] could not spawn background write")

    def _ensure_step(self, label: str) -> bool:
        """Add ``label`` as a step if the card does not already carry it.
        Returns True when the step list changed. The answer step is kept
        last, always."""
        if any(s.get("label") == label for s in self._steps):
            return False
        step = {
            "id": str(uuid.uuid4()),
            "type": f"step_{len(self._steps)}",
            "label": label,
            "status": "pending",
        }
        if not self._steps:
            self._steps = [
                step,
                {
                    "id": str(uuid.uuid4()), "type": "step_answer",
                    "label": self._answer_label, "status": "pending",
                },
            ]
        else:
            self._steps.insert(len(self._steps) - 1, step)
            for i, s in enumerate(self._steps):
                s["type"] = "step_answer" if i == len(self._steps) - 1 else f"step_{i}"
        return True

    def _sync_tracker(self, tracker: Any, *, active: int) -> None:
        tracker.job_id = self._job_id
        tracker.steps = self.step_labels()
        tracker.steps_total = len(self._steps)
        tracker.job_type = self._job_type()
        tracker.active = max(0, min(active, len(self._steps) - 1))

    def _job_type(self) -> Optional[str]:
        try:
            from app.agent.job_type import classify_job_type
            return classify_job_type(self._title, None)
        except Exception:  # noqa: BLE001
            return None

    def _snapshot(self) -> List[Dict[str, Any]]:
        """A copy of the step list — the background writer must not hand the
        turn's own list to the ORM while the turn is still appending to it."""
        return [dict(s) for s in self._steps]

    # -- background writes (serialised behind one lock) ----------------
    async def _open(self) -> None:
        from app.agent.job_runner import JobRunner, TaskSpec
        from app.agent.job_steps import dump_steps, open_first_step
        from app.config import settings

        async with self._lock:
            now = datetime.utcnow()
            # Applied to the LIVE list, then snapshotted — not the other way
            # round. `open_first_step` is what stamps step 0's `started_at`,
            # and stamping it on a throwaway copy would leave the live list
            # unstamped, so the next `advance_steps` would close step 0 with
            # `now` for both edges and print the 0ms Round 8 removed.
            open_first_step(self._steps, now)
            steps = self._snapshot()
            job_type = self._job_type()
            try:
                await JobRunner().create_job(
                    job_id=self._job_id,
                    job_type="agent_task",
                    spec=TaskSpec(
                        user_id=self._user_id,
                        # Same deny-listed channel the create_job tool uses, so
                        # injected content cannot drive a mutating connector
                        # off the back of this row.
                        channel="agent_task",
                        # The agent opened this itself — no upstream event, and
                        # the same source_kind chat's own cards carry, so
                        # Mission Control does not need a voice special case.
                        source_kind="manual",
                        conversation_id=self._conversation_id,
                        # NOT `asst_message_id`: on voice the answer row is
                        # written by the platform relay under an id it mints
                        # itself (`_save_voice_messages`), so the runner's
                        # pre-minted id names a row that never exists. The
                        # reconciler treats that key as PROOF and would then
                        # never find any, stranding a card the watchdog exists
                        # to close. Leaving it unset drops the row onto the
                        # "any later assistant message in this conversation"
                        # proof path, which voice does satisfy.
                        # Durable link back to the Live delegation. A repaired
                        # socket may land on another platform replica, so the
                        # tenant job row is the source of truth for its card.
                        config_json={
                            "job_type": job_type, "voice_turn": True,
                            **({"voice_delegation_id": self._delegation_id}
                               if self._delegation_id else {}),
                        },
                    ),
                    title=self._title,
                    prompt=self._title,
                    status="running",
                    model=settings.agent_model,
                    layer=0,
                    steps_json=dump_steps(steps),
                )
            except Exception:  # noqa: BLE001
                logger.exception("[voice-job] could not open job %s", self._job_id[:8])
                return
            self._row_exists = True
            logger.info(
                "[voice-job] opened %s \"%s\" (%d steps)",
                self._job_id[:8], self._title[:60], len(steps),
            )
            # Inside the lock, deliberately. The push lane orders by insert
            # time, and an announce released before the next `_advance` takes
            # the lock can land a `progress` card ahead of `mission_started`.
            # Nothing on the turn's coroutine ever waits on this lock, so
            # holding it across the push costs the turn nothing.
            await self._announce_running(steps, started=True)

    async def _write_chat_card(self) -> None:
        """The card's row IN THE THREAD, so it survives a reload.

        A chat job gets this row opportunistically, from whichever ws_chat
        path is live when the ``job_update`` frame goes out. A voice turn
        touches neither: its socket is the audio relay on platform-api, and
        the agent's chat WS may not even be open. Without this the card would
        exist everywhere except the one place the user was told to look.

        Same row shape and, crucially, the same ``job-<id>`` primary key as
        the ws_chat writers — so if the phone's chat socket IS connected and
        forwards the frame, its existence check sees this row and the thread
        gets one card, not two.

        Written at CLOSE, not at open, and the reason is ordering. The thread
        is sorted by ``created_at``, and on voice the user's own row is
        written by the relay off a transcription event that races the model's
        `think` call — a card stamped at open can therefore land ABOVE the
        question it answers. By the time the turn ends the transcript row is
        long since in, and the relay writes the spoken reply immediately
        after, so the card lands between them: exactly where chat puts it.
        """
        from sqlalchemy import select as _sel
        from app.api.message_cards import job_marker_content
        from app.db.database import async_session_maker
        from app.db.message_helpers import resolve_day_chat_id_for_now
        from app.db.models import Message as _Msg

        if not self._conversation_id:
            return
        row_id = f"job-{self._job_id}"
        try:
            async with async_session_maker() as db:
                exists = (await db.execute(
                    _sel(_Msg.id).where(_Msg.id == row_id)
                )).first()
                if exists:
                    return
                db.add(_Msg(
                    id=row_id,
                    conversation_id=self._conversation_id,
                    day_chat_id=await resolve_day_chat_id_for_now(db, self._user_id),
                    role="job",
                    # The marker, through the ONE writer — never a local
                    # json.dumps, whose default `ensure_ascii=True` is what
                    # rendered this row's Persian title as a run of
                    # backslash-u escapes on the day a reader printed the
                    # marker instead of the card (Round 16). The readers
                    # blank it now; the shared writer is the second lock on
                    # the same door.
                    content=job_marker_content(
                        self._job_id, self._title, self._job_type(),
                    ),
                ))
                await db.commit()
        except Exception:  # noqa: BLE001 — the card is not worth a raise
            logger.debug("[voice-job] chat card row failed", exc_info=True)

    async def _advance(self) -> None:
        """Re-write the step list and push progress. Cheap and idempotent —
        the card's steps grow as the turn discovers what it needs."""
        from app.agent.job_steps import (
            advance_steps, counts, dump_steps, open_first_step,
        )
        from app.db.database import async_session_maker
        from app.db.models import BuildJob, JobEvent

        async with self._lock:
            if not self._row_exists:
                return
            now = datetime.utcnow()
            # Live list first, snapshot second — see `_open`. Everything
            # before the active step is finished; the active one is running.
            # Same transitions, and the same per-step windows, the
            # `update_job` tool writes, from the same module.
            if self._active > 0:
                advance_steps(self._steps, self._active - 1, now)
            else:
                open_first_step(self._steps, now)
            steps = self._snapshot()
            done, total = counts(steps)
            try:
                async with async_session_maker() as db:
                    job = await db.get(BuildJob, self._job_id)
                    if job is None or job.status != "running":
                        return
                    job.steps_json = dump_steps(steps)
                    db.add(JobEvent(
                        job_id=self._job_id, user_id=self._user_id,
                        kind="info", level="info",
                        label=f"Progress: {done}/{total} steps"[:200],
                    ))
                    await db.commit()
            except Exception:  # noqa: BLE001
                logger.debug("[voice-job] advance failed", exc_info=True)
                return
            await self._announce_running(steps, started=False)

    async def _announce_running(
        self, steps: List[Dict[str, Any]], *, started: bool,
    ) -> None:
        """The live surfaces: the in-app ``job_update`` frame and the phone
        card. Byte-for-byte the frame ``create_job`` / ``update_job`` emit —
        the clients must not be able to tell which producer wrote it."""
        from app.agent.job_steps import counts, current_label
        done, total = counts(steps)
        label = current_label(steps)
        job_type = self._job_type()
        try:
            from app.api.ws_chat import broadcast_to_user
            await broadcast_to_user(self._user_id, {
                "type": "job_update",
                "job_id": self._job_id,
                "job_type": job_type,
                "name": self._title,
                "status": "running",
                "step": label or "Working...",
                "total_steps": total,
                "completed_steps": done,
                "chat_id": self._conversation_id,
                # No `message_id`: on voice the answer row is written by the
                # platform relay under an id it mints itself, so the runner's
                # pre-minted id would deep-link the card to a message that
                # does not exist. The chat_id alone opens the conversation.
                "message_id": None,
            })
        except Exception:  # noqa: BLE001
            logger.debug("[voice-job] job_update broadcast failed", exc_info=True)

        try:
            import time as _t
            from app.agent.subagent_orchestrator import _notify_job_event
            from app.services.plain_text import (
                humanize_label as _hl, plain_preview as _plain,
            )
            if started:
                await _notify_job_event(
                    job_id=self._job_id, label=self._title,
                    kind="mission_started",
                    title=f"Working on: {_hl(_plain(self._title, 150))}",
                    body=_plain(self._title, 200),
                    # Indeterminate timer, never progress=0 — `_content_state`
                    # prefers a timer, and a bare 0 ships a card frozen at
                    # "0%" for the whole turn. Window matches the reaper's
                    # 30-minute stall cutoff, same as create_job's.
                    timer_end_ms=int((_t.time() + 1800) * 1000),
                    dedup_suffix="started",
                    chat_id=self._conversation_id, message_id=None,
                    job_type=job_type, step_name=label or None,
                    steps_done=done, steps_total=total,
                    refresh_if_started=bool(self._conversation_id),
                )
            else:
                # `progress`, NOT a "mission_progress" — KNOWN_NOTIFY_KINDS is
                # a closed enum validated at ingest (agent_notify rejects an
                # unknown kind with a 422), and `progress` is the Live-Activity
                # -update-only lane that never fires an OS alert. Identical
                # call shape to `update_job`'s own progress tick, low priority
                # so it stays outside the APNs alert budget.
                await _notify_job_event(
                    job_id=self._job_id, label=self._title,
                    kind="progress",
                    title=f"Working on: {_hl(_plain(self._title, 150))}",
                    body=label or None,
                    progress=int(done / total * 100) if total else None,
                    priority="low", dedup_suffix="progress",
                    chat_id=self._conversation_id,
                    message_id=None,
                    job_type=job_type, step_name=label or None,
                    steps_done=done, steps_total=total,
                )
        except Exception:  # noqa: BLE001
            logger.debug("[voice-job] card push failed", exc_info=True)

    @staticmethod
    def _voice_of(raw: Any) -> Optional[Dict[str, Any]]:
        """``metadata_json.voice`` of one row, or None when it has none."""
        if isinstance(raw, dict):
            meta: Any = raw
        elif isinstance(raw, str) and raw:
            try:
                meta = json.loads(raw)
            except (ValueError, TypeError):
                return None
        else:
            return None
        if not isinstance(meta, dict):
            return None
        voice = meta.get("voice")
        return voice if isinstance(voice, dict) else None

    def _row_is_proof(self, raw_metadata: Any) -> bool:
        """Is THIS assistant row the answer to THIS turn?

        A row with no ``voice`` provenance at all is the Realtime path, where
        the only assistant writer is the turn itself — unchanged behaviour, and
        the reason this is not simply "voice.source == 'delegated'".

        A ``live_spoken`` row is NEVER proof. That is the whole defect: the
        Live prompt tells the model to "say one short line and delegate", so a
        filler line — and therefore a spoken row — lands in the same
        conversation a second after the job opens, and the cancelled outcome
        became unreachable in exactly the scenario it was written for.
        """
        voice = self._voice_of(raw_metadata)
        if voice is None:
            return True
        if str(voice.get("source") or "") != "delegated":
            return False
        if voice.get("cancelled") or voice.get("superseded"):
            return False
        if self._delegation_id is None:
            return True
        # Two concurrent delegations are an ordinary Live state (D9 allows
        # two, queues the third). The OTHER one's answer is somebody else's
        # proof.
        #
        # Both sides are bounded to the SAME length before comparing. The id is
        # provider-supplied and unbounded, and it reaches the two sides of this
        # comparison down different paths with different bounds: the card's
        # copy came through the relay body (`[:64]`) and `ChatRequest`
        # (max_length=64), while the row's copy is stamped straight onto the
        # persisted message. Compared raw, an id longer than 64 characters
        # makes this permanently False — so every DELIVERED delegated answer
        # would close `cancelled`, which is the A5-05 lie inverted and looks
        # correct from outside.
        return str(voice.get("delegation_id") or "")[:_DELEGATION_ID_MAX] == self._delegation_id

    async def _answer_was_delivered(self) -> bool:
        """Did THIS turn's answer land in this conversation after it opened?

        The one proof of delivery this process can see. On voice the answer row
        is written by the platform relay under an id this container never
        learns, so there is nothing to look up by key — but the relay writes it
        into the SAME conversation, stamped with the provenance
        (``metadata_json.voice``) that says which writer produced it. That is
        the proof path ``job_reconciler.reconcile_delivered_turn_jobs`` trusts,
        narrowed: that watchdog only ever looks at ``running`` rows, so a card
        closed ``cancelled`` can never be upgraded later, and the question has
        to be asked here, once, before the row goes terminal.
        """
        from sqlalchemy import select as _sel
        from app.db.database import async_session_maker
        from app.db.models import Message as _Msg

        if not self._conversation_id:
            return False
        try:
            async with async_session_maker() as db:
                rows = (await db.execute(
                    _sel(_Msg.metadata_json).where(
                        _Msg.conversation_id == self._conversation_id,
                        _Msg.role == "assistant",
                        _Msg.created_at > self._opened_at,
                    ).order_by(_Msg.created_at.desc()).limit(_PROOF_ROW_SCAN)
                )).all()
                return any(self._row_is_proof(r[0]) for r in rows)
        except Exception:  # noqa: BLE001
            # Fail towards the honest word. A close that cannot prove delivery
            # must not claim it.
            logger.debug("[voice-job] delivery proof read failed", exc_info=True)
            return False

    async def _await_answer_proof(self) -> bool:
        """`_answer_was_delivered`, polled to a ceiling.

        Asks IMMEDIATELY first: an answer that landed before the cancellation
        reached us is already in the thread, and the old single sleep paid a
        flat 2 s to learn what the first read would have said. Then re-reads
        until the ceiling, so the window this has to cover is bounded by a
        number instead of assumed to be one.

        Fails towards the honest word: the ceiling expiring means "no proof",
        never "delivered".
        """
        deadline = time.monotonic() + _CANCEL_PROOF_MAX_WAIT_S
        while True:
            if await self._answer_was_delivered():
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(_CANCEL_PROOF_POLL_S)

    async def _close(
        self, final_text: str, total_tokens: Optional[int],
        model: Optional[str], reason: str,
    ) -> None:
        """Terminalise through the SHARED closer, so the row, the step
        windows, the ``job_events`` heartbeat and the terminal push are the
        ones every other agent-authored job gets."""
        from app.agent.job_reconciler import announce_completed, close_job_completed
        from app.db.database import async_session_maker

        if reason == SEAL_CANCELLED:
            # A cancel reaches here because `AgentRunner.run`'s finally ran
            # without `finalize` — so THIS turn produced no answer. It does not
            # follow that the user got nothing: the relay may already have
            # persisted the answer of a delegation that was detached rather
            # than killed. Ask — repeatedly, up to a ceiling — and upgrade to
            # the honest completed word when the thread says so.
            if await self._await_answer_proof():
                reason = SEAL_INTERRUPTED_AFTER_ANSWER

        if reason == SEAL_CANCELLED:
            await self._close_cancelled()
            return

        interrupted = reason == SEAL_INTERRUPTED_AFTER_ANSWER
        async with self._lock:
            if not self._row_exists:
                # The insert never landed (or is still in flight and failed):
                # there is nothing to close, and the reaper has no row either.
                return
            await self._write_chat_card()
            now = datetime.utcnow()
            try:
                async with async_session_maker() as db:
                    closed = await close_job_completed(
                        db, self._job_id, user_id=self._user_id, now=now,
                        # No message_id: the answer row is the relay's, under
                        # an id this process never sees. The reconciler's
                        # conversation-scoped proof covers the back-link.
                        message_id=None,
                        total_tokens=total_tokens, model=model,
                        reason="turn_end",
                    )
                    if closed is None:
                        # Already terminal — the sweep, the reconciler or the
                        # reaper got here first. Guarded UPDATE, so exactly one
                        # of us pushes the card.
                        return
                    await db.commit()
            except Exception:  # noqa: BLE001
                logger.exception("[voice-job] could not close %s", self._job_id[:8])
                return
        logger.info(
            "[voice-job] closed %s \"%s\"%s",
            self._job_id[:8], self._title[:60],
            " (turn interrupted)" if interrupted else "",
        )
        # An interrupted voice turn whose answer WAS delivered still gets a
        # COMPLETED card, not a failed one: the caller hanging up after hearing
        # the reply is how a voice call normally ends, and "Didn't finish" over
        # work the agent already spoke aloud is the exact lie
        # job_status.turn_interrupted was written to stop telling. Only the
        # preview differs — there is no delivered answer text to show. A turn
        # stopped BEFORE delivery never reaches here; see `_close_cancelled`.
        await announce_completed(
            closed, message_id=None,
            preview=None if interrupted else (final_text or None),
            chat_id_fallback=self._conversation_id,
        )

    async def _close_cancelled(self) -> None:
        """Terminalise a card whose work was stopped before anything was
        delivered.

        Not `close_job_completed`: that closer's whole job is to mark every
        remaining step done, which is the claim being corrected here. The steps
        keep the state they actually reached — the searches that ran really
        ran, and the card stays worth opening — and the row goes `cancelled`,
        a value both clients already treat as terminal
        (`job_status.TERMINAL_STATUSES`).

        Same guarded `status == 'running'` UPDATE every other closer uses, so
        the sweep, the reconciler and the reaper still resolve to exactly one
        writer and the phone card is pushed exactly once.
        """
        from sqlalchemy import update as _upd
        from app.agent.job_status import STATUS_CANCELLED, turn_interrupted
        from app.agent.job_steps import counts, dump_steps
        from app.db.database import async_session_maker
        from app.db.models import BuildJob, JobEvent

        async with self._lock:
            if not self._row_exists:
                return
            await self._write_chat_card()
            now = datetime.utcnow()
            steps = _stop_open_steps(self._snapshot(), now)
            done, total = counts(steps)
            verdict = turn_interrupted()
            try:
                async with async_session_maker() as db:
                    res = await db.execute(
                        _upd(BuildJob)
                        .where(BuildJob.id == self._job_id,
                               BuildJob.user_id == self._user_id,
                               BuildJob.status == "running")
                        .values(
                            status=STATUS_CANCELLED,
                            completed_at=now,
                            # The steps as they stood. `finish_all_steps` would
                            # paint the unreached ones done, which is the
                            # overclaim.
                            steps_json=dump_steps(steps),
                            error_class=verdict.error_class,
                            # NOT `verdict.user_message`: that sentence is
                            # English-only, and the card detail surface renders
                            # it beside step labels this module has already
                            # localised. One card, one language.
                            user_message=(_STOPPED_BODY_FA if self._rtl
                                          else _STOPPED_BODY_EN),
                        )
                        .returning(BuildJob.id)
                    )
                    if res.first() is None:
                        return
                    db.add(JobEvent(
                        job_id=self._job_id, user_id=self._user_id,
                        kind="info", level="info",
                        status=STATUS_CANCELLED, ts=now,
                        label=(f"Stopped: {done}/{total} steps"
                               if total else "Stopped")[:200],
                    ))
                    await db.commit()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "[voice-job] could not cancel %s", self._job_id[:8])
                return
        # The title is the user's own utterance; the log carries its length,
        # never its words (ids, counts, hashes only on this surface).
        logger.info(
            "[voice-job] cancelled %s title_len=%d (%d/%d steps, nothing delivered)",
            self._job_id[:8], len(self._title), done, total,
        )
        await self._announce_cancelled(done, total)

    async def _announce_cancelled(self, done: int, total: int) -> None:
        """The live surfaces for a stopped card.

        The in-app frame is the whole announcement, and there is deliberately
        NO push.

        A Live Activity is closed only by a terminal notification, and the two
        terminal kinds are `mission_completed` and `mission_failed`
        (KNOWN_NOTIFY_KINDS is a closed enum validated at ingest, in
        `db/models/notification.py`). `mission_failed` maps to the widget's
        `failed` phase — `JOB_PHASE_BY_KIND` in subagent_orchestrator.py, and
        the phone's own `JobFace.Phase` — which draws a red mark and the words
        "Didn't finish" over work the user STOPPED. Announcing a failure for a
        cancellation is the same class of lie as announcing a completion for
        one, and neither the kind enum nor the widget's phase vocabulary can be
        extended from this module's files.
        [No `phase` override either: an unknown phase string makes the widget
        fall through to its legacy derivation, which reads a non-empty step
        line as work IN PROGRESS — a terminal card drawn as running.]
        So: the `job_update` frame above carries the cancellation to every open
        client, the row carries it to Mission Control and to a card opened
        later, and the lock-screen card is left to the conversation's NEXT turn
        (the Live Activity is keyed `chatjob:<chat_id>`, so the next
        `mission_started` refreshes it in place) or to its own 30-minute stale
        date.
        """
        try:
            from app.api.ws_chat import broadcast_to_user
            await broadcast_to_user(self._user_id, {
                "type": "job_update",
                "job_id": self._job_id,
                "job_type": self._job_type(),
                "name": self._title,
                "status": "cancelled",
                "step": _STOPPED_STEP_FA if self._rtl else _STOPPED_STEP_EN,
                "total_steps": total,
                "completed_steps": done,
                "chat_id": self._conversation_id,
                "message_id": None,
            })
        except Exception:  # noqa: BLE001
            logger.debug("[voice-job] cancel broadcast failed", exc_info=True)
