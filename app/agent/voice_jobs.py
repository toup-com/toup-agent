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

Voice round 2026-10-05 (D-P4/D-P5): the class is ``TurnJob`` now, one
producer for both channels. A Live delegation's card is opened EAGERLY at the
top of the turn with a plan declared from the request (``template_plan``), so
the call has a card before phase 1 ends; steps start when a tool lands on
them, an unforeseen tool kind is inserted before the answer step, and a
planned step that never ran closes ``skipped``. A chat turn whose model does
not call ``create_job`` gets the same card, opened lazily at its first work
round. Every state change is mirrored onto the delegation stream as a
``job.update`` frame (``frame_sink``) and onto the chat socket as
``job_update`` with the step list and the sources it read.

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
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.services.background_tasks import spawn as _spawn_bg

logger = logging.getLogger(__name__)

__all__ = [
    "SEAL_CANCELLED",
    "SEAL_COMPLETED",
    "SEAL_INTERRUPTED_AFTER_ANSWER",
    "CHAT_TURN_JOB_CHANNELS",
    "TurnJob",
    "VoiceTurnJob",
    "clean_request_text",
    "current_voice_job",
    "explicit_search_requested",
    "is_work_tool",
    "set_current_voice_job",
    "step_label_for_tool",
    "sweep_current_voice_job",
    "template_plan",
]

#: Interactive chat channels whose turns get the runner-minted card when the
#: model does not open one itself (D-P4: one producer for voice and chat).
#: Routines, triggers, sub-agents, missions and vibecoding own their jobs.
CHAT_TURN_JOB_CHANNELS: frozenset = frozenset({"web", "mobile", "app"})

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
    "navigate_to", "play_media", "media_control", "play_netflix", "radio",
    "tts", "tts_prefs", "talk_mode",
    "memory_search", "recall_day",
    "session_status", "sessions_list", "sessions_history", "thread",
    "start_mission", "spawn",
    # The supervisor (`delegate`) re-plans THIS card itself (`follow_steps`);
    # counted as work it would also add a step of its own beside that plan.
    "delegate",
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
    "media": "Control the music",
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
    "media": "کنترل پخش",
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
    # D-T1: both media tools land on the card's media step. They stay
    # NON-work (the phone's player is the surface; neither earns a card of
    # its own), but a planned media step is started by them — see `attribute`.
    "play_media": "media", "media_control": "media",
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


# ── An explicit ask to search ─────────────────────────────────────────────
# "search it", «سرچ کن», «چرا سرچ نمی کنی» — the caller named the method, so
# the answer must come from the web this turn, never from the model's memory
# (the 2026-10-05 call: three asks, a guess, then "I didn't search"). Closed
# grammar, both languages. A question ABOUT a search («اینو از کجا سرچ
# کردی», "where did you search") is not a new ask, and a search over the
# user's own data (inbox, calendar, files) is not a web search.
_SEARCH_ASK_RE = re.compile(
    r"\b(?:search(?:\s+for)?|google|look\s+(?:it\s+|that\s+|this\s+)?up|lookup"
    r"|find\s+out|check\s+online|web\s+search)\b"
    r"|سرچ|جستجو|جست\s*و\s*جو|گوگل",
    re.IGNORECASE,
)
_SEARCH_PROVENANCE_RE = re.compile(
    r"\bwhere\s+(?:did|do)\s+you\s+(?:search|look|find|get)\b"
    r"|\bwhat\s+did\s+you\s+(?:search|google|look\s+up)\b"
    r"|از\s*کجا|کجا\s*(?:رو|را)?\s*سرچ",
    re.IGNORECASE,
)
_OWNED_SEARCH_EN_RE = re.compile(
    r"\b(?:my|in\s+my|from\s+my|through\s+my)\s+(?:e-?mails?|inbox|gmail|calendar"
    r"|files?|drive|messages?|slack|notes?|docs?|documents?|chats?)\b",
    re.IGNORECASE,
)


# Phase D (lens-tenant-2): the caller's OWN data in Persian, ANCHORED.
# `query_intent._PERSIAN_OWNED_DATA_RE` (the classifier's, unchanged) is an
# unanchored substring list — «قراره» (is supposed to), «قرارداد», «پیامد»,
# «چت جی پی تی», «گزارش هواشناسی», «جلسه امروز مجلس» all matched — and the
# runner's T4 guard used it to STRIP the relay's `require_web` from ordinary
# research asks (no speculative search, no forced web round).  Own data is a
# data noun that is the caller's: a possessive suffix (تقویمم، ایمیلام،
# پیام‌هام، جلسه‌ام), «… من», a colloquial plural (ایمیلا، پیامای), the noun
# phrase's next word marked possessive (جلسه فردام، قرار بعدیم، فایل
# قراردادم), or a connector named as a place or an object (تو اسلک، تقویمو).
_FA_OWNED_NOUNS = (
    r"ایمیل|رایانامه|جیمیل|پیام|چت|مکالمه|واتساپ|تلگرام|اسلک|مخاطب|تقویم|"
    r"جلسه|قرار|رویداد|فایل|پرونده|پوشه|سند|اسناد|درایو|یادداشت|وظیفه|وظایف|اینباکس"
)
_FA_OWNED_PLACES = r"جیمیل|اسلک|واتساپ|تلگرام|تقویم|درایو|ایمیل|اینباکس"
#: Words ending in «م» that are not a possessive (ordinals and common ones).
_FA_NOT_POSSESSIVE = frozenset(
    "دوم سوم چهارم پنجم ششم هفتم هشتم نهم دهم تمام کدام نام مردم عموم مهم لازم "
    "کم هم کلام سلام پیام قدیم مستقیم اعلام انجام تیم سیستم فیلم علم حجم رقم "
    "خانم آدم رسم مدام دائم عظیم".split()
)
_FA_OWNED_RE = re.compile(
    r"(?<!\S)(?:" + _FA_OWNED_NOUNS + r")(?:\s?(?:ها|های))?\s?(?:م|ام|مو|امو|مم)(?!\S)"
    r"|(?<!\S)(?:" + _FA_OWNED_NOUNS + r")(?:\s?(?:ها|های)|ی)?\s+من(?!\S)"
    r"|(?<!\S)(?:" + _FA_OWNED_NOUNS + r")(?:ا|ای|امون)(?!\S)"
    r"|(?<!\S)(?:تو|توی|در|از|داخل)\s+(?:" + _FA_OWNED_PLACES + r")(?:\s?(?:ها|های))?(?!\S)"
    r"|(?<!\S)(?:" + _FA_OWNED_PLACES + r")(?:\s?رو|و)(?!\S)",
)
_FA_OWNED_PHRASE_RE = re.compile(
    r"(?<!\S)(?:" + _FA_OWNED_NOUNS + r")(?:ی|ای|ا)?\s+(\S{2,}?)(?:م|ام)(?!\S)",
)


def owned_data_request(text: Optional[str]) -> bool:
    """True when ``text`` asks about the caller's OWN data — mail, calendar,
    files, messages — en + fa; reused by the runner's `require_web` guard
    (voice round 2026-10-07, T4): a relay that marks «تقویممو سرچ کن» as
    research must not force a WEB search over it.  The Persian half is the
    ANCHORED rule above (Phase D lens-tenant-2), not the classifier's
    substring list."""
    if not text or not isinstance(text, str):
        return False
    try:
        from app.agent.query_intent import normalize_fa
        norm = normalize_fa(text.lower()).replace("\u200c", " ")
    except Exception:  # noqa: BLE001 — a missing helper keeps the raw text
        norm = text.lower().replace("\u200c", " ")
    norm = re.sub(r"[،,.!?؟:;«»\"']", " ", norm)
    if _OWNED_SEARCH_EN_RE.search(norm) or _FA_OWNED_RE.search(norm):
        return True
    for match in _FA_OWNED_PHRASE_RE.finditer(norm):
        word = match.group(0).split()[-1]
        if word not in _FA_NOT_POSSESSIVE:
            return True
    return False


def explicit_search_requested(text: Optional[str]) -> bool:
    """True when ``text`` (the caller's own words) explicitly asks to search
    the web — the predicate that forces the web tools on the first round and
    plans the card as a search."""
    if not text or not isinstance(text, str):
        return False
    try:
        from app.agent.query_intent import (
            _PERSIAN_OWNED_DATA_RE, normalize_fa,
        )
        norm = normalize_fa(text.lower()).replace("\u200c", " ")
    except Exception:  # noqa: BLE001 — a missing helper must not hide the ask
        _PERSIAN_OWNED_DATA_RE = None  # type: ignore[assignment]
        norm = text.lower().replace("\u200c", " ")
    if not _SEARCH_ASK_RE.search(norm):
        return False
    if _SEARCH_PROVENANCE_RE.search(norm):
        return False
    if _OWNED_SEARCH_EN_RE.search(norm):
        return False
    if _PERSIAN_OWNED_DATA_RE is not None and _PERSIAN_OWNED_DATA_RE.search(norm):
        return False
    return True


# ── The research budget's depth ask (voice round 2026-10-07, T7) ─────────
# A spoken answer gets a small research budget (`voice_research_max_searches`
# / `_max_reads`); a caller who asks for depth gets ×`voice_research_deep_factor`.
# «دقیق» counts only as a whole word: «دقیقا/دقیقاً» is "exactly" and «دقیقه»
# is "minute" — neither asks for depth.
_DEPTH_EN_RE = re.compile(
    # "deep learning", DeepMind, DeepSeek, a deepfake are TOPICS, not depth.
    r"\b(?:thorough(?:ly)?|in[- ]depth|deeply|deep(?!\s*(?:learning|mind|seek|fakes?)\b)"
    r"|detailed|comprehensive"
    r"|research\s+(?:it|this|that))\b",
    re.IGNORECASE,
)
_DEPTH_FA_RE = re.compile(
    r"(?<!\S)(?:کامل|مفصل|عمیق|دقیق)(?!\S)|با\s+جزئیات|با\s+جزییات|همه\s*جانبه"
    r"|تحقیق\s*(?:کن|بکن)",
)


def research_depth_requested(text: Optional[str]) -> bool:
    """True when ``text`` (the caller's words) asks for a thorough answer."""
    if not text or not isinstance(text, str):
        return False
    norm = text.replace("\u200c", " ")
    if _DEPTH_EN_RE.search(norm):
        return True
    try:
        from app.agent.query_intent import normalize_fa
        norm = normalize_fa(norm)
    except Exception:  # noqa: BLE001 — a missing helper keeps the raw text
        pass
    return bool(_DEPTH_FA_RE.search(norm))


# ── The speculative first search (D-T3) ──────────────────────────────────
# A voice delegation that must search starts its first search at t0, before
# the model's first round (3.4–6.4 s measured) asks for it. The query is the
# caller's request with the search ASK removed ("search it", «سرچ کن») — no
# model call. A wrong query costs one extra search the model makes anyway.
_SPEC_FILLER_RE = re.compile(
    r"\b(?:please|can\s+you|could\s+you|for\s+me|it|that|this|now|quickly|ok(?:ay)?)\b"
    r"|(?<!\S)(?:برام|برامون|لطفا|لطفاً|میشه|می\s*شه|اینو|اونو|رو|را|کن|بکن|هم|یه|یک|خب|حالا"
    # Voice round 2026-10-07 (T12): the call's queries carried the caller's
    # hesitation and discourse words («باشه … ببین … اه …»). NOT bare «چیز»
    # (a noun) nor bare «بعد» (after) — only these.
    r"|اه|ببین|ببینم|باشه|اوکی|الان|چیزی|سؤال|سوال|بعد\s+به\s+نظرت|به\s+نظرت)(?!\S)",
    re.IGNORECASE,
)
_SPEC_MIN_WORDS = 2
_SPEC_MAX_CHARS = 200


def speculative_search_query(text: Optional[str]) -> Optional[str]:
    """The query the speculative first search runs, or None when the request
    minus its search ask leaves nothing worth searching (< 2 words)."""
    if not text or not isinstance(text, str):
        return None
    q = text.replace("\u200c", " ")
    q = _SEARCH_ASK_RE.sub(" ", q)
    # Punctuation first (T12): «سرچ کن.» left `کن.`, and the filler's `(?!\S)`
    # does not match a word followed by its own period.
    q = re.sub(r"[\s،,.!?؟:;]+", " ", q)
    q = _SPEC_FILLER_RE.sub(" ", q)
    q = re.sub(r"\s+", " ", q).strip()
    if len(q.split()) < _SPEC_MIN_WORDS:
        return None
    return q[:_SPEC_MAX_CHARS]


def speculative_search_block(query: str, result: str) -> str:
    """The context block a speculative result rides into round 0 as (critic
    R8): a plain user-role block, NOT a fabricated tool_use/tool_result pair —
    a reasoning model's provider can refuse a function_call item it did not
    emit, and nothing here can prove it accepts one. The result string is the
    tool's own, already fenced as external content by the executor."""
    return (
        f"<speculative_search query=\"{query}\">\n"
        "A web_search for the request above ran while you were reading it; its "
        "results are below and already count as this turn's search. Answer from "
        "them, search again only if they do not answer the request, and cite "
        "only URLs that appear in them.\n"
        f"{result}\n"
        "</speculative_search>"
    )


# ── A media command ───────────────────────────────────────────────────────
# Plans the card's media step (D-T1). Closed and conservative — it only picks
# a step label, never a tool: play / stop / pause / resume / skip, en + fa,
# with the clitic forms the 2026-10-06 call used («استاپش کن»).
_MEDIA_COMMAND_RE = re.compile(
    r"^\W*(?:(?:please|ok(?:ay)?|hey)\W+)?(?:stop|pause|resume|unpause|skip|kill)"
    r"(?:\s+(?:it|this|that|playing|the\s+(?:music|song|track)|music|song|track))?\W*$"
    r"|\b(?:stop|pause|resume|unpause|kill|skip)\s+(?:the\s+)?(?:music|song|track|playback)\b"
    r"|\b(?:next|previous|last)\s+(?:song|track)\b"
    r"|\bplay\b.*\b(?:songs?|music|tracks?|albums?|playlists?|by)\b"
    r"|(?:استاپ|استپ|قطع|خاموش|متوقف|مکث|پاز)ش?\s*(?:کن|بکن|شه)"
    r"|^\s*بسه\s*$|نگهش\s*دار"
    r"|آهنگ\s*(?:بعدی|قبلی)|(?:آهنگ|موزیک|موسیقی)\S*\s*.*(?:پخش|بذار|بزار|بزن)",
    re.IGNORECASE,
)


def media_command_requested(text: Optional[str]) -> bool:
    """True when ``text`` (the caller's words) is a play/stop/pause/resume/
    skip request — the card then declares one media step."""
    if not text or not isinstance(text, str):
        return False
    return bool(_MEDIA_COMMAND_RE.search(text.replace("\u200c", " ")))


# ── A multi-part (heavy) request ─────────────────────────────────────────
# Voice round 2026-10-06, D-T2: the supervisor (`delegate`) is for a request
# with several searches or parts — "compare the three best X with sources" —
# and the turn's first round could not reach it: a search ask forced round 0 to
# the web tools only, and a `question` intent never offered it. This predicate
# is what lifts both, so it is CONSERVATIVE: a comparison / ranking of several,
# two real questions, or two clauses joined by an explicit "and also". A
# single lookup («یواف‌سی چی», "one more thing, Miami FC") is never multi-part.
_MULTI_COMPARE_RE = re.compile(
    # Phase D (correctness-tenant-4): a bare "vs" is a FIXTURE ("Lakers vs
    # Celtics score"), not a comparison — only with a comparison ask; a
    # ranking needs a plural noun after its number ("best 2 for 1" is a deal).
    r"\bcompar(?:e|es|ed|ing|ison)\b|\bpros\s+and\s+cons\b"
    r"|(?=.*\b(?:better|worse|which|pros|difference|should\s+i)\b)"
    r".*(?:\bversus\b|\bvs\.?(?=\s))"
    r"|\b(?:top|best)\s+(?:two|three|four|five|six|[2-9])\s+\w+s\b"
    r"|\b(?:two|three|four|five|six|[2-9])\s+(?:best|top)\b"
    r"|مقایسه"
    r"|(?:دو|سه|چهار|پنج|شش|[۲-۹2-9])\s*تا\s*(?:از\s*)?(?:بهترین|برترین)"
    r"|(?:بهترین|برترین)\s*(?:دو|سه|چهار|پنج|شش|[۲-۹2-9])\s*(?:تا|مورد)",
    re.IGNORECASE,
)
_MULTI_JOIN_RE = re.compile(
    # Not a bare "plus" (Disney plus, iPhone 16 plus) nor "and then" (one
    # narrative question, one reminder with two steps) — Phase D, tenant-4.
    r"\b(?:and\s+also|as\s+well\s+as)\b"
    r"|\sو\s*همچنین\s|\sو\s*بعدش\s|\sو\s*بعد\s+هم\s",
    re.IGNORECASE,
)
_MULTI_MIN_WORDS = 4
_MULTI_CLAUSE_MIN_WORDS = 3
_MULTI_QUESTION_MIN_WORDS = 3   # "what is the score? who scored?" is one ask


def multi_part_request(text: Optional[str]) -> bool:
    """True when ``text`` (the caller's own words) is a heavy, multi-part ask
    — the one shape the supervisor exists for. fa + en, closed grammar.

    Never for a media command, an owned-data ask (inbox, calendar, files —
    the same exclusions `explicit_search_requested` makes) or anything under
    four words."""
    if not text or not isinstance(text, str):
        return False
    try:
        from app.agent.query_intent import _PERSIAN_OWNED_DATA_RE, normalize_fa
        norm = normalize_fa(text.lower()).replace("\u200c", " ")
    except Exception:  # noqa: BLE001 — a missing helper must not hide the ask
        _PERSIAN_OWNED_DATA_RE = None  # type: ignore[assignment]
        norm = text.lower().replace("\u200c", " ")
    words = norm.split()
    if len(words) < _MULTI_MIN_WORDS or media_command_requested(text):
        return False
    if _OWNED_SEARCH_EN_RE.search(norm):
        return False
    if _PERSIAN_OWNED_DATA_RE is not None and _PERSIAN_OWNED_DATA_RE.search(norm):
        return False
    if _MULTI_COMPARE_RE.search(norm):
        return True
    questions = [q for q in re.split(r"[?؟]", norm) if q.strip()]
    if (len(re.findall(r"[?؟]", norm)) >= 2 and len(questions) >= 2
            and all(len(q.split()) >= _MULTI_QUESTION_MIN_WORDS for q in questions[:2])):
        return True
    parts = _MULTI_JOIN_RE.split(f" {norm} ", maxsplit=1)
    if len(parts) == 2 and all(
        len(p.split()) >= _MULTI_CLAUSE_MIN_WORDS for p in parts
    ):
        return True
    return False


#: Connector names a request can name, mapped to the brand the connector
#: tool's label carries (``_brand`` of the tool prefix), so a planned
#: "Check Gmail" is the step a ``gmail__…`` call lands on.
_PLAN_BRANDS_RE = re.compile(
    r"\b(gmail|google\s+calendar|calendar|google\s+drive|drive|outlook|github"
    r"|linkedin|notion|linear|slack|stripe|figma)\b"
    r"|(جیمیل|تقویم|اسلک|گیت\s*هاب|نوشن|درایو|لینکدین)",
    re.IGNORECASE,
)
_PLAN_BRAND_NAMES: Dict[str, str] = {
    "google calendar": "Calendar", "google drive": "Drive",
    "جیمیل": "Gmail", "تقویم": "Calendar", "اسلک": "Slack",
    "گیت هاب": "GitHub", "گیتهاب": "GitHub", "نوشن": "Notion",
    "درایو": "Drive", "لینکدین": "LinkedIn",
}


def _planned_brand(text: str) -> Optional[str]:
    m = _PLAN_BRANDS_RE.search(text or "")
    if not m:
        return None
    word = re.sub(r"\s+", " ", (m.group(1) or m.group(2) or "").lower()).strip()
    return _PLAN_BRAND_NAMES.get(word) or _brand(word)


#: The plan kinds a declared plan may carry — the `_LABELS_*` keys. The relay
#: seeds the card with these (`plan: [{kind, label}]`) and hands the same keys
#: to the tenant as `planned_steps` (voice round 2026-10-07, DESIGN-R3 D-R4(6)).
PLAN_KINDS: frozenset = frozenset(_LABELS_EN)
#: The most rows a body `planned_steps` may declare (answer included).
PLANNED_STEPS_MAX = 6

#: Question-shape fallback (voice round 2026-10-07, card 2): a request the
#: classifier reads as `full` («…همین توی آیسآکادمیک چجوریه») but that is a
#: QUESTION gets the search plan, not «انجام کار». Closed: a request ending in
#: ?/؟, or whose last word is one of these question tails. «make it nicer»
#: stays generic.
_QUESTION_TAIL_WORDS: frozenset = frozenset({
    "چیه", "چیست", "کیه", "کیست", "کجاست", "چنده", "چطوره", "چجوریه", "کدومه",
})
_QUESTION_TAIL_PAIRS: frozenset = frozenset({("کی", "بود")})


def _question_shaped(text: str) -> bool:
    raw = (text or "").replace("\u200c", " ").strip()
    if not raw:
        return False
    if raw.endswith(("?", "؟")):
        return True
    try:
        from app.agent.query_intent import normalize_fa
        raw = normalize_fa(raw)
    except Exception:  # noqa: BLE001 — a missing helper keeps the raw text
        pass
    words = re.sub(r"[\s،,.!:;«»\"']+", " ", raw).split()
    if not words:
        return False
    if words[-1] in _QUESTION_TAIL_WORDS:
        return True
    return len(words) >= 2 and (words[-2], words[-1]) in _QUESTION_TAIL_PAIRS


def template_plan_keys(
    display_request: Optional[str], intent: Any = None, rtl: bool = False,
    *, require_web: bool = False,
) -> List[str]:
    """The KINDS of the plan a delegated turn's card declares before any tool
    runs (D-P5) — `_LABELS_*` keys, the answer last. ONE function for both
    sides of the wire: the relay seeds the card's first frame with exactly
    these (DESIGN-R3 D-R4(6)) and `template_plan` renders them, so the seed and
    the tenant's declared plan cannot drift. ``rtl`` does not change a kind;
    it is accepted so the two signatures stay one call shape."""
    category = str(getattr(intent, "category", intent) or "")
    text = display_request or ""
    if require_web or explicit_search_requested(text):
        work = ["web", "fetch"]
    elif category == "media" or media_command_requested(text):
        # D-T1: a delegated «استاپش کن» read "Work on it · 1 skipped" because
        # the only tool it ran (play_media / media_control) is not work and
        # the generic step never started. One media step the tool starts.
        work = ["media"]
    elif category in ("web", "question"):
        work = ["web", "fetch"]
    elif _planned_brand(text):
        work = ["connector_read"]
    elif category == "code":
        work = ["file_read", "other"]
    elif category in ("full", "") and _question_shaped(text) and not owned_data_request(text):
        # Phase D (search-play-title-4): never for the caller's OWN data —
        # «ایمیلای امروزم چیه؟», «جلسه فردام کیه؟» are questions, but the card
        # opened on «جست‌وجو در وب · خواندن منابع» over a Gmail / Calendar
        # read (985e611e planned «انجام کار»).  Ruling 37 is card 2's research
        # question only.
        work = ["web", "fetch"]
    else:
        work = ["other"]
    return work + ["answer"]


def plan_step_label(
    kind: str, *, rtl: bool = False, display_request: Optional[str] = None,
) -> str:
    """The card label of one plan kind, in the request's language. A
    connector kind names the brand the request names (``{brand}``)."""
    labels = _LABELS_FA if rtl else _LABELS_EN
    label = labels.get(str(kind or ""), labels["other"])
    if "{brand}" in label:
        label = label.format(brand=_planned_brand(display_request or "") or "")
    return label


def valid_planned_steps(
    kinds: Any, display_request: Optional[str] = None,
) -> Optional[List[str]]:
    """``kinds`` as a plan this card may declare verbatim, or None. Every kind
    must be known, ``answer`` must be last and only last, there must be at
    least one work step, at most `PLANNED_STEPS_MAX` rows, and a connector kind
    needs a brand the request names — anything else falls back to
    `template_plan` (DESIGN-R3 D-R4(6): "exactly when every key is known and
    answer is last")."""
    if not isinstance(kinds, (list, tuple)):
        return None
    out = [str(k).strip() for k in kinds if isinstance(k, str)]
    if len(out) != len(kinds) or not (2 <= len(out) <= PLANNED_STEPS_MAX):
        return None
    if out[-1] != "answer" or "answer" in out[:-1]:
        return None
    if any(k not in PLAN_KINDS for k in out):
        return None
    if any(k.startswith("connector_") for k in out) and not _planned_brand(display_request or ""):
        return None
    return out


def template_plan(
    display_request: Optional[str], intent: Any = None, rtl: bool = False,
    *, require_web: bool = False,
) -> List[str]:
    """The plan a delegated turn's card declares BEFORE any tool runs (D-P5).

    Chosen from the request alone, in the request's language, from the same
    closed vocabulary the steps are attributed with — so a planned step is the
    step a tool call lands on, and a tool kind the plan did not foresee is
    inserted honestly (``TurnJob.plan``) rather than hidden. The answer step
    is always last. ``intent`` is a ``QueryIntent`` or its category string.
    The labels of `template_plan_keys` — one decision, two renderings.
    """
    return [
        plan_step_label(k, rtl=rtl, display_request=display_request)
        for k in template_plan_keys(display_request, intent, rtl, require_web=require_web)
    ]


def _close_step(step: Dict[str, Any], now: datetime) -> None:
    """A running step is done at ``now``, with the window it really ran."""
    step["status"] = "done"
    if step.get("completed_at"):
        return
    step["completed_at"] = _iso_ms(now)
    start_dt = _parse_iso_ms(step.get("started_at"))
    if start_dt is not None:
        dur = int(max(0.0, (now - start_dt).total_seconds() * 1000))
        step["duration_ms"] = dur
        step["durationMs"] = dur


#: Tools whose results name the pages the work read (`config_json.sources`).
_SOURCE_TOOLS = frozenset({
    "web_search", "web_fetch", "extension_search", "extension_read",
    "extension_research",
})
_SOURCES_MAX = 6


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


def voice_row_is_proof(raw_metadata: Any, delegation_id: Optional[str]) -> bool:
    """Is this assistant row the answer to the voice card for
    ``delegation_id``?

    A row with no ``voice`` provenance at all is the Realtime path, where
    the only assistant writer is the turn itself — unchanged behaviour, and
    the reason this is not simply "voice.source == 'delegated'".

    A ``live_spoken`` row is NEVER proof. That is the whole defect: the
    Live prompt tells the model to "say one short line and delegate", so a
    filler line — and therefore a spoken row — lands in the same
    conversation a second after the job opens, and the cancelled outcome
    became unreachable in exactly the scenario it was written for. The
    delivered-turn watchdog (`job_reconciler`) asks the same question of a
    still-running voice card, where a spoken progress line from a long
    supervised run would otherwise close it while its workers run.
    """
    voice = _voice_of(raw_metadata)
    if voice is None:
        return True
    if str(voice.get("source") or "") != "delegated":
        return False
    if voice.get("cancelled") or voice.get("superseded"):
        return False
    if not delegation_id:
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
    return (str(voice.get("delegation_id") or "")[:_DELEGATION_ID_MAX]
            == str(delegation_id)[:_DELEGATION_ID_MAX])


class TurnJob:
    """The job card for one turn the RUNNER opens — a voice delegation
    (eagerly, with a declared plan) or a chat/voice turn whose first work round
    the model did not put in a job of its own (lazily, steps from the tools).
    Every public method is synchronous, returns immediately, and never raises.
    """

    __slots__ = (
        "_active", "_anchor", "_answer_label", "_config_extra", "_conversation_id",
        "_delegation_id", "_frame_sink", "_job_id", "_lock", "_opened", "_opened_at",
        "_plan_offset", "_plan_source", "_planned_ids", "_provisional", "_row_exists",
        "_row_ready", "_rtl",
        "_sealed", "_sources", "_steps", "_title", "_tracker", "_user_id", "_voice",
    )

    def __init__(
        self,
        *,
        user_id: str,
        conversation_id: Optional[str],
        request_text: Optional[str],
        display_request: Optional[str] = None,
        delegation_id: Optional[str] = None,
        channel: Optional[str] = "voice",
        frame_sink: Optional[Any] = None,
        task_title: Optional[str] = None,
        anchor: Optional[Dict[str, Any]] = None,
    ) -> None:
        """``display_request`` is what the caller SAID, when the caller knows
        it separately from what the model was given. On the Live path they are
        different strings — `request_text` is a constructed prompt — and only
        the first of them may reach a screen.

        ``delegation_id`` names WHICH piece of work this card is, so the
        cancelled close can tell this turn's answer from a different one that
        landed in the same conversation at the same time — two concurrent
        delegations are an ordinary Live state, not an edge case.

        ``frame_sink`` is the delegation stream's synchronous ``put`` (api_v1's
        SSE route): every state change is mirrored onto it as a ``job.update``
        frame so the call draws the card from t0 instead of waiting for a
        chat socket the phone may not have open.
        """
        from app.agent.job_titles import derive_job_title, is_rtl_text

        self._user_id = user_id
        self._conversation_id = conversation_id
        self._delegation_id = (
            str(delegation_id)[:_DELEGATION_ID_MAX] if delegation_id else None
        )
        _display = clean_request_text(display_request) or clean_request_text(request_text)
        self._rtl = is_rtl_text(_display if _display else request_text)
        # Voice round 2026-10-06 (D-T4): the generic no longer reads like a
        # step («جمع‌بندی درخواست صوتی» sat on the card beside the answer
        # step «جمع‌بندی پاسخ» and looked like one).
        _fallback = "درخواست صوتی" if self._rtl else "Voice request"
        # Precedence (D-T4): the relay's own title for the task (its summary,
        # or the request words it carries — the «ها» card was titled with the
        # filler that triggered it), then the caller's words, then the
        # generic. A one-word title is a caption fragment at every rung:
        # `derive_job_title` normalises a REQUEST and has no notion of "this is
        # only part of one", so the floor lives here, where the input is known
        # to be a transcript. Which floor depends on the SCRIPT: a space-free
        # one measures the fragment in characters, because there every request
        # is one `split()` token.
        _given = derive_job_title(clean_request_text(task_title), fallback="")
        if _title_is_fragment(_given):
            _given = ""
        _derived = derive_job_title(_display, fallback="")
        if _title_is_fragment(_derived):
            _derived = ""
        self._title = _given or _derived or derive_job_title(None, fallback=_fallback)
        # D-R8: the caller turn that asked, for the thread marker and the row.
        self._anchor: Dict[str, Any] = {
            k: anchor[k] for k in (
                "parent_user_turn_id", "request_turn_ids", "anchor_occurred_at",
            ) if isinstance(anchor, dict) and anchor.get(k)
        }
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
        # Set once `_open` has an answer (row committed, or the insert
        # failed): the turn may await it, and no `running` frame goes out
        # before it (W3).
        self._row_ready = asyncio.Event()
        self._sealed = False
        self._lock = asyncio.Lock()
        self._voice = (channel or "").strip().lower() == "voice"
        self._frame_sink = frame_sink
        self._plan_source: str = "tools"
        self._planned_ids: set = set()
        self._sources: List[Dict[str, Any]] = []
        self._config_extra: Dict[str, Any] = {}
        # The turn's StepTracker, kept from the last `bind` so a plan adopted
        # from OUTSIDE the turn's own calls (`follow_steps`) re-points it.
        self._tracker: Any = None
        # Rows of steps that ran before an adopted supervisor plan; the
        # plan's step i is card row `_plan_offset + i`.
        self._plan_offset: int = 0
        # D-T3: the card state before a PROVISIONAL begin_answer, or None.
        self._provisional: Optional[Tuple[List[Dict[str, Any]], int]] = None

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

    @property
    def rtl(self) -> bool:
        return self._rtl

    @property
    def conversation_id(self) -> Optional[str]:
        return self._conversation_id

    @property
    def plan_offset(self) -> int:
        """Card row of the followed supervisor plan's step 0."""
        return self._plan_offset if self._plan_source in ("supervisor", "model") else 0

    def step_labels(self) -> List[str]:
        return [str(s.get("label") or "") for s in self._steps]

    def progress(self) -> Tuple[int, int, int]:
        """(done, total, active row) of the card as it stands."""
        from app.agent.job_steps import counts
        done, total = counts(self._steps)
        return done, total, self._active

    # ── declared up front (a voice delegation) ────────────────────────
    def declare(self, labels: Iterable[str], tracker: Any = None) -> None:
        """Open the card NOW with a declared plan — before phase 1, before the
        first model round. Nothing is running yet: the plan is what the turn
        intends, and a step only starts when a tool lands on it, so a planned
        step the turn never reaches closes ``skipped``, never done."""
        try:
            if self._opened:
                return
            for label in labels or ():
                label = str(label or "").strip()
                if not label or label == self._answer_label:
                    continue
                if self._ensure_step(label):
                    self._planned_ids.add(self._steps[-2]["id"])
            if not self._steps:
                return
            self._plan_source = "template"
            self._opened = True
            self.bind(tracker)
            # The first `running` frame is `_open`'s, after the row commits.
            self._spawn(self._open())
        except Exception:  # noqa: BLE001 — a turn must never fail on a card
            logger.exception("[turn-job] declare failed")

    def adopt_plan(self, labels: Iterable[str], tracker: Any = None,
                   *, source: str = "supervisor") -> None:
        """Replace the steps that have not started with ``labels`` — the last
        of them becomes the answer step. For a producer that plans the work
        itself (the supervisor's ``delegate``, or the model's own
        ``create_job`` on a turn whose card is already open): the steps
        already run stay, in order, and the card keeps one writer instead of
        two racing over ``steps_json``. The plan's step i is card row
        ``plan_offset + i``."""
        try:
            labels = [str(x).strip() for x in labels or () if str(x or "").strip()]
            if len(labels) < 2 or self._sealed:
                return
            kept = [st for st in self._steps
                    if st.get("type") != "step_answer"
                    and st.get("status") not in (None, "pending")]
            fresh = [{"id": str(uuid.uuid4()), "label": lb, "status": "pending"}
                     for lb in labels]
            self._planned_ids = {st["id"] for st in fresh[:-1]}
            self._steps = kept + fresh
            self._plan_offset = len(kept)
            for i, st in enumerate(self._steps):
                st["type"] = "step_answer" if i == len(self._steps) - 1 else f"step_{i}"
            self._answer_label = labels[-1]
            self._plan_source = source
            self._active = max(0, len(kept) - 1)
            if not self._opened:
                self._opened = True
                self.bind(tracker)
                self._spawn(self._open())
            else:
                self.bind(tracker)
                self._emit("running")
                self._spawn(self._advance())
        except Exception:  # noqa: BLE001
            logger.exception("[turn-job] adopt_plan failed")

    def retitle(self, task_title: Optional[str]) -> bool:
        """The relay's late title for this card (its six-word summary, landed
        after dispatch) — voice round 2026-10-07, T11. Same floors as the
        constructor: a fragment never replaces a title. The next frame carries
        the new `name`, `_advance` writes `BuildJob.title`, and the thread
        marker is written from it at close. True when the title changed."""
        try:
            if self._sealed:
                return False
            from app.agent.job_titles import derive_job_title
            title = derive_job_title(clean_request_text(task_title), fallback="")
            if not title or _title_is_fragment(title) or title == self._title:
                return False
            self._title = title
            if self._opened:
                self._emit("running")
                self._spawn(self._advance())
            return True
        except Exception:  # noqa: BLE001 — a title never costs a turn
            logger.debug("[turn-job] retitle failed", exc_info=True)
            return False

    def start_step(self, index: int, tracker: Any = None) -> None:
        """Start step ``index`` (a step that already ran is left alone)."""
        try:
            if self._opened and not self._sealed and self._start(int(index)):
                self.bind(tracker if tracker is not None else self._tracker)
                self._emit("running")
                self._spawn(self._advance())
        except Exception:  # noqa: BLE001
            logger.debug("[turn-job] start_step failed", exc_info=True)

    def note_config(self, **values: Any) -> None:
        """Merge keys into the row's ``config_json`` (e.g. ``role``) through
        this card's own writer, so a later card write cannot erase them."""
        try:
            self._config_extra.update(values)
            if self._opened:
                self._spawn(self._advance())
        except Exception:  # noqa: BLE001
            logger.debug("[turn-job] note_config failed", exc_info=True)

    def rebind_conversation(self, conversation_id: Optional[str]) -> None:
        """The session phase 1 resolved, when it differs from the one the
        card was declared with (a conversation the tenant had not seen)."""
        try:
            if not conversation_id or conversation_id == self._conversation_id:
                return
            self._conversation_id = conversation_id
            if self._opened:
                self._spawn(self._advance())
        except Exception:  # noqa: BLE001
            logger.debug("[turn-job] rebind failed", exc_info=True)

    def follow_steps(self, steps: Iterable[Dict[str, Any]]) -> None:
        """Adopt an external plan WITH its live statuses — the supervisor's
        (`delegate`, lane PF), whose workers run concurrently, which
        `adopt_plan`/`start_step` (one running step at a time) cannot say.
        No IO and no frame: the supervisor writes the row and announces it;
        this keeps the card's own later writers (progress, the terminal frame,
        the close) carrying the SAME plan instead of rewriting the template
        over it. The last row is the answer step.

        A FRESH plan (none of its rows is on the card yet) keeps the steps
        that already ran ahead of it, as `adopt_plan` does — a search the
        model made before delegating really ran, and its actions are filed
        under it. A plan that already carries the card's rows (the
        supervisor's own progress writes, built from `followed_steps`)
        replaces them. Either way the turn's tracker is re-pointed, so the
        delegate's own tool frames and the closing step event name rows of
        THIS card rather than the template's."""
        try:
            rows = [dict(st) for st in steps or () if isinstance(st, dict)]
            if len(rows) < 2 or not self._opened or self._sealed:
                return
            incoming = {st.get("id") for st in rows if st.get("id")}
            if not incoming & {st.get("id") for st in self._steps}:
                kept = [dict(st) for st in self._steps
                        if st.get("type") != "step_answer"
                        and st.get("status") not in (None, "pending")]
                # The step that was running ended when the work was handed
                # to the helpers.
                now = datetime.utcnow()
                for st in kept:
                    if st.get("status") == "running":
                        _close_step(st, now)
                self._plan_offset = len(kept)
                rows = kept + rows
            for i, st in enumerate(rows):
                st["type"] = "step_answer" if i == len(rows) - 1 else f"step_{i}"
            self._steps = rows
            self._planned_ids = {st.get("id") for st in rows[:-1]}
            self._answer_label = str(rows[-1].get("label") or self._answer_label)
            self._plan_source = "supervisor"
            running = [i for i, st in enumerate(rows) if st.get("status") == "running"]
            self._active = running[-1] if running else min(self._plan_offset, len(rows) - 1)
            if self._tracker is not None:
                self._sync_tracker(self._tracker, active=self._active)
        except Exception:  # noqa: BLE001
            logger.debug("[turn-job] follow_steps failed", exc_info=True)

    async def open_following(self, steps: Iterable[Dict[str, Any]], **config: Any) -> Optional[str]:
        """Open a card that has not opened yet DIRECTLY on an external plan —
        a chat turn whose first round is `delegate` (TW-5). The supervisor's
        card is then this turn's card, so the runner's lazy open finds it
        open and a later work tool lands on it instead of minting a second
        one. Awaits the row: the plan's workers are inserted under it as
        their parent. Returns the job id, or None when the insert failed."""
        try:
            rows = [dict(st) for st in steps or () if isinstance(st, dict)]
            if len(rows) < 2 or self._opened or self._sealed:
                return None
            for i, st in enumerate(rows):
                st["type"] = "step_answer" if i == len(rows) - 1 else f"step_{i}"
            self._steps = rows
            self._planned_ids = {st.get("id") for st in rows[:-1]}
            self._answer_label = str(rows[-1].get("label") or self._answer_label)
            self._plan_source = "supervisor"
            self._plan_offset = 0
            self._active = 0
            self._config_extra.update(config)
            self._opened = True
            self.bind(self._tracker)
            await self._open()
            return self._job_id if self._row_exists else None
        except Exception:  # noqa: BLE001
            logger.exception("[turn-job] open_following failed")
            return None

    async def wait_row(self, timeout: float) -> bool:
        """Wait (bounded) until the card's row is committed. True when it is."""
        try:
            await asyncio.wait_for(self._row_ready.wait(), timeout)
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001
            pass
        return self._row_exists

    def followed_steps(self) -> Optional[List[Dict[str, Any]]]:
        """The supervisor-followed plan (a copy), else None."""
        return self._snapshot() if self._plan_source == "supervisor" and self._steps else None

    def begin_answer(
        self, started_at: Optional[datetime] = None, *, provisional: bool = False,
    ) -> None:
        """The model's tool-less closing round began at ``started_at``: the
        answer step is running from then, and the step before it closed then.

        ``provisional`` (D-T3, the round-START beat): the round has written
        text but may still call a tool — the card state before the move is
        kept so `retract_answer` can put it back. A later non-provisional call
        (the post-round confirmation) settles it."""
        try:
            if not self._opened or self._sealed or not self._steps:
                return
            if not provisional:
                self._provisional = None
            last = len(self._steps) - 1
            before = ([dict(st) for st in self._steps], self._active)
            if self._move_to(last, started_at or datetime.utcnow()):
                if provisional:
                    self._provisional = before
                self._emit("running")
                self._spawn(self._advance())
        except Exception:  # noqa: BLE001
            logger.debug("[turn-job] begin_answer failed", exc_info=True)

    def retract_answer(self) -> None:
        """Undo a PROVISIONAL `begin_answer`: the round's text turned out to be
        a preamble before a tool call, so the card returns to the step it was
        on (its window reopened, nothing greened). No-op otherwise."""
        try:
            saved = self._provisional
            self._provisional = None
            if saved is None or not self._opened or self._sealed:
                return
            restored, active = saved
            # The walk-back is a BACKWARD move (the prior step done→running,
            # the answer running→pending) and both clients merge steps by
            # status rank — they keep the higher status unless the row's
            # `rev` changed. Without the bump the retraction was discarded
            # and the card sat on "writing the answer" through the search
            # (Phase D, contract-wire-1).
            for i, row in enumerate(restored):
                cur = self._steps[i] if i < len(self._steps) else {}
                if (row.get("status"), row.get("completed_at")) != (
                    cur.get("status"), cur.get("completed_at")
                ):
                    row["rev"] = max(int(row.get("rev") or 0), int(cur.get("rev") or 0)) + 1
            self._steps, self._active = restored, active
            if self._tracker is not None:
                self._sync_tracker(self._tracker, active=self._active)
            self._emit("running")
            self._spawn(self._advance())
        except Exception:  # noqa: BLE001
            logger.debug("[turn-job] retract_answer failed", exc_info=True)

    def note_sources(self, tool_name: Optional[str], tool_input: Any, result: Any) -> None:
        """Keep the pages the work read — deduped by domain, at most six — for
        ``config_json.sources`` and every frame from here on."""
        try:
            if not self._opened or str(tool_name or "") not in _SOURCE_TOOLS:
                return
            from app.api.api_v1 import _vs_sources
            found = _vs_sources(
                str(tool_name), tool_input if isinstance(tool_input, dict) else {},
                result if isinstance(result, str) else str(result or ""),
            )
            seen = {s.get("domain") for s in self._sources}
            for src in found or ():
                dom = (src or {}).get("domain")
                if dom and dom in seen and src.get("icon"):
                    # D-T4: a domain first met in a SEARCH gains the icon its
                    # page carried once that page was FETCHED.
                    for kept in self._sources:
                        if kept.get("domain") == dom and not kept.get("icon"):
                            kept["icon"] = src["icon"]
                    continue
                if not dom or dom in seen or len(self._sources) >= _SOURCES_MAX:
                    continue
                seen.add(dom)
                self._sources.append(dict(src))
                if len(self._sources) >= _SOURCES_MAX:
                    break
        except Exception:  # noqa: BLE001
            logger.debug("[turn-job] note_sources failed", exc_info=True)

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

        With a DECLARED plan the round is attributed to the planned step its
        tool kind names; a kind the plan did not foresee takes over a never
        started generic "Work on it" step, else is inserted before the answer
        step — honest growth, never a hidden tool.
        """
        try:
            # Kept even for a round with no work tool: a card opened later
            # from outside the turn's own calls (`open_following`) binds it.
            if tracker is not None:
                self._tracker = tracker
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
            moved = self._start(self._index_of(first))
            self.bind(tracker)
            if not self._opened:
                self._opened = True
                self.bind(tracker)
                self._spawn(self._open())
            elif fresh or moved:
                self._emit("running")
                self._spawn(self._advance())
        except Exception:  # noqa: BLE001 — a turn must never fail on a card
            logger.exception("[voice-job] plan failed")

    def _index_of(self, label: str) -> int:
        for i, s in enumerate(self._steps):
            if s.get("label") == label:
                return i
        return self._active

    def _start(self, idx: int) -> bool:
        """Start step ``idx`` if it has never run. A revisit of a step that
        already ran leaves the card alone — the tracker still files the action
        under it — so the done count only ever grows."""
        if not (0 <= idx < len(self._steps)):
            return False
        if self._steps[idx].get("status") not in (None, "pending"):
            return False
        return self._move_to(idx, datetime.utcnow())

    def _move_to(self, idx: int, now: datetime) -> bool:
        """Step ``idx`` is the running step from ``now``; the step that was
        running closes then, with the window it really ran. Applied to the
        live list on the turn's coroutine (no IO) so a fast walk through three
        steps cannot be coalesced into one by a background writer."""
        changed = False
        for i, st in enumerate(self._steps):
            if i != idx and st.get("status") == "running":
                _close_step(st, now)
                changed = True
        target = self._steps[idx]
        if target.get("status") in (None, "pending"):
            target["status"] = "running"
            target.setdefault("started_at", _iso_ms(now))
            changed = True
        self._active = idx
        return changed

    def attribute(self, tool_name: Optional[str], tracker: Any) -> None:
        """Point ``tracker`` at the step this tool call belongs to, so the
        tool frames and the persisted ``tool_events`` carry the same
        ``job_id`` / ``step_index`` / ``step_name`` a chat turn's do."""
        try:
            if not self._opened or tracker is None:
                return
            label = step_label_for_tool(tool_name, rtl=self._rtl)
            idx = next(
                (i for i, s in enumerate(self._steps) if s.get("label") == label),
                None,
            )
            # A non-work tool files under a step only when it is a MEDIA tool
            # and the PLAN declared the media step (D-T1). Every other
            # non-work tool still starts nothing — a memory lookup must not
            # green a planned "Work on it".
            if not is_work_tool(tool_name) and (
                _TOOL_KEYS.get(str(tool_name or "")) != "media"
                or idx is None
                or self._steps[idx].get("id") not in self._planned_ids
            ):
                return
            if idx is None:
                return
            # The TRACKER gets the true index — this action belongs to this
            # step whether or not the turn has been here before.
            self._sync_tracker(tracker, active=idx)
            # The CARD only starts steps that never ran. A turn that goes
            # search → fetch → search revisits step 0, and a bar that walked
            # backwards would read as the job losing work it had already done.
            if self._start(idx):
                self._emit("running")
                self._spawn(self._advance())
        except Exception:  # noqa: BLE001
            logger.debug("[voice-job] attribute failed", exc_info=True)

    def bind(self, tracker: Any) -> None:
        """Adopt this job into the turn's :class:`StepTracker`."""
        try:
            if tracker is not None:
                self._tracker = tracker
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
            # The delegation stream is still open on the happy path (`done`
            # follows this call), so its terminal frame goes out now, with the
            # same step rules the background close writes to the row.
            from app.agent.job_steps import finish_all_steps, skip_unstarted_steps
            now = datetime.utcnow()
            if reason == SEAL_CANCELLED:
                self._emit("cancelled", _stop_open_steps(self._snapshot(), now))
            else:
                self._emit("completed", finish_all_steps(
                    skip_unstarted_steps(self._snapshot()), now,
                    keep_failed=self._config_extra.get("role") == "supervisor"))
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
        # A declared generic step ("Work on it") that never started is a
        # placeholder for exactly this: the first concrete kind takes it over
        # instead of standing beside it as a step that will only be skipped.
        generic = (_LABELS_FA if self._rtl else _LABELS_EN)["other"]
        for st in self._steps:
            if (st.get("id") in self._planned_ids and st.get("label") == generic
                    and st.get("status") in (None, "pending")):
                st["label"] = label
                return True
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

    def _config(self) -> Dict[str, Any]:
        """``config_json`` as this card writes it. ``turn_job`` is the marker
        every closer reads (`job_reconciler.close_job_completed`) to skip a
        planned step that never started instead of greening it."""
        return {
            "job_type": self._job_type(), "turn_job": True,
            "plan_source": self._plan_source,
            **({"voice_turn": True} if self._voice else {}),
            **({"voice_delegation_id": self._delegation_id}
               if self._delegation_id else {}),
            **({"sources": list(self._sources)} if self._sources else {}),
            # D-R8: the asking turn — `message_cards.job_card_fields` projects
            # it, and `ws_chat._persist_job_card` stamps its marker from it.
            **self._anchor,
            **self._config_extra,
        }

    def _frame(self, status: str, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """The ``job_update`` body every surface gets — the chat socket and
        the delegation stream alike. Additive over the shape ``create_job``
        emits: ``steps`` (the whole plan with live statuses) and ``sources``,
        so no client waits for a REST read to draw the card."""
        from app.agent.job_steps import counts, public_steps
        done, total = counts(steps)
        running = next((str(s.get("label") or "") for s in steps
                        if s.get("status") == "running"), "")
        if status == "completed":
            done = total
        if status == "cancelled":
            step = _STOPPED_STEP_FA if self._rtl else _STOPPED_STEP_EN
        elif status == "completed":
            step = str(steps[-1].get("label") or "") if steps else "Done"
        else:
            step = running or "Working..."
        out: Dict[str, Any] = {
            "job_id": self._job_id,
            "job_type": self._job_type(),
            "name": self._title,
            "status": status,
            "step": step,
            "total_steps": total,
            "completed_steps": done,
            "chat_id": self._conversation_id,
            # No `message_id`: on voice the answer row is written by the
            # platform relay under an id it mints itself.
            "message_id": None,
            "steps": public_steps(steps),
        }
        if self._sources:
            out["sources"] = list(self._sources)
        if self._anchor.get("parent_user_turn_id"):
            # D-R8 (additive): the live card can fold under its bubble too.
            out["parent_user_turn_id"] = self._anchor["parent_user_turn_id"]
        return out

    def _emit(self, status: str, steps: Optional[List[Dict[str, Any]]] = None) -> None:
        """Mirror the card onto the delegation stream as ``job.update``.
        Synchronous and never raises: the sink is a non-blocking queue put."""
        if self._frame_sink is None:
            return
        if status == "running" and not self._row_exists:
            # W3: the phone reaps a job its next poll does not list, so a
            # card announced before its row commits can be deleted and
            # redrawn. `_open` sends the first frame, with every step change
            # made meanwhile, the moment the row exists.
            return
        try:
            frame = self._frame(status, steps if steps is not None else self._snapshot())
            if self._delegation_id:
                frame["delegation_id"] = self._delegation_id
            self._frame_sink({"type": "job.update", **frame})
        except Exception:  # noqa: BLE001
            logger.debug("[turn-job] job.update sink failed", exc_info=True)

    # -- background writes (serialised behind one lock) ----------------
    async def _open(self) -> None:
        from app.agent.job_runner import JobRunner, TaskSpec
        from app.agent.job_steps import dump_steps
        from app.config import settings

        async with self._lock:
            # The step windows are stamped on the live list by the turn itself
            # (`_move_to`); this only persists them.
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
                        config_json=self._config(),
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
                self._row_ready.set()
                return
            self._row_exists = True
            self._row_ready.set()
            if not self._sealed:
                self._emit("running")
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
        from app.api.message_cards import (
            job_marker_content, job_marker_row_fields, marker_day_instant,
        )
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
                # D-R8 (d): stamped at the asking turn, linked to it —
                # through the one helper both marker writers share — and
                # filed under THAT instant's day (Phase D, tenant-8), not the
                # day the card closed.
                marker_fields = job_marker_row_fields(
                    delegation_id=self._delegation_id, **self._anchor,
                )
                db.add(_Msg(
                    id=row_id,
                    conversation_id=self._conversation_id,
                    day_chat_id=await resolve_day_chat_id_for_now(
                        db, self._user_id,
                        utc_now=marker_day_instant(marker_fields.get("occurred_at")),
                    ),
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
                    **marker_fields,
                ))
                await db.commit()
        except Exception:  # noqa: BLE001 — the card is not worth a raise
            logger.debug("[voice-job] chat card row failed", exc_info=True)

    async def _advance(self) -> None:
        """Re-write the step list and push progress. Cheap and idempotent —
        the card's steps grow as the turn discovers what it needs."""
        from app.agent.job_steps import counts, dump_steps
        from app.db.database import async_session_maker
        from app.db.models import BuildJob, JobEvent

        async with self._lock:
            if not self._row_exists:
                return
            steps = self._snapshot()
            done, total = counts(steps)
            try:
                async with async_session_maker() as db:
                    job = await db.get(BuildJob, self._job_id)
                    if job is None or job.status != "running":
                        return
                    job.steps_json = dump_steps(steps)
                    if job.title != self._title:
                        job.title = self._title     # `retitle` (T11)
                    if self._conversation_id and job.conversation_id != self._conversation_id:
                        job.conversation_id = self._conversation_id
                    if self._sources or self._config_extra:
                        job.config_json = {
                            **(job.config_json or {}), **self._config_extra,
                            **({"sources": list(self._sources)} if self._sources else {}),
                        }
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
        from app.agent.job_steps import counts
        done, total = counts(steps)
        label = next((str(s.get("label") or "") for s in steps
                      if s.get("status") == "running"), "")
        job_type = self._job_type()
        try:
            from app.api.ws_chat import broadcast_to_user
            await broadcast_to_user(self._user_id, {
                "type": "job_update", **self._frame("running", steps),
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

    def _row_is_proof(self, raw_metadata: Any) -> bool:
        """Is THIS assistant row the answer to THIS turn? See
        :func:`voice_row_is_proof` — one rule, shared with the reconciler."""
        return voice_row_is_proof(raw_metadata, self._delegation_id)

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

        if reason == SEAL_CANCELLED and self._voice:
            # A cancel reaches here because `AgentRunner.run`'s finally ran
            # without `finalize` — so THIS turn produced no answer. It does not
            # follow that the user got nothing: the relay may already have
            # persisted the answer of a delegation that was detached rather
            # than killed. Ask — repeatedly, up to a ceiling — and upgrade to
            # the honest completed word when the thread says so.
            #
            # VOICE ONLY. On chat the turn is the only writer of its answer,
            # so `finalize` is the only proof — and ws_chat writes an
            # assistant row with no voice provenance on every Stop that
            # streamed text ("…*[Generation stopped by user]*") and on every
            # error, which `_row_is_proof` would read as the answer and close
            # a stopped turn's card completed, every step green.
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
                    # The turn's own step windows, not the last background
                    # write's: `begin_answer` may still be queued behind us.
                    from app.agent.job_steps import dump_steps
                    from app.db.models import BuildJob
                    _row = await db.get(BuildJob, self._job_id)
                    if _row is not None and _row.status == "running":
                        _row.steps_json = dump_steps(self._snapshot())
                    closed = await close_job_completed(
                        db, self._job_id, user_id=self._user_id, now=now,
                        # No message_id: the answer row is the relay's, under
                        # an id this process never sees. The reconciler's
                        # conversation-scoped proof covers the back-link.
                        message_id=None,
                        total_tokens=total_tokens, model=model,
                        reason="turn_end",
                        config_updates={
                            **self._config_extra,
                            **({"sources": list(self._sources)} if self._sources else {}),
                        } or None,
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
                    # Merged, not replaced: another writer (a supervisor's
                    # `role`, say) may have added keys since this card opened.
                    _row = await db.get(BuildJob, self._job_id)
                    _cfg = {**((_row.config_json if _row is not None else None) or {}),
                            **self._config_extra,
                            **({"sources": list(self._sources)} if self._sources else {})}
                    res = await db.execute(
                        _upd(BuildJob)
                        .where(BuildJob.id == self._job_id,
                               BuildJob.user_id == self._user_id,
                               BuildJob.status == "running")
                        .values(
                            status=STATUS_CANCELLED,
                            completed_at=now,
                            config_json=_cfg or self._config(),
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
        await self._announce_cancelled(steps)

    async def _announce_cancelled(self, steps: List[Dict[str, Any]]) -> None:
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
                "type": "job_update", **self._frame("cancelled", steps),
            })
        except Exception:  # noqa: BLE001
            logger.debug("[voice-job] cancel broadcast failed", exc_info=True)


#: The Round-13 name. Every call site and test that imported the voice-only
#: class keeps working; the class is channel-agnostic now.
VoiceTurnJob = TurnJob
