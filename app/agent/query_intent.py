"""
Query Intent Classifier — lightweight, zero-LLM classification for tool filtering.

Classifies user messages into intent categories to determine which tools
and system prompt sections to include in the LLM call. This directly reduces
input token count and TTFT (time to first token).

Categories:
  - "greeting"  → No tools needed (hi, hello, thanks, bye)
  - "question"  → No tools needed (general knowledge, opinion, explanation)
  - "memory"    → Only memory tools (remember, recall, what did I say)
  - "media"     → Media tools (play, music, video, send photo)
  - "web"       → Web tools (search, look up, find online, URL)
  - "code"      → Code/file tools + skill tools (build, deploy, fix, file)
  - "agent"     → Agent management tools (spawn, process, session)
  - "full"      → All tools (unclear, complex, multi-intent)

Two orthogonal merges ride on top of the category:
  - every work intent carries TOOLS_WORK_TRACKING (create_job/update_job/
    start_mission) because the decision rules mandate them first;
  - TOOLS_DOCGEN (generate_pdf & co.) is added only when
    `has_document_intent` sees an explicit format/artifact word.

Performance target: <1ms. No regex compilation at call time (all pre-compiled).
"""

import re
import logging
from dataclasses import dataclass, field, replace
from typing import FrozenSet, Set

# Shared explicit-remember predicate (D-mem-C) — single definition, also used
# by the extraction gate in agent_runner. query_classifier is a leaf module
# (stdlib-only imports), so this cannot cycle.
from app.services.query_classifier import is_explicit_remember_request

logger = logging.getLogger(__name__)


def _remember_boost_enabled() -> bool:
    """settings.memory_tools_on_remember, resolved lazily so this module keeps
    importing without app.config on the hot path (and tests can monkeypatch
    the flag)."""
    try:
        from app.config import settings
        return bool(getattr(settings, "memory_tools_on_remember", True))
    except Exception:
        return True

# ---------------------------------------------------------------------------
# Tool name sets by category
# ---------------------------------------------------------------------------

# Baseline recall tools — exposed in EVERY non-greeting intent. Users reference
# past days across every category ("build an app based on yesterday's idea"
# is code-intent, "what did we decide about Stripe yesterday" could route
# anywhere). recall_day must be available on turn 1, not only after tool
# escalation, so it lives here and is merged into each intent's tool set below.
TOOLS_RECALL: FrozenSet[str] = frozenset({
    "recall_day",
})

# Document generation — PDF/DOCX/XLSX/PPTX/Markdown exports. NOT merged
# into the base work intents any more: it rides on `has_document_intent`
# (below), which fires on an explicit format/artifact word ("PDF",
# "spreadsheet", "slide deck", "report", "export"…) whatever the subject
# vocabulary classifies to. History, both directions:
#
#   * 2026-07-28 (canary 533354ce, #371): "Make me a one-page PDF
#     summarizing the water cycle" scored code (make…page), generate_pdf
#     was filtered out, and the model wrote the PDF into the workspace
#     ROOT with write_file — invisible to the document pane. The fix
#     then was to merge TOOLS_DOCGEN into EVERY work intent.
#   * 2026-08-18 (founder, iOS): "Can you search for me the latest ai
#     news and give me the 5 most important one in this mounth" — a pure
#     search-and-summarize ask — opened with
#     `generate_pdf({"filename": "x", "content": []})` and shipped a
#     952-byte empty x.pdf. Root cause is the create_job gap fixed by
#     TOOLS_WORK_TRACKING below, but with the export tools in every
#     work intent they were the tools the constrained decode snapped
#     to, and there is no reason a research ask should be OFFERED an
#     exporter on turn 1 at all.
#
# `has_document_intent` keeps the first incident fixed (every phrasing
# it pinned still exposes the tools — see test_docgen_path_normalize)
# while a search/summarize/question turn no longer carries seven export
# tools it was never going to use legitimately. Turn 1 is the only turn
# this gates: after any tool call both layouts open the full set, so a
# "research X, then write it up" plan still reaches generate_pdf on the
# iteration that needs it. The runtime gate (settings.feature_doc_generation
# + the tenant entitlement) still applies at tool registration, so
# visibility here bypasses nothing.
TOOLS_DOCGEN: FrozenSet[str] = frozenset({
    "generate_pdf", "generate_docx", "generate_xlsx", "generate_pptx",
    "generate_markdown", "generate_html_to_pdf", "convert_document",
})

# Work tracking — the progress-card tools the platform-knowledge decision
# rules MANDATE for any do-something ask: "research / build / fix /
# produce … → call `create_job` FIRST, then `update_job`", and "'while I'm
# away' → call `start_mission`". Until 2026-08-18 no intent set carried
# them, so on every non-full intent the prompt told the model to open
# with a tool the gate forbade. Under the legacy layout that was merely
# unhelpful (the tool was absent from the array; the model went on to
# web_search). Under the prefix-stable layout the gate is an OpenAI
# `tool_choice: allowed_tools` restriction over the FULL wire array — the
# model can SEE create_job and the prompt says call it first, so the
# constrained decode snapped to the next allowed name in wire order
# (create_job, update_job, save_streaming_credential, generate_pdf, …)
# and filled the required args with placeholders. Measured on the
# founder's tenant, 2026-08-17/18: two web-intent turns, both opened
# with `generate_pdf({"filename": "x", "content": []})`, then create_job
# on iteration 2 exactly as the prompt asked. Prompts are advisory;
# tool-list omission is hard (prompt_profile.py) — the allowed set has
# to include everything the decision rules name, or the model invents
# a call to satisfy the grammar. Voice/trigger/subagent/autopilot still
# lose these through their disabled sets, which are applied to the
# array before intent gating and subtracted from the allowed names.
TOOLS_WORK_TRACKING: FrozenSet[str] = frozenset({
    "create_job", "update_job", "start_mission",
})

TOOLS_MEMORY: FrozenSet[str] = frozenset({
    "memory_search", "memory_read_file", "memory_store", "memory_delete",
}) | TOOLS_RECALL | TOOLS_WORK_TRACKING

TOOLS_WEB: FrozenSet[str] = frozenset({
    "web_search", "web_fetch", "browser",
}) | TOOLS_RECALL | TOOLS_WORK_TRACKING

TOOLS_MEDIA: FrozenSet[str] = frozenset({
    "send_file", "send_photo", "analyze_image", "generate_image", "edit_image",
    "tts", "play_media", "tts_prefs", "canvas",
}) | TOOLS_RECALL | TOOLS_WORK_TRACKING

TOOLS_CODE: FrozenSet[str] = frozenset({
    "exec", "pty_exec", "read_file", "write_file", "edit_file",
    "grep", "find", "ls", "apply_patch",
    # Image generation rides along with code intent because the document
    # nouns added to _CODE_PATTERNS_RE (presentation|slides|deck|report|
    # invoice|document) pull compound doc+image asks — "make me a
    # presentation with images", "make a slide deck with photos from my
    # trip" — from media into code (ties break code-before-media, and a
    # format keyword like "pdf" can outscore media outright). On main
    # those asks classified media and exposed generate_image/edit_image;
    # without this merge the code intent would hide both on turn 1 — the
    # exact tool-hiding failure class of #310/routines__remind/A6-3 and
    # of this file's has_document_intent gate. Runtime gates still apply
    # at tool registration, so visibility here bypasses nothing.
    "generate_image", "edit_image",
}) | TOOLS_RECALL | TOOLS_WORK_TRACKING

TOOLS_AGENT: FrozenSet[str] = frozenset({
    "spawn", "process", "sessions_list", "sessions_history",
    "sessions_send", "session_status", "agents_list", "lanes_status",
    "thread",
}) | TOOLS_RECALL | TOOLS_WORK_TRACKING

TOOLS_ADMIN: FrozenSet[str] = frozenset({
    "cron", "config_reload", "doctor", "moderate", "poll",
    "message", "skill_marketplace", "talk_mode",
})

# Scheduling — reminders + recurring routines (the routines skill).
# Founder bug 2026-07-16: "Remind me two minutes later" and "Every
# 11:10 pm send me a joke" scored ZERO in every category (no keyword
# list contained remind/every/daily/schedule), fell to 'question', and
# the routines__ tools were filtered out — the agent truthfully told
# the user "the reminder tool isn't available to me in this turn".
TOOLS_SCHEDULING: FrozenSet[str] = frozenset({
    "routines__create", "routines__remind", "routines__list",
    "routines__update", "routines__delete", "routines__run_now",
    # A "while I'm away, keep me updated" ask can read as scheduling —
    # keep the mission hand-off reachable from this intent (also in
    # TOOLS_WORK_TRACKING; listed here so the intent reads complete).
    "start_mission",
}) | TOOLS_RECALL | TOOLS_WORK_TRACKING

# Composite sets for multi-intent categories
TOOLS_CODE_FULL: FrozenSet[str] = TOOLS_CODE | TOOLS_WEB | TOOLS_MEMORY | TOOLS_ADMIN
TOOLS_WEB_FULL: FrozenSet[str] = TOOLS_WEB | TOOLS_MEMORY


# ---------------------------------------------------------------------------
# Intent result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QueryIntent:
    """Result of query intent classification."""
    category: str                                    # One of the categories above
    tool_names: FrozenSet[str] = field(default_factory=frozenset)  # Tool names to include (empty = all)
    include_skills: bool = False                     # Whether to include skill tool definitions
    include_skill_prompts: bool = False              # Whether to include skill system prompt sections
    include_environment: bool = False                # Whether to include environment section
    include_media_section: bool = False              # Whether to include media section
    # NOTE: skip_memory_retrieval was removed — memories are always retrieved now.
    # 200ms cost is acceptable for context quality. See Day-as-Chat architecture.


# ---------------------------------------------------------------------------
# Pre-compiled patterns (compiled once at import time)
# ---------------------------------------------------------------------------

# Greetings — multilingual, emoji-only, filler messages
_GREETING_EXACT: Set[str] = {
    # English
    "hi", "hey", "hello", "yo", "sup", "heya", "hiya", "howdy",
    "heyy", "hiii", "helloo", "heyyy", "hellooo",
    "thanks", "thank you", "thx", "ty", "tysm", "thank u",
    "ok", "okay", "ok!", "okay!", "k", "kk", "okey",
    "yes", "no", "yep", "yup", "nope", "nah", "yeah", "yea", "ya",
    "bye", "goodbye", "see ya", "see you", "cya", "later",
    "good", "great", "nice", "cool", "awesome", "perfect", "got it",
    "sure", "alright", "fine", "np", "no problem", "no worries",
    "lol", "lmao", "haha", "hahaha", "lmfao", "😂",
    "gm", "gn", "good morning", "good night", "good evening",
    "good afternoon", "morning", "night", "evening",
    # Arabic
    "سلام", "مرحبا", "اهلا", "شكرا", "مرسي", "هلا", "هاي",
    "صباح الخير", "مساء الخير",
    # Spanish
    "hola", "gracias", "buenas", "buenos dias", "buenas noches",
    "buenas tardes", "adios", "vale", "si", "claro",
    # French
    "bonjour", "salut", "merci", "bonsoir", "au revoir", "coucou",
    "oui", "non", "d'accord",
    # German
    "hallo", "danke", "tschüss", "ja", "nein", "gut", "moin",
    # Portuguese
    "oi", "olá", "obrigado", "obrigada", "tchau", "valeu",
    # Russian
    "привет", "здравствуйте", "спасибо", "пока", "да", "нет",
    # Chinese
    "你好", "谢谢", "再见", "好的", "嗨",
    # Japanese
    "こんにちは", "ありがとう", "さようなら", "はい", "いいえ",
    # Korean
    "안녕하세요", "감사합니다", "안녕", "네", "아니요",
    # Hindi
    "नमस्ते", "धन्यवाद", "शुक्रिया", "हाँ", "नहीं",
    # Turkish
    "merhaba", "teşekkürler", "evet", "hayır", "selam",
    # Italian
    "ciao", "grazie", "buongiorno", "buonasera", "arrivederci",
    # Persian
    "درود", "ممنون", "خداحافظ", "بله", "خیر",
    "سلام چطوری", "سلام خوبی", "سلام علیکم", "چطوری", "خوبی",
    "صبح بخیر", "شب بخیر", "عصر بخیر", "خسته نباشی",
    # Swahili
    "jambo", "habari", "asante",
}

# Greeting patterns (starts-with or contains)
_GREETING_PREFIXES = (
    "hey there", "hi there", "hello there", "what's up", "whats up",
    "how are you", "how's it going", "how are things", "how do you do",
    "what's good", "how you doing", "how have you been",
    "good to see you", "nice to see you", "long time no see",
    "pleased to meet", "nice to meet",
    # Persian
    "سلام چطوری", "سلام خوبی", "سلام علیکم",
)

# Emoji-only pattern (message is entirely emoji/whitespace/punctuation)
_EMOJI_ONLY_RE = re.compile(
    r'^[\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
    r'\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251'
    r'\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF'
    r'\U00002600-\U000026FF\U0000FE00-\U0000FE0F\U0000200D'
    r'!?.,:;\'"()\-_~*#@&+=%\[\]{}/<>\\|`^$]+$'
)

# ---------------------------------------------------------------------------
# Intent keyword sets (lowercase, checked against normalized message)
# ---------------------------------------------------------------------------

# Memory intent — user wants to recall or store information
_MEMORY_KEYWORDS = {
    "remember", "recall", "what did i say", "what did i tell",
    "do you remember", "did i mention", "what do you know about me",
    "forget", "delete memory", "remove memory", "clear memory",
    "my memories", "what have i told you", "save this",
    "store this", "note this", "remember this",
    # Temporal recall — references to past days the agent should fetch.
    "yesterday", "last week", "last monday", "last tuesday", "last wednesday",
    "last thursday", "last friday", "last saturday", "last sunday",
    "what we discussed", "what we talked about", "what you taught me",
    "based on yesterday", "from yesterday", "on monday", "on tuesday",
    "on wednesday", "on thursday", "on friday", "on saturday", "on sunday",
}

_MEMORY_PATTERNS_RE = re.compile(
    r'\b(?:remember|recall|memorize|forget)\b.*\b(?:that|this|about|when)\b'
    r'|\bwhat (?:did i|do you (?:know|remember))\b'
    r'|\b(?:save|store|note|delete|remove|clear)\s+(?:this|that|my|the)\s*(?:memory|memories|info)?\b'
    # Temporal recall: "yesterday", weekday names, "N days/weeks ago" — these
    # signal the user is referencing a past day the agent should load.
    r'|\byesterday\b'
    r'|\b(?:last|past|this past)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week)\b'
    r'|\b\d+\s+(?:days?|weeks?|months?)\s+ago\b'
    r'|\b(?:what|which)\b.*\b(?:we|you)\b.*\b(?:discussed|talked|taught|learned|said|did|built|decided)\b.*\b(?:yesterday|last\s+\w+|\d+\s+days?\s+ago)\b',
    re.IGNORECASE,
)

# Web intent — user wants to search or fetch from internet
_WEB_KEYWORDS = {
    "search", "google", "look up", "lookup", "find online",
    "search the web", "search online", "web search",
    "what is the latest", "latest news", "current",
}

_WEB_PATTERNS_RE = re.compile(
    r'\b(?:search|google|look\s*up|find)\b.*\b(?:online|web|internet|for me)\b'
    r'|https?://'
    r'|\b(?:latest|current|recent|today)\b.*\b(?:news|price|weather|score|status|update)\b',
    re.IGNORECASE,
)

# Media intent — user wants to play, send, or view media
_MEDIA_KEYWORDS = {
    "play", "listen", "watch", "music", "song", "video",
    "youtube", "netflix", "spotify",
    "send photo", "send file", "send image", "send video",
    "voice", "speak", "read aloud", "tts",
    "analyze image", "describe image", "what's in this image",
    "draw", "paint", "sketch", "canvas",
    # Image GENERATION (gpt-image-1 / generate_image tool)
    "generate image", "generate an image", "create image", "create an image",
    "make an image", "make me an image", "image of", "picture of",
    "generate a picture", "create a picture", "illustration of", "render an image",
    # Image EDITING (gpt-image-1 edits / edit_image tool)
    "edit image", "edit this image", "edit the image", "edit this photo",
    "edit the photo", "modify this image", "modify the image", "change the image",
    "change this photo", "edit my photo", "edit my picture", "retouch",
    "add to this image", "remove from this image", "make this image",
}

_MEDIA_PATTERNS_RE = re.compile(
    r'\b(?:play|listen to|watch|put on|queue)\b.*\b(?:song|music|video|movie|show|episode|track|album|playlist|netflix|youtube|spotify)\b'
    r'|\b(?:send|share)\s+(?:this|a|the|my)?\s*(?:photo|image|picture|file|document|video|voice)\b'
    r'|\b(?:play|listen|watch)\s+(?:me\s+)?(?:some|a)\b'
    # Image generate/edit — a verb + an image-noun, plus the extremely common
    # slang "pic"/"pics". Word boundaries keep "pic" from matching inside
    # epic / typical / picnic / spicy / pickup. Without this, "generate pic" or
    # "make me a photo" scored 0 for media and generate_image was never offered
    # to the model, which then wrongly claimed it couldn't generate images.
    r'|\b(?:generate|create|make|draw|paint|render|design|edit|modify|change|retouch|redraw|redo|upscale)\b(?:\s+\w+){0,4}?\s+(?:photos?|pics?|images?|pictures?|selfies?|portraits?|avatars?|wallpapers?|logos?|posters?|illustrations?)\b'
    r'|\bpics?\b',
    re.IGNORECASE,
)

# Code/file intent — user wants to build, deploy, fix, or manipulate files
_CODE_KEYWORDS = {
    "code", "build", "deploy", "fix", "debug", "run", "execute",
    "compile", "install", "script", "function", "class", "method",
    "error", "bug", "crash", "exception", "traceback", "stack trace",
    "file", "directory", "folder", "path",
    "git", "commit", "push", "pull", "branch", "merge",
    "docker", "container", "server", "service", "port",
    "database", "sql", "query", "migration", "schema", "table",
    "api", "endpoint", "route", "request", "response",
    "test", "lint", "format", "refactor", "optimize",
    "pip", "npm", "yarn", "package", "dependency",
    "create app", "build app", "make app", "scaffold",
    "read file", "write file", "edit file",
    "ssh", "systemctl", "nginx", "cron",
    # Document-export formats — a short "make me a PDF" (5 words) scored
    # zero everywhere and fell to the tool-less question intent, so the
    # generate_* tools were never offered (canary 533354ce, 2026-07-28).
    # Code intent carries TOOLS_DOCGEN, so format words route there.
    "pdf", "docx", "pptx", "xlsx", "spreadsheet", "slide deck", "powerpoint",
}

_CODE_PATTERNS_RE = re.compile(
    r'\b(?:create|build|make|scaffold|deploy|fix|debug|run|execute|install|write|read|edit|delete|remove|update|modify|change|add|implement|refactor)\b'
    # Document-export nouns (pdf…invoice) live in this alternation so a
    # verb+format ask ("write me an invoice", "create a slide deck")
    # lands in code intent, which carries TOOLS_DOCGEN.
    r'.*\b(?:app|code|script|function|class|file|server|service|database|api|endpoint|test|package|component|module|feature|bug|error|page|route|config|pdf|docx|pptx|xlsx|document|spreadsheet|presentation|slides|deck|report|invoice)\b'
    r'|```'  # Code blocks in message
    r'|\b(?:pip|npm|yarn|apt|brew|cargo|go)\s+(?:install|add|remove|update)\b'
    r'|\b(?:git|docker|kubectl|systemctl|nginx|crontab)\s+\w+',
    re.IGNORECASE,
)

# Agent management intent
_AGENT_KEYWORDS = {
    "spawn", "subprocess", "background task", "agent",
    "process", "kill process", "stop process", "list processes",
    "session", "sessions", "history", "conversation",
    "lane", "lanes",
    # Sub-agent vocabulary — caught 2026-05-24 when "spin up 3 sub-agents
    # in parallel" classified as web intent (research/news beat the lone
    # "agent" match) and spawn was filtered out. These keywords give the
    # agent intent enough weight to win on explicit spawn phrasings.
    "sub-agent", "subagent", "sub agents", "sub-agents",
    "spin up", "fire off", "kick off", "in parallel",
}

_AGENT_PATTERNS_RE = re.compile(
    r'\b(?:spawn|kill|stop|list|show)\s+(?:a\s+)?(?:agent|process|subprocess|task|session|lane)\b'
    r'|\b(?:background|parallel)\s+(?:task|process|job)\b'
    # "spin up N sub-agents" / "spawn 3 agents" — explicit fan-out
    # phrasing that the original regex missed.
    r'|\b(?:spin\s+up|fire\s+off|kick\s+off|spawn)\s+\d*\s*(?:sub[\-\s]?agents?|agents?|workers?|tasks?)\b'
    r'|\b(?:in\s+parallel|simultaneously)\b.*\b(?:agent|task|research|job)\b',
    re.IGNORECASE,
)

# Scheduling intent — reminders and recurring routines. Time-anchored
# phrasings are the signal; plain "every"/"tomorrow" alone are not
# (too common in prose).
_SCHEDULING_KEYWORDS = {
    "remind", "reminder", "reminders", "schedule", "scheduled",
    "routine", "routines", "briefing", "daily", "weekly", "hourly",
    "every morning", "every night", "every evening", "every day",
    "every week", "each morning", "each day", "wake me",
}

_SCHEDULING_PATTERNS_RE = re.compile(
    # "remind me…", "set a reminder", "don't let me forget"
    r'\bremind(?:er)?s?\b'
    r'|\bdon\'?t let me forget\b'
    # "every 11:10 pm", "every day at 7", "at 7 am every morning"
    r'|\bevery\s+(?:\d{1,2}(?::\d{2})?\s*(?:am|pm)?|day|morning|night|evening|week|hour|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
    # "at 11:10 pm send/tell/nudge/wake/message me"
    r'|\bat\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\b.*\b(?:send|tell|text|message|nudge|wake|ping|joke)\b'
    r'|\b(?:send|tell|text|message|nudge|wake|ping)\b.*\bat\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\b'
    # "in 20 minutes/2 hours" paired with an action verb
    r'|\bin\s+\d+\s*(?:min(?:ute)?s?|hours?|hrs?)\b'
    # daily/weekly/morning briefing or summary
    r'|\b(?:daily|weekly|morning|evening)\b.*\b(?:briefing|summary|digest|report|joke|quote|news)\b',
    re.IGNORECASE,
)

# Document intent — the user names a FILE they want to keep, share, or
# edit. This is the only turn-1 gate on TOOLS_DOCGEN (see the note on that
# set). Deliberately generous on the artifact side — a hidden exporter
# is the #371 failure (the model improvises with write_file) — and
# deliberately silent on subject vocabulary: "search the latest AI news
# and give me the 5 most important" names no artifact, and must not
# match. Three alternations:
#   1. Export formats — pdf, docx/word doc, xlsx/excel/spreadsheet/
#      workbook/csv, pptx/powerpoint/slides/slide deck/presentation,
#      markdown/.md.
#   2. Artifact nouns the user expects as a file — document/file/report/
#      invoice/… — with a make/write/create/generate/draft/prepare/put
#      together/export/turn-into verb somewhere before them, so "the
#      report said" and "any file will do" don't fire.
#   3. Export verbs on their own — export / download(able) / printable /
#      save as / as a file / attach(ment).
# Follow-ups after an offer ("yes please, PDF") reach here too: a
# format word skips the greeting shortcuts in classify_query_intent.
_DOCUMENT_INTENT_RE = re.compile(
    r'\b(?:pdfs?|docx?|word\s+(?:doc(?:ument)?|file)s?|xlsx?|excel|spreadsheets?'
    r'|workbooks?|csv|pptx?|powerpoint|slides?|slide\s*decks?|decks?'
    r'|presentations?|markdown)\b'
    r'|\.md\b'
    r'|\b(?:make|write|create|generate|draft|prepare|produce|build|compile'
    r'|put\s+together|turn\s+(?:this|that|it)\s+into|convert|export|give\s+me'
    r'|send\s+me|i\s+(?:need|want))\b(?:\s+\S+){0,6}?\s+'
    r'(?:documents?|files?|reports?|invoices?|receipts?|contracts?|proposals?'
    r'|memos?|letters?|resumes?|cvs?|one[\s-]?pagers?|handouts?'
    r'|worksheets?|templates?|write[\s-]?ups?|whitepapers?|summar(?:y|ies)\s+(?:document|file|pdf))\b'
    r'|\b(?:export|download(?:able)?|printable|save\s+(?:\S+\s+){0,3}?as|as\s+a\s+file'
    r'|in\s+a\s+file|attach(?:ment|ed)?)\b',
    re.IGNORECASE,
)


def has_document_intent(message: str) -> bool:
    """True when the message names a file/export the user wants — the gate
    that puts TOOLS_DOCGEN into the turn-1 allowed set. Pure regex, no LLM,
    <0.1ms; see _DOCUMENT_INTENT_RE for what counts."""
    if not message:
        return False
    return bool(_DOCUMENT_INTENT_RE.search(message))


def with_document_tools(intent: QueryIntent) -> QueryIntent:
    """Merge the export tools onto an intent (turn-1 allowed set only)."""
    if intent.category == "full":
        return intent
    return replace(intent, tool_names=frozenset(intent.tool_names) | TOOLS_DOCGEN)


# Short but meaningful messages that should NOT be classified as greetings
_SHORT_BUT_MEANINGFUL_RE = re.compile(
    r'\b(?:delete|remove|cancel|stop|abort|undo|redo|reset|clear|wipe)\b'
    r'|\b(?:what|where|when|who|how|why|which|show|tell|give|get|set|list)\b.*\b(?:my|the|this|that|is|are|was|were)\b'
    r'|\b(?:balance|account|password|subscription|plan|billing|payment|order|status|settings|config|profile)\b'
    r'|\b(?:help me|i need|i want|can you|could you|please|would you)\b',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Pre-built intent results (singletons — no allocation per call)
# ---------------------------------------------------------------------------

INTENT_GREETING = QueryIntent(
    category="greeting",
    tool_names=frozenset(),
    include_skills=False,
    include_skill_prompts=False,
    include_environment=False,
    include_media_section=False,
)

INTENT_QUESTION = QueryIntent(
    category="question",
    tool_names=frozenset(),
    include_skills=False,
    include_skill_prompts=False,
    include_environment=False,
    include_media_section=False,
)

INTENT_MEMORY = QueryIntent(
    category="memory",
    tool_names=TOOLS_MEMORY,
    include_skills=False,
    include_skill_prompts=False,
    include_environment=False,
    include_media_section=False,
)

INTENT_WEB = QueryIntent(
    category="web",
    tool_names=TOOLS_WEB_FULL,
    include_skills=False,
    include_skill_prompts=False,
    include_environment=False,
    include_media_section=False,
)

INTENT_MEDIA = QueryIntent(
    category="media",
    tool_names=TOOLS_MEDIA,
    include_skills=False,
    include_skill_prompts=False,
    include_environment=False,
    include_media_section=True,
)

INTENT_CODE = QueryIntent(
    category="code",
    tool_names=TOOLS_CODE_FULL,
    include_skills=True,
    include_skill_prompts=True,
    include_environment=True,
    include_media_section=False,
)

INTENT_SCHEDULING = QueryIntent(
    category="scheduling",
    tool_names=TOOLS_SCHEDULING,
    include_skills=False,           # routines__ names match explicitly
    include_skill_prompts=False,
    include_environment=True,       # reminders need the current clock
    include_media_section=False,
)

INTENT_AGENT = QueryIntent(
    category="agent",
    tool_names=TOOLS_AGENT,
    include_skills=False,
    include_skill_prompts=False,
    include_environment=True,
    include_media_section=False,
)

INTENT_FULL = QueryIntent(
    category="full",
    tool_names=frozenset(),  # empty = send all
    include_skills=True,
    include_skill_prompts=True,
    include_environment=True,
    include_media_section=True,
)


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def classify_query_intent(message: str) -> QueryIntent:
    """
    Classify a user message into an intent category for tool/prompt filtering.

    Rules (evaluated in priority order — first match wins):
      1. Empty/whitespace/emoji-only → greeting
      2. Exact match against greeting word list → greeting
      3. Short (≤4 words) greeting prefix match → greeting
         (but NOT if it contains meaningful action keywords)
      4. Explicit memory keywords/patterns → memory
      5. Explicit media keywords/patterns → media
      6. Explicit web keywords/patterns → web
      7. Explicit code keywords/patterns → code
      8. Explicit agent keywords/patterns → agent
      9. Short (≤6 words) with no tool indicators → question
     10. Default → full (send everything)

    Returns:
        QueryIntent with the classification result.
    """
    if not message or not message.strip():
        return INTENT_GREETING

    stripped = message.strip()
    normalized = stripped.lower()

    # Remove trailing punctuation/emoji for matching
    clean = normalized.rstrip("!?.,;:) \t\n")

    # Document intent is orthogonal to the category — "make me a PDF of X"
    # classifies by X — so it is decided once here and merged onto
    # whatever category wins below. It also disqualifies the greeting
    # shortcuts: "yes pdf please" (first word a greeting, ≤4 words) is a
    # follow-up accepting an offered export, not a greeting.
    _doc = has_document_intent(normalized)

    def _finish(intent: QueryIntent) -> QueryIntent:
        return with_document_tools(intent) if _doc else intent

    # 1. Emoji-only messages
    if _EMOJI_ONLY_RE.match(stripped):
        return INTENT_GREETING

    # 2. Exact greeting match (after lowering + stripping punctuation)
    if clean in _GREETING_EXACT:
        return INTENT_GREETING

    # 2b. First word is a known greeting and message is short (≤4 words)
    #     Catches "سلام چطوری", "hi how are you", "hola que tal", etc.
    first_word = clean.split()[0] if clean else ""
    if first_word in _GREETING_EXACT and len(clean.split()) <= 4 and not _doc:
        return INTENT_GREETING

    # 3. Short greeting prefix match — but check for meaningful content first
    word_count = len(stripped.split())
    if word_count <= 5 and not _doc:
        # Check if it's actually meaningful despite being short
        if _SHORT_BUT_MEANINGFUL_RE.search(normalized):
            pass  # Fall through to intent detection
        elif any(normalized.startswith(p) for p in _GREETING_PREFIXES):
            return INTENT_GREETING

    # 4-8. Check each intent category
    # Score each category and pick the strongest match

    scores = {
        "memory": 0,
        "media": 0,
        "web": 0,
        "code": 0,
        "agent": 0,
        "scheduling": 0,
    }

    # Scheduling — checked like the others; wins ties (see priority
    # order below) because a time-anchored ask ("every morning send me
    # the news") is a routine to CREATE, not a one-off web search.
    for kw in _SCHEDULING_KEYWORDS:
        if kw in normalized:
            scores["scheduling"] += 2
    if _SCHEDULING_PATTERNS_RE.search(normalized):
        scores["scheduling"] += 3

    # Memory
    for kw in _MEMORY_KEYWORDS:
        if kw in normalized:
            scores["memory"] += 2
    if _MEMORY_PATTERNS_RE.search(normalized):
        scores["memory"] += 3
    # Explicit remember-phrasing — D-mem-C (2026-07-29), #371 TOOLS_DOCGEN
    # precedent: realistic save asks scored memory=2 at most, and any payload
    # containing a _CODE_KEYWORDS word tied them into code intent on the
    # priority order ("Please remember: my assigned parking spot CODE is …"
    # → code). Tool EXPOSURE was never the gap (memory_store/memory_search
    # are always-included) — but intent drives the prompt sections and the
    # allowed-tools restriction under the stable-prefix layout, so an
    # explicit save request must land in memory intent. +4 beats a keyword
    # collision (2) without outscoring a real work ask (patterns 3 + stacked
    # keywords). Kill switch: settings.memory_tools_on_remember.
    if _remember_boost_enabled() and is_explicit_remember_request(normalized):
        scores["memory"] += 4

    # Media
    for kw in _MEDIA_KEYWORDS:
        if kw in normalized:
            scores["media"] += 2
    if _MEDIA_PATTERNS_RE.search(normalized):
        scores["media"] += 3

    # Web
    for kw in _WEB_KEYWORDS:
        if kw in normalized:
            scores["web"] += 2
    if _WEB_PATTERNS_RE.search(normalized):
        scores["web"] += 3
    # URLs are a strong web signal
    if "http://" in normalized or "https://" in normalized:
        scores["web"] += 5

    # Code
    for kw in _CODE_KEYWORDS:
        if kw in normalized:
            scores["code"] += 2
    if _CODE_PATTERNS_RE.search(normalized):
        scores["code"] += 3
    # Code blocks are a very strong code signal
    if "```" in stripped:
        scores["code"] += 10

    # Agent
    for kw in _AGENT_KEYWORDS:
        if kw in normalized:
            scores["agent"] += 2
    if _AGENT_PATTERNS_RE.search(normalized):
        scores["agent"] += 3

    # Find the highest-scoring category
    max_score = max(scores.values())

    if max_score >= 2:
        # Pick the winner (ties resolved by priority: scheduling > code
        # > web > media > memory > agent — a time-anchored ask is a
        # routine to create, not a search to run).
        priority = ["scheduling", "code", "web", "media", "memory", "agent"]
        for cat in priority:
            if scores[cat] == max_score:
                intent_map = {
                    "memory": INTENT_MEMORY,
                    "media": INTENT_MEDIA,
                    "web": INTENT_WEB,
                    "code": INTENT_CODE,
                    "agent": INTENT_AGENT,
                    "scheduling": INTENT_SCHEDULING,
                }
                return _finish(intent_map[cat])

    # 9. Short messages with no tool indicators → question (no tools needed)
    #    A short ask that names a file ("as a word doc please") keeps the
    #    question category but carries the export tools.
    if word_count <= 8 and max_score == 0:
        return _finish(INTENT_QUESTION)

    # 10. Default: full toolset for anything complex or ambiguous
    return INTENT_FULL


# Transversal UX-affordance tools — ALWAYS included regardless of intent.
# These don't fit a single category because the user can ask for them in
# greetings, questions, anywhere. Without this, the LLM sees the tool
# described in the system prompt but can't actually invoke it on a
# greeting/question turn — and then hallucinates fake XML/JSX syntax in
# its text output (caught 2026-05-08 when "transfer me to user brain"
# came back as `<navigate_to path="/brain/user" />` plain text instead
# of an actual tool call).
_ALWAYS_INCLUDED_TOOLS = frozenset({
    "navigate_to",  # Page transfers — needed in any intent
    "recall_day",   # Past-day questions can fall under any category
    "memory_search",  # "what do you remember about X" can hit any intent
    # Same reasoning, and stronger: the `# User Brain` index NAMES the
    # user's files on every non-trivial turn, so any intent can end up
    # holding a slug it needs to open. A tool the prompt advertises and the
    # intent gate hides is the A6-3 failure class (the model either
    # hallucinates the content or says it cannot read its own memory).
    "memory_read_file",
    # "Remember <fact>" is an explicit-save ask the system prompt mandates
    # a memory_store call for, yet greeting/question/web/media intents
    # filtered the tool out — the model either hallucinated "saved!" or
    # truthfully said it can't (audit A6-3, 2026-07-23). Same failure
    # class as routines__remind below: short/typo'd asks score zero in
    # the memory keyword list and land in question intent. Background
    # auto-extraction still runs either way; exposing the tool just makes
    # explicit saves real.
    "memory_store",
    # spawn is a meta-affordance: any sufficiently expensive turn — even a
    # "research" prompt that classifies as web intent — may want to be
    # backgrounded into a sub-agent. Hiding spawn from non-agent intents
    # forces the LLM to do the work inline and lie to the user that it
    # "doesn't have a sub-agent spawning tool" (caught 2026-05-24 when
    # "Spin up 3 sub-agents to research…" scored web > agent and spawn
    # was filtered out). spawn's runtime gate still applies (the kill
    # switch + per-tenant subagent_spawning_enabled flag), so making it
    # always-visible to the LLM doesn't bypass the rollout controls.
    "spawn",
    # Reminders are the canonical short-message ask, and typos defeat
    # keyword classification entirely: founder repro 2026-07-16 —
    # "Temind me teo means later" (→ question, zero score) left the
    # agent truthfully saying "the reminder tool isn't available to me
    # in this turn". Same failure class as the spawn incident above.
    # The runtime flag gates (ROUTINES_REMINDERS_ENABLED) still apply
    # at execution, so visibility here bypasses nothing.
    "routines__remind",
    "routines__create",
    "routines__list",
    # ND-18: "what automations do I have?" classifies `question`, and
    # `routines__list` was visible on those turns while the automations
    # inventory was not — so the only list the model could reach was the
    # wrong one, and it answered from reminders. Prompt wording cannot
    # fix a tool that is not on the turn. Read-only, and the skill's own
    # flag gate still applies at execution, so visibility bypasses
    # nothing: a tenant without automations still has no such tool.
    "automations__list",
})


def filter_tools_by_intent(
    all_tools: list,
    intent: QueryIntent,
) -> list:
    """
    Filter a list of tool definitions to only those relevant for the given intent.

    Args:
        all_tools: Full list of tool definitions (Anthropic format with "name" key).
        intent: The classified intent.

    Returns:
        Filtered list of tool definitions. If intent.tool_names is empty,
        returns all_tools for "full" intent, or only the always-included
        UX-affordance tools for "greeting"/"question".
    """
    if intent.category == "full":
        return all_tools

    if not intent.tool_names:
        # greeting, question — no work tools, but still surface the
        # always-included UX affordances so the LLM can navigate / recall
        # / search memory without an intent escalation round-trip.
        return [t for t in all_tools if t.get("name", "") in _ALWAYS_INCLUDED_TOOLS]

    # Filter to only matching tool names + always-included UX affordances.
    # Skill tools have prefixed names (e.g., "app_builder__build_app")
    # Include them if the intent says include_skills.
    filtered = []
    for tool in all_tools:
        name = tool.get("name", "")
        if name in intent.tool_names or name in _ALWAYS_INCLUDED_TOOLS:
            filtered.append(tool)
        elif intent.include_skills and "__" in name:
            # Skill tool — include it
            filtered.append(tool)

    return filtered


def with_inbound_image(intent: QueryIntent) -> QueryIntent:
    """Augment a text-classified intent for a turn that carries an inbound image.

    Tool-gating above is *text-only*. When a user attaches a photo they almost
    always want it looked at or edited, but a short caption that names no
    image-noun — "make a six pack", "fix this", "make me look muscular", or no
    caption at all — classifies as ``question``/``greeting``/``code``/``memory``
    and never exposes ``edit_image``/``analyze_image``. The model is then handed
    the image (it can *see* it) but has no tool to act on it, so it falsely
    claims it "can't edit/render the image in this chat".

    Given an inbound image we merge the media toolset onto whatever the text
    classified into, so ``edit_image``/``generate_image``/``analyze_image`` are
    always available. We only ADD tools — the model still decides whether to
    call one (e.g. it can still just answer "what's in this?" without editing).
    """
    if intent.category == "full":
        return intent  # "full" already exposes every tool, including media
    return replace(intent, tool_names=frozenset(intent.tool_names) | TOOLS_MEDIA)
