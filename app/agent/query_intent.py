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

Performance target: <1ms. No regex compilation at call time (all pre-compiled).
"""

import re
import logging
from dataclasses import dataclass, field
from typing import FrozenSet, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool name sets by category
# ---------------------------------------------------------------------------

TOOLS_MEMORY: FrozenSet[str] = frozenset({
    "memory_search", "memory_store", "memory_delete",
})

TOOLS_WEB: FrozenSet[str] = frozenset({
    "web_search", "web_fetch", "browser",
})

TOOLS_MEDIA: FrozenSet[str] = frozenset({
    "send_file", "send_photo", "analyze_image", "tts", "play_media",
    "tts_prefs", "canvas",
})

TOOLS_CODE: FrozenSet[str] = frozenset({
    "exec", "pty_exec", "read_file", "write_file", "edit_file",
    "grep", "find", "ls", "apply_patch",
})

TOOLS_AGENT: FrozenSet[str] = frozenset({
    "spawn", "process", "sessions_list", "sessions_history",
    "sessions_send", "session_status", "agents_list", "lanes_status",
    "thread",
})

TOOLS_ADMIN: FrozenSet[str] = frozenset({
    "cron", "config_reload", "doctor", "moderate", "poll",
    "message", "skill_marketplace", "talk_mode",
})

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
    skip_memory_retrieval: bool = False              # Whether to skip hybrid memory search


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
}

_MEMORY_PATTERNS_RE = re.compile(
    r'\b(?:remember|recall|memorize|forget)\b.*\b(?:that|this|about|when)\b'
    r'|\bwhat (?:did i|do you (?:know|remember))\b'
    r'|\b(?:save|store|note|delete|remove|clear)\s+(?:this|that|my|the)\s*(?:memory|memories|info)?\b',
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
}

_MEDIA_PATTERNS_RE = re.compile(
    r'\b(?:play|listen to|watch|put on|queue)\b.*\b(?:song|music|video|movie|show|episode|track|album|playlist|netflix|youtube|spotify)\b'
    r'|\b(?:send|share)\s+(?:this|a|the|my)?\s*(?:photo|image|picture|file|document|video|voice)\b'
    r'|\b(?:play|listen|watch)\s+(?:me\s+)?(?:some|a)\b',
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
}

_CODE_PATTERNS_RE = re.compile(
    r'\b(?:create|build|make|scaffold|deploy|fix|debug|run|execute|install|write|read|edit|delete|remove|update|modify|change|add|implement|refactor)\b'
    r'.*\b(?:app|code|script|function|class|file|server|service|database|api|endpoint|test|package|component|module|feature|bug|error|page|route|config)\b'
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
}

_AGENT_PATTERNS_RE = re.compile(
    r'\b(?:spawn|kill|stop|list|show)\s+(?:a\s+)?(?:agent|process|subprocess|task|session|lane)\b'
    r'|\b(?:background|parallel)\s+(?:task|process|job)\b',
    re.IGNORECASE,
)

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
    skip_memory_retrieval=True,
)

INTENT_QUESTION = QueryIntent(
    category="question",
    tool_names=frozenset(),
    include_skills=False,
    include_skill_prompts=False,
    include_environment=False,
    include_media_section=False,
    skip_memory_retrieval=False,
)

INTENT_MEMORY = QueryIntent(
    category="memory",
    tool_names=TOOLS_MEMORY,
    include_skills=False,
    include_skill_prompts=False,
    include_environment=False,
    include_media_section=False,
    skip_memory_retrieval=False,
)

INTENT_WEB = QueryIntent(
    category="web",
    tool_names=TOOLS_WEB_FULL,
    include_skills=False,
    include_skill_prompts=False,
    include_environment=False,
    include_media_section=False,
    skip_memory_retrieval=False,
)

INTENT_MEDIA = QueryIntent(
    category="media",
    tool_names=TOOLS_MEDIA,
    include_skills=False,
    include_skill_prompts=False,
    include_environment=False,
    include_media_section=True,
    skip_memory_retrieval=True,
)

INTENT_CODE = QueryIntent(
    category="code",
    tool_names=TOOLS_CODE_FULL,
    include_skills=True,
    include_skill_prompts=True,
    include_environment=True,
    include_media_section=False,
    skip_memory_retrieval=False,
)

INTENT_AGENT = QueryIntent(
    category="agent",
    tool_names=TOOLS_AGENT,
    include_skills=False,
    include_skill_prompts=False,
    include_environment=True,
    include_media_section=False,
    skip_memory_retrieval=True,
)

INTENT_FULL = QueryIntent(
    category="full",
    tool_names=frozenset(),  # empty = send all
    include_skills=True,
    include_skill_prompts=True,
    include_environment=True,
    include_media_section=True,
    skip_memory_retrieval=False,
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

    # 1. Emoji-only messages
    if _EMOJI_ONLY_RE.match(stripped):
        return INTENT_GREETING

    # 2. Exact greeting match (after lowering + stripping punctuation)
    if clean in _GREETING_EXACT:
        return INTENT_GREETING

    # 2b. First word is a known greeting and message is short (≤4 words)
    #     Catches "سلام چطوری", "hi how are you", "hola que tal", etc.
    first_word = clean.split()[0] if clean else ""
    if first_word in _GREETING_EXACT and len(clean.split()) <= 4:
        return INTENT_GREETING

    # 3. Short greeting prefix match — but check for meaningful content first
    word_count = len(stripped.split())
    if word_count <= 5:
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
    }

    # Memory
    for kw in _MEMORY_KEYWORDS:
        if kw in normalized:
            scores["memory"] += 2
    if _MEMORY_PATTERNS_RE.search(normalized):
        scores["memory"] += 3

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
        # Pick the winner (ties resolved by priority: code > web > media > memory > agent)
        priority = ["code", "web", "media", "memory", "agent"]
        for cat in priority:
            if scores[cat] == max_score:
                intent_map = {
                    "memory": INTENT_MEMORY,
                    "media": INTENT_MEDIA,
                    "web": INTENT_WEB,
                    "code": INTENT_CODE,
                    "agent": INTENT_AGENT,
                }
                return intent_map[cat]

    # 9. Short messages with no tool indicators → question (no tools needed)
    if word_count <= 8 and max_score == 0:
        return INTENT_QUESTION

    # 10. Default: full toolset for anything complex or ambiguous
    return INTENT_FULL


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
        returns all_tools for "full" intent, or empty list for "greeting"/"question".
    """
    if intent.category == "full":
        return all_tools

    if not intent.tool_names:
        # greeting, question — no tools
        return []

    # Filter to only matching tool names
    # Skill tools have prefixed names (e.g., "app_builder__build_app")
    # Include them if the intent says include_skills
    filtered = []
    for tool in all_tools:
        name = tool.get("name", "")
        if name in intent.tool_names:
            filtered.append(tool)
        elif intent.include_skills and "__" in name:
            # Skill tool — include it
            filtered.append(tool)

    return filtered
