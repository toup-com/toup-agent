"""
Lightweight query classifier for optimized memory retrieval.

Classifies user queries by type (identity, preference, entity, project, goal,
temporal, general) so the retrieval engine can prioritize the right strategies
and filter to relevant categories — no LLM call needed.
"""

import re
from typing import Dict, List, Optional


def classify_query(query: str) -> Dict:
    """
    Classify a query type to optimize retrieval strategy.

    Returns:
        {
            "type": str,            # identity|preference|entity|project|goal|temporal|general
            "categories": list|None,  # suggested category filter (None = all)
            "entity_hint": str|None,  # extracted entity name if detected
            "temporal": bool,       # whether temporal language was detected
            "strategies": list|None,  # suggested strategies (None = all defaults)
        }
    """
    q = query.lower().strip()

    result = {
        "type": "general",
        "categories": None,
        "entity_hint": None,
        "temporal": False,
        "strategies": None,
    }

    # --- Temporal queries ---
    if re.search(
        r"\b(yesterday|last week|last month|recently|today|when did|how long ago"
        r"|this week|this month|lately|recent)\b",
        q,
    ):
        result["type"] = "temporal"
        result["temporal"] = True
        result["strategies"] = ["vector", "keyword", "temporal"]

    # --- Identity queries ---
    elif re.search(
        r"\b(my name|who am i|where do i live|how old|my age|my background"
        r"|my email|my phone|my birthday|about me)\b",
        q,
    ):
        result["type"] = "identity"
        result["categories"] = ["identity", "people", "locations"]
        result["strategies"] = ["vector", "keyword"]

    # --- Preference queries ---
    elif re.search(
        r"\b(do i like|my fav|i prefer|what.*like|taste|preference"
        r"|my style|what kind of)\b",
        q,
    ):
        result["type"] = "preference"
        result["categories"] = ["preferences", "media", "habits"]

    # --- People queries ---
    elif re.search(
        r"\b(who is|tell me about|my friend|my colleague|my brother"
        r"|my sister|my boss|my partner|my wife|my husband"
        r"|my mom|my dad|my father|my mother)\b",
        q,
    ):
        result["type"] = "entity"
        result["categories"] = ["people", "relationships", "work"]
        result["strategies"] = ["vector", "keyword", "graph"]
        # Extract the entity name (skip possessives like "my")
        name_match = re.search(
            r"(?:who is|about|know about)\s+(?:my\s+(?:friend|colleague|brother|sister|boss|partner)\s+)?(\w+)",
            q, re.I,
        )
        if name_match:
            candidate = name_match.group(1).lower()
            # Skip common non-name words
            if candidate not in {"my", "the", "a", "an", "this", "that"}:
                result["entity_hint"] = name_match.group(1)

    # --- Project queries ---
    elif re.search(
        r"\b(my project|working on|building|app builder"
        r"|what.*build|current project)\b",
        q,
    ):
        result["type"] = "project"
        result["categories"] = ["work", "goals", "active_task"]

    # --- Goal queries ---
    elif re.search(
        r"\b(my goal|plan|want to|aspir|dream|aim|objective"
        r"|trying to|hope to)\b",
        q,
    ):
        result["type"] = "goal"
        result["categories"] = ["goals", "skills", "experiences"]

    # --- Work queries ---
    elif re.search(
        r"\b(my job|where.*work|my company|my career|my role"
        r"|my team|my manager)\b",
        q,
    ):
        result["type"] = "work"
        result["categories"] = ["work", "identity", "active_task"]

    # --- Health queries ---
    elif re.search(
        r"\b(my health|exercise|workout|diet|sleep|medical"
        r"|fitness|symptom)\b",
        q,
    ):
        result["type"] = "health"
        result["categories"] = ["health", "habits"]

    # --- Learning queries ---
    elif re.search(
        r"\b(learning|studying|course|exam|skill|tutorial"
        r"|what.*study|school|university)\b",
        q,
    ):
        result["type"] = "learning"
        result["categories"] = ["skills", "knowledge", "goals"]

    return result


# ---------------------------------------------------------------------------
# Trivial-query classifier — TKT-LAT-019 (context trim)
# ---------------------------------------------------------------------------
#
# Returns True for messages that don't need the full memory/portrait/entity
# context to answer well: short greetings, acknowledgments, simple time/date
# questions, and other low-information utterances. When True, the agent
# runner skips ~5-15 k tokens of optional context (portrait, hybrid memory
# search, entities, active tasks) so a one-word answer like "what time is
# it?" doesn't pay a 11-second tax.
#
# Conservative by design — favor false negatives (run with full context)
# over false positives (drop context that the user needed). If a query
# might benefit from memory, leave it on the non-trivial path.

# Single-word/short acknowledgments + greetings + thanks. Token-level set
# because we want exact-match on the trimmed lowercase string.
_TRIVIAL_EXACT = {
    # Acknowledgments
    "ok", "okay", "k", "kk", "alright", "right", "sure", "yep", "yup", "yeah",
    "yes", "no", "nope", "nah", "got it", "gotcha", "noted", "understood",
    "copy", "copy that", "roger", "roger that", "10-4",
    # Thanks
    "ty", "thx", "tysm", "thanks", "thank you", "thanks so much",
    "much appreciated", "appreciate it", "appreciated", "cheers",
    # Affirmations
    "cool", "nice", "great", "awesome", "perfect", "sweet", "rad",
    "love it", "love that", "amazing", "brilliant", "excellent",
    "good", "good stuff", "good one", "well done", "nicely done",
    "sounds good", "looks good", "lgtm", "works for me", "wfm",
    "no worries", "np", "no problem", "all good", "of course",
    "of course!", "ofc", "exactly", "right on", "true", "fair", "fair enough",
    # Greetings
    "hi", "hello", "hey", "yo", "sup", "howdy",
    "good morning", "good afternoon", "good evening", "good night",
    "gn", "gm", "gnight", "morning", "evening",
    # Farewells
    "bye", "goodbye", "cya", "see ya", "see you", "later", "ttyl",
    "talk later", "catch you later", "take care", "peace",
    # Reactions
    "lol", "haha", "ha", "hahaha", "lmao", "lmfao", "wow", "omg",
    "yikes", "oof", "oops", "huh", "hmm", "uh", "uhh",
    # Conversational openers (apostrophe-stripped variants too, since the
    # is_trivial_query() check strips trailing punctuation but not interior).
    "how's it going", "hows it going", "how is it going",
    "how are you", "how r u", "how are ya", "how're you", "how are things",
    "how's your day", "hows your day", "how's everything", "hows everything",
    "what's up", "whats up", "wassup", "wazzup", "sup",
    "you there", "are you there", "are you alive",
    "long time no see", "good to see you", "miss you",
}

# Substring patterns for short conversational openers + factual questions
# that the agent can answer from current context alone. Two families:
#
#   1. Time/date/day questions (no user memory needed — agent has tz +
#      current time in its system prompt).
#   2. Conversational greetings / status-check openers — "how's it going",
#      "how are you", "what's up", etc. These are social openers; the
#      agent's response is a natural-language hello, not a memory recall.
#
# Both families are bounded by _TRIVIAL_MAX_WORDS (6) so a longer message
# that happens to start with one of these phrases stays on the full path.
_TRIVIAL_RE = re.compile(
    r"^("
    # ── Time / date / day questions ───────────────────────────────
    r"what(?:'s| is| s)?\s+the\s+(?:time|date|day|hour)"
    r"|what(?:'s|s)?\s+time\s+is\s+it"
    r"|what(?:'s|s)?\s+day\s+is\s+(?:it|today)"
    r"|what(?:'s| is| s)?\s+today(?:'s\s+date)?"
    r"|tell\s+me\s+the\s+(?:time|date)"
    # ── Conversational greetings / "how are you" family ───────────
    # Apostrophe-optional to catch typos like "hows it going" without
    # the apostrophe — see Toup chat 2026-05-23 12:06 PM screenshot.
    r"|how(?:'s|s|\s+is)?\s+it\s+(?:going|been|hanging)"
    r"|how(?:'re|re|\s+are)?\s+(?:you|ya|things|you\s+doing|you\s+doin|things\s+going)"
    r"|how(?:'s|s|\s+is)?\s+(?:your\s+day|everything|life|stuff)"
    r"|how(?:'ve|ve|\s+have)?\s+you\s+been(?:\s+(?:today|lately|recently|so\s+far|this\s+week))?"
    r"|what(?:'s|s)?\s+(?:up|good|new|happening|going\s+on)"
    r"|you\s+(?:there|good|ok|okay|alive|around)"
    r"|are\s+you\s+(?:there|alive|good|ok|okay|around|busy)"
    r"|still\s+(?:there|alive|with\s+me)"
    r"|good\s+to\s+(?:see|hear\s+from)\s+you"
    r"|long\s+time\s+no\s+(?:see|chat|talk)"
    r"|miss\s+(?:you|me)"
    r")\s*[?!.]*\s*$",
    re.IGNORECASE,
)

# Hard cap on word count for the trivial classifier to even look at the
# message. Anything longer almost certainly has enough content to benefit
# from full context.
_TRIVIAL_MAX_WORDS = 8  # bumped from 6 to fit "how have you been today"-class openers


def is_trivial_query(text: str) -> bool:
    """Return True iff the message can be answered without
    portrait/memory/entity context.

    See module-level comment for the design constraints. Always lossless
    against the existing behavior: when this returns False, downstream
    callers run the full context pipeline exactly as before.
    """
    if not text:
        return False
    stripped = text.strip().lower()
    # Drop trailing punctuation that doesn't change meaning.
    stripped = stripped.rstrip("?!.")
    if not stripped:
        return False
    # 1. Exact-string acknowledgments / greetings.
    if stripped in _TRIVIAL_EXACT:
        return True
    # 2. Short, simple time/date questions (regex above already enforces
    #    structure; the word-count cap is belt-and-suspenders).
    if len(stripped.split()) <= _TRIVIAL_MAX_WORDS and _TRIVIAL_RE.match(stripped + "?"):
        return True
    return False


# ---------------------------------------------------------------------------
# Explicit remember-phrasing (D-mem-C, 2026-07-29)
# ---------------------------------------------------------------------------
# A user telling the assistant to SAVE something ("Please remember: my
# assigned parking spot code is …", "For my records: …", "Note this down: …").
# The memory-quality harness measured 5 of 8 such explicit requests silently
# dropped (retrieval_events logged has_extraction_gap=true): the phrasing does
# not make the model call memory_store, so capture rides entirely on
# background extraction — which must therefore never be skipped, and must be
# told the save was explicit, on exactly these turns.
#
# Shared by three consumers (single definition on purpose):
#   * query_intent.classify_query_intent — boosts memory-intent scoring so the
#     ask isn't stolen by a keyword collision ("…spot CODE is…" ties into code
#     intent); flag-gated there via settings.memory_tools_on_remember.
#   * agent_runner._extract_memories — bypasses the trivial-turn skip gate for
#     very short remember-asks.
#   * agent_runner → memory_extractor.extract_memories_with_llm — threads
#     explicit_save_requested=True so token-like payloads (codes, passphrases)
#     are not discarded as noise.
#
# Deliberately NOT matched: "remember to <do something>" — that is reminder
# phrasing owned by the routines pipeline, not a fact-save.
_EXPLICIT_REMEMBER_RE = re.compile(
    r"\bplease\s+remember\b(?!\s+to\b)"          # "please remember (this|:|it) …"
    r"|\bremember\s+(?:this|that|it)\b"          # "remember this: …"
    r"|\bremember\s*:"                           # "remember: my … is …"
    r"|\bfor\s+(?:my|the|your)\s+records?\b"     # "for my records: …"
    r"|\bnote\s+(?:this|that|it)\s+down\b"       # "note this down: …"
    r"|\bnote\s+down\b"                          # "note down: …"
    r"|\b(?:save|store)\s+(?:this|that)\b"       # "save this: …"
    r"|\bdon'?t\s+forget\s*:",                   # "don't forget: …"
    re.IGNORECASE,
)


def is_explicit_remember_request(text: Optional[str]) -> bool:
    """True iff the message explicitly asks the assistant to remember/save a
    fact. See the block comment above for consumers and design constraints."""
    if not text:
        return False
    return bool(_EXPLICIT_REMEMBER_RE.search(text))
