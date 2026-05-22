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
        result["categories"] = ["identity", "family", "places"]
        result["strategies"] = ["vector", "keyword"]

    # --- Preference queries ---
    elif re.search(
        r"\b(do i like|my fav|i prefer|what.*like|taste|preference"
        r"|my style|what kind of)\b",
        q,
    ):
        result["type"] = "preference"
        result["categories"] = ["preferences", "food", "media", "habits"]

    # --- People queries ---
    elif re.search(
        r"\b(who is|tell me about|my friend|my colleague|my brother"
        r"|my sister|my boss|my partner|my wife|my husband"
        r"|my mom|my dad|my father|my mother)\b",
        q,
    ):
        result["type"] = "entity"
        result["categories"] = ["people", "family", "work"]
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
        result["categories"] = ["projects", "work", "goals"]

    # --- Goal queries ---
    elif re.search(
        r"\b(my goal|plan|want to|aspir|dream|aim|objective"
        r"|trying to|hope to)\b",
        q,
    ):
        result["type"] = "goal"
        result["categories"] = ["goals", "learning", "travel"]

    # --- Work queries ---
    elif re.search(
        r"\b(my job|where.*work|my company|my career|my role"
        r"|my team|my manager)\b",
        q,
    ):
        result["type"] = "work"
        result["categories"] = ["work", "projects", "identity"]

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
        result["categories"] = ["learning", "knowledge", "goals"]

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
}

# Substring patterns for short factual questions that the agent can answer
# from current context alone (date/time/day of week). These are intentionally
# narrow — broader "what is X" stays non-trivial because the answer often
# depends on user memory.
_TRIVIAL_RE = re.compile(
    r"^("
    r"what(?:'s| is)\s+the\s+(?:time|date|day|hour)"
    r"|what\s+time\s+is\s+it"
    r"|what\s+day\s+is\s+(?:it|today)"
    r"|what(?:'s| is)\s+today"
    r"|tell\s+me\s+the\s+(?:time|date)"
    r")\s*[?!.]*\s*$",
    re.IGNORECASE,
)

# Hard cap on word count for the trivial classifier to even look at the
# message. Anything longer almost certainly has enough content to benefit
# from full context.
_TRIVIAL_MAX_WORDS = 6


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
