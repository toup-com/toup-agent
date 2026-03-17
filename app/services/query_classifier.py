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
