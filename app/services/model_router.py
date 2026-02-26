"""
Model Auto-Router — Heuristic-based model selection for cost optimization.

Classifies each user request by complexity/intent and routes to the
cheapest model that can handle it well.

Tiers:
  - light:   Simple greetings, short Q&A, chit-chat, single-fact lookups  → Claude Sonnet 4.5
  - medium:  Multi-step tasks, summarization, moderate reasoning, tool use  → Claude Opus 4.6
  - heavy:   Complex coding, multi-file edits, deep analysis, planning,
             long-form generation, math proofs, multi-tool orchestration   → Claude Opus 4.6

The router uses fast heuristics (no LLM call) to classify in <1ms.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)


def _is_claude_model(model: str) -> bool:
    """Check if a model name refers to an Anthropic Claude model."""
    return model.startswith("claude-")


# ── Model Tiers ──────────────────────────────────────────────────────

@dataclass
class ModelTier:
    """A model tier with its model ID and cost characteristics."""
    name: str           # "light", "medium", "heavy"
    model: str          # actual model ID
    label: str          # display label
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0


# Default tier configs (can be overridden via settings)
DEFAULT_TIERS: Dict[str, ModelTier] = {
    "light": ModelTier(
        name="light",
        model="claude-sonnet-4-5",
        label="Claude Sonnet 4.5",
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
    ),
    "medium": ModelTier(
        name="medium",
        model="claude-opus-4-6",
        label="Claude Opus 4.6",
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
    ),
    "heavy": ModelTier(
        name="heavy",
        model="claude-opus-4-6",
        label="Claude Opus 4.6",
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
    ),
}


# ── Complexity Signals ───────────────────────────────────────────────

# Patterns that suggest heavy/complex tasks
HEAVY_PATTERNS = [
    # Coding
    r"\b(implement|refactor|debug|fix\s+the\s+bug|write\s+(?:a\s+)?(?:full|complete)\s+(?:program|app|script|function|class))\b",
    r"\b(build\s+(?:a|an|the)\s+\w+|create\s+(?:a|an)\s+(?:project|app|api|server|system))\b",
    r"\b(multi[- ]?file|codebase|architecture|design\s+pattern|system\s+design)\b",
    # Analysis / reasoning
    r"\b(analyze|compare\s+and\s+contrast|evaluate|critique|pros?\s+and\s+cons?)\b",
    r"\b(step[- ]?by[- ]?step|chain\s+of\s+thought|think\s+through|reason\s+about)\b",
    r"\b(prove|derive|theorem|mathematical|equation|integral|differential)\b",
    # Planning / multi-step
    r"\b(plan|strategy|roadmap|breakdown|decompose|outline\s+a\s+(?:plan|strategy))\b",
    r"\b(research\s+(?:and|&)\s+(?:summarize|write)|write\s+(?:a|an)\s+(?:essay|report|article|paper))\b",
    # Long-form
    r"\b(comprehensive|detailed|thorough|in[- ]?depth|exhaustive)\b",
    # Multi-tool
    r"\b(search\s+(?:the\s+web|online)\s+(?:and|then)|fetch\s+(?:and|then)|download\s+(?:and|then))\b",
]

# Patterns that suggest light/simple tasks
LIGHT_PATTERNS = [
    # Greetings
    r"^(?:hi|hello|hey|sup|yo|morning|evening|night|salam|سلام|hola|bonjour)[\s!?.]*$",
    # Simple questions
    r"^(?:what(?:'s| is)\s+(?:your\s+name|the\s+(?:time|date|weather)))",
    r"^(?:who\s+are\s+you|how\s+are\s+you|what\s+can\s+you\s+do)",
    # Short commands
    r"^(?:thanks?|thank\s+you|ok|okay|got\s+it|cool|nice|great|awesome|bye|goodbye)[\s!?.]*$",
    # Yes/No
    r"^(?:yes|no|yeah|nah|yep|nope|sure|of\s+course)[\s!?.]*$",
    # Simple translations / definitions
    r"^(?:translate|define|what\s+does\s+\w+\s+mean)",
    # Status / simple lookups  
    r"^(?:show|list|get)\s+(?:my\s+)?(?:memories|sessions|status|stats)",
]

# Tool-heavy keywords (medium → heavy)
TOOL_KEYWORDS = [
    "exec", "execute", "run", "shell", "terminal", "command",
    "file", "read", "write", "edit", "create", "delete",
    "search", "fetch", "download", "web", "scrape", "crawl",
    "cron", "schedule", "spawn", "subagent",
]

# Code indicators (bump complexity)
CODE_INDICATORS = [
    r"```",                           # code blocks
    r"\b(?:def|class|function|const|let|var|import|from|async|await)\b",
    r"\b(?:python|javascript|typescript|react|node|sql|html|css|rust|go)\b",
]


@dataclass
class RoutingDecision:
    """Result of the auto-routing classification."""
    tier: str                       # "light", "medium", "heavy"
    model: str                      # actual model ID to use
    label: str                      # human-readable label
    confidence: float               # 0.0 - 1.0
    reason: str                     # short explanation
    signals: Dict[str, float] = field(default_factory=dict)


def classify_request(
    user_message: str,
    conversation_history: Optional[List[Dict]] = None,
    has_media: bool = False,
) -> RoutingDecision:
    """
    Classify a user request and select the best model tier.
    
    Uses fast heuristics — no LLM call, runs in <1ms.
    
    Scoring approach:
      - Start with a base score of 0
      - Add/subtract based on message characteristics
      - Map final score to tier: <30 = light, 30-65 = medium, >65 = heavy
    """
    if not user_message.strip():
        return RoutingDecision(
            tier="light",
            model=DEFAULT_TIERS["light"].model,
            label=DEFAULT_TIERS["light"].label,
            confidence=0.95,
            reason="Empty message",
        )

    text = user_message.strip()
    text_lower = text.lower()
    score = 0.0
    signals: Dict[str, float] = {}

    # ── 1. Length signal ──────────────────────────────────────────
    word_count = len(text.split())
    char_count = len(text)

    if word_count <= 5:
        signals["length"] = -10
    elif word_count <= 20:
        signals["length"] = 5
    elif word_count <= 80:
        signals["length"] = 15
    elif word_count <= 200:
        signals["length"] = 25
    else:
        signals["length"] = 35

    # ── 2. Light pattern matching ─────────────────────────────────
    for pattern in LIGHT_PATTERNS:
        if re.search(pattern, text_lower):
            signals["light_pattern"] = -30
            break

    # ── 3. Heavy pattern matching ─────────────────────────────────
    heavy_matches = 0
    for pattern in HEAVY_PATTERNS:
        if re.search(pattern, text_lower):
            heavy_matches += 1
    if heavy_matches > 0:
        signals["heavy_pattern"] = min(heavy_matches * 20, 60)

    # ── 4. Tool-related keywords ──────────────────────────────────
    tool_hits = sum(1 for kw in TOOL_KEYWORDS if kw in text_lower)
    if tool_hits > 0:
        signals["tool_keywords"] = min(tool_hits * 8, 25)

    # ── 5. Code indicators ────────────────────────────────────────
    code_hits = 0
    for pattern in CODE_INDICATORS:
        if re.search(pattern, text):
            code_hits += 1
    if code_hits > 0:
        signals["code_indicators"] = min(code_hits * 10, 30)

    # ── 6. Question complexity ────────────────────────────────────
    question_marks = text.count("?")
    if question_marks > 2:
        signals["multi_question"] = 15

    # Numbered lists / multiple requirements
    numbered = len(re.findall(r"^\s*\d+[\.\)]\s", text, re.MULTILINE))
    bullet_points = len(re.findall(r"^\s*[-•*]\s", text, re.MULTILINE))
    constraints = numbered + bullet_points
    if constraints >= 3:
        signals["structured_request"] = min(constraints * 8, 30)

    # ── 7. Conversation context ───────────────────────────────────
    if conversation_history:
        history_len = len(conversation_history)
        if history_len > 15:
            signals["long_conversation"] = 10
        # Check if previous messages involved tool use
        recent_tools = sum(
            1 for msg in conversation_history[-5:]
            if isinstance(msg.get("content"), list)
            and any(
                isinstance(c, dict) and c.get("type") in ("tool_use", "tool_result")
                for c in msg["content"]
            )
        )
        if recent_tools > 0:
            signals["recent_tool_use"] = 15

    # ── 8. Media attachment ───────────────────────────────────────
    if has_media:
        signals["has_media"] = 15

    # ── 9. Explicit quality requests ──────────────────────────────
    if re.search(r"\b(best|highest\s+quality|most\s+accurate|be\s+careful|double[- ]check)\b", text_lower):
        signals["quality_request"] = 15

    # ── 10. Language signal (non-English may need better model) ────
    non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / max(len(text), 1)
    if non_ascii_ratio > 0.3:
        signals["non_english"] = 10

    # ── Sum up ────────────────────────────────────────────────────
    score = sum(signals.values())

    # ── Map to tier ───────────────────────────────────────────────
    if score < 15:
        tier_name = "light"
        confidence = min(0.95, 0.7 + (15 - score) / 100)
    elif score < 50:
        tier_name = "medium"
        confidence = min(0.9, 0.6 + abs(score - 32.5) / 100)
    else:
        tier_name = "heavy"
        confidence = min(0.95, 0.7 + (score - 50) / 100)

    tier = DEFAULT_TIERS[tier_name]

    # If the selected tier uses a Claude model but no Anthropic key is set,
    # fall back to the configured agent_model (OpenAI)
    if _is_claude_model(tier.model) and not settings.anthropic_api_key:
        fallback_model = settings.agent_model
        logger.info(f"[ROUTER] No Anthropic API key — overriding {tier.model} → {fallback_model}")
        tier = ModelTier(
            name=tier_name,
            model=fallback_model,
            label=f"{fallback_model} (fallback)",
        )

    # Build reason
    top_signals = sorted(signals.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    reason_parts = [f"{k}={v:+.0f}" for k, v in top_signals if v != 0]
    reason = f"score={score:.0f} → {tier_name} ({', '.join(reason_parts)})"

    decision = RoutingDecision(
        tier=tier_name,
        model=tier.model,
        label=tier.label,
        confidence=confidence,
        reason=reason,
        signals=signals,
    )

    logger.info(f"[ROUTER] {reason} → {tier.model}")

    return decision


def get_model_for_auto(
    user_message: str,
    conversation_history: Optional[List[Dict]] = None,
    has_media: bool = False,
) -> str:
    """
    Convenience function: classify and return just the model ID.
    Used when model_override is "auto".
    """
    decision = classify_request(user_message, conversation_history, has_media)
    return decision.model
