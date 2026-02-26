"""
Multi-Agent Routing — Route messages to specialized agent personas.

Each agent definition is a named configuration that can override:
- System prompt
- Model
- Tool allow/deny lists
- Memory scope
- Temperature

The router examines incoming messages and delegates to the best-fit
agent based on keyword rules, regex patterns, or LLM classification.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentPersona:
    """A named agent configuration."""
    name: str
    description: str = ""
    system_prompt_prefix: str = ""  # Prepended to default system prompt
    model_override: Optional[str] = None
    temperature: float = 0.7
    tool_allow: Optional[List[str]] = None  # If set, ONLY these tools are available
    tool_deny: Optional[List[str]] = None   # If set, these tools are blocked
    keywords: List[str] = field(default_factory=list)  # Trigger keywords
    patterns: List[str] = field(default_factory=list)   # Regex patterns
    priority: int = 0  # Higher = checked first


class MultiAgentRouter:
    """
    Routes incoming messages to the best-matching agent persona.

    Default routing uses keyword + regex matching. For complex
    scenarios, register a custom classifier function.
    """

    def __init__(self):
        self._personas: Dict[str, AgentPersona] = {}
        self._classifier: Optional[Callable] = None
        self._default: str = "default"

        # Register the default persona
        self.register(AgentPersona(
            name="default",
            description="General-purpose HexBrain agent",
        ))

    def register(self, persona: AgentPersona) -> None:
        """Register a new agent persona."""
        self._personas[persona.name] = persona
        logger.info(f"[MULTI-AGENT] Registered persona: {persona.name}")

    def unregister(self, name: str) -> bool:
        """Remove a persona. Cannot remove 'default'."""
        if name == "default":
            return False
        return self._personas.pop(name, None) is not None

    def set_classifier(self, fn: Callable) -> None:
        """Set a custom classifier function: (message, personas) -> persona_name."""
        self._classifier = fn

    def set_default(self, name: str) -> None:
        """Set the default persona name (fallback when no match)."""
        if name in self._personas:
            self._default = name

    def route(self, message: str, context: Optional[Dict[str, Any]] = None) -> AgentPersona:
        """
        Determine which persona should handle a message.

        Priority:
          1. Custom classifier (if set)
          2. Keyword/regex matching (highest priority first)
          3. Default persona
        """
        # 1. Custom classifier
        if self._classifier:
            try:
                name = self._classifier(message, self._personas)
                if name and name in self._personas:
                    logger.debug(f"[MULTI-AGENT] Classifier routed to: {name}")
                    return self._personas[name]
            except Exception as e:
                logger.warning(f"[MULTI-AGENT] Classifier error: {e}")

        # 2. Keyword / regex matching
        msg_lower = message.lower()
        candidates = sorted(
            [p for p in self._personas.values() if p.name != self._default],
            key=lambda p: p.priority,
            reverse=True,
        )

        for persona in candidates:
            # Check keywords
            for kw in persona.keywords:
                if kw.lower() in msg_lower:
                    logger.debug(f"[MULTI-AGENT] Keyword match '{kw}' → {persona.name}")
                    return persona

            # Check regex patterns
            for pat in persona.patterns:
                try:
                    if re.search(pat, message, re.IGNORECASE):
                        logger.debug(f"[MULTI-AGENT] Pattern match '{pat}' → {persona.name}")
                        return persona
                except re.error:
                    pass

        # 3. Default
        return self._personas.get(self._default, self._personas["default"])

    def list_personas(self) -> List[Dict[str, Any]]:
        """List all registered personas."""
        return [
            {
                "name": p.name,
                "description": p.description,
                "model": p.model_override,
                "priority": p.priority,
                "keywords": p.keywords,
                "tool_allow": p.tool_allow,
                "tool_deny": p.tool_deny,
            }
            for p in sorted(self._personas.values(), key=lambda x: x.priority, reverse=True)
        ]

    def get(self, name: str) -> Optional[AgentPersona]:
        """Get a persona by name."""
        return self._personas.get(name)

    @property
    def count(self) -> int:
        return len(self._personas)


# ── Singleton ────────────────────────────────────────────────
_router: Optional[MultiAgentRouter] = None


def get_multi_agent_router() -> MultiAgentRouter:
    global _router
    if _router is None:
        _router = MultiAgentRouter()

        # Register built-in personas
        _router.register(AgentPersona(
            name="coder",
            description="Software engineering specialist",
            system_prompt_prefix="You are in CODER mode. Focus on writing clean, production-quality code. Prefer tool usage over explanations.",
            keywords=["code", "implement", "function", "class", "refactor", "debug", "fix bug"],
            patterns=[r"\b(def |class |import |async def)\b", r"```"],
            priority=5,
        ))

        _router.register(AgentPersona(
            name="researcher",
            description="Research and analysis specialist",
            system_prompt_prefix="You are in RESEARCHER mode. Provide thorough, well-sourced analysis. Use web search when needed.",
            keywords=["research", "analyze", "compare", "explain", "summarize"],
            patterns=[r"\bwhy\b.*\?$", r"\bhow does\b.*\?$"],
            priority=3,
        ))

        _router.register(AgentPersona(
            name="writer",
            description="Creative and technical writing specialist",
            system_prompt_prefix="You are in WRITER mode. Focus on clear, engaging prose. Adapt tone to the request.",
            keywords=["write", "draft", "essay", "blog", "email", "letter"],
            priority=2,
        ))

    return _router
