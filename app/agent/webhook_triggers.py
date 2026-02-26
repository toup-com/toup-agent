"""
Webhook Triggers — HTTP endpoints that trigger agent runs.

Supports:
- POST /hooks/agent — Run the agent with a message (like CLI send)
- POST /hooks/ingest — Ingest a document via webhook
- Custom match rules for filtering incoming webhooks

Usage:
    from app.agent.webhook_triggers import get_webhook_manager

    mgr = get_webhook_manager()
    mgr.register_rule("github_push", match={"event": "push"}, action="agent")
    result = await mgr.process({"event": "push", "repo": "my-repo"})
"""

import hashlib
import hmac
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WebhookRule:
    """A rule that matches incoming webhook payloads and triggers an action."""
    name: str
    match: Dict[str, Any]  # Key-value conditions to match against payload
    action: str = "agent"  # "agent" | "ingest" | "event" | "custom"
    agent_prompt_template: str = ""  # Template for agent message: "New {event} from {source}"
    target_event: str = ""  # For action="event", the wake event to fire
    enabled: bool = True
    priority: int = 0
    created_at: float = 0.0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    def matches(self, payload: Dict[str, Any]) -> bool:
        """Check if a payload matches this rule."""
        for key, expected in self.match.items():
            actual = payload.get(key)
            if actual is None:
                return False
            if isinstance(expected, str) and expected.startswith("regex:"):
                pattern = expected[6:]
                if not re.search(pattern, str(actual), re.IGNORECASE):
                    return False
            elif actual != expected:
                return False
        return True

    def build_prompt(self, payload: Dict[str, Any]) -> str:
        """Build the agent prompt from template and payload."""
        if not self.agent_prompt_template:
            import json
            return f"Webhook received: {json.dumps(payload, default=str)[:2000]}"

        prompt = self.agent_prompt_template
        for key, value in payload.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        return prompt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "match": self.match,
            "action": self.action,
            "agent_prompt_template": self.agent_prompt_template,
            "target_event": self.target_event,
            "enabled": self.enabled,
            "priority": self.priority,
        }


@dataclass
class WebhookResult:
    """Result of processing a webhook."""
    rule_name: str
    action: str
    success: bool
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule_name,
            "action": self.action,
            "success": self.success,
            "message": self.message,
            "data": self.data,
        }


class WebhookManager:
    """
    Manages webhook rules and processes incoming payloads.

    When a webhook arrives, it's matched against all rules (highest
    priority first). Matching rules trigger their configured action.
    """

    def __init__(self):
        self._rules: Dict[str, WebhookRule] = {}
        self._secrets: Dict[str, str] = {}  # name → signing secret
        self._handlers: Dict[str, Callable] = {}  # custom action handlers
        self._history: List[Dict[str, Any]] = []
        self._max_history: int = 100

    def register_rule(
        self,
        name: str,
        match: Dict[str, Any],
        action: str = "agent",
        *,
        prompt_template: str = "",
        target_event: str = "",
        priority: int = 0,
    ) -> WebhookRule:
        """Register a webhook matching rule."""
        rule = WebhookRule(
            name=name,
            match=match,
            action=action,
            agent_prompt_template=prompt_template,
            target_event=target_event,
            priority=priority,
        )
        self._rules[name] = rule
        logger.info(f"[WEBHOOK] Registered rule: {name} (action={action})")
        return rule

    def unregister_rule(self, name: str) -> bool:
        """Remove a webhook rule."""
        return self._rules.pop(name, None) is not None

    def set_secret(self, name: str, secret: str) -> None:
        """Set a signing secret for webhook verification."""
        self._secrets[name] = secret

    def verify_signature(
        self,
        name: str,
        payload_bytes: bytes,
        signature: str,
        algorithm: str = "sha256",
    ) -> bool:
        """Verify a webhook signature using HMAC."""
        secret = self._secrets.get(name)
        if not secret:
            return True  # No secret configured, skip verification

        if algorithm == "sha256":
            expected = hmac.new(
                secret.encode(), payload_bytes, hashlib.sha256
            ).hexdigest()
        elif algorithm == "sha1":
            expected = hmac.new(
                secret.encode(), payload_bytes, hashlib.sha1
            ).hexdigest()
        else:
            return False

        # Strip prefix like "sha256=" if present
        if "=" in signature:
            signature = signature.split("=", 1)[1]

        return hmac.compare_digest(expected, signature)

    def register_custom_handler(self, action_name: str, handler: Callable) -> None:
        """Register a custom action handler."""
        self._handlers[action_name] = handler

    async def process(self, payload: Dict[str, Any]) -> List[WebhookResult]:
        """
        Process an incoming webhook payload against all rules.

        Returns list of results for each matching rule.
        """
        results = []
        sorted_rules = sorted(
            [r for r in self._rules.values() if r.enabled],
            key=lambda r: r.priority,
            reverse=True,
        )

        for rule in sorted_rules:
            if not rule.matches(payload):
                continue

            try:
                if rule.action == "agent":
                    prompt = rule.build_prompt(payload)
                    result = WebhookResult(
                        rule_name=rule.name,
                        action="agent",
                        success=True,
                        message=prompt,
                        data={"prompt": prompt},
                    )
                elif rule.action == "event":
                    result = WebhookResult(
                        rule_name=rule.name,
                        action="event",
                        success=True,
                        message=f"Would fire event: {rule.target_event}",
                        data={"event": rule.target_event, "payload": payload},
                    )
                elif rule.action in self._handlers:
                    handler = self._handlers[rule.action]
                    handler_result = await handler(payload, rule)
                    result = WebhookResult(
                        rule_name=rule.name,
                        action=rule.action,
                        success=True,
                        data=handler_result if isinstance(handler_result, dict) else {},
                    )
                else:
                    result = WebhookResult(
                        rule_name=rule.name,
                        action=rule.action,
                        success=False,
                        message=f"No handler for action: {rule.action}",
                    )

                results.append(result)

            except Exception as e:
                logger.error(f"[WEBHOOK] Rule {rule.name} failed: {e}")
                results.append(WebhookResult(
                    rule_name=rule.name,
                    action=rule.action,
                    success=False,
                    message=str(e),
                ))

        # Record in history
        self._history.append({
            "payload_keys": list(payload.keys()),
            "matched_rules": [r.rule_name for r in results],
            "timestamp": time.time(),
        })
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return results

    def list_rules(self) -> List[Dict[str, Any]]:
        """List all registered webhook rules."""
        return [r.to_dict() for r in self._rules.values()]

    def get_rule(self, name: str) -> Optional[WebhookRule]:
        """Get a specific rule by name."""
        return self._rules.get(name)

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent webhook processing history."""
        return self._history[-limit:]

    @property
    def rule_count(self) -> int:
        return len(self._rules)


# ── Singleton ────────────────────────────────────────────
_manager: Optional[WebhookManager] = None


def get_webhook_manager() -> WebhookManager:
    """Get the global webhook manager."""
    global _manager
    if _manager is None:
        _manager = WebhookManager()
    return _manager
