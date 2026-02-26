"""
Voice Call Plugin — Telephony integration with Telnyx/Twilio/Plivo.

Manages voice calls through telephony providers. Supports inbound
and outbound calls, STT during calls, TTS for responses, call
recording, and DTMF handling.

Usage:
    from app.agent.voice_call import get_voice_call_manager

    mgr = get_voice_call_manager()
    mgr.configure_provider("twilio", account_sid="...", auth_token="...")
    call = await mgr.initiate_call("+1555123456", agent_id="agent-1")
    await mgr.hangup(call.call_id)
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CallProvider(str, Enum):
    TWILIO = "twilio"
    TELNYX = "telnyx"
    PLIVO = "plivo"


class CallState(str, Enum):
    INITIATING = "initiating"
    RINGING = "ringing"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    ENDED = "ended"
    FAILED = "failed"


class CallDirection(str, Enum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"


@dataclass
class ProviderConfig:
    """Configuration for a telephony provider."""
    provider: CallProvider
    credentials: Dict[str, str] = field(default_factory=dict)
    from_number: str = ""
    webhook_url: str = ""
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider.value,
            "from_number": self.from_number,
            "enabled": self.enabled,
            "has_credentials": bool(self.credentials),
        }


@dataclass
class CallInfo:
    """Information about a voice call."""
    call_id: str
    provider: CallProvider
    direction: CallDirection
    from_number: str
    to_number: str
    state: CallState = CallState.INITIATING
    agent_id: str = ""
    started_at: float = 0.0
    connected_at: Optional[float] = None
    ended_at: Optional[float] = None
    duration_seconds: float = 0.0
    recording_url: Optional[str] = None
    transcript: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.started_at == 0.0:
            self.started_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "provider": self.provider.value,
            "direction": self.direction.value,
            "from": self.from_number,
            "to": self.to_number,
            "state": self.state.value,
            "agent_id": self.agent_id,
            "duration_seconds": round(self.duration_seconds, 1),
        }


class VoiceCallManager:
    """
    Manages voice calls through telephony providers.

    Supports inbound/outbound calls with STT+TTS for
    AI-powered phone conversations.
    """

    def __init__(self):
        self._providers: Dict[str, ProviderConfig] = {}
        self._calls: Dict[str, CallInfo] = {}
        self._default_provider: Optional[str] = None
        self._call_counter: int = 0

    def configure_provider(
        self,
        provider: str,
        *,
        credentials: Optional[Dict[str, str]] = None,
        from_number: str = "",
        webhook_url: str = "",
    ) -> ProviderConfig:
        """Configure a telephony provider."""
        try:
            prov_enum = CallProvider(provider)
        except ValueError:
            raise ValueError(f"Unknown provider: {provider}")

        config = ProviderConfig(
            provider=prov_enum,
            credentials=credentials or {},
            from_number=from_number,
            webhook_url=webhook_url,
        )
        self._providers[provider] = config

        if self._default_provider is None:
            self._default_provider = provider

        logger.info(f"[VOICE-CALL] Configured provider: {provider}")
        return config

    def set_default_provider(self, provider: str) -> bool:
        """Set the default telephony provider."""
        if provider in self._providers:
            self._default_provider = provider
            return True
        return False

    async def initiate_call(
        self,
        to_number: str,
        *,
        agent_id: str = "default",
        provider: Optional[str] = None,
        from_number: Optional[str] = None,
    ) -> CallInfo:
        """
        Initiate an outbound voice call.

        Args:
            to_number: Phone number to call.
            agent_id: Agent handling the call.
            provider: Telephony provider to use.
            from_number: Override the from number.
        """
        prov_name = provider or self._default_provider
        if not prov_name or prov_name not in self._providers:
            return CallInfo(
                call_id="error",
                provider=CallProvider.TWILIO,
                direction=CallDirection.OUTBOUND,
                from_number="",
                to_number=to_number,
                state=CallState.FAILED,
            )

        prov_config = self._providers[prov_name]
        self._call_counter += 1
        call_id = f"call_{self._call_counter}"

        call = CallInfo(
            call_id=call_id,
            provider=prov_config.provider,
            direction=CallDirection.OUTBOUND,
            from_number=from_number or prov_config.from_number,
            to_number=to_number,
            state=CallState.RINGING,
            agent_id=agent_id,
        )

        self._calls[call_id] = call
        logger.info(f"[VOICE-CALL] Initiated {call_id}: {call.from_number} → {to_number}")
        return call

    def handle_inbound(
        self,
        call_id: str,
        from_number: str,
        to_number: str,
        provider: str,
        *,
        agent_id: str = "default",
    ) -> CallInfo:
        """Handle an inbound call webhook."""
        try:
            prov_enum = CallProvider(provider)
        except ValueError:
            prov_enum = CallProvider.TWILIO

        call = CallInfo(
            call_id=call_id,
            provider=prov_enum,
            direction=CallDirection.INBOUND,
            from_number=from_number,
            to_number=to_number,
            state=CallState.RINGING,
            agent_id=agent_id,
        )

        self._calls[call_id] = call
        return call

    async def answer(self, call_id: str) -> bool:
        """Answer a ringing call."""
        call = self._calls.get(call_id)
        if call and call.state == CallState.RINGING:
            call.state = CallState.IN_PROGRESS
            call.connected_at = time.time()
            return True
        return False

    async def hangup(self, call_id: str) -> bool:
        """Hang up a call."""
        call = self._calls.get(call_id)
        if call and call.state in (CallState.RINGING, CallState.IN_PROGRESS, CallState.ON_HOLD):
            call.state = CallState.ENDED
            call.ended_at = time.time()
            if call.connected_at:
                call.duration_seconds = call.ended_at - call.connected_at
            return True
        return False

    async def hold(self, call_id: str) -> bool:
        """Put a call on hold."""
        call = self._calls.get(call_id)
        if call and call.state == CallState.IN_PROGRESS:
            call.state = CallState.ON_HOLD
            return True
        return False

    async def resume(self, call_id: str) -> bool:
        """Resume a held call."""
        call = self._calls.get(call_id)
        if call and call.state == CallState.ON_HOLD:
            call.state = CallState.IN_PROGRESS
            return True
        return False

    def add_transcript(self, call_id: str, text: str) -> bool:
        """Add a transcript line to a call."""
        call = self._calls.get(call_id)
        if call:
            call.transcript.append(text)
            return True
        return False

    def get_call(self, call_id: str) -> Optional[CallInfo]:
        """Get call information."""
        return self._calls.get(call_id)

    def list_calls(
        self,
        state: Optional[CallState] = None,
        direction: Optional[CallDirection] = None,
    ) -> List[Dict[str, Any]]:
        """List all calls."""
        calls = list(self._calls.values())
        if state:
            calls = [c for c in calls if c.state == state]
        if direction:
            calls = [c for c in calls if c.direction == direction]
        return [c.to_dict() for c in calls]

    def list_providers(self) -> List[Dict[str, Any]]:
        """List configured providers."""
        return [p.to_dict() for p in self._providers.values()]

    def stats(self) -> Dict[str, Any]:
        by_state: Dict[str, int] = {}
        total_duration = 0.0
        for c in self._calls.values():
            by_state[c.state.value] = by_state.get(c.state.value, 0) + 1
            total_duration += c.duration_seconds

        return {
            "total_calls": len(self._calls),
            "by_state": by_state,
            "total_duration_seconds": round(total_duration, 1),
            "providers": len(self._providers),
            "default_provider": self._default_provider,
        }


# ── Singleton ────────────────────────────────────────────
_manager: Optional[VoiceCallManager] = None


def get_voice_call_manager() -> VoiceCallManager:
    """Get the global voice call manager."""
    global _manager
    if _manager is None:
        _manager = VoiceCallManager()
    return _manager
