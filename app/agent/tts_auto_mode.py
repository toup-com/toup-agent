"""
TTS Auto-Mode — Automatic text-to-speech mode control.

Controls when the agent automatically converts text replies to voice:
  - off:      Never auto-TTS (user must explicitly request)
  - always:   Every reply gets TTS
  - inbound:  TTS only if the user sent a voice message
  - tagged:   TTS only if the model tags the reply with [voice:...]

Per-session and per-channel auto-mode settings.

Usage:
    from app.agent.tts_auto_mode import get_tts_mode_manager

    mgr = get_tts_mode_manager()
    mgr.set_mode("s1", TTSAutoMode.INBOUND)
    mode = mgr.get_mode("s1")  # TTSAutoMode.INBOUND

    should_speak = mgr.should_tts("s1", inbound_is_voice=True)  # True
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TTSAutoMode(str, Enum):
    OFF = "off"
    ALWAYS = "always"
    INBOUND = "inbound"
    TAGGED = "tagged"


@dataclass
class TTSModeConfig:
    """TTS mode configuration for a session."""
    session_id: str
    mode: TTSAutoMode = TTSAutoMode.OFF
    voice: str = "alloy"
    speed: float = 1.0
    provider: str = "openai"
    last_changed: float = 0.0
    tts_count: int = 0
    skip_count: int = 0

    def __post_init__(self):
        if self.last_changed == 0.0:
            self.last_changed = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "mode": self.mode.value,
            "voice": self.voice,
            "speed": self.speed,
            "provider": self.provider,
            "tts_count": self.tts_count,
            "skip_count": self.skip_count,
        }


class TTSModeManager:
    """
    Manages TTS auto-mode settings per session.

    Controls whether agent replies are automatically converted
    to voice based on the configured mode.
    """

    def __init__(self, default_mode: TTSAutoMode = TTSAutoMode.OFF):
        self._configs: Dict[str, TTSModeConfig] = {}
        self._default_mode = default_mode
        self._channel_modes: Dict[str, TTSAutoMode] = {}

    @property
    def default_mode(self) -> TTSAutoMode:
        return self._default_mode

    @default_mode.setter
    def default_mode(self, value: TTSAutoMode):
        self._default_mode = value

    def set_mode(
        self,
        session_id: str,
        mode: TTSAutoMode,
        *,
        voice: str = "alloy",
        speed: float = 1.0,
        provider: str = "openai",
    ) -> TTSModeConfig:
        """Set TTS mode for a session."""
        config = TTSModeConfig(
            session_id=session_id,
            mode=mode,
            voice=voice,
            speed=speed,
            provider=provider,
        )
        self._configs[session_id] = config
        logger.info(f"[TTS-MODE] Session {session_id}: {mode.value}")
        return config

    def get_mode(self, session_id: str) -> TTSAutoMode:
        """Get the TTS mode for a session."""
        config = self._configs.get(session_id)
        if config:
            return config.mode
        return self._default_mode

    def get_config(self, session_id: str) -> Optional[TTSModeConfig]:
        """Get full TTS config for a session."""
        return self._configs.get(session_id)

    def set_channel_mode(self, channel_type: str, mode: TTSAutoMode) -> None:
        """Set default TTS mode for a channel type."""
        self._channel_modes[channel_type] = mode

    def get_channel_mode(self, channel_type: str) -> TTSAutoMode:
        """Get TTS mode for a channel type."""
        return self._channel_modes.get(channel_type, self._default_mode)

    def should_tts(
        self,
        session_id: str,
        *,
        inbound_is_voice: bool = False,
        has_voice_tag: bool = False,
        channel_type: str = "",
    ) -> bool:
        """
        Determine if a reply should be converted to TTS.

        Args:
            session_id: The session ID.
            inbound_is_voice: Whether the user's message was a voice message.
            has_voice_tag: Whether the model tagged the reply with [voice:...].
            channel_type: The channel type (for channel-level defaults).
        """
        config = self._configs.get(session_id)
        mode = config.mode if config else self._default_mode

        # Check channel-level override
        if not config and channel_type:
            mode = self._channel_modes.get(channel_type, mode)

        result = False

        if mode == TTSAutoMode.OFF:
            result = False
        elif mode == TTSAutoMode.ALWAYS:
            result = True
        elif mode == TTSAutoMode.INBOUND:
            result = inbound_is_voice
        elif mode == TTSAutoMode.TAGGED:
            result = has_voice_tag

        # Track stats
        if config:
            if result:
                config.tts_count += 1
            else:
                config.skip_count += 1

        return result

    def cycle_mode(self, session_id: str) -> TTSAutoMode:
        """Cycle through TTS modes: off → inbound → always → tagged → off."""
        order = [TTSAutoMode.OFF, TTSAutoMode.INBOUND, TTSAutoMode.ALWAYS, TTSAutoMode.TAGGED]
        current = self.get_mode(session_id)
        idx = order.index(current) if current in order else 0
        next_mode = order[(idx + 1) % len(order)]

        config = self._configs.get(session_id)
        if config:
            config.mode = next_mode
            config.last_changed = time.time()
        else:
            self.set_mode(session_id, next_mode)

        return next_mode

    def remove_config(self, session_id: str) -> bool:
        """Remove session TTS config (revert to default)."""
        return self._configs.pop(session_id, None) is not None

    def list_configs(self) -> List[Dict[str, Any]]:
        """List all session TTS configs."""
        return [c.to_dict() for c in self._configs.values()]

    def stats(self) -> Dict[str, Any]:
        """Get TTS mode statistics."""
        by_mode: Dict[str, int] = {}
        total_tts = 0
        total_skip = 0
        for c in self._configs.values():
            by_mode[c.mode.value] = by_mode.get(c.mode.value, 0) + 1
            total_tts += c.tts_count
            total_skip += c.skip_count

        return {
            "total_configs": len(self._configs),
            "default_mode": self._default_mode.value,
            "by_mode": by_mode,
            "total_tts_generated": total_tts,
            "total_skipped": total_skip,
            "channel_modes": {k: v.value for k, v in self._channel_modes.items()},
        }


# ── Singleton ────────────────────────────────────────────
_manager: Optional[TTSModeManager] = None


def get_tts_mode_manager() -> TTSModeManager:
    """Get the global TTS mode manager."""
    global _manager
    if _manager is None:
        _manager = TTSModeManager()
    return _manager
