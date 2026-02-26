"""
TTS Preferences — Per-user voice and speech preferences.

Stores and retrieves TTS preferences (voice, speed, provider,
language) per user across sessions.

Usage:
    from app.agent.tts_preferences import get_tts_prefs_manager

    mgr = get_tts_prefs_manager()
    mgr.set_preferences("user_123", voice="nova", speed=1.2)
    prefs = mgr.get_preferences("user_123")
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Available voices per provider
AVAILABLE_VOICES: Dict[str, List[str]] = {
    "openai": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
    "elevenlabs": ["rachel", "drew", "clyde", "paul", "domi", "bella", "antoni", "elli", "josh", "arnold", "adam", "sam"],
    "edge": ["en-US-AriaNeural", "en-US-GuyNeural", "en-US-JennyNeural", "en-GB-SoniaNeural"],
}


@dataclass
class TTSPreferences:
    """TTS preferences for a user."""
    user_id: str
    voice: str = "alloy"
    speed: float = 1.0
    provider: str = "openai"
    language: str = "en"
    pitch: float = 1.0
    volume: float = 1.0
    auto_tts: bool = False
    updated_at: float = 0.0

    def __post_init__(self):
        if self.updated_at == 0.0:
            self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "voice": self.voice,
            "speed": self.speed,
            "provider": self.provider,
            "language": self.language,
            "pitch": self.pitch,
            "volume": self.volume,
            "auto_tts": self.auto_tts,
        }


class TTSPreferencesManager:
    """
    Manages per-user TTS preferences.

    Stores voice, speed, provider, and language preferences
    that persist across sessions for each user.
    """

    def __init__(self):
        self._prefs: Dict[str, TTSPreferences] = {}
        self._defaults = TTSPreferences(user_id="__default__")

    def set_preferences(
        self,
        user_id: str,
        *,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        provider: Optional[str] = None,
        language: Optional[str] = None,
        pitch: Optional[float] = None,
        volume: Optional[float] = None,
        auto_tts: Optional[bool] = None,
    ) -> TTSPreferences:
        """Set or update TTS preferences for a user."""
        prefs = self._prefs.get(user_id)
        if not prefs:
            prefs = TTSPreferences(user_id=user_id)
            self._prefs[user_id] = prefs

        if voice is not None:
            prefs.voice = voice
        if speed is not None:
            prefs.speed = max(0.25, min(4.0, speed))
        if provider is not None:
            prefs.provider = provider
        if language is not None:
            prefs.language = language
        if pitch is not None:
            prefs.pitch = max(0.5, min(2.0, pitch))
        if volume is not None:
            prefs.volume = max(0.0, min(1.0, volume))
        if auto_tts is not None:
            prefs.auto_tts = auto_tts

        prefs.updated_at = time.time()
        logger.info(f"[TTS-PREFS] Updated for {user_id}: voice={prefs.voice}, speed={prefs.speed}")
        return prefs

    def get_preferences(self, user_id: str) -> TTSPreferences:
        """Get TTS preferences for a user (returns defaults if not set)."""
        return self._prefs.get(user_id, self._defaults)

    def has_preferences(self, user_id: str) -> bool:
        """Check if a user has custom preferences."""
        return user_id in self._prefs

    def remove_preferences(self, user_id: str) -> bool:
        """Remove a user's preferences (revert to defaults)."""
        return self._prefs.pop(user_id, None) is not None

    def set_defaults(
        self,
        *,
        voice: str = "alloy",
        speed: float = 1.0,
        provider: str = "openai",
    ) -> TTSPreferences:
        """Set default TTS preferences for all users."""
        self._defaults = TTSPreferences(
            user_id="__default__",
            voice=voice,
            speed=speed,
            provider=provider,
        )
        return self._defaults

    def get_defaults(self) -> Dict[str, Any]:
        """Get default preferences."""
        return self._defaults.to_dict()

    def get_available_voices(self, provider: Optional[str] = None) -> Dict[str, List[str]]:
        """Get available voices, optionally filtered by provider."""
        if provider:
            return {provider: AVAILABLE_VOICES.get(provider, [])}
        return dict(AVAILABLE_VOICES)

    def validate_voice(self, provider: str, voice: str) -> bool:
        """Check if a voice is valid for a provider."""
        voices = AVAILABLE_VOICES.get(provider, [])
        return voice in voices if voices else True  # Allow unknown providers

    def list_users(self) -> List[Dict[str, Any]]:
        """List all users with custom preferences."""
        return [p.to_dict() for p in self._prefs.values()]

    def format_preferences(self, user_id: str) -> str:
        """Format preferences for display."""
        prefs = self.get_preferences(user_id)
        is_custom = user_id in self._prefs
        return (
            f"🔊 **TTS Preferences** {'(custom)' if is_custom else '(defaults)'}\n"
            f"  Voice: {prefs.voice}\n"
            f"  Speed: {prefs.speed}x\n"
            f"  Provider: {prefs.provider}\n"
            f"  Language: {prefs.language}\n"
            f"  Auto-TTS: {'ON' if prefs.auto_tts else 'OFF'}"
        )

    def stats(self) -> Dict[str, Any]:
        """Get preference statistics."""
        by_provider: Dict[str, int] = {}
        by_voice: Dict[str, int] = {}
        for p in self._prefs.values():
            by_provider[p.provider] = by_provider.get(p.provider, 0) + 1
            by_voice[p.voice] = by_voice.get(p.voice, 0) + 1

        return {
            "total_users": len(self._prefs),
            "by_provider": by_provider,
            "by_voice": by_voice,
            "default_voice": self._defaults.voice,
            "default_provider": self._defaults.provider,
        }


# ── Singleton ────────────────────────────────────────────
_manager: Optional[TTSPreferencesManager] = None


def get_tts_prefs_manager() -> TTSPreferencesManager:
    """Get the global TTS preferences manager."""
    global _manager
    if _manager is None:
        _manager = TTSPreferencesManager()
    return _manager
