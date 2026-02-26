"""
Voice Handler — Multi-provider TTS and transcription.

Layer 6 enhancements:
  * ElevenLabs TTS provider (streaming support)
  * Edge TTS provider (Microsoft free TTS)
  * Talk Mode protocol (continuous voice conversation via WebSocket)
  * Voice Wake detection (hotword trigger)
  * Provider abstraction for easy switching
"""

import asyncio
import logging
import os
import tempfile
import time
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# TTS Provider Abstraction
# ──────────────────────────────────────────────────────────────

class TTSProvider(str, Enum):
    OPENAI = "openai"
    ELEVENLABS = "elevenlabs"
    EDGE = "edge"


VALID_VOICES = {
    "alloy", "ash", "ballad", "coral", "echo",
    "fable", "nova", "onyx", "sage", "shimmer",
}

ELEVENLABS_DEFAULT_VOICES = {
    "rachel": "21m00Tcm4TlvDq8ikWAM",
    "adam": "pNInz6obpgDQGcFmaJgB",
    "sam": "yoZ06aMxZJJ28mfd3POQ",
    "elli": "MF3mGyEYCl7XYWbV9V6O",
    "josh": "TxGEqnHWrfWFTfGW9XjX",
    "bella": "EXAVITQu4vr4xnSDxMaL",
}


# ──────────────────────────────────────────────────────────────
# Transcription (Whisper)
# ──────────────────────────────────────────────────────────────

async def transcribe_voice(file_path: str, language: Optional[str] = None) -> str:
    """Transcribe audio using OpenAI Whisper API."""
    api_key = settings.openai_api_key
    if not api_key:
        return "ERROR: OpenAI API key not configured for transcription."

    if not os.path.isfile(file_path):
        return f"ERROR: Audio file not found: {file_path}"

    file_size = os.path.getsize(file_path)
    if file_size > 25 * 1024 * 1024:
        return "ERROR: Audio file too large (max 25MB)."

    logger.info(f"[AGENT] Transcribing voice: {file_path} ({file_size} bytes)")

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f, "audio/ogg")}
                data = {"model": "whisper-1"}
                if language:
                    data["language"] = language

                resp = await client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    files=files,
                    data=data,
                )
                resp.raise_for_status()

        result = resp.json()
        text = result.get("text", "").strip()
        logger.info(f"[AGENT] Transcription result ({len(text)} chars): {text[:100]}")
        return text or "(empty transcription)"

    except httpx.HTTPStatusError as exc:
        logger.error(f"[AGENT] Whisper API error: {exc.response.status_code}")
        return f"ERROR: Whisper API returned {exc.response.status_code}"
    except Exception as exc:
        logger.exception("[AGENT] Voice transcription failed")
        return f"ERROR: Transcription failed: {exc}"


# ──────────────────────────────────────────────────────────────
# TTS — Multi-provider synthesis
# ──────────────────────────────────────────────────────────────

async def synthesize_speech(
    text: str,
    voice: str = "nova",
    model: str = "gpt-4o-mini-tts",
    speed: float = 1.0,
    instructions: Optional[str] = None,
    provider: Optional[str] = None,
) -> str:
    """
    Generate speech audio from text using the configured TTS provider.

    Returns path to the generated audio file, or "ERROR: ...".
    """
    prov = provider or getattr(settings, "tts_provider", "openai")

    if prov == TTSProvider.ELEVENLABS or prov == "elevenlabs":
        return await _tts_elevenlabs(text, voice)
    elif prov == TTSProvider.EDGE or prov == "edge":
        return await _tts_edge(text, voice)
    else:
        return await _tts_openai(text, voice, model, speed, instructions)


async def _tts_openai(text: str, voice: str, model: str,
                      speed: float, instructions: Optional[str]) -> str:
    """OpenAI TTS synthesis."""
    api_key = settings.openai_api_key
    if not api_key:
        return "ERROR: OpenAI API key not configured for TTS."

    if not text or not text.strip():
        return "ERROR: Empty text for TTS."

    voice = voice.lower() if voice else "nova"
    if voice not in VALID_VOICES:
        voice = "nova"

    speed = max(0.25, min(4.0, speed))
    if len(text) > 4096:
        text = text[:4096]

    logger.info("[TTS] OpenAI: voice=%s model=%s len=%d", voice, model, len(text))

    try:
        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": "opus",
            "speed": speed,
        }
        if instructions and model == "gpt-4o-mini-tts":
            payload["instructions"] = instructions

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()

        tmp = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
        tmp.write(resp.content)
        tmp.close()
        logger.info("[TTS] OpenAI generated %d bytes → %s", len(resp.content), tmp.name)
        return tmp.name

    except httpx.HTTPStatusError as exc:
        logger.error("[TTS] OpenAI API error: %s", exc.response.status_code)
        return f"ERROR: TTS API returned {exc.response.status_code}"
    except Exception as exc:
        logger.exception("[TTS] OpenAI synthesis failed")
        return f"ERROR: TTS failed: {exc}"


async def _tts_elevenlabs(text: str, voice: str) -> str:
    """ElevenLabs TTS synthesis with streaming support."""
    api_key = getattr(settings, "elevenlabs_api_key", None)
    if not api_key:
        return "ERROR: ElevenLabs API key not configured."

    if not text or not text.strip():
        return "ERROR: Empty text for TTS."

    # Resolve voice name to ID
    voice_id = ELEVENLABS_DEFAULT_VOICES.get(voice.lower(), voice)
    if not voice_id:
        voice_id = getattr(settings, "elevenlabs_voice_id", "21m00Tcm4TlvDq8ikWAM")

    model_id = getattr(settings, "elevenlabs_model", "eleven_multilingual_v2")

    if len(text) > 5000:
        text = text[:5000]

    logger.info("[TTS] ElevenLabs: voice=%s model=%s len=%d", voice_id[:8], model_id, len(text))

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": api_key,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg",
                },
                json={
                    "text": text,
                    "model_id": model_id,
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                    },
                },
            )
            resp.raise_for_status()

        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.write(resp.content)
        tmp.close()
        logger.info("[TTS] ElevenLabs generated %d bytes → %s", len(resp.content), tmp.name)
        return tmp.name

    except httpx.HTTPStatusError as exc:
        logger.error("[TTS] ElevenLabs API error: %s", exc.response.status_code)
        return f"ERROR: ElevenLabs TTS API returned {exc.response.status_code}"
    except Exception as exc:
        logger.exception("[TTS] ElevenLabs synthesis failed")
        return f"ERROR: ElevenLabs TTS failed: {exc}"


async def _tts_edge(text: str, voice: str) -> str:
    """Microsoft Edge TTS (free, no API key required)."""
    if not text or not text.strip():
        return "ERROR: Empty text for TTS."

    if len(text) > 5000:
        text = text[:5000]

    # Map simple names to full Edge TTS voice names
    edge_voices = {
        "nova": "en-US-JennyNeural",
        "alloy": "en-US-GuyNeural",
        "echo": "en-US-AriaNeural",
        "shimmer": "en-US-JaneNeural",
        "onyx": "en-US-DavisNeural",
        "coral": "en-US-SaraNeural",
    }
    edge_voice = edge_voices.get(voice.lower(), voice)

    logger.info("[TTS] Edge: voice=%s len=%d", edge_voice, len(text))

    try:
        import edge_tts

        communicate = edge_tts.Communicate(text, edge_voice)
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()
        await communicate.save(tmp.name)

        file_size = os.path.getsize(tmp.name)
        if file_size == 0:
            return "ERROR: Edge TTS produced empty audio"

        logger.info("[TTS] Edge generated %d bytes → %s", file_size, tmp.name)
        return tmp.name

    except ImportError:
        return "ERROR: edge-tts package not installed. Run: pip install edge-tts"
    except Exception as exc:
        logger.exception("[TTS] Edge synthesis failed")
        return f"ERROR: Edge TTS failed: {exc}"


# ──────────────────────────────────────────────────────────────
# ElevenLabs Streaming TTS
# ──────────────────────────────────────────────────────────────

async def stream_tts_elevenlabs(text: str, voice: str = "rachel") -> AsyncIterator[bytes]:
    """
    Stream TTS audio chunks from ElevenLabs.
    Yields raw audio bytes as they arrive.
    """
    api_key = getattr(settings, "elevenlabs_api_key", None)
    if not api_key:
        return

    voice_id = ELEVENLABS_DEFAULT_VOICES.get(voice.lower(), voice)
    model_id = getattr(settings, "elevenlabs_model", "eleven_multilingual_v2")

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream",
                headers={
                    "xi-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": model_id,
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
                },
            ) as response:
                async for chunk in response.aiter_bytes(1024):
                    yield chunk
    except Exception as e:
        logger.exception("[TTS] ElevenLabs streaming failed")


# ──────────────────────────────────────────────────────────────
# Talk Mode — Continuous voice conversation protocol
# ──────────────────────────────────────────────────────────────

class TalkModeSession:
    """
    Manages a continuous voice conversation session.

    Protocol (over WebSocket):
    1. Client sends audio chunks → server transcribes
    2. Server sends text to agent → gets response
    3. Server synthesises response → sends audio back
    4. Repeat until session ends

    States: idle → listening → transcribing → thinking → speaking → idle
    """

    class State(str, Enum):
        IDLE = "idle"
        LISTENING = "listening"
        TRANSCRIBING = "transcribing"
        THINKING = "thinking"
        SPEAKING = "speaking"
        ENDED = "ended"

    def __init__(self, user_id: str, session_id: Optional[str] = None):
        self.user_id = user_id
        self.session_id = session_id
        self.state = self.State.IDLE
        self.started_at = time.time()
        self.turn_count = 0
        self.total_audio_bytes = 0
        self._audio_buffer = bytearray()
        self._active = False

    def start(self):
        """Start the talk mode session."""
        self._active = True
        self.state = self.State.LISTENING
        logger.info("[TALK_MODE] Session started for user %s", self.user_id[:8])

    def stop(self):
        """Stop the talk mode session."""
        self._active = False
        self.state = self.State.ENDED
        logger.info("[TALK_MODE] Session ended for user %s (turns=%d)",
                     self.user_id[:8], self.turn_count)

    @property
    def is_active(self) -> bool:
        return self._active

    def add_audio_chunk(self, chunk: bytes):
        """Add an audio chunk to the buffer."""
        self._audio_buffer.extend(chunk)
        self.total_audio_bytes += len(chunk)

    def get_audio_buffer(self) -> bytes:
        """Get and clear the accumulated audio buffer."""
        data = bytes(self._audio_buffer)
        self._audio_buffer.clear()
        return data

    def set_state(self, state: State):
        """Update session state."""
        self.state = state

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "state": self.state.value,
            "is_active": self._active,
            "turn_count": self.turn_count,
            "total_audio_bytes": self.total_audio_bytes,
            "uptime_seconds": int(time.time() - self.started_at),
        }


class TalkModeManager:
    """Manages talk mode sessions across users."""

    def __init__(self):
        self._sessions: Dict[str, TalkModeSession] = {}

    def start_session(self, user_id: str, session_id: Optional[str] = None) -> TalkModeSession:
        """Start a new talk mode session."""
        sess = TalkModeSession(user_id, session_id)
        sess.start()
        self._sessions[user_id] = sess
        return sess

    def get_session(self, user_id: str) -> Optional[TalkModeSession]:
        """Get an active talk mode session."""
        return self._sessions.get(user_id)

    def end_session(self, user_id: str) -> bool:
        """End a talk mode session."""
        sess = self._sessions.pop(user_id, None)
        if sess:
            sess.stop()
            return True
        return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active talk mode sessions."""
        return [s.to_dict() for s in self._sessions.values() if s.is_active]

    @property
    def active_count(self) -> int:
        return len([s for s in self._sessions.values() if s.is_active])


# Global talk mode manager
_talk_mode_manager: Optional[TalkModeManager] = None


def get_talk_mode_manager() -> TalkModeManager:
    global _talk_mode_manager
    if _talk_mode_manager is None:
        _talk_mode_manager = TalkModeManager()
    return _talk_mode_manager


# ──────────────────────────────────────────────────────────────
# Voice Wake — Hotword detection
# ──────────────────────────────────────────────────────────────

class VoiceWakeDetector:
    """
    Simple voice wake / hotword detection.

    Uses a keyword-based approach: if the transcribed text starts with
    a wake word, it triggers talk mode. This avoids heavy on-device
    hotword models while still providing a natural activation mechanism.
    """

    DEFAULT_WAKE_WORDS = {"hey hex", "ok hex", "hexbrain", "hex brain"}

    def __init__(self, wake_words: Optional[set] = None):
        self.wake_words = wake_words or self.DEFAULT_WAKE_WORDS
        self.enabled = True

    def check(self, text: str) -> bool:
        """Check if text contains a wake word."""
        if not self.enabled:
            return False
        text_lower = text.lower().strip()
        return any(text_lower.startswith(w) for w in self.wake_words)

    def strip_wake_word(self, text: str) -> str:
        """Remove the wake word from the beginning of text."""
        text_lower = text.lower().strip()
        for w in sorted(self.wake_words, key=len, reverse=True):
            if text_lower.startswith(w):
                return text[len(w):].strip()
        return text

    def add_wake_word(self, word: str):
        self.wake_words.add(word.lower())

    def remove_wake_word(self, word: str):
        self.wake_words.discard(word.lower())

    def list_wake_words(self) -> list:
        return sorted(self.wake_words)
