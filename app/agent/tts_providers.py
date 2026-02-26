"""
TTS Providers — Multi-backend text-to-speech engine.

Supports:
  - OpenAI TTS (gpt-4o-mini-tts, tts-1, tts-1-hd)
  - ElevenLabs (eleven_multilingual_v2, eleven_turbo_v2)
  - Edge TTS (free, offline-capable via edge-tts package)

Per-user preferences are stored in a JSON file at /app/workspace/.tts_prefs.json
"""

import json
import logging
import os
import tempfile
from typing import Any, Dict, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# ── Per-user TTS preferences ────────────────────────────────
_PREFS_PATH = os.path.join(settings.agent_workspace_dir, ".tts_prefs.json")
_prefs_cache: Dict[str, Dict[str, Any]] = {}


def _load_prefs() -> Dict[str, Dict[str, Any]]:
    """Load per-user TTS preferences from disk."""
    global _prefs_cache
    if _prefs_cache:
        return _prefs_cache
    try:
        if os.path.isfile(_PREFS_PATH):
            with open(_PREFS_PATH, "r") as f:
                _prefs_cache = json.load(f)
    except Exception:
        _prefs_cache = {}
    return _prefs_cache


def _save_prefs() -> None:
    """Persist per-user TTS preferences to disk."""
    try:
        os.makedirs(os.path.dirname(_PREFS_PATH), exist_ok=True)
        with open(_PREFS_PATH, "w") as f:
            json.dump(_prefs_cache, f, indent=2)
    except Exception as e:
        logger.warning(f"[TTS] Failed to save prefs: {e}")


def get_user_tts_prefs(user_id: str) -> Dict[str, Any]:
    """Get TTS preferences for a user (provider, voice, speed, model)."""
    prefs = _load_prefs()
    return prefs.get(user_id, {
        "provider": settings.tts_provider,
        "voice": settings.tts_default_voice,
        "speed": settings.tts_speed,
        "model": settings.tts_model,
    })


def set_user_tts_prefs(user_id: str, **kwargs) -> Dict[str, Any]:
    """Update TTS preferences for a user. Returns updated prefs."""
    prefs = _load_prefs()
    current = prefs.get(user_id, {
        "provider": settings.tts_provider,
        "voice": settings.tts_default_voice,
        "speed": settings.tts_speed,
        "model": settings.tts_model,
    })
    current.update({k: v for k, v in kwargs.items() if v is not None})
    prefs[user_id] = current
    _prefs_cache.update(prefs)
    _save_prefs()
    return current


# ── ElevenLabs TTS ───────────────────────────────────────────

ELEVENLABS_VOICES = {
    "rachel": "21m00Tcm4TlvDq8ikWAM",
    "domi": "AZnzlk1XvdvUeBnXmlld",
    "bella": "EXAVITQu4vr4xnSDxMaL",
    "antoni": "ErXwobaYiN019PkySvjV",
    "elli": "MF3mGyEYCl7XYWbV9V6O",
    "josh": "TxGEqnHWrfWFTfGW9XjX",
    "arnold": "VR6AewLTigWG4xSOukaG",
    "adam": "pNInz6obpgDQGcFmaJgB",
    "sam": "yoZ06aMxZJJ28mfd3POQ",
}


async def synthesize_elevenlabs(
    text: str,
    voice_id: Optional[str] = None,
    model: Optional[str] = None,
    speed: float = 1.0,
) -> str:
    """Generate speech via ElevenLabs API. Returns path to .ogg file or ERROR string."""
    api_key = settings.elevenlabs_api_key
    if not api_key:
        return "ERROR: ElevenLabs API key not configured. Set ELEVENLABS_API_KEY."

    voice_id = voice_id or settings.elevenlabs_voice_id
    # Resolve voice name to ID
    if voice_id.lower() in ELEVENLABS_VOICES:
        voice_id = ELEVENLABS_VOICES[voice_id.lower()]
    model = model or settings.elevenlabs_model

    if len(text) > 5000:
        text = text[:5000]

    logger.info("[TTS:ElevenLabs] voice=%s model=%s len=%d", voice_id, model, len(text))

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
                    "model_id": model,
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                        "speed": speed,
                    },
                },
            )
            resp.raise_for_status()

        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.write(resp.content)
        tmp.close()
        logger.info("[TTS:ElevenLabs] Generated %d bytes → %s", len(resp.content), tmp.name)
        return tmp.name

    except httpx.HTTPStatusError as exc:
        logger.error("[TTS:ElevenLabs] API error: %s", exc.response.text[:200])
        return f"ERROR: ElevenLabs API returned {exc.response.status_code}"
    except Exception as exc:
        logger.exception("[TTS:ElevenLabs] Failed")
        return f"ERROR: ElevenLabs TTS failed: {exc}"


# ── Edge TTS (free) ──────────────────────────────────────────

EDGE_VOICES = {
    "alloy": "en-US-GuyNeural",
    "nova": "en-US-JennyNeural",
    "echo": "en-US-EricNeural",
    "shimmer": "en-US-AriaNeural",
    "onyx": "en-GB-RyanNeural",
    "fable": "en-AU-WilliamNeural",
}


async def synthesize_edge_tts(
    text: str,
    voice: str = "nova",
    speed: float = 1.0,
) -> str:
    """Generate speech via Edge TTS (free). Returns path to .mp3 file or ERROR string."""
    try:
        import edge_tts
    except ImportError:
        return "ERROR: edge-tts not installed. Run: pip install edge-tts"

    # Map OpenAI voice names to Edge TTS voice names
    edge_voice = EDGE_VOICES.get(voice.lower(), "en-US-JennyNeural")

    if len(text) > 5000:
        text = text[:5000]

    # Convert speed multiplier to Edge TTS rate string (e.g., +50% or -25%)
    rate_pct = int((speed - 1.0) * 100)
    rate_str = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"

    logger.info("[TTS:Edge] voice=%s rate=%s len=%d", edge_voice, rate_str, len(text))

    try:
        tmp_path = tempfile.mktemp(suffix=".mp3")
        communicate = edge_tts.Communicate(text, edge_voice, rate=rate_str)
        await communicate.save(tmp_path)
        logger.info("[TTS:Edge] Generated → %s", tmp_path)
        return tmp_path

    except Exception as exc:
        logger.exception("[TTS:Edge] Failed")
        return f"ERROR: Edge TTS failed: {exc}"


# ── Dispatcher ───────────────────────────────────────────────

async def synthesize_speech_multi(
    text: str,
    provider: Optional[str] = None,
    voice: Optional[str] = None,
    model: Optional[str] = None,
    speed: float = 1.0,
    instructions: Optional[str] = None,
    user_id: Optional[str] = None,
) -> str:
    """
    Dispatch TTS to the appropriate provider.
    If user_id is given and per-user prefs are enabled, use their saved preferences.
    Falls back through providers: primary → edge (free).
    """
    # Load per-user preferences
    if user_id and settings.tts_per_user_prefs:
        prefs = get_user_tts_prefs(user_id)
        provider = provider or prefs.get("provider", settings.tts_provider)
        voice = voice or prefs.get("voice", settings.tts_default_voice)
        speed = speed or prefs.get("speed", settings.tts_speed)
        model = model or prefs.get("model", settings.tts_model)
    else:
        provider = provider or settings.tts_provider
        voice = voice or settings.tts_default_voice
        model = model or settings.tts_model

    if provider == "elevenlabs":
        result = await synthesize_elevenlabs(text, voice_id=voice, model=model, speed=speed)
        if result.startswith("ERROR:"):
            logger.warning("[TTS] ElevenLabs failed, falling back to Edge TTS")
            result = await synthesize_edge_tts(text, voice=voice, speed=speed)
        return result

    elif provider == "edge":
        return await synthesize_edge_tts(text, voice=voice, speed=speed)

    else:
        # Default: OpenAI
        from app.agent.voice_handler import synthesize_speech
        result = await synthesize_speech(text, voice=voice, model=model, speed=speed, instructions=instructions)
        if result.startswith("ERROR:"):
            logger.warning("[TTS] OpenAI TTS failed, falling back to Edge TTS")
            result = await synthesize_edge_tts(text, voice=voice, speed=speed)
        return result
