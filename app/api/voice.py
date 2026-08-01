"""
Voice API — Transcription and TTS.

The continuous-conversation surface lives at /api/ws/realtime (ws_realtime.py).
"""

import logging
import os
import re
import time
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.config import settings
from app.db import get_db, AgentConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["Voice"])


class TTSRequest(BaseModel):
    text: str
    voice: str = "alloy"


async def _get_user_openai_key(user_id: str, db: AsyncSession) -> Optional[str]:
    """Retrieve the user's stored OpenAI API key from their agent config."""
    result = await db.execute(
        select(AgentConfig.openai_api_key).where(AgentConfig.user_id == user_id)
    )
    return result.scalar_one_or_none()


# Whisper only accepts these containers; anything else 400s. The temp-file
# extension MUST match the actual bytes, because downstream we hand Whisper a
# filename + content-type derived from that extension. The web client uploads
# webm; mobile uploads m4a (AAC). Hard-coding .webm here makes Whisper try to
# demux mobile's AAC as webm → "Whisper API returned 400" on every dictation.
_WHISPER_EXTS = {".flac", ".m4a", ".mp3", ".mp4", ".mpeg", ".mpga", ".oga", ".ogg", ".wav", ".webm"}
_CONTENT_TYPE_EXT = {
    "audio/webm": ".webm", "audio/ogg": ".ogg", "audio/oga": ".oga",
    "audio/mpeg": ".mp3", "audio/mp3": ".mp3", "audio/mp4": ".m4a",
    "audio/m4a": ".m4a", "audio/x-m4a": ".m4a",
    "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/flac": ".flac",
}


def _audio_suffix(upload: UploadFile) -> str:
    """Pick a Whisper-supported file extension from the upload, never trusting
    a single source: filename first, then content-type, then webm as the
    last-resort default (matches the web client)."""
    ext = os.path.splitext(upload.filename or "")[1].lower()
    if ext in _WHISPER_EXTS:
        return ext
    content_type = (upload.content_type or "").split(";")[0].strip().lower()
    return _CONTENT_TYPE_EXT.get(content_type, ".webm")


@router.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Transcribe an audio file to text using OpenAI Whisper."""
    from app.agent.voice_handler import transcribe_voice
    # Imported here, not at module scope: ws_realtime imports THIS module for
    # detect_script_language, so a top-level import would close the cycle.
    from app.api.ws_realtime import resolve_voice_language

    user_key = await _get_user_openai_key(current_user.id, db)

    # An explicit client-supplied language always wins; this only fills the gap
    # when the caller said nothing, which today is every caller.
    if not language:
        language = await resolve_voice_language(current_user.id)

    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="ERROR: Empty audio upload.")

    tmp_path = f"/tmp/voice_{current_user.id}_{int(time.time())}{_audio_suffix(audio)}"
    try:
        with open(tmp_path, "wb") as f:
            f.write(data)
        text = await transcribe_voice(tmp_path, language=language, api_key=user_key)
        if text.startswith("ERROR:"):
            raise HTTPException(status_code=400, detail=text)
        return {"text": text}
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# Script -> ISO-639-1, highest-priority first. THE single copy of this
# vocabulary: TTS instruction selection and the Whisper `language` hint
# (ws_realtime.resolve_voice_language) both read it, so a new script gets
# added once. Order is load-bearing -- it is the original _detect_tts_config
# branch order, and the first pattern that matches wins.
_SCRIPT_LANGUAGES: tuple = (
    # Arabic block. Reported as "fa" because it drives a Farsi TTS instruction
    # and always has; an actual Arabic speaker is mis-served by that branch and
    # has been since this function was written. Separating them needs the
    # Persian-only letters (pe/che/zhe/gaf) plus an Arabic instruction --
    # deliberately NOT done here, so this refactor changes no behavior.
    ("fa", re.compile(r'[\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF]')),
    ("zh", re.compile(r'[\u4e00-\u9fff]')),
    ("ja", re.compile(r'[\u3040-\u309f\u30a0-\u30ff]')),  # Hiragana / Katakana
    ("ko", re.compile(r'[\uac00-\ud7af]')),
)

_LATIN_RE = re.compile(r'[A-Za-z]')

_TTS_INSTRUCTIONS: dict = {
    "fa": (
        "Speak in natural, fluent Persian (Farsi). "
        "Use a warm and conversational tone with correct Farsi pronunciation. "
        "Do NOT read this as English or Arabic — it is Farsi."
    ),
    "zh": "Speak in natural Mandarin Chinese with clear pronunciation.",
    "ja": "Speak in natural Japanese.",
    "ko": "Speak in natural Korean.",
}


def detect_script_language(
    text: str, *, min_share: float = 0.0, min_chars: int = 1
) -> Optional[str]:
    """ISO-639-1 code of the non-Latin script `text` is written in, else None.

    `min_share` / `min_chars` raise the bar from "contains one such character"
    to "is actually written in this script" -- what the transcription hint
    needs, since one quoted foreign song title inside an English sentence must
    not be read as the speaker's language. The share is measured against Latin
    letters only, so punctuation, digits and emoji cannot dilute an otherwise
    wholly non-Latin sentence.

    The defaults reproduce the original substring test exactly, so callers that
    want the loose behavior (TTS) pass nothing.
    """
    latin = len(_LATIN_RE.findall(text))
    for code, pattern in _SCRIPT_LANGUAGES:
        found = len(pattern.findall(text))
        if found < min_chars:
            continue
        if min_share and found < min_share * (found + latin):
            continue
        return code
    return None


def _detect_tts_config(text: str) -> tuple:
    """Auto-detect language from text and return (model, instructions).

    Uses gpt-4o-mini-tts with language-specific instructions for non-Latin
    scripts (Farsi, Arabic, etc.) to get proper pronunciation and accent.
    Falls back to tts-1 (fastest) for Latin-script text.
    """
    code = detect_script_language(text)
    if code:
        return ("gpt-4o-mini-tts", _TTS_INSTRUCTIONS[code])
    # Default: fast model, no instructions
    return ("tts-1", None)


@router.post("/tts")
async def text_to_speech(
    body: TTSRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Convert text to speech using OpenAI TTS. Streams audio directly."""
    import httpx

    # Fall back to the platform key exactly like /transcribe (voice_handler.py:58)
    # so voice replies work for managed/bundle users who have no personal OpenAI
    # key — the same users whose STT already rides the platform key. Without this
    # the mobile Talk Mode is voice-in / text-out only for them.
    user_key = await _get_user_openai_key(current_user.id, db) or settings.openai_api_key
    if not user_key:
        raise HTTPException(status_code=400, detail="OpenAI API key not configured for TTS.")

    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text for TTS.")
    if len(text) > 4096:
        text = text[:4096]

    voice = (body.voice or "alloy").lower()
    valid_voices = {"alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"}
    if voice not in valid_voices:
        voice = "alloy"

    # Auto-detect language → pick model + instructions
    model, instructions = _detect_tts_config(text)
    logger.info("[TTS] model=%s, voice=%s, len=%d, instructions=%s",
                model, voice, len(text), bool(instructions))

    payload = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": "mp3",
        "speed": 1.15,
    }
    if instructions:
        payload["instructions"] = instructions

    async def stream_tts():
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                "https://api.openai.com/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {user_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes(chunk_size=4096):
                    yield chunk

    return StreamingResponse(stream_tts(), media_type="audio/mpeg")
