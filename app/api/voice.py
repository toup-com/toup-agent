"""
Voice / Talk Mode API — Transcription, TTS, and continuous voice conversation.
"""

import asyncio
import json
import logging
import os
import re
import tempfile
import time
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.db import get_db, AgentConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["Voice / Talk Mode"])

# Reference to agent runner (set from main.py lifespan)
_agent_runner = None


def set_voice_refs(agent_runner):
    """Set references needed by voice endpoints."""
    global _agent_runner
    _agent_runner = agent_runner


class TTSRequest(BaseModel):
    text: str
    voice: str = "alloy"


async def _get_user_openai_key(user_id: str, db: AsyncSession) -> Optional[str]:
    """Retrieve the user's stored OpenAI API key from their agent config."""
    result = await db.execute(
        select(AgentConfig.openai_api_key).where(AgentConfig.user_id == user_id)
    )
    return result.scalar_one_or_none()


@router.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Transcribe an audio file to text using OpenAI Whisper."""
    from app.agent.voice_handler import transcribe_voice

    user_key = await _get_user_openai_key(current_user.id, db)

    tmp_path = f"/tmp/voice_{current_user.id}_{int(time.time())}.webm"
    try:
        with open(tmp_path, "wb") as f:
            f.write(await audio.read())
        text = await transcribe_voice(tmp_path, language=language, api_key=user_key)
        if text.startswith("ERROR:"):
            raise HTTPException(status_code=400, detail=text)
        return {"text": text}
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _detect_tts_config(text: str) -> tuple:
    """Auto-detect language from text and return (model, instructions).

    Uses gpt-4o-mini-tts with language-specific instructions for non-Latin
    scripts (Farsi, Arabic, etc.) to get proper pronunciation and accent.
    Falls back to tts-1 (fastest) for Latin-script text.
    """
    # Farsi / Arabic script
    if re.search(r'[\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF]', text):
        return (
            "gpt-4o-mini-tts",
            "Speak in natural, fluent Persian (Farsi). "
            "Use a warm and conversational tone with correct Farsi pronunciation. "
            "Do NOT read this as English or Arabic — it is Farsi.",
        )
    # Chinese
    if re.search(r'[\u4e00-\u9fff]', text):
        return ("gpt-4o-mini-tts", "Speak in natural Mandarin Chinese with clear pronunciation.")
    # Japanese (Hiragana / Katakana)
    if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
        return ("gpt-4o-mini-tts", "Speak in natural Japanese.")
    # Korean
    if re.search(r'[\uac00-\ud7af]', text):
        return ("gpt-4o-mini-tts", "Speak in natural Korean.")
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

    user_key = await _get_user_openai_key(current_user.id, db)
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


@router.get("/talk/status")
async def talk_mode_status():
    """Get all active talk mode sessions."""
    from app.agent.voice_handler import get_talk_mode_manager
    mgr = get_talk_mode_manager()
    return {
        "active_sessions": mgr.list_sessions(),
        "active_count": mgr.active_count,
    }


@router.websocket("/ws/talk/{user_id}")
async def talk_mode_ws(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint for Talk Mode — continuous voice conversation.

    Protocol:
    Client sends:
      { "type": "audio", "data": "<base64_audio>" }  — audio chunk
      { "type": "end_turn" }  — signal end of speech
      { "type": "stop" }  — end talk mode
      { "type": "ping" }

    Server sends:
      { "type": "state", "state": "listening|transcribing|thinking|speaking" }
      { "type": "transcript", "text": "..." }  — what user said
      { "type": "response_text", "text": "..." }  — agent response
      { "type": "audio_chunk", "data": "<base64_audio>" }  — TTS audio
      { "type": "turn_complete", "turn": 1 }
      { "type": "pong" }
      { "type": "error", "message": "..." }
    """
    await websocket.accept()

    from app.agent.voice_handler import (
        TalkModeSession, get_talk_mode_manager,
        transcribe_voice, synthesize_speech,
    )
    import base64

    mgr = get_talk_mode_manager()
    session = mgr.start_session(user_id)

    async def send_state(state: str):
        session.set_state(TalkModeSession.State(state))
        try:
            await websocket.send_json({"type": "state", "state": state})
        except Exception:
            pass

    logger.info("[TALK_MODE] WS session started for %s", user_id[:8])
    await send_state("listening")

    try:
        while session.is_active:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=300)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "error", "message": "Session timed out"})
                break
            except WebSocketDisconnect:
                break

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if msg_type == "stop":
                break

            if msg_type == "audio":
                # Accumulate audio data
                audio_b64 = msg.get("data", "")
                if audio_b64:
                    try:
                        audio_bytes = base64.b64decode(audio_b64)
                        session.add_audio_chunk(audio_bytes)
                    except Exception:
                        pass
                continue

            if msg_type == "end_turn":
                # Process accumulated audio
                audio_data = session.get_audio_buffer()
                if not audio_data:
                    await websocket.send_json({"type": "error", "message": "No audio received"})
                    await send_state("listening")
                    continue

                # Transcribe
                await send_state("transcribing")
                tmp_audio = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
                tmp_audio.write(audio_data)
                tmp_audio.close()

                transcript = await transcribe_voice(tmp_audio.name)

                if transcript.startswith("ERROR:"):
                    await websocket.send_json({"type": "error", "message": transcript})
                    await send_state("listening")
                    continue

                await websocket.send_json({"type": "transcript", "text": transcript})

                # Think (agent response)
                await send_state("thinking")
                if _agent_runner:
                    try:
                        response = await _agent_runner.run(
                            user_message=transcript,
                            user_id=user_id,
                        )
                        response_text = response.text
                    except Exception as e:
                        response_text = f"Error: {e}"
                else:
                    response_text = "Agent not available"

                await websocket.send_json({"type": "response_text", "text": response_text})

                # Speak (TTS)
                await send_state("speaking")
                audio_path = await synthesize_speech(response_text)

                if not audio_path.startswith("ERROR:"):
                    try:
                        with open(audio_path, "rb") as af:
                            audio_data = af.read()
                        audio_b64 = base64.b64encode(audio_data).decode()
                        await websocket.send_json({
                            "type": "audio_chunk",
                            "data": audio_b64,
                            "format": "ogg",
                        })
                    except Exception as e:
                        logger.warning("[TALK_MODE] TTS send failed: %s", e)

                session.turn_count += 1
                await websocket.send_json({
                    "type": "turn_complete",
                    "turn": session.turn_count,
                })
                await send_state("listening")
                continue

    finally:
        mgr.end_session(user_id)
        logger.info("[TALK_MODE] WS session ended for %s (turns=%d)",
                     user_id[:8], session.turn_count)
        try:
            await websocket.close()
        except Exception:
            pass
