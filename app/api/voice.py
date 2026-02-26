"""
Voice / Talk Mode WebSocket API — Continuous voice conversation.
"""

import asyncio
import json
import logging
import tempfile
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["Voice / Talk Mode"])

# Reference to agent runner (set from main.py lifespan)
_agent_runner = None


def set_voice_refs(agent_runner):
    """Set references needed by voice endpoints."""
    global _agent_runner
    _agent_runner = agent_runner


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
