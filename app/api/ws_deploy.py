"""
WebSocket endpoint for live agent deployment log streaming.

ws://toup.ai/api/ws/deploy?token=JWT_TOKEN

Streams deploy log lines in real-time as the SSH deployment runs.
"""

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from jose import jwt, JWTError

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Agent Deploy WS"])


@router.websocket("/ws/deploy")
async def ws_deploy(ws: WebSocket):
    """
    WebSocket that streams deployment logs.

    Messages sent to client:
        {"type": "deploy_log", "line": "...", "level": "info|success|error|cmd|step"}
        {"type": "deploy_complete", "success": true/false}
    """
    token = ws.query_params.get("token", "")
    if not token:
        await ws.close(code=4001, reason="Missing token")
        return

    # Verify JWT
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        user_id = payload.get("sub")
        if not user_id:
            await ws.close(code=4001, reason="Invalid token")
            return
    except JWTError:
        await ws.close(code=4001, reason="Invalid token")
        return

    await ws.accept()
    logger.info("Deploy WS connected: user=%s", user_id)

    # Import here to avoid circular imports
    from app.api.agent_setup import get_deploy_logs

    sent_count = 0

    try:
        while True:
            log_state = get_deploy_logs(user_id)

            if log_state is None:
                # No deploy in progress — wait and check again
                await asyncio.sleep(1)
                continue

            # Stream any new log lines
            lines = log_state["lines"]
            while sent_count < len(lines):
                entry = lines[sent_count]
                await ws.send_json({
                    "type": "deploy_log",
                    "line": entry["line"],
                    "level": entry["level"],
                })
                sent_count += 1

            # Check if deploy is done
            if log_state["done"]:
                await ws.send_json({
                    "type": "deploy_complete",
                    "success": log_state["success"],
                })
                break

            await asyncio.sleep(0.3)

    except WebSocketDisconnect:
        logger.info("Deploy WS disconnected: user=%s", user_id)
    except Exception as e:
        logger.exception("Deploy WS error: %s", e)
        try:
            await ws.close(code=4500, reason="Internal error")
        except Exception:
            pass
