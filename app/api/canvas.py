"""
Canvas / A2UI API Router — WebSocket endpoint for canvas push + REST for snapshots.
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/canvas", tags=["Canvas / A2UI"])


@router.get("/state/{user_id}")
async def get_canvas_state(user_id: str):
    """Get the current canvas state for a user."""
    from app.agent.canvas import get_canvas_manager
    mgr = get_canvas_manager()
    state = mgr.get_state(user_id)
    if not state:
        return {"user_id": user_id, "visible": False, "frames": {}, "frame_count": 0}
    return state.to_dict()


@router.post("/present")
async def present_canvas(user_id: str, content: str, content_type: str = "html",
                         title: str = "", frame_id: Optional[str] = None):
    """Present content on a user's canvas (REST fallback)."""
    from app.agent.canvas import get_canvas_manager
    mgr = get_canvas_manager()
    result = await mgr.present(user_id, content, content_type, title, frame_id)
    return result


@router.post("/hide/{user_id}")
async def hide_canvas(user_id: str):
    """Hide the canvas for a user."""
    from app.agent.canvas import get_canvas_manager
    mgr = get_canvas_manager()
    return await mgr.hide(user_id)


@router.post("/clear/{user_id}")
async def clear_canvas(user_id: str, frame_id: Optional[str] = None):
    """Clear canvas frames."""
    from app.agent.canvas import get_canvas_manager
    mgr = get_canvas_manager()
    return await mgr.clear(user_id, frame_id)


@router.websocket("/ws/{user_id}")
async def canvas_ws(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint for real-time canvas updates.

    Client connects and receives canvas events:
      canvas_present, canvas_hide, canvas_show, canvas_clear_frame,
      canvas_clear_all, canvas_layout, canvas_eval

    Client can send events back:
      { "type": "canvas_event", "event": "click", "data": {...} }
    """
    await websocket.accept()

    from app.agent.canvas import get_canvas_manager
    mgr = get_canvas_manager()

    async def push_to_ws(msg: dict):
        try:
            await websocket.send_json(msg)
        except Exception:
            pass

    mgr.add_listener(user_id, push_to_ws)
    logger.info("[CANVAS-WS] Client connected: %s", user_id[:8])

    try:
        # Send current state on connect
        state = mgr.get_state(user_id)
        if state and state.visible:
            await websocket.send_json({
                "type": "canvas_state",
                "data": state.to_dict(),
            })

        while True:
            try:
                raw = await websocket.receive_text()
                msg = json.loads(raw)
                msg_type = msg.get("type", "")

                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                elif msg_type == "canvas_event":
                    logger.info("[CANVAS-WS] Event from %s: %s", user_id[:8], msg.get("event"))
                    # Future: route user interactions back to agent
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                pass
    finally:
        mgr.remove_listener(user_id, push_to_ws)
        logger.info("[CANVAS-WS] Client disconnected: %s", user_id[:8])
