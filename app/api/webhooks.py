"""
Webhook endpoints for external event triggers.
Fires wake events on the CronService to trigger event-based cron jobs.
"""
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/webhooks", tags=["webhooks"])

_cron_service = None

def set_cron_ref(cron_service):
    """Called from main.py lifespan to inject cron service reference."""
    global _cron_service
    _cron_service = cron_service


class WebhookPayload(BaseModel):
    event: str
    payload: Optional[Dict[str, Any]] = None


@router.post("/trigger")
async def trigger_wake_event(body: WebhookPayload, request: Request):
    """
    Fire a wake event to trigger matching cron jobs.
    
    POST /api/v1/webhooks/trigger
    {
        "event": "gmail.new_email",
        "payload": {"from": "user@example.com", "subject": "Hello"}
    }
    """
    if not _cron_service:
        raise HTTPException(status_code=503, detail="Cron service not available")

    results = await _cron_service.fire_wake_event(body.event, body.payload)
    return {
        "status": "ok",
        "event": body.event,
        "triggered_jobs": len(results),
        "results": results,
    }


@router.post("/gmail")
async def gmail_pubsub_webhook(request: Request):
    """
    Gmail Pub/Sub push notification endpoint.
    Receives notifications when new emails arrive and fires 'gmail.new_email' wake event.
    
    Expects Google Cloud Pub/Sub push format:
    {
        "message": {
            "data": "<base64-encoded>",
            "messageId": "...",
            "publishTime": "..."
        },
        "subscription": "..."
    }
    """
    if not _cron_service:
        raise HTTPException(status_code=503, detail="Cron service not available")

    import base64, json

    try:
        body = await request.json()
        message = body.get("message", {})
        data_b64 = message.get("data", "")
        
        if data_b64:
            decoded = base64.b64decode(data_b64).decode("utf-8")
            try:
                payload = json.loads(decoded)
            except json.JSONDecodeError:
                payload = {"raw": decoded}
        else:
            payload = {"raw": str(body)}

        logger.info(f"📧 Gmail webhook received: {str(payload)[:200]}")

        results = await _cron_service.fire_wake_event("gmail.new_email", payload)
        return {"status": "ok", "triggered": len(results)}

    except Exception as e:
        logger.error(f"Gmail webhook error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/events")
async def list_registered_events():
    """List all unique wake events registered across cron jobs."""
    if not _cron_service:
        raise HTTPException(status_code=503, detail="Cron service not available")

    events = set()
    for jinfo in _cron_service._jobs.values():
        we = jinfo.get("wake_event")
        if we:
            events.add(we)

    return {
        "events": sorted(events),
        "total_jobs_with_events": sum(
            1 for j in _cron_service._jobs.values() if j.get("wake_event")
        ),
    }


# Alias for main.py compatibility
_agent_runner = None
_telegram_bot = None

def set_webhook_refs(agent_runner, telegram_bot):
    """Called from main.py lifespan to inject agent runner and telegram bot refs."""
    global _agent_runner, _telegram_bot, _cron_service
    _agent_runner = agent_runner
    _telegram_bot = telegram_bot
    # Also set cron_service ref from agent_runner if available
    if hasattr(agent_runner, 'cron_service'):
        _cron_service = agent_runner.cron_service
    elif hasattr(telegram_bot, 'cron_service'):
        _cron_service = telegram_bot.cron_service
