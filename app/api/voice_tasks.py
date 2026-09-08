"""Authenticated tenant-only durable voice task control plane."""
import secrets
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.config import settings
from app.agent.voice_tasks import VoiceTaskError, get_voice_task_service

router = APIRouter(prefix="/v1/internal/voice-tasks", include_in_schema=False)


class SubmitVoiceTask(BaseModel):
    request_id: str = Field(min_length=1, max_length=120)
    session_id: str = Field(min_length=1, max_length=36)
    message: str = Field(min_length=1, max_length=20000)
    model: Optional[str] = Field(default=None, max_length=50)
    user_message: Optional[str] = Field(default=None, max_length=20000)
    client_tz: Optional[str] = Field(default=None, max_length=64)


class SteerVoiceTask(BaseModel):
    request_id: str = Field(min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=20000)


class AckVoiceTask(BaseModel):
    revision: int = Field(ge=0)
    kind: str = Field(pattern="^(delivery|received|spoken)$")


def _authenticated_service(request: Request):
    if settings.run_mode != "agent":
        raise HTTPException(404, "Not Found")
    key = request.headers.get("X-Agent-Key", "")
    if not settings.agent_api_key or not secrets.compare_digest(key, settings.agent_api_key):
        raise HTTPException(401, "Invalid agent key")
    if not settings.voice_tasks_enabled:
        raise HTTPException(404, "Voice tasks are not enabled")
    from app.api.api_v1 import _agent_runner
    if not _agent_runner or not settings.user_id:
        raise HTTPException(503, "Agent not available")
    return get_voice_task_service(_agent_runner)


async def _call(awaitable):
    try:
        return await awaitable
    except VoiceTaskError as exc:
        raise HTTPException(exc.status_code, exc.message) from exc


@router.post("")
async def submit_voice_task(req: SubmitVoiceTask, service=Depends(_authenticated_service)):
    if not req.message.strip():
        raise HTTPException(422, "Task message cannot be blank")
    return await _call(service.submit(user_id=settings.user_id, **req.model_dump()))


@router.get("")
async def list_voice_tasks(session_id: str = Query(min_length=1, max_length=36),
                           service=Depends(_authenticated_service)):
    return await _call(service.list(user_id=settings.user_id, session_id=session_id))


@router.get("/{task_id}")
async def get_voice_task(task_id: str, service=Depends(_authenticated_service)):
    return await _call(service.get(user_id=settings.user_id, task_id=task_id))


@router.post("/{task_id}/cancel")
async def cancel_voice_task(task_id: str, service=Depends(_authenticated_service)):
    return await _call(service.cancel(user_id=settings.user_id, task_id=task_id))


@router.post("/{task_id}/steer")
async def steer_voice_task(task_id: str, req: SteerVoiceTask, service=Depends(_authenticated_service)):
    if not req.message.strip():
        raise HTTPException(422, "Correction cannot be blank")
    return await _call(service.steer(user_id=settings.user_id, task_id=task_id, **req.model_dump()))


@router.post("/{task_id}/ack")
async def ack_voice_task(
    task_id: str, req: AckVoiceTask, service=Depends(_authenticated_service),
):
    return await _call(service.ack(
        user_id=settings.user_id, task_id=task_id, **req.model_dump()
    ))


async def local_voice_task_request(runner, user_id: str, method: str, path: str,
                                   body=None, params=None):
    """Monolith adapter; the WS caller already authenticated this user.

    Identical request validation and service ownership checks to hosted HTTP.
    No tool execution through an unscoped executor or direct model fallback.
    """
    if not settings.voice_tasks_enabled:
        raise VoiceTaskError(404, "Voice tasks are not enabled")
    service = get_voice_task_service(runner)
    suffix = path.split("/internal/voice-tasks", 1)[-1].strip("/")
    if not suffix:
        if method.upper() == "POST":
            req = SubmitVoiceTask.model_validate(body or {})
            if not req.message.strip():
                raise VoiceTaskError(422, "Task message cannot be blank")
            return await service.submit(user_id=user_id, **req.model_dump())
        if method.upper() == "GET":
            return await service.list(user_id=user_id, session_id=str((params or {}).get("session_id", "")))
    parts = suffix.split("/")
    if len(parts) == 1 and method.upper() == "GET":
        return await service.get(user_id=user_id, task_id=parts[0])
    if len(parts) == 2 and method.upper() == "POST":
        if parts[1] == "cancel":
            return await service.cancel(user_id=user_id, task_id=parts[0])
        if parts[1] == "steer":
            req = SteerVoiceTask.model_validate(body or {})
            if not req.message.strip():
                raise VoiceTaskError(422, "Correction cannot be blank")
            return await service.steer(user_id=user_id, task_id=parts[0], **req.model_dump())
        if parts[1] == "ack":
            req = AckVoiceTask.model_validate(body or {})
            return await service.ack(user_id=user_id, task_id=parts[0], **req.model_dump())
    raise VoiceTaskError(404, "Voice task endpoint not found")
