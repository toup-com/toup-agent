"""
Workflow Builder — Conversational AI that asks questions then generates workflows.

Uses SSE (Server-Sent Events) to stream the AI conversation to the frontend.
The AI asks clarifying questions about the user's needs, then generates
workflow JSON that updates the preview in real-time.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user
from app.config import settings
from app.db import get_db, User, Workflow, Memory

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/workflows/builder", tags=["Workflow Builder"])

# Node types available for workflow generation
AVAILABLE_NODES = """
Triggers: trigger_manual, trigger_schedule (cron), trigger_webhook, trigger_telegram, trigger_event (memory events)
AI Nodes: ai_agent (full agent with tools), ai_chat (simple LLM call), ai_classify, ai_extract, ai_embedding
Actions: action_memory_search, action_memory_store, action_http (HTTP request), action_exec (shell), action_send_telegram
Logic: logic_if (branching), logic_switch, logic_merge, logic_loop, logic_wait (delay)
Data: data_transform (JS code), data_set (variable), data_filter
Output: output_respond (reply to trigger), output_log
"""

SYSTEM_PROMPT = f"""You are Toup's workflow builder AI. Your job is to understand what the user needs and create agentic workflows for them.

## Your Process:
1. UNDERSTAND: Ask 2-3 focused questions to understand exactly what the user needs. Be specific — ask about their tools, frequency, triggers, etc.
2. PLAN: Once you understand, briefly describe the workflows you'll create (1-2 sentences each).
3. BUILD: Generate the workflows one by one. For each workflow, output a special JSON block that the frontend renders as a visual workflow.

## Available Node Types:
{AVAILABLE_NODES}

## When generating a workflow, output it in this exact format:
```workflow
{{
  "name": "Workflow Name",
  "description": "What this workflow does",
  "nodes": [
    {{"id": "node_1", "type": "trigger_schedule", "position": {{"x": 0, "y": 200}}, "data": {{"label": "Daily at 9am", "templateType": "trigger_schedule", "config": {{"cron": "0 9 * * *"}}}}}}
  ],
  "edges": [
    {{"id": "e1-2", "source": "node_1", "target": "node_2"}}
  ]
}}
```

## Rules:
- Nodes must use ONLY the types listed above
- Each workflow needs exactly ONE trigger node (leftmost, x=0)
- Position nodes left-to-right: x=0, x=300, x=600, etc. Keep y around 200.
- Connect nodes with edges (source → target)
- Keep workflows focused: 3-6 nodes each
- Generate 2-4 workflows based on the user's needs
- Be conversational and friendly, not robotic
- After generating all workflows, ask if they want to adjust anything

## Important:
- Do NOT generate workflows until you've asked questions and understand the user's needs
- Be specific in your questions — "What tools do you use?" is better than "Tell me more"
- Each workflow block must be valid JSON inside ```workflow``` fences
"""


class BuilderMessage(BaseModel):
    messages: list[dict]  # [{role: "user"/"assistant", content: "..."}]


async def _get_user_context(user_id: str, db: AsyncSession) -> str:
    """Get user context from onboarding memories."""
    try:
        rows = (await db.execute(
            select(Memory.content).where(
                Memory.user_id == user_id,
                Memory.brain_type == "user",
            ).limit(10)
        )).scalars().all()
        if rows:
            return "User context from their profile:\n" + "\n".join(f"- {r}" for r in rows)
    except Exception:
        pass
    return ""


async def _stream_openai(messages: list[dict]):
    """Stream response from OpenAI."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    model = getattr(settings, "default_model", "gpt-4o-mini")
    # Use a stronger model for the builder if available
    agent_model = getattr(settings, "agent_model", None)
    if agent_model and agent_model != "gpt-4o-mini":
        model = agent_model

    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=4096,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            yield delta.content


async def _stream_anthropic(messages: list[dict]):
    """Stream response from Anthropic."""
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    model = getattr(settings, "anthropic_model", "claude-opus-4-6")

    # Convert system message
    system_msg = ""
    chat_messages = []
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        else:
            chat_messages.append(m)

    async with client.messages.stream(
        model=model,
        max_tokens=4096,
        system=system_msg,
        messages=chat_messages,
    ) as stream:
        async for text in stream.text_stream:
            yield text


@router.post("/chat")
async def builder_chat(
    request: BuilderMessage,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Stream AI responses for the workflow builder conversation."""

    if not settings.openai_api_key and not settings.anthropic_api_key:
        raise HTTPException(500, "No LLM API key configured")

    # Build conversation with system prompt
    user_context = await _get_user_context(current_user.id, db)
    system_content = SYSTEM_PROMPT
    if user_context:
        system_content += f"\n\n{user_context}"

    messages = [{"role": "system", "content": system_content}]
    for msg in request.messages:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Pick provider
    use_anthropic = bool(settings.anthropic_api_key)

    async def event_stream():
        try:
            streamer = _stream_anthropic(messages) if use_anthropic else _stream_openai(messages)
            async for token in streamer:
                # SSE format
                data = json.dumps({"type": "token", "content": token})
                yield f"data: {data}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            logger.error("Builder chat stream error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


class SaveWorkflowsRequest(BaseModel):
    workflows: list[dict]  # Array of {name, description, nodes, edges}


@router.post("/save")
async def save_built_workflows(
    request: SaveWorkflowsRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Save workflows generated by the builder conversation."""
    created = []
    for wf_data in request.workflows:
        wf = Workflow(
            id=str(uuid.uuid4()),
            user_id=current_user.id,
            name=wf_data.get("name", "Untitled Workflow"),
            description=wf_data.get("description", ""),
            status="draft",
            nodes_json=json.dumps(wf_data.get("nodes", [])),
            edges_json=json.dumps(wf_data.get("edges", [])),
            run_count=0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        db.add(wf)
        created.append({"id": wf.id, "name": wf.name})

    await db.commit()
    return {"status": "saved", "count": len(created), "workflows": created}
