"""
App Builder — Conversational AI that asks questions then generates React apps.

Uses SSE (Server-Sent Events) to stream the AI conversation to the frontend.
The AI asks clarifying questions about the user's needs, then generates
React component code that renders as a live preview via Sandpack.
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
router = APIRouter(prefix="/workflows/builder", tags=["App Builder"])


SYSTEM_PROMPT = """You are Toup's app builder AI. You build custom React apps for users based on their needs — like v0 or Bolt.

## Your Process:
1. UNDERSTAND: Ask 2-3 focused questions to understand exactly what the user needs. Ask about their specific use case, what data they want to track, what features matter most.
2. PLAN: Once you understand, briefly describe the app you'll create (2-3 sentences).
3. BUILD: Generate a complete React component. The code renders live in the user's browser via Sandpack.

## Code Rules:
- Output code in a single ```app code fence (NOT ```tsx or ```jsx, use ```app)
- Export a default function component called `App`
- Use ONLY inline styles (style={{...}}) — no CSS imports, no className with Tailwind (Tailwind CDN is loaded but inline styles are more reliable in Sandpack)
- Use React hooks (useState, useEffect, useRef, useMemo, useCallback) — they're available
- NO external imports — no axios, no lodash, no external packages. Only React is available.
- Build a COMPLETE, functional app — not a skeleton or placeholder
- Include realistic sample data so the app looks populated and useful
- Use a modern dark theme by default (dark backgrounds #0f172a, #1e293b, white/gray text)
- Make it responsive and polished — this should look like a real product
- Include interactive elements: buttons that work, inputs that filter, tabs that switch, etc.

## Design Guidelines:
- Use a clean, modern design with good spacing and visual hierarchy
- Use color accents sparingly (violet/purple #8b5cf6 for primary actions)
- Include icons as emoji or Unicode symbols (not imported icon libraries)
- Add smooth hover effects and transitions via inline styles
- Make the app feel complete — include headers, stats cards, lists, forms as appropriate

## Example Output:
```app
export default function App() {
  const [activeTab, setActiveTab] = React.useState('dashboard');
  const [items, setItems] = React.useState([
    { id: 1, name: 'Example Item', status: 'active' },
  ]);

  return (
    <div style={{ minHeight: '100vh', background: '#0f172a', color: '#e2e8f0', fontFamily: 'system-ui' }}>
      {/* Header */}
      <header style={{ padding: '1rem 2rem', borderBottom: '1px solid #1e293b' }}>
        <h1 style={{ fontSize: '1.25rem', fontWeight: 700 }}>My App</h1>
      </header>
      {/* Content */}
      <main style={{ padding: '2rem' }}>
        {/* ... full app content ... */}
      </main>
    </div>
  );
}
```

## Important:
- Do NOT generate code until you've asked questions and understand the user's needs
- Be specific in your questions — "What data do you want to track?" is better than "Tell me more"
- When iterating, output the FULL updated component (not just the changed parts)
- After generating, ask if they want to adjust anything (colors, features, layout, etc.)
- Keep conversation friendly and concise — you're a builder, not a lecturer
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
    agent_model = getattr(settings, "agent_model", None)
    if agent_model and agent_model != "gpt-4o-mini":
        model = agent_model

    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=8192,
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

    system_msg = ""
    chat_messages = []
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        else:
            chat_messages.append(m)

    async with client.messages.stream(
        model=model,
        max_tokens=8192,
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
    """Stream AI responses for the app builder conversation."""

    if not settings.openai_api_key and not settings.anthropic_api_key:
        raise HTTPException(500, "No LLM API key configured")

    user_context = await _get_user_context(current_user.id, db)
    system_content = SYSTEM_PROMPT
    if user_context:
        system_content += f"\n\n{user_context}"

    messages = [{"role": "system", "content": system_content}]
    for msg in request.messages:
        messages.append({"role": msg["role"], "content": msg["content"]})

    use_anthropic = bool(settings.anthropic_api_key)

    async def event_stream():
        try:
            streamer = _stream_anthropic(messages) if use_anthropic else _stream_openai(messages)
            async for token in streamer:
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


# ── Save endpoints ────────────────────────────────────────────────

class SaveWorkflowsRequest(BaseModel):
    workflows: list[dict]

@router.post("/save")
async def save_built_workflows(
    request: SaveWorkflowsRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Save workflows generated by the builder (legacy)."""
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


class SaveAppRequest(BaseModel):
    name: str
    code: str

@router.post("/save-app")
async def save_built_app(
    request: SaveAppRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Save an app generated by the builder as a workflow with embedded code."""
    wf = Workflow(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        name=request.name,
        description="Generated app",
        status="draft",
        nodes_json=json.dumps([{
            "id": "app",
            "type": "app_component",
            "position": {"x": 0, "y": 0},
            "data": {
                "label": request.name,
                "templateType": "app_component",
                "code": request.code,
                "config": {},
            },
        }]),
        edges_json="[]",
        run_count=0,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(wf)
    await db.commit()
    return {"status": "saved", "id": wf.id, "name": wf.name}
