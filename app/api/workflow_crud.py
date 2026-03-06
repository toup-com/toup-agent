"""
Workflow CRUD API — runs on the VPS agent (data owner).

Provides full CRUD for workflows stored in the local toup_brain database.
The platform proxies all workflow requests here via X-Agent-Key auth.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.db.models import Workflow
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/workflows", tags=["Workflows"])


# ── Request / Response schemas ───────────────────────────────────────

class WorkflowCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None


class WorkflowUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    nodes_json: Optional[str] = None
    edges_json: Optional[str] = None


class WorkflowOut(BaseModel):
    id: str
    name: str
    description: Optional[str]
    status: str
    nodes_json: str
    edges_json: str
    run_count: int
    last_run_at: Optional[str]
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


def _to_out(w: Workflow) -> dict:
    return {
        "id": w.id,
        "name": w.name,
        "description": w.description,
        "status": w.status,
        "nodes_json": w.nodes_json,
        "edges_json": w.edges_json,
        "run_count": w.run_count,
        "last_run_at": w.last_run_at.isoformat() if w.last_run_at else None,
        "created_at": w.created_at.isoformat() if w.created_at else None,
        "updated_at": w.updated_at.isoformat() if w.updated_at else None,
    }


# ── Routes ───────────────────────────────────────────────────────────

@router.get("")
@router.get("/")
async def list_workflows(db: AsyncSession = Depends(get_db)):
    """List all workflows for the agent owner."""
    user_id = settings.user_id
    query = select(Workflow).order_by(Workflow.updated_at.desc())
    if user_id:
        query = query.where(Workflow.user_id == user_id)
    result = await db.execute(query)
    workflows = result.scalars().all()
    return [_to_out(w) for w in workflows]


@router.get("/{workflow_id}")
async def get_workflow(workflow_id: str, db: AsyncSession = Depends(get_db)):
    """Get a single workflow by ID."""
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    w = result.scalar_one_or_none()
    if not w:
        raise HTTPException(404, "Workflow not found")
    return _to_out(w)


@router.post("")
@router.post("/")
async def create_workflow(req: WorkflowCreate, db: AsyncSession = Depends(get_db)):
    """Create a new workflow."""
    w = Workflow(
        id=str(uuid.uuid4()),
        user_id=settings.user_id,
        name=req.name,
        description=req.description,
        status="draft",
        nodes_json="[]",
        edges_json="[]",
        run_count=0,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(w)
    await db.commit()
    await db.refresh(w)
    return _to_out(w)


@router.put("/{workflow_id}")
async def update_workflow(
    workflow_id: str, req: WorkflowUpdate, db: AsyncSession = Depends(get_db)
):
    """Update a workflow (name, description, status, nodes, edges)."""
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    w = result.scalar_one_or_none()
    if not w:
        raise HTTPException(404, "Workflow not found")

    if req.name is not None:
        w.name = req.name
    if req.description is not None:
        w.description = req.description
    if req.status is not None:
        w.status = req.status
    if req.nodes_json is not None:
        w.nodes_json = req.nodes_json
    if req.edges_json is not None:
        w.edges_json = req.edges_json
    w.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(w)
    return _to_out(w)


@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a workflow."""
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    w = result.scalar_one_or_none()
    if not w:
        raise HTTPException(404, "Workflow not found")
    await db.delete(w)
    await db.commit()
    return {"status": "deleted", "id": workflow_id}


@router.post("/{workflow_id}/duplicate")
async def duplicate_workflow(workflow_id: str, db: AsyncSession = Depends(get_db)):
    """Duplicate a workflow."""
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    original = result.scalar_one_or_none()
    if not original:
        raise HTTPException(404, "Workflow not found")

    copy = Workflow(
        id=str(uuid.uuid4()),
        user_id=original.user_id,
        name=f"{original.name} (Copy)",
        description=original.description,
        status="draft",
        nodes_json=original.nodes_json,
        edges_json=original.edges_json,
        run_count=0,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(copy)
    await db.commit()
    await db.refresh(copy)
    return _to_out(copy)


# ── Execution ────────────────────────────────────────────────────

# Module-level reference set by agent_main.py lifespan
_workflow_engine = None


def set_workflow_engine(engine):
    """Set the workflow engine reference (called from agent_main.py)."""
    global _workflow_engine
    _workflow_engine = engine


@router.post("/{workflow_id}/run")
async def run_workflow(
    workflow_id: str,
    req: Optional[dict] = None,
    db: AsyncSession = Depends(get_db),
):
    """Execute a workflow and return the results."""
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    w = result.scalar_one_or_none()
    if not w:
        raise HTTPException(404, "Workflow not found")

    if not _workflow_engine:
        raise HTTPException(503, "Workflow engine not initialized")

    trigger_data = req or {}
    ctx = await _workflow_engine.execute(w, trigger_data=trigger_data, user_id=settings.user_id)
    return ctx.to_dict()


# ── App Builder (AI chat → SSE stream) ───────────────────────────

BUILDER_SYSTEM_PROMPT = """You are Toup's app builder AI. You build custom React apps for users based on their needs — like v0 or Bolt.

## Your Process:
1. UNDERSTAND: Ask 2-3 focused questions to understand exactly what the user needs. Ask about their specific use case, what data they want to track, what features matter most.
2. PLAN: Once you understand, briefly describe the app you'll create (2-3 sentences).
3. BUILD: Generate a complete React component. The code renders live in the user's browser via Sandpack.

## Code Rules:
- Output code in a single ```app code fence (NOT ```tsx or ```jsx, use ```app)
- Export a default function component called `App`
- Use ONLY inline styles (style={{...}}) — no CSS imports, no className with Tailwind
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
      <header style={{ padding: '1rem 2rem', borderBottom: '1px solid #1e293b' }}>
        <h1 style={{ fontSize: '1.25rem', fontWeight: 700 }}>My App</h1>
      </header>
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


class BuilderChatRequest(BaseModel):
    messages: list[dict]  # [{role: "user"/"assistant", content: "..."}]


async def _stream_openai(messages: list[dict]):
    """Stream from OpenAI using the agent's configured model."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    model = getattr(settings, "agent_model", "gpt-4o-mini")

    try:
        stream = await client.chat.completions.create(
            model=model, messages=messages,
            max_completion_tokens=16384, stream=True,
        )
    except Exception:
        stream = await client.chat.completions.create(
            model=model, messages=messages,
            temperature=0.7, max_tokens=16384, stream=True,
        )

    async for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            yield delta.content


async def _stream_anthropic(messages: list[dict]):
    """Stream from Anthropic using the agent's strongest model."""
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    model = getattr(settings, "anthropic_model", "claude-sonnet-4-20250514")

    system_msg = ""
    chat_messages = []
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        else:
            chat_messages.append(m)

    async with client.messages.stream(
        model=model, max_tokens=16384,
        system=system_msg, messages=chat_messages,
    ) as stream:
        async for text in stream.text_stream:
            yield text


def _get_strongest_provider() -> str:
    """Return 'anthropic' or 'openai' based on available keys."""
    if getattr(settings, "anthropic_api_key", None):
        return "anthropic"
    if getattr(settings, "openai_api_key", None):
        return "openai"
    return "none"


@router.post("/builder/chat")
async def builder_chat(request: BuilderChatRequest, db: AsyncSession = Depends(get_db)):
    """Stream AI app builder responses. Runs on the VPS agent with user's own API keys."""
    provider = _get_strongest_provider()
    if provider == "none":
        raise HTTPException(500, "No LLM API key configured on this agent")

    # Get user context from memories
    user_context = ""
    try:
        from app.db.models import Memory
        rows = (await db.execute(
            select(Memory.content).where(
                Memory.user_id == settings.user_id,
                Memory.brain_type == "user",
            ).limit(10)
        )).scalars().all()
        if rows:
            user_context = "\nUser context:\n" + "\n".join(f"- {r}" for r in rows)
    except Exception:
        pass

    system_content = BUILDER_SYSTEM_PROMPT + user_context
    messages = [{"role": "system", "content": system_content}]
    for msg in request.messages:
        messages.append({"role": msg["role"], "content": msg["content"]})

    model_name = (
        getattr(settings, "anthropic_model", "claude-sonnet-4-20250514")
        if provider == "anthropic"
        else getattr(settings, "agent_model", "gpt-4o-mini")
    )

    async def event_stream():
        try:
            streamer = _stream_anthropic(messages) if provider == "anthropic" else _stream_openai(messages)
            # Send model info as first event
            yield f"data: {json.dumps({'type': 'meta', 'model': model_name, 'provider': provider})}\n\n"
            async for token in streamer:
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            logger.error("Builder chat stream error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Workspace Generation (post-onboarding) ──────────────────────

@router.post("/generate-from-onboarding")
async def generate_workspace_from_onboarding(
    background_tasks: BackgroundTasks,
):
    """Auto-generate personalized workflows from onboarding goals. Runs in background."""
    from app.agent.workspace.workspace_generator import generate_workspace
    background_tasks.add_task(generate_workspace, settings.user_id)
    return {"status": "generating", "message": "Workspace generation started"}
