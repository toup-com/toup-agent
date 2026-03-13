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

# ── App skill auto-registration ─────────────────────────────────
_skill_loader = None


def set_skill_loader(loader):
    """Called by agent_main.py to wire skill registration on workflow CRUD."""
    global _skill_loader
    _skill_loader = loader


async def _maybe_register_app_skill(workflow):
    """If the workflow has an app_component node, register an AppSkill."""
    if not _skill_loader:
        return
    try:
        nodes = json.loads(workflow.nodes_json or "[]")
        from app.agent.skills.builtins.app_skill import AppSkill, slugify, _find_app_node
        node = _find_app_node(nodes)
        if node:
            slug = slugify(workflow.name, workflow.id)
            skill = AppSkill(workflow.id, workflow.name, slug)
            await _skill_loader.register_dynamic(skill)
            logger.info("[APP-SKILL] Registered skill for app '%s'", workflow.name)
    except Exception as e:
        logger.warning("[APP-SKILL] Registration failed: %s", e)


async def _maybe_unload_app_skill(workflow):
    """Unload the app skill for a deleted workflow."""
    if not _skill_loader:
        return
    try:
        from app.agent.skills.builtins.app_skill import slugify
        slug = slugify(workflow.name, workflow.id)
        skill_name = f"app_{slug}"
        await _skill_loader.unload_skill(skill_name)
    except Exception as e:
        logger.debug("[APP-SKILL] Unload skipped: %s", e)


# ── Request / Response schemas ───────────────────────────────────────

class WorkflowCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    nodes_json: Optional[str] = None
    edges_json: Optional[str] = None


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
        nodes_json=req.nodes_json or "[]",
        edges_json=req.edges_json or "[]",
        run_count=0,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(w)
    await db.commit()
    await db.refresh(w)
    await _maybe_register_app_skill(w)
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
    await _maybe_register_app_skill(w)
    return _to_out(w)


@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a workflow."""
    result = await db.execute(select(Workflow).where(Workflow.id == workflow_id))
    w = result.scalar_one_or_none()
    if not w:
        raise HTTPException(404, "Workflow not found")
    await _maybe_unload_app_skill(w)
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

BUILDER_SYSTEM_PROMPT = """You are Toup's app builder AI. You build full-stack React apps — like Lovable, v0, or Bolt.

## Your Process:
1. UNDERSTAND: Ask 1-2 focused questions. What are they building? What data do they track? Keep it brief.
2. BUILD: Generate a complete multi-file React app. The code renders live in the user's browser.

## Multi-File Output Format:
Output each file in a separate fenced block tagged with its path:

```file:/App.jsx
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
export default function App() { return <BrowserRouter>...</BrowserRouter>; }
```

```file:/pages/Dashboard.jsx
import { useState } from 'react';
export default function Dashboard() { return <div>...</div>; }
```

```file:/components/Sidebar.jsx
export default function Sidebar({ items }) { return <nav>...</nav>; }
```

```file:/lib/data.js
export const sampleData = [...];
```

## Available Packages (import freely):
- react, react-dom, react-router-dom (routing)
- lucide-react (icons: import { Home, Settings, Users, Search, Plus, Trash2, Edit, Check, X, ChevronRight, Bell, Star, Heart, BarChart3, Calendar, Mail, Phone, MapPin, Clock, Filter, Download, Upload, Share2, MoreHorizontal, ArrowLeft, ArrowRight, LogOut, Moon, Sun, Menu, Globe, Zap, Shield, Award, TrendingUp, DollarSign, ShoppingCart, Package, Layers, GitBranch, Activity, Eye, EyeOff, Lock, Unlock, Bookmark, Tag, Folder, File, Image, Video, Music, Mic, Camera, Send, MessageSquare, MessageCircle, UserPlus, UserMinus, AlertCircle, AlertTriangle, Info, HelpCircle, ExternalLink, Copy, Clipboard, RefreshCw, RotateCcw, Save, Printer, Wifi, WifiOff, Battery, BatteryCharging, Cloud, CloudOff, Database, Server, Terminal, Code, GitCommit, GitPullRequest, GitMerge } from 'lucide-react')
- recharts (charts: LineChart, BarChart, PieChart, AreaChart, etc.)
- date-fns (date utilities)
- clsx (conditional classNames)
- @supabase/supabase-js (if user wants backend/auth — use placeholder URL)
- uuid (for generating IDs)
- zustand (state management for complex apps)

When you use a package, also output a dependencies block.
IMPORTANT: If using recharts, you MUST also include "react-is": "latest" (peer dependency).
```dependencies
{
  "lucide-react": "latest",
  "recharts": "latest",
  "react-is": "latest",
  "react-router-dom": "latest"
}
```

## Styling:
- Use Tailwind CSS classes (className="...") — Tailwind is available via CDN
- Dark theme by default: bg-slate-950, bg-slate-900, bg-slate-800, text-white, text-slate-400
- Accent: violet-500, violet-600 for primary; emerald-500 for success; red-500 for danger
- Use modern patterns: rounded-xl, shadow-lg, backdrop-blur, gradients
- Responsive: use sm:, md:, lg: breakpoints

## Code Rules:
- ALWAYS use ```file:/path``` blocks — NEVER use ```app, ```tsx, or ```jsx alone
- /App.jsx is the entry point — MUST export default
- Split into pages/, components/, lib/ directories for apps with 3+ components
- Use named imports: import { useState, useEffect } from 'react'
- Use lucide-react icons instead of emoji
- Include realistic sample data in lib/data.js
- Build COMPLETE, functional apps — not skeletons
- Interactive: buttons work, inputs filter, tabs switch, forms submit
- Polished: hover effects, transitions, proper spacing, visual hierarchy

## File Structure for a typical app:
/App.jsx — entry point with routing
/pages/Dashboard.jsx — main page
/pages/Settings.jsx — settings page (if needed)
/components/Sidebar.jsx — navigation sidebar
/components/Header.jsx — top header bar
/components/Card.jsx — reusable card component
/lib/data.js — sample data and constants
/lib/utils.js — helper functions

## Iterative Edits:
- When user asks for changes, ONLY output the files that changed
- DO NOT re-output unchanged files
- Always include the full content of changed files (not partial diffs)

## Agent Integration:
Your app will be automatically integrated with the user's AI agent.
The agent can inspect and modify all files. Write clean, modular code
with clear file separation so changes are easy to apply per-file.

## App Name:
When you build an app, ALWAYS output a short, catchy app name at the start of your response:
```appname
My Cool App
```
This name is used to save the app in the user's workspace. Keep it short (2-4 words), descriptive, and title-cased.

## Important:
- Keep questions minimal (1-2 max), then BUILD immediately
- After generating, ask "Want to adjust anything?"
- Keep conversation concise — you're a builder, not a lecturer
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
    model = getattr(settings, "anthropic_model", "claude-opus-4-6")

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
        getattr(settings, "anthropic_model", "claude-opus-4-6")
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


# ── Builder suggestions (personalized from user memories) ─────

SUGGESTIONS_PROMPT = """Based on this user's goals and context, suggest 4 specific app ideas they would find useful. Each suggestion should be a short phrase (under 60 chars) describing a concrete app.

User context:
{context}

Return a JSON object with a "suggestions" key containing an array of 4 strings. Example:
{{"suggestions": ["IELTS vocabulary flashcard trainer", "Speaking practice timer with scoring", "Essay structure planner", "Listening test simulator"]}}"""


@router.get("/builder/suggestions")
async def builder_suggestions(db: AsyncSession = Depends(get_db)):
    """Return personalized app suggestions based on user memories."""
    fallback = [
        "Project management dashboard with kanban board",
        "Personal CRM to track clients and deals",
        "Habit tracker with streaks and analytics",
        "E-commerce admin panel with order management",
    ]

    try:
        from app.db.models import Memory

        # Read user memories — goals first, then general
        rows = (await db.execute(
            select(Memory.content).where(
                Memory.user_id == settings.user_id,
                Memory.brain_type == "user",
            ).order_by(
                # goals category first
                Memory.category.desc(),
            ).limit(15)
        )).scalars().all()

        if not rows:
            logger.info("Builder suggestions: no memories found for user %s", settings.user_id[:8])
            return {"suggestions": fallback, "personalized": False}

        context = "\n".join(f"- {r}" for r in rows)
        logger.info("Builder suggestions: found %d memories, using context (%d chars)", len(rows), len(context))

        # Try LLM-based suggestions
        provider = _get_strongest_provider()
        if provider == "none":
            logger.warning("Builder suggestions: no LLM provider available")
            return {"suggestions": fallback, "personalized": False}

        suggestions = None

        if provider == "anthropic":
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
            # Use Haiku for fast, cheap suggestions (no need for Opus here)
            model = "claude-opus-4-6"
            resp = await client.messages.create(
                model=model, max_tokens=300,
                messages=[{"role": "user", "content": SUGGESTIONS_PROMPT.format(context=context)}],
            )
            text = resp.content[0].text if resp.content else "[]"
            if "[" in text:
                suggestions = json.loads(text[text.index("["):text.rindex("]") + 1])
        else:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=settings.openai_api_key)
            model = getattr(settings, "agent_model", "gpt-4o-mini")
            try:
                resp = await client.chat.completions.create(
                    model=model, max_completion_tokens=300,
                    messages=[{"role": "user", "content": SUGGESTIONS_PROMPT.format(context=context)}],
                    response_format={"type": "json_object"},
                )
            except Exception as oai_err:
                logger.info("Builder suggestions: max_completion_tokens failed (%s), retrying with max_tokens", oai_err)
                resp = await client.chat.completions.create(
                    model=model, max_tokens=300, temperature=0.7,
                    messages=[{"role": "user", "content": SUGGESTIONS_PROMPT.format(context=context)}],
                    response_format={"type": "json_object"},
                )
            text = resp.choices[0].message.content or "[]"
            logger.info("Builder suggestions: raw OpenAI response: %s", text[:500])
            data = json.loads(text)
            suggestions = data if isinstance(data, list) else data.get("suggestions", data.get("ideas", data.get("apps", [])))

        if suggestions and len(suggestions) >= 3:
            logger.info("Builder suggestions: generated %d personalized suggestions", len(suggestions))
            return {"suggestions": suggestions[:4], "personalized": True}
        logger.warning("Builder suggestions: LLM returned insufficient suggestions: %s", suggestions)

    except Exception as e:
        logger.warning("Builder suggestions failed: %s", e, exc_info=True)

    return {"suggestions": fallback, "personalized": False}


# ── Workspace Generation (post-onboarding) ──────────────────────

@router.post("/generate-from-onboarding")
async def generate_workspace_from_onboarding(
    background_tasks: BackgroundTasks,
):
    """Auto-generate personalized workflows from onboarding goals. Runs in background."""
    from app.agent.workspace.workspace_generator import generate_workspace
    background_tasks.add_task(generate_workspace, settings.user_id)
    return {"status": "generating", "message": "Workspace generation started"}


class GenerateRequest(BaseModel):
    description: str


@router.post("/generate")
async def generate_workspace_from_description(
    request: GenerateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generate personalized workflows from a user description."""
    from app.agent.workspace.workspace_generator import (
        load_template_packs, match_packs, _keyword_match,
    )

    user_id = settings.user_id
    description = request.description.strip()
    if not description:
        raise HTTPException(400, "Description is required")

    # Guard: skip if workflows already exist
    existing = (await db.execute(
        select(Workflow.id).where(Workflow.user_id == user_id).limit(1)
    )).scalar_one_or_none()
    if existing:
        return {"status": "exists", "message": "Workflows already exist"}

    # Match template packs
    packs = load_template_packs()
    if not packs:
        raise HTTPException(500, "No template packs available")

    try:
        matched = await match_packs(description)
    except Exception:
        matched = _keyword_match(description, packs)

    # Create workflows in DB
    created = []
    for pack_name in matched:
        pack = packs.get(pack_name)
        if not pack:
            continue
        for wf_template in pack.get("workflows", []):
            wf = Workflow(
                id=str(uuid.uuid4()),
                user_id=user_id,
                name=wf_template["name"],
                description=wf_template.get("description", ""),
                status="draft",
                nodes_json=json.dumps(wf_template["nodes_json"]),
                edges_json=json.dumps(wf_template["edges_json"]),
                run_count=0,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            db.add(wf)
            created.append({"id": wf.id, "name": wf.name})

    await db.commit()
    return {"status": "created", "count": len(created), "packs": matched, "workflows": created}
