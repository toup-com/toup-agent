"""
Dashboard API — runs on the VPS agent (data owner).

Provides task management, inbox, docs, alerts, goals, activity log,
and agent status. Data stored as JSON files in the agent workspace.
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Dashboard"])

# ── Storage helpers ──────────────────────────────────────────────────

def _data_dir() -> str:
    base = getattr(settings, "agent_workspace_dir", None) or "./workspace"
    d = os.path.join(base, ".dashboard")
    os.makedirs(d, exist_ok=True)
    return d


def _read_json(name: str, default=None):
    path = os.path.join(_data_dir(), f"{name}.json")
    if not os.path.exists(path):
        return default if default is not None else []
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else []


def _write_json(name: str, data):
    path = os.path.join(_data_dir(), f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ── Schemas ──────────────────────────────────────────────────────────

class InboxCreate(BaseModel):
    id: Optional[str] = None
    title: str = ""
    source: str = "manual"
    raw: Optional[str] = ""
    created_at: Optional[str] = None
    processed: bool = False


class DocSave(BaseModel):
    name: str
    content: str




# ── Combined dashboard endpoint ──────────────────────────────────────

@router.get("/dashboard")
async def get_dashboard():
    """Return combined dashboard data (status, goals, alerts).

    Tasks/jobs are now served from the build_jobs DB table via /apps/jobs/.
    Stats are computed client-side from jobs.
    """
    goals = _read_json("goals", [])
    alerts = [a for a in _read_json("alerts", []) if not a.get("dismissed")]

    return {
        "tasks": [],  # Deprecated — use /apps/jobs/ instead
        "status": {"state": "running", "status": "active"},
        "stats": {"today": {}, "week": {}, "allTime": {}, "byCategory": {}},
        "goals": goals,
        "alerts": alerts,
    }


# ── Tasks (REMOVED — canonical source is build_jobs DB table via /apps/jobs/) ──


# ── Agent status ─────────────────────────────────────────────────────

@router.get("/status")
async def get_status():
    return {"state": "running", "status": "active", "current_task": None}


# ── Agents (sub-agents list) ────────────────────────────────────────

@router.get("/agents")
async def list_agents():
    agents = _read_json("agents", None)
    if agents is None:
        return [
            {"name": "agent", "role": "Toup Agent", "emoji": "robot", "status": "idle"},
        ]
    return agents


# ── Activity log REMOVED — nothing wrote to .dashboard/activity.json ──


# ── Stats ────────────────────────────────────────────────────────────

@router.get("/stats")
async def get_stats():
    """Stats are now computed client-side from build_jobs. Kept for backward compat."""
    return {
        "today": {"done": 0, "added": 0},
        "week": {"done": 0, "added": 0},
        "allTime": {"done": 0, "added": 0, "total": 0},
        "byCategory": {},
    }


# ── Goals ────────────────────────────────────────────────────────────

@router.get("/goals")
async def list_goals():
    return _read_json("goals", [])


# ── Alerts ───────────────────────────────────────────────────────────

@router.get("/alerts")
async def list_alerts():
    return _read_json("alerts", [])


@router.post("/alerts/generate")
async def generate_alerts():
    """Auto-generate alerts based on agent state. Placeholder for now."""
    return {"ok": True}


@router.post("/alerts/{alert_id}/dismiss")
async def dismiss_alert(alert_id: str):
    alerts = _read_json("alerts", [])
    for a in alerts:
        if a.get("id") == alert_id:
            a["dismissed"] = True
    _write_json("alerts", alerts)
    return {"ok": True}


# ── Inbox ────────────────────────────────────────────────────────────

@router.get("/inbox")
async def list_inbox():
    return _read_json("inbox", [])


@router.post("/inbox")
async def create_inbox_item(item: InboxCreate):
    items = _read_json("inbox", [])
    d = item.model_dump()
    if not d.get("id"):
        d["id"] = f"inbox-{uuid.uuid4().hex[:8]}"
    if not d.get("created_at"):
        d["created_at"] = datetime.now(timezone.utc).isoformat()
    items.append(d)
    _write_json("inbox", items)
    return {"id": d["id"]}


@router.delete("/inbox/{item_id}")
async def delete_inbox_item(item_id: str):
    items = _read_json("inbox", [])
    items = [i for i in items if i.get("id") != item_id]
    _write_json("inbox", items)
    return {"ok": True}


# ── Jobs/delegate REMOVED — tasks execute via /apps/jobs/ + AgentRunner ──


# ── Docs ─────────────────────────────────────────────────────────────

@router.get("/docs")
async def list_docs():
    return _read_json("docs", [])


@router.post("/docs")
async def save_doc(doc: DocSave):
    docs = _read_json("docs", [])
    now = datetime.now(timezone.utc).isoformat()
    existing = next((d for d in docs if d.get("name") == doc.name), None)
    if existing:
        existing["content"] = doc.content
        existing["updated_at"] = now
    else:
        docs.append({
            "id": f"doc-{uuid.uuid4().hex[:8]}",
            "name": doc.name,
            "content": doc.content,
            "created_at": now,
            "updated_at": now,
        })
    _write_json("docs", docs)
    return docs[-1] if not existing else existing


@router.delete("/docs")
async def delete_doc(name: str = Query(...)):
    docs = _read_json("docs", [])
    docs = [d for d in docs if d.get("name") != name]
    _write_json("docs", docs)
    return {"ok": True}


# ── Memory files (workspace .md files) ──────────────────────────────

@router.get("/memory/{directory:path}")
async def list_memory_dir(directory: str):
    base = getattr(settings, "agent_workspace_dir", None) or "./workspace"
    target = os.path.join(base, directory)
    if not os.path.isdir(target):
        return []
    result = []
    for name in sorted(os.listdir(target)):
        if name.startswith("."):
            continue
        full = os.path.join(target, name)
        if os.path.isfile(full) and name.endswith(".md"):
            try:
                with open(full, "r") as f:
                    lines = f.readlines()
                entries = sum(1 for l in lines if l.strip().startswith("- "))
                result.append({"name": name, "entries": entries})
            except Exception:
                result.append({"name": name, "entries": 0})
    return result


@router.get("/memory/file/{filepath:path}")
async def get_memory_file(filepath: str):
    base = getattr(settings, "agent_workspace_dir", None) or "./workspace"
    target = os.path.join(base, filepath)
    if not os.path.isfile(target):
        raise HTTPException(404, "File not found")
    with open(target, "r") as f:
        return {"content": f.read()}


@router.get("/memory/search")
async def search_memory(q: str = Query("")):
    if not q:
        return []
    base = getattr(settings, "agent_workspace_dir", None) or "./workspace"
    results = []
    q_lower = q.lower()
    for root, _, files in os.walk(base):
        for name in files:
            if not name.endswith(".md"):
                continue
            full = os.path.join(root, name)
            try:
                with open(full, "r") as f:
                    content = f.read()
                if q_lower in content.lower():
                    # Find first matching line as snippet
                    for line in content.split("\n"):
                        if q_lower in line.lower():
                            rel = os.path.relpath(full, base)
                            results.append({"file": rel, "snippet": line.strip()[:200]})
                            break
            except Exception:
                continue
            if len(results) >= 20:
                break
    return results


@router.post("/memory/entry")
async def add_memory_entry(req: dict):
    file_path = req.get("file", "")
    entry = req.get("entry", "")
    if not file_path or not entry:
        raise HTTPException(400, "file and entry are required")
    base = getattr(settings, "agent_workspace_dir", None) or "./workspace"
    target = os.path.join(base, file_path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    with open(target, "a") as f:
        f.write(f"\n- [{now}] {entry}")
    return {"ok": True}
