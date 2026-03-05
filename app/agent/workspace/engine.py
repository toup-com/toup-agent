"""
Workflow Execution Engine — runs n8n-style node graphs on the VPS agent.

The engine:
1. Loads a workflow (nodes + edges) from DB
2. Finds the trigger node (entry point)
3. Executes nodes in topological order, passing data along edges
4. Supports branching (IF/Switch), merging, and looping
5. Logs execution state for debugging

Each node receives input data and produces output data.
Template expressions like {{input.text}} are resolved at runtime.
"""

import asyncio
import json
import logging
import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from app.db.models import Workflow

logger = logging.getLogger(__name__)


class ExecutionContext:
    """Runtime context for a single workflow execution."""

    def __init__(self, workflow_id: str, trigger_data: Optional[Dict] = None):
        self.run_id = str(uuid.uuid4())
        self.workflow_id = workflow_id
        self.started_at = time.time()
        self.trigger_data = trigger_data or {}
        self.node_outputs: Dict[str, Any] = {}
        self.variables: Dict[str, Any] = {}
        self.logs: List[Dict] = []
        self.status: str = "running"
        self.error: Optional[str] = None

    def log(self, node_id: str, level: str, message: str):
        entry = {
            "ts": time.time(),
            "node_id": node_id,
            "level": level,
            "message": message,
        }
        self.logs.append(entry)
        log_fn = getattr(logger, level, logger.info)
        log_fn("[WF:%s][%s] %s", self.workflow_id[:8], node_id, message)

    def to_dict(self) -> Dict:
        return {
            "run_id": self.run_id,
            "workflow_id": self.workflow_id,
            "status": self.status,
            "started_at": self.started_at,
            "duration_ms": int((time.time() - self.started_at) * 1000),
            "node_outputs": {k: _safe_serialize(v) for k, v in self.node_outputs.items()},
            "variables": self.variables,
            "logs": self.logs,
            "error": self.error,
        }


def _safe_serialize(obj: Any, max_len: int = 2000) -> Any:
    """Safely serialize an object for logging/API output."""
    try:
        s = json.dumps(obj, default=str)
        if len(s) > max_len:
            return json.loads(s[:max_len] + '..."')
        return obj
    except Exception:
        return str(obj)[:max_len]


def resolve_template(template: str, data: Dict[str, Any]) -> str:
    """Resolve {{variable}} expressions in a template string."""
    if not isinstance(template, str):
        return template

    def replacer(match):
        expr = match.group(1).strip()
        parts = expr.split(".")
        value = data
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return match.group(0)
            if value is None:
                return ""
        return str(value) if value is not None else ""

    return re.sub(r"\{\{(.+?)\}\}", replacer, template)


def resolve_config(config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve all template expressions in a node's config."""
    resolved = {}
    for key, value in config.items():
        if isinstance(value, str):
            resolved[key] = resolve_template(value, data)
        else:
            resolved[key] = value
    return resolved


class WorkflowEngine:
    """Executes workflow node graphs."""

    def __init__(self, agent_runner=None, tool_executor=None):
        self._agent_runner = agent_runner
        self._tool_executor = tool_executor
        # Import node executors lazily
        self._executors: Optional[Dict] = None

    def _get_executors(self):
        if self._executors is None:
            from app.agent.workspace.nodes import get_node_executors
            self._executors = get_node_executors(
                agent_runner=self._agent_runner,
                tool_executor=self._tool_executor,
            )
        return self._executors

    async def execute(
        self,
        workflow: Workflow,
        trigger_data: Optional[Dict] = None,
        user_id: Optional[str] = None,
    ) -> ExecutionContext:
        """Execute a workflow and return the execution context."""
        ctx = ExecutionContext(workflow.id, trigger_data)

        try:
            nodes = json.loads(workflow.nodes_json or "[]")
            edges = json.loads(workflow.edges_json or "[]")
        except json.JSONDecodeError as e:
            ctx.status = "error"
            ctx.error = f"Invalid workflow JSON: {e}"
            return ctx

        if not nodes:
            ctx.status = "error"
            ctx.error = "Workflow has no nodes"
            return ctx

        # Build adjacency: source_id -> [(target_id, source_handle, target_handle)]
        adj: Dict[str, List[Dict]] = {}
        for edge in edges:
            src = edge.get("source")
            if src:
                adj.setdefault(src, []).append({
                    "target": edge.get("target"),
                    "sourceHandle": edge.get("sourceHandle"),
                    "targetHandle": edge.get("targetHandle"),
                })

        # Build node lookup
        node_map = {n["id"]: n for n in nodes}

        # Find trigger nodes (nodes with no incoming edges)
        targets = {e.get("target") for e in edges}
        trigger_nodes = [n for n in nodes if n["id"] not in targets]
        if not trigger_nodes:
            trigger_nodes = [nodes[0]]

        ctx.log("engine", "info", f"Starting workflow with {len(nodes)} nodes, {len(edges)} edges")

        # Execute from each trigger
        for trigger in trigger_nodes:
            try:
                await self._execute_node(ctx, trigger, adj, node_map, trigger_data or {}, user_id)
            except Exception as e:
                ctx.log(trigger["id"], "error", f"Trigger execution failed: {e}")
                ctx.status = "error"
                ctx.error = str(e)
                return ctx

        if ctx.status == "running":
            ctx.status = "completed"

        # Update workflow run stats
        try:
            from app.db.database import async_session_maker
            from sqlalchemy import select
            async with async_session_maker() as db:
                result = await db.execute(select(Workflow).where(Workflow.id == workflow.id))
                wf = result.scalar_one_or_none()
                if wf:
                    wf.run_count = (wf.run_count or 0) + 1
                    wf.last_run_at = datetime.utcnow()
                    await db.commit()
        except Exception as e:
            logger.warning("Could not update workflow run stats: %s", e)

        ctx.log("engine", "info", f"Workflow completed in {int((time.time() - ctx.started_at) * 1000)}ms")
        return ctx

    async def _execute_node(
        self,
        ctx: ExecutionContext,
        node: Dict,
        adj: Dict[str, List[Dict]],
        node_map: Dict[str, Dict],
        input_data: Dict[str, Any],
        user_id: Optional[str] = None,
    ):
        """Execute a single node and recurse to its outputs."""
        node_id = node["id"]
        node_data = node.get("data", {})
        template_type = node_data.get("templateType", "")
        config = node_data.get("config", {})

        # Resolve template expressions in config
        resolved_config = resolve_config(config, {"input": input_data, "vars": ctx.variables})

        ctx.log(node_id, "info", f"Executing {template_type}")

        # Get the executor for this node type
        executors = self._get_executors()
        executor = executors.get(template_type)

        if executor is None:
            ctx.log(node_id, "warning", f"No executor for node type: {template_type}")
            output = input_data  # pass through
        else:
            try:
                output = await executor(
                    config=resolved_config,
                    input_data=input_data,
                    ctx=ctx,
                    node_id=node_id,
                    user_id=user_id,
                )
            except Exception as e:
                ctx.log(node_id, "error", f"Node execution failed: {e}")
                ctx.status = "error"
                ctx.error = f"Node {node_id} ({template_type}): {e}"
                return

        ctx.node_outputs[node_id] = output

        # Follow outgoing edges
        for edge_info in adj.get(node_id, []):
            target_id = edge_info.get("target")
            source_handle = edge_info.get("sourceHandle")
            target_node = node_map.get(target_id)

            if not target_node:
                continue

            # For branching nodes (IF/Switch), only follow the matching handle
            if template_type == "logic_if" and isinstance(output, dict):
                branch = output.get("_branch")
                if branch is not None and source_handle != branch:
                    continue
                # Pass the actual data, not the branch marker
                branch_data = output.get("data", output)
                await self._execute_node(ctx, target_node, adj, node_map, branch_data, user_id)
            elif template_type == "logic_switch" and isinstance(output, dict):
                route = output.get("_route")
                if route is not None and source_handle != route:
                    continue
                route_data = output.get("data", output)
                await self._execute_node(ctx, target_node, adj, node_map, route_data, user_id)
            elif template_type == "logic_loop" and isinstance(output, dict):
                items = output.get("_items", [])
                if source_handle == "item":
                    for item in items:
                        await self._execute_node(ctx, target_node, adj, node_map, item, user_id)
                elif source_handle == "done":
                    await self._execute_node(ctx, target_node, adj, node_map, {"items": items}, user_id)
            else:
                await self._execute_node(ctx, target_node, adj, node_map, output if isinstance(output, dict) else {"result": output}, user_id)
