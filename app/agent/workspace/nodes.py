"""
Node executors — one async function per node type.

Each executor receives:
  config:     dict — resolved node configuration (template expressions already evaluated)
  input_data: dict — output from the previous node
  ctx:        ExecutionContext — shared execution state
  node_id:    str — this node's ID (for logging)
  user_id:    str | None — owner user ID

Returns: dict — the output data passed to downstream nodes
"""

import asyncio
import json
import logging
import subprocess
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Type alias for node executor functions
NodeExecutor = Callable[..., Any]


def get_node_executors(agent_runner=None, tool_executor=None) -> Dict[str, NodeExecutor]:
    """Build and return the executor registry."""

    executors: Dict[str, NodeExecutor] = {}

    # ── Triggers ────────────────────────────────────────────────────

    async def trigger_manual(config, input_data, ctx, node_id, user_id=None, **kw):
        ctx.log(node_id, "info", "Manual trigger fired")
        return input_data or {}

    async def trigger_schedule(config, input_data, ctx, node_id, user_id=None, **kw):
        ctx.log(node_id, "info", f"Schedule trigger (cron: {config.get('cron', '?')})")
        return {"triggered_at": ctx.started_at, "cron": config.get("cron", "")}

    async def trigger_webhook(config, input_data, ctx, node_id, user_id=None, **kw):
        ctx.log(node_id, "info", f"Webhook trigger ({config.get('method', 'POST')} {config.get('path', '/')})")
        return input_data or {}

    async def trigger_telegram(config, input_data, ctx, node_id, user_id=None, **kw):
        ctx.log(node_id, "info", "Telegram trigger")
        return input_data or {}

    async def trigger_event(config, input_data, ctx, node_id, user_id=None, **kw):
        ctx.log(node_id, "info", f"Memory event trigger ({config.get('event_type', '?')})")
        return input_data or {}

    executors["trigger_manual"] = trigger_manual
    executors["trigger_schedule"] = trigger_schedule
    executors["trigger_webhook"] = trigger_webhook
    executors["trigger_telegram"] = trigger_telegram
    executors["trigger_event"] = trigger_event

    # ── AI Nodes ────────────────────────────────────────────────────

    async def ai_agent(config, input_data, ctx, node_id, user_id=None, **kw):
        """Run the full agent (with tools) on input."""
        if not agent_runner:
            ctx.log(node_id, "warning", "No agent_runner available")
            return {"error": "Agent not available"}

        prompt = config.get("system_prompt", "")
        user_msg = input_data.get("text") or input_data.get("result") or json.dumps(input_data)

        ctx.log(node_id, "info", f"Running AI agent (model: {config.get('model', 'default')})")
        try:
            response = await agent_runner.run(
                user_message=user_msg,
                user_id=user_id,
                system_prompt_override=prompt or None,
                model_override=config.get("model"),
            )
            return {
                "text": response.text,
                "result": response.text,
                "model": response.model,
                "tokens": response.tokens_total,
                "tool_calls": len(response.tool_calls),
            }
        except Exception as e:
            ctx.log(node_id, "error", f"AI agent error: {e}")
            return {"error": str(e)}

    async def ai_chat(config, input_data, ctx, node_id, user_id=None, **kw):
        """Simple LLM chat completion (no tools)."""
        if not agent_runner:
            ctx.log(node_id, "warning", "No agent_runner available")
            return {"error": "Agent not available"}

        prompt_template = config.get("prompt", "{{input}}")
        from app.agent.workspace.engine import resolve_template
        prompt = resolve_template(prompt_template, {"input": input_data})

        ctx.log(node_id, "info", f"AI chat completion (model: {config.get('model', 'default')})")
        try:
            response = await agent_runner.run(
                user_message=prompt,
                user_id=user_id,
                model_override=config.get("model"),
                disable_tools=True,
            )
            return {"text": response.text, "result": response.text}
        except Exception as e:
            ctx.log(node_id, "error", f"AI chat error: {e}")
            return {"error": str(e)}

    async def ai_classify(config, input_data, ctx, node_id, user_id=None, **kw):
        """Classify input into categories using LLM."""
        if not agent_runner:
            return {"error": "Agent not available"}

        categories = config.get("categories", "")
        text = input_data.get("text") or input_data.get("result") or str(input_data)
        prompt = (
            f"Classify the following text into exactly one of these categories: {categories}\n\n"
            f"Text: {text}\n\n"
            f"Respond with ONLY the category name, nothing else."
        )

        try:
            response = await agent_runner.run(
                user_message=prompt,
                user_id=user_id,
                model_override=config.get("model"),
                disable_tools=True,
            )
            category = response.text.strip().lower()
            return {"category": category, "text": text, "result": category}
        except Exception as e:
            return {"error": str(e)}

    async def ai_extract(config, input_data, ctx, node_id, user_id=None, **kw):
        """Extract structured data from text using AI."""
        if not agent_runner:
            return {"error": "Agent not available"}

        schema = config.get("schema", "{}")
        text = input_data.get("text") or input_data.get("result") or str(input_data)
        prompt = (
            f"Extract the following fields from the text. Return ONLY valid JSON matching this schema:\n"
            f"{schema}\n\nText: {text}\n\nJSON:"
        )

        try:
            response = await agent_runner.run(
                user_message=prompt,
                user_id=user_id,
                model_override=config.get("model"),
                disable_tools=True,
            )
            try:
                extracted = json.loads(response.text.strip())
            except json.JSONDecodeError:
                extracted = {"raw": response.text.strip()}
            return {"extracted": extracted, "result": extracted}
        except Exception as e:
            return {"error": str(e)}

    async def ai_embedding(config, input_data, ctx, node_id, user_id=None, **kw):
        """Generate text embeddings."""
        text = input_data.get("text") or input_data.get("result") or str(input_data)
        try:
            from app.services import get_embedding_service
            svc = get_embedding_service()
            embedding = svc.embed(text)
            return {"embedding": embedding, "dimensions": len(embedding), "text": text}
        except Exception as e:
            return {"error": str(e)}

    executors["ai_agent"] = ai_agent
    executors["ai_chat"] = ai_chat
    executors["ai_classify"] = ai_classify
    executors["ai_extract"] = ai_extract
    executors["ai_embedding"] = ai_embedding

    # ── Actions ─────────────────────────────────────────────────────

    async def action_memory_search(config, input_data, ctx, node_id, user_id=None, **kw):
        """Search agent memories."""
        query = config.get("query") or input_data.get("text", "")
        brain_type = config.get("brain_type", "user")
        limit = int(config.get("limit", 10))

        ctx.log(node_id, "info", f"Memory search: '{query[:50]}' (brain={brain_type})")
        try:
            from app.services import get_embedding_service
            from app.services.memory_service import MemoryService
            from app.db.database import async_session_maker

            svc = get_embedding_service()
            embedding = svc.embed(query)
            async with async_session_maker() as db:
                mem_svc = MemoryService(db)
                results = await mem_svc.search_memories_by_embedding(
                    user_id=user_id,
                    embedding=embedding,
                    limit=limit,
                    min_similarity=0.1,
                    brain_types=[brain_type] if brain_type != "all" else None,
                )
            return {"results": results, "count": len(results), "query": query}
        except Exception as e:
            ctx.log(node_id, "error", f"Memory search failed: {e}")
            return {"error": str(e), "results": []}

    async def action_memory_store(config, input_data, ctx, node_id, user_id=None, **kw):
        """Store a new memory."""
        content = config.get("content") or input_data.get("text", "")
        category = config.get("category", "knowledge")
        importance = float(config.get("importance", 5)) / 10.0

        ctx.log(node_id, "info", f"Storing memory: '{content[:50]}...'")
        try:
            from app.services.memory_service import MemoryService
            from app.db.database import async_session_maker

            async with async_session_maker() as db:
                mem_svc = MemoryService(db)
                memory = await mem_svc.create_memory(
                    user_id=user_id,
                    content=content,
                    category=category,
                    brain_type="user",
                    importance=importance,
                )
                return {"memory_id": memory.id, "content": content, "result": "stored"}
        except Exception as e:
            return {"error": str(e)}

    async def action_http(config, input_data, ctx, node_id, user_id=None, **kw):
        """Make an HTTP request."""
        import httpx
        method = config.get("method", "GET")
        url = config.get("url", "")
        headers_raw = config.get("headers", "{}")
        body_raw = config.get("body", "{}")

        try:
            headers = json.loads(headers_raw) if isinstance(headers_raw, str) else headers_raw
        except json.JSONDecodeError:
            headers = {}
        try:
            body = json.loads(body_raw) if isinstance(body_raw, str) else body_raw
        except json.JSONDecodeError:
            body = {}

        ctx.log(node_id, "info", f"HTTP {method} {url}")
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                if method in ("GET", "HEAD"):
                    resp = await client.request(method, url, headers=headers)
                else:
                    resp = await client.request(method, url, headers=headers, json=body)
                try:
                    resp_json = resp.json()
                except Exception:
                    resp_json = None
                return {
                    "status": resp.status_code,
                    "body": resp_json or resp.text[:5000],
                    "headers": dict(resp.headers),
                    "result": resp_json or resp.text[:5000],
                }
        except Exception as e:
            return {"error": str(e)}

    async def action_exec(config, input_data, ctx, node_id, user_id=None, **kw):
        """Execute a shell command."""
        command = config.get("command", "echo 'no command'")
        timeout = int(config.get("timeout", 30))

        ctx.log(node_id, "info", f"Exec: {command[:80]}")
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return {
                "stdout": stdout.decode(errors="replace")[:10000],
                "stderr": stderr.decode(errors="replace")[:5000],
                "exit_code": proc.returncode,
                "result": stdout.decode(errors="replace")[:10000],
            }
        except asyncio.TimeoutError:
            return {"error": f"Command timed out after {timeout}s"}
        except Exception as e:
            return {"error": str(e)}

    async def action_send_telegram(config, input_data, ctx, node_id, user_id=None, **kw):
        """Send a Telegram message."""
        message = config.get("message") or input_data.get("text") or input_data.get("result", "")
        chat_id = config.get("chat_id")

        ctx.log(node_id, "info", f"Send Telegram: '{message[:50]}...'")
        try:
            from app.config import settings
            if not settings.telegram_bot_token:
                return {"error": "Telegram not configured"}

            import httpx
            url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
            payload = {"text": message, "parse_mode": "Markdown"}
            if chat_id:
                payload["chat_id"] = chat_id
            else:
                # Use trigger chat_id if available
                trigger_chat = ctx.trigger_data.get("chat_id")
                if trigger_chat:
                    payload["chat_id"] = trigger_chat
                else:
                    return {"error": "No chat_id specified and no trigger chat_id"}

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload)
                return {"sent": resp.status_code == 200, "result": resp.json()}
        except Exception as e:
            return {"error": str(e)}

    executors["action_memory_search"] = action_memory_search
    executors["action_memory_store"] = action_memory_store
    executors["action_http"] = action_http
    executors["action_exec"] = action_exec
    executors["action_send_telegram"] = action_send_telegram

    # ── Logic ───────────────────────────────────────────────────────

    async def logic_if(config, input_data, ctx, node_id, user_id=None, **kw):
        """Branch based on condition. Returns _branch='true' or 'false'."""
        condition = config.get("condition", "")
        ctx.log(node_id, "info", f"IF: {condition}")

        # Evaluate condition — simple expression evaluation
        try:
            # Build safe evaluation context
            eval_ctx = {"input": input_data, "vars": ctx.variables, "len": len, "str": str, "int": int, "float": float}
            result = bool(eval(condition, {"__builtins__": {}}, eval_ctx))
        except Exception as e:
            ctx.log(node_id, "warning", f"Condition eval error: {e}, defaulting to false")
            result = False

        return {"_branch": "true" if result else "false", "data": input_data}

    async def logic_switch(config, input_data, ctx, node_id, user_id=None, **kw):
        """Route to different outputs based on field value."""
        field_expr = config.get("field", "")
        from app.agent.workspace.engine import resolve_template
        field_value = resolve_template(field_expr, {"input": input_data})

        case1 = config.get("case1_value", "")
        case2 = config.get("case2_value", "")

        if field_value == case1:
            route = "case1"
        elif field_value == case2:
            route = "case2"
        else:
            route = "default"

        ctx.log(node_id, "info", f"Switch: {field_value} → {route}")
        return {"_route": route, "data": input_data}

    async def logic_merge(config, input_data, ctx, node_id, user_id=None, **kw):
        """Merge inputs (simplified — just pass through)."""
        ctx.log(node_id, "info", "Merge")
        return input_data

    async def logic_loop(config, input_data, ctx, node_id, user_id=None, **kw):
        """Loop over items. Returns _items list for the engine to iterate."""
        items_field = config.get("items_field", "")
        from app.agent.workspace.engine import resolve_template

        if items_field:
            raw = resolve_template(items_field, {"input": input_data})
            try:
                items = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                items = input_data.get("results") or input_data.get("items") or []
        else:
            items = input_data.get("results") or input_data.get("items") or []

        if not isinstance(items, list):
            items = [items]

        ctx.log(node_id, "info", f"Loop: {len(items)} items")
        return {"_items": items}

    async def logic_wait(config, input_data, ctx, node_id, user_id=None, **kw):
        """Wait/delay before continuing."""
        seconds = float(config.get("seconds", 5))
        ctx.log(node_id, "info", f"Waiting {seconds}s")
        await asyncio.sleep(min(seconds, 300))  # Cap at 5 minutes
        return input_data

    executors["logic_if"] = logic_if
    executors["logic_switch"] = logic_switch
    executors["logic_merge"] = logic_merge
    executors["logic_loop"] = logic_loop
    executors["logic_wait"] = logic_wait

    # ── Data ────────────────────────────────────────────────────────

    async def data_transform(config, input_data, ctx, node_id, user_id=None, **kw):
        """Transform data with a simple expression."""
        code = config.get("code", "return input")
        ctx.log(node_id, "info", "Transform data")
        try:
            # Wrap in function for 'return' support
            func_code = f"def _transform(input, vars):\n"
            for line in code.split("\n"):
                func_code += f"  {line}\n"
            local_ns: Dict[str, Any] = {}
            exec(func_code, {"__builtins__": {"str": str, "int": int, "float": float, "len": len, "json": json, "list": list, "dict": dict, "bool": bool, "range": range, "enumerate": enumerate, "zip": zip}}, local_ns)
            result = local_ns["_transform"](input_data, ctx.variables)
            return result if isinstance(result, dict) else {"result": result}
        except Exception as e:
            ctx.log(node_id, "error", f"Transform error: {e}")
            return {"error": str(e), **input_data}

    async def data_set(config, input_data, ctx, node_id, user_id=None, **kw):
        """Set a variable in execution context."""
        name = config.get("name", "")
        value = config.get("value", "")
        if name:
            ctx.variables[name] = value
            ctx.log(node_id, "info", f"Set {name} = {str(value)[:50]}")
        return {**input_data, name: value} if name else input_data

    async def data_filter(config, input_data, ctx, node_id, user_id=None, **kw):
        """Filter items by condition."""
        condition = config.get("condition", "")
        items = input_data.get("results") or input_data.get("items") or []
        if not isinstance(items, list):
            return input_data

        filtered = []
        for item in items:
            try:
                eval_ctx = {"item": item, "vars": ctx.variables, "len": len, "str": str, "int": int, "float": float}
                if eval(condition, {"__builtins__": {}}, eval_ctx):
                    filtered.append(item)
            except Exception:
                pass

        ctx.log(node_id, "info", f"Filter: {len(items)} → {len(filtered)} items")
        return {"items": filtered, "results": filtered, "count": len(filtered)}

    executors["data_transform"] = data_transform
    executors["data_set"] = data_set
    executors["data_filter"] = data_filter

    # ── Output ──────────────────────────────────────────────────────

    async def output_respond(config, input_data, ctx, node_id, user_id=None, **kw):
        """Send response back to trigger (stored in context)."""
        message = config.get("message") or input_data.get("text") or input_data.get("result", "")
        ctx.log(node_id, "info", f"Response: {str(message)[:100]}")
        ctx.variables["_response"] = message
        return {"response": message, "result": message}

    async def output_log(config, input_data, ctx, node_id, user_id=None, **kw):
        """Log data for debugging."""
        level = config.get("level", "info")
        ctx.log(node_id, level, f"Log output: {json.dumps(input_data, default=str)[:500]}")
        return input_data

    executors["output_respond"] = output_respond
    executors["output_log"] = output_log

    return executors
