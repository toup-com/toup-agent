"""
Tool Executor — Runs tools requested by the LLM and returns string results.

Supported tools:
  exec, read_file, write_file, edit_file,
  memory_search, memory_store, web_search, web_fetch

Per-tool output limits prevent bloating context:
  exec:       10 KB
  read_file:  50 KB
  write_file: N/A (short confirmation)
  edit_file:  N/A (short confirmation)
  web_search: 10 KB
  web_fetch:  15 KB
  memory_*:   10 KB
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import Dict, Any, Optional, List, Set

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# Per-tool output limits (bytes)
TOOL_OUTPUT_LIMITS: Dict[str, int] = {
    "exec": 10_000,
    "read_file": 50_000,
    "write_file": 1_000,
    "edit_file": 1_000,
    "memory_search": 10_000,
    "memory_store": 1_000,
    "web_search": 10_000,
    "web_fetch": 15_000,
    "send_file": 1_000,
    "send_photo": 1_000,
    "analyze_image": 10_000,
    "cron": 5_000,
    "spawn": 1_000,
    "process": 10_000,
    "tts": 1_000,
    "browser": 50_000,
    "sessions_list": 10_000,
    "sessions_history": 30_000,
    "grep": 30_000,
    "find": 15_000,
    "ls": 15_000,
    "apply_patch": 5_000,
    "sessions_send": 1_000,
    "session_status": 2_000,
    "agents_list": 5_000,
    "message": 2_000,
    "moderate": 1_000,
    "config_reload": 5_000,
    "lanes_status": 3_000,
    "poll": 1_000,
    "thread": 2_000,
    "tts_prefs": 1_000,
}

# Default if tool not in the table
DEFAULT_OUTPUT_LIMIT = 15_000

# Dangerous command patterns — always blocked (catastrophic)
BLOCKED_PATTERNS = [
    r"rm\s+-rf\s+/\s*$",
    r"rm\s+-rf\s+/\s+",
    r"mkfs\.",
    r"dd\s+if=.*of=/dev/",
    r":\(\)\{.*\}",
    r"chmod\s+-R\s+777\s+/\s*$",
]

# Destructive command patterns — require explicit user confirmation.
# When detected, the tool returns a safety message instead of executing.
# The agent must ask the user to confirm, then re-call with confirmed=true.
DESTRUCTIVE_PATTERNS = [
    r"\brm\s+",           # rm (any form: rm file, rm -f, rm -r, rm -rf)
    r"\brmdir\b",         # rmdir
    r"\bunlink\b",        # unlink
    r"\bshred\b",         # shred
    r"\btrash\b",         # trash
    r"\bmv\b.*(/dev/null|/tmp/)",  # mv to /dev/null or /tmp (disguised delete)
    r">\s*/dev/null",     # redirect to /dev/null (truncate)
]


class ToolExecutor:
    """Executes agent tools and returns results as strings."""
    
    def __init__(self, workspace: Optional[str] = None, telegram_bot=None, cron_service=None, subagent_manager=None):
        self.workspace = workspace or settings.agent_workspace_dir
        os.makedirs(self.workspace, exist_ok=True)
        self.telegram_bot = telegram_bot  # Set after bot starts
        self.cron_service = cron_service  # Set after cron service starts
        self.subagent_manager = subagent_manager  # Set after subagent manager created
        self.skill_loader = None  # Set after skills are loaded
        self._chat_id: Optional[int] = None
        self._on_tool_progress: Optional[Any] = None  # Callback for streaming tool output
        # Track which user workspaces have been bootstrapped this session
        self._bootstrapped_users: Set[str] = set()
        # Background process tracking {proc_id: {...}}
        self._processes: Dict[str, Dict[str, Any]] = {}
        self._proc_counter: int = 0
        # Per-user disabled tools (loaded from AgentConfig)
        self.user_disabled_tools: Set[str] = set()

    def set_chat_id(self, chat_id: Optional[int]):
        """Set the current Telegram chat ID for send_file/send_photo tools."""
        self._chat_id = chat_id

    # ------------------------------------------------------------------
    # Workspace Bootstrap
    # ------------------------------------------------------------------
    def _get_user_workspace(self) -> str:
        """
        Return the effective workspace path for the current user.

        When ``workspace_per_user`` is enabled, each user gets an isolated
        subdirectory: ``<workspace_root>/<user_id>/``.  Otherwise the
        shared root workspace is returned.
        """
        if settings.workspace_per_user and getattr(self, "_user_id", ""):
            return os.path.join(self.workspace, self._user_id)
        return self.workspace

    def _ensure_workspace(self) -> str:
        """
        Ensure the workspace directory exists for the current user.

        Called lazily on the first file/exec tool invocation per user.
        Creates the directory tree and an optional README.md.

        Returns the workspace path.
        """
        ws = self._get_user_workspace()
        user_id = getattr(self, "_user_id", "")

        # Fast path — already bootstrapped this session
        if user_id and user_id in self._bootstrapped_users:
            return ws

        if not os.path.isdir(ws):
            os.makedirs(ws, exist_ok=True)
            logger.info(f"[WORKSPACE] Created workspace directory: {ws}")

            if settings.workspace_create_readme:
                readme_path = os.path.join(ws, "README.md")
                if not os.path.exists(readme_path):
                    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
                    readme = (
                        f"# HexBrain Workspace\n\n"
                        f"Created: {now}\n"
                        f"User: {user_id or 'shared'}\n\n"
                        f"This directory is used by the HexBrain agent for file operations.\n"
                        f"Files created here can be sent to you via Telegram.\n"
                    )
                    try:
                        with open(readme_path, "w", encoding="utf-8") as f:
                            f.write(readme)
                        logger.info(f"[WORKSPACE] Created README.md in {ws}")
                    except Exception as e:
                        logger.warning(f"[WORKSPACE] Failed to create README: {e}")
        else:
            logger.debug(f"[WORKSPACE] Workspace already exists: {ws}")

        if user_id:
            self._bootstrapped_users.add(user_id)

        return ws
    
    async def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """
        Dispatch a tool call and return the result as a string.
        Applies per-tool output limits and appends truncation notice.
        On error the string starts with "ERROR: …".

        Routing order:
          0. Check tool policy (deny list blocks, elevated list logs warning)
          1. Built-in tool handler (_tool_<name>)
          2. Skill tool (if skill_loader recognises the name)
          3. ERROR: Unknown tool
        """
        # ── Tool Policy Enforcement ──────────────────────────
        if tool_name in settings.tool_deny_list:
            return f"ERROR: Tool '{tool_name}' is blocked by administrator policy."
        if tool_name in self.user_disabled_tools:
            return f"ERROR: Tool '{tool_name}' has been disabled by the user."
        if tool_name in settings.tool_elevated_list:
            logger.warning(f"[TOOL-POLICY] Elevated tool invoked: {tool_name}")

        # ── Per-Tool Timeout ──────────────────────────────────
        tool_timeout = settings.tool_timeout_overrides.get(
            tool_name, settings.tool_timeout_default
        )

        try:
            handler = getattr(self, f"_tool_{tool_name}", None)
            if handler is not None:
                try:
                    result = await asyncio.wait_for(handler(tool_input), timeout=tool_timeout)
                except asyncio.TimeoutError:
                    return f"ERROR: Tool '{tool_name}' timed out after {tool_timeout}s"
            elif self.skill_loader and self.skill_loader.is_skill_tool(tool_name):
                from app.agent.skills.base import SkillContext
                ctx = SkillContext(
                    workspace=self.workspace,
                    user_id=self._current_user_id,
                    chat_id=self._chat_id,
                )
                result = await self.skill_loader.execute_tool(tool_name, tool_input, ctx)
            else:
                return f"ERROR: Unknown tool '{tool_name}'"

            # Apply per-tool output limit
            limit = TOOL_OUTPUT_LIMITS.get(tool_name, DEFAULT_OUTPUT_LIMIT)
            if len(result) > limit:
                truncated_bytes = len(result) - limit
                result = result[:limit] + f"\n\n[truncated, {truncated_bytes} more bytes]"

            return result
        except Exception as exc:
            logger.exception(f"Tool {tool_name} raised")
            return f"ERROR: {type(exc).__name__}: {exc}"
    
    # ------------------------------------------------------------------
    # 1. exec — shell command execution
    # ------------------------------------------------------------------
    async def _tool_exec(self, inp: Dict[str, Any]) -> str:
        command = inp.get("command", "").strip()
        if not command:
            return "ERROR: Empty command"

        # Safety check — always blocked (catastrophic commands)
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, command):
                return f"ERROR: Blocked dangerous command pattern: {pattern}"

        # Destructive command check — requires explicit user confirmation
        confirmed = inp.get("confirmed", False)
        if not confirmed:
            for pattern in DESTRUCTIVE_PATTERNS:
                if re.search(pattern, command):
                    return (
                        f"SAFETY: This command is destructive (matches: {pattern}). "
                        f"You MUST ask the user for explicit confirmation before executing. "
                        f"Tell the user exactly what will be deleted and ask 'Are you sure?'. "
                        f"Only if they clearly say yes, re-call exec with confirmed=true."
                    )
        
        # Bootstrap workspace on first use
        default_ws = self._ensure_workspace()
        workdir = inp.get("workdir", default_ws)
        timeout = min(int(inp.get("timeout", 30)), 120)

        # Docker sandbox mode
        if settings.sandbox_enabled:
            from app.agent.sandbox import SandboxExecutor
            if not hasattr(self, "_sandbox"):
                self._sandbox = SandboxExecutor()
            return await self._sandbox.exec(
                command=command,
                user_id=self._current_user_id or "default",
                workdir="/workspace",
                timeout=timeout,
            )
        
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=workdir,
                env={**os.environ, "TERM": "dumb"},
            )
            try:
                # Stream output progressively if callback available
                if self._on_tool_progress and proc.stdout:
                    chunks = []
                    try:
                        while True:
                            line = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout)
                            if not line:
                                break
                            decoded = line.decode("utf-8", errors="replace")
                            chunks.append(decoded)
                            try:
                                await self._on_tool_progress("exec", decoded)
                            except Exception:
                                pass
                    except asyncio.TimeoutError:
                        proc.kill()
                        await proc.wait()
                        partial = "".join(chunks)
                        return f"{partial}\nERROR: Command timed out after {timeout}s"
                    await proc.wait()
                    stdout = "".join(chunks).encode()
                else:
                    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return f"ERROR: Command timed out after {timeout}s"
            
            output = stdout.decode("utf-8", errors="replace")
            
            exit_code = proc.returncode
            if exit_code != 0:
                return f"{output}\n[exit code: {exit_code}]"
            return output or "(no output)"
        
        except FileNotFoundError:
            return f"ERROR: Working directory not found: {workdir}"
    

    # ------------------------------------------------------------------
    # 1b. pty_exec — pseudo-terminal exec for TTY-requiring CLIs
    # ------------------------------------------------------------------
    async def _tool_pty_exec(self, inp: Dict[str, Any]) -> str:
        """Execute a command in a pseudo-terminal (for TTY-requiring CLIs like top, vim, etc.)."""
        self._ensure_workspace()
        command = inp.get("command", "")
        if not command:
            return "ERROR: 'command' is required"

        # Destructive command check
        confirmed = inp.get("confirmed", False)
        if not confirmed:
            for pattern in DESTRUCTIVE_PATTERNS:
                if re.search(pattern, command):
                    return (
                        f"SAFETY: This command is destructive (matches: {pattern}). "
                        f"You MUST ask the user for explicit confirmation before executing. "
                        f"Only if they clearly say yes, re-call with confirmed=true."
                    )

        default_ws = self._get_user_workspace()
        workdir = inp.get("workdir", default_ws)
        timeout = min(int(inp.get("timeout", 30)), 120)
        rows = int(inp.get("rows", 24))
        cols = int(inp.get("cols", 80))

        try:
            import pty as pty_mod
            import select as select_mod

            master_fd, slave_fd = pty_mod.openpty()

            # Set terminal size
            import struct, fcntl, termios
            winsize = struct.pack("HHHH", rows, cols, 0, 0)
            fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)

            proc = await asyncio.create_subprocess_shell(
                command,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                cwd=workdir,
                env={**os.environ, "TERM": "xterm-256color", "COLUMNS": str(cols), "LINES": str(rows)},
            )
            os.close(slave_fd)

            output_chunks = []
            loop = asyncio.get_event_loop()

            async def read_pty():
                while True:
                    try:
                        readable, _, _ = await loop.run_in_executor(
                            None, select_mod.select, [master_fd], [], [], 0.1
                        )
                        if readable:
                            data = os.read(master_fd, 4096)
                            if not data:
                                break
                            decoded = data.decode("utf-8", errors="replace")
                            output_chunks.append(decoded)
                            if self._on_tool_progress:
                                try:
                                    await self._on_tool_progress("pty_exec", decoded)
                                except Exception:
                                    pass
                        else:
                            if proc.returncode is not None:
                                break
                    except OSError:
                        break

            try:
                await asyncio.wait_for(read_pty(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()

            await proc.wait()
            os.close(master_fd)

            output = "".join(output_chunks)
            # Strip ANSI escape sequences for clean output
            import re as _re
            output = _re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", output)
            output = _re.sub(r"\x1b\][^\x07]*\x07", "", output)

            exit_code = proc.returncode
            if len(output) > 10000:
                output = output[:5000] + "\n... (truncated) ...\n" + output[-2000:]
            if exit_code != 0:
                return f"{output}\n[exit code: {exit_code}]"
            return output or "(no output)"

        except ImportError:
            return "ERROR: PTY support not available on this platform"
        except Exception as e:
            return f"ERROR: PTY exec failed: {e}"

    # ------------------------------------------------------------------
    # 2. read_file
    # ------------------------------------------------------------------
    async def _tool_read_file(self, inp: Dict[str, Any]) -> str:
        self._ensure_workspace()
        path = self._resolve_path(inp.get("path", ""))
        if not os.path.isfile(path):
            return f"ERROR: File not found: {path}"
        
        try:
            read_limit = TOOL_OUTPUT_LIMITS.get("read_file", DEFAULT_OUTPUT_LIMIT)
            with open(path, "rb") as f:
                raw = f.read(read_limit + 1)
            
            # Binary detection
            if b"\x00" in raw[:8192]:
                return f"Binary file ({len(raw)} bytes): {path}"
            
            text = raw.decode("utf-8", errors="replace")
            lines = text.splitlines(keepends=True)
            
            offset = int(inp.get("offset", 0))
            limit = int(inp.get("limit", 0)) or len(lines)
            selected = lines[offset:offset + limit]
            
            result = "".join(selected)
            return result or "(empty file)"
        
        except PermissionError:
            return f"ERROR: Permission denied: {path}"
    
    # ------------------------------------------------------------------
    # 3. write_file
    # ------------------------------------------------------------------
    async def _tool_write_file(self, inp: Dict[str, Any]) -> str:
        self._ensure_workspace()
        path = self._resolve_path(inp.get("path", ""))
        content = inp.get("content", "")
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Written {len(content)} bytes to {path}"
        except PermissionError:
            return f"ERROR: Permission denied: {path}"
    
    # ------------------------------------------------------------------
    # 4. edit_file (find & replace)
    # ------------------------------------------------------------------
    async def _tool_edit_file(self, inp: Dict[str, Any]) -> str:
        self._ensure_workspace()
        path = self._resolve_path(inp.get("path", ""))
        old_text = inp.get("old_text", "")
        new_text = inp.get("new_text", "")
        
        if not os.path.isfile(path):
            return f"ERROR: File not found: {path}"
        if not old_text:
            return "ERROR: old_text is required"
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            if old_text not in content:
                return f"ERROR: old_text not found in {path}"
            
            count = content.count(old_text)
            content = content.replace(old_text, new_text, 1)
            
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            
            return f"Replaced 1 of {count} occurrence(s) in {path}"
        except PermissionError:
            return f"ERROR: Permission denied: {path}"
    
    # ------------------------------------------------------------------
    # 5. memory_search
    # ------------------------------------------------------------------
    async def _tool_memory_search(self, inp: Dict[str, Any]) -> str:
        query = inp.get("query", "")
        brain_type = inp.get("brain_type")
        limit = int(inp.get("limit", 5))
        
        if not query:
            return "ERROR: query is required"
        
        try:
            from app.db.database import async_session_maker
            from app.services.memory_service import MemoryService
            from app.services.embedding_service import get_embedding_service
            
            embedding_svc = get_embedding_service()
            embedding = embedding_svc.embed(query)
            
            async with async_session_maker() as db:
                svc = MemoryService(db)
                results = await svc.search_memories_by_embedding(
                    user_id=self._current_user_id,
                    embedding=embedding,
                    limit=limit,
                    min_similarity=0.1,
                    brain_types=[brain_type] if brain_type else None,
                )
            
            if not results:
                return "No memories found."
            
            lines = []
            for i, mem in enumerate(results, 1):
                score = mem.get("similarity_score", 0)
                cat = mem.get("category", "")
                content = mem.get("content", "")
                lines.append(f"{i}. [{cat}] (sim={score:.2f}) {content}")
            return "\n".join(lines)
        
        except Exception as exc:
            logger.exception("memory_search failed")
            return f"ERROR: {exc}"
    
    # ------------------------------------------------------------------
    # 6. memory_store
    # ------------------------------------------------------------------
    async def _tool_memory_store(self, inp: Dict[str, Any]) -> str:
        content = inp.get("content", "")
        category = inp.get("category", "context")
        brain_type = inp.get("brain_type", "user")
        importance = float(inp.get("importance", 0.5))
        
        if not content:
            return "ERROR: content is required"
        
        try:
            from app.db.database import async_session_maker
            from app.services.memory_dedup_service import MemoryDedupService
            from app.schemas import MemoryCreate, BrainType, MemoryType, MemoryLevel
            
            memory_data = MemoryCreate(
                content=content,
                summary=content[:100],
                brain_type=BrainType(brain_type),
                category=category,
                memory_type=MemoryType.FACT,
                importance=importance,
                confidence=0.9,
                memory_level=MemoryLevel.EPISODIC,
                emotional_salience=0.5,
                source_type="agent_tool",
            )
            
            async with async_session_maker() as db:
                dedup = MemoryDedupService(db)
                memory, action = await dedup.smart_create_memory(
                    new_memory=memory_data,
                    user_id=self._current_user_id,
                )
            
            return f"Memory {action}: {memory.id} — {memory.content[:80]}"
        
        except Exception as exc:
            logger.exception("memory_store failed")
            return f"ERROR: {exc}"
    
    # ------------------------------------------------------------------
    # 7. web_search
    # ------------------------------------------------------------------
    async def _tool_web_search(self, inp: Dict[str, Any]) -> str:
        query = inp.get("query", "")
        count = min(int(inp.get("count", 5)), 10)
        
        if not query:
            return "ERROR: query is required"
        
        # Try Brave Search first, fallback to DuckDuckGo HTML
        if settings.brave_api_key:
            return await self._brave_search(query, count)
        return await self._ddg_search(query, count)
    
    async def _brave_search(self, query: str, count: int) -> str:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": count},
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": settings.brave_api_key,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        
        results = data.get("web", {}).get("results", [])
        if not results:
            return "No results found."
        
        lines = []
        for i, r in enumerate(results[:count], 1):
            lines.append(f"{i}. {r.get('title', '')}")
            lines.append(f"   {r.get('url', '')}")
            lines.append(f"   {r.get('description', '')}")
            lines.append("")
        return "\n".join(lines)
    
    async def _ddg_search(self, query: str, count: int) -> str:
        """Fallback search via DuckDuckGo HTML (no API key needed)."""
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    headers={"User-Agent": "Mozilla/5.0 (HexBrain Agent)"},
                )
                resp.raise_for_status()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            results = soup.select(".result")[:count]
            
            if not results:
                return "No results found."
            
            lines = []
            for i, r in enumerate(results, 1):
                title_el = r.select_one(".result__title a")
                snippet_el = r.select_one(".result__snippet")
                title = title_el.get_text(strip=True) if title_el else "Untitled"
                url = title_el.get("href", "") if title_el else ""
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                lines.append(f"{i}. {title}")
                lines.append(f"   {url}")
                lines.append(f"   {snippet}")
                lines.append("")
            return "\n".join(lines)
        
        except Exception as exc:
            return f"ERROR: DuckDuckGo search failed: {exc}"
    
    # ------------------------------------------------------------------
    # 8. web_fetch
    # ------------------------------------------------------------------
    async def _tool_web_fetch(self, inp: Dict[str, Any]) -> str:
        url = inp.get("url", "")
        max_chars = int(inp.get("max_chars", 10000))
        
        if not url:
            return "ERROR: url is required"
        
        try:
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                resp = await client.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (HexBrain Agent)"},
                )
                resp.raise_for_status()
            
            content_type = resp.headers.get("content-type", "")
            
            # If it's plain text or JSON, return directly
            if "text/plain" in content_type or "application/json" in content_type:
                text = resp.text[:max_chars]
                return text
            
            # Parse HTML
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # Remove scripts, styles, nav, footer
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                tag.decompose()
            
            # Try article/main content first
            main = soup.find("article") or soup.find("main") or soup.find("body")
            if main is None:
                return "(empty page)"
            
            text = main.get_text(separator="\n", strip=True)
            
            # Collapse multiple blank lines
            import re as _re
            text = _re.sub(r"\n{3,}", "\n\n", text)
            
            if len(text) > max_chars:
                text = text[:max_chars] + "\n... (truncated)"
            
            return text or "(empty page)"
        
        except httpx.HTTPStatusError as exc:
            return f"ERROR: HTTP {exc.response.status_code} for {url}"
        except Exception as exc:
            return f"ERROR: {exc}"
    
    # ------------------------------------------------------------------
    # 9. send_file — send a document to the user via Telegram
    # ------------------------------------------------------------------
    async def _tool_send_file(self, inp: Dict[str, Any]) -> str:
        path = self._resolve_path(inp.get("path", ""))
        caption = inp.get("caption", None)

        if not os.path.isfile(path):
            return f"ERROR: File not found: {path}"

        if not self.telegram_bot or not self._chat_id:
            return "ERROR: Telegram bot not available or no active chat"

        try:
            file_size = os.path.getsize(path)
            if file_size > 50 * 1024 * 1024:  # Telegram 50MB limit
                return f"ERROR: File too large ({file_size} bytes). Telegram limit is 50MB."

            bot = self.telegram_bot.app.bot
            with open(path, "rb") as f:
                await bot.send_document(
                    chat_id=self._chat_id,
                    document=f,
                    filename=os.path.basename(path),
                    caption=caption,
                )
            fname = os.path.basename(path)
            return f"File sent to user: {fname} ({file_size} bytes)"
        except Exception as exc:
            logger.exception("send_file failed")
            return f"ERROR: Failed to send file: {exc}"

    # ------------------------------------------------------------------
    # 10. send_photo — send an image to the user via Telegram
    # ------------------------------------------------------------------
    async def _tool_send_photo(self, inp: Dict[str, Any]) -> str:
        path = self._resolve_path(inp.get("path", ""))
        caption = inp.get("caption", None)

        if not os.path.isfile(path):
            return f"ERROR: File not found: {path}"

        if not self.telegram_bot or not self._chat_id:
            return "ERROR: Telegram bot not available or no active chat"

        try:
            bot = self.telegram_bot.app.bot
            with open(path, "rb") as f:
                await bot.send_photo(
                    chat_id=self._chat_id,
                    photo=f,
                    caption=caption,
                )
            return f"Photo sent to user: {os.path.basename(path)}"
        except Exception as exc:
            logger.exception("send_photo failed")
            return f"ERROR: Failed to send photo: {exc}"

    # ------------------------------------------------------------------
    # 11. analyze_image — GPT vision on URL or workspace file
    # ------------------------------------------------------------------
    async def _tool_analyze_image(self, inp: Dict[str, Any]) -> str:
        image = inp.get("image", "").strip()
        if not image:
            return "ERROR: 'image' is required (URL or file path)"

        question = inp.get("question", "Describe this image in detail.").strip()

        import base64

        # Determine if URL or file path
        if image.startswith(("http://", "https://")):
            image_content = {"type": "image_url", "image_url": {"url": image}}
        else:
            path = self._resolve_path(image)
            if not os.path.isfile(path):
                return f"ERROR: Image file not found: {path}"

            ext = os.path.splitext(path)[1].lower()
            mime_map = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".gif": "image/gif",
                ".webp": "image/webp", ".bmp": "image/bmp",
            }
            mime = mime_map.get(ext, "image/jpeg")

            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            image_content = {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"},
            }

        # Call OpenAI vision API
        api_key = settings.openai_api_key
        if not api_key:
            return "ERROR: OpenAI API key not configured"

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "gpt-4o",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": question},
                                    image_content,
                                ],
                            }
                        ],
                        "max_tokens": 1024,
                    },
                )
                resp.raise_for_status()

            result = resp.json()
            return result["choices"][0]["message"]["content"]

        except httpx.HTTPStatusError as exc:
            return f"ERROR: Vision API returned {exc.response.status_code}"
        except Exception as exc:
            logger.exception("analyze_image failed")
            return f"ERROR: Image analysis failed: {exc}"

    # ------------------------------------------------------------------
    # 12. cron — scheduled tasks
    # ------------------------------------------------------------------
    async def _tool_cron(self, inp: Dict[str, Any]) -> str:
        action = inp.get("action", "").strip().lower()

        if not self.cron_service:
            return "ERROR: Cron service not available"

        user_id = self._current_user_id
        chat_id = self._chat_id

        if action == "add":
            name = inp.get("name", "Unnamed task")
            schedule = inp.get("schedule", "")
            message = inp.get("message", "")
            if not schedule:
                return "ERROR: 'schedule' is required for add"
            if not message:
                return "ERROR: 'message' is required for add"
            if not chat_id:
                return "ERROR: No active Telegram chat"

            result = await self.cron_service.add_job(
                user_id=user_id,
                chat_id=chat_id,
                name=name,
                schedule=schedule,
                message=message,
            )
            return json.dumps(result)

        elif action == "list":
            jobs = await self.cron_service.list_jobs(user_id)
            if not jobs:
                return "No scheduled jobs."
            lines = []
            for j in jobs:
                status = "enabled" if j["enabled"] else "disabled"
                lines.append(
                    f"• {j['name']} (id={j['id'][:8]}...)\n"
                    f"  Schedule: {j['schedule']} ({j['kind']})\n"
                    f"  Status: {status} | Runs: {j['run_count']}"
                )
            return "\n\n".join(lines)

        elif action == "remove":
            job_id = inp.get("job_id", "")
            if not job_id:
                return "ERROR: 'job_id' is required for remove"
            result = await self.cron_service.remove_job(job_id, user_id)
            return json.dumps(result)

        elif action == "run":
            job_id = inp.get("job_id", "")
            if not job_id:
                return "ERROR: 'job_id' is required for run"
            result = await self.cron_service.run_job_now(job_id, user_id)
            return json.dumps(result)

        else:
            return f"ERROR: Unknown action '{action}'. Use: add, list, remove, run"

    # ------------------------------------------------------------------
    # 13. process — long-running background shell process management
    # ------------------------------------------------------------------
    async def _tool_process(self, inp: Dict[str, Any]) -> str:
        action = inp.get("action", "").strip().lower()

        if action == "start":
            return await self._process_start(inp)
        elif action == "list":
            return self._process_list()
        elif action == "status":
            return self._process_status(inp)
        elif action == "output":
            return self._process_output(inp)
        elif action == "stop":
            return await self._process_stop(inp)
        else:
            return f"ERROR: Unknown action '{action}'. Use: start, list, status, output, stop"

    async def _process_start(self, inp: Dict[str, Any]) -> str:
        command = inp.get("command", "").strip()
        if not command:
            return "ERROR: 'command' is required for start"

        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, command):
                return f"ERROR: Blocked dangerous command pattern"

        # Destructive command check
        if not inp.get("confirmed", False):
            for pattern in DESTRUCTIVE_PATTERNS:
                if re.search(pattern, command):
                    return (
                        f"SAFETY: This command is destructive. "
                        f"Ask the user for explicit confirmation first."
                    )

        label = inp.get("label", f"proc-{self._proc_counter}")
        workdir = self._get_user_workspace()

        self._proc_counter += 1
        proc_id = f"p{self._proc_counter}"

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=workdir,
                env={**os.environ, "TERM": "dumb"},
            )
        except Exception as e:
            return f"ERROR: Failed to start process: {e}"

        self._processes[proc_id] = {
            "id": proc_id,
            "label": label,
            "command": command,
            "proc": proc,
            "pid": proc.pid,
            "started_at": datetime.utcnow().isoformat(),
            "output_buffer": [],
            "user_id": self._current_user_id,
        }

        # Start background reader
        asyncio.create_task(self._process_reader(proc_id))

        logger.info("[PROCESS] Started %s (pid=%s): %s", proc_id, proc.pid, command)
        return json.dumps({
            "id": proc_id,
            "label": label,
            "pid": proc.pid,
            "status": "running",
        })

    async def _process_reader(self, proc_id: str):
        """Read stdout in background and buffer lines."""
        entry = self._processes.get(proc_id)
        if not entry:
            return

        proc = entry["proc"]
        buf = entry["output_buffer"]
        max_lines = 500  # Keep last N lines in memory

        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip("\n")
                buf.append(text)
                if len(buf) > max_lines:
                    buf.pop(0)
        except Exception:
            pass

    def _process_list(self) -> str:
        if not self._processes:
            return "No background processes."

        lines = []
        for entry in self._processes.values():
            proc = entry["proc"]
            running = proc.returncode is None
            status = "running" if running else f"exited ({proc.returncode})"
            lines.append(
                f"• {entry['id']} [{entry['label']}] — {status}\n"
                f"  PID: {entry['pid']} | Started: {entry['started_at']}\n"
                f"  Command: {entry['command'][:80]}"
            )
        return "\n\n".join(lines)

    def _process_status(self, inp: Dict[str, Any]) -> str:
        proc_id = inp.get("process_id", "").strip()
        if not proc_id:
            return "ERROR: 'process_id' is required"

        entry = self._processes.get(proc_id)
        if not entry:
            return f"ERROR: Process not found: {proc_id}"

        proc = entry["proc"]
        running = proc.returncode is None
        return json.dumps({
            "id": proc_id,
            "label": entry["label"],
            "pid": entry["pid"],
            "status": "running" if running else "exited",
            "exit_code": proc.returncode,
            "started_at": entry["started_at"],
            "output_lines": len(entry["output_buffer"]),
        })

    def _process_output(self, inp: Dict[str, Any]) -> str:
        proc_id = inp.get("process_id", "").strip()
        if not proc_id:
            return "ERROR: 'process_id' is required"

        entry = self._processes.get(proc_id)
        if not entry:
            return f"ERROR: Process not found: {proc_id}"

        tail = int(inp.get("tail_lines", 50))
        lines = entry["output_buffer"][-tail:]

        if not lines:
            return "(no output yet)"

        return "\n".join(lines)

    async def _process_stop(self, inp: Dict[str, Any]) -> str:
        proc_id = inp.get("process_id", "").strip()
        if not proc_id:
            return "ERROR: 'process_id' is required"

        entry = self._processes.get(proc_id)
        if not entry:
            return f"ERROR: Process not found: {proc_id}"

        proc = entry["proc"]
        if proc.returncode is not None:
            return json.dumps({
                "id": proc_id,
                "status": "already_exited",
                "exit_code": proc.returncode,
            })

        # Try SIGTERM first, then SIGKILL after 5s
        import signal
        try:
            proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        except ProcessLookupError:
            pass

        logger.info("[PROCESS] Stopped %s (pid=%s)", proc_id, entry["pid"])
        return json.dumps({
            "id": proc_id,
            "label": entry["label"],
            "status": "stopped",
            "exit_code": proc.returncode,
        })

    # ------------------------------------------------------------------
    # 14. spawn — background sub-agent task
    # ------------------------------------------------------------------
    async def _tool_spawn(self, inp: Dict[str, Any]) -> str:
        task = inp.get("task", "").strip()
        if not task:
            return "ERROR: 'task' is required"

        if not self.subagent_manager:
            return "ERROR: Sub-agent manager not available"

        # Agent spawn policy — restrict which agents can be spawned
        from app.config import settings as _cfg
        if _cfg.allow_agents:
            agent_id = inp.get("agent_id", "default")
            if agent_id not in _cfg.allow_agents and agent_id != "default":
                return f"ERROR: Agent '{agent_id}' not in allow_agents policy: {_cfg.allow_agents}"

        user_id = self._current_user_id
        chat_id = self._chat_id
        if not chat_id:
            return "ERROR: No active Telegram chat"

        label = inp.get("label", None)
        model = inp.get("model", None)
        timeout = min(int(inp.get("timeout_seconds", 300)), 600)

        result = await self.subagent_manager.spawn(
            task=task,
            user_id=user_id,
            telegram_chat_id=chat_id,
            label=label,
            model=model,
            timeout_seconds=timeout,
        )
        return json.dumps(result)

    # ------------------------------------------------------------------
    # 15. tts — text-to-speech voice message
    # ------------------------------------------------------------------
    async def _tool_tts(self, inp: Dict[str, Any]) -> str:
        text = inp.get("text", "").strip()
        if not text:
            return "ERROR: 'text' is required"

        chat_id = self._chat_id
        if not chat_id:
            return "ERROR: No active Telegram chat — TTS only works via Telegram"

        if not self.telegram_bot or not self.telegram_bot.app:
            return "ERROR: Telegram bot not available"

        voice = inp.get("voice", "nova")
        speed = float(inp.get("speed", 1.0))
        instructions = inp.get("instructions", None)

        from app.agent.tts_providers import synthesize_speech_multi

        provider = inp.get("provider", None)
        audio_path = await synthesize_speech_multi(
            text=text,
            provider=provider,
            voice=voice,
            speed=speed,
            instructions=instructions,
            user_id=self._current_user_id,
        )

        if audio_path.startswith("ERROR:"):
            return audio_path

        try:
            bot = self.telegram_bot.app.bot
            with open(audio_path, "rb") as audio_file:
                await bot.send_voice(chat_id=chat_id, voice=audio_file)
            return f"Voice message sent ({len(text)} chars, voice={voice})"
        except Exception as e:
            logger.exception("[TTS] Failed to send voice message")
            return f"ERROR: Failed to send voice: {e}"
        finally:
            # Clean up temp file
            try:
                os.unlink(audio_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # 16. sessions_list — list conversation sessions
    # ------------------------------------------------------------------
    async def _tool_sessions_list(self, inp: Dict[str, Any]) -> str:
        limit = int(inp.get("limit", 10))
        active_only = inp.get("active_only", True)

        user_id = self._current_user_id
        if not user_id:
            return "ERROR: No user context"

        try:
            from app.db.database import async_session_maker
            from app.db.models import Conversation
            from sqlalchemy import select, and_

            async with async_session_maker() as db:
                query = select(Conversation).where(
                    Conversation.user_id == user_id
                )
                if active_only:
                    query = query.where(Conversation.is_active == True)
                query = query.order_by(Conversation.updated_at.desc()).limit(limit)

                result = await db.execute(query)
                sessions = result.scalars().all()

            if not sessions:
                return "No sessions found."

            lines = []
            for s in sessions:
                status = "active" if s.is_active else "ended"
                lines.append(
                    f"• Session {s.id[:8]}...\n"
                    f"  Channel: {s.channel} | Status: {status}\n"
                    f"  Messages: {s.message_count} | Tokens: {s.total_tokens:,}\n"
                    f"  Created: {s.created_at.strftime('%Y-%m-%d %H:%M')}\n"
                    f"  Updated: {s.updated_at.strftime('%Y-%m-%d %H:%M')}"
                )
            return "\n\n".join(lines)

        except Exception as e:
            logger.exception("sessions_list failed")
            return f"ERROR: {e}"

    # ------------------------------------------------------------------
    # 17. sessions_history — view messages from a session
    # ------------------------------------------------------------------
    async def _tool_sessions_history(self, inp: Dict[str, Any]) -> str:
        session_id = inp.get("session_id", "").strip()
        limit = int(inp.get("limit", 20))

        if not session_id:
            return "ERROR: 'session_id' is required"

        user_id = self._current_user_id
        if not user_id:
            return "ERROR: No user context"

        try:
            from app.db.database import async_session_maker
            from app.db.models import Conversation, Message
            from sqlalchemy import select, and_

            async with async_session_maker() as db:
                # Verify session belongs to user
                result = await db.execute(
                    select(Conversation).where(
                        and_(
                            Conversation.id == session_id,
                            Conversation.user_id == user_id,
                        )
                    )
                )
                conv = result.scalar_one_or_none()
                if not conv:
                    return f"ERROR: Session not found or not yours: {session_id}"

                # Get messages
                result = await db.execute(
                    select(Message)
                    .where(Message.conversation_id == session_id)
                    .order_by(Message.created_at.desc())
                    .limit(limit)
                )
                messages = list(reversed(result.scalars().all()))

            if not messages:
                return "No messages in this session."

            lines = [f"Session {session_id[:8]}... ({len(messages)} messages)\n"]
            for msg in messages:
                role_label = "You" if msg.role == "user" else "Hex"
                ts = msg.created_at.strftime("%H:%M")
                content = msg.content[:500]
                if len(msg.content) > 500:
                    content += "..."
                lines.append(f"[{ts}] {role_label}: {content}")

            return "\n\n".join(lines)

        except Exception as e:
            logger.exception("sessions_history failed")
            return f"ERROR: {e}"

    # ------------------------------------------------------------------
    # 18. browser — headless browser automation
    # ------------------------------------------------------------------
    async def _tool_browser(self, inp: Dict[str, Any]) -> str:
        action = inp.get("action", "").strip().lower()
        url = inp.get("url", "").strip()

        if not action:
            return "ERROR: 'action' is required"
        if not url:
            return "ERROR: 'url' is required"

        try:
            from app.agent import browser as browser_svc
        except ImportError:
            return "ERROR: Playwright not installed. Run: pip install playwright && playwright install chromium"

        try:
            if action == "navigate":
                result = await browser_svc.navigate(url)
                return json.dumps(result)

            elif action == "screenshot":
                full_page = bool(inp.get("full_page", False))
                img_path = await browser_svc.screenshot(url, full_page=full_page)
                if img_path.startswith("ERROR:"):
                    return img_path
                # Send the screenshot to the user via Telegram
                if self.telegram_bot and self._chat_id:
                    try:
                        bot = self.telegram_bot.app.bot
                        with open(img_path, "rb") as f:
                            await bot.send_photo(chat_id=self._chat_id, photo=f, caption=f"Screenshot: {url[:80]}")
                    except Exception as e:
                        logger.warning("[BROWSER] Failed to send screenshot: %s", e)
                return f"Screenshot captured and sent: {url}"

            elif action == "extract_text":
                selector = inp.get("selector", None)
                return await browser_svc.extract_text(url, selector=selector)

            elif action in ("click", "fill", "evaluate"):
                selector = inp.get("selector", None)
                value = inp.get("value", None)
                return await browser_svc.run_action(url, action, selector=selector, value=value)

            else:
                return f"ERROR: Unknown action '{action}'. Use: navigate, screenshot, extract_text, click, fill, evaluate"

        except Exception as e:
            logger.exception("[BROWSER] Action '%s' failed", action)
            return f"ERROR: Browser action failed: {e}"


    # ------------------------------------------------------------------
    # 19. grep — search files for pattern
    # ------------------------------------------------------------------
    async def _tool_grep(self, inp: Dict[str, Any]) -> str:
        import fnmatch

        pattern = inp.get("pattern", "").strip()
        if not pattern:
            return "ERROR: 'pattern' is required"

        self._ensure_workspace()
        search_path = self._resolve_path(inp.get("path", ""))
        include_glob = inp.get("include", "")
        ignore_case = inp.get("ignore_case", True)
        max_results = min(int(inp.get("max_results", 50)), 200)

        flags = re.IGNORECASE if ignore_case else 0
        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            return f"ERROR: Invalid regex pattern: {e}"

        matches = []
        files_searched = 0

        def _walk(root: str):
            nonlocal files_searched
            for dirpath, dirnames, filenames in os.walk(root):
                # Skip hidden and common non-code dirs
                dirnames[:] = [d for d in dirnames if not d.startswith('.') and d not in (
                    'node_modules', '__pycache__', '.git', 'venv', '.venv', 'dist', 'build',
                )]
                for fname in filenames:
                    if include_glob and not fnmatch.fnmatch(fname, include_glob):
                        continue
                    fpath = os.path.join(dirpath, fname)
                    files_searched += 1
                    try:
                        with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                            for lineno, line in enumerate(f, 1):
                                if compiled.search(line):
                                    rel = os.path.relpath(fpath, search_path)
                                    matches.append(f"{rel}:{lineno}: {line.rstrip()}")
                                    if len(matches) >= max_results:
                                        return
                    except (PermissionError, OSError):
                        continue

        if os.path.isfile(search_path):
            try:
                with open(search_path, 'r', encoding='utf-8', errors='replace') as f:
                    for lineno, line in enumerate(f, 1):
                        if compiled.search(line):
                            matches.append(f"{os.path.basename(search_path)}:{lineno}: {line.rstrip()}")
                            if len(matches) >= max_results:
                                break
                files_searched = 1
            except (PermissionError, OSError) as e:
                return f"ERROR: {e}"
        else:
            _walk(search_path)

        if not matches:
            return f"No matches found ({files_searched} files searched)"
        header = f"Found {len(matches)} matches ({files_searched} files searched)"
        if len(matches) >= max_results:
            header += f" [limited to {max_results}]"
        return header + "\n\n" + "\n".join(matches)

    # ------------------------------------------------------------------
    # 20. find — find files by name pattern
    # ------------------------------------------------------------------
    async def _tool_find(self, inp: Dict[str, Any]) -> str:
        import fnmatch

        pattern = inp.get("pattern", "").strip()
        if not pattern:
            return "ERROR: 'pattern' is required"

        self._ensure_workspace()
        search_path = self._resolve_path(inp.get("path", ""))
        filter_type = inp.get("type", "all")
        max_depth = min(int(inp.get("max_depth", 10)), 20)
        max_results = min(int(inp.get("max_results", 100)), 500)

        results = []

        for dirpath, dirnames, filenames in os.walk(search_path):
            depth = dirpath.replace(search_path, "").count(os.sep)
            if depth >= max_depth:
                dirnames.clear()
                continue
            # Skip hidden/ignored dirs
            dirnames[:] = [d for d in dirnames if not d.startswith('.') and d not in (
                'node_modules', '__pycache__', '.git', 'venv', '.venv',
            )]

            if filter_type in ("dir", "all"):
                for d in dirnames:
                    if fnmatch.fnmatch(d, pattern):
                        rel = os.path.relpath(os.path.join(dirpath, d), search_path)
                        results.append(f"�� {rel}/")
                        if len(results) >= max_results:
                            break

            if filter_type in ("file", "all"):
                for f in filenames:
                    if fnmatch.fnmatch(f, pattern):
                        rel = os.path.relpath(os.path.join(dirpath, f), search_path)
                        size = os.path.getsize(os.path.join(dirpath, f))
                        results.append(f"📄 {rel}  ({self._human_size(size)})")
                        if len(results) >= max_results:
                            break

            if len(results) >= max_results:
                break

        if not results:
            return f"No matches for '{pattern}' in {search_path}"
        header = f"Found {len(results)} matches"
        if len(results) >= max_results:
            header += f" [limited to {max_results}]"
        return header + "\n\n" + "\n".join(results)

    # ------------------------------------------------------------------
    # 21. ls — list directory contents
    # ------------------------------------------------------------------
    async def _tool_ls(self, inp: Dict[str, Any]) -> str:
        self._ensure_workspace()
        path = self._resolve_path(inp.get("path", ""))
        show_all = inp.get("all", False)
        recursive = inp.get("recursive", False)
        max_depth = min(int(inp.get("max_depth", 2)), 5)

        if not os.path.isdir(path):
            return f"ERROR: Not a directory: {path}"

        lines = []

        def _list_dir(dirpath: str, depth: int, prefix: str = ""):
            try:
                entries = sorted(os.listdir(dirpath))
            except PermissionError:
                lines.append(f"{prefix}(permission denied)")
                return

            for entry in entries:
                if not show_all and entry.startswith('.'):
                    continue
                full = os.path.join(dirpath, entry)
                if os.path.isdir(full):
                    lines.append(f"{prefix}📁 {entry}/")
                    if recursive and depth < max_depth:
                        _list_dir(full, depth + 1, prefix + "  ")
                else:
                    try:
                        size = os.path.getsize(full)
                        mtime = datetime.fromtimestamp(os.path.getmtime(full)).strftime("%Y-%m-%d %H:%M")
                        lines.append(f"{prefix}📄 {entry}  {self._human_size(size)}  {mtime}")
                    except OSError:
                        lines.append(f"{prefix}📄 {entry}")

                if len(lines) > 500:
                    lines.append("... (truncated)")
                    return

        _list_dir(path, 0)

        if not lines:
            return "(empty directory)"
        return "\n".join(lines)

    @staticmethod
    def _human_size(size: int) -> str:
        """Convert bytes to human-readable size."""
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"

    # ------------------------------------------------------------------
    # 22. apply_patch — apply unified diff
    # ------------------------------------------------------------------
    async def _tool_apply_patch(self, inp: Dict[str, Any]) -> str:
        patch_text = inp.get("patch", "").strip()
        strip_n = int(inp.get("strip", 0))

        if not patch_text:
            return "ERROR: 'patch' is required"

        self._ensure_workspace()
        workspace = self._get_user_workspace()

        # Parse unified diff hunks
        files_patched = 0
        errors = []
        current_file = None
        hunks = []

        for line in patch_text.splitlines():
            if line.startswith("+++ "):
                # Save previous file's hunks
                if current_file and hunks:
                    result = self._apply_hunks(current_file, hunks, workspace)
                    if result.startswith("ERROR"):
                        errors.append(result)
                    else:
                        files_patched += 1
                    hunks = []

                path = line[4:].strip()
                if path.startswith("b/"):
                    path = path[2:]
                # Strip leading components
                parts = path.split("/")
                if strip_n < len(parts):
                    path = "/".join(parts[strip_n:])
                current_file = path

            elif line.startswith("@@ "):
                hunks.append({"header": line, "lines": []})
            elif hunks:
                hunks[-1]["lines"].append(line)

        # Apply last file
        if current_file and hunks:
            result = self._apply_hunks(current_file, hunks, workspace)
            if result.startswith("ERROR"):
                errors.append(result)
            else:
                files_patched += 1

        if errors:
            return f"Patched {files_patched} file(s) with {len(errors)} error(s):\n" + "\n".join(errors)
        if files_patched == 0:
            return "ERROR: No files were patched. Check the patch format."
        return f"Successfully patched {files_patched} file(s)."

    def _apply_hunks(self, rel_path: str, hunks: list, workspace: str) -> str:
        """Apply parsed hunks to a single file."""
        full_path = os.path.join(workspace, rel_path)
        if not os.path.isfile(full_path):
            return f"ERROR: File not found: {rel_path}"

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                original_lines = f.readlines()
        except Exception as e:
            return f"ERROR: Cannot read {rel_path}: {e}"

        # Simple line-based patch application
        result_lines = list(original_lines)
        offset = 0

        for hunk in hunks:
            header = hunk["header"]
            # Parse @@ -old_start,old_count +new_start,new_count @@
            import re as _re
            m = _re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', header)
            if not m:
                continue
            old_start = int(m.group(1)) - 1  # 0-indexed
            old_count = int(m.group(2) or 1)

            idx = old_start + offset
            new_lines = []
            removed = 0
            for line in hunk["lines"]:
                if line.startswith("-"):
                    removed += 1
                elif line.startswith("+"):
                    new_lines.append(line[1:] + "\n")
                elif line.startswith(" ") or line == "":
                    new_lines.append((line[1:] if line.startswith(" ") else line) + "\n")

            # Replace old lines with new lines
            result_lines[idx:idx + old_count] = new_lines
            offset += len(new_lines) - old_count

        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.writelines(result_lines)
            return f"Patched: {rel_path}"
        except Exception as e:
            return f"ERROR: Cannot write {rel_path}: {e}"

    # ------------------------------------------------------------------
    # 23. sessions_send — cross-session messaging
    # ------------------------------------------------------------------
    async def _tool_sessions_send(self, inp: Dict[str, Any]) -> str:
        message = inp.get("message", "").strip()
        if not message:
            return "ERROR: 'message' is required"

        session_id = inp.get("session_id")
        channel = inp.get("channel")

        # For now, send via Telegram if that's the active channel
        if self.telegram_bot and self._chat_id and (not channel or channel == "telegram"):
            try:
                bot = self.telegram_bot.app.bot
                await bot.send_message(chat_id=self._chat_id, text=message)
                return f"Message sent to Telegram chat {self._chat_id}"
            except Exception as e:
                return f"ERROR: Failed to send: {e}"

        return "ERROR: No active channel to send to. Specify a valid session_id or channel."


    async def _tool_session_status(self, inp: Dict[str, Any]) -> str:
        """Show current session status: model, tokens, messages, uptime."""
        try:
            from app.db.database import async_session_maker
            from app.db.models import Session, Message
            from sqlalchemy import select, func

            user_id = self._current_user_id
            chat_id = self._chat_id

            async with async_session_maker() as db:
                # Find current session
                from sqlalchemy import and_
                result = await db.execute(
                    select(Session).where(
                        and_(
                            Session.user_id == user_id,
                            Session.is_active == True,
                        )
                    ).order_by(Session.updated_at.desc()).limit(1)
                )
                session = result.scalar_one_or_none()
                if not session:
                    return "No active session found."

                # Count messages
                msg_count = await db.execute(
                    select(func.count()).where(Message.session_id == session.id)
                )
                count = msg_count.scalar() or 0

                # Token totals
                token_result = await db.execute(
                    select(
                        func.sum(Message.tokens_input),
                        func.sum(Message.tokens_output),
                    ).where(Message.session_id == session.id)
                )
                row = token_result.one()
                total_in = row[0] or 0
                total_out = row[1] or 0

                lines = [
                    f"Session: {session.id}",
                    f"Created: {session.created_at}",
                    f"Messages: {count}",
                    f"Tokens: {total_in} in / {total_out} out / {total_in + total_out} total",
                    f"Model: {settings.agent_model}",
                    f"Thinking budget: {settings.thinking_budget_default}",
                    f"Reranker: {'enabled' if settings.enable_reranker else 'disabled'}",
                ]
                return "\n".join(lines)
        except Exception as e:
            return f"Session status: {e}"

    async def _tool_agents_list(self, inp: Dict[str, Any]) -> str:
        """List all available agent personas from multi-agent router."""
        try:
            from app.agent.multi_agent import get_multi_agent_router
            router = get_multi_agent_router()
            personas = router.list_personas()
            if not personas:
                return "No agent personas registered."
            lines = []
            for p in personas:
                model = p.get("model") or "default"
                kw = ", ".join(p.get("keywords", [])) or "none"
                lines.append(
                    f"• {p['name']} (priority={p['priority']}, model={model})\n"
                    f"  {p.get('description', '')}\n"
                    f"  keywords: {kw}"
                )
            return f"Available agent personas ({len(personas)}):\n\n" + "\n\n".join(lines)
        except Exception as e:
            return f"ERROR: {e}"


    async def _tool_message(self, inp: Dict[str, Any]) -> str:
        """Cross-channel messaging: send/react/edit/delete/pin."""
        from app.agent.cross_channel import (
            send_cross_channel, react_cross_channel,
            edit_cross_channel, delete_cross_channel,
            pin_cross_channel,
        )
        import json as _json

        action = inp.get("action", "").lower()
        channel = inp.get("channel", "telegram").lower()
        target = inp.get("target", "")
        bot_refs = {"telegram_bot": self.telegram_bot}

        if not target:
            return "ERROR: 'target' (chat_id/channel_id) is required."

        if action == "send":
            text = inp.get("text", "")
            if not text:
                return "ERROR: 'text' is required for send action."
            result = await send_cross_channel(
                channel, target, text,
                reply_to=inp.get("reply_to"),
                thread_id=inp.get("thread_id"),
                bot_refs=bot_refs,
            )
            return _json.dumps(result)

        elif action == "react":
            message_id = inp.get("message_id", "")
            emoji = inp.get("emoji", "👍")
            if not message_id:
                return "ERROR: 'message_id' is required for react action."
            result = await react_cross_channel(
                channel, target, message_id, emoji, bot_refs=bot_refs,
            )
            return _json.dumps(result)

        elif action == "edit":
            message_id = inp.get("message_id", "")
            new_text = inp.get("text", "")
            if not message_id or not new_text:
                return "ERROR: 'message_id' and 'text' are required for edit action."
            result = await edit_cross_channel(
                channel, target, message_id, new_text, bot_refs=bot_refs,
            )
            return _json.dumps(result)

        elif action == "delete":
            message_id = inp.get("message_id", "")
            if not message_id:
                return "ERROR: 'message_id' is required for delete action."
            result = await delete_cross_channel(
                channel, target, message_id, bot_refs=bot_refs,
            )
            return _json.dumps(result)

        elif action == "pin":
            message_id = inp.get("message_id", "")
            if not message_id:
                return "ERROR: 'message_id' is required for pin action."
            result = await pin_cross_channel(
                channel, target, message_id, bot_refs=bot_refs,
            )
            return _json.dumps(result)

        else:
            return f"ERROR: Unknown action '{action}'. Use: send, react, edit, delete, pin."

    async def _tool_moderate(self, inp: Dict[str, Any]) -> str:
        """Execute moderation actions in group chats."""
        from app.agent.moderation import moderate_user
        import json as _json

        action = inp.get("action", "").lower()
        channel = inp.get("channel", "telegram").lower()
        chat_id = inp.get("chat_id", "")
        user_id = inp.get("user_id", "")

        if not all([action, chat_id, user_id]):
            return "ERROR: 'action', 'chat_id', and 'user_id' are required."

        if not settings.moderation_enabled:
            return "ERROR: Moderation is disabled. Set MODERATION_ENABLED=true to enable."

        result = await moderate_user(
            action=action,
            channel=channel,
            chat_id=chat_id,
            user_id=user_id,
            duration_seconds=int(inp.get("duration_seconds", 0)),
            reason=inp.get("reason", ""),
            bot_refs={"telegram_bot": self.telegram_bot},
        )
        return _json.dumps(result)

    async def _tool_config_reload(self, inp: Dict[str, Any]) -> str:
        """Hot-reload configuration settings."""
        from app.agent.config_reload import (
            reload_config, get_reloadable_fields, get_current_config,
        )
        import json as _json

        action = inp.get("action", "list").lower()

        if action == "list":
            fields = get_reloadable_fields()
            return "Reloadable config fields:\n" + "\n".join(f"  • {f}" for f in fields)

        elif action == "get":
            field = inp.get("field")
            if field:
                values = get_current_config([field])
            else:
                values = get_current_config()
            return _json.dumps(values, indent=2, default=str)

        elif action == "set":
            field = inp.get("field", "")
            value = inp.get("value", "")
            if not field:
                return "ERROR: 'field' is required for set action."
            results = reload_config({field: value})
            return _json.dumps(results)

        elif action == "reload_env":
            results = reload_config()
            if not results:
                return "No environment variables found for reloadable fields."
            return _json.dumps(results)

        else:
            return f"ERROR: Unknown action '{action}'. Use: list, get, set, reload_env."

    async def _tool_lanes_status(self, inp: Dict[str, Any]) -> str:
        """Show agent execution lane statistics."""
        from app.agent.lanes import get_lane_manager
        import json as _json

        lm = get_lane_manager()
        stats = lm.get_stats()

        active_runs = lm.get_active_runs()
        runs_info = []
        for r in active_runs:
            import time
            elapsed = time.time() - r.started_at
            runs_info.append({
                "run_id": r.run_id,
                "lane": r.lane.value,
                "user": r.user_id[:8] + "...",
                "model": r.model or "default",
                "elapsed_s": round(elapsed, 1),
            })

        output = {
            "summary": stats,
            "active_runs": runs_info,
        }
        return _json.dumps(output, indent=2)

    async def _tool_poll(self, inp: Dict[str, Any]) -> str:
        """Create a poll in a Telegram group chat."""
        if not self.telegram_bot or not self.telegram_bot.bot:
            return "ERROR: Telegram bot not connected."

        question = inp.get("question", "")
        options = inp.get("options", [])

        if not question:
            return "ERROR: 'question' is required."
        if len(options) < 2:
            return "ERROR: At least 2 options are required."
        if len(options) > 10:
            return "ERROR: Maximum 10 options allowed."

        chat_id = inp.get("chat_id") or self._chat_id
        if not chat_id:
            return "ERROR: 'chat_id' is required (or must be in a chat context)."

        try:
            kwargs = {
                "chat_id": int(chat_id),
                "question": question,
                "options": options,
                "is_anonymous": inp.get("is_anonymous", True),
            }

            poll_type = inp.get("type", "regular")
            if poll_type == "quiz":
                kwargs["type"] = "quiz"
                correct_id = inp.get("correct_option_id")
                if correct_id is not None:
                    kwargs["correct_option_id"] = int(correct_id)

            msg = await self.telegram_bot.bot.send_poll(**kwargs)
            return f"Poll created! Message ID: {msg.message_id}"

        except Exception as e:
            return f"ERROR creating poll: {e}"

    # ------------------------------------------------------------------
    # 31. thread — Telegram forum topic management
    # ------------------------------------------------------------------
    async def _tool_thread(self, inp: Dict[str, Any]) -> str:
        """Manage Telegram forum topics (threads)."""
        if not self.telegram_bot or not self.telegram_bot.bot:
            return "ERROR: Telegram bot not connected."

        action = inp.get("action", "")
        chat_id = inp.get("chat_id") or self._chat_id
        if not chat_id:
            return "ERROR: 'chat_id' is required."

        try:
            chat_id = int(chat_id)
        except (ValueError, TypeError):
            return "ERROR: Invalid chat_id."

        bot = self.telegram_bot.bot

        try:
            if action == "create":
                name = inp.get("name", "").strip()
                if not name:
                    return "ERROR: 'name' is required for creating a topic."
                kwargs = {"chat_id": chat_id, "name": name}
                icon_color = inp.get("icon_color")
                if icon_color:
                    kwargs["icon_color"] = int(icon_color)
                topic = await bot.create_forum_topic(**kwargs)
                return json.dumps({
                    "status": "created",
                    "topic_id": topic.message_thread_id,
                    "name": topic.name,
                })

            elif action == "close":
                topic_id = inp.get("topic_id")
                if not topic_id:
                    return "ERROR: 'topic_id' is required for close action."
                await bot.close_forum_topic(chat_id=chat_id, message_thread_id=int(topic_id))
                return f"Topic {topic_id} closed."

            elif action == "reopen":
                topic_id = inp.get("topic_id")
                if not topic_id:
                    return "ERROR: 'topic_id' is required for reopen action."
                await bot.reopen_forum_topic(chat_id=chat_id, message_thread_id=int(topic_id))
                return f"Topic {topic_id} reopened."

            elif action == "list":
                # Telegram Bot API doesn't have a list_forum_topics method,
                # but we can use getForumTopicIconStickers as a proxy or just note it
                return json.dumps({
                    "note": "Telegram Bot API does not provide a list_forum_topics endpoint. "
                            "Use the group's topic sidebar to see topics. "
                            "You can create new topics or close/reopen existing ones by ID.",
                })

            else:
                return f"ERROR: Unknown action '{action}'. Use: create, list, close, reopen."

        except Exception as e:
            return f"ERROR: Thread operation failed: {e}"

    # ------------------------------------------------------------------
    # 32. tts_prefs — per-user TTS preferences
    # ------------------------------------------------------------------
    async def _tool_tts_prefs(self, inp: Dict[str, Any]) -> str:
        """Get or set per-user TTS preferences."""
        from app.agent.tts_providers import get_user_tts_prefs, set_user_tts_prefs

        action = inp.get("action", "get")
        user_id = self._current_user_id
        if not user_id:
            return "ERROR: No user context — cannot manage TTS prefs."

        if action == "get":
            prefs = get_user_tts_prefs(user_id)
            return json.dumps({"user_id": user_id, "tts_preferences": prefs})

        elif action == "set":
            updates = {}
            for key in ("provider", "voice", "speed", "model"):
                val = inp.get(key)
                if val is not None:
                    updates[key] = val
            if not updates:
                return "ERROR: No preferences to set. Provide provider, voice, speed, or model."
            prefs = set_user_tts_prefs(user_id, **updates)
            return json.dumps({"user_id": user_id, "tts_preferences": prefs, "updated": list(updates.keys())})

        else:
            return f"ERROR: Unknown action '{action}'. Use: get, set."

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_path(self, path: str) -> str:
        """Resolve a path relative to the user's workspace if not absolute."""
        if not path:
            return self._get_user_workspace()
        # Expand ~ to the user's home directory
        path = os.path.expanduser(path)
        if os.path.isabs(path):
            return path
        return os.path.join(self._get_user_workspace(), path)
    
    def set_user_id(self, user_id: str):
        """Set the current user ID for memory tools."""
        self._current_user_id = user_id
    
    @property
    def _current_user_id(self) -> str:
        return getattr(self, "_user_id", "")
    
    @_current_user_id.setter
    def _current_user_id(self, value: str):
        self._user_id = value

    # ─────────────────────────────────────────────────────────
    # 33. canvas — Agent-to-UI push
    # ─────────────────────────────────────────────────────────
    async def _tool_canvas(self, inp: Dict[str, Any]) -> str:
        from app.agent.canvas import get_canvas_manager
        import json as _json

        mgr = get_canvas_manager()
        action = inp.get("action", "")
        user_id = self._current_user_id or "default"

        if action == "present":
            content = inp.get("content", "")
            content_type = inp.get("content_type", "html")
            title = inp.get("title", "")
            frame_id = inp.get("frame_id")
            result = await mgr.present(user_id, content, content_type, title, frame_id)
            return _json.dumps(result)
        elif action == "hide":
            result = await mgr.hide(user_id)
            return _json.dumps(result)
        elif action == "show":
            result = await mgr.show(user_id)
            return _json.dumps(result)
        elif action == "clear":
            frame_id = inp.get("frame_id")
            result = await mgr.clear(user_id, frame_id)
            return _json.dumps(result)
        elif action == "set_layout":
            layout = inp.get("layout", "stack")
            result = await mgr.set_layout(user_id, layout)
            return _json.dumps(result)
        elif action == "eval_js":
            code = inp.get("code", "")
            result = await mgr.evaluate_js(user_id, code)
            return _json.dumps(result)
        elif action == "snapshot":
            result = await mgr.snapshot(user_id)
            return _json.dumps(result, indent=2)
        else:
            return f"ERROR: Unknown canvas action '{action}'"

    # ─────────────────────────────────────────────────────────
    # 34. skill_marketplace — Skill discovery and management
    # ─────────────────────────────────────────────────────────
    async def _tool_skill_marketplace(self, inp: Dict[str, Any]) -> str:
        from app.agent.skills.marketplace import get_marketplace
        import json as _json

        mp = get_marketplace()
        action = inp.get("action", "")

        if action == "search":
            query = inp.get("query", "")
            tags = inp.get("tags")
            results = await mp.search(query, tags)
            return _json.dumps({"results": results, "count": len(results)}, indent=2)
        elif action == "install":
            name = inp.get("skill_name", "")
            if not name:
                return "ERROR: 'skill_name' required for install"
            result = await mp.install(name)
            return _json.dumps(result)
        elif action == "uninstall":
            name = inp.get("skill_name", "")
            if not name:
                return "ERROR: 'skill_name' required for uninstall"
            result = await mp.uninstall(name)
            return _json.dumps(result)
        elif action == "update":
            name = inp.get("skill_name", "")
            if not name:
                return "ERROR: 'skill_name' required for update"
            result = await mp.update(name)
            return _json.dumps(result)
        elif action == "list_installed":
            installed = mp.list_installed()
            return _json.dumps({"installed": installed, "count": len(installed)}, indent=2)
        elif action == "enable":
            name = inp.get("skill_name", "")
            ok = mp.enable_skill(name)
            return f"Enabled: {name}" if ok else f"ERROR: Skill not found: {name}"
        elif action == "disable":
            name = inp.get("skill_name", "")
            ok = mp.disable_skill(name)
            return f"Disabled: {name}" if ok else f"ERROR: Skill not found: {name}"
        else:
            return f"ERROR: Unknown marketplace action '{action}'"

    # ─────────────────────────────────────────────────────────
    # 35. doctor — System health checks
    # ─────────────────────────────────────────────────────────
    async def _tool_doctor(self, inp: Dict[str, Any]) -> str:
        from app.agent.cli_doctor import run_doctor
        import json as _json

        checks = inp.get("checks")
        fmt = inp.get("format", "text")

        report = await run_doctor(include=checks)

        if fmt == "json":
            return _json.dumps(report.to_dict(), indent=2)
        return report.to_text()

    # ─────────────────────────────────────────────────────────
    # 36. talk_mode — Continuous voice conversation management
    # ─────────────────────────────────────────────────────────
    async def _tool_talk_mode(self, inp: Dict[str, Any]) -> str:
        from app.agent.voice_handler import get_talk_mode_manager
        import json as _json

        mgr = get_talk_mode_manager()
        action = inp.get("action", "")

        if action == "status":
            sessions = mgr.list_sessions()
            return _json.dumps({
                "active_sessions": sessions,
                "active_count": mgr.active_count,
            }, indent=2)
        elif action == "start":
            user_id = self._current_user_id or "default"
            sess = mgr.start_session(user_id)
            return _json.dumps(sess.to_dict())
        elif action == "stop":
            user_id = self._current_user_id or "default"
            ended = mgr.end_session(user_id)
            return f"Talk mode ended" if ended else "No active talk mode session"
        else:
            return f"ERROR: Unknown talk_mode action '{action}'"
