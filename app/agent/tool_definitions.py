"""
Tool definitions for the HexBrain Agent Runtime.

Each tool is defined in Anthropic's tool format:
{ name, description, input_schema (JSON Schema) }
"""

from typing import List, Dict, Any


def get_agent_tools() -> List[Dict[str, Any]]:
    """Return all tool definitions available to the agent."""
    return [
        # ------------------------------------------------------------------
        # 1. Shell execution
        # ------------------------------------------------------------------
        {
            "name": "exec",
            "description": (
                "Execute a shell command and return stdout/stderr. "
                "Use for running scripts, checking system state, installing packages, etc. "
                "Commands run in a sandboxed workspace. A timeout is enforced. "
                "IMPORTANT: Destructive commands (rm, rmdir, unlink, shred) require "
                "confirmed=true. You MUST ask the user for explicit confirmation first."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                    "workdir": {
                        "type": "string",
                        "description": "Working directory (defaults to agent workspace).",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 30, max 120).",
                    },
                    "confirmed": {
                        "type": "boolean",
                        "description": "Set to true ONLY after the user explicitly confirmed a destructive operation. Never set this without asking the user first.",
                    },
                },
                "required": ["command"],
            },
        },
        # ------------------------------------------------------------------
        # 1b. PTY exec
        # ------------------------------------------------------------------
        {
            "name": "pty_exec",
            "description": (
                "Execute a command in a pseudo-terminal (PTY). "
                "Use for TTY-requiring commands like top, htop, less, vim, "
                "or interactive CLIs. Returns captured output with ANSI codes stripped."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute in a PTY",
                    },
                    "workdir": {
                        "type": "string",
                        "description": "Working directory (optional)",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (max 120, default 30)",
                    },
                    "rows": {
                        "type": "integer",
                        "description": "Terminal rows (default 24)",
                    },
                    "cols": {
                        "type": "integer",
                        "description": "Terminal columns (default 80)",
                    },
                },
                "required": ["command"],
            },
        },
        # ------------------------------------------------------------------
        # 2. Read file
        # ------------------------------------------------------------------
        {
            "name": "read_file",
            "description": (
                "Read the contents of a file. Returns text content, "
                "optionally from a specific byte offset and limited to a number of lines."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or workspace-relative file path.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line offset to start reading from (0-based).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of lines to return.",
                    },
                },
                "required": ["path"],
            },
        },
        # ------------------------------------------------------------------
        # 3. Write file
        # ------------------------------------------------------------------
        {
            "name": "write_file",
            "description": (
                "Create or overwrite a file with the given content. "
                "Parent directories are created automatically."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or workspace-relative file path.",
                    },
                    "content": {
                        "type": "string",
                        "description": "File content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
        # ------------------------------------------------------------------
        # 4. Edit file (find & replace)
        # ------------------------------------------------------------------
        {
            "name": "edit_file",
            "description": (
                "Find and replace text in a file. The old_text must match exactly "
                "(including whitespace). Returns confirmation or error."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or workspace-relative file path.",
                    },
                    "old_text": {
                        "type": "string",
                        "description": "The exact text to find in the file.",
                    },
                    "new_text": {
                        "type": "string",
                        "description": "The replacement text.",
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
        # ------------------------------------------------------------------
        # 5. Memory search
        # ------------------------------------------------------------------
        {
            "name": "memory_search",
            "description": (
                "Search HexBrain's memory system using semantic search. "
                "Returns matching memories ranked by relevance."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query.",
                    },
                    "brain_type": {
                        "type": "string",
                        "description": "Filter by brain type: 'user', 'agent', or 'work'.",
                        "enum": ["user", "agent", "work"],
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
        # ------------------------------------------------------------------
        # 6. Memory store
        # ------------------------------------------------------------------
        {
            "name": "memory_store",
            "description": (
                "Store a new memory into HexBrain. "
                "Automatically deduplicates and merges with existing memories."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The memory content to store.",
                    },
                    "category": {
                        "type": "string",
                        "description": (
                            "Memory category: identity, preferences, beliefs, emotions, people, "
                            "places, family, experiences, projects, schedule, work, learning, "
                            "knowledge, tools, media, health, habits, food, travel, goals, context."
                        ),
                    },
                    "brain_type": {
                        "type": "string",
                        "description": "Brain type (default 'user').",
                        "enum": ["user", "agent", "work"],
                    },
                    "importance": {
                        "type": "number",
                        "description": "Importance score 0.0-1.0 (default 0.5).",
                    },
                },
                "required": ["content", "category"],
            },
        },
        # ------------------------------------------------------------------
        # 7. Web search
        # ------------------------------------------------------------------
        {
            "name": "web_search",
            "description": (
                "Search the web and return a list of results with titles, URLs, and snippets."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of results to return (default 5, max 10).",
                    },
                },
                "required": ["query"],
            },
        },
        # ------------------------------------------------------------------
        # 8. Web fetch
        # ------------------------------------------------------------------
        {
            "name": "web_fetch",
            "description": (
                "Fetch a URL and extract readable text content. "
                "Strips HTML, scripts, and styles. Good for reading articles and documentation."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Max characters to return (default 10000).",
                    },
                },
                "required": ["url"],
            },
        },
        # ------------------------------------------------------------------
        # 9. Send file to user
        # ------------------------------------------------------------------
        {
            "name": "send_file",
            "description": (
                "Send a file from the workspace to the user via Telegram. "
                "Use this after creating a file (e.g. .docx, .pdf, .csv, .zip) "
                "that the user asked for. The file must exist on disk first — "
                "create it with write_file or exec, then send it with this tool."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or workspace-relative path to the file to send.",
                    },
                    "caption": {
                        "type": "string",
                        "description": "Optional caption/message to include with the file.",
                    },
                },
                "required": ["path"],
            },
        },
        # ------------------------------------------------------------------
        # 10. Send image/photo to user
        # ------------------------------------------------------------------
        {
            "name": "send_photo",
            "description": (
                "Send an image/photo from the workspace to the user via Telegram. "
                "Use this after creating or downloading an image file. "
                "Supports .jpg, .png, .gif, .webp formats."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or workspace-relative path to the image file.",
                    },
                    "caption": {
                        "type": "string",
                        "description": "Optional caption/message to include with the photo.",
                    },
                },
                "required": ["path"],
            },
        },
        # ------------------------------------------------------------------
        # 11. Analyze image — GPT vision on URL or workspace file
        # ------------------------------------------------------------------
        {
            "name": "analyze_image",
            "description": (
                "Analyze an image using GPT vision. Accepts an image URL or a workspace file path. "
                "Use when you need to describe, extract text (OCR), or answer questions about an image "
                "that the user referenced by URL or that you downloaded/created."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": "Image URL (https://...) or workspace-relative file path.",
                    },
                    "question": {
                        "type": "string",
                        "description": "What to analyze/look for in the image (default: 'Describe this image in detail').",
                    },
                },
                "required": ["image"],
            },
        },
        # ------------------------------------------------------------------
        # 12. Cron — scheduled tasks
        # ------------------------------------------------------------------
        {
            "name": "cron",
            "description": (
                "Manage scheduled tasks. The agent can create reminders, periodic checks, "
                "or any recurring task. Actions: add (create job), list (show jobs), "
                "remove (delete job), run (trigger a job now)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "list", "remove", "run"],
                        "description": "The action to perform.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Job name (for add).",
                    },
                    "schedule": {
                        "type": "string",
                        "description": (
                            "Schedule expression (for add). Accepts: "
                            "cron expr ('*/30 * * * *'), interval ('30m', '2h', '1d'), "
                            "ISO datetime ('2025-12-31 09:00'), or relative ('in 5m')."
                        ),
                    },
                    "message": {
                        "type": "string",
                        "description": "The prompt/message to run on schedule (for add).",
                    },
                    "job_id": {
                        "type": "string",
                        "description": "Job ID (for remove/run).",
                    },
                },
                "required": ["action"],
            },
        },
        # ------------------------------------------------------------------
        # 13. Spawn — background sub-agent task
        # ------------------------------------------------------------------
        {
            "name": "spawn",
            "description": (
                "Spawn a background task that runs independently and reports back when done. "
                "Use for: long-running research, complex multi-step tasks, work that shouldn't block "
                "the conversation. The result will be announced in chat when the task completes."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Detailed task description. Be specific about what to do and what to return.",
                    },
                    "label": {
                        "type": "string",
                        "description": "Short label for the task (shown in status updates).",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model override (optional). Use a cheaper model for simple tasks.",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Max time in seconds (default 300, max 600).",
                    },
                },
                "required": ["task"],
            },
        },
        # ------------------------------------------------------------------
        # 14. Process — long-running background shell process management
        # ------------------------------------------------------------------
        {
            "name": "process",
            "description": (
                "Manage long-running background shell processes (servers, watchers, etc). "
                "Unlike exec which blocks until done, process starts commands in the background "
                "and lets you check output or stop them later. "
                "Actions: start (launch process), status (check running/stopped), "
                "output (get stdout/stderr tail), stop (kill process), list (all processes)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["start", "status", "output", "stop", "list"],
                        "description": "Action to perform.",
                    },
                    "command": {
                        "type": "string",
                        "description": "Shell command to run (for start).",
                    },
                    "label": {
                        "type": "string",
                        "description": "Short label for the process (for start, e.g. 'dev-server').",
                    },
                    "process_id": {
                        "type": "string",
                        "description": "Process ID (for status/output/stop).",
                    },
                    "tail_lines": {
                        "type": "integer",
                        "description": "Number of output lines to return (for output, default 50).",
                    },
                },
                "required": ["action"],
            },
        },
        # ------------------------------------------------------------------
        # 15. TTS — text-to-speech voice messages
        # ------------------------------------------------------------------
        {
            "name": "tts",
            "description": (
                "Convert text to speech and send as a voice message in Telegram. "
                "Use when the user asks to read something aloud, wants a voice reply, "
                "or you think audio would be better than text (e.g. pronunciation, language learning). "
                "Supports multiple voices and speed control."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to speak (max 4096 chars).",
                    },
                    "voice": {
                        "type": "string",
                        "enum": ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"],
                        "description": "Voice to use (default: nova).",
                    },
                    "speed": {
                        "type": "number",
                        "description": "Playback speed 0.25–4.0 (default 1.0).",
                    },
                    "instructions": {
                        "type": "string",
                        "description": "Optional instructions for how to speak (tone, emotion, emphasis). Only works with gpt-4o-mini-tts model.",
                    },
                    "provider": {
                        "type": "string",
                        "enum": ["openai", "elevenlabs", "edge"],
                        "description": "TTS provider. 'openai' (default), 'elevenlabs' (natural voices), 'edge' (free).",
                    },
                },
                "required": ["text"],
            },
        },
        # ------------------------------------------------------------------
        # 16. Sessions list — view conversation sessions
        # ------------------------------------------------------------------
        {
            "name": "sessions_list",
            "description": (
                "List the user's conversation sessions. "
                "Returns session IDs, message counts, and timestamps."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max sessions to return (default 10).",
                    },
                    "active_only": {
                        "type": "boolean",
                        "description": "Only show active sessions (default true).",
                    },
                },
            },
        },
        # ------------------------------------------------------------------
        # 17. Sessions history — view messages from a session
        # ------------------------------------------------------------------
        {
            "name": "sessions_history",
            "description": (
                "View message history from a specific conversation session. "
                "Returns the last N messages with roles and timestamps."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session/conversation ID to view.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max messages to return (default 20).",
                    },
                },
                "required": ["session_id"],
            },
        },
        # ------------------------------------------------------------------
        # 18. Browser — headless browser automation
        # ------------------------------------------------------------------
        {
            "name": "browser",
            "description": (
                "Control a headless browser (Chromium). "
                "Actions: navigate (go to URL), screenshot (capture page), "
                "extract_text (get page text), click (click element), "
                "fill (type into input), evaluate (run JavaScript). "
                "Use for: web scraping when web_fetch isn't enough, "
                "interacting with dynamic pages, taking screenshots, "
                "form submission, testing web apps."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["navigate", "screenshot", "extract_text", "click", "fill", "evaluate"],
                        "description": "Action to perform.",
                    },
                    "url": {
                        "type": "string",
                        "description": "URL to navigate to.",
                    },
                    "selector": {
                        "type": "string",
                        "description": "CSS selector for click/fill/extract_text actions.",
                    },
                    "value": {
                        "type": "string",
                        "description": "Text to fill (for fill action) or JavaScript code (for evaluate).",
                    },
                    "full_page": {
                        "type": "boolean",
                        "description": "Capture full page screenshot (default false).",
                    },
                },
                "required": ["action", "url"],
            },
        },
    ]


def get_extended_tools():
    """Return additional platform tools (grep, find, ls, apply_patch, sessions_send, webhook)."""
    return [
        # ------------------------------------------------------------------
        # 19. grep — search files for pattern
        # ------------------------------------------------------------------
        {
            "name": "grep",
            "description": (
                "Search for a pattern across files in the workspace. "
                "Returns matching lines with file paths and line numbers. "
                "Supports regex patterns and case-insensitive search. "
                "Much faster than exec + grep for workspace searches."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern (regex supported).",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search in (defaults to workspace root).",
                    },
                    "include": {
                        "type": "string",
                        "description": "File glob pattern to include (e.g. '*.py', '*.ts').",
                    },
                    "ignore_case": {
                        "type": "boolean",
                        "description": "Case-insensitive search (default true).",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max matching lines to return (default 50).",
                    },
                },
                "required": ["pattern"],
            },
        },
        # ------------------------------------------------------------------
        # 20. find — find files by name pattern
        # ------------------------------------------------------------------
        {
            "name": "find",
            "description": (
                "Find files and directories by name pattern in the workspace. "
                "Supports glob patterns. Returns paths of matching files."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Filename glob pattern (e.g. '*.py', 'README*', 'test_*').",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (defaults to workspace root).",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["file", "dir", "all"],
                        "description": "Filter by type: file, dir, or all (default: all).",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum directory depth to search (default: 10).",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max results to return (default 100).",
                    },
                },
                "required": ["pattern"],
            },
        },
        # ------------------------------------------------------------------
        # 21. ls — list directory contents
        # ------------------------------------------------------------------
        {
            "name": "ls",
            "description": (
                "List contents of a directory with file sizes, types, and modification times. "
                "More informative than exec + ls."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path (defaults to workspace root).",
                    },
                    "all": {
                        "type": "boolean",
                        "description": "Include hidden files (default false).",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "List recursively (default false). Use max_depth to limit.",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Max depth for recursive listing (default 2).",
                    },
                },
            },
        },
        # ------------------------------------------------------------------
        # 22. apply_patch — apply unified diff
        # ------------------------------------------------------------------
        {
            "name": "apply_patch",
            "description": (
                "Apply a unified diff patch to one or more files. "
                "Accepts standard unified diff format (output of `diff -u` or `git diff`). "
                "Use this for multi-line or complex edits that are hard to express with edit_file."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "string",
                        "description": "The unified diff patch content.",
                    },
                    "strip": {
                        "type": "integer",
                        "description": "Number of leading path components to strip (like patch -pN, default 0).",
                    },
                },
                "required": ["patch"],
            },
        },
        # ------------------------------------------------------------------
        # 23. sessions_send — send message to another session / channel
        # ------------------------------------------------------------------
        {
            "name": "sessions_send",
            "description": (
                "Send a message to a different conversation session or channel. "
                "Use this to notify the user on another channel, or send a message "
                "to a sub-agent session. Requires a target session_id or channel."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Target session ID to send the message to.",
                    },
                    "message": {
                        "type": "string",
                        "description": "The message to send.",
                    },
                    "channel": {
                        "type": "string",
                        "description": "Target channel type (telegram, discord, slack, web).",
                    },
                },
                "required": ["message"],
            },
        },
    
        # ------------------------------------------------------------------
        # 24. session_status — current session statistics
        # ------------------------------------------------------------------
        {
            "name": "session_status",
            "description": (
                "Show current session status including model, token usage, "
                "message count, uptime, and configuration settings. Use this "
                "when the user asks about session info or usage stats."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        },
        # ------------------------------------------------------------------
        # 25. agents_list — list available agent personas
        # ------------------------------------------------------------------
        {
            "name": "agents_list",
            "description": (
                "List all available agent personas in the multi-agent router. "
                "Shows persona names, descriptions, models, priority, and keywords."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        },
    
        # ------------------------------------------------------------------
        # 26. message — cross-channel messaging
        # ------------------------------------------------------------------
        {
            "name": "message",
            "description": (
                "Send, react, edit, delete, or pin a message on any connected channel "
                "(telegram, discord, slack, whatsapp). Use action='send' to send a new "
                "message, 'react' to add a reaction, 'edit' to update a message, "
                "'delete' to remove, or 'pin' to pin."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["send", "react", "edit", "delete", "pin"],
                        "description": "The messaging action to perform.",
                    },
                    "channel": {
                        "type": "string",
                        "enum": ["telegram", "discord", "slack", "whatsapp"],
                        "description": "Target channel.",
                    },
                    "target": {
                        "type": "string",
                        "description": "Target chat/channel/phone ID.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Message text (for send/edit).",
                    },
                    "message_id": {
                        "type": "string",
                        "description": "Message ID (for react/edit/delete/pin).",
                    },
                    "emoji": {
                        "type": "string",
                        "description": "Emoji for reaction (e.g., '👍').",
                    },
                    "thread_id": {
                        "type": "string",
                        "description": "Thread/topic ID for threaded messages.",
                    },
                },
                "required": ["action", "channel", "target"],
            },
        },
        # ------------------------------------------------------------------
        # 27. moderate — group moderation actions
        # ------------------------------------------------------------------
        {
            "name": "moderate",
            "description": (
                "Execute moderation actions in group chats: timeout, kick, ban, "
                "unban, mute, or unmute a user. Only works in groups where the "
                "bot has admin privileges."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["timeout", "kick", "ban", "unban", "mute", "unmute"],
                        "description": "The moderation action.",
                    },
                    "channel": {
                        "type": "string",
                        "enum": ["telegram", "discord", "slack"],
                        "description": "Channel where the group is.",
                    },
                    "chat_id": {
                        "type": "string",
                        "description": "Group/chat identifier.",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Target user identifier.",
                    },
                    "duration_seconds": {
                        "type": "integer",
                        "description": "Duration in seconds (for timeout/mute/ban). 0=permanent.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for the moderation action.",
                    },
                },
                "required": ["action", "channel", "chat_id", "user_id"],
            },
        },
        # ------------------------------------------------------------------
        # 28. config_reload — hot-reload configuration
        # ------------------------------------------------------------------
        {
            "name": "config_reload",
            "description": (
                "Hot-reload configuration settings without restarting the server. "
                "Use action='list' to see reloadable fields, 'get' to read current "
                "values, or 'set' to update a field. Security-sensitive fields "
                "(API keys, DB URL) cannot be reloaded."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "get", "set", "reload_env"],
                        "description": "list=show fields, get=read values, set=update, reload_env=re-read from env.",
                    },
                    "field": {
                        "type": "string",
                        "description": "Config field name (for get/set).",
                    },
                    "value": {
                        "type": "string",
                        "description": "New value (for set action).",
                    },
                },
                "required": ["action"],
            },
        },
        # ------------------------------------------------------------------
        # 29. lanes_status — view agent lane statistics
        # ------------------------------------------------------------------
        {
            "name": "lanes_status",
            "description": (
                "View agent execution lane statistics. Shows active runs by lane "
                "(main, subagent, cron, hook), concurrency usage, and run history."
            ),
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        },
        # ------------------------------------------------------------------
        # 30. poll — create a poll in a group chat
        # ------------------------------------------------------------------
        {
            "name": "poll",
            "description": (
                "Create a poll in a Telegram group chat. Supports regular polls "
                "and quiz-style polls with a correct answer."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "chat_id": {
                        "type": "string",
                        "description": "Target chat ID for the poll.",
                    },
                    "question": {
                        "type": "string",
                        "description": "The poll question.",
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of poll options (2-10 items).",
                    },
                    "is_anonymous": {
                        "type": "boolean",
                        "description": "Whether the poll is anonymous. Default: true.",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["regular", "quiz"],
                        "description": "Poll type. Default: regular.",
                    },
                    "correct_option_id": {
                        "type": "integer",
                        "description": "0-based index of correct answer (for quiz type).",
                    },
                },
                "required": ["question", "options"],
            },
        },
        # ------------------------------------------------------------------
        # 31. thread — Telegram forum topic / thread management
        # ------------------------------------------------------------------
        {
            "name": "thread",
            "description": (
                "Manage Telegram forum topics (threads). Create new topics, "
                "list existing topics, or reply to a specific topic in a forum-enabled group."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "list", "close", "reopen"],
                        "description": "Action: create a topic, list topics, close or reopen a topic.",
                    },
                    "chat_id": {
                        "type": "string",
                        "description": "Target group chat ID (must be a forum-enabled supergroup).",
                    },
                    "name": {
                        "type": "string",
                        "description": "Topic name (for create action).",
                    },
                    "icon_color": {
                        "type": "integer",
                        "description": "Topic icon color as integer (for create action).",
                    },
                    "topic_id": {
                        "type": "integer",
                        "description": "Topic/thread ID (for close/reopen actions).",
                    },
                },
                "required": ["action", "chat_id"],
            },
        },
        # ------------------------------------------------------------------
        # 32. tts_prefs — manage per-user TTS preferences
        # ------------------------------------------------------------------
        {
            "name": "tts_prefs",
            "description": (
                "Get or set the user's text-to-speech preferences. "
                "Includes provider (openai/elevenlabs/edge), voice, speed, and model."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["get", "set"],
                        "description": "get = show current prefs, set = update prefs.",
                    },
                    "provider": {
                        "type": "string",
                        "enum": ["openai", "elevenlabs", "edge"],
                        "description": "TTS provider to set.",
                    },
                    "voice": {
                        "type": "string",
                        "description": "Voice ID or name to set.",
                    },
                    "speed": {
                        "type": "number",
                        "description": "TTS speed multiplier (0.25–4.0).",
                    },
                },
                "required": ["action"],
            },
        },
    
        # ─────────────────────────────────────────────────────────
        # 33. canvas — Agent-to-UI push
        # ─────────────────────────────────────────────────────────
        {
            "name": "canvas",
            "description": (
                "Present content on the user's visual canvas (A2UI). "
                "Actions: present (push HTML/markdown/code/chart), hide, show, clear, "
                "set_layout (stack/grid/tabs/split), eval_js, snapshot."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["present", "hide", "show", "clear", "set_layout", "eval_js", "snapshot"],
                        "description": "Canvas action to perform.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to present (HTML, markdown, code, JSON data).",
                    },
                    "content_type": {
                        "type": "string",
                        "enum": ["html", "markdown", "json_data", "chart", "code", "image", "iframe", "custom"],
                        "description": "Type of content being presented.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Title for the canvas frame.",
                    },
                    "frame_id": {
                        "type": "string",
                        "description": "Frame ID to update or clear. Auto-generated if omitted.",
                    },
                    "layout": {
                        "type": "string",
                        "enum": ["stack", "grid", "tabs", "split"],
                        "description": "Canvas layout mode (for set_layout action).",
                    },
                    "code": {
                        "type": "string",
                        "description": "JavaScript code (for eval_js action).",
                    },
                },
                "required": ["action"],
            },
        },
        # ─────────────────────────────────────────────────────────
        # 34. skill_marketplace — Discover and install skills
        # ─────────────────────────────────────────────────────────
        {
            "name": "skill_marketplace",
            "description": (
                "Search, install, update, or uninstall agent skills from the marketplace. "
                "Actions: search (query/tags), install, uninstall, update, list_installed, enable, disable."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["search", "install", "uninstall", "update", "list_installed", "enable", "disable"],
                        "description": "Marketplace action.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query (for search action).",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by tags (for search action).",
                    },
                    "skill_name": {
                        "type": "string",
                        "description": "Skill name (for install/uninstall/update/enable/disable).",
                    },
                },
                "required": ["action"],
            },
        },
        # ─────────────────────────────────────────────────────────
        # 35. doctor — System health check
        # ─────────────────────────────────────────────────────────
        {
            "name": "doctor",
            "description": (
                "Run system health checks and diagnostics. "
                "Checks: python_deps, config, database, disk_space, docker, "
                "api_key_openai, api_key_anthropic, workspace, memory_system, telegram_bot, browser."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "checks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific checks to run. Omit for all checks.",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["json", "text"],
                        "description": "Output format: json (structured) or text (human-readable).",
                    },
                },
                "required": [],
            },
        },
        # ─────────────────────────────────────────────────────────
        # 36. talk_mode — Continuous voice conversation
        # ─────────────────────────────────────────────────────────
        {
            "name": "talk_mode",
            "description": (
                "Manage talk mode (continuous voice conversation). "
                "Actions: status (list active sessions), start, stop."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["status", "start", "stop"],
                        "description": "Talk mode action.",
                    },
                },
                "required": ["action"],
            },
        },
    ]