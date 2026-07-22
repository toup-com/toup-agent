"""
Tool definitions for the Toup Agent Runtime.

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
                "Search your memory system using semantic search. "
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
                "Store a new memory into your brain. "
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
                "Search the web; returns ranked results as title + URL + snippet "
                "(no page bodies). Reason over the snippets FIRST — they often "
                "answer the question outright. When a task needs several angles "
                "(comparisons, multi-part questions), issue MULTIPLE web_search "
                "calls in the SAME turn — they run concurrently — instead of "
                "searching, waiting, then searching again. Only web_fetch a "
                "result when its snippet is insufficient. Keep it lean — a "
                "few well-chosen queries beat a dozen near-duplicate ones; do "
                "NOT re-search the same thing with slight rewordings."
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
                        "description": "Number of results to return (default 8, max 10).",
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
                "Fetch a URL and extract readable text (HTML/scripts/styles "
                "stripped). Use it on-demand — only for the few results whose "
                "search snippet wasn't enough, not for every result. When you "
                "need several pages, call web_fetch for ALL of them in the SAME "
                "turn: they run concurrently (≈ the time of one fetch) and the "
                "combined output is token-budgeted, so batching is both faster "
                "and safe. Output is truncated to a token budget automatically. "
                "Fetch sparingly: each fetch renders a full page and is slow, so "
                "open only the 2-3 most promising results — the snippets (now "
                "with extra passages) usually already contain the answer."
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
                        "description": (
                            "Optional upper bound on characters returned. Usually "
                            "omit it — output is token-budgeted automatically."
                        ),
                    },
                },
                "required": ["url"],
            },
        },
        # ------------------------------------------------------------------
        # 8b. Extension-routed web search (runs in the user's real Chrome)
        # ------------------------------------------------------------------
        {
            "name": "extension_search",
            "description": (
                "Run a web search inside the user's real Chrome via the Toup Chrome "
                "extension. Uses the user's residential IP and session — far more "
                "resilient to bot detection than server-side scraping. Opens a "
                "visible tab. Falls back to web_search transparently if the user "
                "hasn't installed/paired the extension; agents can call this "
                "freely without checking first."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query":  {"type": "string", "description": "The search query."},
                    "engine": {"type": "string", "enum": ["google", "bing", "duckduckgo"], "description": "Search engine (default google)."},
                    "top_n":  {"type": "integer", "description": "Number of ranked results (1-20, default 10)."},
                    "locale": {"type": "string", "description": "Optional locale hint, e.g. 'en-US'."},
                },
                "required": ["query"],
            },
        },
        # ------------------------------------------------------------------
        # 8c. Extension-routed page read (Readability-clean text)
        # ------------------------------------------------------------------
        {
            "name": "extension_read",
            "description": (
                "Fetch a URL in the user's real Chrome and return cleaned readable "
                "text (nav/ads/boilerplate stripped). Use this for sites that block "
                "server-side fetch (LinkedIn, Twitter/X, Substack, Medium paywalls), "
                "or when you specifically need pages behind the user's login. "
                "Falls back to web_fetch if the extension isn't connected."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "url":       {"type": "string", "description": "The URL to read."},
                    "max_chars": {"type": "integer", "description": "Max characters to return (default 12000)."},
                    "use_existing_tab": {"type": "boolean", "description": "Reuse an open tab on the same hostname if available (default true)."},
                },
                "required": ["url"],
            },
        },
        # ------------------------------------------------------------------
        # 8d. Extension-routed research (search + read top N)
        # ------------------------------------------------------------------
        {
            "name": "extension_research",
            "description": (
                "Multi-step research: run a search, open the top results in the "
                "user's Chrome, extract clean text from each, and return a "
                "structured bundle. Far faster than chaining extension_search + "
                "extension_read manually. Falls back to web_search + web_fetch "
                "if the extension is unavailable."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query":  {"type": "string", "description": "The research question."},
                    "depth":  {"type": "integer", "description": "How many top results to open (1-10, default 5)."},
                    "engine": {"type": "string", "enum": ["google", "bing", "duckduckgo"], "description": "Search engine (default google)."},
                    "per_page_chars": {"type": "integer", "description": "Max chars to extract per page (default 4000)."},
                },
                "required": ["query"],
            },
        },
        # ------------------------------------------------------------------
        # 8e. browser_session_start — open a controlled tab in user's Chrome
        # ------------------------------------------------------------------
        {
            "name": "browser_session_start",
            "description": (
                "Start a stateful browser-control session inside the user's real "
                "Chrome (via the Toup extension). Returns a session_id you MUST "
                "pass to every subsequent browser_action / browser_screenshot / "
                "browser_session_end call. Sessions persist across turns and "
                "auto-end after 10 min of inactivity. Requires the extension to "
                "be paired; without it, every browser_* call returns an error "
                "(use web_search / extension_search for stateless lookups)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "name":             {"type": "string", "description": "Human-readable label for the session (shown in the UI)."},
                    "hint_url":         {"type": "string", "description": "Optional starting URL; extension opens a fresh tab here."},
                    "share_active_tab": {"type": "boolean", "description": "If true and hint_url is omitted, take over the user's currently-focused tab."},
                },
                "required": [],
            },
        },
        # ------------------------------------------------------------------
        # 8f. browser_session_end
        # ------------------------------------------------------------------
        {
            "name": "browser_session_end",
            "description": "End a browser session. Optionally closes the underlying tab.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "From browser_session_start."},
                    "close_tab":  {"type": "boolean", "description": "If true, also close the tab in Chrome (default false)."},
                },
                "required": ["session_id"],
            },
        },
        # ------------------------------------------------------------------
        # 8g. browser_action — the fat tool (navigate/click/type/etc.)
        # ------------------------------------------------------------------
        {
            "name": "browser_action",
            "description": (
                "Execute a single browser action inside a session. The `kind` "
                "field selects: navigate, click, type, scroll, select, "
                "wait_for, evaluate, extract, dom_snapshot, read_logs. "
                "`args` is kind-specific. Set `capture.screenshot=true` to get "
                "a JPEG of the post-action viewport in the result. Element "
                "targeting prefers `ref` (from a prior dom_snapshot) over CSS "
                "`selector` over `(x,y)`. "
                "For dynamic apps where selectors are unreliable (Gmail, "
                "LinkedIn, Notion, canvas/PDF, shadow DOM), set "
                "`args.use_vision=true` and provide `args.target_description` "
                "in plain English — a Claude Vision call grounds the action "
                "to pixel coordinates from the current screenshot."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": ["navigate", "click", "type", "scroll", "select",
                                  "wait_for", "evaluate", "extract", "dom_snapshot", "read_logs"],
                    },
                    "args":    {"type": "object", "description": "Kind-specific arguments. See PROTOCOL.md §3."},
                    "capture": {
                        "type": "object",
                        "properties": {
                            "screenshot": {"type": "boolean", "description": "Include a JPEG of the post-action viewport."},
                            "snapshot":   {"type": "boolean", "description": "Include a fresh dom_snapshot in the result."},
                            "quality":    {"type": "integer", "description": "JPEG quality 1-100 (default 80)."},
                        },
                    },
                    "timeout_s": {"type": "integer", "description": "Per-action timeout (default 30)."},
                },
                "required": ["session_id", "kind"],
            },
        },
        # ------------------------------------------------------------------
        # 8h. browser_screenshot — JPEG of the session's viewport
        # ------------------------------------------------------------------
        {
            "name": "browser_screenshot",
            "description": (
                "Capture a JPEG screenshot of the session's current viewport. "
                "Cheaper than browser_action(kind=dom_snapshot) when you just "
                "want vision. Returns base64-encoded JPEG."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "quality":    {"type": "integer", "description": "JPEG quality 1-100 (default 80)."},
                },
                "required": ["session_id"],
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
        # 11. Analyze image — vision on URL or workspace file
        # ------------------------------------------------------------------
        {
            "name": "analyze_image",
            "description": (
                "Analyze an image (describe it, extract text/OCR, answer questions about it). "
                "Accepts an image URL or a workspace file path. "
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
        # 11b. Generate image — text-to-image
        # ------------------------------------------------------------------
        {
            "name": "generate_image",
            "description": (
                "Generate a brand-new image from a text prompt. "
                "Use this whenever the user asks you "
                "to create, draw, design, paint, illustrate, or make a picture, "
                "logo, icon, artwork, poster, or diagram. The image is delivered "
                "to the user automatically as an inline attachment and also saved "
                "to the workspace (so you can send_photo or reference it later). "
                "Do NOT use this to describe/read an existing image — that's "
                "analyze_image. Higher quality and larger sizes cost more credits."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Detailed description of the image to generate. Be "
                            "specific about subject, style, composition, colors, "
                            "lighting, and mood for best results. When the user "
                            "wants a REALISTIC photo (a person, product, or scene "
                            "they might post to social media), explicitly ask for "
                            "a photorealistic result with natural skin texture, "
                            "real lighting, and authentic detail, and say to avoid "
                            "an over-smoothed, plastic, or obviously AI-generated "
                            "look — otherwise the result skews artificial."
                        ),
                    },
                    "size": {
                        "type": "string",
                        "enum": ["1024x1024", "1024x1536", "1536x1024"],
                        "description": "Dimensions: square (1024x1024, default), portrait (1024x1536), or landscape (1536x1024).",
                    },
                    "quality": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Rendering quality. Higher = better but pricier. Default: high.",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Optional output filename ending in .png (e.g. 'logo.png').",
                    },
                },
                "required": ["prompt"],
            },
        },
        # ------------------------------------------------------------------
        {
            "name": "edit_image",
            "description": (
                "Edit / modify an EXISTING image from a text instruction. "
                "Use this whenever the user attaches a photo (or "
                "references one they sent) and asks you to change, modify, edit, "
                "add, remove, replace, recolor, restyle, retouch, or transform it "
                "— e.g. 'make the sky purple', 'remove the person in the back', "
                "'add sunglasses', 'turn this into a watercolor'. By default it "
                "edits the image the user most recently uploaded in the "
                "conversation; pass `image` to target a specific workspace file or "
                "URL. If the user references a photo they already sent (even a few "
                "turns ago), just CALL this tool — it automatically finds their "
                "most recently uploaded image. Do NOT ask them to re-upload first; "
                "only ask if this returns an error saying no image is on record. "
                "The result is delivered inline and saved to the workspace. "
                "Do NOT use this to create a picture from scratch (that's "
                "generate_image) or to merely describe one (that's analyze_image). "
                "Higher quality and larger sizes cost more credits."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "What to change about the image. Describe the edit "
                            "clearly, e.g. 'replace the background with a beach at "
                            "sunset' or 'make the car red instead of blue'."
                        ),
                    },
                    "image": {
                        "type": "string",
                        "description": (
                            "Optional. A workspace file path or https image URL to "
                            "edit. Omit to edit the user's most recently uploaded "
                            "image in the conversation."
                        ),
                    },
                    "size": {
                        "type": "string",
                        "enum": ["1024x1024", "1024x1536", "1536x1024"],
                        "description": "Output dimensions: square (1024x1024, default), portrait, or landscape.",
                    },
                    "quality": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Rendering quality. Higher = better but pricier. Default: high.",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Optional output filename ending in .png (e.g. 'edited.png').",
                    },
                },
                "required": ["prompt"],
            },
        },
        # ------------------------------------------------------------------
        # Spawn — background sub-agent task
        # ------------------------------------------------------------------
        {
            "name": "spawn",
            "description": (
                "Spawn a background sub-agent to do an independent task. "
                "**This is non-blocking** — the call returns immediately with "
                "a job_id; the sub-agent runs separately and announces its "
                "result as a new assistant message in the conversation when "
                "done. **After calling spawn, END YOUR TURN.** Do not wait or "
                "poll. Do not chain more tool calls expecting the sub-agent's "
                "result. Tell the user briefly what you've spawned, then stop. "
                "Use for: quick, bounded sub-tasks (a few minutes at most) "
                "that support THIS conversation — a lookup, a comparison, a "
                "draft the user is waiting on right now. Do NOT use spawn "
                "when the user wants work to continue after they leave, says "
                "things like 'while I'm away/asleep', 'keep working on it', "
                "'in the background and keep me updated', or the work could "
                "outlast this chat — call start_mission for those instead. "
                "The sub-agent has tools but cannot itself spawn further "
                "sub-agents (no grandchildren) and cannot write to your "
                "user's memory."
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
                    "credit_budget_usd": {
                        "type": "number",
                        "description": (
                            "Optional USD spend cap for the sub-agent run. "
                            "If the run exceeds this, it terminates with "
                            "outcome=budget_exhausted and a partial-result "
                            "announce. Use for expensive research tasks "
                            "where you want to bound cost. Leave unset for "
                            "no cap."
                        ),
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
        # 17b. recall_day — recall a past day's conversation across channels
        # ------------------------------------------------------------------
        {
            "name": "recall_day",
            "description": (
                "Recall a past day's conversation across ALL channels (web, telegram, voice, app, vibecoding). "
                "Returns the day's archival summary by default. Set include_full_conversation=true to get the "
                "raw messages when the summary is not detailed enough for the task. "
                "Pass an optional `query` to filter within the day when it mixes unrelated topics "
                "(e.g. query='calculus course' when the user asks to build a quiz from yesterday's lesson "
                "and the day also has unrelated chatter). "
                "Accepts natural language dates: 'yesterday', 'last Monday', '3 days ago', 'April 10', "
                "or ISO 'YYYY-MM-DD'. Weekday names always resolve to the most recent PAST occurrence. "
                "Use this whenever the user references a previous day — never say you can't remember."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date reference: 'yesterday', 'last Tuesday', '3 days ago', '2026-04-15', 'April 10', etc.",
                    },
                    "include_full_conversation": {
                        "type": "boolean",
                        "description": "If true, return raw messages annotated with channel + time. Use when you need specific content — e.g. to make a quiz from a lesson. Default: false (summary only).",
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional topic filter. When the day mixes unrelated topics, pass this to narrow the returned messages (keyword match with ±2-message context window).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max messages to return when include_full_conversation=true. Default 200.",
                    },
                },
                "required": ["date"],
            },
        },
        # ------------------------------------------------------------------
        # 18. play_media — play music/video for the user
        # ------------------------------------------------------------------
        {
            "name": "play_media",
            "description": (
                "Play a song or video for the user. Searches YouTube and streams it "
                "directly in the user's browser — no navigation needed. "
                "Pass the song/artist name or movie title as the query. "
                "For Netflix content, pass channel='netflix' and the agent will use "
                "the user's connected Netflix account via the browser agent."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Song name, artist, or video title to search for (e.g. 'Adele Hello', 'Dark Knight trailer').",
                    },
                    "channel": {
                        "type": "string",
                        "enum": ["youtube", "netflix"],
                        "description": "Streaming channel to use. Default: youtube.",
                    },
                },
                "required": ["query"],
            },
        },
        # ------------------------------------------------------------------
        # 19. Browser — headless browser automation
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
                "Run system health checks and diagnostics across dependencies, "
                "config, database, disk, runtime, provider keys, workspace, "
                "memory, messaging, and browser. Omit 'checks' to run everything."
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
        # ------------------------------------------------------------------
        # Autopilot — hand off a goal to the autonomous mission engine
        # ------------------------------------------------------------------
        {
            "name": "start_mission",
            "description": (
                "Hand a goal to Autopilot: an autonomous background engine that "
                "keeps working on it in ticks AFTER this conversation ends — even "
                "while the user is offline or asleep — and notifies them on their "
                "phone when it finishes or needs a decision. Use when the user asks "
                "you to handle something over time ('book me a dentist appointment', "
                "'keep researching X and have a draft by morning', 'do this while I "
                "sleep'). ALSO use start_mission — not spawn — whenever the user "
                "asks for background work they will walk away from, wants live "
                "progress or notifications on their phone, or the task could take "
                "more than a few minutes: missions drive the lock-screen progress "
                "card; spawn is only for quick sub-tasks inside this conversation. "
                "Do NOT use for things you can finish in this turn. The "
                "mission runs unsupervised with a credit budget; anything requiring "
                "an outward action (sending email, purchases) will pause and ask the "
                "user for approval. Tell the user the mission was created and that "
                "they can watch or stop it in Mission Control. The mission itself "
                "appears in Mission Control as the tracked task — do NOT also call "
                "create_job for the same ask."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": (
                            "The full goal, self-contained — the mission won't see "
                            "this conversation. Include every constraint the user "
                            "gave (deadline, preferences, links)."
                        ),
                    },
                    "name": {
                        "type": "string",
                        "description": "Short display name for Mission Control (e.g. 'Dentist appointment').",
                    },
                    "budget_credits": {
                        "type": "number",
                        "description": "Optional credit budget ceiling (default 100 \u2248 $1).",
                    },
                    "urgent": {
                        "type": "boolean",
                        "description": "True only if the user explicitly wants to be woken up \u2014 urgent notifications bypass quiet hours.",
                    },
                },
                "required": ["goal"],
            },
        },
        # ------------------------------------------------------------------
        # Job management — create/update jobs visible in the dashboard
        # ------------------------------------------------------------------
        {
            "name": "create_job",
            "description": (
                "Create a trackable job for multi-step work you are doing NOW, in this "
                "conversation. It appears live in the user's dashboard, sidebar, and on "
                "their phone's lock screen / Dynamic Island. Declare steps up front and "
                "call update_job as you complete each one. CONTRACT — never end your "
                "reply while a job you created is still 'running': before finishing, "
                "either (a) complete the work and mark it 'completed', (b) mark it "
                "'failed' with an error_message, or (c) hand the remaining work to "
                "spawn or start_mission so something actually continues it. A job left "
                "'running' when your turn ends freezes every progress surface and is "
                "auto-failed as stalled after 30 minutes. Never create a job for work "
                "you are handing to start_mission — the mission appears as its own "
                "tracked task in Mission Control."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short title for the job (e.g. 'Research AI papers', 'Fix login bug').",
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description of what needs to be done.",
                    },
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of step labels (e.g. ['Research', 'Draft outline', 'Write report']). Optional.",
                    },
                },
                "required": ["title"],
            },
        },
        {
            "name": "update_job",
            "description": (
                "Update the status of an existing job. Use this to mark steps as done, "
                "update the overall job status, or add an error message. "
                "Call this as you complete each step so the user sees live progress."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job ID returned by create_job.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["running", "completed", "failed"],
                        "description": "New overall job status.",
                    },
                    "current_step": {
                        "type": "integer",
                        "description": "Index (0-based) of the step to mark as done. All steps before this index will also be marked done.",
                    },
                    "error_message": {
                        "type": "string",
                        "description": "Error message if status is 'failed'.",
                    },
                },
                "required": ["job_id"],
            },
        },
        # ------------------------------------------------------------------
        # save_streaming_credential — Vault CP4 chat-save
        # ------------------------------------------------------------------
        {
            "name": "save_streaming_credential",
            "description": (
                "Send the user a confirmation card to save a streaming service login "
                "(Netflix, Prime Video, Disney+, etc.) into their encrypted Vault. "
                "The user will fill in email + password through a secure card — do NOT "
                "ask for the password in chat, and NEVER include a password in this tool's input. "
                "NEVER call on Telegram or voice channels — the tool will reject and the user "
                "should be told to use the web or mobile app. "
                "After calling this tool, wait for the user to confirm via the card before "
                "taking further action."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "enum": [
                            "netflix",
                            "prime_video",
                            "disney_plus",
                            "apple_tv",
                            "hbo_max",
                            "hulu",
                            "paramount_plus",
                            "peacock",
                            "crave",
                        ],
                        "description": "Streaming service slug.",
                    },
                    "email_hint": {
                        "type": "string",
                        "description": (
                            "OPTIONAL. If the user already mentioned an email, pass it as a hint "
                            "so the card pre-fills the field. Leave absent otherwise. "
                            "NEVER include a password."
                        ),
                    },
                },
                "required": ["channel"],
            },
        },
    ]


def get_doc_generation_tools() -> List[Dict[str, Any]]:
    """Document-generation tools (PDF/DOCX/XLSX/PPTX/MD/HTML→PDF).

    Gated by settings.feature_doc_generation. Each tool writes a file to
    the per-user storage backend and returns a summary string; the
    actual attachment metadata is emitted to the client over the WS
    `attachment` event after the tool completes.

    The LLM should pick these over inline markdown when the user asks
    for an export, report, invoice, spreadsheet, slide deck, etc.
    """
    _content_block = {
        "type": "array",
        "description": (
            "Structured content blocks. Each block has a `type`: "
            "'heading' (+level 1-4, +text), 'paragraph' (+text), "
            "'table' (+headers, +rows), 'image' (+path, +caption?), "
            "'bullet_list' (+items), 'numbered_list' (+items), 'page_break'."
        ),
        "items": {"type": "object"},
    }
    return [
        {
            "name": "generate_pdf",
            "description": (
                "Produce a PDF from structured content blocks and deliver it to the user. "
                "Prefer this over inline markdown when the user asks for a report, export, "
                "invoice, or any document they'd want to download or share."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": _content_block,
                    "filename": {"type": "string", "description": "Output filename (e.g. 'report.pdf')."},
                    "title": {"type": "string", "description": "Optional document title, used for the cover page and PDF metadata."},
                    "cover_page": {"type": "boolean", "description": "If true, add a cover page with title + today's date."},
                },
                "required": ["content", "filename"],
            },
        },
        {
            "name": "generate_docx",
            "description": (
                "Produce a Word (.docx) document from structured content blocks. "
                "Use when the user wants an editable document (vs. a read-only PDF)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": _content_block,
                    "filename": {"type": "string"},
                    "title": {"type": "string", "description": "Optional H0 title at the top."},
                },
                "required": ["content", "filename"],
            },
        },
        {
            "name": "generate_xlsx",
            "description": (
                "Produce an Excel (.xlsx) workbook with one or more sheets. "
                "Use for tabular data — expense summaries, datasets, schedules."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "sheets": {
                        "type": "array",
                        "description": "List of sheets, each {name, headers, rows}. `rows` is a list of lists.",
                        "items": {"type": "object"},
                    },
                    "filename": {"type": "string"},
                },
                "required": ["sheets", "filename"],
            },
        },
        {
            "name": "generate_pptx",
            "description": (
                "Produce a PowerPoint (.pptx) deck. Use for presentations, briefings, slide summaries."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "slides": {
                        "type": "array",
                        "description": (
                            "List of slides. Each slide has `type`: 'title' (+title, +subtitle), "
                            "'content' (+title, +bullets[]), 'image' (+title, +path), 'section' (+title)."
                        ),
                        "items": {"type": "object"},
                    },
                    "filename": {"type": "string"},
                },
                "required": ["slides", "filename"],
            },
        },
        {
            "name": "generate_markdown",
            "description": (
                "Save a Markdown document. Use when the user wants plain-text output "
                "they can import elsewhere (notes, docs sites, etc.)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Markdown source."},
                    "filename": {"type": "string"},
                },
                "required": ["content", "filename"],
            },
        },
        {
            "name": "convert_document",
            "description": (
                "Convert an EXISTING generated DOCX or PPTX file to PDF via "
                "LibreOffice — faithful WYSIWYG conversion, NOT a new document "
                "from scratch. Use this when the user says \"make it a PDF\", "
                "\"give me the PDF version\", \"convert to PDF\", or similar. "
                "Produces a PDF that looks identical to what Word / PowerPoint "
                "would print. Do NOT call generate_pdf for this — that builds "
                "a new PDF from structured blocks, which loses the original "
                "layout."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "source_filename": {
                        "type": "string",
                        "description": "Filename of the previously-generated file (e.g. 'hiring-application-form.docx').",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Output filename (optional; defaults to source basename + .pdf).",
                    },
                },
                "required": ["source_filename"],
            },
        },
        {
            "name": "generate_html_to_pdf",
            "description": (
                "Render an HTML string to PDF via weasyprint. Use when you need CSS-styled PDF "
                "output (custom fonts, complex layouts). Only available on agent-side runtimes "
                "(requires Pango/Cairo); falls back to an error on platform-side. Prefer "
                "generate_pdf with structured blocks when possible."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "html": {"type": "string", "description": "Full HTML document or fragment."},
                    "filename": {"type": "string"},
                },
                "required": ["html", "filename"],
            },
        },
        # Navigate the user's browser to a different platform page. The handler
        # broadcasts a {"type":"navigate", "path":...} frame to the user's chat
        # WebSocket; the frontend listens for it and routes via React Router
        # without losing chat session continuity. Use ONLY when the user
        # explicitly asks to be taken somewhere ("take me to settings", "open
        # my brain"). For passive suggestions (e.g. "your portrait is on the
        # brain page — want to see?"), emit a [[navigate:/path]] chip in the
        # message text instead — the user clicks if interested.
        {
            "name": "navigate_to",
            "description": (
                "Transfer the user to a different page in the Toup platform. "
                "Use when the user EXPLICITLY asks to go somewhere ('take me to "
                "X', 'open Y', 'go to Z'). For passive suggestions, prefer a "
                "[[navigate:/path]] chip in your message instead so the user "
                "can choose. Available paths:\n"
                "- / — Hub (home)\n"
                "- /chat — Chat (this page)\n"
                "- /brain — Brain (their stored memories about themselves)\n"
                "- /browser — Live Browser (watch your headless browser in real time)\n"
                "- /workspace — Workspace (apps you've built for them)\n"
                "- /jobs — Jobs (long-running tasks, status + logs)\n"
                "- /dashboard — Dashboard (metrics, inbox, daily summary)\n"
                "- /agent — Agent home (soul, channels, LLM keys)\n"
                "- /agent/soul — Soul (your personality config)\n"
                "- /agent/settings — Channels & Settings (WhatsApp, Telegram, voice wiring)\n"
                "- /agent/tools — Tools catalog\n"
                "- /agent/skills — Skills catalog\n"
                "- /account — Account (profile, password, billing)\n"
                "- /movies — Movies (Netflix integration)"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The URL path to navigate to.",
                        "enum": [
                            "/", "/chat", "/brain", "/brain/user", "/browser",
                            "/workspace", "/jobs", "/dashboard",
                            "/agent", "/agent/soul", "/agent/settings",
                            "/agent/tools", "/agent/skills",
                            "/account", "/movies",
                        ],
                    },
                },
                "required": ["path"],
            },
        },
    ]