# Toup Agent

Personal AI agent with tools, persistent memory, and multi-channel support.

## Quick Start

Set up your agent from [toup.ai](https://toup.ai) — the setup wizard generates a one-liner install command that configures everything automatically.

## Features

- **Tool execution** — Shell commands, file I/O, web search, web fetch, and 30+ built-in tools
- **Persistent memory** — Vector + keyword + graph hybrid search across conversations
- **Multi-channel** — Telegram, Discord, Slack, WhatsApp, and web chat
- **Auto-routing** — Automatically selects the best model (Sonnet/Opus/GPT) based on task complexity
- **Background service** — Runs as launchd (macOS) or systemd (Linux), auto-restarts on crash
- **Self-update** — Update from the web dashboard or CLI with one command

## CLI

After installation, manage your agent with:

```bash
~/toup-agent/toup status     # Check if agent is running
~/toup-agent/toup update     # Pull latest code & restart
~/toup-agent/toup logs       # Tail agent logs
~/toup-agent/toup stop       # Stop the agent
~/toup-agent/toup start      # Start the agent
~/toup-agent/toup restart    # Restart the agent
```

## Architecture

```
agent_main.py          — FastAPI entry point (uvicorn)
app/
  agent/
    agent_runner.py    — Core orchestration loop (LLM ↔ tools)
    tool_executor.py   — Tool dispatch and execution
    tool_definitions.py— Tool schemas for LLM
    telegram_bot.py    — Telegram channel integration
    cron_service.py    — Scheduled tasks
    skills/            — Pluggable skill system
  api/
    ws_chat.py         — WebSocket streaming chat
    chat.py            — REST chat endpoint
    api_v1.py          — REST API v1
  services/
    openai_agent_service.py   — OpenAI API wrapper
    anthropic_service.py      — Anthropic API wrapper
    model_router.py           — Auto-routing by complexity
    memory_service.py         — Memory CRUD + hybrid search
  db/
    models.py          — SQLAlchemy models
    database.py        — Async engine + session factory
  config.py            — Settings from environment
```

## License

Proprietary — see [toup.ai](https://toup.ai) for terms.
