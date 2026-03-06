# HexBrain — System Architecture

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           HexBrain Platform                             │
├─────────────┬───────────────────┬──────────────────┬────────────────────┤
│  Telegram   │   Web Chat (WS)   │  REST API v1     │  Frontend (React)  │
│  Bot        │   /ws/chat         │  /api/v1/*       │  Brain Viz + Chat  │
├─────────────┴───────────────────┴──────────────────┴────────────────────┤
│                           FastAPI Backend                                │
│  ┌────────────┐  ┌────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │ AgentRunner│  │ Tool       │  │ Skill Loader  │  │ Memory        │   │
│  │ (LLM Loop) │  │ Executor   │  │ (Plugin Mgr)  │  │ Service       │   │
│  └─────┬──────┘  └─────┬──────┘  └──────┬────────┘  └──────┬────────┘   │
│        │               │               │                   │            │
│        └───────────────┴───────────────┴───────────────────┘            │
│                              │                                          │
│  ┌───────────────────────────┴────────────────────────────────┐         │
│  │                    PostgreSQL + pgvector                    │         │
│  │  users, conversations, messages, memories, api_keys, ...   │         │
│  └────────────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Diagram

### 1. Entry Points (Multi-Channel)

| Channel | Protocol | Auth | Handler |
|---------|----------|------|---------|
| Telegram | Polling (long-poll) | `TELEGRAM_ALLOWED_USER_IDS` | `telegram_bot.py` |
| Web Chat | WebSocket | JWT (query param or first message) | `ws_chat.py` |
| REST API v1 | HTTP | API Key (`hx_...`) | `api_v1.py` |
| Frontend API | HTTP | JWT | `api/*.py` routers |

All channels route through the same `AgentRunner` → same memory, same tools, same skills.

### 2. Agent Runtime

```
User Message
    │
    ▼
┌──────────────────────────────────────────────────┐
│  AgentRunner.run()                                │
│                                                    │
│  1. Build system prompt                           │
│     ├─ Identity + rules                            │
│     ├─ Memory recall (semantic search)             │
│     ├─ Runtime context (time, tools, etc.)         │
│     └─ Skill prompt sections                       │
│                                                    │
│  2. Context window management                     │
│     └─ Auto-compact if > threshold                 │
│                                                    │
│  3. LLM Call (streaming)                          │
│     ├─ Model: gpt-5.2 (fallback: gpt-4o)         │
│     └─ Tools: 10 built-in + skill tools            │
│                                                    │
│  4. Tool execution loop                           │
│     ├─ Built-in tool → ToolExecutor               │
│     ├─ Skill tool → SkillLoader → Skill.execute   │
│     └─ Feed results back → LLM → repeat           │
│                                                    │
│  5. Post-processing                               │
│     ├─ Save messages to DB                         │
│     ├─ Extract & evolve memories                   │
│     └─ Return response                             │
└──────────────────────────────────────────────────┘
```

### 3. Tool System

**Built-in Tools (10):**

| Tool | Description |
|------|-------------|
| `exec` | Run shell commands in workspace |
| `read_file` | Read file contents |
| `write_file` | Write/create files |
| `edit_file` | Find & replace in files |
| `memory_search` | Semantic memory search |
| `memory_store` | Store new memories |
| `web_search` | Brave/DuckDuckGo search |
| `web_fetch` | Fetch web page content |
| `send_file` | Send document to user |
| `send_photo` | Send image to user |

**Skill Tools:** Dynamically registered by loaded skills (e.g., `toup__create_spec`).

**Additional Tools:**
| Tool | Description |
|------|-------------|
| `cron` | Schedule recurring tasks |
| `spawn` | Spawn background sub-agents |
| `sessions_list` | List conversation sessions |
| `sessions_history` | Read session message history |

### 4. Memory Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Evolution Pipeline                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  New Memory → Embed (text-embedding-3-small)                │
│      │                                                       │
│      ▼                                                       │
│  Search Similar (pgvector cosine similarity)                │
│      │                                                       │
│      ├─ < 35% similarity → CREATE new memory                │
│      ├─ 35-45% similarity → LINK (create relationship)      │
│      ├─ 45-70% similarity → MERGE (LLM combines content)   │
│      └─ > 70% similarity → REINFORCE (strengthen existing)  │
│      │                                                       │
│      ▼                                                       │
│  Record MemoryEvent (immutable audit log)                   │
│  Re-embed if content changed                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Three Brains:**

| Brain | Categories | Purpose |
|-------|-----------|---------|
| User | 21 categories | Personal memories, preferences, knowledge |
| Agent | 6 categories | Hex's learned patterns and capabilities |
| Work | 2 categories | Workflows and processes |

**Memory Levels (Cognitive Hierarchy):**

| Level | Description | Decay Rate |
|-------|-------------|------------|
| Episodic | Specific events | Fast decay |
| Semantic | General knowledge | Slow decay |
| Procedural | How-to knowledge | Very slow |
| Strategic | High-level patterns | Minimal decay |

### 5. Skills / Plugin System

```
SkillLoader
    │
    ├─ Scan builtins/   (backend/app/agent/skills/builtins/)
    ├─ Scan external/   (/app/skills/ or SKILLS_DIR)
    │
    ▼
For each skill directory with skill.py:
    1. Import module
    2. Find Skill subclass
    3. Validate tool name prefixes (<skill>__)
    4. Call on_load()
    5. Register tools in tool index
    6. Inject system prompt sections
```

### 6. Scheduled Tasks

| Task | Schedule | Purpose |
|------|----------|---------|
| Memory Decay | Every 6h | Ebbinghaus forgetting curve |
| Memory Consolidation | Daily 3 AM | Episodic → Semantic promotion |
| Memory Health Check | Every 60 min | Integrity validation |
| Cron Jobs | User-defined | Custom scheduled tasks |

### 7. Database Schema (Key Tables)

```
users
├── conversations (1:N)
│   └── messages (1:N)
├── memories (1:N)
│   ├── memory_events (1:N)  — audit log
│   └── memory_relationships (N:N)  — links between memories
├── api_keys (1:N)
├── telegram_user_mappings (1:N)
├── identities (1:N)
├── documents (1:N)
│   └── document_chunks (1:N)
├── cron_jobs (1:N)
└── agent_errors (1:N)
```

## Tech Stack

| Layer | Technology | Version |
|-------|------------|---------|
| Frontend | React + TypeScript | 18 |
| 3D Viz | Three.js + React Three Fiber | 0.160 |
| State | Zustand | 4.4 |
| Styling | TailwindCSS | 3.4 |
| Animation | Framer Motion | 10.18 |
| Backend | Python + FastAPI | 3.12 / 0.109+ |
| ORM | SQLAlchemy (async) | 2.0+ |
| Database | PostgreSQL + pgvector | 16 |
| LLM | OpenAI GPT-5.2 | — |
| Embeddings | text-embedding-3-small | 1536 dims |
| Telegram | python-telegram-bot | 21.0.1 |
| Containers | Docker Compose | v2 |
| Proxy | nginx | — |

## Request Flow Example

```
User sends "I love pizza" via Telegram
    │
    ▼
TelegramBot._handle_message()
    ├─ Download media (if any)
    ├─ Transcribe voice (Whisper)
    ├─ Get HexBrain user ID (TelegramUserMapping)
    │
    ▼
AgentRunner.run(message, user_id, session_id)
    ├─ Load conversation history (20 msgs)
    ├─ Search memories ("I love pizza" → embedding → pgvector)
    ├─ Build system prompt (identity + memories + tools + skills)
    ├─ Call GPT-5.2 (streaming)
    │   ├─ LLM uses memory_store tool → "User loves pizza"
    │   │   └─ MemoryService → deduplicate → merge/create → event log
    │   └─ LLM returns final text
    ├─ Save messages to DB
    ├─ Extract additional memories (LLM-based extraction)
    └─ Return AgentResponse
    │
    ▼
TelegramBot streams response back
    ├─ Progressive message edits (every 300ms)
    ├─ ACK reaction (👀) while processing
    └─ Final message with full response
```
