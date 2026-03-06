# HexBrain — Setup Guide

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Docker | 20.10+ | Docker Desktop recommended |
| Docker Compose | v2+ | Included with Docker Desktop |
| OpenAI API key | — | Required for embeddings + agent LLM |

## Quick Start (5 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/toup-com/hex-brain.git
cd hex-brain

# 2. Create your .env file
cp .env.example .env

# 3. Add your OpenAI API key
#    Open .env and set: OPENAI_API_KEY=sk-...

# 4. Start all services
docker compose up -d

# 5. Access the app
#    Frontend: http://localhost
#    API:      http://localhost:8000
#    Health:   http://localhost:8000/health
```

## Default Login

| Field | Value |
|-------|-------|
| Username | `hex` |
| Password | `Nariman123!` |

## Configuration

All settings are controlled via environment variables in `.env`. See [`.env.example`](../.env.example) for the full list.

### Required

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings and LLM |

### Telegram Bot (Optional)

| Variable | Description |
|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather |
| `TELEGRAM_ALLOWED_USER_IDS` | JSON array of allowed Telegram user IDs (empty = all) |
| `TELEGRAM_USER_MAP` | JSON map of Telegram user ID → HexBrain user ID |

### Agent LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MODEL` | `gpt-5.2` | Primary agent model |
| `AGENT_FALLBACK_MODEL` | `gpt-4o` | Fallback if primary fails |

### Web Search (Optional)

| Variable | Description |
|----------|-------------|
| `BRAVE_API_KEY` | Brave Search API key (falls back to DuckDuckGo) |

### Database

The database is auto-configured in Docker. For custom setups:

| Variable | Default |
|----------|---------|
| `DATABASE_URL` | `postgresql+asyncpg://hexbrain:hexbrain_secret@db:5432/hexbrain` |

### Scheduler

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_SCHEDULER` | `true` | Memory decay/consolidation scheduler |
| `DECAY_INTERVAL_HOURS` | `6` | How often to run Ebbinghaus decay |
| `CONSOLIDATION_CRON_HOUR` | `3` | Hour (UTC) for daily memory consolidation |

## Services

When running via Docker Compose, three containers start:

| Service | Container | Port | Description |
|---------|-----------|------|-------------|
| `db` | hexbrain-db | 5432 | PostgreSQL 16 + pgvector |
| `backend` | hexbrain-backend | 8000 | FastAPI Python backend |
| `frontend` | hexbrain-frontend | 80 | React + nginx |

## Volumes

| Volume | Mount | Purpose |
|--------|-------|---------|
| `postgres_data` | `/var/lib/postgresql/data` | Database persistence |
| `agent_workspace` | `/app/workspace` | Agent file operations |
| `agent_skills` | `/app/skills` | External skill plugins |

## Verifying the Install

```bash
# Check all containers are running
docker compose ps

# Check backend health
curl http://localhost:8000/health

# Check backend logs
docker compose logs backend --tail 20

# You should see:
# ✅ Database initialized
# 🧩 Loaded 1 skill(s): ['toup']
# 🤖 HexBrain Telegram bot started (polling mode)
# Uvicorn running on http://0.0.0.0:8000
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker compose logs backend

# Rebuild from scratch
docker compose down -v
docker compose build --no-cache
docker compose up -d
```

### Database connection issues
```bash
# Check if DB is healthy
docker compose exec db pg_isready -U hexbrain

# Connect directly
docker compose exec db psql -U hexbrain -d hexbrain
```

### Port conflicts
If ports 80, 5432, or 8000 are in use, update `docker-compose.yml`:
```yaml
ports:
  - "3001:80"    # frontend on 3001 instead of 80
  - "8001:8000"  # backend on 8001 instead of 8000
```

## Updating

```bash
git pull
docker compose build --no-cache
docker compose up -d
```
