# Toup — API Reference

## Authentication

Toup supports two authentication methods:

### JWT Tokens (Web/Frontend)

Used by the web UI and WebSocket chat.

```bash
# Login
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "hex", "password": "Nariman123!"}'

# Response: { "access_token": "eyJ...", "token_type": "bearer" }
```

Include the token in subsequent requests:
```
Authorization: Bearer eyJ...
```

### API Keys (Public API v1)

Used for programmatic access. Keys are prefixed with `hx_` and stored as SHA-256 hashes.

```bash
# Create an API key (requires JWT auth)
curl -X POST http://localhost:8000/api/v1/keys \
  -H "Authorization: Bearer eyJ..." \
  -H "Content-Type: application/json" \
  -d '{"name": "My CI Key", "rate_limit": 60}'

# Response: { "key": "hx_abc123...", "id": "...", "name": "My CI Key", "key_prefix": "hx_abc123" }
# ⚠️ The raw key is only shown ONCE — save it immediately.
```

Use the API key:
```
Authorization: Bearer hx_abc123...
```

---

## Public API v1

All v1 endpoints require API key authentication.

Base URL: `http://localhost:8000/api/v1`

### Chat

#### `POST /v1/chat`

Send a message and get a complete response.

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer hx_..." \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What do you know about me?",
    "session_id": null,
    "model": null
  }'
```

**Response:**
```json
{
  "text": "Based on my memories, you...",
  "session_id": "abc-123-def",
  "tokens_input": 1500,
  "tokens_output": 300,
  "tokens_total": 1800,
  "model": "gpt-5.2",
  "tool_calls": 1,
  "processing_time_ms": 2400
}
```

#### `POST /v1/chat/stream`

Send a message and receive Server-Sent Events (SSE) for streaming.

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Authorization: Bearer hx_..." \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me a story"}'
```

**Events:**
```
data: {"type": "tool_start", "tool": "memory_search"}
data: {"type": "tool_end", "tool": "memory_search", "summary": "..."}
data: {"type": "done", "text": "...", "session_id": "...", "tokens_input": 0, "tokens_output": 0, "model": "gpt-5.2", "tool_calls": 1, "processing_time_ms": 2000}
```

### Sessions

#### `GET /v1/sessions`

List conversation sessions.

```bash
curl http://localhost:8000/api/v1/sessions?limit=20&active_only=false \
  -H "Authorization: Bearer hx_..."
```

**Response:**
```json
[
  {
    "id": "abc-123",
    "channel": "telegram",
    "is_active": true,
    "message_count": 42,
    "total_tokens": 15000,
    "created_at": "2026-02-01T00:00:00",
    "updated_at": "2026-02-07T12:00:00"
  }
]
```

#### `GET /v1/sessions/{session_id}/messages`

Get messages from a session.

```bash
curl http://localhost:8000/api/v1/sessions/abc-123/messages?limit=50 \
  -H "Authorization: Bearer hx_..."
```

**Response:**
```json
[
  {
    "role": "user",
    "content": "Hello",
    "created_at": "2026-02-07T12:00:00",
    "model_used": null
  },
  {
    "role": "assistant",
    "content": "Hi! How can I help?",
    "created_at": "2026-02-07T12:00:01",
    "model_used": "gpt-5.2"
  }
]
```

### Memories

#### `POST /v1/memories/search`

Search memories by semantic similarity.

```bash
curl -X POST http://localhost:8000/api/v1/memories/search \
  -H "Authorization: Bearer hx_..." \
  -H "Content-Type: application/json" \
  -d '{
    "query": "soccer",
    "brain_type": "user",
    "limit": 10
  }'
```

### Skills

#### `GET /v1/skills`

List all loaded skills and their tools.

```bash
curl http://localhost:8000/api/v1/skills \
  -H "Authorization: Bearer hx_..."
```

**Response:**
```json
{
  "skills": [
    {
      "name": "toup",
      "version": "1.0.0",
      "description": "Software engineering tools: specs, scaffolds, changesets, code review, sprint planning.",
      "author": "Toup",
      "tools": [
        "toup__create_spec",
        "toup__scaffold",
        "toup__changeset",
        "toup__review_diff",
        "toup__plan_sprint"
      ]
    }
  ],
  "count": 1
}
```

### API Key Management

#### `POST /v1/keys`

Create a new API key.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `name` | string | Yes | — | Key name (e.g. "CI Pipeline") |
| `rate_limit` | int | No | 60 | Requests per minute |
| `expires_in_days` | int | No | null | Auto-expire after N days |

#### `GET /v1/keys`

List your API keys (key values are never returned, only prefixes).

#### `DELETE /v1/keys/{key_id}`

Revoke (deactivate) an API key.

---

## WebSocket Chat

Real-time streaming chat via WebSocket.

### Connection

```
ws://localhost:8000/api/ws/chat?token=JWT_TOKEN
```

Or connect without a token and send auth as the first message:
```json
{"type": "auth", "token": "JWT_TOKEN"}
```

### Client → Server Messages

```json
{"type": "message", "text": "Hello!", "session_id": "optional-session-id"}
{"type": "ping"}
```

### Server → Client Messages

```json
{"type": "text_chunk", "text": "partial response..."}
{"type": "tool_start", "tool": "memory_search"}
{"type": "tool_end", "tool": "memory_search", "summary": "Found 3 memories"}
{"type": "done", "text": "full response", "session_id": "...", "tokens": {"input": 100, "output": 50, "total": 150}, "model": "gpt-5.2", "tool_calls": 1, "processing_time_ms": 2000}
{"type": "error", "message": "Something went wrong"}
{"type": "pong"}
```

---

## Internal API (JWT Auth)

These endpoints are used by the frontend and require JWT authentication.

### Memories

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/memories` | Create memory (with smart dedup + evolution) |
| `GET` | `/api/memories/search?query=...&brain=user` | Semantic search |
| `GET` | `/api/memories/{id}` | Get memory with evolution history |
| `GET` | `/api/memories/{id}/timeline` | Get evolution timeline |

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | Send message to agent |
| `POST` | `/api/chat/stream` | Streaming chat (SSE) |

### Auth

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/auth/login` | Login, get JWT token |
| `POST` | `/api/auth/register` | Register new user |
| `GET` | `/api/auth/me` | Get current user |

### Admin

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/admin/status` | System status + stats |
| `GET` | `/api/admin/errors` | Recent errors |
| `GET` | `/api/admin/cron` | Cron job status |

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/documents/upload` | Upload PDF/DOCX/etc |
| `GET` | `/api/documents` | List documents |

---

## Rate Limiting

API key requests are rate-limited with a sliding window (default: 60 requests/minute per key).

When rate-limited, the API returns:
```json
HTTP 429
{"detail": "Rate limit exceeded (60/min)"}
```

## Error Responses

All errors follow this format:
```json
{
  "detail": "Error description"
}
```

| Status | Meaning |
|--------|---------|
| 401 | Missing or invalid authentication |
| 403 | Insufficient permissions |
| 404 | Resource not found |
| 429 | Rate limit exceeded |
| 500 | Internal server error |
| 503 | Agent not available |
