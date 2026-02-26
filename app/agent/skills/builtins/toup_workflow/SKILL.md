---
name: Toup Workflow Skills
description: Expert guidance for building HexBrain agentic workflows. Use when creating workflows, configuring nodes, connecting edges, validating, or troubleshooting workflow errors on the /work page.
version: 1.0.0
---

# Toup Workflow Skills

Expert knowledge for building production-ready HexBrain workflows using the hexbrain-workflows MCP server.

---

## Quick Reference

### The 6 Node Categories (25 total)

| Category | Nodes | Purpose |
|----------|-------|---------|
| **trigger** (5) | manual, schedule, webhook, telegram, event | Start the workflow |
| **ai** (5) | agent, chat, classify, extract, embedding | AI-powered processing |
| **action** (5) | memory_search, memory_store, http, exec, send_telegram | External actions |
| **logic** (5) | if, switch, merge, loop, wait | Control flow |
| **data** (3) | transform, set, filter | Data manipulation |
| **output** (2) | respond, log | Final outputs |

### Workflow Building Sequence

```
1. list_node_templates()          → See what's available
2. create_workflow(name, desc)    → Empty canvas
3. add_node(type, config)         → Add nodes one by one
4. connect_nodes(source, target)  → Wire them up
5. validate_workflow()            → Check for issues
6. set_workflow_status('active')  → Go live
```

---

## Core Concepts

### 1. Every Workflow Starts with a Trigger

A workflow MUST have exactly one trigger node. No trigger = validation error.

| Trigger | Use When |
|---------|----------|
| `trigger_manual` | Testing, one-off runs |
| `trigger_schedule` | Cron-based recurring tasks |
| `trigger_webhook` | External API calls, webhooks |
| `trigger_telegram` | Chat bot messages |
| `trigger_event` | React to memory changes |

### 2. Nodes Have Typed Handles

Every node has **inputs** and **outputs** with specific handle IDs:

- Standard: `in` → `out`
- IF node: `in` → `true` / `false`
- Switch: `in` → `case1` / `case2` / `default`
- Loop: `in` → `item` (each iteration) / `done` (after loop)
- Merge: `in1` + `in2` → `out`
- Output nodes: `in` → (no output)
- Trigger nodes: (no input) → `out`

### 3. Config Fields Are Template-Specific

Each node type has specific config fields. Always use `get_node_template_details(type)` to see available fields before configuring. Don't guess field names.

### 4. Workflow Status Lifecycle

```
draft → active → paused
         ↑        ↓
         └────────┘
```

- `draft` — Default. Can edit freely.
- `active` — Running. Validated before activation.
- `paused` — Temporarily stopped.

---

## Common Patterns

### Pattern 1: Chatbot (Telegram → AI → Reply)

```
trigger_telegram → ai_agent → output_respond
```

Best for: Interactive bots, Q&A systems, customer support.

Config tips:
- `ai_agent.tools_enabled = true` for function calling
- `ai_agent.system_prompt` defines personality
- `ai_agent.model` defaults to gpt-5.2

### Pattern 2: Webhook Pipeline (Webhook → Process → Store)

```
trigger_webhook → ai_extract → action_memory_store → output_log
```

Best for: Ingesting data from external services, API integrations.

Config tips:
- `trigger_webhook.method = "POST"` for receiving data
- `ai_extract.schema` must be valid JSON schema

### Pattern 3: Smart Router (Input → Classify → Switch → Handlers)

```
trigger_webhook → ai_classify → logic_switch ─→ [case1] → ai_agent
                                               ├→ [case2] → action_send_telegram
                                               └→ [default] → action_memory_store
```

Best for: Routing different types of input to specialized handlers.

### Pattern 4: Scheduled Task (Cron → Search → AI → Notify)

```
trigger_schedule → action_memory_search → ai_chat → action_send_telegram
```

Best for: Daily digests, periodic reports, memory consolidation.

Config tips:
- `trigger_schedule.cron` uses standard cron format: `"0 9 * * *"` = daily at 9am
- `trigger_schedule.timezone` defaults to UTC

### Pattern 5: Loop Processing (Trigger → Loop → Process → Done)

```
trigger_webhook → logic_loop ─→ [item] → ai_chat → action_memory_store
                              └→ [done] → output_respond
```

Best for: Batch processing arrays of items.

---

## Node Configuration Guide

### AI Nodes — Model Selection

| Model | Best For | Speed | Cost |
|-------|----------|-------|------|
| `gpt-5.2` | Complex reasoning, tool use | Slower | Higher |
| `gpt-4o-mini` | Simple tasks, classification | Fast | Low |
| `text-embedding-3-small` | Embeddings (default) | Fast | Low |
| `text-embedding-3-large` | Higher quality embeddings | Medium | Medium |

### Logic Nodes — Handle Routing

**IF node** — Use `source_handle`:
- `"true"` → condition is met
- `"false"` → condition is not met

**Switch node** — Use `source_handle`:
- `"case1"` → matches case1_value
- `"case2"` → matches case2_value
- `"default"` → no match

**Loop node** — Use `source_handle`:
- `"item"` → for each iteration
- `"done"` → after all items processed

**Merge node** — Use `target_handle`:
- `"in1"` → first input
- `"in2"` → second input

### Canvas Layout Best Practices

Position nodes left-to-right:
- Triggers: `x=100`
- First processing: `x=400`
- Second processing: `x=700`
- Output: `x=1000`
- Vertical spacing: `y` increments of `200`

---

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| No trigger node | Add a trigger as the first node |
| Using wrong `source_handle` | Check node template: IF uses `true`/`false`, not `out` |
| Forgetting `connect_nodes()` | Every node except trigger needs an incoming edge |
| Wrong `node_type` name | Use `list_node_templates()` to see exact type IDs |
| Setting status to `active` without validation | Always `validate_workflow()` first |
| Using `out` handle on IF node | IF has `true` and `false`, not `out` |
| Merge node with only one input | Merge needs connections to both `in1` AND `in2` |
| Config field name guessing | Use `get_node_template_details(type)` first |

---

## Related Files

- [NODE_REFERENCE.md](NODE_REFERENCE.md) — Complete node catalog with all fields
- [WORKFLOW_PATTERNS.md](WORKFLOW_PATTERNS.md) — 5 detailed patterns with examples
- [VALIDATION_GUIDE.md](VALIDATION_GUIDE.md) — Validation rules and error fixing
- [MCP_TOOLS_GUIDE.md](MCP_TOOLS_GUIDE.md) — How to use each MCP tool effectively
- [COMMON_MISTAKES.md](COMMON_MISTAKES.md) — Error catalog with solutions
