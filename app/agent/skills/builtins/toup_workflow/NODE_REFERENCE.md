# HexBrain Node Reference — Complete Catalog

All 25 node types with fields, handles, and configuration details.

---

## Triggers (5 nodes)

### trigger_manual
**Start workflow manually**
- Inputs: none
- Outputs: `out`
- Fields: none
- Use: Testing, debugging, one-off runs

### trigger_schedule
**Run on a cron schedule**
- Inputs: none
- Outputs: `out`
- Fields:
  - `cron` (text, **required**) — Cron expression (e.g. `"0 9 * * *"` = daily 9am)
  - `timezone` (text, default: `"UTC"`) — IANA timezone
- Use: Recurring tasks, daily reports, periodic cleanup

**Cron Quick Reference:**
```
┌───────── minute (0-59)
│ ┌─────── hour (0-23)
│ │ ┌───── day of month (1-31)
│ │ │ ┌─── month (1-12)
│ │ │ │ ┌─ day of week (0-7, 0=Sun)
│ │ │ │ │
* * * * *

"0 9 * * *"      → Every day at 9:00 AM
"*/15 * * * *"   → Every 15 minutes
"0 0 * * 1"      → Every Monday at midnight
"0 */6 * * *"    → Every 6 hours
"0 9 * * 1-5"    → Weekdays at 9 AM
```

### trigger_webhook
**Trigger via HTTP webhook**
- Inputs: none
- Outputs: `out`
- Fields:
  - `method` (select: GET/POST/PUT, default: `"POST"`) — HTTP method
  - `path` (text) — Webhook path (e.g. `"/incoming"`)
- Use: API integrations, external service callbacks

### trigger_telegram
**Trigger on Telegram message**
- Inputs: none
- Outputs: `out`
- Fields:
  - `filter` (text) — Message filter expression
- Use: Chatbots, command handlers

### trigger_event
**Trigger when a memory is created/updated**
- Inputs: none
- Outputs: `out`
- Fields:
  - `event_type` (select: created/updated/consolidated) — Which event to listen for
  - `category_filter` (text) — Only trigger for memories in this category
- Use: Memory-driven automations, reactive workflows

---

## AI Nodes (5 nodes)

### ai_agent
**Run an AI agent with tools (function calling)**
- Inputs: `in`
- Outputs: `out`
- Fields:
  - `model` (model-select, default: `"gpt-5.2"`) — LLM model
  - `system_prompt` (textarea) — Agent personality and instructions
  - `max_tokens` (number, default: `4096`) — Max response tokens
  - `temperature` (number, default: `0.7`) — Creativity (0.0–2.0)
  - `tools_enabled` (toggle, default: `true`) — Enable function calling
- Use: Complex tasks requiring reasoning and tool use
- **Most powerful AI node** — use for tasks requiring multi-step reasoning

### ai_chat
**Simple LLM chat call (no tools)**
- Inputs: `in`
- Outputs: `out`
- Fields:
  - `model` (model-select, default: `"gpt-4o-mini"`) — LLM model
  - `prompt` (textarea) — Prompt template
  - `temperature` (number, default: `0.7`)
- Use: Summarization, rewriting, simple Q&A
- **Faster and cheaper** than ai_agent — use when tools aren't needed

### ai_classify
**Classify input into categories using an LLM**
- Inputs: `in`
- Outputs: `out`
- Fields:
  - `categories` (text) — Comma-separated categories (e.g. `"question, complaint, feedback"`)
  - `model` (model-select, default: `"gpt-4o-mini"`)
- Use: Content routing, sentiment analysis, intent detection
- **Pair with logic_switch** for routing workflows

### ai_extract
**Extract structured data from text using AI**
- Inputs: `in`
- Outputs: `out`
- Fields:
  - `schema` (code) — JSON schema for extraction (e.g. `{"name": "string", "email": "string"}`)
  - `model` (model-select, default: `"gpt-4o-mini"`)
- Use: Entity extraction, form parsing, data normalization

### ai_embedding
**Generate text embeddings for semantic search**
- Inputs: `in`
- Outputs: `out`
- Fields:
  - `model` (select: text-embedding-3-small / text-embedding-3-large, default: `"text-embedding-3-small"`)
- Use: Semantic search, similarity matching, clustering

---

## Action Nodes (5 nodes)

### action_memory_search
**Search HexBrain memories by semantic similarity**
- Inputs: `in`
- Outputs: `out`
- Fields:
  - `query` (text) — Search query
  - `brain_type` (select: user/agent/all, default: `"user"`) — Which brain to search
  - `limit` (number, default: `10`) — Max results
- Use: RAG, knowledge retrieval, context gathering

### action_memory_store
**Save a new memory to HexBrain**
- Inputs: `in`
- Outputs: `out`
- Fields:
  - `content` (textarea) — Memory content
  - `category` (text, default: `"knowledge"`) — Memory category
  - `importance` (number, default: `5`) — Importance 1–10
- Use: Learning from conversations, storing extracted data

### action_http
**Make an HTTP request to any API**
- Inputs: `in`
- Outputs: `out`
- Fields:
  - `method` (select: GET/POST/PUT/DELETE, default: `"GET"`)
  - `url` (text) — Full URL
  - `headers` (code) — JSON headers
  - `body` (code) — JSON body
- Use: API integrations, webhooks, external service calls

### action_exec
**Run a shell command in the workspace**
- Inputs: `in`
- Outputs: `out`
- Fields:
  - `command` (text) — Shell command
  - `timeout` (number, default: `30`) — Timeout in seconds
- Use: Script execution, file operations, system tasks
- ⚠️ **Security**: Be careful with user-provided input in commands

### action_send_telegram
**Send a Telegram message via HexBrain bot**
- Inputs: `in`
- Outputs: `out`
- Fields:
  - `message` (textarea) — Message text
  - `chat_id` (text, optional) — Target chat ID
- Use: Notifications, alerts, bot responses

---

## Logic Nodes (5 nodes)

### logic_if
**Branch workflow based on a boolean condition**
- Inputs: `in`
- Outputs: `true`, `false`
- Fields:
  - `condition` (text) — Boolean expression
- Use: Conditional branching

**⚠️ CRITICAL**: Uses `true`/`false` handles, NOT `out`. When connecting:
```
connect_nodes(source_handle="true", ...)   → condition met
connect_nodes(source_handle="false", ...)  → condition not met
```

### logic_switch
**Route to different outputs based on a value**
- Inputs: `in`
- Outputs: `case1`, `case2`, `default`
- Fields:
  - `field` (text) — Field to switch on
  - `case1_value` (text) — Value for case 1
  - `case2_value` (text) — Value for case 2
- Use: Multi-way routing, content-based routing

**Handle mapping:**
```
connect_nodes(source_handle="case1", ...)   → matches case1_value
connect_nodes(source_handle="case2", ...)   → matches case2_value
connect_nodes(source_handle="default", ...) → no match
```

### logic_merge
**Merge multiple inputs into a single output**
- Inputs: `in1`, `in2`
- Outputs: `out`
- Fields:
  - `mode` (select: wait_all/first/append, default: `"wait_all"`)
- Use: Joining parallel branches, aggregating results

**Modes:**
- `wait_all` — Wait for both inputs before continuing
- `first` — Continue as soon as first input arrives
- `append` — Combine all inputs into a list

**⚠️ CRITICAL**: Merge has TWO input handles (`in1`, `in2`). When connecting:
```
connect_nodes(target_handle="in1", ...)  → first source
connect_nodes(target_handle="in2", ...)  → second source
```

### logic_loop
**Iterate over a list of items**
- Inputs: `in`
- Outputs: `item`, `done`
- Fields:
  - `items_field` (text) — Field containing the array to iterate
- Use: Batch processing, iterating over search results

**Handle mapping:**
```
connect_nodes(source_handle="item", ...) → each iteration body
connect_nodes(source_handle="done", ...) → after all items processed
```

### logic_wait
**Pause execution for a specified duration**
- Inputs: `in`
- Outputs: `out`
- Fields:
  - `seconds` (number, default: `5`) — Duration in seconds
- Use: Rate limiting, delays between API calls

---

## Data Nodes (3 nodes)

### data_transform
**Transform data with JavaScript/Python code**
- Inputs: `in`
- Outputs: `out`
- Fields:
  - `code` (code) — Transformation code
- Use: Complex data manipulation, format conversion

### data_set
**Set a named variable in the workflow context**
- Inputs: `in`
- Outputs: `out`
- Fields:
  - `name` (text) — Variable name
  - `value` (text) — Variable value
- Use: Setting variables for downstream nodes

### data_filter
**Filter items by a condition expression**
- Inputs: `in`
- Outputs: `out`
- Fields:
  - `condition` (text) — Keep items matching this condition
- Use: Filtering search results, removing unwanted data

---

## Output Nodes (2 nodes)

### output_respond
**Send response back to the trigger source**
- Inputs: `in`
- Outputs: none
- Fields:
  - `message` (textarea) — Response message
- Use: Reply to webhook caller, bot response

### output_log
**Log data for debugging and monitoring**
- Inputs: `in`
- Outputs: none
- Fields:
  - `level` (select: info/warning/error, default: `"info"`)
- Use: Debugging, audit trail, monitoring
