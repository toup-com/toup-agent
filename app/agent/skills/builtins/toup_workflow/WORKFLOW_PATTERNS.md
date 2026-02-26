# Workflow Patterns — Proven Architectures

5 battle-tested patterns for building HexBrain workflows.

---

## Pattern Selection Guide

| Need | Pattern | Complexity |
|------|---------|------------|
| Bot / interactive agent | Chatbot | Simple |
| Receive data from external service | Webhook Pipeline | Simple |
| Route different inputs to different handlers | Smart Router | Medium |
| Recurring task / report | Scheduled Task | Simple |
| Process arrays / batches | Loop Processing | Medium |
| Parallel processing + merge | Fan-out / Fan-in | Advanced |

---

## Pattern 1: Chatbot

**Structure**: `Trigger → AI Agent → Response`

**When to use**: Interactive bots, Q&A, customer support, conversational AI.

**Full build sequence**:
```
1. create_workflow("Customer Support Bot", "AI-powered support agent")
2. add_node("trigger_telegram", position_x=100, position_y=200)
3. add_node("ai_agent", config={
     "model": "gpt-5.2",
     "system_prompt": "You are a helpful customer support agent. Be concise and friendly.",
     "tools_enabled": true,
     "temperature": 0.7
   }, position_x=400, position_y=200)
4. add_node("output_respond", position_x=700, position_y=200)
5. connect_nodes(trigger → ai_agent)
6. connect_nodes(ai_agent → respond)
7. validate_workflow()
8. set_workflow_status("active")
```

**Variations**:
- **Webhook chatbot**: Replace `trigger_telegram` with `trigger_webhook`
- **With memory**: Add `action_memory_search` before AI Agent for context
- **With logging**: Add `output_log` after response

**Node selection**:
- Use `ai_agent` (not `ai_chat`) when you need tool-calling capability
- Use `ai_chat` for simple prompt-in/text-out without tools

---

## Pattern 2: Webhook Pipeline

**Structure**: `Webhook → Extract → Store → Log`

**When to use**: Ingesting data from APIs, processing form submissions, ETL.

**Full build sequence**:
```
1. create_workflow("Data Ingestion Pipeline", "Receives data via webhook and stores as memories")
2. add_node("trigger_webhook", config={
     "method": "POST",
     "path": "/ingest"
   }, position_x=100, position_y=200)
3. add_node("ai_extract", config={
     "schema": "{\"name\": \"string\", \"topic\": \"string\", \"summary\": \"string\"}",
     "model": "gpt-4o-mini"
   }, position_x=400, position_y=200)
4. add_node("action_memory_store", config={
     "category": "knowledge",
     "importance": 7
   }, position_x=700, position_y=200)
5. add_node("output_log", position_x=1000, position_y=200)
6. connect_nodes(webhook → extract → store → log)
7. validate_workflow()
8. set_workflow_status("active")
```

**Variations**:
- **With validation**: Add `logic_if` after extract to check data quality
- **Multi-source**: Use `trigger_event` instead for memory-triggered pipelines
- **API forwarding**: Add `action_http` to forward processed data

---

## Pattern 3: Smart Router

**Structure**: `Trigger → Classify → Switch → [Handler1, Handler2, Default]`

**When to use**: Routing different types of input to specialized handlers.

**Full build sequence**:
```
1. create_workflow("Smart Message Router", "Routes messages by intent")
2. add_node("trigger_webhook", config={"method": "POST"}, position_x=100, position_y=300)
3. add_node("ai_classify", config={
     "categories": "question, complaint, feedback, other"
   }, position_x=400, position_y=300)
4. add_node("logic_switch", config={
     "field": "category",
     "case1_value": "question",
     "case2_value": "complaint"
   }, position_x=700, position_y=300)
5. add_node("ai_agent", label="Answer Questions", config={
     "system_prompt": "Answer the user's question helpfully."
   }, position_x=1000, position_y=100)
6. add_node("action_send_telegram", label="Alert Team", config={
     "message": "⚠️ New complaint received"
   }, position_x=1000, position_y=300)
7. add_node("action_memory_store", label="Save Feedback", config={
     "category": "feedback", "importance": 6
   }, position_x=1000, position_y=500)
8. connect_nodes(webhook → classify)
9. connect_nodes(classify → switch)
10. connect_nodes(switch →[case1] answer)
11. connect_nodes(switch →[case2] alert)
12. connect_nodes(switch →[default] save)
13. validate_workflow()
```

**⚠️ Critical**: When connecting switch outputs, specify `source_handle`:
```
connect_nodes(switch_id, answer_id, source_handle="case1")
connect_nodes(switch_id, alert_id, source_handle="case2")
connect_nodes(switch_id, save_id, source_handle="default")
```

---

## Pattern 4: Scheduled Task

**Structure**: `Schedule → Gather → Process → Notify`

**When to use**: Daily reports, periodic cleanup, recurring analysis.

**Full build sequence**:
```
1. create_workflow("Daily Memory Digest", "Sends daily summary of new memories")
2. add_node("trigger_schedule", config={
     "cron": "0 9 * * *",
     "timezone": "UTC"
   }, position_x=100, position_y=200)
3. add_node("action_memory_search", config={
     "query": "recent memories from today",
     "brain_type": "all",
     "limit": 20
   }, position_x=400, position_y=200)
4. add_node("ai_chat", config={
     "prompt": "Summarize these memories into a brief daily digest:",
     "model": "gpt-4o-mini"
   }, position_x=700, position_y=200)
5. add_node("action_send_telegram", config={
     "message": "📋 Daily Memory Digest"
   }, position_x=1000, position_y=200)
6. connect_nodes(schedule → search → chat → telegram)
7. validate_workflow()
8. set_workflow_status("active")
```

---

## Pattern 5: Loop Processing

**Structure**: `Trigger → Loop → [Per-item processing] → Done handler`

**When to use**: Batch processing arrays, iterating over search results.

**Full build sequence**:
```
1. create_workflow("Batch Analyzer", "Processes each item in a list")
2. add_node("trigger_webhook", config={
     "method": "POST", "path": "/batch"
   }, position_x=100, position_y=300)
3. add_node("logic_loop", config={
     "items_field": "items"
   }, position_x=400, position_y=300)
4. add_node("ai_chat", label="Analyze Item", config={
     "prompt": "Analyze this item and provide insights:",
     "model": "gpt-4o-mini"
   }, position_x=700, position_y=200)
5. add_node("action_memory_store", label="Save Analysis", config={
     "category": "analysis"
   }, position_x=1000, position_y=200)
6. add_node("output_respond", label="Batch Complete", position_x=700, position_y=500)
7. connect_nodes(webhook → loop)
8. connect_nodes(loop →[item] analyze)     // source_handle="item"
9. connect_nodes(analyze → store)
10. connect_nodes(loop →[done] respond)    // source_handle="done"
11. validate_workflow()
```

**⚠️ Critical**: Loop has TWO outputs:
- `source_handle="item"` → per-iteration processing
- `source_handle="done"` → after all items

---

## Pattern 6: Fan-out / Fan-in (Advanced)

**Structure**: `Trigger → [Parallel branches] → Merge → Output`

**When to use**: Running multiple operations in parallel, then combining results.

```
trigger_webhook → logic_merge ← action_memory_search
                             ← action_http
                  ↓
                  ai_chat → output_respond
```

**Build sequence**:
```
1. create_workflow("Parallel Enrichment")
2. add_node("trigger_webhook", position_x=100, position_y=300)
3. add_node("action_memory_search", position_x=400, position_y=200)
4. add_node("action_http", config={"url": "https://api.example.com"}, position_x=400, position_y=400)
5. add_node("logic_merge", config={"mode": "wait_all"}, position_x=700, position_y=300)
6. add_node("ai_chat", position_x=1000, position_y=300)
7. add_node("output_respond", position_x=1300, position_y=300)

// Fan-out
8. connect_nodes(webhook → memory_search)
9. connect_nodes(webhook → http_request)

// Fan-in (note different target_handles!)
10. connect_nodes(memory_search → merge, target_handle="in1")
11. connect_nodes(http_request → merge, target_handle="in2")

// Continue
12. connect_nodes(merge → ai_chat → respond)
```

**⚠️ Critical**: Merge needs TWO inputs:
- `target_handle="in1"` → first source
- `target_handle="in2"` → second source
