# Validation Guide — Catching Errors Before Execution

How workflows are validated, what errors mean, and how to fix them.

---

## Validation Overview

Call `validate_workflow(workflow_id)` to check a workflow before activating it.
Returns a list of errors and warnings with fix suggestions.

---

## Validation Rules

### Rule 1: Must Have Exactly One Trigger
```
Error: "Workflow has no trigger node"
Error: "Multiple trigger nodes found: use only one"
```
**Fix**: Every workflow needs exactly ONE trigger node (type starts with `trigger_`).

### Rule 2: All Nodes Must Be Connected
```
Error: "Node 'Set Variables' has no incoming connections"
Warning: "Node 'Log Output' has no outgoing connections" (OK for terminal nodes)
```
**Fix**: Every non-trigger node must have at least one incoming edge.
Terminal nodes (output_respond, output_log, action_send_telegram) are OK without outgoing edges.

### Rule 3: No Cycles (Except Loop)
```
Error: "Cycle detected: node_a → node_b → node_a"
```
**Fix**: Only `logic_loop` is allowed to create cycles. Regular nodes cannot loop back.

### Rule 4: Required Config Fields Must Be Set
```
Error: "Node 'HTTP Request' missing required field 'url'"
```
**Fix**: Check NODE_REFERENCE.md for required fields (marked with `required: true`).

**Required fields per node type**:

| Node Type | Required Fields |
|-----------|----------------|
| trigger_schedule | cron |
| trigger_webhook | method |
| ai_agent | model |
| ai_chat | model, prompt |
| ai_classify | categories |
| ai_extract | schema |
| ai_embedding | model |
| action_http | url, method |
| action_exec | command |
| action_memory_search | query |
| logic_switch | field |

### Rule 5: Data Types Must Match
```
Warning: "Output of 'Extract Data' may not match expected input of 'IF Condition'"
```
**Fix**: Ensure upstream nodes produce the data expected by downstream nodes.

### Rule 6: Handle Names Must Be Valid
```
Error: "Source handle 'case3' not valid for node type 'logic_switch'"
```
**Valid handles per node type**:

| Node | Source Handles | Target Handles |
|------|--------------|----------------|
| logic_if | true, false | (default) |
| logic_switch | case1, case2, case3, default | (default) |
| logic_loop | item, done | (default) |
| logic_merge | (default) | in1, in2 |
| All others | (default) | (default) |

---

## Validation Response Format

```json
{
  "is_valid": false,
  "errors": [
    {
      "type": "missing_trigger",
      "message": "Workflow has no trigger node",
      "node_id": null,
      "fix": "Add a trigger node (trigger_manual, trigger_schedule, trigger_webhook, trigger_telegram, or trigger_event)"
    },
    {
      "type": "missing_required_field",
      "message": "Node 'HTTP Request' missing required field 'url'",
      "node_id": "abc123",
      "fix": "Set config.url on the HTTP Request node"
    }
  ],
  "warnings": [
    {
      "type": "disconnected_terminal",
      "message": "Node 'Log Output' has no outgoing connections",
      "node_id": "def456"
    }
  ]
}
```

---

## Error Categories

### Critical (blocks activation)
- **missing_trigger** — No trigger node
- **multiple_triggers** — More than one trigger node
- **missing_connection** — Node has no incoming edges
- **cycle_detected** — Non-loop cycle found
- **missing_required_field** — Required config field not set
- **invalid_handle** — Source/target handle name invalid

### Warning (workflow may still run)
- **disconnected_terminal** — Terminal node has no outgoing (usually OK)
- **empty_config** — Node uses all defaults (may be intentional)
- **unused_branch** — Switch/IF branch leads nowhere
- **type_mismatch** — Data type may not match

---

## Fixing Strategies

### Strategy 1: Fix From Top Down
Start at the trigger, follow the flow, fix each error in order.
Upstream fixes often resolve downstream warnings.

### Strategy 2: Fix Missing Fields First
Most common error. Use `update_node_config()` to set required fields:
```
update_node_config(node_id, config={"url": "https://api.example.com"})
```

### Strategy 3: Fix Connections
If nodes are disconnected:
```
list_workflow_nodes(workflow_id)  // find the node IDs
connect_nodes(source_id, target_id)
```

### Strategy 4: Handle Routing
For IF/Switch/Loop, always specify `source_handle`:
```
// IF node
connect_nodes(if_id, true_path_id, source_handle="true")
connect_nodes(if_id, false_path_id, source_handle="false")

// Switch node
connect_nodes(switch_id, handler1_id, source_handle="case1")
connect_nodes(switch_id, handler2_id, source_handle="case2")
connect_nodes(switch_id, default_id, source_handle="default")
```

### Strategy 5: Re-validate After Fixes
Always call `validate_workflow()` again after making changes.
Fix → Validate → Fix → Validate until clean.

---

## Pre-activation Checklist

1. ✅ Exactly one trigger node
2. ✅ All nodes connected (data flows from trigger to end)
3. ✅ Required config fields set on every node
4. ✅ IF/Switch/Loop handles connected correctly
5. ✅ Merge node has both inputs connected
6. ✅ `validate_workflow()` returns `is_valid: true`
7. ✅ `set_workflow_status("active")` succeeds
