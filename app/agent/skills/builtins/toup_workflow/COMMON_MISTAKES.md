# Common Mistakes — Error Catalog & Solutions

Every mistake you can make and exactly how to fix it.

---

## Connection Mistakes

### ❌ Forgetting source_handle on IF node
```
WRONG:  connect_nodes(if_id, handler_id)
RIGHT:  connect_nodes(if_id, handler_id, source_handle="true")
        connect_nodes(if_id, other_id, source_handle="false")
```
**Why**: IF has TWO outputs. Without `source_handle`, the system doesn't know which branch.

### ❌ Forgetting source_handle on Switch node
```
WRONG:  connect_nodes(switch_id, handler_id)
RIGHT:  connect_nodes(switch_id, handler_id, source_handle="case1")
```
**Why**: Switch has up to 4 outputs (case1, case2, case3, default).

### ❌ Using wrong handle on Loop node
```
WRONG:  connect_nodes(loop_id, process_id, source_handle="loop")
RIGHT:  connect_nodes(loop_id, process_id, source_handle="item")
        connect_nodes(loop_id, done_id, source_handle="done")
```
**Handles are**: `item` (per-iteration) and `done` (after all items).

### ❌ Forgetting target_handle on Merge node
```
WRONG:  connect_nodes(branch1_id, merge_id)
        connect_nodes(branch2_id, merge_id)
RIGHT:  connect_nodes(branch1_id, merge_id, target_handle="in1")
        connect_nodes(branch2_id, merge_id, target_handle="in2")
```
**Why**: Merge has TWO inputs that must be differentiated.

### ❌ Connecting nodes in wrong direction
```
WRONG:  connect_nodes(ai_agent_id, trigger_id)   // backwards!
RIGHT:  connect_nodes(trigger_id, ai_agent_id)   // trigger → agent
```
**Remember**: source = upstream, target = downstream. Data flows left → right.

---

## Node Configuration Mistakes

### ❌ Using wrong model name
```
WRONG:  config={"model": "claude-3"}
RIGHT:  config={"model": "claude-sonnet-4-20250514"}
```
**Use**: Full model identifiers. Check your LLM provider for exact names.

### ❌ Missing required config fields
```
WRONG:  add_node(workflow_id, "ai_chat")                    // no model or prompt
RIGHT:  add_node(workflow_id, "ai_chat", config={"model": "gpt-4o", "prompt": "..."})
```
**Required fields** vary by node. See NODE_REFERENCE.md or call `list_templates()`.

### ❌ Config as string instead of object
```
WRONG:  config='{"model": "gpt-4o"}'     // string!
RIGHT:  config={"model": "gpt-4o"}       // object
```

### ❌ Setting cron without timezone
```
RISKY:  config={"cron": "0 9 * * *"}
BETTER: config={"cron": "0 9 * * *", "timezone": "UTC"}
```
**Why**: Without timezone, schedule behavior may be unpredictable.

---

## Workflow Structure Mistakes

### ❌ Multiple trigger nodes
```
WRONG:  add_node(wf_id, "trigger_manual")
        add_node(wf_id, "trigger_webhook")     // second trigger!
RIGHT:  One trigger per workflow.
```
**Fix**: Use only ONE trigger. If you need multiple entry points, create separate workflows.

### ❌ No trigger node
```
WRONG:  add_node(wf_id, "ai_agent")           // no trigger!
RIGHT:  add_node(wf_id, "trigger_manual")     // add trigger first
        add_node(wf_id, "ai_agent")
```

### ❌ Disconnected nodes (islands)
```
WRONG:  add_node → add_node → add_node        // no connect_nodes calls!
RIGHT:  add_node → add_node → connect_nodes → add_node → connect_nodes
```
**Fix**: Connect nodes as you add them. Every non-trigger node needs an incoming edge.

### ❌ Creating cycles
```
WRONG:  connect_nodes(A → B → C → A)          // cycle!
RIGHT:  Use logic_loop for iteration patterns
```

---

## Ordering Mistakes

### ❌ Connecting before adding nodes
```
WRONG:  connect_nodes(node1_id, node2_id)      // nodes don't exist yet!
RIGHT:  1. add_node → get node1_id
        2. add_node → get node2_id
        3. connect_nodes(node1_id, node2_id)
```

### ❌ Activating before validating
```
WRONG:  set_workflow_status("active")          // might have errors!
RIGHT:  validate_workflow(wf_id)               // check first
        set_workflow_status("active")           // then activate
```

### ❌ Skipping node IDs from add_node response
```
WRONG:  add_node(wf_id, "trigger_manual")     // ignoring returned ID
        connect_nodes(???, ???)                 // don't know the IDs!
RIGHT:  result = add_node(wf_id, "trigger_manual")
        trigger_id = result.id                  // save the ID
        result2 = add_node(wf_id, "ai_agent")
        agent_id = result2.id                   // save this too
        connect_nodes(trigger_id, agent_id)     // use saved IDs
```

---

## Canvas Layout Mistakes

### ❌ All nodes at (0, 0)
```
WRONG:  add_node(..., position_x=0, position_y=0)   // everything stacked!
RIGHT:  add_node(..., position_x=100, position_y=200)
        add_node(..., position_x=400, position_y=200)
        add_node(..., position_x=700, position_y=200)
```
**Rule**: Space nodes ~300px apart horizontally, align branches vertically.

### ❌ Not accounting for branches
```
WRONG:  IF node and both paths at same Y position
RIGHT:  IF at y=300
        True path at y=150
        False path at y=450
```
**Rule**: Spread branches ±150px vertically from the branching node.

---

## Quick Fix Checklist

When things go wrong:
1. `get_workflow(id)` — see full current state
2. `validate_workflow(id)` — find all errors
3. Fix errors one by one, most critical first
4. `validate_workflow(id)` — verify fix worked
5. Repeat until `is_valid: true`
6. `set_workflow_status("active")`
