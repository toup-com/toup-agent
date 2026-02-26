# MCP Tools Guide — Complete Tool Reference

All 16 tools, 2 resources, and 3 prompts available for workflow building.

---

## Quick Tool Index

| # | Tool | Purpose |
|---|------|---------|
| 1 | create_workflow | Create new empty workflow |
| 2 | list_workflows | Get all workflows |
| 3 | get_workflow | Get one workflow with all nodes/edges |
| 4 | update_workflow | Change name/description |
| 5 | delete_workflow | Remove workflow permanently |
| 6 | add_node | Add node to workflow |
| 7 | update_node_config | Change node config/position |
| 8 | remove_node | Delete node and its edges |
| 9 | connect_nodes | Create edge between nodes |
| 10 | disconnect_nodes | Remove edge between nodes |
| 11 | validate_workflow | Check for errors |
| 12 | set_workflow_status | Activate/deactivate |
| 13 | export_workflow | Export as JSON |
| 14 | import_workflow | Import from JSON |
| 15 | list_templates | Get available node templates |
| 16 | list_workflow_nodes | Get nodes + edges for a workflow |

---

## Tool Details

### 1. create_workflow

Creates a new empty workflow (no nodes, no edges).

**Parameters**:
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| name | string | ✅ | Workflow name |
| description | string | ❌ | What it does |

**Returns**: Workflow object with `id`, `name`, `description`, `status="draft"`.

**Example**:
```
create_workflow(name="My Bot", description="Telegram chatbot with memory")
→ { id: "abc-123", name: "My Bot", status: "draft" }
```

---

### 2. list_workflows

Lists all workflows with basic info (no nodes/edges).

**Parameters**: None

**Returns**: Array of workflow summaries.

---

### 3. get_workflow

Gets full workflow detail including all nodes and edges.

**Parameters**:
| Param | Type | Required |
|-------|------|----------|
| workflow_id | string | ✅ |

**Returns**: Full workflow with `nodes[]` and `edges[]`.

**Use when**: You need to see the current state, check connections, get node IDs.

---

### 4. update_workflow

Updates workflow name and/or description.

**Parameters**:
| Param | Type | Required |
|-------|------|----------|
| workflow_id | string | ✅ |
| name | string | ❌ |
| description | string | ❌ |

---

### 5. delete_workflow

Permanently deletes a workflow and all its nodes/edges.

**Parameters**:
| Param | Type | Required |
|-------|------|----------|
| workflow_id | string | ✅ |

**⚠️ Irreversible!** Export first if you might need it later.

---

### 6. add_node ⭐ (Most Used)

Adds a node to a workflow.

**Parameters**:
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| workflow_id | string | ✅ | Target workflow |
| node_type | string | ✅ | One of 25 node types |
| label | string | ❌ | Custom display name |
| config | object | ❌ | Node configuration |
| position_x | number | ❌ | Canvas X (default 0) |
| position_y | number | ❌ | Canvas Y (default 0) |

**Valid node_type values**:
```
Triggers:  trigger_manual, trigger_schedule, trigger_webhook, trigger_telegram, trigger_event
AI:        ai_agent, ai_chat, ai_classify, ai_extract, ai_embedding
Actions:   action_memory_search, action_memory_store, action_http, action_exec, action_send_telegram
Logic:     logic_if, logic_switch, logic_merge, logic_loop, logic_wait
Data:      data_transform, data_set, data_filter
Output:    output_respond, output_log
```

**Example**:
```
add_node(
  workflow_id="abc-123",
  node_type="ai_agent",
  label="Support Bot",
  config={"model": "gpt-5.2", "system_prompt": "Be helpful", "temperature": 0.7},
  position_x=400,
  position_y=200
)
```

**Returns**: Node object with `id`, `node_type`, `label`, `config`.

---

### 7. update_node_config

Changes a node's configuration, label, or position.

**Parameters**:
| Param | Type | Required |
|-------|------|----------|
| workflow_id | string | ✅ |
| node_id | string | ✅ |
| label | string | ❌ |
| config | object | ❌ |
| position_x | number | ❌ |
| position_y | number | ❌ |

**⚠️ Config is merged**: New fields are added, existing fields updated. To remove a field, set it to `null`.

---

### 8. remove_node

Deletes a node and ALL edges connected to it.

**Parameters**:
| Param | Type | Required |
|-------|------|----------|
| workflow_id | string | ✅ |
| node_id | string | ✅ |

---

### 9. connect_nodes ⭐ (Critical)

Creates an edge between two nodes.

**Parameters**:
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| workflow_id | string | ✅ | |
| source_node_id | string | ✅ | Upstream node |
| target_node_id | string | ✅ | Downstream node |
| source_handle | string | ❌ | Output port name |
| target_handle | string | ❌ | Input port name |

**Default handles**: Most nodes use default (omit the param).

**Special handles** — MUST specify for these nodes:
```
logic_if:     source_handle = "true" | "false"
logic_switch: source_handle = "case1" | "case2" | "case3" | "default"
logic_loop:   source_handle = "item" | "done"
logic_merge:  target_handle = "in1" | "in2"
```

---

### 10. disconnect_nodes

Removes an edge between two nodes.

**Parameters**:
| Param | Type | Required |
|-------|------|----------|
| workflow_id | string | ✅ |
| source_node_id | string | ✅ |
| target_node_id | string | ✅ |

---

### 11. validate_workflow

Checks workflow for errors and warnings.

**Parameters**:
| Param | Type | Required |
|-------|------|----------|
| workflow_id | string | ✅ |

**Returns**: `{ is_valid, errors[], warnings[] }`

**Always call before activating!**

---

### 12. set_workflow_status

Changes workflow status.

**Parameters**:
| Param | Type | Required | Description |
|-------|------|----------|-------------|
| workflow_id | string | ✅ | |
| status | string | ✅ | "draft", "active", "paused", "archived" |

**Status lifecycle**: `draft → active → paused → active → archived`

---

### 13. export_workflow

Exports workflow as portable JSON.

**Parameters**:
| Param | Type | Required |
|-------|------|----------|
| workflow_id | string | ✅ |

**Returns**: Complete JSON with nodes, edges, config — ready for `import_workflow`.

---

### 14. import_workflow

Creates a new workflow from exported JSON.

**Parameters**:
| Param | Type | Required |
|-------|------|----------|
| workflow_data | object | ✅ |

---

### 15. list_templates

Gets all available node templates with their fields and defaults.

**Parameters**: None (or optional `category` filter)

**Returns**: All 25 templates organized by category.

**Use when**: You need to know what config fields a node type accepts.

---

### 16. list_workflow_nodes

Gets just the nodes and edges for a workflow (lighter than get_workflow).

**Parameters**:
| Param | Type | Required |
|-------|------|----------|
| workflow_id | string | ✅ |

---

## Resources

### workflow://templates
Full node template catalog. Same as `list_templates()` but as a readable resource.

### workflow://guide
Quick-start guide text for building workflows.

---

## Prompts

### build_workflow
Prompt template for building a new workflow from a description.
**Arg**: `description` — what the workflow should do.

### debug_workflow
Prompt template for debugging a broken workflow.
**Arg**: `workflow_id` — the workflow to debug.

### optimize_workflow
Prompt template for optimizing an existing workflow.
**Arg**: `workflow_id` — the workflow to optimize.

---

## Common Tool Sequences

### Build a new workflow
```
1. create_workflow(name, description)
2. add_node(trigger_type)
3. add_node(processing_nodes...)   // repeat
4. connect_nodes(...)              // repeat
5. validate_workflow()
6. set_workflow_status("active")
```

### Debug a workflow
```
1. get_workflow(id)                // see full state
2. validate_workflow(id)           // find errors
3. update_node_config(...)         // fix issues
4. connect_nodes(...)              // fix connections
5. validate_workflow(id)           // re-check
```

### Clone and modify
```
1. export_workflow(source_id)
2. import_workflow(exported_data)  // creates new
3. update_workflow(new_id, new_name)
4. update_node_config(...)         // modify as needed
```

### Replace a node
```
1. list_workflow_nodes(id)         // find connected nodes
2. remove_node(old_node_id)       // removes node + edges
3. add_node(new_type, config)     // add replacement
4. connect_nodes(...)             // reconnect
5. validate_workflow()
```
