# Toup Workflow Skill

> **Version**: 1.0.0 | **Author**: Toup | **Category**: Workflow Building

A comprehensive knowledge skill that teaches the HexBrain agent how to build, validate, and manage visual workflows using the 25 available node types and 16 MCP tools.

## Inspired By

Architecture modeled after [n8n-skills](https://github.com/czlonkowski/n8n-skills) — structured knowledge files with SKILL.md frontmatter, reference documents, and pattern libraries.

## What's Inside

| File | Purpose | Lines |
|------|---------|-------|
| `SKILL.md` | Core skill guide — quick reference, concepts, building sequence | ~170 |
| `NODE_REFERENCE.md` | All 25 node types with fields, handles, defaults | ~250 |
| `WORKFLOW_PATTERNS.md` | 6 proven patterns with step-by-step build instructions | ~200 |
| `VALIDATION_GUIDE.md` | Validation rules, error types, fixing strategies | ~150 |
| `MCP_TOOLS_GUIDE.md` | All 16 MCP tools with parameters and examples | ~250 |
| `COMMON_MISTAKES.md` | Error catalog: wrong/right patterns for every mistake | ~150 |
| `skill.py` | Python Skill subclass with 7 tools for knowledge retrieval | ~300 |

## Tools Provided

| Tool | Description |
|------|-------------|
| `toup_workflow__get_skill_guide` | Core workflow-building knowledge |
| `toup_workflow__get_node_reference` | Node catalog (filterable by category) |
| `toup_workflow__get_pattern` | Workflow patterns (filterable by name) |
| `toup_workflow__get_validation_guide` | Validation rules & error fixing |
| `toup_workflow__get_tools_guide` | MCP tool reference |
| `toup_workflow__get_common_mistakes` | Error catalog with solutions |
| `toup_workflow__lookup` | Smart keyword search across all knowledge |

## Node Categories Covered

- **Triggers** (5): manual, schedule, webhook, telegram, event
- **AI** (5): agent, chat, classify, extract, embedding
- **Actions** (5): memory_search, memory_store, http, exec, send_telegram
- **Logic** (5): if, switch, merge, loop, wait
- **Data** (3): transform, set, filter
- **Output** (2): respond, log

## How It Works

The skill auto-loads from `builtins/toup_workflow/` via `SkillLoader`.
It injects a system prompt section with quick rules for correct workflow building.
When the agent needs details, it calls the appropriate `toup_workflow__*` tool
to retrieve the relevant knowledge file.

## Integration

Works alongside the MCP workflow server (`backend/mcp_server/`) and the
REST API (`/api/workflows/`). The skill provides the *knowledge*,
the MCP server provides the *actions*.
