# HexBrain — Building Skills (Plugins)

Skills are modular plugins that extend the HexBrain agent with new tools, system prompt sections, and behaviors.

## Directory Structure

```
skills/
└── my_skill/
    ├── skill.py        # Required: contains your Skill subclass
    └── README.md       # Optional: documentation
```

Skills can live in two locations:
1. **Built-in:** `backend/app/agent/skills/builtins/` — shipped with HexBrain
2. **External:** `/app/skills/` (Docker volume `agent_skills`) — user-installed

## Creating a Skill

### 1. Create the skill directory

```bash
mkdir -p skills/my_skill
```

### 2. Create `skill.py`

Every skill must have a `skill.py` file containing a class that extends `Skill`:

```python
from app.agent.skills.base import Skill, SkillContext, SkillMeta
from typing import Any, Dict, List, Optional


class MySkill(Skill):
    """My custom HexBrain skill."""

    meta = SkillMeta(
        name="my_skill",              # Unique slug (used as tool prefix)
        version="1.0.0",
        description="What this skill does",
        author="Your Name",
    )

    def get_tools(self) -> List[Dict[str, Any]]:
        """Return tool definitions. Names MUST be prefixed with '<skill_name>__'."""
        return [
            {
                "name": "my_skill__hello",
                "description": "Says hello to the user",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name to greet",
                        },
                    },
                    "required": ["name"],
                },
            },
        ]

    async def execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        ctx: SkillContext,
    ) -> str:
        """Execute a tool call. Return the result as a string."""
        if tool_name == "my_skill__hello":
            return f"Hello, {args['name']}! 👋"
        return f"ERROR: Unknown tool: {tool_name}"

    def get_system_prompt_section(self) -> Optional[str]:
        """Optional: inject text into the agent's system prompt."""
        return (
            "# My Skill\n"
            "You have a greeting tool. Use `my_skill__hello` when asked to greet someone."
        )

    async def on_load(self) -> None:
        """Optional: called once when the skill loads at startup."""
        print("MySkill loaded!")

    async def on_unload(self) -> None:
        """Optional: called on shutdown."""
        print("MySkill unloaded!")
```

### 3. Deploy

**Docker (production):**
Copy the skill folder into the `agent_skills` volume:
```bash
docker cp skills/my_skill hexbrain-backend:/app/skills/my_skill
docker compose restart backend
```

**Built-in (development):**
Place the folder under `backend/app/agent/skills/builtins/my_skill/`.

### 4. Verify

```bash
# Check logs
docker compose logs backend | grep SKILLS

# Via Telegram
/skills

# Via API
curl -H "Authorization: Bearer hx_..." http://localhost:8000/api/v1/skills
```

## Skill Base Class Reference

### `SkillMeta` (dataclass)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `str` | Yes | Unique slug (e.g. `"toup"`) |
| `version` | `str` | No | Semver (default `"0.1.0"`) |
| `description` | `str` | No | Short description |
| `author` | `str` | No | Author name |
| `requires_opt_in` | `bool` | No | If `True`, only loaded when explicitly enabled |

### `SkillContext` (dataclass)

Passed to every `execute_tool` call:

| Field | Type | Description |
|-------|------|-------------|
| `workspace` | `str` | Agent workspace directory path |
| `user_id` | `str` | Current user's ID |
| `session_id` | `str` | Current session ID |
| `chat_id` | `Optional[int]` | Telegram chat ID (if applicable) |
| `extra` | `Dict` | Additional context |

### `Skill` (ABC) Methods

| Method | Required | Description |
|--------|----------|-------------|
| `get_tools()` | **Yes** | Return list of tool definitions |
| `execute_tool(name, args, ctx)` | **Yes** | Execute a tool, return result string |
| `get_system_prompt_section()` | No | Return text to inject into system prompt |
| `on_load()` | No | Called once at startup |
| `on_unload()` | No | Called at shutdown |
| `_prefix(name)` | Helper | Returns `f"{self.meta.name}__{name}"` |

## Tool Naming Convention

All tool names **must** be prefixed with the skill's name followed by double underscores:

```
<skill_name>__<tool_name>
```

Examples:
- `toup__create_spec`
- `toup__scaffold`
- `my_skill__hello`

The loader validates this at registration time and rejects skills with incorrectly named tools.

## Tool Definition Schema

Each tool in `get_tools()` must follow this format:

```python
{
    "name": "skill_name__tool_name",
    "description": "What this tool does",
    "input_schema": {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "..."},
            "param2": {"type": "integer", "description": "..."},
        },
        "required": ["param1"],
    },
}
```

## Built-in Skills

### Toup (`toup`)

Software engineering assistant with 5 tools:

| Tool | Description |
|------|-------------|
| `toup__create_spec` | Turn a feature idea into a product spec |
| `toup__scaffold` | Generate project scaffolds (fastapi/nextjs/fullstack) |
| `toup__changeset` | Plan code changes as a structured changeset |
| `toup__review_diff` | Review diffs for bugs, security, style |
| `toup__plan_sprint` | Break an epic into sprint tasks |

## Tips

- **Keep tools focused.** One tool per action, clear input schemas.
- **Return strings.** All `execute_tool` results must be strings (the agent reads them as text).
- **Use `ctx.workspace`** for file operations — it's the agent's working directory.
- **Log errors** but don't crash — return `"ERROR: ..."` strings for graceful handling.
- **Test locally** by placing the skill in `backend/app/agent/skills/builtins/` during development.
