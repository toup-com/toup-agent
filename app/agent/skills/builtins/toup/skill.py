"""
Toup Skill — Software engineering assistant for the Toup platform.

Provides 5 tools:
  toup__create_spec     — Turn a feature description into a structured product spec
  toup__scaffold        — Generate a project scaffold (FastAPI, Next.js, etc.)
  toup__changeset       — Generate a code changeset from a description
  toup__review_diff     — Review a diff/patch and provide feedback
  toup__plan_sprint     — Break an epic into sprint-sized tasks
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from app.agent.skills.base import Skill, SkillContext, SkillMeta

logger = logging.getLogger(__name__)


# ======================================================================
# Project scaffolds (templates)
# ======================================================================

FASTAPI_SCAFFOLD = {
    "backend/app/__init__.py": "",
    "backend/app/main.py": textwrap.dedent("""\
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware

        app = FastAPI(title="{name}", version="0.1.0")

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )


        @app.get("/")
        async def root():
            return {{"status": "ok", "service": "{name}"}}


        @app.get("/health")
        async def health():
            return {{"status": "healthy"}}
    """),
    "backend/app/config.py": textwrap.dedent("""\
        from pydantic_settings import BaseSettings


        class Settings(BaseSettings):
            app_name: str = "{name}"
            debug: bool = True
            database_url: str = "sqlite+aiosqlite:///./app.db"

            class Config:
                env_file = ".env"


        settings = Settings()
    """),
    "backend/app/models.py": textwrap.dedent("""\
        from sqlalchemy.orm import DeclarativeBase


        class Base(DeclarativeBase):
            pass


        # Add your models here
    """),
    "backend/requirements.txt": textwrap.dedent("""\
        fastapi>=0.109.0
        uvicorn[standard]>=0.27.0
        sqlalchemy[asyncio]>=2.0.25
        pydantic-settings>=2.1.0
        python-dotenv>=1.0.0
    """),
    "backend/Dockerfile": textwrap.dedent("""\
        FROM python:3.12-slim
        WORKDIR /app
        COPY requirements.txt .
        RUN pip install --no-cache-dir -r requirements.txt
        COPY app/ app/
        CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
    """),
    ".env.example": "DATABASE_URL=sqlite+aiosqlite:///./app.db\nDEBUG=true\n",
    "README.md": "# {name}\n\n{description}\n\n## Setup\n\n```bash\npip install -r backend/requirements.txt\nuvicorn app.main:app --reload\n```\n",
    ".gitignore": "__pycache__/\n*.pyc\n.env\n*.db\n.venv/\n",
}

NEXTJS_SCAFFOLD = {
    "src/app/page.tsx": textwrap.dedent("""\
        export default function Home() {{
          return (
            <main className="flex min-h-screen flex-col items-center justify-center p-24">
              <h1 className="text-4xl font-bold">{name}</h1>
              <p className="mt-4 text-lg text-gray-600">{description}</p>
            </main>
          );
        }}
    """),
    "src/app/layout.tsx": textwrap.dedent("""\
        import type {{ Metadata }} from "next";
        import "./globals.css";

        export const metadata: Metadata = {{
          title: "{name}",
          description: "{description}",
        }};

        export default function RootLayout({{
          children,
        }}: {{
          children: React.ReactNode;
        }}) {{
          return (
            <html lang="en">
              <body>{{children}}</body>
            </html>
          );
        }}
    """),
    "src/app/globals.css": "@tailwind base;\n@tailwind components;\n@tailwind utilities;\n",
    "package.json": json.dumps({
        "name": "{name}",
        "version": "0.1.0",
        "private": True,
        "scripts": {
            "dev": "next dev",
            "build": "next build",
            "start": "next start",
            "lint": "next lint",
        },
        "dependencies": {
            "next": "^14.0.0",
            "react": "^18.0.0",
            "react-dom": "^18.0.0",
        },
        "devDependencies": {
            "@types/node": "^20.0.0",
            "@types/react": "^18.0.0",
            "typescript": "^5.0.0",
            "tailwindcss": "^3.4.0",
            "postcss": "^8.4.0",
            "autoprefixer": "^10.4.0",
        },
    }, indent=2),
    "tsconfig.json": json.dumps({
        "compilerOptions": {
            "target": "es5",
            "lib": ["dom", "dom.iterable", "esnext"],
            "allowJs": True,
            "skipLibCheck": True,
            "strict": True,
            "noEmit": True,
            "esModuleInterop": True,
            "module": "esnext",
            "moduleResolution": "bundler",
            "resolveJsonModule": True,
            "isolatedModules": True,
            "jsx": "preserve",
            "incremental": True,
            "paths": {"@/*": ["./src/*"]},
        },
        "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx"],
        "exclude": ["node_modules"],
    }, indent=2),
    "tailwind.config.js": textwrap.dedent("""\
        /** @type {import('tailwindcss').Config} */
        module.exports = {
          content: ["./src/**/*.{js,ts,jsx,tsx}"],
          theme: { extend: {} },
          plugins: [],
        };
    """),
    ".gitignore": "node_modules/\n.next/\n.env\n",
    "README.md": "# {name}\n\n{description}\n\n## Setup\n\n```bash\nnpm install\nnpm run dev\n```\n",
}

FULLSTACK_SCAFFOLD = {**FASTAPI_SCAFFOLD, **{f"frontend/{k}": v for k, v in NEXTJS_SCAFFOLD.items()}}


# ======================================================================
# Toup Skill
# ======================================================================

class ToupSkill(Skill):
    """Software engineering assistant — specs, scaffolds, changesets, reviews, sprint planning."""

    meta = SkillMeta(
        name="toup",
        version="1.0.0",
        description="Software engineering tools: specs, scaffolds, changesets, code review, sprint planning.",
        author="Toup",
    )

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------
    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "toup__create_spec",
                "description": (
                    "Turn a feature description into a structured product specification. "
                    "Outputs a markdown spec with goals, user stories, acceptance criteria, "
                    "tech stack, and milestones."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "feature": {
                            "type": "string",
                            "description": "Description of the feature or product to spec out.",
                        },
                        "context": {
                            "type": "string",
                            "description": "Additional context (tech stack, constraints, existing code, etc.).",
                        },
                    },
                    "required": ["feature"],
                },
            },
            {
                "name": "toup__scaffold",
                "description": (
                    "Generate a project scaffold in the workspace. "
                    "Supports: fastapi, nextjs, fullstack (fastapi+nextjs). "
                    "Creates all boilerplate files ready to run."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Project name (slug, e.g. 'my-app').",
                        },
                        "template": {
                            "type": "string",
                            "enum": ["fastapi", "nextjs", "fullstack"],
                            "description": "Scaffold template to use.",
                        },
                        "description": {
                            "type": "string",
                            "description": "Short project description.",
                        },
                    },
                    "required": ["name", "template"],
                },
            },
            {
                "name": "toup__changeset",
                "description": (
                    "Generate a detailed code changeset (list of file changes) from a description. "
                    "Returns a structured plan of files to create/modify/delete with content."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "What change to make. Be specific about files and logic.",
                        },
                        "project_dir": {
                            "type": "string",
                            "description": "Project directory (workspace-relative or absolute).",
                        },
                        "files_context": {
                            "type": "string",
                            "description": "Relevant existing file contents for context.",
                        },
                    },
                    "required": ["description"],
                },
            },
            {
                "name": "toup__review_diff",
                "description": (
                    "Review a code diff/patch and provide structured feedback: "
                    "bugs, security issues, style, performance, and suggestions."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "diff": {
                            "type": "string",
                            "description": "The diff/patch text to review (unified diff format).",
                        },
                        "context": {
                            "type": "string",
                            "description": "PR description or additional context.",
                        },
                    },
                    "required": ["diff"],
                },
            },
            {
                "name": "toup__plan_sprint",
                "description": (
                    "Break an epic or large feature into sprint-sized tasks. "
                    "Returns a structured task list with estimates, priorities, and dependencies."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "epic": {
                            "type": "string",
                            "description": "Description of the epic / large feature to break down.",
                        },
                        "sprint_days": {
                            "type": "integer",
                            "description": "Sprint length in days (default 14).",
                        },
                        "team_size": {
                            "type": "integer",
                            "description": "Number of developers (default 1).",
                        },
                    },
                    "required": ["epic"],
                },
            },
        ]

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------
    def get_system_prompt_section(self) -> Optional[str]:
        return (
            "# Toup Skill — Software Engineering Tools\n"
            "You have access to the Toup software engineering skill with these tools:\n"
            "- `toup__create_spec` — Turn a feature idea into a structured product spec\n"
            "- `toup__scaffold` — Generate a project scaffold (fastapi, nextjs, fullstack)\n"
            "- `toup__changeset` — Plan code changes as a structured changeset\n"
            "- `toup__review_diff` — Review a code diff for bugs, security, style\n"
            "- `toup__plan_sprint` — Break an epic into sprint tasks\n\n"
            "Use these tools proactively when the user asks about building software, "
            "planning features, reviewing code, or scaffolding projects."
        )

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------
    async def execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        ctx: SkillContext,
    ) -> str:
        dispatch = {
            "toup__create_spec": self._create_spec,
            "toup__scaffold": self._scaffold,
            "toup__changeset": self._changeset,
            "toup__review_diff": self._review_diff,
            "toup__plan_sprint": self._plan_sprint,
        }
        handler = dispatch.get(tool_name)
        if not handler:
            return f"ERROR: Unknown toup tool: {tool_name}"
        return await handler(args, ctx)

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    async def _create_spec(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        feature = args.get("feature", "").strip()
        context = args.get("context", "")

        if not feature:
            return "ERROR: 'feature' is required"

        now = datetime.utcnow().strftime("%Y-%m-%d")

        spec = f"""# Product Specification

## Feature
{feature}

## Date
{now}

---

## 1. Overview
{feature}

{f"### Additional Context{chr(10)}{context}" if context else ""}

## 2. Goals
- Deliver a working implementation of: {feature}
- Ensure production-quality code with tests
- Provide clear documentation

## 3. User Stories
- As a user, I want to {feature.lower()}, so that I can achieve the intended outcome.
- As a developer, I want clean, modular code, so that the feature is maintainable.
- As a stakeholder, I want the feature delivered on time with quality assurance.

## 4. Acceptance Criteria
- [ ] Feature works as described in the overview
- [ ] Unit tests pass with >80% coverage on new code
- [ ] No regressions in existing functionality
- [ ] Code reviewed and approved
- [ ] Documentation updated

## 5. Technical Approach
- Identify affected components and data models
- Implement backend logic with proper error handling
- Add API endpoints if needed
- Build frontend UI components if applicable
- Write unit and integration tests
- Update configuration and environment variables

## 6. Milestones
| # | Milestone | Target |
|---|-----------|--------|
| 1 | Spec finalized | {now} |
| 2 | Backend implementation | +3 days |
| 3 | Frontend integration | +5 days |
| 4 | Testing & QA | +7 days |
| 5 | Deploy to staging | +8 days |
| 6 | Production release | +10 days |

## 7. Risks & Mitigations
- **Scope creep** — Stick to defined acceptance criteria
- **Technical unknowns** — Spike/prototype early
- **Integration issues** — Test with mocks, then real services

## 8. Open Questions
- [ ] Are there existing APIs or services to integrate with?
- [ ] What is the expected load / scale?
- [ ] Are there specific UI/UX requirements?
"""

        # Save to workspace
        spec_dir = os.path.join(ctx.workspace, "specs")
        os.makedirs(spec_dir, exist_ok=True)
        slug = feature[:40].lower().replace(" ", "-").replace("/", "-")
        slug = "".join(c for c in slug if c.isalnum() or c == "-")
        filepath = os.path.join(spec_dir, f"spec-{slug}-{now}.md")
        with open(filepath, "w") as f:
            f.write(spec)

        return f"Spec created: {filepath}\n\n{spec}"

    async def _scaffold(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        name = args.get("name", "").strip()
        template = args.get("template", "").strip().lower()
        description = args.get("description", f"A {template} project")

        if not name:
            return "ERROR: 'name' is required"
        if template not in ("fastapi", "nextjs", "fullstack"):
            return f"ERROR: Unknown template '{template}'. Use: fastapi, nextjs, fullstack"

        # Select scaffold
        scaffold_map = {
            "fastapi": FASTAPI_SCAFFOLD,
            "nextjs": NEXTJS_SCAFFOLD,
            "fullstack": FULLSTACK_SCAFFOLD,
        }
        scaffold = scaffold_map[template]

        # Write files
        project_dir = os.path.join(ctx.workspace, name)
        files_created: List[str] = []

        for rel_path, content in scaffold.items():
            # Template substitution
            rendered = content.replace("{name}", name).replace("{description}", description)
            full_path = os.path.join(project_dir, rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(rendered)
            files_created.append(rel_path)

        summary = (
            f"Project '{name}' scaffolded with template '{template}'\n"
            f"Directory: {project_dir}\n"
            f"Files created ({len(files_created)}):\n"
        )
        for fp in sorted(files_created):
            summary += f"  • {fp}\n"

        return summary

    async def _changeset(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        description = args.get("description", "").strip()
        project_dir = args.get("project_dir", "")
        files_context = args.get("files_context", "")

        if not description:
            return "ERROR: 'description' is required"

        # Resolve project directory
        if project_dir:
            if not os.path.isabs(project_dir):
                project_dir = os.path.join(ctx.workspace, project_dir)
        else:
            project_dir = ctx.workspace

        changeset = {
            "description": description,
            "project_dir": project_dir,
            "timestamp": datetime.utcnow().isoformat(),
            "changes": [],
        }

        # Analyze the description and create a structured changeset
        changeset["changes"] = self._analyze_changes(description, files_context)

        # Save changeset to file
        changeset_dir = os.path.join(ctx.workspace, "changesets")
        os.makedirs(changeset_dir, exist_ok=True)
        slug = description[:30].lower().replace(" ", "-")
        slug = "".join(c for c in slug if c.isalnum() or c == "-")
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        filepath = os.path.join(changeset_dir, f"changeset-{slug}-{ts}.json")

        with open(filepath, "w") as f:
            json.dump(changeset, f, indent=2)

        # Format output
        lines = [f"Changeset saved: {filepath}\n"]
        lines.append(f"Description: {description}\n")
        lines.append(f"Changes ({len(changeset['changes'])}):")
        for i, ch in enumerate(changeset["changes"], 1):
            action = ch.get("action", "modify")
            path = ch.get("path", "unknown")
            reason = ch.get("reason", "")
            lines.append(f"  {i}. [{action.upper()}] {path}")
            if reason:
                lines.append(f"     Reason: {reason}")

        return "\n".join(lines)

    def _analyze_changes(self, description: str, files_context: str) -> List[Dict[str, str]]:
        """Parse the description into a list of file changes."""
        changes: List[Dict[str, str]] = []
        desc_lower = description.lower()

        # Heuristic analysis based on keywords
        if any(k in desc_lower for k in ["api", "endpoint", "route"]):
            changes.append({
                "action": "modify",
                "path": "app/api/routes.py",
                "reason": "Add/modify API endpoint",
            })

        if any(k in desc_lower for k in ["model", "database", "table", "schema"]):
            changes.append({
                "action": "modify",
                "path": "app/models.py",
                "reason": "Add/modify database model",
            })

        if any(k in desc_lower for k in ["test", "spec"]):
            changes.append({
                "action": "create",
                "path": "tests/test_new_feature.py",
                "reason": "Add tests for new feature",
            })

        if any(k in desc_lower for k in ["config", "setting", "env"]):
            changes.append({
                "action": "modify",
                "path": "app/config.py",
                "reason": "Update configuration",
            })

        if any(k in desc_lower for k in ["frontend", "component", "page", "ui"]):
            changes.append({
                "action": "create",
                "path": "src/components/NewComponent.tsx",
                "reason": "Add frontend component",
            })

        if any(k in desc_lower for k in ["migration", "migrate"]):
            changes.append({
                "action": "create",
                "path": "migrations/new_migration.py",
                "reason": "Database migration",
            })

        if not changes:
            changes.append({
                "action": "modify",
                "path": "(to be determined)",
                "reason": description[:100],
            })

        return changes

    async def _review_diff(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        diff = args.get("diff", "").strip()
        context = args.get("context", "")

        if not diff:
            return "ERROR: 'diff' is required"

        # Parse diff statistics
        added = diff.count("\n+") - diff.count("\n+++")
        removed = diff.count("\n-") - diff.count("\n---")
        files_changed = diff.count("diff --git") or diff.count("---")

        issues: List[str] = []
        suggestions: List[str] = []

        # Static analysis checks
        lines = diff.split("\n")
        for i, line in enumerate(lines):
            stripped = line[1:] if line.startswith("+") and not line.startswith("+++") else ""

            # Security checks
            if any(p in stripped.lower() for p in ["password", "secret", "api_key", "token"]):
                if "=" in stripped and not stripped.strip().startswith("#"):
                    issues.append(f"⚠️ Possible hardcoded secret at diff line {i + 1}: `{stripped.strip()[:60]}`")

            if "eval(" in stripped or "exec(" in stripped:
                issues.append(f"🔴 Dangerous eval/exec at diff line {i + 1}")

            if "import *" in stripped:
                issues.append(f"⚠️ Wildcard import at diff line {i + 1}: `{stripped.strip()}`")

            # Style checks
            if len(stripped) > 120:
                suggestions.append(f"Line {i + 1}: Line exceeds 120 chars ({len(stripped)} chars)")

            if "TODO" in stripped or "FIXME" in stripped or "HACK" in stripped:
                suggestions.append(f"Line {i + 1}: Contains TODO/FIXME marker: `{stripped.strip()[:60]}`")

            if "print(" in stripped and not stripped.strip().startswith("#"):
                suggestions.append(f"Line {i + 1}: Consider using logger instead of print()")

        # Build review
        review = [
            "# Code Review\n",
            f"## Stats",
            f"- Files changed: ~{max(files_changed, 1)}",
            f"- Lines added: +{added}",
            f"- Lines removed: -{removed}\n",
        ]

        if context:
            review.append(f"## Context\n{context}\n")

        if issues:
            review.append("## Issues Found")
            for issue in issues:
                review.append(f"- {issue}")
            review.append("")

        if suggestions:
            review.append("## Suggestions")
            for s in suggestions[:10]:
                review.append(f"- {s}")
            review.append("")

        if not issues and not suggestions:
            review.append("## Verdict\n✅ No issues found. LGTM!\n")
        elif issues:
            review.append("## Verdict\n🔴 Issues found — please address before merging.\n")
        else:
            review.append("## Verdict\n🟡 Minor suggestions — can merge with optional improvements.\n")

        return "\n".join(review)

    async def _plan_sprint(self, args: Dict[str, Any], ctx: SkillContext) -> str:
        epic = args.get("epic", "").strip()
        sprint_days = int(args.get("sprint_days", 14))
        team_size = int(args.get("team_size", 1))

        if not epic:
            return "ERROR: 'epic' is required"

        total_capacity_hours = sprint_days * team_size * 6  # ~6 productive hours/day
        now = datetime.utcnow()
        sprint_end = now + timedelta(days=sprint_days)

        # Break down the epic into phases
        phases = [
            ("Planning & Design", 0.10),
            ("Backend / Core Implementation", 0.35),
            ("Frontend / Integration", 0.25),
            ("Testing & QA", 0.15),
            ("Documentation & Deployment", 0.10),
            ("Buffer / Bug Fixes", 0.05),
        ]

        plan = [
            f"# Sprint Plan\n",
            f"## Epic\n{epic}\n",
            f"## Sprint Details",
            f"- Duration: {sprint_days} days ({now.strftime('%Y-%m-%d')} → {sprint_end.strftime('%Y-%m-%d')})",
            f"- Team size: {team_size} developer(s)",
            f"- Capacity: ~{total_capacity_hours} hours\n",
            "## Task Breakdown\n",
            "| # | Phase | Hours | Priority | Status |",
            "|---|-------|-------|----------|--------|",
        ]

        tasks: List[Dict[str, Any]] = []
        for i, (phase, pct) in enumerate(phases, 1):
            hours = round(total_capacity_hours * pct)
            priority = "P0" if i <= 2 else "P1" if i <= 4 else "P2"
            plan.append(f"| {i} | {phase} | {hours}h | {priority} | ⬜ |")
            tasks.append({"phase": phase, "hours": hours, "priority": priority})

        plan.append("")
        plan.append("## Detailed Tasks\n")

        # Generate detailed subtasks for each phase
        subtask_templates = {
            "Planning & Design": [
                "Define requirements and acceptance criteria",
                "Create technical design document",
                "Review and finalize architecture decisions",
            ],
            "Backend / Core Implementation": [
                "Set up data models and database schema",
                "Implement core business logic",
                "Build API endpoints",
                "Add input validation and error handling",
            ],
            "Frontend / Integration": [
                "Build UI components",
                "Integrate with backend APIs",
                "Add loading states and error handling",
                "Implement responsive design",
            ],
            "Testing & QA": [
                "Write unit tests",
                "Write integration tests",
                "Manual testing and bug fixes",
            ],
            "Documentation & Deployment": [
                "Update README and API docs",
                "Configure CI/CD pipeline",
                "Deploy to staging and verify",
            ],
            "Buffer / Bug Fixes": [
                "Address review feedback",
                "Fix any remaining bugs",
            ],
        }

        task_num = 1
        for phase_data in tasks:
            phase = phase_data["phase"]
            plan.append(f"### {phase} ({phase_data['hours']}h)")
            subtasks = subtask_templates.get(phase, ["Implement as needed"])
            for st in subtasks:
                plan.append(f"- [ ] {task_num}. {st}")
                task_num += 1
            plan.append("")

        plan.append("## Definition of Done")
        plan.append("- [ ] All tasks completed")
        plan.append("- [ ] Tests passing")
        plan.append("- [ ] Code reviewed")
        plan.append("- [ ] Documentation updated")
        plan.append("- [ ] Deployed to staging")
        plan.append("")

        result = "\n".join(plan)

        # Save plan to workspace
        plans_dir = os.path.join(ctx.workspace, "plans")
        os.makedirs(plans_dir, exist_ok=True)
        slug = epic[:30].lower().replace(" ", "-")
        slug = "".join(c for c in slug if c.isalnum() or c == "-")
        filepath = os.path.join(plans_dir, f"sprint-{slug}-{now.strftime('%Y%m%d')}.md")
        with open(filepath, "w") as f:
            f.write(result)

        return f"Sprint plan saved: {filepath}\n\n{result}"
