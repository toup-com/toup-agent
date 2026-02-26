"""
CLI Doctor — System health checks and diagnostics.

Provides comprehensive health checks for the HexBrain platform:
  * Database connectivity and schema
  * API endpoint availability
  * Docker / sandbox status
  * Memory system health
  * Disk space and resource usage
  * Configuration validation
  * Dependency verification
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CheckStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class CheckResult:
    """Result of a single health check."""
    name: str
    status: CheckStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "duration_ms": round(self.duration_ms, 1),
        }


@dataclass
class DoctorReport:
    """Full health check report."""
    checks: List[CheckResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    platform_info: Dict[str, str] = field(default_factory=dict)

    @property
    def ok_count(self) -> int:
        return len([c for c in self.checks if c.status == CheckStatus.OK])

    @property
    def warning_count(self) -> int:
        return len([c for c in self.checks if c.status == CheckStatus.WARNING])

    @property
    def error_count(self) -> int:
        return len([c for c in self.checks if c.status == CheckStatus.ERROR])

    @property
    def overall_status(self) -> CheckStatus:
        if self.error_count > 0:
            return CheckStatus.ERROR
        if self.warning_count > 0:
            return CheckStatus.WARNING
        return CheckStatus.OK

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "summary": {
                "ok": self.ok_count,
                "warnings": self.warning_count,
                "errors": self.error_count,
                "total": len(self.checks),
            },
            "platform": self.platform_info,
            "checks": [c.to_dict() for c in self.checks],
            "timestamp": self.timestamp,
        }

    def to_text(self) -> str:
        """Format report as human-readable text."""
        icons = {
            CheckStatus.OK: "✅",
            CheckStatus.WARNING: "⚠️",
            CheckStatus.ERROR: "❌",
            CheckStatus.SKIPPED: "⏭️",
        }
        lines = [
            "╔══════════════════════════════════════════╗",
            "║       HexBrain Doctor Report             ║",
            "╚══════════════════════════════════════════╝",
            "",
        ]

        for check in self.checks:
            icon = icons[check.status]
            lines.append(f"  {icon} {check.name}: {check.message}")
            if check.details:
                for k, v in check.details.items():
                    lines.append(f"     └─ {k}: {v}")

        lines.extend([
            "",
            f"Summary: {self.ok_count} ok, {self.warning_count} warnings, {self.error_count} errors",
            f"Overall: {icons[self.overall_status]} {self.overall_status.value.upper()}",
        ])
        return "\n".join(lines)


async def run_doctor(include: Optional[List[str]] = None) -> DoctorReport:
    """
    Run all health checks and return a report.

    Args:
        include: Optional list of check names to run. If None, run all.
    """
    report = DoctorReport()
    report.platform_info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pid": str(os.getpid()),
    }

    all_checks = [
        ("python_deps", _check_python_deps),
        ("config", _check_config),
        ("database", _check_database),
        ("disk_space", _check_disk_space),
        ("docker", _check_docker),
        ("api_key_openai", _check_openai_key),
        ("api_key_anthropic", _check_anthropic_key),
        ("workspace", _check_workspace),
        ("memory_system", _check_memory_system),
        ("telegram_bot", _check_telegram),
        ("browser", _check_browser),
    ]

    for name, check_fn in all_checks:
        if include and name not in include:
            continue
        start = time.time()
        try:
            result = await check_fn()
            result.duration_ms = (time.time() - start) * 1000
            report.checks.append(result)
        except Exception as e:
            report.checks.append(CheckResult(
                name=name,
                status=CheckStatus.ERROR,
                message=f"Check crashed: {e}",
                duration_ms=(time.time() - start) * 1000,
            ))

    return report


# ──────────────────────────────────────────────────────────────
# Individual Checks
# ──────────────────────────────────────────────────────────────

async def _check_python_deps() -> CheckResult:
    """Check required Python packages."""
    required = [
        "fastapi", "uvicorn", "sqlalchemy", "httpx", "pydantic",
        "alembic", "bcrypt", "python-jose",
    ]
    optional = [
        "playwright", "edge_tts", "openai", "anthropic",
    ]

    missing_req = []
    missing_opt = []

    for pkg in required:
        try:
            importlib.import_module(pkg.replace("-", "_"))
        except ImportError:
            missing_req.append(pkg)

    for pkg in optional:
        try:
            importlib.import_module(pkg.replace("-", "_"))
        except ImportError:
            missing_opt.append(pkg)

    if missing_req:
        return CheckResult(
            name="python_deps",
            status=CheckStatus.ERROR,
            message=f"Missing required packages: {missing_req}",
            details={"missing_required": missing_req, "missing_optional": missing_opt},
        )
    elif missing_opt:
        return CheckResult(
            name="python_deps",
            status=CheckStatus.WARNING,
            message=f"Missing optional packages: {missing_opt}",
            details={"missing_optional": missing_opt},
        )
    return CheckResult(
        name="python_deps",
        status=CheckStatus.OK,
        message="All required packages installed",
    )


async def _check_config() -> CheckResult:
    """Validate configuration."""
    try:
        from app.config import settings
        issues = []

        if settings.jwt_secret == "hexbrain-dev-secret-change-in-production":
            issues.append("JWT secret is still the default — change in production!")

        if not settings.openai_api_key:
            issues.append("OPENAI_API_KEY not set")

        if settings.database_url.startswith("sqlite"):
            issues.append("Using SQLite — switch to PostgreSQL for production")

        if issues:
            return CheckResult(
                name="config",
                status=CheckStatus.WARNING,
                message=f"{len(issues)} config issues",
                details={"issues": issues},
            )
        return CheckResult(
            name="config",
            status=CheckStatus.OK,
            message="Configuration valid",
        )
    except Exception as e:
        return CheckResult(
            name="config",
            status=CheckStatus.ERROR,
            message=f"Config load failed: {e}",
        )


async def _check_database() -> CheckResult:
    """Test database connectivity."""
    try:
        from app.db.database import async_session_maker
        from sqlalchemy import text

        async with async_session_maker() as session:
            result = await session.execute(text("SELECT 1"))
            row = result.scalar()
            if row == 1:
                return CheckResult(
                    name="database",
                    status=CheckStatus.OK,
                    message="Database connected",
                )
        return CheckResult(
            name="database",
            status=CheckStatus.ERROR,
            message="Database query returned unexpected result",
        )
    except Exception as e:
        return CheckResult(
            name="database",
            status=CheckStatus.ERROR,
            message=f"Database connection failed: {e}",
        )


async def _check_disk_space() -> CheckResult:
    """Check available disk space."""
    try:
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024 ** 3)
        total_gb = total / (1024 ** 3)
        used_pct = (used / total) * 100

        if free_gb < 1:
            status = CheckStatus.ERROR
            msg = f"Critical: only {free_gb:.1f}GB free"
        elif free_gb < 5:
            status = CheckStatus.WARNING
            msg = f"Low disk space: {free_gb:.1f}GB free"
        else:
            status = CheckStatus.OK
            msg = f"{free_gb:.1f}GB free of {total_gb:.1f}GB"

        return CheckResult(
            name="disk_space",
            status=status,
            message=msg,
            details={"free_gb": round(free_gb, 1), "total_gb": round(total_gb, 1),
                      "used_pct": round(used_pct, 1)},
        )
    except Exception as e:
        return CheckResult(
            name="disk_space",
            status=CheckStatus.ERROR,
            message=f"Disk check failed: {e}",
        )


async def _check_docker() -> CheckResult:
    """Check Docker availability."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "info", "--format", "{{.ServerVersion}}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)

        if proc.returncode == 0:
            version = stdout.decode().strip()
            return CheckResult(
                name="docker",
                status=CheckStatus.OK,
                message=f"Docker {version} available",
                details={"version": version},
            )
        return CheckResult(
            name="docker",
            status=CheckStatus.WARNING,
            message="Docker not responding",
        )
    except FileNotFoundError:
        return CheckResult(
            name="docker",
            status=CheckStatus.WARNING,
            message="Docker CLI not found (sandbox features disabled)",
        )
    except Exception as e:
        return CheckResult(
            name="docker",
            status=CheckStatus.WARNING,
            message=f"Docker check failed: {e}",
        )


async def _check_openai_key() -> CheckResult:
    """Validate OpenAI API key."""
    try:
        from app.config import settings
        key = settings.openai_api_key
        if not key:
            return CheckResult(
                name="api_key_openai",
                status=CheckStatus.WARNING,
                message="OPENAI_API_KEY not configured",
            )
        if not key.startswith("sk-"):
            return CheckResult(
                name="api_key_openai",
                status=CheckStatus.WARNING,
                message="OPENAI_API_KEY format looks invalid",
            )
        return CheckResult(
            name="api_key_openai",
            status=CheckStatus.OK,
            message=f"OpenAI key configured ({key[:8]}...)",
        )
    except Exception as e:
        return CheckResult(
            name="api_key_openai",
            status=CheckStatus.ERROR,
            message=f"OpenAI key check failed: {e}",
        )


async def _check_anthropic_key() -> CheckResult:
    """Check Anthropic API key."""
    try:
        from app.config import settings
        key = settings.anthropic_api_key
        if not key:
            return CheckResult(
                name="api_key_anthropic",
                status=CheckStatus.OK,
                message="Anthropic key not configured (optional)",
            )
        return CheckResult(
            name="api_key_anthropic",
            status=CheckStatus.OK,
            message=f"Anthropic key configured ({key[:8]}...)",
        )
    except Exception as e:
        return CheckResult(
            name="api_key_anthropic",
            status=CheckStatus.ERROR,
            message=f"Anthropic key check failed: {e}",
        )


async def _check_workspace() -> CheckResult:
    """Check agent workspace directory."""
    try:
        from app.config import settings
        ws = settings.agent_workspace_dir

        if not os.path.isdir(ws):
            return CheckResult(
                name="workspace",
                status=CheckStatus.WARNING,
                message=f"Workspace directory missing: {ws}",
            )

        writable = os.access(ws, os.W_OK)
        if not writable:
            return CheckResult(
                name="workspace",
                status=CheckStatus.ERROR,
                message=f"Workspace not writable: {ws}",
            )

        # Count files
        file_count = sum(len(files) for _, _, files in os.walk(ws))
        return CheckResult(
            name="workspace",
            status=CheckStatus.OK,
            message=f"Workspace OK ({file_count} files)",
            details={"path": ws, "file_count": file_count},
        )
    except Exception as e:
        return CheckResult(
            name="workspace",
            status=CheckStatus.ERROR,
            message=f"Workspace check failed: {e}",
        )


async def _check_memory_system() -> CheckResult:
    """Check memory/embedding system."""
    try:
        from app.config import settings

        if "pgvector" in settings.database_url or "postgresql" in settings.database_url:
            return CheckResult(
                name="memory_system",
                status=CheckStatus.OK,
                message=f"pgvector + {settings.embedding_model}",
                details={
                    "provider": settings.embedding_provider,
                    "model": settings.embedding_model,
                    "dimension": settings.embedding_dimension,
                },
            )
        return CheckResult(
            name="memory_system",
            status=CheckStatus.WARNING,
            message="SQLite backend — pgvector recommended for production",
        )
    except Exception as e:
        return CheckResult(
            name="memory_system",
            status=CheckStatus.ERROR,
            message=f"Memory check failed: {e}",
        )


async def _check_telegram() -> CheckResult:
    """Check Telegram bot configuration."""
    try:
        from app.config import settings
        token = settings.telegram_bot_token
        if not token:
            return CheckResult(
                name="telegram_bot",
                status=CheckStatus.OK,
                message="Telegram bot not configured (optional)",
            )
        return CheckResult(
            name="telegram_bot",
            status=CheckStatus.OK,
            message="Telegram bot token configured",
        )
    except Exception as e:
        return CheckResult(
            name="telegram_bot",
            status=CheckStatus.ERROR,
            message=f"Telegram check failed: {e}",
        )


async def _check_browser() -> CheckResult:
    """Check Playwright browser availability."""
    try:
        import playwright
        return CheckResult(
            name="browser",
            status=CheckStatus.OK,
            message="Playwright installed",
        )
    except ImportError:
        return CheckResult(
            name="browser",
            status=CheckStatus.WARNING,
            message="Playwright not installed (browser tools disabled)",
        )


# ──────────────────────────────────────────────────────────────
# CLI Send — Send a message from the terminal
# ──────────────────────────────────────────────────────────────

async def cli_send(
    message: str,
    user_id: str = "cli_user",
    session_id: Optional[str] = None,
    base_url: str = "http://localhost:8000",
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Send a message to the agent from the CLI.

    Connects to the API and sends a message, returning the response.
    """
    import httpx

    url = f"{base_url}/api/chat"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload = {
        "message": message,
        "user_id": user_id,
    }
    if session_id:
        payload["session_id"] = session_id

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"API returned {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"error": f"Failed to connect: {e}"}
