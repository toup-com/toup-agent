"""
Structured Logging — Per-subsystem structured logging with JSON output.

Provides contextual logging with subsystem tags, request correlation IDs,
and configurable log levels per subsystem.
"""

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# Context variable for request correlation
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
user_id_var: ContextVar[str] = ContextVar("user_id", default="")
session_id_var: ContextVar[str] = ContextVar("session_id", default="")


class Subsystem(str, Enum):
    AGENT = "agent"
    MEMORY = "memory"
    CHANNEL = "channel"
    TOOL = "tool"
    API = "api"
    DB = "db"
    SCHEDULER = "scheduler"
    AUTH = "auth"
    WEBHOOK = "webhook"
    VOICE = "voice"
    CANVAS = "canvas"
    SANDBOX = "sandbox"
    EVENT_BUS = "event_bus"
    QUEUE = "queue"


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "subsystem": getattr(record, "subsystem", "general"),
            "message": record.getMessage(),
            "logger": record.name,
        }

        # Add context vars if set
        req_id = request_id_var.get("")
        if req_id:
            log_entry["request_id"] = req_id
        usr_id = user_id_var.get("")
        if usr_id:
            log_entry["user_id"] = usr_id
        sess_id = session_id_var.get("")
        if sess_id:
            log_entry["session_id"] = sess_id

        # Add extra fields
        extra = getattr(record, "extra_data", None)
        if extra:
            log_entry["data"] = extra

        # Add exception info
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }

        return json.dumps(log_entry)


class SubsystemLogger:
    """Logger wrapper that adds subsystem context."""

    def __init__(self, subsystem: Subsystem, logger: logging.Logger):
        self._subsystem = subsystem
        self._logger = logger

    def _log(self, level: int, msg: str, extra_data: Any = None, **kwargs):
        extra = {"subsystem": self._subsystem.value}
        if extra_data:
            extra["extra_data"] = extra_data
        self._logger.log(level, msg, extra=extra, **kwargs)

    def debug(self, msg: str, data: Any = None, **kwargs):
        self._log(logging.DEBUG, msg, data, **kwargs)

    def info(self, msg: str, data: Any = None, **kwargs):
        self._log(logging.INFO, msg, data, **kwargs)

    def warning(self, msg: str, data: Any = None, **kwargs):
        self._log(logging.WARNING, msg, data, **kwargs)

    def error(self, msg: str, data: Any = None, **kwargs):
        self._log(logging.ERROR, msg, data, **kwargs)

    def critical(self, msg: str, data: Any = None, **kwargs):
        self._log(logging.CRITICAL, msg, data, **kwargs)


# ── Logger Registry ──
_loggers: Dict[str, SubsystemLogger] = {}
_structured_enabled = False


def get_subsystem_logger(subsystem: Subsystem) -> SubsystemLogger:
    """Get a structured logger for a subsystem."""
    key = subsystem.value
    if key not in _loggers:
        logger = logging.getLogger(f"hexbrain.{key}")
        _loggers[key] = SubsystemLogger(subsystem, logger)
    return _loggers[key]


def enable_structured_logging(level: int = logging.INFO):
    """Enable JSON structured logging globally."""
    global _structured_enabled
    if _structured_enabled:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())

    root = logging.getLogger("hexbrain")
    root.addHandler(handler)
    root.setLevel(level)
    _structured_enabled = True


def set_request_context(request_id: str = "", user_id: str = "", session_id: str = ""):
    """Set context variables for the current request."""
    if request_id:
        request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)
    if session_id:
        session_id_var.set(session_id)


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())[:12]


# ── Convenience loggers ──
agent_log = get_subsystem_logger(Subsystem.AGENT)
memory_log = get_subsystem_logger(Subsystem.MEMORY)
channel_log = get_subsystem_logger(Subsystem.CHANNEL)
tool_log = get_subsystem_logger(Subsystem.TOOL)
api_log = get_subsystem_logger(Subsystem.API)
db_log = get_subsystem_logger(Subsystem.DB)
