"""Stable identities and freshness scopes for durable tool checkpoints.

Mutation outcomes are replay guards.  Eligible read outcomes are bounded
snapshots: useful within one observation boundary until a later operation can
have changed the resource they observed.  Live/polling reads are never cache
entries.  Keeping that distinction here lets both the runner's in-memory loop
and the durable voice ledger apply the same rule.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
from typing import Any, Callable, Optional


FilesystemPathResolver = Callable[[str, dict[str, Any]], Optional[str]]


_READ_ACTIONS = frozenset({
    "read", "read_file", "list", "list_files", "get", "search", "find",
    "fetch", "query", "lookup", "describe", "download", "export", "check",
    "status", "logs", "inspect", "view", "show", "analyze",
})
_READ_TOOLS = frozenset({
    "read_file", "ls", "grep", "find", "memory_search", "memory_read_file",
    "web_search", "web_fetch", "extension_search", "extension_read",
    "extension_research", "smart_fetch", "analyze_image", "sessions_list",
    "sessions_history", "session_status", "agents_list", "lanes_status",
    "doctor",
})
_FILE_READ_ACTIONS = frozenset({
    "read_file", "list_files", "ls", "grep", "find",
})
_FILE_WRITE_ACTIONS = frozenset({
    "write_file", "edit_file", "delete_file", "remove_file", "apply_patch",
})
_VOLATILE_READ_ACTIONS = frozenset({
    "status", "get_status", "check_status", "progress", "get_progress",
    "poll", "wait", "wait_for", "watch", "logs", "get_logs", "tail_logs",
})
_VOLATILE_READ_TOOLS = frozenset({
    "process", "sessions_list", "sessions_history", "session_status",
    "agents_list", "lanes_status", "doctor", "config_reload", "thread",
    "tts_prefs", "skill_marketplace", "talk_mode",
})
_NATIVE_ACTION_READS = {
    "process": frozenset({"status", "output", "list"}),
    "config_reload": frozenset({"list", "get"}),
    "thread": frozenset({"list"}),
    "tts_prefs": frozenset({"get"}),
    "skill_marketplace": frozenset({"search", "list_installed"}),
    "talk_mode": frozenset({"status"}),
}
_FRESH_READ_KEYS = frozenset({
    "fresh", "refresh", "force_refresh", "reload", "bypass_cache",
    "no_cache",
})
_CACHE_CONTROL_KEYS = frozenset({"cache", "use_cache", "allow_cache"})


def _action_name(tool_name: str) -> str:
    name = str(tool_name or "").strip().lower()
    if "__" in name:
        return name.rsplit("__", 1)[-1]
    return name


def tool_operation_kind(tool_name: str, tool_input: Any = None) -> str:
    """Return ``read`` or ``mutation`` using manifest truth when available.

    Unknown tools default to mutation.  Reusing an unknown read less often is
    preferable to replaying an unknown side effect.
    """
    name = str(tool_name or "").strip().lower()
    payload = tool_input if isinstance(tool_input, dict) else {}
    if name == "app__query_db" or name.endswith("__query_db"):
        return "read" if str(payload.get("type") or "").lower() == "query" else "mutation"
    if name in _NATIVE_ACTION_READS:
        action = str(payload.get("action") or "").strip().lower()
        return "read" if action in _NATIVE_ACTION_READS[name] else "mutation"
    if name in _READ_TOOLS:
        return "read"

    try:
        from app.services.connector_registry import get_registry

        spec = get_registry().get_tool_spec(str(tool_name or ""))
        if spec is not None:
            return "mutation" if bool(spec.mutates) else "read"
    except Exception:
        # Registry availability is optional in the isolated agent process and
        # in unit tests.  The conservative name fallback below is deterministic.
        pass

    action = _action_name(name)
    if action in _READ_ACTIONS or any(
        action.startswith(prefix + "_") for prefix in _READ_ACTIONS
    ):
        return "read"
    return "mutation"


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off", "none"}
    return bool(value)


def tool_operation_cacheable_read(tool_name: str, tool_input: Any = None) -> bool:
    """Whether an exact read may reuse one bounded snapshot.

    Read-only is a side-effect property, not a freshness promise.  Live status,
    progress, log, wait, and process/session observations must execute every
    time because the observed work can advance without a mutation in this
    runner.  Explicit cache-bypass arguments have the same effect.  Other reads
    may share an exact result only inside the runner's current model-iteration
    snapshot, or as a durable managed-task checkpoint after a crash/continuation.
    """
    name = str(tool_name or "").strip().lower()
    payload = tool_input if isinstance(tool_input, dict) else {}
    if tool_operation_kind(name, payload) != "read":
        return False
    action = _action_name(name)
    if (name in _VOLATILE_READ_TOOLS or action in _VOLATILE_READ_ACTIONS
            or "status" in action or "progress" in action
            or action.startswith(("poll", "wait", "watch"))
            or action.endswith(("_logs", "_events"))):
        return False
    lowered = {str(key).strip().lower(): value for key, value in payload.items()}
    if any(_truthy(lowered.get(key)) for key in _FRESH_READ_KEYS if key in lowered):
        return False
    if any(
        key in lowered and not _truthy(lowered[key])
        for key in _CACHE_CONTROL_KEYS
    ):
        return False
    if "cache_ttl" in lowered:
        try:
            if float(lowered["cache_ttl"]) <= 0:
                return False
        except (TypeError, ValueError):
            pass
    return True


def _tool_domain(tool_name: str, tool_input: dict[str, Any]) -> str:
    name = str(tool_name or "").strip().lower()
    action = _action_name(name)
    if name.startswith("memory_"):
        return "memory"
    if name in {"web_search", "web_fetch", "smart_fetch"} or name.startswith("extension_"):
        return "web"
    if name.startswith("browser"):
        return "browser"
    if (name in {"read_file", "write_file", "edit_file", "apply_patch", "ls", "grep", "find", "exec", "pty_exec", "process"}
            or action in _FILE_READ_ACTIONS or action in _FILE_WRITE_ACTIONS):
        if name.startswith("app__"):
            slug = str(tool_input.get("app_slug") or tool_input.get("slug") or "*").strip()
            return f"app:{slug}"
        if "__" in name and name.split("__", 1)[0].startswith("app_"):
            return f"app:{name.split('__', 1)[0]}"
        return "filesystem:workspace"
    if "__" in name:
        return f"connector:{name.split('__', 1)[0]}"
    # Connector test doubles and older manifests commonly use
    # ``gmail_send_email`` rather than ``gmail__send_email``.
    prefix = name.split("_", 1)[0]
    if prefix in {
        "gmail", "gcal", "calendar", "drive", "gdrive", "docs", "sheets",
        "slides", "outlook", "github", "linkedin", "notion", "linear",
        "slack", "stripe", "whatsapp", "telegram",
    }:
        return f"connector:{prefix}"
    return f"tool:{prefix or 'unknown'}"


def _normal_path(value: Any) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    return posixpath.normpath(raw) if raw else ""


def _filesystem_resource_path(
    tool_name: str,
    payload: dict[str, Any],
    resolver: Optional[FilesystemPathResolver],
) -> tuple[str, bool]:
    """Return the path spelling used by the native filesystem tools.

    Without an executor resolver the lexical value remains useful for older
    checkpoints and isolated tests.  A resolver failure means identity is
    unknown; callers then omit resource keys so a later mutation invalidates
    reads conservatively instead of claiming that two spellings are distinct.
    """
    raw = payload.get("path") or payload.get("file_path") or payload.get("filename")
    if resolver is None:
        path = _normal_path(raw)
        return path, bool(path)
    try:
        path = _normal_path(resolver(str(tool_name or ""), payload))
    except Exception:
        return "", False
    return path, bool(path)


def tool_operation_checkpoint(
    tool_name: str,
    tool_input: Any,
    *,
    filesystem_path_resolver: Optional[FilesystemPathResolver] = None,
) -> dict[str, Any]:
    """Describe the replay/freshness properties without retaining arguments."""
    payload = tool_input if isinstance(tool_input, dict) else {}
    name = str(tool_name or "")
    action = _action_name(name)
    domain = _tool_domain(name, payload)
    resources: set[str] = set()
    path_operation = (
        action in _FILE_READ_ACTIONS or action in _FILE_WRITE_ACTIONS
        or name.lower() in _FILE_READ_ACTIONS
        or name.lower() in _FILE_WRITE_ACTIONS
    )
    path, path_known = _filesystem_resource_path(
        name,
        payload,
        filesystem_path_resolver if domain == "filesystem:workspace" else None,
    )
    if path_known and path_operation:
        if action in {"list_files", "ls", "grep", "find"} or name.lower() in {"ls", "grep", "find"}:
            resources.add(f"tree:{domain}:{path}")
        else:
            resources.add(f"file:{domain}:{path}")
        if tool_operation_kind(name, payload) == "mutation":
            parent = posixpath.dirname(path) or "."
            # Directory reads of any ancestor become stale after a file write.
            for _ in range(32):
                resources.add(f"tree:{domain}:{parent}")
                if parent in {".", "/"}:
                    break
                next_parent = posixpath.dirname(parent)
                if not next_parent:
                    next_parent = "."
                if next_parent == parent:
                    break
                parent = next_parent
    return {
        "operation_kind": tool_operation_kind(name, payload),
        "resource_domains": [domain],
        "resource_keys": sorted(resources),
    }


def _resolved_read_resources(
    read_operation: dict[str, Any],
    resolver: Optional[FilesystemPathResolver],
) -> tuple[set[str], bool]:
    """Resolve stored lexical file keys through the current native workspace.

    Durable checkpoints intentionally omit full arguments.  Their resource
    keys retain enough path information to migrate old lexical checkpoints at
    comparison time, including relative/absolute and ``.``/``..`` aliases.
    """
    resources = set(read_operation.get("resource_keys") or [])
    if resolver is None or not resources:
        return resources, bool(resources)
    domains = set(read_operation.get("resource_domains") or [])
    filesystem_domains = {
        domain for domain in domains if domain == "filesystem:workspace"
    }
    if not filesystem_domains:
        return resources, True

    resolved: set[str] = set()
    tool_name = str(read_operation.get("tool_name") or "read_file")
    for resource in resources:
        matched = False
        for domain in filesystem_domains:
            for resource_type in ("file", "tree"):
                prefix = f"{resource_type}:{domain}:"
                if not str(resource).startswith(prefix):
                    continue
                matched = True
                stored_path = str(resource)[len(prefix):]
                try:
                    path = _normal_path(resolver(tool_name, {"path": stored_path}))
                except Exception:
                    return set(), False
                if not path:
                    return set(), False
                resolved.add(f"{resource_type}:{domain}:{path}")
                break
            if matched:
                break
        if not matched:
            # A key in the filesystem domain that cannot be interpreted is not
            # evidence that the read survived a write.
            if str(resource).startswith(("file:filesystem:", "tree:filesystem:")):
                return set(), False
            resolved.add(str(resource))
    return resolved, True


def resolved_operation_is_read(operation: Any) -> bool:
    if not isinstance(operation, dict):
        return False
    kind = str(operation.get("operation_kind") or "")
    if kind in {"read", "mutation"}:
        return kind == "read"
    return tool_operation_kind(str(operation.get("tool_name") or "")) == "read"


def mutation_invalidates_read(
    mutation_tool: str,
    mutation_input: Any,
    read_operation: Any,
    *,
    filesystem_path_resolver: Optional[FilesystemPathResolver] = None,
) -> bool:
    """Whether a successful write makes one stored read result stale."""
    if not resolved_operation_is_read(read_operation):
        return False
    mutation = tool_operation_checkpoint(
        mutation_tool,
        mutation_input,
        filesystem_path_resolver=filesystem_path_resolver,
    )
    if mutation["operation_kind"] != "mutation":
        return False
    read_domains = set(read_operation.get("resource_domains") or [])
    if not read_domains:
        read_domains = {
            _tool_domain(str(read_operation.get("tool_name") or ""), {})
        }
    if not (set(mutation["resource_domains"]) & read_domains):
        return False
    mutation_resources = set(mutation.get("resource_keys") or [])
    read_resources, read_identity_known = _resolved_read_resources(
        read_operation, filesystem_path_resolver,
    )
    if not read_identity_known:
        return True
    if mutation_resources and read_resources:
        if mutation_resources & read_resources:
            return True
        if filesystem_path_resolver is None and "filesystem:workspace" in read_domains:
            # One old checkpoint may use an absolute path and the other a
            # relative path.  With no effective workspace there is no sound
            # basis for declaring them different resources.
            mutation_absolute = any(
                key.startswith(("file:filesystem:workspace:/", "tree:filesystem:workspace:/"))
                for key in mutation_resources
            )
            read_absolute = any(
                key.startswith(("file:filesystem:workspace:/", "tree:filesystem:workspace:/"))
                for key in read_resources
            )
            if mutation_absolute != read_absolute:
                return True
        return False
    # Older checkpoints lack resource keys.  Within the same resource domain,
    # discard the read rather than claim its evidence survived an unknown write.
    return True


def tool_operation_key(tool_name: str, tool_input: Any) -> str:
    """Hash one tool plus its canonical arguments without persisting secrets."""
    payload = json.dumps(
        [str(tool_name), tool_input if isinstance(tool_input, dict) else {}],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tool_result_is_confirmation(value: Any) -> bool:
    """A staged approval card is not an executed mutation checkpoint."""
    text = value if isinstance(value, str) else str(value)
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        return "[confirmation_required]" in text
    if isinstance(parsed, dict) and set(parsed) == {"result"}:
        parsed = parsed.get("result")
    return isinstance(parsed, dict) and parsed.get("kind") == "confirmation_required"
