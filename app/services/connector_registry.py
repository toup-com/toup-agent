"""
T1c — Connector registry.

Discovers connector subdirectories under `backend/app/connectors/`,
validates each one's manifest.yaml + provider.py, indexes the
resulting tools, and exposes a query API for the rest of the platform.

Loaded once at platform boot; cached for process lifetime. No
hot-reload — connectors are first-party and ship with the deploy.

Comparison with the existing skill loader (`app/agent/skills/loader.py`),
since the operator asked for prior-art alignment:

  Skill loader                       | Connector registry
  -----------------------------------|-----------------------------------
  Filesystem walk + dynamic exec     | Fixed module path
  (spec_from_file_location loads     | (importlib.import_module —
   ANY skill.py from external dirs)  |  first-party providers only)
  All-Python: meta + tools in class  | Split: YAML manifest + Python
  `on_load`/`on_unload` lifecycle    | Boot-load only, no hot-reload
  Hard raise on collision            | Skip + alarm + continue (per
                                     | architecture §9.3 — bad PR
                                     | cannot wedge platform startup)

What we copy:
  - `<id>__<tool>` tool naming convention; same regex.
  - `tool_index: dict[tool_name, owner_id]` for O(1) dispatch lookups.
  - `_register()`-shape: validate-then-add atomically per connector.
  - Logger on every load, summary on completion.

Lint rules at load time (each failure → log + alarm + skip the
manifest, never crash boot):
  - tool name regex: `^[a-z][a-z0-9]*__[a-z0-9_]+$`
  - manifest `id` matches the directory name
  - provider class `manifest_id` matches manifest `id`
  - `health.probe` matches one of the manifest's tool names
  - `oauth.provider_app` is in the known set (`google`, `github` for v1)
  - tool name unique within manifest
  - tool name not in built-in set (`tool_definitions.py`)
  - tool name not under reserved skill prefix (`<skill_dirname>__*`)
  - tool name not in another already-loaded manifest
"""

from __future__ import annotations

import importlib
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.connectors.base import BaseConnectorProvider

logger = logging.getLogger(__name__)


# ─── Tool-name regex (matches skill loader convention) ────────────────


TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9]*__[a-z0-9_]+$")


# ─── Provider apps known to the platform (v1) ────────────────────────
#
# This is intentionally a frozen set rather than a dynamic registry. T3a
# adds `google` config; T4a adds `github`. Stub providers in tests use
# `stub_provider_app`. New entries here = new ticket.

KNOWN_PROVIDER_APPS: frozenset[str] = frozenset({
    "google",
    "github",
    # 2026-05-11 — Outlook + LinkedIn connectors. Templates +
    # env-var registration live in `provider_apps.py`.
    "microsoft",
    "linkedin",
    "stub_provider_app",
})


# ─── Pydantic schema for manifest.yaml ───────────────────────────────


class ChannelPolicy(BaseModel):
    """Per-tool channel allow/deny policy (architecture §4.4)."""

    model_config = ConfigDict(extra="forbid")

    default: Literal["allow", "deny"] = "allow"
    deny: list[str] = Field(default_factory=list)


class OAuthSpec(BaseModel):
    """Per-connector OAuth shape."""

    model_config = ConfigDict(extra="forbid")

    provider_app: str
    scopes: list[str] = Field(default_factory=list)
    scopes_optional: list[str] = Field(default_factory=list)
    pkce: bool = True
    refresh: bool = True


class HealthSpec(BaseModel):
    """Health probe declaration. `probe` is the tool name to invoke
    cheaply for an authenticated round-trip — must reference one of
    the manifest's own tools (lint-checked at load)."""

    model_config = ConfigDict(extra="forbid")

    probe: str


class ConnectorTool(BaseModel):
    """One tool definition inside a manifest."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    input_schema: dict
    mutates: bool = False
    elevation: bool = False
    rate_limit: Optional[str] = None
    output_redaction: list[str] = Field(default_factory=list)
    channel_policy: ChannelPolicy = Field(default_factory=ChannelPolicy)
    pii: Optional[Literal["low", "medium", "high"]] = None


class ConnectorManifest(BaseModel):
    """Top-level manifest schema (architecture §2.1)."""

    model_config = ConfigDict(extra="forbid")

    manifest_version: int = 1
    id: str
    name: str
    short_description: str
    icon: Optional[str] = None
    docs_url: Optional[str] = None
    status: Literal["stable", "beta", "experimental"] = "experimental"
    category: str
    oauth: OAuthSpec
    health: HealthSpec
    tools: list[ConnectorTool]
    # Per-scope plain-language strings for the OAuth scope-review UI
    # (T2b). Map keys are full scope URIs (Google) or short names
    # (GitHub); values are user-readable. Falls back to the catalog
    # endpoint's DEFAULT_SCOPE_DESCRIPTIONS when missing.
    scope_descriptions: dict[str, str] = Field(default_factory=dict)


# ─── Registry entry ──────────────────────────────────────────────────


@dataclass(frozen=True)
class ConnectorEntry:
    """One registered connector — manifest + live provider instance."""

    manifest: ConnectorManifest
    provider: BaseConnectorProvider


@dataclass
class _ManifestRejection:
    """Why one specific manifest was skipped during load. Surfaced via
    `registry.alarms()` so ops can grep at boot."""

    connector_id: str
    reason: str


# ─── Registry ────────────────────────────────────────────────────────


class ConnectorRegistry:
    """In-memory registry of validated connectors.

    Boot the platform → call `load_all()` once → query via `get(id)`,
    `list_all()`, `is_connector_tool(name)`, `get_owner(name)`.
    """

    def __init__(self) -> None:
        self._entries: dict[str, ConnectorEntry] = {}
        self._tool_index: dict[str, str] = {}
        self._alarms: list[_ManifestRejection] = []

    # ── Public query API ──

    def get(self, connector_id: str) -> Optional[ConnectorEntry]:
        return self._entries.get(connector_id)

    def list_all(self) -> list[ConnectorEntry]:
        """All loaded connectors, ordered by id."""
        return [self._entries[k] for k in sorted(self._entries.keys())]

    def is_connector_tool(self, tool_name: str) -> bool:
        return tool_name in self._tool_index

    def get_owner(self, tool_name: str) -> Optional[str]:
        """connector_id of the manifest that registered this tool."""
        return self._tool_index.get(tool_name)

    def get_tool_spec(self, tool_name: str) -> Optional["ConnectorTool"]:
        """Return the manifest tool entry for `tool_name`, or None.

        Lets callers ask things like "is this a mutating tool?" without
        re-walking the manifest list. Used by the MCP tool filter to
        drop write-tools when the identity is in read-only mode.
        """
        owner = self._tool_index.get(tool_name)
        if owner is None:
            return None
        entry = self._entries.get(owner)
        if entry is None:
            return None
        for t in entry.manifest.tools:
            if t.name == tool_name:
                return t
        return None

    def alarms(self) -> list[_ManifestRejection]:
        """List of manifests rejected during the last `load_all()`.
        Empty on a clean boot."""
        return list(self._alarms)

    def summary(self) -> list[dict]:
        """For CLI dump / health-check endpoint. Read-only."""
        return [
            {
                "id": e.manifest.id,
                "name": e.manifest.name,
                "status": e.manifest.status,
                "tools": [t.name for t in e.manifest.tools],
                "scopes": e.manifest.oauth.scopes,
            }
            for e in self.list_all()
        ]

    # ── Load ──

    def load_all(
        self,
        connectors_dir: Optional[str] = None,
        *,
        include_experimental: bool = False,
    ) -> int:
        """Discover, validate, and register every connector subdir.

        Returns the number of connectors successfully loaded. Failed
        manifests go to `self._alarms` rather than raising — per
        architecture §9.3, a bad PR must NOT wedge platform startup.

        `include_experimental=False` (the default) drops manifests with
        `status: experimental` — that's how the test stub stays out of
        production.
        """
        # Reset (idempotent — re-load replaces the previous registry)
        self._entries.clear()
        self._tool_index.clear()
        self._alarms.clear()

        if connectors_dir is None:
            connectors_dir = str(Path(__file__).resolve().parent.parent / "connectors")

        if not os.path.isdir(connectors_dir):
            logger.info(
                "[connectors] No connectors dir at %s — empty registry",
                connectors_dir,
            )
            return 0

        # Pre-compute the cross-system collision sets ONCE per load.
        builtin_tool_names = _collect_builtin_tool_names()
        skill_prefixes = _collect_skill_prefixes()

        for entry in sorted(os.listdir(connectors_dir)):
            subdir = os.path.join(connectors_dir, entry)
            manifest_path = os.path.join(subdir, "manifest.yaml")
            provider_path = os.path.join(subdir, "provider.py")

            # Skip non-connector directories silently — `__init__.py` etc.
            if not os.path.isdir(subdir):
                continue
            if entry.startswith("__"):
                continue
            if not os.path.isfile(manifest_path):
                continue

            try:
                self._load_one(
                    entry,
                    manifest_path,
                    provider_path,
                    builtin_tool_names=builtin_tool_names,
                    skill_prefixes=skill_prefixes,
                    include_experimental=include_experimental,
                )
            except _SkipConnector as e:
                self._alarms.append(_ManifestRejection(
                    connector_id=entry, reason=str(e),
                ))
                logger.warning(
                    "[connectors] REJECTED %s: %s", entry, e,
                )
            except Exception as e:
                # Unexpected — don't let one bad provider crash boot.
                self._alarms.append(_ManifestRejection(
                    connector_id=entry,
                    reason=f"unexpected {type(e).__name__}: {e}",
                ))
                logger.exception(
                    "[connectors] REJECTED %s with unexpected error", entry,
                )

        logger.info(
            "[connectors] Loaded %d connector(s): %s. Rejected %d.",
            len(self._entries),
            sorted(self._entries.keys()),
            len(self._alarms),
        )
        return len(self._entries)

    # ── Internal: per-connector load + lint ──

    def _load_one(
        self,
        dir_name: str,
        manifest_path: str,
        provider_path: str,
        *,
        builtin_tool_names: set[str],
        skill_prefixes: set[str],
        include_experimental: bool,
    ) -> None:
        """Validate one connector and add it to `_entries` / `_tool_index`.

        Raises `_SkipConnector` on any lint failure — caller logs +
        alarms + moves on. NEVER raises for valid manifests.
        """
        # 1. Parse manifest.yaml.
        try:
            with open(manifest_path) as f:
                raw = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise _SkipConnector(f"manifest.yaml is not valid YAML: {e}")

        if not isinstance(raw, dict):
            raise _SkipConnector(
                f"manifest.yaml must be a YAML mapping, got {type(raw).__name__}"
            )

        # 2. Validate against the Pydantic schema.
        try:
            manifest = ConnectorManifest.model_validate(raw)
        except ValidationError as e:
            raise _SkipConnector(f"manifest schema validation failed: {e}")

        # 3. id ↔ directory name.
        if manifest.id != dir_name:
            raise _SkipConnector(
                f"manifest.id={manifest.id!r} does not match directory "
                f"name={dir_name!r}"
            )

        # 4. Filter experimental manifests in production.
        if manifest.status == "experimental" and not include_experimental:
            raise _SkipConnector(
                "manifest.status=experimental and include_experimental=False"
            )

        # 5. Validate oauth.provider_app is known.
        if manifest.oauth.provider_app not in KNOWN_PROVIDER_APPS:
            raise _SkipConnector(
                f"oauth.provider_app={manifest.oauth.provider_app!r} not in "
                f"KNOWN_PROVIDER_APPS={sorted(KNOWN_PROVIDER_APPS)}"
            )

        # 6. Each tool name must match the regex.
        for tool in manifest.tools:
            if not TOOL_NAME_RE.match(tool.name):
                raise _SkipConnector(
                    f"tool name {tool.name!r} does not match {TOOL_NAME_RE.pattern}"
                )

        # 7. Tool names must start with the manifest id prefix.
        prefix = f"{manifest.id}__"
        for tool in manifest.tools:
            if not tool.name.startswith(prefix):
                raise _SkipConnector(
                    f"tool name {tool.name!r} must start with {prefix!r}"
                )

        # 8. No duplicates within this manifest.
        seen: set[str] = set()
        for tool in manifest.tools:
            if tool.name in seen:
                raise _SkipConnector(f"duplicate tool name {tool.name!r}")
            seen.add(tool.name)

        # 9. health.probe must reference one of this manifest's own tools.
        own_tool_names = {t.name for t in manifest.tools}
        if manifest.health.probe not in own_tool_names:
            raise _SkipConnector(
                f"health.probe={manifest.health.probe!r} is not declared "
                f"among this manifest's tools ({sorted(own_tool_names)})"
            )

        # 10. Cross-system collision: built-ins, skills, other connectors.
        for tool in manifest.tools:
            if tool.name in builtin_tool_names:
                raise _SkipConnector(
                    f"tool name {tool.name!r} collides with built-in tool — "
                    f"built-ins always win"
                )
            tool_prefix = tool.name.split("__", 1)[0]
            if tool_prefix in skill_prefixes:
                raise _SkipConnector(
                    f"tool name {tool.name!r} uses prefix {tool_prefix!r} "
                    f"reserved by an existing skill — skills win over connectors"
                )
            if tool.name in self._tool_index:
                raise _SkipConnector(
                    f"tool name {tool.name!r} already registered by "
                    f"connector {self._tool_index[tool.name]!r}"
                )

        # 11. Import the provider module + locate the BaseConnectorProvider
        #     subclass. Raises if missing. Fixed module path — no arbitrary
        #     file loading (tighter than the skill loader).
        if not os.path.isfile(provider_path):
            raise _SkipConnector("provider.py is missing")

        try:
            provider_module = importlib.import_module(
                f"app.connectors.{dir_name}.provider",
            )
        except Exception as e:
            raise _SkipConnector(
                f"importing app.connectors.{dir_name}.provider failed: "
                f"{type(e).__name__}: {e}"
            )

        provider_cls = _find_provider_class(provider_module)
        if provider_cls is None:
            raise _SkipConnector(
                "no BaseConnectorProvider subclass exported from provider.py"
            )

        # 12. Provider's manifest_id ClassVar must match manifest.id.
        if getattr(provider_cls, "manifest_id", None) != manifest.id:
            raise _SkipConnector(
                f"provider class {provider_cls.__name__}.manifest_id="
                f"{getattr(provider_cls, 'manifest_id', None)!r} does not "
                f"match manifest.id={manifest.id!r}"
            )

        # 13. Instantiate. Provider __init__ takes no args (matches skill
        #     loader convention; per-call state lives in ConnectorContext).
        try:
            provider = provider_cls()
        except Exception as e:
            raise _SkipConnector(
                f"provider class {provider_cls.__name__} raised on instantiate: "
                f"{type(e).__name__}: {e}"
            )

        # ── All checks passed. Atomically register. ──
        connector_entry = ConnectorEntry(manifest=manifest, provider=provider)
        self._entries[manifest.id] = connector_entry
        for tool in manifest.tools:
            self._tool_index[tool.name] = manifest.id

        logger.info(
            "[connectors] Registered %r v%s (status=%s, %d tool(s): %s)",
            manifest.id,
            manifest.manifest_version,
            manifest.status,
            len(manifest.tools),
            [t.name for t in manifest.tools],
        )


# ─── Helpers ─────────────────────────────────────────────────────────


class _SkipConnector(Exception):
    """Internal sentinel — caller turns this into an alarm and moves on."""


def _find_provider_class(module: Any) -> Optional[type[BaseConnectorProvider]]:
    """Return the FIRST BaseConnectorProvider subclass exported by the
    module, excluding BaseConnectorProvider itself."""
    import inspect

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj is BaseConnectorProvider:
            continue
        if issubclass(obj, BaseConnectorProvider):
            return obj
    return None


def _collect_builtin_tool_names() -> set[str]:
    """Read tool names declared in `app.agent.tool_definitions`. The
    function calls return pure dicts — no execution side effects.

    Imported lazily (inside the function, not at module top level) so a
    test that doesn't need the registry doesn't pay the
    `tool_definitions` import cost."""
    try:
        from app.agent.tool_definitions import (
            get_agent_tools,
            get_extended_tools,
            get_doc_generation_tools,
            # `navigate_to` moved out of the doc-generation group (it is
            # navigation, not export). It must stay in the reserved-name set
            # or a connector could claim it.
            get_navigation_tools,
        )
    except ImportError as e:
        # Tool definitions might not import in some test contexts; treat
        # as empty rather than wedging the registry. CI runs the full
        # platform stack and will hit the real names.
        logger.warning(
            "[connectors] could not import tool_definitions for collision check: %s",
            e,
        )
        return set()

    names: set[str] = set()
    for fn in (
        get_agent_tools,
        get_extended_tools,
        get_doc_generation_tools,
        get_navigation_tools,
    ):
        try:
            for tool in fn():
                name = tool.get("name")
                if name:
                    names.add(name)
        except Exception:
            logger.exception(
                "[connectors] tool_definitions.%s raised during name extraction",
                fn.__name__,
            )
    return names


def _collect_skill_prefixes() -> set[str]:
    """Reserve `<skill_dirname>__*` prefixes for skills that ship with
    the platform.

    The platform process doesn't actually load skills (those run in
    agent containers), but the skill directories are part of the same
    codebase. Walking them for directory names is a lightweight way to
    forbid collisions without instantiating anything.

    Tighter exact-match would require running the skill loader at
    platform boot — much more code for marginal benefit. Agent-side
    dispatch (T1g) provides the second line of defense if a connector
    somehow registers a skill-prefix tool.
    """
    skills_dir = (
        Path(__file__).resolve().parent.parent
        / "agent" / "skills" / "builtins"
    )
    if not skills_dir.is_dir():
        return set()

    prefixes: set[str] = set()
    for entry in os.listdir(skills_dir):
        path = skills_dir / entry
        if not path.is_dir():
            continue
        if entry.startswith("__"):
            continue
        prefixes.add(entry)
    return prefixes


# ─── Module-level singleton ──────────────────────────────────────────


_registry: Optional[ConnectorRegistry] = None


def get_registry() -> ConnectorRegistry:
    """Return the process-wide registry singleton.

    Lazily instantiated. Callers that need the registry populated should
    call `get_registry().load_all()` at platform boot (typically in
    `platform_main.py` lifespan).
    """
    global _registry
    if _registry is None:
        _registry = ConnectorRegistry()
    return _registry


def reset_registry_for_tests() -> None:
    """Drop the module-level singleton. Tests only — production never
    needs to reset (registry is boot-loaded and immutable thereafter)."""
    global _registry
    _registry = None
