"""
AppManifest — Pydantic schema for agentSkill.json manifests.

Each generated app ships with a manifest at {app_dir}/lib/agentSkill.json
describing the domain-specific actions the agent can execute within the app
(SQL queries, mutations, screen navigations).

The manifest is an agent action catalog, NOT a file structure map or
customization guide. See docs/checkpoint-4a-skill-json-api.md for details.
"""

import json
import logging
import os
import re
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

# Regex for SQL :param_name placeholders
_SQL_PARAM_RE = re.compile(r":([a-zA-Z_][a-zA-Z0-9_]*)")


# ── Parameter definition ─────────────────────────────────────────

class ParamDefinition(BaseModel):
    """A single parameter accepted by an action."""
    model_config = ConfigDict(extra="allow")

    type: str  # "string", "integer", "number", "boolean"
    description: str = ""
    default: Optional[Any] = None


# ── Action types (discriminated union on `type` field) ───────────

class QueryAction(BaseModel):
    """SELECT action — reads data via SQL."""
    model_config = ConfigDict(extra="allow")

    name: str
    description: str
    type: Literal["query"]
    sql: str
    params: Dict[str, ParamDefinition] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_sql_params(self) -> "QueryAction":
        _check_sql_params(self.sql, self.params, self.name)
        return self


class MutationAction(BaseModel):
    """INSERT/UPDATE/DELETE action — modifies data via SQL."""
    model_config = ConfigDict(extra="allow")

    name: str
    description: str
    type: Literal["mutation"]
    sql: str
    params: Dict[str, ParamDefinition] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_sql_params(self) -> "MutationAction":
        _check_sql_params(self.sql, self.params, self.name)
        return self


class NavigateAction(BaseModel):
    """Screen navigation action."""
    model_config = ConfigDict(extra="allow")

    name: str
    description: str
    type: Literal["navigate"]
    screen: str
    params: Dict[str, ParamDefinition] = Field(default_factory=dict)


Action = Annotated[
    Union[QueryAction, MutationAction, NavigateAction],
    Field(discriminator="type"),
]


# ── Top-level manifest ───────────────────────────────────────────

class AppManifest(BaseModel):
    """Parsed agentSkill.json manifest for a generated app."""
    model_config = ConfigDict(extra="allow")

    domain: str
    description: str
    actions: List[Action]

    @classmethod
    def from_file(cls, path: str) -> "AppManifest":
        """Parse a manifest from a JSON file. Raises on any failure."""
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls.model_validate(raw)

    @classmethod
    def from_app_dir(cls, app_dir: str) -> "AppManifest":
        """Parse a manifest from an app's working directory.
        Expects the file at {app_dir}/lib/agentSkill.json."""
        path = os.path.join(app_dir, "lib", "agentSkill.json")
        return cls.from_file(path)


# ── SQL parameter validation helper ──────────────────────────────

def _check_sql_params(sql: str, params: Dict[str, ParamDefinition], action_name: str) -> None:
    """Validate that SQL :placeholders match the declared params dict."""
    sql_params = set(_SQL_PARAM_RE.findall(sql))
    declared_params = set(params.keys())

    undeclared = sql_params - declared_params
    unused = declared_params - sql_params

    errors = []
    if undeclared:
        errors.append(
            f"SQL parameter(s) {', '.join(':' + p for p in sorted(undeclared))} "
            f"used in SQL but not declared in params"
        )
    if unused:
        errors.append(
            f"Parameter(s) {', '.join(sorted(unused))} "
            f"declared in params but not used in SQL string"
        )

    if errors:
        raise ValueError(f"Action '{action_name}': {'; '.join(errors)}")


# ── Unknown field detection ──────────────────────────────────────

_KNOWN_TOP_FIELDS = {"domain", "description", "actions"}
_KNOWN_ACTION_FIELDS = {"name", "description", "type", "sql", "screen", "params"}
_KNOWN_PARAM_FIELDS = {"type", "description", "default"}


def detect_unknown_fields(raw: dict) -> List[str]:
    """Walk a raw manifest dict and find fields not in the known schema.
    Returns a list of dotpath strings for unknown fields."""
    unknown = []

    for key in raw:
        if key not in _KNOWN_TOP_FIELDS:
            unknown.append(key)

    for i, action in enumerate(raw.get("actions", [])):
        if not isinstance(action, dict):
            continue
        for key in action:
            if key not in _KNOWN_ACTION_FIELDS:
                unknown.append(f"actions[{i}].{key}")
        for pname, pdef in (action.get("params") or {}).items():
            if isinstance(pdef, dict):
                for key in pdef:
                    if key not in _KNOWN_PARAM_FIELDS:
                        unknown.append(f"actions[{i}].params.{pname}.{key}")

    return unknown
