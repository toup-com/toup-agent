"""Tenant-stable entitlement gate for optional tool families.

The problem
-----------
Every agent container ships the *same* tool array to every tenant, and a
sizeable slice of it is dead weight for most of them. Measured on the real
OpenAI chat wire (o200k_base, `_anthropic_tools_to_openai` + `json.dumps`,
delta against the full 95-def / 18,105-tok array):

    doc_generation tools (7 defs)      1,160 tok
    app_builder skill tools (6 defs)   1,036 tok
    toup skill tools (5 defs)            684 tok
    app_builder system-prompt section    547 tok
                                       ---------
                                       3,427 tok

Those bytes head the provider cache prefix, so a tenant who never asks for
a spreadsheet still pays for the schema on the first turn of every lineage
and carries it in every cached prefix thereafter.

THE CACHE INVARIANT — read before changing anything here
---------------------------------------------------------
The tools array serializes AHEAD of system+history in the prompt prefix.
**Any byte difference starts a separate provider cache lineage**, and every
turn in the new lineage re-bills the entire system+history tail at full
price. A gate that varies per turn therefore costs far more than the tokens
it saves — it is the exact defect class this remediation program exists to
close (see `[PERF] tools_array_changed` in `agent_runner.run()`, and
`tests/test_all_channels_one_lineage.py`).

So the gate here is a pure function of the TENANT:

  * its only input is one env-backed settings string, `AGENT_TOOL_FAMILIES`;
  * it is resolved ONCE per process and memoized (`_RESOLVED`), so even a
    hot-reload of `settings` cannot fork the array mid-life;
  * it takes no turn, message, channel, time or request argument — there is
    deliberately nowhere to pass one.

Changing a tenant's entitlement is a container restart, by design.

Default
-------
`AGENT_TOOL_FAMILIES` defaults to ``"*"`` = every family entitled = exactly
today's behaviour. Nothing changes for anyone on merge; the byte-identity
of the default array is pinned in
`tests/test_tool_family_entitlements.py::test_default_wire_array_is_byte_identical_to_today`.

Value grammar
-------------
    "*"                    every family (default; also the value used when
                           the var is unset or blank, so a bridge that
                           writes an empty string cannot silently strip a
                           tenant's tools)
    "doc_generation,toup"  exactly those families
    "none"                 no optional family (also "-" )

Unknown names are logged and ignored rather than raising — a typo in a
per-tenant env var must not stop a container from booting.
"""

from __future__ import annotations

import logging
from typing import Dict, FrozenSet, Optional

logger = logging.getLogger(__name__)


class _Family:
    """One optional tool family.

    `tool_names` are core tool definitions (from `tool_definitions.py`);
    `skills` are skill-loader skill names, whose tools are all prefixed
    ``<skill>__`` by the loader's own contract (`SkillLoader._register`).
    """

    __slots__ = ("id", "tool_names", "skills", "label")

    def __init__(self, id: str, label: str, tool_names=(), skills=()):
        self.id = id
        self.label = label
        self.tool_names: FrozenSet[str] = frozenset(tool_names)
        self.skills: FrozenSet[str] = frozenset(skills)


# NOTE `navigate_to` is deliberately NOT here. It lived inside
# `get_doc_generation_tools()` purely by accident of where it was appended,
# so gating that group used to take page navigation ("take me to my brain")
# down with it — 400 of the 1,560 tokens people attributed to document
# generation were navigation. It now has its own
# `tool_definitions.get_navigation_tools()` and ships unconditionally.
FAMILIES: Dict[str, _Family] = {
    f.id: f
    for f in (
        _Family(
            "doc_generation",
            "document generation (PDF / Word / Excel / PowerPoint export)",
            tool_names=(
                "generate_pdf",
                "generate_docx",
                "generate_xlsx",
                "generate_pptx",
                "generate_markdown",
                "convert_document",
                "generate_html_to_pdf",
            ),
        ),
        _Family(
            "app_builder",
            "the app builder",
            skills=("app_builder",),
        ),
        _Family(
            "toup",
            "the software-engineering toolkit (specs, scaffolds, reviews)",
            skills=("toup",),
        ),
    )
}

ALL_FAMILIES: FrozenSet[str] = frozenset(FAMILIES)

_ALL = "*"
_NONE_TOKENS = frozenset({"none", "-"})

# Resolved once per process. See the cache invariant above: this memo is
# what makes the gate container-stable rather than settings-stable.
_RESOLVED: Optional[FrozenSet[str]] = None


def _parse(raw: str) -> FrozenSet[str]:
    value = (raw or "").strip()
    if not value or value == _ALL:
        return ALL_FAMILIES
    if value.lower() in _NONE_TOKENS:
        return frozenset()
    wanted = {p.strip() for p in value.split(",") if p.strip()}
    unknown = sorted(wanted - ALL_FAMILIES)
    if unknown:
        logger.warning(
            "[entitlements] AGENT_TOOL_FAMILIES lists unknown families %s — "
            "ignored. Known: %s",
            unknown,
            sorted(ALL_FAMILIES),
        )
    return frozenset(wanted & ALL_FAMILIES)


def entitled_families() -> FrozenSet[str]:
    """The families this TENANT is entitled to, for the life of the process.

    Resolved on first call and memoized. Takes no arguments on purpose:
    there is no turn, channel or user to key on, because keying on any of
    them would fork the provider cache lineage (see the module docstring).
    """
    global _RESOLVED
    if _RESOLVED is None:
        try:
            from app.config import settings

            raw = getattr(settings, "agent_tool_families", _ALL)
        except Exception:  # pragma: no cover - settings must never break boot
            logger.warning(
                "[entitlements] settings unavailable — defaulting to all families",
                exc_info=True,
            )
            raw = _ALL
        _RESOLVED = _parse(raw if isinstance(raw, str) else _ALL)
        if _RESOLVED != ALL_FAMILIES:
            logger.info(
                "[entitlements] tool families for this tenant: %s (withheld: %s)",
                sorted(_RESOLVED) or "none",
                sorted(ALL_FAMILIES - _RESOLVED),
            )
    return _RESOLVED


def family_enabled(family_id: str) -> bool:
    """True when `family_id` is entitled. Unknown ids are treated as
    entitled — a family this build has never heard of must not vanish."""
    if family_id not in FAMILIES:
        return True
    return family_id in entitled_families()


def skill_enabled(skill_name: str) -> bool:
    """True unless `skill_name` belongs to a family that is withheld."""
    for fam in FAMILIES.values():
        if skill_name in fam.skills and fam.id not in entitled_families():
            return False
    return True


def family_for_tool(tool_name: str) -> Optional[str]:
    """The family id owning `tool_name`, or None if it is unconditional."""
    if not tool_name:
        return None
    for fam in FAMILIES.values():
        if tool_name in fam.tool_names:
            return fam.id
        for skill in fam.skills:
            if tool_name.startswith(f"{skill}__"):
                return fam.id
    return None


def refusal_for_tool(tool_name: str) -> Optional[str]:
    """A user-facing refusal when `tool_name` is withheld, else None.

    Used as the executor backstop. Withholding the *definition* keeps the
    model from picking the tool; this keeps a call that arrives some other
    way (a replayed tool_use block, an MCP alias, a hallucinated name) from
    reaching a handler, and gives the model wording that does not invent
    the capability.
    """
    fam_id = family_for_tool(tool_name)
    if fam_id is None or fam_id in entitled_families():
        return None
    return (
        f"ERROR: '{tool_name}' is not enabled on this account "
        f"({FAMILIES[fam_id].label} is not part of the current plan). "
        "Tell the user plainly that this capability is not enabled for them "
        "and offer what you CAN do instead — do not describe it as "
        "temporarily broken, and do not improvise a substitute."
    )


def reset_cache_for_tests() -> None:
    """Drop the memo. TEST ONLY.

    Production must never call this: re-resolving mid-life is precisely the
    turn-conditional array this module exists to prevent.
    """
    global _RESOLVED
    _RESOLVED = None
