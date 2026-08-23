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
            # BOTH skills, and the second one is the bigger half. The
            # `app_builder` skill declares 6 tools (`app_builder__*`);
            # `AppGatewaySkill` — same directory, registered under the
            # name "app" — declares 13 more (`app__*`: read_file,
            # write_file, git_push, query_db, logs, restart, publish_web…).
            #
            # Listing only "app_builder" made a family labelled "the app
            # builder" withhold 6 of the app builder's 19 tools, so a
            # tenant who set AGENT_TOOL_FAMILIES to exclude it lost the
            # entry points and kept the entire machine behind them —
            # ~2/3 of the tokens, and 13 tools the agent could still call
            # with no way to have built anything to call them on.
            #
            # Not a token-accounting nicety: with four connectors added
            # in the last two days the wire array is over OpenAI's hard
            # 128-tool cap, where the overflow is TRUNCATED, so the tools
            # this family fails to withhold are paid for by dropping
            # whichever tools sort last.
            #
            # `app_html` (round 12) is the single-file HTML artifact
            # pipeline that replaces the Expo one. It belongs to the SAME
            # family for the same reason the two Expo skills do: a tenant
            # who withholds "the app builder" means "I cannot build apps",
            # not "I keep whichever half is newer". It is 5 tool defs
            # against the Expo path's 19, so the family also gets ~3x
            # cheaper once `app_builder_expo_enabled` is off.
            skills=("app_builder", "app", "app_html"),
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


# ── Pipeline selection (round 12) ─────────────────────────────────────
# Orthogonal to the family gate above. A family answers "may this tenant
# build apps at all?"; this answers "with WHICH pipeline?" — the legacy
# Expo one (`app_builder` + `app`), the single-file HTML one (`app_html`),
# or both during the canary.
#
# Same container-stability rule: resolved once, no per-turn input, flipping
# it is a restart. A pipeline that appeared or vanished mid-life would fork
# the provider cache lineage exactly like a family that did.
_EXPO_PIPELINE_SKILLS: FrozenSet[str] = frozenset({"app_builder", "app"})
_HTML_PIPELINE_SKILLS: FrozenSet[str] = frozenset({"app_html"})

# ── The Expo pipeline is RETIRED IN CODE (2026-08-21) ────────────────
# Not "off by default" — unreachable. `pipeline_enabled("expo")` returns
# False in every process, and no environment variable, settings value,
# `AGENT_TOOL_FAMILIES` string or bridge env pin can make it return True.
#
# Why a constant and not a default
# --------------------------------
# A default is not a live value. The 2026-08-21 default flip
# (`app_builder_expo_enabled: bool = False`) left THREE ways for a
# container to go on building Expo apps:
#
#   1. the canary set `APP_BUILDER_EXPO_ENABLED=1` explicitly on the
#      bridge env, and `pool_addon._FEATURE_FLAG_ENVS` forwarded it into
#      every container it spawned — a forwarded value outranks a default;
#   2. a container built from an image predating the flip has the old
#      `= True` compiled in, and keeps it until it is recreated;
#   3. `AGENT_TOOL_FAMILIES` gates the FAMILY, not the pipeline, so it
#      could not be used to close the gap either.
#
# Telling those three apart takes a shell on the box, and that box's SSH
# key was rotated on 2026-08-20. So the gate stops being a value to look
# up and becomes a fact about the build, asserted right here.
#
# The Expo code is deliberately NOT deleted — `app_builder/skill.py`,
# `app_gateway_skill.py`, `app_manager.py` and the 8-phase build_jobs
# pipeline all still exist, and `/api/apps` still serves apps that were
# built with them. What is gone is every route that STARTS one. Bringing
# it back is a code change and a review, which is the right price for a
# path that costs 452 MiB and 27k inodes per app.
EXPO_PIPELINE_RETIRED: bool = True

#: Skills that can never register, on any tenant, under any configuration.
#: Checked first in `skill_enabled`, which `SkillLoader._register` funnels
#: BOTH filesystem discovery and `register_dynamic()` through — so neither
#: the builtins scan nor agent_main's late-bound construction can slip one
#: in past the pipeline gate.
RETIRED_SKILLS: FrozenSet[str] = (
    _EXPO_PIPELINE_SKILLS if EXPO_PIPELINE_RETIRED else frozenset()
)

_PIPELINES: Optional[Dict[str, bool]] = None


def _resolved_pipelines() -> Dict[str, bool]:
    global _PIPELINES
    if _PIPELINES is None:
        try:
            from app.config import settings
            expo = bool(getattr(settings, "app_builder_expo_enabled", True))
            html = bool(getattr(settings, "app_html_enabled", True))
        except Exception:  # pragma: no cover - settings must never break boot
            expo, html = True, True
        if EXPO_PIPELINE_RETIRED:
            if expo:
                # Say so out loud. A container whose bridge env still pins
                # `APP_BUILDER_EXPO_ENABLED=1` is not misconfigured any
                # more — it is carrying a dead pin — and this is the line
                # that tells a rollout investigation which ones still have
                # it without needing a shell on the host.
                logger.warning(
                    "[entitlements] ignoring APP_BUILDER_EXPO_ENABLED=1 — the "
                    "Expo app pipeline is retired in code "
                    "(EXPO_PIPELINE_RETIRED). Apps build as single-file HTML "
                    "artifacts. The pin can be cleared from the bridge env; "
                    "it no longer does anything.",
                )
            expo = False
        _PIPELINES = {"expo": expo, "html": html}
        logger.info(
            "[entitlements] app pipelines — expo=%s html=%s", expo, html,
        )
    return _PIPELINES


def pipeline_enabled(name: str) -> bool:
    """`name` \u2208 {"expo", "html"}. Unknown names are enabled."""
    return _resolved_pipelines().get(name, True)


def skill_enabled(skill_name: str) -> bool:
    """True unless `skill_name` is RETIRED, or withheld by a family OR by
    pipeline selection."""
    # First, and unconditional: a retired skill has no configuration that
    # brings it back. Checked ahead of everything else so that no settings
    # value, env var or entitlement string is even consulted.
    if skill_name in RETIRED_SKILLS:
        return False
    # Automations (Round 26): dark-launch gate. The skill's tools head
    # the provider cache prefix like every skill's, so registration is
    # per-process and flag-resolved once at boot — a dark tenant's wire
    # array stays byte-identical to today's.
    if skill_name == "automations":
        from app.config import settings
        if not getattr(settings, "automations_enabled", False):
            return False
    if skill_name in _EXPO_PIPELINE_SKILLS and not pipeline_enabled("expo"):
        return False
    if skill_name in _HTML_PIPELINE_SKILLS and not pipeline_enabled("html"):
        return False
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
    global _RESOLVED, _PIPELINES
    _RESOLVED = None
    _PIPELINES = None
