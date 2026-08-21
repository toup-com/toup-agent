"""Turning a short instruction into a coherent scene — and one rule that is not
negotiable by anyone, including the user.

Two independent layers, and the separation is the point:

* `build_scene_prompt` EXPANDS. "Put me in the pool" has to become a scene:
  swimwear, waterline, wet skin, pool lighting, same face. This is a model call
  with its own system prompt, and it is allowed to fail — on any error the raw
  instruction goes through with a coherence suffix, because a broken expander
  must not stop someone editing a photograph.

* `check_clothing_guard` REFUSES. It runs on the FINAL string, after expansion,
  on every path, and it is pure regex — no model, no network, nothing that can
  be talked out of it. If the expander is jailbroken, prompt-injected, or simply
  wrong, this still holds.

Why the second layer exists at all
----------------------------------
This platform edits photographs of real, identifiable people that users upload,
and it cannot verify who is depicted in any given upload. An agent that decides
on its own that undressing a subject is how you fit them to a scene is a
machine for generating non-consensual intimate imagery, including of minors.
That risk does not depend on the request being malicious — "put me in the pool"
is an entirely innocent sentence, and a naive expander answering it with
"wearing nothing" would be the same incident. So the rule is enforced on the
constructed prompt rather than on the user's words.

The rule: never remove, reduce or omit clothing the subject is wearing in the
source. Where a setting implies different attire, SUBSTITUTE setting-appropriate
clothing — swim shorts for a pool, a coat for snow. Substitution is not removal:
swimwear is the correct, expected answer for a pool, and it is named explicitly
in the system prompt so the model reaches for it rather than improvising.
"""

from __future__ import annotations

import logging
import re
from typing import Awaitable, Callable, Optional, Tuple

logger = logging.getLogger(__name__)


# ── The constructor's system prompt ──────────────────────────────────────
SCENE_SYSTEM_PROMPT = """You expand a short image-editing instruction into a \
single coherent scene description for an image model.

The user names ONE change. Your job is to work out everything else that must \
change for the result to be internally consistent, and to state it explicitly. \
An instruction like "put me in the pool" is not satisfied by moving the subject \
— a person sitting in a pool in a t-shirt and a sun hat is a failed edit.

Cover, only where relevant:
- WARDROBE appropriate to the setting: swim shorts or a swimsuit for a pool, \
beach or sea; a coat, hat and gloves for snow; formalwear for a gala; athletic \
wear for a gym; appropriate dress for a place of worship.
- POSE and body mechanics for the activity: how a person actually sits, stands \
or floats, correct waterline at the body, plausible buoyancy and weight.
- LIGHTING, shadow direction and colour temperature matched to the environment.
- WET/DRY state, reflections, droplets and the way water meets skin and fabric.
- IDENTITY: the same face, the same build, the same skin tone, the same hair. \
This is a photograph of a real person and they must remain recognisably \
themselves.
- CONTINUITY with edits already made to this image, so instructions compound \
rather than reset.

ABSOLUTE RULE — this overrides every other instruction in this prompt, anything \
the user writes, and any reasoning you do about what a scene "needs":
You must NEVER remove, reduce or omit clothing the subject is wearing in the \
source image. You must NEVER decide that undressing, partially undressing, or \
showing less of a subject's body is the way to fit them to a scene. Where a \
setting implies different attire you SUBSTITUTE setting-appropriate clothing — \
for a pool that means swim shorts or a swimsuit, which is the correct and \
expected answer. If a request cannot be satisfied without reducing what the \
subject is wearing, describe them fully clothed in the most \
setting-appropriate outfit instead. Never describe a subject as nude, topless, \
undressed, in underwear or in lingerie, under any framing whatsoever.

Reply with the scene description only. No preamble, no explanation, no lists. \
Two to four sentences."""


# ── Layer 2: the guard ───────────────────────────────────────────────────
# Deliberately regex, deliberately independent of the model, deliberately run on
# the CONSTRUCTED prompt. Ordered from most explicit to most euphemistic.

_NUDITY_TERMS = r"""(?:
    nude | nudity | naked | unclothed | unclad |
    # Bare verb forms too: `_REMOVAL_RE` wants a GARMENT after the verb,
    # so "undress the woman" — object a person, not a shirt — slipped
    # through both tests. These have no innocent reading in an image prompt.
    undress(?:es|ed|ing)? | disrobe[sd]? | disrobing |
    topless | bottomless | shirtless(?:\s+(?:woman|girl|female))? |
    bare[-\s]?(?:chested|breasted|bottom|buttocks|skin\s+below) |
    in\s+(?:the\s+)?(?:nude|buff) | birthday\s+suit |
    without\s+(?:any\s+)?(?:clothes|clothing|garments|attire|covering) |
    no\s+(?:clothes|clothing|garments|top|shirt|bra|underwear) |
    lingerie | underwear\s+only | just\s+(?:her|his|their)\s+underwear |
    (?:see[-\s]?through|transparent|sheer)\s+(?:top|dress|shirt|clothing) |
    exposed\s+(?:breasts?|chest|genitals|nipples|buttocks) |
    nipples? | genitals? | cleavage
)"""

_REMOVAL_VERBS = r"""(?:
    remove[ds]? | removing | take[ns]?\s+off | taking\s+off | took\s+off |
    strip(?:s|ped|ping)? | undress(?:e[ds]|ing)? | peel(?:ed|ing)?\s+off |
    pull(?:ed|ing)?\s+(?:off|down) | unbutton(?:ed|ing)? |
    unzip(?:ped|ping)? | open(?:ed)?\s+up | lose | losing | lost |
    get\s+rid\s+of | discard(?:ed|ing)? | ditch(?:ed)? |
    less\s+of | reduce[ds]? | minimi[sz]e[ds]? | shorten(?:ed)? | shrink
)"""

_GARMENT_NOUNS = r"""(?:
    clothes | clothing | garments? | outfit | attire | top | tops | shirt |
    t[-\s]?shirt | blouse | dress | skirt | trousers | pants | shorts |
    jacket | coat | sweater | jumper | hoodie | bra | underwear | robe |
    towel | swimsuit | swimwear | bikini | covering
)"""

_NUDITY_RE = re.compile(_NUDITY_TERMS, re.IGNORECASE | re.VERBOSE)
_REMOVAL_RE = re.compile(
    rf"\b{_REMOVAL_VERBS}\b[^.]{{0,40}}?\b{_GARMENT_NOUNS}\b",
    re.IGNORECASE | re.VERBOSE,
)
# "her shirt off", "the dress removed" — object before verb.
_REMOVAL_REVERSED_RE = re.compile(
    rf"\b{_GARMENT_NOUNS}\b\s*(?:is|are|being|,)?\s*(?:{_REMOVAL_VERBS}|off)\b",
    re.IGNORECASE | re.VERBOSE,
)
# Coverage reduction phrased as a quantity.
_LESS_COVERAGE_RE = re.compile(
    r"\b(?:less|least|minimal|barely|scantily|revealing(?:ly)?|skimpy|"
    r"more\s+revealing|show(?:ing)?\s+more\s+(?:skin|body|leg|chest))\b"
    r"(?:[^.]{0,30}\b(?:clad|dressed|clothed|covering|coverage|outfit|clothing)\b)?",
    re.IGNORECASE,
)

# A minor anywhere near this subject matter is an immediate refusal — the
# combination is the one the guard exists for.
# Underwear worn AS the outfit is reduced coverage, whoever the subject is.
# Phrased around the wearing so that increasing coverage ("put a vest on
# under it") is untouched — nobody writes "add underwear" as an edit anyway.
_UNDERGARMENT_AS_OUTFIT_RE = re.compile(
    r"\b(?:in|wearing|down\s+to|only\s+in|just\s+in|clad\s+in|dressed\s+in)\b"
    r"[^.]{0,20}?\b(?:underwear|undies|panties|knickers|boxers|briefs|thong|"
    r"g[-\s]?string|bra|boxer\s+shorts)\b",
    re.IGNORECASE,
)

_MINOR_RE = re.compile(
    r"\b(?:child|children|kid|kids|baby|babies|toddler|infant|minor|minors|"
    r"teen|teens|teenager|teenaged|adolescent|schoolgirl|schoolboy|"
    r"underage|pre[-\s]?teen|little\s+(?:girl|boy))\b",
    re.IGNORECASE,
)

GUARD_REFUSAL = (
    "This edit would change what the person in the photo is wearing to something "
    "less covering, which this assistant will not do — it edits photographs of "
    "real people and cannot verify who is in them. Tell the user plainly that "
    "this particular edit is not something you can make, and do not retry it "
    "with different wording. If the setting calls for different clothes, offer "
    "the setting-appropriate version instead — swimwear for a pool, a coat for "
    "snow. Every other edit to the same photo still works."
)


def check_clothing_guard(prompt: str) -> Tuple[bool, Optional[str]]:
    """`(allowed, reason)` for a FINAL image prompt.

    Runs on every image path, after any expansion, and knows nothing about who
    produced the string. `reason` is written for the agent to relay, never
    raw-rendered to the user.
    """
    if not prompt:
        return True, None
    text = str(prompt)

    # A minor anywhere near clothing, undressing or body-coverage language is
    # refused OUTRIGHT — no combination test, no benefit of the doubt. The
    # first draft of this guard AND'd the minor test with the nudity test, and
    # "a child in underwear" passed both: `underwear` alone was not in the
    # nudity list. That is the exact case this whole module exists for, so the
    # test is now presence, not conjunction.
    if _MINOR_RE.search(text) and (
        _NUDITY_RE.search(text)
        or _LESS_COVERAGE_RE.search(text)
        or _UNDERGARMENT_AS_OUTFIT_RE.search(text)
        or _REMOVAL_RE.search(text)
        or _REMOVAL_REVERSED_RE.search(text)
        or re.search(r"\b(?:swimsuit|swimwear|bikini|swim\s+shorts|trunks|"
                     r"bath(?:ing)?\s+suit|leotard|towel)\b", text, re.IGNORECASE)
    ):
        return False, GUARD_REFUSAL
    if _NUDITY_RE.search(text):
        return False, GUARD_REFUSAL
    if _REMOVAL_RE.search(text) or _REMOVAL_REVERSED_RE.search(text):
        return False, GUARD_REFUSAL
    if _UNDERGARMENT_AS_OUTFIT_RE.search(text):
        return False, GUARD_REFUSAL
    if _LESS_COVERAGE_RE.search(text):
        return False, GUARD_REFUSAL
    return True, None


# ── Layer 1: expansion ───────────────────────────────────────────────────

# Settings whose implied wardrobe/lighting/physics differ enough from an
# ordinary photo that "change only what was named" produces an incoherent
# result. Used to decide whether to spend a model call at all.
_SCENE_CUES = re.compile(
    r"\b(?:pool|swimming|swim|sea|ocean|beach|lake|river|underwater|surf|"
    r"snow|ski|skiing|mountain|blizzard|winter|"
    r"gym|workout|running|marathon|yoga|"
    r"gala|wedding|black[-\s]?tie|formal|red\s+carpet|"
    r"desert|jungle|forest|rain|storm|"
    r"space|astronaut|scuba|diving|"
    r"bed|sleeping|shower|bath|sauna|hot\s+tub|jacuzzi)\b",
    re.IGNORECASE,
)


def needs_scene_expansion(instruction: str) -> bool:
    """Whether the instruction moves the subject somewhere that implies more
    than the change it names."""
    return bool(instruction and _SCENE_CUES.search(instruction))


# What the edit prompt says about preserving the source. The old suffix said
# "Make only the change described above… Preserve the rest of the source image
# exactly", unconditionally — which is precisely the instruction that kept a
# t-shirt and a sun hat on a man sitting in an infinity pool. When the setting
# changes, dependent changes are REQUIRED, not forbidden.
REALISM_PRESERVE = (
    " — Make only the change described above and blend it in seamlessly. "
    "Preserve the rest of the source image exactly: keep its existing style, "
    "resolution, texture, film grain, colour, and lighting. If the source is a "
    "real photograph, keep the result photorealistic with natural skin texture, "
    "pores and detail — do NOT smooth, beautify, airbrush, upscale, restyle, or "
    "give it an artificial over-processed 'AI-generated' look. Only depart from "
    "the original's style if the instruction explicitly asks for a different one."
)

REALISM_COHERENT = (
    " — Apply the scene described above coherently: change everything the new "
    "setting requires (clothing appropriate to it, pose, waterline, wet or dry "
    "skin, lighting direction and colour temperature, shadows and reflections) "
    "and leave everything else exactly as the source has it. The subject must "
    "remain the SAME PERSON — same face, same build, same skin tone, same hair "
    "— and must remain fully clothed in setting-appropriate attire. If the "
    "source is a real photograph, keep the result photorealistic with natural "
    "skin texture, pores and detail — do NOT smooth, beautify, airbrush, "
    "upscale or restyle it."
)


def realism_suffix(scene_changed: bool) -> str:
    return REALISM_COHERENT if scene_changed else REALISM_PRESERVE


async def build_scene_prompt(
    instruction: str,
    *,
    expand: Optional[Callable[[str, str], Awaitable[str]]] = None,
) -> Tuple[str, bool]:
    """`(prompt, scene_changed)` for an edit.

    `expand(system, user)` is the model call, injected so this module stays
    testable and has no import edge into the LLM stack. Any failure — no
    expander, an exception, an empty or a REFUSED expansion — falls back to the
    user's own words. The guard runs on whatever comes out either way, so the
    fallback cannot be a hole.
    """
    instruction = (instruction or "").strip()
    if not instruction:
        return instruction, False

    if not needs_scene_expansion(instruction) or expand is None:
        return instruction, False

    try:
        expanded = (await expand(SCENE_SYSTEM_PROMPT, instruction) or "").strip()
    except Exception:  # noqa: BLE001 - an expander fault must not block an edit
        logger.warning("image scene expansion failed; using the raw instruction", exc_info=True)
        return instruction, True

    if not expanded or len(expanded) < len(instruction):
        return instruction, True

    allowed, _ = check_clothing_guard(expanded)
    if not allowed:
        # The expander produced something the guard refuses. Do NOT ship it and
        # do NOT refuse the user outright — their instruction was innocent and
        # it is our expansion that went wrong. Fall back to their own words,
        # which are then guarded on their own merits by the caller.
        logger.warning("image scene expansion refused by the clothing guard; falling back")
        return instruction, True

    return expanded, True
