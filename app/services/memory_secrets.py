"""The never-store secret tier — v3's only inherited gate rule.

Round 8's `memory_gate.py` was a nine-rule write gate whose whole design
assumed the writer was fed the user's own sentence and could measure a
candidate's overlap against it. v3 retires that: the curator is the one
writer, its source gate is structural (only the user's clean text is a
source of facts), and the eval set is what proves it. Exactly ONE rule
survives, because it never measured provenance at all — it reads the
content and nothing else, so it is as correct for a curated bullet as it
was for an extracted sentence.

It lives here rather than in `memory_gate` so that cutting the gate cannot
strand it. `memory_gate` re-exports these names for the row-era callers it
still serves; nothing v3 writes may import them from there.

Two tiers, because "secret" covers two different things.

NEVER_STORE has no legitimate "please remember this" use and a high blast
radius if it leaks: payment cards, CVVs, government identity numbers,
provider API keys and bearer tokens. Refused on every path, whatever anyone
asked for.

The password/passphrase/PIN tier is different. A user who says "remember my
storage locker passphrase is kestrel-dbf7" is stating a fact about their own
life and expecting the agent to keep it. So that tier is refused on
AUTOMATIC capture (nobody asked; the value was merely observed) and allowed
on an EXPLICIT save. The tier split is the whole safety argument: an
explicit save can never reach a card number or an API key, no matter how it
is phrased.
"""

from __future__ import annotations

import re
from typing import Optional

# ── Sensitive values ─────────────────────────────────────────────────────
#
# Policy (also stated in MEMORY_SYSTEM_MAP.md §2.1): a small set of secret
# categories is NEVER written to long-term memory verbatim — payment card
# numbers, card verification values, government identity numbers, API keys,
# bearer tokens, private keys, credentials inside connection strings, and
# declared passwords/passphrases/PINs.
#
# This exists because the alternative was enforcement by LLM judgment. The
# labeled privacy scenarios (K01–K04) passed before this rule existed, purely
# because gpt-4o-mini chose not to extract the values — which is a disposition,
# not a control. Nothing stopped a different model, a different temperature or a
# differently-phrased turn from writing a card number into a permanent,
# plaintext, embedded row. Memory content is not encrypted at column level, so a
# stored secret is a stored secret.
#
# Scoped tightly on purpose. Everything NOT in this list stays storable,
# including the things a blunter rule would eat:
#   * health details and medications  — K05, durable user facts, must be kept
#   * user-chosen door/locker/garage codes the user ASKED to save — A21
#   * flight numbers, addresses, dates, phone-shaped identifiers
_LUHN_CANDIDATE_RE = re.compile(r"(?<!\w)(?:\d[ -]?){12,18}\d(?!\w)")
_CARD_CONTEXT_RE = re.compile(
    r"(?i)\b(?:visa|mastercard|amex|american\s+express|discover|credit\s+card|"
    r"debit\s+card|card\s+number|cardholder|pan)\b"
)

# The trigger nouns for a declared password, in the two languages this product
# is actually spoken in. Persian is not decoration here: the extractor is fed
# Farsi turns and returns Farsi content (see the Farsi clock patterns above), so
# an English-only noun list means a Persian user's password is written verbatim
# into a plaintext, embedded, permanent row while an English user's is refused.
#
# This is NOT the cross-lingual case the provenance rules abstain on.
# `assistant_echo_reason` and `unsupported_claim_reason` abstain on script
# mismatch because they MEASURE the memory against the user's own words, and a
# different alphabet makes that measurement meaningless (see `_dominant_script`
# below). `sensitive_content_reason` never looks at `from_user` at all — it
# reads the content and nothing else — so it inherits no such obligation, and
# adding a script costs the abstention nothing.
_FA_PASSWORD_NOUN = (
    r"(?: رمز\s*(?:عبور|ورود) | رمزعبور | پسورد | کلمه\s*(?:ی\s*)?عبور | گذرواژه )"
)
# What a password VALUE looks like, for the Persian rules. Requiring at least
# one Latin letter, ASCII digit or Persian/Arabic-Indic digit inside a >=4-char
# token is what keeps these two rules narrow: Persian is SOV, so a copula rule
# with no shape constraint on the value would read "رمز عبور من خیلی ضعیف است"
# ("my password is very weak") as a disclosed secret. It contains no such
# character, so it is not a value and the rule stays quiet.
#
# Both constraints are LOOKAHEADS and the token is consumed once, by a single
# `\S+`. Written the obvious way \u2014 `\S* [alnum] \S*` \u2014 the engine retries every
# split point of the token and the rule goes quadratic in token length:
# measured 3.2ms on a single 590-character run, against 0.02ms here. That
# matters because `MemoryService.create_memory`'s storage backstop calls this
# function WITHOUT the MAX_MEMORY_CHARS cap on purpose (document ingestion
# legitimately stores long RAG chunks), so the input is not bounded at 600.
_FA_SECRET_VALUE = r"(?=\S{4,}) (?= \S* [A-Za-z0-9\u06F0-\u06F9\u0660-\u0669] ) \S+"

_SENSITIVE_RES = (
    # Government identity numbers. Canadian SIN / US SSN shapes.
    ("government_id", re.compile(r"(?<!\d)\d{3}[ -]\d{3}[ -]\d{3}(?!\d)")),
    # US SSN. `[ -]` and not `-`: "123 45 6789" is the same number written the
    # same way people write it, and 3-2-4 does not collide with the shapes this
    # rule must keep (a phone number is 3-3-4, a date is not nine digits).
    ("government_id", re.compile(r"(?<!\d)\d{3}[ -]\d{2}[ -]\d{4}(?!\d)")),
    ("government_id", re.compile(r"(?ix) \b (?: sin | ssn | passport \s* (?:no|number|\#)? ) \b [^.\n]{0,20}? \b [a-z]?\d{6,9} \b")),
    # Provider API keys and bearer tokens, by their own published prefixes.
    #
    # HYPHEN-separated keys: OpenAI (`sk-proj-…`), Anthropic (`sk-ant-…`),
    # OpenAI admin keys. This alternation used to read `proj|live|test|ant|admin`
    # — `live` and `test` are Stripe's words, but Stripe separates with an
    # UNDERSCORE, so naming them here made the line look like Stripe coverage
    # while matching nothing Stripe emits. They are dropped rather than made
    # to work, because the trailing `[A-Za-z0-9_-]{16,}` already absorbs any
    # middle segment; Stripe gets its own honest pattern below.
    ("api_key", re.compile(r"(?i)\b(?:sk|pk|rk)-(?:proj|ant|admin)?-?[A-Za-z0-9_-]{16,}")),
    # Stripe, whose separator is an underscore: sk_live_…, pk_test_…, rk_live_…,
    # and webhook signing secrets.
    ("api_key", re.compile(r"\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{16,}")),
    ("api_key", re.compile(r"\bwhsec_[A-Za-z0-9]{16,}")),
    ("api_key", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}")),
    # Slack bot/user/app/refresh tokens, and app-level tokens.
    ("api_key", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("api_key", re.compile(r"\bxapp-[A-Za-z0-9-]{10,}")),
    # Google API keys are AIza + exactly 35 more. The exact length is the whole
    # control: `AIza\w+` would eat "my flight confirmation is AIzaSyD".
    ("api_key", re.compile(r"(?<![A-Za-z0-9_-])AIza[A-Za-z0-9_-]{35}(?![A-Za-z0-9_-])")),
    ("api_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    # AWS temporary/STS credentials, same shape as AKIA with a different prefix.
    ("api_key", re.compile(r"\bASIA[0-9A-Z]{16}\b")),
    # SendGrid: SG.<id>.<secret>.
    ("api_key", re.compile(r"\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}")),
    # npm automation/publish tokens are npm_ + exactly 36.
    ("api_key", re.compile(r"\bnpm_[A-Za-z0-9]{36}(?![A-Za-z0-9])")),
    ("api_key", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._-]{20,}")),
    ("api_key", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.")),
    # PEM private key blocks — RSA/DSA/EC/OPENSSH/PGP/ENCRYPTED and bare.
    ("api_key", re.compile(r"-----BEGIN\s(?:[A-Z0-9]+\s)*PRIVATE\sKEY(?:\sBLOCK)?-----")),
    # Credentials embedded in a connection string: postgres://, mongodb+srv://,
    # redis://, amqp://, https:// basic auth — anything of the shape
    # scheme://[user]:password@host.
    #
    # The password segment is what this fires on, not the URL. A docs link with
    # no credentials in it ("the postgres docs at postgres://localhost/mydb")
    # has no `:pass@` and is kept, which is the point: `[^\s:/@]` forbids the
    # separators, so a host:port path can never be read as user:password.
    ("api_key", re.compile(r"(?i)\b[a-z][a-z0-9+.-]{1,15}://[^\s:/@]*:[^\s:/@]+@[^\s/]+")),
    # A declared password / passphrase / PIN, with its value.
    # "code" is deliberately absent: "my garage door code is nightjar-4417" is a
    # fact the user explicitly asked to be remembered.
    # Allows a few intervening words: "my password FOR THE ADMIN PANEL is X".
    ("password", re.compile(r"(?ix) \b (?: password | passwd | passphrase | pin \s* (?:code|number)? ) \b (?: \s+ \w+){0,5} \s* (?: is | = | : ) \s* \S{4,}")),
    # The same declaration in Persian, in the two shapes Persian writes it.
    #   separator form:  «پسورد من: fakePass9912»
    ("password", re.compile(
        r"(?ix)" + _FA_PASSWORD_NOUN + r"[^\n:=]{0,24}? [:=] \s* " + _FA_SECRET_VALUE
    )),
    #   copula form:     «رمز عبور من fakePass9912 است»  (Persian is SOV, so the
    #   verb trails the value and an `is|=|:` rule can never see it)
    ("password", re.compile(
        r"(?ix)" + _FA_PASSWORD_NOUN + r"[^\n:=]{0,24}? \s "
        + _FA_SECRET_VALUE + r" \s* (?: است | هست | بود )\b"
    )),
    # Card verification value, only when named.
    ("card_cvv", re.compile(r"(?i)\b(?:cvv|cvc|cv2|security\s+code)\b\s*(?:is|=|:)?\s*\d{3,4}\b")),
)


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


# Two tiers, because "secret" covers two different things.
#
# NEVER_STORE has no legitimate "please remember this" use and a high blast
# radius if it leaks: payment cards, CVVs, government identity numbers, provider
# API keys and bearer tokens. Refused on every path, whatever anyone asked for.
#
# The password/passphrase/PIN tier is different. A user who says "remember my
# storage locker passphrase is kestrel-dbf7" is stating a fact about their own
# life and expecting the agent to keep it — the same product behaviour as the
# garage door code in A21, and the behaviour D-mem-A's supersede tests exercise.
# Refusing that on a deliberate save would be the agent overruling its owner
# about their own locker.
#
# So the passphrase tier is refused on AUTOMATIC capture (where nobody asked for
# anything and the value was merely observed) and allowed on an EXPLICIT save.
# The tier split is the whole safety argument: an explicit save can never reach
# a card number or an API key, no matter how it is phrased.
_NEVER_STORE_LABELS = frozenset({"government_id", "api_key", "card_cvv"})


def sensitive_content_reason(
    content: str, *, explicit_save: bool = False
) -> Optional[str]:
    """Reject memories carrying a secret value verbatim. None == store it."""
    text = content or ""
    if not text:
        return None

    for label, pattern in _SENSITIVE_RES:
        if explicit_save and label not in _NEVER_STORE_LABELS:
            continue
        if pattern.search(text):
            return f"sensitive_{label}"

    # Payment cards last. A bare digit run is rejected only when it is
    # Luhn-valid OR the sentence names a card — so ordinary long numbers (order
    # ids, account numbers, phone numbers) stay storable, while a mistyped or
    # partially-redacted PAN in an obviously card-shaped sentence still does not
    # get written down.
    card_context = _CARD_CONTEXT_RE.search(text) is not None
    for match in _LUHN_CANDIDATE_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and (_luhn_ok(digits) or card_context):
            return "sensitive_card_number"

    return None


__all__ = ["sensitive_content_reason", "NEVER_STORE_LABELS"]

#: Public alias — the tier no caller may opt out of.
NEVER_STORE_LABELS = _NEVER_STORE_LABELS
