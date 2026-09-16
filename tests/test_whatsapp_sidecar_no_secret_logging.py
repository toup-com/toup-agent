"""GA run D-6: the WhatsApp sidecar logged a cryptographic secret.

`messages.upsert.entry` dumped `m.message.messageContextInfo` wholesale.
That object carries **`messageSecret`** — a 32-byte per-message
cryptographic secret (44 chars base64) — plus
`deviceListMetadata.{recipient,sender}KeyHash`, and the surrounding
fields carried `participantPn` (a subscriber phone number) and
`pushName`. These lines ship to Loki with 14-day retention.

Measured on the live fleet 2026-08-10: 6 `messageSecret` values, 4
`recipientKeyHash`, 2 `senderKeyHash`, 26 `participantPn`, 26
`pushName`. Message BODIES were never logged — `messageKeys` holds
protobuf field names only — so this was a metadata leak, not a content
leak, but a 32-byte message key is a secret by any definition.

It survived the repo secret scanner because 44-char base64 sits below
its length threshold and the value has no secret-ish identifier next to
it; only field-path analysis surfaced it. That is the lesson worth
keeping: a scanner tuned for `KEY = value` shapes cannot see a secret
that arrives as one field inside a logged object.

The diagnostic intent of the line — "so a future WhatsApp
identity-format change surfaces from logs alone" — lives entirely in the
KEY NAMES and the shape, both of which are still logged.
"""
from __future__ import annotations

import re
from pathlib import Path

SIDECAR = (
    Path(__file__).resolve().parents[1] / "whatsapp_sidecar" / "sidecar.mjs"
)


def _src() -> str:
    return SIDECAR.read_text()


def test_message_context_info_object_is_never_logged():
    """The object itself must not reach a log call — only its keys."""
    src = _src()
    # `messageContextInfoKeys: ...Object.keys(...)` is fine. A bare
    # `messageContextInfo: m.message?.messageContextInfo` is the leak.
    leak = re.search(
        r"messageContextInfo\s*:\s*[^\n]*messageContextInfo(?!\s*\?\s*Object\.keys)",
        src,
    )
    assert leak is None, (
        "messageContextInfo is being logged as an object — it carries "
        "messageSecret, a 32-byte per-message cryptographic secret, and "
        "these lines are retained in Loki for 14 days"
    )


def test_message_context_info_keys_are_still_logged():
    """The diagnostic value was always the key names; keep them."""
    assert "messageContextInfoKeys" in _src()


def test_raw_key_object_values_are_not_logged():
    """`m.key` carries participantPn — a subscriber phone number."""
    src = _src()
    assert "Object.fromEntries(Object.entries(m.key))" not in src, (
        "the raw key object is logged verbatim, which puts subscriber "
        "phone numbers (participantPn) into 14-day log retention"
    )
    assert "redactValues(" in src, "no redaction helper is applied to key fields"


def test_redact_values_keeps_shape_and_drops_content():
    """Run the shipped helper: names and shapes survive, values do not."""
    import json
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        import pytest

        pytest.skip("node not available")

    src = _src()
    m = re.search(r"function redactValues\(obj\) \{.*?\n\}", src, re.S)
    assert m, "redactValues() not found in the sidecar"

    script = (
        m.group(0)
        + "\nconst k={remoteJid:'1234@s.whatsapp.net',fromMe:false,"
        "id:'ABC123',participantPn:'15551234567@s.whatsapp.net'};"
        "console.log(JSON.stringify(redactValues(k)));"
    )
    out = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, timeout=30
    )
    assert out.returncode == 0, out.stderr[:400]
    flat = out.stdout.strip()

    assert "15551234567" not in flat, "the phone number survived redaction"
    assert "s.whatsapp.net" not in flat, "the JID survived redaction"

    parsed = json.loads(flat)
    assert set(parsed) == {"remoteJid", "fromMe", "id", "participantPn"}, (
        "field NAMES must survive — they are what a format change shows up in"
    )
    assert parsed["fromMe"] is False, "booleans are safe and stay readable"
    assert parsed["participantPn"].startswith("str("), parsed


# ── fix lane A (R46): the masking must survive one more level ─────────


def _mask_fields_harness(payload_js: str) -> dict:
    """Run the SHIPPED `maskFields` (and the helpers it calls) under node."""
    import json
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        import pytest

        pytest.skip("node not available")

    src = _src()
    start = src.index("const _ID_KEYS = new Set([")
    end = src.index("function maskFields(")
    tail = re.search(r"function maskFields\(.*?\n\}", src[end:], re.S)
    assert tail, "maskFields() not found in the sidecar"
    block = src[start:end] + tail.group(0)

    script = (
        "const {createHash} = require('crypto');\n"
        + block
        + f"\nconsole.log(JSON.stringify(maskFields({payload_js})));"
    )
    out = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, timeout=30
    )
    assert out.returncode == 0, out.stderr[:400]
    return json.loads(out.stdout.strip())


def test_a_nested_identifier_is_masked_too():
    """`maskFields` used to mask top-level strings and `_ID_KEYS` scalars only,
    so `{ meta: { jid } }` reached pino verbatim. The masking lives in the log
    helpers precisely because a NEW call site is where the next leak comes
    from — and a nested payload is exactly such a call site."""
    got = _mask_fields_harness(
        "{stage:'x', meta:{jid:'15551234567@s.whatsapp.net', "
        "note:'code for +15551234567'}, peers:[{phone:'+15551234567'}]}"
    )
    flat = repr(got)
    assert "15551234567@s.whatsapp.net" not in flat, flat
    assert "+15551234567" not in flat, flat
    assert got["stage"] == "x", "non-identifier values must stay readable"
    assert isinstance(got["meta"], dict) and set(got["meta"]) == {"jid", "note"}, (
        "the SHAPE is the diagnostic — key names must survive"
    )


def test_the_walk_is_bounded():
    """A Baileys payload may not become the cost of one log line."""
    deep = "{a:{b:{c:{d:{jid:'15551234567@s.whatsapp.net'}}}}}"
    got = _mask_fields_harness(deep)
    leaf = got["a"]["b"]["c"]["d"]
    assert isinstance(leaf, str) and leaf.startswith("object("), (
        f"the walk did not stop at the depth cap: {got}"
    )
