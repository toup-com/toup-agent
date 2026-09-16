"""`estimate_message_tokens` on image blocks — the input the compaction
trigger reads.

It charged a FLAT 300 tokens per image (`context_manager.py:174-175`), which
could not scale with resolution at all — the one property an image's cost
actually has. Production evidence for the order of magnitude, cited so the
numbers below are not taste:

    Loki, container toup-agent-pool-78, 2026-09-15 14:32:56.497
        [PERF] build_system_prompt: 237ms — 95872 chars (~23968 tokens)
    Loki, container toup-agent-pool-87, 2026-09-14 11:06:06.545
        [PERF] build_system_prompt: 147ms — 81526 chars (~20381 tokens)

Under-counting an image-heavy turn by 5x is how a thread sails past the window
compaction was supposed to protect.
"""

from __future__ import annotations

import base64

from app.agent.context_manager import estimate_image_tokens, estimate_message_tokens


def block(w=None, h=None, b64_len=0):
    b = {"type": "image_url", "image_url": {"url": "data:image/webp;base64," + ("A" * b64_len)}}
    if w:
        b["width"] = w
        b["height"] = h
    return b


def test_the_estimate_is_strictly_monotonic_in_pixels():
    sizes = [(256, 256), (512, 512), (1024, 1024), (1536, 1536)]
    vals = [estimate_image_tokens(block(w, h)) for w, h in sizes]
    assert vals == sorted(vals)
    assert len(set(vals)) > 1, "still a flat constant"


def test_a_phone_photo_is_no_longer_charged_300_tokens():
    v = estimate_image_tokens(block(1280, 960))
    assert v > 300
    assert 500 <= v <= 4000


def test_a_tiny_thumbnail_is_cheap():
    assert estimate_image_tokens(block(64, 48)) < 400


def test_dimensions_are_used_when_present_and_derived_when_not():
    with_dims = estimate_image_tokens(block(1280, 960))
    # ~600 KB of base64 stands in for a large photo when nobody stamped dims
    derived = estimate_image_tokens(block(b64_len=600_000))
    assert with_dims > 300 and derived > 300


def test_a_block_with_no_evidence_at_all_uses_the_fallback_not_300():
    assert estimate_image_tokens({"type": "image_url"}) == 1500


def test_a_message_of_four_images_costs_four_images():
    msg = {"role": "user", "content": [block(1280, 960) for _ in range(4)]}
    one = {"role": "user", "content": [block(1280, 960)]}
    total = estimate_message_tokens(msg)
    single = estimate_message_tokens(one)
    assert total > 3 * (single - 4)


def test_text_blocks_are_unaffected():
    msg = {"role": "user", "content": [{"type": "text", "text": "x" * 400}]}
    assert 90 <= estimate_message_tokens(msg) <= 120
