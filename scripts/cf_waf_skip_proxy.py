"""
Production-level Cloudflare WAF / Bot Fight Mode skip rule for /api/llm/*.

Why this exists
---------------
Cloudflare's Bot Fight Mode + managed bot detection on toup.ai blocks the
bare Anthropic / OpenAI Python SDK signatures (user-agent + x-stainless-*)
as automation. Our agents call /api/llm/v1/messages and /api/llm/openai/v1/
chat/completions through these SDKs. We have a defensive header normalizer
in app/services/bundle_client.py:_normalize_headers_for_waf, but a vendor
bumping their SDK could re-introduce a new bot signature we don't strip
yet. The proper fix is to tell Cloudflare "don't run bot rules on
/api/llm/*" — those requests are already authenticated by our proxy.

This script is idempotent: re-running it updates the existing rule rather
than creating duplicates. Tag the rule with a stable description so we
recognize it on subsequent runs.

Coverage
--------
The rule covers ONLY /api/llm/* — the proxy mount point. It does NOT
affect:
  - Direct OAuth Anthropic CLI traffic (sk-ant-oat → api.anthropic.com),
    which bypasses our domain entirely.
  - Browser → toup.ai (chat UI, dashboards) — full Bot Fight Mode stays
    enforced, protecting the user-facing surface.

Required token scopes
---------------------
Generate a Cloudflare API token at:
  https://dash.cloudflare.com/profile/api-tokens

Custom token with permissions:
  - Zone : Zone WAF : Edit
  - Zone : Zone : Read
  - Resources: Include → Specific zone → toup.ai

Then export it and run:
  CLOUDFLARE_API_TOKEN=<token> python backend/scripts/cf_waf_skip_proxy.py

Exits 0 on success or no-op (rule already current). Exits 1 on any error.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

ZONE_NAME = "toup.ai"
RULE_DESCRIPTION = "Skip bot detection on LLM proxy paths"
RULE_EXPRESSION = '(http.request.uri.path matches "^/api/llm/")'
# `skip` action with `phases: [http_ratelimit, http_request_sbfm]` and
# `products: ["bic","hot","securityLevel","uaBlock","zoneLockdown"]`
# disables Bot Fight Mode + UA Block + IP Reputation gating + Zone Lockdown
# for matching requests. Free plan supports this combination.
RULE_ACTION = "skip"
RULE_ACTION_PARAMETERS = {
    "products": ["bic", "hot", "securityLevel", "uaBlock", "zoneLockdown"],
    "phases": ["http_ratelimit", "http_request_sbfm"],
}


def _api(token: str, method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        f"https://api.cloudflare.com/client/v4{path}",
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        data=json.dumps(body).encode() if body is not None else None,
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode())


def main() -> int:
    token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    if not token:
        print("ERROR: CLOUDFLARE_API_TOKEN not set", file=sys.stderr)
        return 1

    # 1. Resolve zone
    zones = _api(token, "GET", f"/zones?name={ZONE_NAME}")
    if not zones.get("success") or not zones.get("result"):
        print(f"ERROR: zone lookup failed: {zones}", file=sys.stderr)
        return 1
    zone_id = zones["result"][0]["id"]
    print(f"zone_id={zone_id} name={ZONE_NAME}")

    # 2. Fetch the http_request_firewall_custom ruleset entrypoint
    rs = _api(
        token,
        "GET",
        f"/zones/{zone_id}/rulesets/phases/http_request_firewall_custom/entrypoint",
    )
    if not rs.get("success"):
        print(f"ERROR: ruleset fetch failed (need 'Zone WAF: Edit' scope): {rs}",
              file=sys.stderr)
        return 1
    rules = rs["result"].get("rules") or []
    existing = next(
        (r for r in rules if r.get("description") == RULE_DESCRIPTION),
        None,
    )

    rule_body = {
        "description": RULE_DESCRIPTION,
        "expression": RULE_EXPRESSION,
        "action": RULE_ACTION,
        "action_parameters": RULE_ACTION_PARAMETERS,
        "enabled": True,
    }

    ruleset_id = rs["result"]["id"]

    if existing:
        # Idempotent: update in place. Compare to avoid noisy no-op writes.
        same = (
            existing.get("expression") == RULE_EXPRESSION
            and existing.get("action") == RULE_ACTION
            and existing.get("action_parameters") == RULE_ACTION_PARAMETERS
            and existing.get("enabled") is True
        )
        if same:
            print(f"NO-OP: rule already current (id={existing['id']})")
            return 0
        result = _api(
            token,
            "PATCH",
            f"/zones/{zone_id}/rulesets/{ruleset_id}/rules/{existing['id']}",
            rule_body,
        )
        action = "UPDATED"
    else:
        result = _api(
            token,
            "POST",
            f"/zones/{zone_id}/rulesets/{ruleset_id}/rules",
            rule_body,
        )
        action = "CREATED"

    if not result.get("success"):
        print(f"ERROR: rule {action.lower()} failed: {result}", file=sys.stderr)
        return 1

    print(f"{action} rule id={result['result']['id']}  expr={RULE_EXPRESSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
