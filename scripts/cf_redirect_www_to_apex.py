"""
Production-level Cloudflare Redirect Rule: www.toup.ai → toup.ai (301).

Why this exists
---------------
toup.ai is the canonical brand. We do NOT want www.toup.ai to be visible in
the address bar — but users still type it (browser autocomplete, old links,
copy-pasted URLs from Stripe receipts, Google search results, etc.).

Without this rule:
  - DNS for www.toup.ai must be set up separately. If it's missing, traffic
    goes nowhere.
  - If DNS is a CNAME to toup.ai, traffic reaches our Railway frontend, but
    Railway routes by Host header and 404s any host that isn't on its
    custom-domains allowlist.
  - Even if Railway *did* serve traffic for www, the URL bar would stay
    `www.toup.ai`, which fragments brand consistency.

This script installs a Cloudflare Single Redirect at the zone level that
301-redirects every www.toup.ai request to the corresponding toup.ai URL,
preserving path AND query string. Browsers cache 301s indefinitely, so
repeat visitors skip the redirect entirely.

Coverage
--------
The rule covers ALL www.toup.ai/* paths. Apex toup.ai is unaffected.

Required token scopes
---------------------
Generate a Cloudflare API token at:
  https://dash.cloudflare.com/profile/api-tokens

Custom token with permissions:
  - Zone : Single Redirect : Edit
  - Zone : Zone : Read
  - Resources: Include → Specific zone → toup.ai

Then export it and run:
  CLOUDFLARE_API_TOKEN=<token> python backend/scripts/cf_redirect_www_to_apex.py

Idempotent
----------
Re-running this script:
  - Detects an existing rule by description match and updates in place
  - Creates the dynamic_redirect entrypoint ruleset on first run if missing
  - Exits 0 on success or no-op (rule already current)
  - Exits 1 on any error

DNS prerequisite
----------------
www.toup.ai must already be a CNAME (proxied) → toup.ai. The redirect rule
fires inside Cloudflare's edge; DNS has to deliver the request to the edge
first. If www has an A record pointing elsewhere (e.g. a Network Solutions
parking IP — the actual state we hit on 2026-04-29 before fixing),
the redirect rule never gets a chance to run.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

ZONE_NAME = "toup.ai"
RULE_DESCRIPTION = "Redirect www.toup.ai → toup.ai (canonical apex)"
RULE_EXPRESSION = '(http.host eq "www.toup.ai")'
RULE_ACTION = "redirect"
RULE_ACTION_PARAMETERS = {
    "from_value": {
        "status_code": 301,
        "target_url": {
            "expression": 'concat("https://toup.ai", http.request.uri.path)',
        },
        "preserve_query_string": True,
    },
}
PHASE = "http_request_dynamic_redirect"


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

    rule_body = {
        "description": RULE_DESCRIPTION,
        "expression": RULE_EXPRESSION,
        "action": RULE_ACTION,
        "action_parameters": RULE_ACTION_PARAMETERS,
        "enabled": True,
    }

    # 2. Fetch the dynamic_redirect ruleset entrypoint.
    # On a zone with no Redirect Rules ever created, the entrypoint doesn't
    # exist yet (error 10003). PUT seeds it with our single rule in one call.
    rs = _api(
        token,
        "GET",
        f"/zones/{zone_id}/rulesets/phases/{PHASE}/entrypoint",
    )
    if not rs.get("success"):
        errs = rs.get("errors") or []
        if any(e.get("code") == 10003 for e in errs):
            put_result = _api(
                token,
                "PUT",
                f"/zones/{zone_id}/rulesets/phases/{PHASE}/entrypoint",
                {"rules": [rule_body]},
            )
            if not put_result.get("success"):
                print(f"ERROR: entrypoint create failed: {put_result}", file=sys.stderr)
                return 1
            created = put_result["result"]["rules"][0]
            print(f"CREATED entrypoint ruleset + rule id={created['id']}")
            return 0
        print(f"ERROR: ruleset fetch failed (need 'Single Redirect: Edit' scope): {rs}",
              file=sys.stderr)
        return 1

    rules = rs["result"].get("rules") or []
    existing = next(
        (r for r in rules if r.get("description") == RULE_DESCRIPTION),
        None,
    )
    ruleset_id = rs["result"]["id"]

    if existing:
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
