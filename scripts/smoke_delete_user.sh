#!/usr/bin/env bash
# §2 — Cascade smoke against a throwaway PROD user.
#
# CRITICAL OPERATOR CONSTRAINTS:
# 1. This script targets PRODUCTION. The §1 guard ensures the backend
#    is correctly configured with sk_live_ for that environment.
# 2. The smoke user MUST NOT complete any paid checkout or subscription
#    flow during the manual agent-setup phase. No real Stripe customer
#    should ever be created for this user.
# 3. This script asserts the resulting user_id does NOT match any of
#    the three real prod tenants before any destructive call.
# 4. If anything looks wrong, ABORT. The cascade is irrevocable.
#
# Operator flow (the script handles 1, 2, 4, 5; you do 3 manually):
#   1. Script registers smoketest+{ts}@toup.ai via the real prod signup.
#   2. Script prints credentials + the new user_id, asserts it is NOT a
#      real-tenant prefix, and asks you to confirm by typing the user_id
#      back in.
#   3. YOU complete agent setup in the iOS app or web UI using the
#      printed JWT. Send a few messages, attach one document. SKIP any
#      paid-checkout / subscription flow — no Stripe customer must be
#      created. Press ENTER in this script's terminal when done.
#   4. Script runs reauth → delete-account, asserts replay is rejected,
#      and prints a verification checklist with the exact psql / SSH /
#      Stripe commands you can run for each item.
#   5. You run the verification checklist and report green/red.
#
# Tz-shift trap (preserved from §1.3):
#   `datetime.utcnow().timestamp()` interprets the naive value as LOCAL
#   time and shifts JWT exp claims by the process's UTC offset. Every
#   JWT timestamp in the SAT module goes through `time.time()` directly.
#   This script does the same.
#
# Usage
# -----
#   ./scripts/smoke_delete_user.sh https://toup.ai
#   TOUP_PROD_BASE_URL=https://toup.ai ./scripts/smoke_delete_user.sh
#
# Required (one of):
#   - first positional arg: prod base URL (no trailing slash)
#   - TOUP_PROD_BASE_URL env var: prod base URL (no trailing slash)
#
# Optional env:
#   SMOKE_USER_PASSWORD — password for the throwaway user (default:
#                         a deterministic test value).
#
# Exits 0 on full HTTP-side pass; exits 1 on any unexpected response,
# any tenant-allowlist trip, or any operator confirmation mismatch.

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────

BASE="${1:-${TOUP_PROD_BASE_URL:-${TOUP_DEV_BASE_URL:-}}}"
if [[ -z "$BASE" ]]; then
  echo "Usage: $0 <PROD_BASE_URL>"  >&2
  echo "   or: TOUP_PROD_BASE_URL=https://toup.ai $0" >&2
  exit 1
fi
BASE="${BASE%/}"

# Real-tenant prefixes (8-char user_id prefixes). If signup somehow
# returns a user_id starting with any of these, abort hard before any
# destructive call. UUID prefix collision is vanishingly unlikely;
# this is belt-and-suspenders defense.
REAL_TENANT_PREFIXES=(
  "c47c5b4b"
  "3134fece"
  "ad6060cc"
)

# Use seconds + nanoseconds (where supported) + PID + $RANDOM so two
# concurrent runs against the same backend don't collide on email.
NANO="$(date -u +%N)"
[[ "$NANO" =~ ^[0-9]+$ ]] || NANO="$(printf '%06d' $((RANDOM * RANDOM % 1000000)))"
SUFFIX="$(date -u +%s)${NANO:0:6}$$${RANDOM}"
EMAIL="smoketest+${SUFFIX}@toup.ai"
EMAIL_PATTERN='^smoketest\+[0-9]+@toup\.ai$'
PASSWORD="${SMOKE_USER_PASSWORD:-smoke-test-pw-9f4a8b2c}"
JWT=""
SAT=""
USER_ID=""
RECEIPT=""

# ── Helpers ─────────────────────────────────────────────────────────

red()    { printf "\033[31m%s\033[0m\n" "$*" >&2; }
green()  { printf "\033[32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }
gray()   { printf "\033[90m%s\033[0m\n" "$*"; }
bold()   { printf "\033[1m%s\033[0m\n" "$*"; }

step() {
  echo
  echo "===== $* ====="
}

die() {
  red "FAIL: $*"
  exit 1
}

# Verify required tooling is present BEFORE we start poking at the API.
for cmd in curl jq; do
  command -v "$cmd" >/dev/null 2>&1 || die "missing required tool: $cmd"
done

require_status() {
  local label="$1" expected="$2" actual="$3" body="${4:-}"
  if [[ "$actual" != "$expected" ]]; then
    red "[$label] expected HTTP $expected, got $actual"
    [[ -n "$body" ]] && red "body: $body"
    exit 1
  fi
}

# `http METHOD PATH HEADER...` — performs the call and exports the
# response body in $LAST_BODY and HTTP status in $LAST_STATUS.
LAST_BODY=""
LAST_STATUS=""
http() {
  local method="$1" path="$2"
  shift 2
  local tmp
  tmp="$(mktemp)"
  LAST_STATUS="$(curl -sS -o "$tmp" -w '%{http_code}' \
    -X "$method" "$BASE$path" "$@")"
  LAST_BODY="$(cat "$tmp")"
  rm -f "$tmp"
}

# ── 0. Header banner ────────────────────────────────────────────────

bold "=========================================================="
bold "  §2 — Cascade smoke against PRODUCTION"
bold "  Target: $BASE"
bold "  Time:   $(date -u +%FT%TZ)"
bold "=========================================================="
echo
yellow "Read the operator constraints at the top of this file before proceeding."
yellow "If the cascade fires against a real tenant, recovery is impossible."
echo

# ── 0a. Email-pattern self-check ────────────────────────────────────

if ! [[ "$EMAIL" =~ $EMAIL_PATTERN ]]; then
  die "internal error: generated email '$EMAIL' does not match $EMAIL_PATTERN"
fi
gray "Generated email: $EMAIL (matches required pattern)"

# ── 1. Register throwaway user ──────────────────────────────────────

step "1. Register throwaway user"
gray "POST $BASE/api/auth/register  email=$EMAIL"
http POST /api/auth/register \
  -H "Content-Type: application/json" \
  -d "$(jq -nc --arg e "$EMAIL" --arg p "$PASSWORD" \
        '{email: $e, password: $p, name: "Smoke Test"}')"
require_status "register" "201" "$LAST_STATUS" "$LAST_BODY"
USER_ID="$(echo "$LAST_BODY" | jq -r '.id')"
[[ -n "$USER_ID" && "$USER_ID" != "null" ]] || die "no id in register response"
green "✓ Registered user $USER_ID"

# ── 1a. Tenant-allowlist guard (HARD BLOCK) ─────────────────────────

step "1a. Tenant-allowlist guard"
USER_ID_PREFIX="${USER_ID:0:8}"
echo
bold "  USER ID:        $USER_ID"
bold "  USER ID PREFIX: $USER_ID_PREFIX"
echo
for forbidden in "${REAL_TENANT_PREFIXES[@]}"; do
  if [[ "$USER_ID_PREFIX" == "$forbidden" ]]; then
    red "ABORT: registered user_id prefix '$USER_ID_PREFIX' matches a real prod tenant."
    red "       The cascade MUST NOT run against this user."
    red "       Investigate before re-running the smoke."
    exit 1
  fi
done
green "✓ user_id prefix is not a real-tenant prefix"

# ── 1b. Email-shape re-check from server response ───────────────────

ME_REGISTERED_EMAIL="$(echo "$LAST_BODY" | jq -r '.email')"
if ! [[ "$ME_REGISTERED_EMAIL" =~ $EMAIL_PATTERN ]]; then
  die "server returned email '$ME_REGISTERED_EMAIL' which does not match $EMAIL_PATTERN — abort"
fi
green "✓ server-side email matches smoketest pattern"

# ── 2. Login → JWT ──────────────────────────────────────────────────

step "2. Login → JWT"
http POST /api/auth/login \
  -H "Content-Type: application/json" \
  -d "$(jq -nc --arg e "$EMAIL" --arg p "$PASSWORD" \
        '{email: $e, password: $p}')"
require_status "login" "200" "$LAST_STATUS" "$LAST_BODY"
JWT="$(echo "$LAST_BODY" | jq -r '.access_token')"
[[ -n "$JWT" && "$JWT" != "null" ]] || die "no access_token in login response"
green "✓ Got JWT (len=${#JWT})"

# ── 3. Sanity: /me returns user ─────────────────────────────────────

step "3. Sanity: GET /api/auth/me"
http GET /api/auth/me -H "Authorization: Bearer $JWT"
require_status "me-pre-delete" "200" "$LAST_STATUS" "$LAST_BODY"
ME_EMAIL="$(echo "$LAST_BODY" | jq -r '.email')"
[[ "$ME_EMAIL" == "$EMAIL" ]] || die "/me email mismatch: expected $EMAIL got $ME_EMAIL"
green "✓ /me returns user info"

# ── 4. Operator manual setup (PAUSE) ────────────────────────────────

step "4. OPERATOR — manual setup phase"
echo
bold "Operator: paste the following credentials into the iOS app or web UI"
bold "          (or curl the API directly) to seed realistic agent state."
echo
echo "  EMAIL:    $EMAIL"
echo "  PASSWORD: $PASSWORD"
echo "  JWT:      $JWT"
echo "  USER_ID:  $USER_ID"
echo
yellow "Now do the following manually:"
echo "  • Complete agent setup (the wizard provisions a Contabo container)."
echo "  • Send 3-4 messages to the agent."
echo "  • Upload one document."
echo "  • SKIP any paid checkout / subscription flow."
echo "    Confirm: the smoke user must NOT have a Stripe customer."
echo
yellow "When seeding is complete, return here and confirm to proceed."

# Belt-and-suspenders: operator must type the user_id back in to confirm
# they're aware of which user is about to be deleted.
echo
read -r -p "Type the user_id ($USER_ID) to confirm and proceed: " TYPED_ID
if [[ "$TYPED_ID" != "$USER_ID" ]]; then
  die "operator confirmation mismatch — typed '$TYPED_ID', expected '$USER_ID'"
fi
green "✓ Operator confirmed"

# ── 5. /reauth → SAT ────────────────────────────────────────────────

step "5. POST /api/auth/reauth"
http POST /api/auth/reauth -H "Authorization: Bearer $JWT"
require_status "reauth" "200" "$LAST_STATUS" "$LAST_BODY"
SAT="$(echo "$LAST_BODY" | jq -r '.sensitive_action_token')"
[[ -n "$SAT" && "$SAT" != "null" ]] || die "no sensitive_action_token in reauth response"
EXPIRES_IN="$(echo "$LAST_BODY" | jq -r '.expires_in')"
[[ "$EXPIRES_IN" == "300" ]] || die "expected expires_in=300, got $EXPIRES_IN"
green "✓ Got SAT, expires_in=$EXPIRES_IN"

# ── 6. /delete-account with SAT → expect 200 ────────────────────────

step "6. POST /api/auth/delete-account (DESTRUCTIVE)"
yellow "About to fire the cascade against $USER_ID — irrevocable."
http POST /api/auth/delete-account \
  -H "Authorization: Bearer $JWT" \
  -H "X-Sensitive-Action-Token: $SAT"
require_status "delete-account" "200" "$LAST_STATUS" "$LAST_BODY"
SUCCESS="$(echo "$LAST_BODY" | jq -r '.success')"
[[ "$SUCCESS" == "true" ]] || die "delete-account body: success != true"
RECEIPT="$(echo "$LAST_BODY" | jq -c '.receipt')"
green "✓ /delete-account returned success"
gray "Receipt: $RECEIPT"

# ── 7. /me with old JWT → expect 401 ────────────────────────────────

step "7. GET /api/auth/me (post-delete) → expect 401"
http GET /api/auth/me -H "Authorization: Bearer $JWT"
require_status "me-post-delete" "401" "$LAST_STATUS" "$LAST_BODY"
green "✓ /me correctly rejects the old JWT"

# ── 8. /delete-account replay → expect 401 ─────────────────────────

step "8. POST /api/auth/delete-account replay → expect 401"
# Two rejection paths can produce 401 here:
#   (a) The auth gate sees the now-deleted user_id in the JWT and
#       rejects (User-not-found).
#   (b) The SAT verifier sees the redeemed jti and rejects (replay).
# Both are valid passes for this smoke; the pytest suite covers
# the SAT-replay path in isolation against a still-existing user.
http POST /api/auth/delete-account \
  -H "Authorization: Bearer $JWT" \
  -H "X-Sensitive-Action-Token: $SAT"
require_status "delete-account-replay" "401" "$LAST_STATUS" "$LAST_BODY"
green "✓ Replay correctly rejected"

# ── Summary + operator verification checklist ───────────────────────

step "Summary + operator verification checklist"
green "All HTTP-side checks passed."
echo
bold "USER_ID:  $USER_ID"
bold "EMAIL:    $EMAIL"
bold "RECEIPT:  $RECEIPT"
echo
yellow "=== OPERATOR — RUN EACH CHECK BELOW ==="
echo
echo "[ ] Container gone from \`docker ps\` on Contabo"
gray "    ssh -i ~/.ssh/toup_mrhx mrhx@144.126.138.62 \\"
gray "      \"docker ps --format '{{.Names}}' | grep '${USER_ID_PREFIX}' || echo 'gone (PASS)'\""
echo
echo "[ ] Caddy route removed (no subdomain for prefix $USER_ID_PREFIX)"
gray "    ssh -i ~/.ssh/toup_mrhx mrhx@144.126.138.62 \\"
gray "      \"sudo grep -c 'agent-${USER_ID_PREFIX}' /etc/caddy/Caddyfile || echo '0 (PASS)'\""
echo
echo "[ ] Per-tenant Postgres role + DB dropped"
gray "    ssh -i ~/.ssh/toup_mrhx mrhx@144.126.138.62 \\"
gray "      \"docker exec toup-postgres psql -U postgres -tAc \\\\\""
gray "        \\\"SELECT 1 FROM pg_database WHERE datname='toup_agent_${USER_ID_PREFIX}'\\\\\"\""
gray "    -- expect: empty"
echo
echo "[ ] User row gone from platform users table"
gray "    psql \"\$PLATFORM_DB_URL\" -c \\"
gray "      \"SELECT id, email FROM users WHERE id = '$USER_ID'\""
gray "    -- expect: 0 rows"
echo
echo "[ ] All cascade tables empty for user_id"
gray "    psql \"\$PLATFORM_DB_URL\" -f - <<SQL"
gray "      SELECT 'agent_configs', count(*) FROM agent_configs WHERE user_id = '$USER_ID'"
gray "      UNION ALL SELECT 'messages', count(*) FROM messages WHERE user_id = '$USER_ID'"
gray "      UNION ALL SELECT 'memories', count(*) FROM memories WHERE user_id = '$USER_ID'"
gray "      UNION ALL SELECT 'documents', count(*) FROM documents WHERE user_id = '$USER_ID'"
gray "      UNION ALL SELECT 'day_chats', count(*) FROM day_chats WHERE user_id = '$USER_ID'"
gray "      UNION ALL SELECT 'sessions', count(*) FROM sessions WHERE user_id = '$USER_ID'"
gray "      ORDER BY 1;"
gray "    SQL"
gray "    -- expect: every row count = 0"
echo
echo "[ ] deletion_audit_events row populated, status=succeeded, all flags set"
gray "    psql \"\$PLATFORM_DB_URL\" -c \\"
gray "      \"SELECT user_id, status, container_destroyed, stripe_cleared,\""
gray "      \"        openai_archived, failure_step, completed_at\""
gray "      \" FROM deletion_audit_events WHERE user_id = '$USER_ID'\""
gray "    -- expect: 1 row, status='succeeded', container_destroyed=true,"
gray "    --         stripe_cleared=true, openai_archived=true (or n/a),"
gray "    --         failure_step IS NULL, completed_at IS NOT NULL"
echo
echo "[ ] No Stripe customer created for smoke user (Option 2 invariant)"
gray "    Open Stripe live dashboard → Customers → search by email '$EMAIL'"
gray "    -- expect: no results"
gray "    Optional API form (requires sk_live_, careful):"
gray "      curl -s 'https://api.stripe.com/v1/customers/search?query=email:%22$EMAIL%22' \\"
gray "        -u sk_live_***: | jq '.data | length'"
gray "      -- expect: 0"
echo
echo "[ ] Three real tenants completely unaffected"
gray "    psql \"\$PLATFORM_DB_URL\" -c \\"
gray "      \"SELECT id FROM users WHERE id::text LIKE 'c47c5b4b%'\""
gray "      \"  OR id::text LIKE '3134fece%'\""
gray "      \"  OR id::text LIKE 'ad6060cc%'\""
gray "    -- expect: 3 rows, all unchanged"
gray "    AND verify their containers still running:"
gray "    ssh -i ~/.ssh/toup_mrhx mrhx@144.126.138.62 \\"
gray "      \"docker ps --format '{{.Names}}' | grep -E 'c47c5b4b|3134fece|ad6060cc'\""
gray "    -- expect: 3 lines"
echo
green "Smoke complete (HTTP side). Run the verification list above and report."
