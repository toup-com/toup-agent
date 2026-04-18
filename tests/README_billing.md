# Billing / Subscription test coverage

Automated coverage for the embedded Stripe Payment Element flow shipped in
commit `47c3ef5`.

## What's tested today (Phase 1 — this PR)

Backend integration tests (`test_billing_subscription.py`) via pytest +
httpx.AsyncClient + self-signed Stripe webhook events. No live Stripe API
calls required for the deterministic tier.

### Deterministic tier (runs always in CI)

| Area | Test |
|------|------|
| Endpoint auth | `test_create_subscription_rejects_unauthenticated` |
| Endpoint auth | `test_billing_status_rejects_unauthenticated` |
| Endpoint shape | `test_billing_status_returns_none_for_fresh_user` |
| Webhook signing | `test_webhook_rejects_invalid_signature` |
| Webhook signing | `test_webhook_rejects_missing_signature` |
| First activation | `test_invoice_payment_succeeded_activates_bundle` |
| Idempotent webhook replay | `test_duplicate_invoice_webhook_is_idempotent` |
| Cancel-at-period-end | `test_subscription_updated_cancelling_transitions_state` |
| Past-due transition | `test_subscription_updated_past_due_transitions_state` |
| Hard cancellation | `test_subscription_deleted_marks_cancelled` |
| Renewal failure | `test_invoice_payment_failed_marks_past_due` |
| Out-of-order events | `test_events_out_of_order_converge_to_correct_state` |

These tests construct real Stripe-signed webhook events server-side (using
the same HMAC-SHA256 scheme Stripe uses), POST them to the mounted webhook
endpoint, and assert the resulting DB state. No network egress.

### Live-Stripe tier (runs when `STRIPE_SECRET_KEY` is set)

| Area | Test |
|------|------|
| End-to-end subscription creation + confirmation | `test_create_subscription_end_to_end_live_stripe` |
| Active-user idempotency | `test_create_subscription_is_idempotent_for_active_user` |

Skipped cleanly when the secret is absent. CI wires `STRIPE_TEST_SECRET_KEY`
and `STRIPE_TEST_LLM_BUNDLE_PRICE_ID` from repo secrets. Tests clean up every
Customer and Subscription they create via the `stripe_cleanup` fixture.

## What's NOT yet tested (Phase 2 — follow-up PR)

Items flagged in the original task spec as "manual verification needed" that
require infrastructure Phase 1 doesn't yet stand up:

1. **Frontend user journeys** — Playwright suite for the 11 journeys listed
   in the Layer 2 plan (happy path card, 3DS challenge, decline + retry,
   refresh mid-pay, refresh post-pay, back from payment view, BYOK path,
   already-onboarded redirect, session-expired during polling, test-mode
   pill visibility, wallet button rendering).
2. **Layer 3 live-webhook timing** — needs a public ingress for real
   `stripe listen` to forward events into the test backend. A nightly
   workflow targeting staging with wall-clock SLO assertions.
3. **Docker test stack** (`docker-compose.test.yml`) — boot frontend + backend
   + Postgres for E2E.
4. **Slack notification on nightly failure.**

Phase 2 needs a dedicated PR — Playwright setup alone is a multi-hour task
and deserves its own focused review.

## Running locally

```bash
cd backend
pip install -r requirements.txt
pip install pytest-cov

# Deterministic tests only
pytest tests/test_billing_subscription.py -v

# With live-Stripe tier (requires your own sk_test_* key)
STRIPE_SECRET_KEY=sk_test_... \
STRIPE_LLM_BUNDLE_PRICE_ID=price_... \
  pytest tests/test_billing_subscription.py -v
```

## Intentional regression test (smoke)

Verify the deterministic tier actually catches regressions by commenting
out the `agent_config.bundle_status = "active"` line in
`_handle_invoice_succeeded` ([vps.py:542](../app/api/vps.py#L542)). Run:

```bash
pytest tests/test_billing_subscription.py::test_invoice_payment_succeeded_activates_bundle -v
```

Should fail with an assertion error on `bundle_status == "active"`.

## Webhook routing note

Before this PR, the Stripe webhook endpoint (`POST /api/vps/webhook/stripe`
in [app/api/vps.py](../app/api/vps.py)) existed as code but was **never
mounted** in `platform_main.py`. The embedded subscription flow would have
hung indefinitely at "Activating your subscription..." because Stripe had
no route to POST events to. This PR mounts the vps router alongside
billing + checkout.
