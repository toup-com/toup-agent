-- ── Who was charged twice for one image request? ─────────────────────────
--
-- Context: until `fix(chat): one tap, one turn` (toup-platform-app d14531833),
-- ChatScreen's `sendText` gated on the `isStreaming` STATE, which cannot be
-- true in the same tick as the first dispatch. Two taps therefore ran two agent
-- turns carrying two distinct `client_msg_id`s — legitimately distinct messages
-- as far as the exactly-once ledger is concerned — so each ran the image tool,
-- persisted its own attachment, and charged.
--
-- Nothing downstream could have collapsed them. The OpenAI path keys its charge
-- on `image_gen:{user_id}:{attachment_id}` and the Kie path on
-- `kie_task:{task_id}`; both are fresh per generation. The two charges are
-- real, distinct, correct-looking rows, identifiable only by their SHAPE —
-- same user, same price, minutes apart.
--
-- ⚠️ TWO CHARGE PATHS, and the default one does NOT write an `image_generation`
-- ledger row. `settings.image_provider` defaults to "kie", and on that path
-- `_tool_generate_image` returns at the Kie branch BEFORE its
-- `report_image_charge` block (tool_executor.py). The money moves in the LLM
-- proxy instead: `/kie/image/start` places a HOLD (a `credit_reservations` row
-- with `event_type='image_generation'`), and `/poll` settles it — writing only
-- `LEDGER_SETTLEMENT` (amount 0) and possibly `LEDGER_REFUND`. So a query over
-- `credit_ledger WHERE event_type='image_generation'` finds only the
-- OpenAI/BYO path and the legacy no-hold fallback, and would report zero
-- affected users on a fleet that is almost entirely Kie. The union below is the
-- point of this file.
--
-- ⚠️ READ-ONLY. This finds candidates; it refunds nothing. `credit_ledger` is
-- immutable by contract ("never UPDATE these rows. Corrections happen via
-- compensating inserts"), so remediation is an INSERT of a refund row with a
-- positive `amount` through the credit service — never an UPDATE or DELETE.
--
-- Window: an image turn runs ~60-90 s and the agent serialises turns per
-- session, so a duplicated pair lands roughly one generation apart. 10 minutes
-- is deliberately generous — someone legitimately asking for two variations of
-- one prompt will also appear, which is why this outputs a REVIEW LIST with the
-- gap spread, not a refund script.

WITH charges AS (
    -- (a) The DEFAULT path: the hold IS the charge; `settled_amount` is what
    --     the user actually paid once /poll clamped the estimate.
    SELECT
        user_id,
        id                                AS charge_id,
        'reservation'                     AS source,
        COALESCE(settled_amount, estimated_amount) AS credits,
        created_at,
        metadata ->> 'mode'               AS mode,
        idempotency_key
    FROM credit_reservations
    WHERE event_type = 'image_generation'
      AND status = 'settled'
      AND created_at >= NOW() - INTERVAL '90 days'

    UNION ALL

    -- (b) OpenAI / BYO, and the legacy Kie path that charges directly when no
    --     hold is found. A charge is a NEGATIVE ledger amount.
    SELECT
        user_id,
        id                                AS charge_id,
        'ledger'                          AS source,
        ABS(amount)                       AS credits,
        created_at,
        metadata ->> 'size'               AS mode,
        idempotency_key
    FROM credit_ledger
    WHERE event_type = 'image_generation'
      AND amount < 0
      AND created_at >= NOW() - INTERVAL '90 days'
),
paired AS (
    SELECT
        a.user_id,
        a.charge_id                        AS first_charge_id,
        b.charge_id                        AS second_charge_id,
        a.source,
        a.created_at                       AS first_at,
        b.created_at                       AS second_at,
        EXTRACT(EPOCH FROM (b.created_at - a.created_at)) AS gap_seconds,
        a.credits                          AS credits_each
    FROM charges a
    JOIN charges b
      ON b.user_id = a.user_id
     AND b.created_at > a.created_at
     AND b.created_at <= a.created_at + INTERVAL '10 minutes'
     -- Same price and same shape. A different size/mode is a different ask,
     -- however close together it landed.
     AND b.credits = a.credits
     AND b.mode IS NOT DISTINCT FROM a.mode
     AND b.charge_id <> a.charge_id
)
SELECT
    user_id,
    COUNT(*)                    AS suspect_pairs,
    SUM(credits_each)           AS credits_to_refund,
    MIN(first_at)               AS earliest,
    MAX(second_at)              AS latest,
    -- The discriminator to eye before refunding anyone: a genuine double
    -- dispatch produces a TIGHT, repeated gap, because the gap IS the turn
    -- length (the agent serialises per session). A person iterating on a
    -- prompt does not — their gaps scatter with how long they looked at the
    -- result before asking again.
    ROUND(MIN(gap_seconds))     AS min_gap_seconds,
    ROUND(AVG(gap_seconds))     AS avg_gap_seconds,
    ROUND(MAX(gap_seconds))     AS max_gap_seconds,
    ROUND(STDDEV_POP(gap_seconds)) AS gap_stddev
FROM paired
GROUP BY user_id
ORDER BY credits_to_refund DESC;

-- Per-pair detail for one user, once the summary above has narrowed it:
--
--   SELECT * FROM paired WHERE user_id = '<uuid>' ORDER BY first_at;
--
-- The decisive cross-check is in the TENANT database rather than this one: two
-- assistant messages in the same day chat, minutes apart, with byte-identical
-- `content` and two different attachment ids. THAT is a duplicated turn, which
-- is the thing being remediated. Two charges whose messages differ are a user
-- who asked twice and should not be refunded.
--
--   SELECT id, created_at, LEFT(content, 80) AS preview, attachments
--   FROM messages
--   WHERE role = 'assistant'
--     AND created_at BETWEEN '<first_at>'::timestamp  - INTERVAL '5 minutes'
--                        AND '<second_at>'::timestamp + INTERVAL '5 minutes'
--   ORDER BY created_at;
