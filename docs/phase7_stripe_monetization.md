# Phase 7 — Stripe monetization setup

## Scope

This document describes the MVP Stripe integration for NeuroPrice subscriptions.

Implemented flow:

```text
Pricing card -> Next.js checkout route -> Stripe Checkout subscription -> success/cancel redirect
```

## Frontend route

Checkout route:

```text
frontend/app/api/stripe/checkout/route.ts
```

The route creates a Stripe Checkout session in subscription mode.

Supported plans:

- `quant`
- `enterprise`

The `free` plan does not create a Stripe Checkout session.

## Required environment variables

Set these variables in the frontend deployment environment:

```text
STRIPE_SECRET_KEY=sk_test_...
STRIPE_QUANT_PRICE_ID=price_...
STRIPE_ENTERPRISE_PRICE_ID=price_...
STRIPE_WEBHOOK_SECRET=whsec_...
INTERNAL_API_SECRET=change_me_long_random_secret
NEXT_PUBLIC_APP_URL=http://localhost:3000
NEXT_PUBLIC_API_URL=http://localhost:8000
```

For production:

```text
NEXT_PUBLIC_APP_URL=https://your-production-domain.com
```

## Security notes

- `STRIPE_SECRET_KEY` must remain server-side only.
- Do not prefix the secret key with `NEXT_PUBLIC_`.
- Price IDs can be public, but they are currently read server-side.
- Webhook signature verification is required through `STRIPE_WEBHOOK_SECRET` before automatically upgrading user plans.

## Current limitation

The implementation now includes a Stripe webhook endpoint.

Webhook route:

```text
frontend/app/api/stripe/webhook/route.ts
```

Internal API route called by the webhook:

```text
POST /api/v1/internal/users/plan
```

The internal API route requires:

```text
X-NeuroPrice-Internal-Secret: INTERNAL_API_SECRET
```

## Webhook behavior

Supported Stripe event:

```text
checkout.session.completed
```

The webhook reads:

```text
session.customer_email
session.metadata.plan
```

Then it updates the API quota database:

```text
users.plan = quant / enterprise
```

## Local webhook test with Stripe CLI

Install and login to Stripe CLI, then run:

```bash
stripe listen --forward-to localhost:3000/api/stripe/webhook
```

Stripe CLI will print a webhook signing secret:

```text
whsec_...
```

Copy it to:

```text
STRIPE_WEBHOOK_SECRET=whsec_...
```

Then restart the frontend:

```bash
npm run dev
```

Complete a Checkout flow. The webhook should call the API and persist the paid plan.

## API prerequisites

The FastAPI backend must be running and reachable at:

```text
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Manual test steps

1. Create products and recurring prices in Stripe dashboard.
2. Add the required environment variables.
3. Start the frontend.
4. Click `S'abonner` on the Quant plan.
5. Complete checkout with Stripe test card.
6. Verify redirect to `/?checkout=success&plan=quant`.
7. Verify the Stripe CLI receives `checkout.session.completed`.
8. Verify the backend user plan is updated.

## Stripe test card

Use Stripe's standard test card:

```text
4242 4242 4242 4242
```

Use any future expiry date, any CVC and any postal code.
