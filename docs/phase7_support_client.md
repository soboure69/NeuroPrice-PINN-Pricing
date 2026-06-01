# Phase 7 Support Client

## MVP support

The initial customer support implementation uses email-based support.

Frontend component:

```text
frontend/components/SupportWidget.tsx
```

Landing integration:

```text
frontend/app/page.tsx#support
```

## Environment variable

Set this variable in the frontend environment:

```text
NEXT_PUBLIC_SUPPORT_EMAIL=support@your-domain.com
```

If the variable is missing, the frontend falls back to:

```text
support@neuroprice.app
```

## User context

The support email body is prefilled with:

```text
user email
current plan
free-form issue description
```

## Analytics

The support widget tracks:

```text
support_contact_clicked
```

Properties:

```text
channel
plan
signed_in
```

## Future Intercom upgrade

When support volume increases, replace or complement email with Intercom:

1. Add Intercom workspace.
2. Add `NEXT_PUBLIC_INTERCOM_APP_ID`.
3. Load Intercom only on client side.
4. Identify signed-in users with email and plan.
5. Route Enterprise users to priority support.
