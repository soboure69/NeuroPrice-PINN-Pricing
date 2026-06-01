import { NextResponse } from "next/server";

type CheckoutPlan = "quant" | "enterprise";

const priceEnvByPlan: Record<CheckoutPlan, string> = {
  quant: "STRIPE_QUANT_PRICE_ID",
  enterprise: "STRIPE_ENTERPRISE_PRICE_ID",
};

function getBaseUrl() {
  return process.env.NEXT_PUBLIC_APP_URL ?? "http://localhost:3000";
}

export async function POST(request: Request) {
  const stripeSecretKey = process.env.STRIPE_SECRET_KEY;
  if (!stripeSecretKey) {
    return NextResponse.json({ error: "Stripe is not configured." }, { status: 500 });
  }

  const body = await request.json().catch(() => null) as { plan?: string; email?: string } | null;
  const plan = body?.plan;

  if (plan !== "quant" && plan !== "enterprise") {
    return NextResponse.json({ error: "Unsupported checkout plan." }, { status: 400 });
  }

  const priceId = process.env[priceEnvByPlan[plan]];
  if (!priceId) {
    return NextResponse.json({ error: `Stripe price is not configured for plan ${plan}.` }, { status: 500 });
  }

  const baseUrl = getBaseUrl().replace(/\/$/, "");
  const params = new URLSearchParams({
    mode: "subscription",
    "line_items[0][price]": priceId,
    "line_items[0][quantity]": "1",
    success_url: `${baseUrl}/?checkout=success&plan=${plan}`,
    cancel_url: `${baseUrl}/?checkout=cancelled&plan=${plan}`,
    "metadata[plan]": plan,
    "subscription_data[metadata][plan]": plan,
  });

  if (body?.email) {
    params.set("customer_email", body.email);
  }

  const stripeResponse = await fetch("https://api.stripe.com/v1/checkout/sessions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${stripeSecretKey}`,
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: params,
  });
  const session = await stripeResponse.json() as { url?: string; error?: { message?: string } };

  if (!stripeResponse.ok || !session.url) {
    return NextResponse.json({ error: session.error?.message ?? "Stripe checkout session creation failed." }, { status: 502 });
  }

  return NextResponse.json({ url: session.url });
}
