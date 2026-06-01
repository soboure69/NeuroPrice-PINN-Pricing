import crypto from "node:crypto";
import { NextResponse } from "next/server";

type StripeCheckoutSession = {
  customer_email?: string | null;
  metadata?: { plan?: string } | null;
};

type StripeWebhookEvent = {
  type?: string;
  data?: {
    object?: StripeCheckoutSession;
  };
};

const supportedPlans = new Set(["quant", "enterprise"]);

function verifyStripeSignature(payload: string, signatureHeader: string, secret: string) {
  const parts = signatureHeader.split(",").reduce<Record<string, string[]>>((acc, part) => {
    const [key, value] = part.split("=", 2);
    if (key && value) {
      acc[key] = [...(acc[key] ?? []), value];
    }
    return acc;
  }, {});
  const timestamp = parts.t?.[0];
  const signatures = parts.v1 ?? [];
  if (!timestamp || signatures.length === 0) {
    return false;
  }
  const signedPayload = `${timestamp}.${payload}`;
  const expectedSignature = crypto.createHmac("sha256", secret).update(signedPayload, "utf8").digest("hex");
  return signatures.some((signature) => {
    const expected = Buffer.from(expectedSignature, "hex");
    const actual = Buffer.from(signature, "hex");
    return expected.length === actual.length && crypto.timingSafeEqual(expected, actual);
  });
}

async function updateApiUserPlan(email: string, plan: string) {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL;
  const internalSecret = process.env.INTERNAL_API_SECRET;
  if (!apiUrl || !internalSecret) {
    throw new Error("Missing NEXT_PUBLIC_API_URL or INTERNAL_API_SECRET for user plan update.");
  }
  const response = await fetch(`${apiUrl.replace(/\/$/, "")}/api/v1/internal/users/plan`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-NeuroPrice-Internal-Secret": internalSecret,
    },
    body: JSON.stringify({ email, plan }),
  });
  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(`Plan update failed: ${response.status} ${errorBody}`);
  }
}

export async function POST(request: Request) {
  const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;
  if (!webhookSecret) {
    return NextResponse.json({ error: "Stripe webhook is not configured." }, { status: 500 });
  }

  const signatureHeader = request.headers.get("stripe-signature");
  if (!signatureHeader) {
    return NextResponse.json({ error: "Missing Stripe signature." }, { status: 400 });
  }

  const payload = await request.text();
  if (!verifyStripeSignature(payload, signatureHeader, webhookSecret)) {
    return NextResponse.json({ error: "Invalid Stripe signature." }, { status: 400 });
  }

  const event = JSON.parse(payload) as StripeWebhookEvent;
  if (event.type !== "checkout.session.completed") {
    return NextResponse.json({ received: true, ignored: event.type ?? "unknown" });
  }

  const session = event.data?.object;
  const email = session?.customer_email?.toLowerCase().trim();
  const plan = session?.metadata?.plan;

  if (!email || !plan || !supportedPlans.has(plan)) {
    return NextResponse.json({ error: "Missing or unsupported checkout session email/plan." }, { status: 422 });
  }

  await updateApiUserPlan(email, plan);
  return NextResponse.json({ received: true, email, plan });
}
