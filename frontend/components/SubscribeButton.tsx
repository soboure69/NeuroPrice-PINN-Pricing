"use client";

import { useState } from "react";
import { trackEvent } from "@/components/analytics";
import { useAuth } from "@/components/AuthProvider";

type Plan = "free" | "quant" | "enterprise";

export function SubscribeButton({ plan }: { plan: Plan }) {
  const { user, setPlan } = useAuth();
  const [loading, setLoading] = useState(false);
  const isFree = plan === "free";

  async function handleClick() {
    if (isFree) {
      setPlan("free");
      return;
    }

    setLoading(true);
    trackEvent("stripe_checkout_started", { plan, email: user?.email });
    try {
      const response = await fetch("/api/stripe/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ plan, email: user?.email }),
      });
      const data = await response.json().catch(() => null) as { url?: string; error?: string } | null;
      if (!response.ok || !data?.url) {
        throw new Error(data?.error ?? "Impossible de démarrer Stripe Checkout.");
      }
      window.location.href = data.url;
    } catch (exc) {
      const message = exc instanceof Error ? exc.message : "Erreur Stripe inconnue";
      trackEvent("stripe_checkout_error", { plan, message });
      alert(message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <button
      onClick={handleClick}
      disabled={loading}
      className="mt-6 inline-flex w-full items-center justify-center rounded-xl bg-accent px-4 py-3 font-semibold text-slate-950 hover:bg-sky-300 disabled:cursor-not-allowed disabled:opacity-60"
    >
      {loading ? "Redirection Stripe..." : isFree ? "Démarrer gratuitement" : "S'abonner"}
    </button>
  );
}
