"use client";

import { LifeBuoy, Mail, MessageCircle, Send } from "lucide-react";
import { trackEvent } from "@/components/analytics";
import { useAuth } from "@/components/AuthProvider";

const supportEmail = process.env.NEXT_PUBLIC_SUPPORT_EMAIL ?? "support@neuroprice.app";

export function SupportWidget() {
  const { user, plan } = useAuth();
  const subject = encodeURIComponent("Support NeuroPrice");
  const body = encodeURIComponent([
    "Bonjour NeuroPrice,",
    "",
    "J'ai besoin d'aide sur :",
    "",
    `Email utilisateur : ${user?.email ?? "non connecté"}`,
    `Plan : ${user?.plan ?? plan}`,
    "",
    "Description :",
  ].join("\n"));
  const mailtoHref = `mailto:${supportEmail}?subject=${subject}&body=${body}`;

  function handleSupportClick(channel: string) {
    trackEvent("support_contact_clicked", { channel, plan: user?.plan ?? plan, signed_in: Boolean(user) });
  }

  return (
    <section id="support" className="mx-auto max-w-7xl px-6 py-16">
      <div className="grid gap-6 rounded-3xl border border-slate-800 bg-panel/80 p-6 shadow-glow lg:grid-cols-[1fr_360px] lg:items-center">
        <div>
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-sky-400/30 bg-sky-400/10 px-3 py-1 text-sm text-sky-200">
            <LifeBuoy size={16} /> Support client
          </div>
          <h2 className="text-3xl font-bold text-white">Besoin d'aide sur un pricing, un quota ou un abonnement ?</h2>
          <p className="mt-3 max-w-2xl text-sm leading-6 text-muted">
            Le MVP support démarre par email avec contexte utilisateur prérempli. Intercom peut être ajouté ensuite quand le volume de tickets augmente.
          </p>
        </div>

        <div className="grid gap-3">
          <a onClick={() => handleSupportClick("email")} href={mailtoHref} className="inline-flex items-center justify-center gap-2 rounded-xl bg-accent px-5 py-3 font-semibold text-slate-950 hover:bg-sky-300">
            <Mail size={18} /> Contacter le support
          </a>
          <a onClick={() => handleSupportClick("sales")} href={`mailto:${supportEmail}?subject=${encodeURIComponent("Contact Enterprise NeuroPrice")}`} className="inline-flex items-center justify-center gap-2 rounded-xl border border-slate-700 px-5 py-3 font-semibold text-white hover:border-sky-400">
            <Send size={18} /> Contact Enterprise
          </a>
          <div className="rounded-2xl border border-slate-800 bg-slate-950/50 p-4 text-sm text-muted">
            <div className="mb-2 flex items-center gap-2 text-slate-200"><MessageCircle size={16} /> SLA MVP</div>
            <p>Réponse email cible : 24-48h ouvrées. Priorité aux plans Quant et Enterprise.</p>
          </div>
        </div>
      </div>
    </section>
  );
}
