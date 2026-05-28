"use client";

import { useAuth } from "@/components/AuthProvider";

type Plan = "free" | "quant" | "enterprise";

export function AuthQuotaPanel() {
  const { user, usage, quota, remaining, plan, setPlan, signIn, signOut } = useAuth();

  const usedPercent = Math.min((usage.count / quota) * 100, 100);

  return (
    <section className="mx-auto max-w-7xl px-6 pt-6">
      <div className="rounded-3xl border border-slate-800 bg-panel/80 p-6 shadow-glow">
        <div className="grid gap-5 lg:grid-cols-[1fr_420px] lg:items-center">
          <div>
            <div className="mb-3 flex items-center gap-3 text-accent">
              <span className="flex h-8 w-8 items-center justify-center rounded-full bg-accent/10 text-sm font-bold">U</span>
              <span className="text-sm font-semibold uppercase tracking-[0.2em]">Auth production & quotas</span>
            </div>
            <h2 className="text-2xl font-semibold text-white">{user ? `Connecté : ${user.email}` : "Connexion Google"}</h2>
            <p className="mt-2 text-sm text-muted">La session utilise NextAuth Google. Les quotas mensuels sont contrôlés côté API avec PostgreSQL Neon.</p>
          </div>

          {user ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted">Plan {user.plan}</span>
                <span className="font-semibold text-white">{remaining} / {quota} restants</span>
              </div>
              <div className="h-3 overflow-hidden rounded-full bg-slate-800">
                <div className="h-full rounded-full bg-accent" style={{ width: `${usedPercent}%` }} />
              </div>
              <button onClick={signOut} className="inline-flex w-full items-center justify-center gap-2 rounded-xl border border-slate-700 px-4 py-3 font-semibold text-white hover:border-sky-400">
                Déconnexion
              </button>
            </div>
          ) : (
            <div className="grid gap-3">
              <label className="grid gap-2 text-sm font-medium text-slate-300">
                Plan
                <select value={plan} onChange={(event) => setPlan(event.target.value as Plan)} className="rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-accent">
                  <option value="free">Free — 50 pricings / mois</option>
                  <option value="quant">Quant — 10k pricings / mois</option>
                  <option value="enterprise">Enterprise — quota élevé</option>
                </select>
              </label>
              <button onClick={signIn} className="inline-flex items-center justify-center gap-2 rounded-xl bg-accent px-4 py-3 font-semibold text-slate-950 hover:bg-sky-300">
                Continuer avec Google
              </button>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
