"use client";

import { BarChart3, Euro, RefreshCw, Users, Zap } from "lucide-react";
import { useState } from "react";

type AdminSummary = {
  backend: string;
  users_total: number;
  usage_current_month: number;
  mrr_eur: number;
  plans: { plan: string; users: number }[];
  recent_users: { email: string; plan: string; created_at: string; updated_at: string }[];
};

export function AdminDashboard() {
  const [summary, setSummary] = useState<AdminSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function loadSummary() {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("/api/admin/summary", { cache: "no-store" });
      const data = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(data?.error ?? `Erreur admin ${response.status}`);
      }
      setSummary(data);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "Erreur inconnue");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section id="admin" className="mx-auto max-w-7xl px-6 py-16">
      <div className="rounded-3xl border border-slate-800 bg-panel/80 p-6 shadow-glow">
        <div className="mb-6 flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-sky-400/30 bg-sky-400/10 px-3 py-1 text-sm text-sky-200">
              <BarChart3 size={16} /> Admin MVP
            </div>
            <h2 className="text-3xl font-bold text-white">Dashboard admin</h2>
            <p className="mt-2 text-sm text-muted">Analytics utilisateurs, consommation API et MRR estimé.</p>
          </div>
          <button onClick={loadSummary} disabled={loading} className="inline-flex items-center justify-center gap-2 rounded-xl bg-accent px-5 py-3 font-semibold text-slate-950 hover:bg-sky-300 disabled:opacity-70">
            <RefreshCw className={loading ? "animate-spin" : ""} size={18} />
            {summary ? "Rafraîchir" : "Charger"}
          </button>
        </div>

        {error && <div className="mb-6 rounded-2xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-200">{error}</div>}

        <div className="grid gap-4 md:grid-cols-4">
          <AdminMetric icon={<Users size={20} />} title="Utilisateurs" value={summary ? String(summary.users_total) : "—"} />
          <AdminMetric icon={<Zap size={20} />} title="Pricings ce mois" value={summary ? String(summary.usage_current_month) : "—"} />
          <AdminMetric icon={<Euro size={20} />} title="MRR estimé" value={summary ? `${summary.mrr_eur}€` : "—"} />
          <AdminMetric icon={<BarChart3 size={20} />} title="Backend quota" value={summary?.backend ?? "—"} />
        </div>

        <div className="mt-6 grid gap-6 lg:grid-cols-[360px_1fr]">
          <div className="rounded-2xl border border-slate-800 bg-slate-950/50 p-5">
            <h3 className="mb-4 text-lg font-semibold text-white">Répartition des plans</h3>
            <div className="space-y-3">
              {summary?.plans.length ? summary.plans.map((item) => (
                <div key={item.plan} className="flex items-center justify-between rounded-xl border border-slate-800 px-4 py-3 text-sm">
                  <span className="capitalize text-slate-300">{item.plan}</span>
                  <span className="font-semibold text-white">{item.users}</span>
                </div>
              )) : <p className="text-sm text-muted">Aucune donnée chargée.</p>}
            </div>
          </div>

          <div className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-950/50 p-5">
            <h3 className="mb-4 text-lg font-semibold text-white">Derniers utilisateurs</h3>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[640px] text-left text-sm">
                <thead className="text-muted">
                  <tr>
                    <th className="pb-3 font-medium">Email</th>
                    <th className="pb-3 font-medium">Plan</th>
                    <th className="pb-3 font-medium">Créé</th>
                    <th className="pb-3 font-medium">Mis à jour</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800">
                  {summary?.recent_users.length ? summary.recent_users.map((user) => (
                    <tr key={user.email}>
                      <td className="py-3 text-white">{user.email}</td>
                      <td className="py-3 capitalize text-accent">{user.plan}</td>
                      <td className="py-3 text-slate-300">{formatDate(user.created_at)}</td>
                      <td className="py-3 text-slate-300">{formatDate(user.updated_at)}</td>
                    </tr>
                  )) : (
                    <tr><td className="py-3 text-muted" colSpan={4}>Aucun utilisateur chargé.</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function AdminMetric({ icon, title, value }: { icon: React.ReactNode; title: string; value: string }) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-950/50 p-5">
      <div className="mb-3 text-accent">{icon}</div>
      <p className="text-sm text-muted">{title}</p>
      <p className="mt-1 text-2xl font-semibold text-white">{value}</p>
    </div>
  );
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat("fr-FR", { dateStyle: "short", timeStyle: "short" }).format(new Date(value));
}
