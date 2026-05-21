import { ArrowRight, BrainCircuit, CheckCircle2, Cloud, Cpu, Shield } from "lucide-react";
import type { ReactNode } from "react";
import { AuthQuotaPanel } from "@/components/AuthQuotaPanel";
import { PricingDashboard } from "@/components/PricingDashboard";

const plans = [
  { name: "Free", price: "0€", features: ["50 pricings / mois", "European calls", "Dashboard public"] },
  { name: "Quant", price: "29€", features: ["10k pricings / mois", "Options exotiques", "Cache Redis prioritaire"] },
  { name: "Enterprise", price: "Sur devis", features: ["Batch pricing", "SLA API", "Déploiement dédié"] },
];

export default function Home() {
  return (
    <main>
      <nav className="mx-auto flex max-w-7xl items-center justify-between px-6 py-6">
        <div className="flex items-center gap-3 text-xl font-bold">
          <div className="rounded-2xl bg-accent/10 p-2 text-accent"><BrainCircuit size={24} /></div>
          NeuroPrice
        </div>
        <div className="hidden items-center gap-6 text-sm text-muted md:flex">
          <a href="#product" className="hover:text-white">Produit</a>
          <a href="#dashboard" className="hover:text-white">Dashboard</a>
          <a href="#pricing" className="hover:text-white">Plans</a>
        </div>
        <a href="#dashboard" className="rounded-full bg-white px-4 py-2 text-sm font-semibold text-slate-950">Essayer</a>
      </nav>

      <section id="product" className="mx-auto grid max-w-7xl items-center gap-10 px-6 py-20 lg:grid-cols-[1.1fr_0.9fr]">
        <div>
          <div className="mb-6 inline-flex rounded-full border border-sky-400/30 bg-sky-400/10 px-4 py-2 text-sm text-sky-200">PINN pricing engine · API + Dashboard SaaS</div>
          <h1 className="max-w-4xl text-5xl font-bold tracking-tight text-white md:text-7xl">Pricer des options complexes en quelques millisecondes.</h1>
          <p className="mt-6 max-w-2xl text-lg leading-8 text-muted">NeuroPrice transforme les modèles PINN, Monte Carlo et Black-Scholes en un produit utilisable par desks quant, équipes risk et étudiants finance quantitative.</p>
          <div className="mt-8 flex flex-col gap-3 sm:flex-row">
            <a href="#dashboard" className="inline-flex items-center justify-center gap-2 rounded-xl bg-accent px-6 py-3 font-semibold text-slate-950 hover:bg-sky-300">Lancer un pricing <ArrowRight size={18} /></a>
            <a href="http://127.0.0.1:8000/docs" className="inline-flex items-center justify-center rounded-xl border border-slate-700 px-6 py-3 font-semibold text-white hover:border-sky-400">Swagger API</a>
          </div>
        </div>

        <div className="rounded-3xl border border-slate-800 bg-panel/80 p-6 shadow-glow">
          <div className="grid gap-4">
            <Feature icon={<Cpu />} title="PINN + références" text="Vanilla, barrière, asiatique et lookback." />
            <Feature icon={<Shield />} title="API testée" text="FastAPI, Redis, Docker, CI et tests Locust." />
            <Feature icon={<Cloud />} title="Cloud-ready" text="Préparé pour Render/Railway + Vercel." />
          </div>
        </div>
      </section>

      <AuthQuotaPanel />

      <PricingDashboard />

      <section id="pricing" className="mx-auto max-w-7xl px-6 py-16">
        <div className="mb-10 text-center">
          <h2 className="text-4xl font-bold text-white">Plans SaaS MVP</h2>
          <p className="mt-3 text-muted">Freemium, quota et montée en gamme pour utilisateurs beta.</p>
        </div>
        <div className="grid gap-6 md:grid-cols-3">
          {plans.map((plan) => (
            <div key={plan.name} className="rounded-3xl border border-slate-800 bg-panel/80 p-6">
              <h3 className="text-2xl font-semibold text-white">{plan.name}</h3>
              <p className="mt-3 text-4xl font-bold text-accent">{plan.price}</p>
              <ul className="mt-6 space-y-3 text-sm text-slate-300">
                {plan.features.map((feature) => (
                  <li key={feature} className="flex items-center gap-2"><CheckCircle2 className="text-success" size={18} />{feature}</li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}

function Feature({ icon, title, text }: { icon: ReactNode; title: string; text: string }) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-5">
      <div className="mb-3 text-accent">{icon}</div>
      <h3 className="text-lg font-semibold text-white">{title}</h3>
      <p className="mt-1 text-sm text-muted">{text}</p>
    </div>
  );
}
