"use client";

import { Activity, Calculator, Gauge, LineChart } from "lucide-react";
import { useMemo, useState } from "react";
import { useAuth } from "@/components/AuthProvider";
import { PriceSurfacePlot } from "@/components/PriceSurfacePlot";

type Instrument = "european_call" | "down_out_barrier_call" | "asian_arithmetic_call" | "lookback_floating_call";

type PricingResponse = {
  instrument: Instrument;
  price: number;
  method: string;
  model_version: string;
  inference_time_ms: number;
  greeks: Record<string, number> | null;
  warnings: string[];
};

const instruments: { value: Instrument; label: string }[] = [
  { value: "european_call", label: "European Call" },
  { value: "down_out_barrier_call", label: "Down-and-Out Barrier" },
  { value: "asian_arithmetic_call", label: "Asian Arithmetic" },
  { value: "lookback_floating_call", label: "Lookback Floating" },
];

const apiUrl = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

export function PricingDashboard() {
  const { user, remaining, consumeQuota } = useAuth();
  const [instrument, setInstrument] = useState<Instrument>("european_call");
  const [S0, setS0] = useState(100);
  const [K, setK] = useState(100);
  const [sigma, setSigma] = useState(0.2);
  const [r, setR] = useState(0.05);
  const [T, setT] = useState(1);
  const [barrier, setBarrier] = useState(70);
  const [result, setResult] = useState<PricingResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const surface = useMemo(() => {
    const spotGrid = Array.from({ length: 18 }, (_, index) => 50 + index * 7.5);
    const maturityGrid = Array.from({ length: 12 }, (_, index) => 0.1 + index * 0.18);
    const z = maturityGrid.map((tau) => spotGrid.map((spot) => Math.max(spot - K * Math.exp(-r * tau), 0) * Math.exp(-0.5 * sigma * tau)));
    return { spotGrid, maturityGrid, z };
  }, [K, r, sigma]);

  const priceSurfacePlot = useMemo(
    () => <PriceSurfacePlot spotGrid={surface.spotGrid} maturityGrid={surface.maturityGrid} z={surface.z} />,
    [surface],
  );

  async function handleSubmit() {
    if (!user) {
      setError("Connecte-toi avec le panneau Auth & quotas avant de lancer un pricing.");
      return;
    }
    if (remaining <= 0) {
      setError("Quota mensuel épuisé pour ce plan. Sélectionne un plan supérieur ou réinitialise le mois prochain.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const endpoint = `${apiUrl}/api/v1/price`;
      const payload: Record<string, number | string | boolean> = { instrument, S0, sigma, r, T, greeks: true };
      if (instrument !== "lookback_floating_call") {
        payload.K = K;
      }
      if (instrument === "down_out_barrier_call") {
        payload.barrier = barrier;
      }
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(data?.detail ?? `Erreur API ${response.status} sur ${endpoint}`);
      }
      setResult(data);
      consumeQuota();
    } catch (exc) {
      if (exc instanceof TypeError) {
        setError(`API inaccessible à ${apiUrl}. Vérifie que FastAPI tourne sur le port 8000 et que NEXT_PUBLIC_API_URL pointe vers la bonne URL.`);
      } else {
        setError(exc instanceof Error ? exc.message : "Unknown pricing error");
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <section id="dashboard" className="mx-auto grid max-w-7xl gap-6 px-6 py-16 lg:grid-cols-[420px_1fr]">
      <div className="rounded-3xl border border-slate-800 bg-panel/80 p-6 shadow-glow backdrop-blur">
        <div className="mb-6 flex items-center gap-3">
          <div className="rounded-2xl bg-accent/10 p-3 text-accent">
            <Calculator size={24} />
          </div>
          <div>
            <h2 className="text-2xl font-semibold">Pricing dashboard</h2>
            <p className="text-sm text-muted">Connecté à {apiUrl}</p>
          </div>
        </div>

        <div className="space-y-5">
          <label className="block text-sm font-medium text-slate-300">
            Instrument
            <select value={instrument} onChange={(event) => setInstrument(event.target.value as Instrument)} className="mt-2 w-full rounded-xl border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-accent">
              {instruments.map((item) => (
                <option key={item.value} value={item.value}>{item.label}</option>
              ))}
            </select>
          </label>

          <Slider label="Spot S0" value={S0} min={40} max={220} step={1} onChange={setS0} />
          {instrument !== "lookback_floating_call" && <Slider label="Strike K" value={K} min={40} max={220} step={1} onChange={setK} />}
          {instrument === "down_out_barrier_call" && <Slider label="Barrier B" value={barrier} min={20} max={Math.max(30, K - 1)} step={1} onChange={setBarrier} />}
          <Slider label="Volatility σ" value={sigma} min={0.05} max={0.8} step={0.01} onChange={setSigma} />
          <Slider label="Rate r" value={r} min={0} max={0.15} step={0.005} onChange={setR} />
          <Slider label="Maturity T" value={T} min={0.1} max={5} step={0.1} onChange={setT} />

          <button onClick={handleSubmit} disabled={loading} className="flex w-full items-center justify-center gap-2 rounded-xl bg-accent px-5 py-3 font-semibold text-slate-950 transition hover:bg-sky-300 disabled:cursor-not-allowed disabled:opacity-70">
            <Activity className={loading ? "animate-spin" : ""} size={18} />
            <span>{loading ? "Pricing..." : "Pricer l'instrument"}</span>
          </button>
          <p className="text-xs text-muted">{user ? `${remaining} pricing(s) restants sur ton plan ${user.plan}.` : "Connexion requise pour consommer l'API."}</p>
        </div>
      </div>

      <div className="grid gap-6">
        <div className="grid gap-4 md:grid-cols-3">
          <Metric title="Price" value={result ? result.price.toFixed(4) : "—"} icon={<Gauge size={20} />} />
          <Metric title="Inference" value={result ? `${result.inference_time_ms.toFixed(2)} ms` : "—"} icon={<Activity size={20} />} />
          <Metric title="Model" value={result ? result.model_version : "—"} icon={<LineChart size={20} />} />
        </div>

        {error && <div className="rounded-2xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-200">{error}</div>}

        <div className="rounded-3xl border border-slate-800 bg-panel/80 p-6">
          <h3 className="mb-4 text-xl font-semibold">Price surface preview</h3>
          {priceSurfacePlot}
        </div>

        <div className="rounded-3xl border border-slate-800 bg-panel/80 p-6">
          <h3 className="mb-4 text-xl font-semibold">Greeks</h3>
          <div className="grid gap-3 md:grid-cols-2">
            {result?.greeks ? Object.entries(result.greeks).map(([key, value]) => <Metric key={key} title={key.toUpperCase()} value={value.toFixed(6)} />) : <p className="text-muted">Les Greeks apparaissent pour les instruments supportés.</p>}
          </div>
        </div>
      </div>
    </section>
  );
}

function Slider({ label, value, min, max, step, onChange }: { label: string; value: number; min: number; max: number; step: number; onChange: (value: number) => void }) {
  return (
    <label className="block text-sm font-medium text-slate-300">
      <span className="flex justify-between"><span>{label}</span><span className="text-accent">{value}</span></span>
      <input type="range" value={value} min={min} max={max} step={step} onChange={(event) => onChange(Number(event.target.value))} className="mt-2 w-full accent-sky-400" />
    </label>
  );
}

function Metric({ title, value, icon }: { title: string; value: string; icon?: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-5">
      <div className="mb-2 flex items-center gap-2 text-sm text-muted">{icon}<span>{title}</span></div>
      <div className="truncate text-2xl font-semibold text-white">{value}</div>
    </div>
  );
}
