export type Plan = "free" | "quant" | "enterprise";

export const planConfig: Record<Plan, { id: Plan; name: string; price: string; quota: number; features: string[] }> = {
  free: {
    id: "free",
    name: "Free",
    price: "0€",
    quota: 50,
    features: ["50 pricings / mois", "European calls", "Dashboard public"],
  },
  quant: {
    id: "quant",
    name: "Quant",
    price: "29€",
    quota: 10000,
    features: ["10k pricings / mois", "Options exotiques", "Cache Redis prioritaire"],
  },
  enterprise: {
    id: "enterprise",
    name: "Enterprise",
    price: "Sur devis",
    quota: 1000000,
    features: ["Batch pricing", "SLA API", "Déploiement dédié"],
  },
};

export const plans = [planConfig.free, planConfig.quant, planConfig.enterprise] as const;
