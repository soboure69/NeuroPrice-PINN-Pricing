"use client";

import { SessionProvider, signIn as nextAuthSignIn, signOut as nextAuthSignOut, useSession } from "next-auth/react";
import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { identifyUser, resetAnalytics, trackEvent } from "@/components/analytics";

type Plan = "free" | "quant" | "enterprise";

type User = {
  email: string;
  plan: Plan;
};

type Usage = {
  month: string;
  count: number;
};

type AuthContextValue = {
  user: User | null;
  usage: Usage;
  quota: number;
  remaining: number;
  plan: Plan;
  setPlan: (plan: Plan) => void;
  signIn: () => void;
  signOut: () => void;
  consumeQuota: () => boolean;
};

const quotas: Record<Plan, number> = {
  free: 50,
  quant: 10000,
  enterprise: 1000000,
};

const AuthContext = createContext<AuthContextValue | null>(null);

function currentMonth() {
  return new Date().toISOString().slice(0, 7);
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  return <SessionProvider><AuthStateProvider>{children}</AuthStateProvider></SessionProvider>;
}

function AuthStateProvider({ children }: { children: React.ReactNode }) {
  const { data: session } = useSession();
  const [plan, setPlanState] = useState<Plan>("free");
  const [user, setUser] = useState<User | null>(null);
  const [usage, setUsage] = useState<Usage>({ month: currentMonth(), count: 0 });

  useEffect(() => {
    const savedPlan = window.localStorage.getItem("neuroprice:plan") as Plan | null;
    const savedUsage = window.localStorage.getItem("neuroprice:usage");
    if (savedPlan && savedPlan in quotas) {
      setPlanState(savedPlan);
    }
    if (savedUsage) {
      const parsedUsage = JSON.parse(savedUsage) as Usage;
      setUsage(parsedUsage.month === currentMonth() ? parsedUsage : { month: currentMonth(), count: 0 });
    }
  }, []);

  useEffect(() => {
    const email = session?.user?.email;
    if (email) {
      setUser({ email, plan });
      identifyUser(email, { plan });
    } else {
      setUser(null);
    }
  }, [plan, session?.user?.email]);

  const quota = quotas[plan];
  const remaining = Math.max(quota - usage.count, 0);

  const value = useMemo<AuthContextValue>(() => ({
    user,
    usage,
    quota,
    remaining,
    plan,
    setPlan: (nextPlan: Plan) => {
      setPlanState(nextPlan);
      window.localStorage.setItem("neuroprice:plan", nextPlan);
      trackEvent("plan_selected", { plan: nextPlan });
    },
    signIn: () => {
      trackEvent("user_sign_in_started", { provider: "google", plan });
      nextAuthSignIn("google");
    },
    signOut: () => {
      trackEvent("user_signed_out", { plan });
      resetAnalytics();
      setUser(null);
      nextAuthSignOut();
    },
    consumeQuota: () => {
      if (remaining <= 0) {
        return false;
      }
      const nextUsage = { month: currentMonth(), count: usage.month === currentMonth() ? usage.count + 1 : 1 };
      setUsage(nextUsage);
      window.localStorage.setItem("neuroprice:usage", JSON.stringify(nextUsage));
      return true;
    },
  }), [plan, quota, remaining, usage, user]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return context;
}
