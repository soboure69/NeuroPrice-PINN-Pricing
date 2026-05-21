"use client";

import { createContext, useContext, useEffect, useMemo, useState } from "react";

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
  signIn: (email: string, plan: Plan) => void;
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
  const [user, setUser] = useState<User | null>(null);
  const [usage, setUsage] = useState<Usage>({ month: currentMonth(), count: 0 });

  useEffect(() => {
    const savedUser = window.localStorage.getItem("neuroprice:user");
    const savedUsage = window.localStorage.getItem("neuroprice:usage");
    if (savedUser) {
      setUser(JSON.parse(savedUser));
    }
    if (savedUsage) {
      const parsedUsage = JSON.parse(savedUsage) as Usage;
      setUsage(parsedUsage.month === currentMonth() ? parsedUsage : { month: currentMonth(), count: 0 });
    }
  }, []);

  const quota = user ? quotas[user.plan] : quotas.free;
  const remaining = Math.max(quota - usage.count, 0);

  const value = useMemo<AuthContextValue>(() => ({
    user,
    usage,
    quota,
    remaining,
    signIn: (email: string, plan: Plan) => {
      const nextUser = { email, plan };
      setUser(nextUser);
      window.localStorage.setItem("neuroprice:user", JSON.stringify(nextUser));
    },
    signOut: () => {
      setUser(null);
      window.localStorage.removeItem("neuroprice:user");
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
  }), [quota, remaining, usage, user]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return context;
}
