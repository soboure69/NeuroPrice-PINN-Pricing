from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime

try:
    import psycopg
except ImportError:
    psycopg = None


PLAN_QUOTAS = {
    "free": 50,
    "quant": 10000,
    "enterprise": 1000000,
}

PLAN_DETAILS = {
    "free": {
        "name": "Free",
        "price": "0€",
        "features": ["50 pricings / mois", "European calls", "Dashboard public"],
    },
    "quant": {
        "name": "Quant",
        "price": "29€",
        "features": ["10k pricings / mois", "Options exotiques", "Cache Redis prioritaire"],
    },
    "enterprise": {
        "name": "Enterprise",
        "price": "Sur devis",
        "features": ["Batch pricing", "SLA API", "Déploiement dédié"],
    },
}


@dataclass(frozen=True)
class QuotaStatus:
    backend: str
    allowed: bool
    plan: str
    quota: int
    used: int
    remaining: int


class QuotaStore:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self.backend = "postgres" if psycopg is not None and self.database_url else "disabled"

    def init_schema(self) -> None:
        if self.backend != "postgres":
            return
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        email TEXT PRIMARY KEY,
                        plan TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS usage_events (
                        id BIGSERIAL PRIMARY KEY,
                        email TEXT NOT NULL REFERENCES users(email) ON DELETE CASCADE,
                        event_month TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                cur.execute("CREATE INDEX IF NOT EXISTS idx_usage_events_email_month ON usage_events(email, event_month)")
                conn.commit()

    def check(self, email: str | None, plan: str | None) -> QuotaStatus:
        normalized_plan = plan if plan in PLAN_QUOTAS else "free"
        if self.backend != "postgres" or not email:
            quota = PLAN_QUOTAS[normalized_plan]
            return QuotaStatus(self.backend, True, normalized_plan, quota, 0, quota)
        month = current_month()
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                normalized_plan = get_or_create_user_plan(cur, email, normalized_plan)
                quota = PLAN_QUOTAS[normalized_plan]
                cur.execute("SELECT count(*) FROM usage_events WHERE email = %s AND event_month = %s", (email, month))
                used = int(cur.fetchone()[0])
                conn.commit()
        remaining = max(quota - used, 0)
        return QuotaStatus(self.backend, remaining > 0, normalized_plan, quota, used, remaining)

    def consume(self, email: str | None, plan: str | None) -> QuotaStatus:
        normalized_plan = plan if plan in PLAN_QUOTAS else "free"
        if self.backend != "postgres" or not email:
            quota = PLAN_QUOTAS[normalized_plan]
            return QuotaStatus(self.backend, True, normalized_plan, quota, 0, quota)
        month = current_month()
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                normalized_plan = get_or_create_user_plan(cur, email, normalized_plan)
                quota = PLAN_QUOTAS[normalized_plan]
                cur.execute("SELECT count(*) FROM usage_events WHERE email = %s AND event_month = %s", (email, month))
                used = int(cur.fetchone()[0])
                if used >= quota:
                    conn.commit()
                    return QuotaStatus(self.backend, False, normalized_plan, quota, used, 0)
                cur.execute("INSERT INTO usage_events(email, event_month) VALUES (%s, %s)", (email, month))
                conn.commit()
        used += 1
        return QuotaStatus(self.backend, True, normalized_plan, quota, used, max(quota - used, 0))

    def set_plan(self, email: str, plan: str) -> dict[str, str]:
        if plan not in PLAN_QUOTAS:
            raise ValueError(f"Unsupported plan: {plan}")
        normalized_email = email.lower().strip()
        if self.backend != "postgres":
            return {"backend": self.backend, "email": normalized_email, "plan": plan}
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                upsert_user(cur, normalized_email, plan)
                conn.commit()
        return {"backend": self.backend, "email": normalized_email, "plan": plan}

    def admin_summary(self) -> dict[str, object]:
        if self.backend != "postgres":
            return {
                "backend": self.backend,
                "users_total": 0,
                "usage_current_month": 0,
                "mrr_eur": 0,
                "plans": [],
                "recent_users": [],
            }
        month = current_month()
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT plan, count(*) FROM users GROUP BY plan ORDER BY plan")
                plan_counts = [{"plan": row[0], "users": int(row[1])} for row in cur.fetchall()]
                cur.execute("SELECT count(*) FROM users")
                users_total = int(cur.fetchone()[0])
                cur.execute("SELECT count(*) FROM usage_events WHERE event_month = %s", (month,))
                usage_current_month = int(cur.fetchone()[0])
                cur.execute(
                    """
                    SELECT email, plan, created_at, updated_at
                    FROM users
                    ORDER BY updated_at DESC
                    LIMIT 10
                    """
                )
                recent_users = [
                    {
                        "email": row[0],
                        "plan": row[1],
                        "created_at": row[2].isoformat(),
                        "updated_at": row[3].isoformat(),
                    }
                    for row in cur.fetchall()
                ]
        mrr_eur = sum(plan_monthly_price_eur(item["plan"]) * item["users"] for item in plan_counts)
        return {
            "backend": self.backend,
            "users_total": users_total,
            "usage_current_month": usage_current_month,
            "mrr_eur": mrr_eur,
            "plans": plan_counts,
            "recent_users": recent_users,
        }


def current_month() -> str:
    return datetime.now(UTC).strftime("%Y-%m")


def upsert_user(cur, email: str, plan: str) -> None:
    cur.execute(
        """
        INSERT INTO users(email, plan, updated_at)
        VALUES (%s, %s, now())
        ON CONFLICT (email) DO UPDATE SET plan = EXCLUDED.plan, updated_at = now()
        """,
        (email.lower().strip(), plan),
    )


def get_or_create_user_plan(cur, email: str, default_plan: str) -> str:
    normalized_email = email.lower().strip()
    cur.execute("SELECT plan FROM users WHERE email = %s", (normalized_email,))
    row = cur.fetchone()
    if row and row[0] in PLAN_QUOTAS:
        return row[0]
    upsert_user(cur, normalized_email, default_plan)
    return default_plan


_quota_store = QuotaStore()


def get_quota_store() -> QuotaStore:
    return _quota_store


def get_plan_catalog() -> list[dict[str, object]]:
    return [
        {
            "id": plan,
            "quota": PLAN_QUOTAS[plan],
            **PLAN_DETAILS[plan],
        }
        for plan in ("free", "quant", "enterprise")
    ]


def plan_monthly_price_eur(plan: str) -> int:
    if plan == "quant":
        return 29
    if plan == "enterprise":
        return 199
    return 0
