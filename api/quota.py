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
        quota = PLAN_QUOTAS[normalized_plan]
        if self.backend != "postgres" or not email:
            return QuotaStatus(self.backend, True, normalized_plan, quota, 0, quota)
        month = current_month()
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                upsert_user(cur, email, normalized_plan)
                cur.execute("SELECT count(*) FROM usage_events WHERE email = %s AND event_month = %s", (email, month))
                used = int(cur.fetchone()[0])
                conn.commit()
        remaining = max(quota - used, 0)
        return QuotaStatus(self.backend, remaining > 0, normalized_plan, quota, used, remaining)

    def consume(self, email: str | None, plan: str | None) -> QuotaStatus:
        normalized_plan = plan if plan in PLAN_QUOTAS else "free"
        quota = PLAN_QUOTAS[normalized_plan]
        if self.backend != "postgres" or not email:
            return QuotaStatus(self.backend, True, normalized_plan, quota, 0, quota)
        month = current_month()
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                upsert_user(cur, email, normalized_plan)
                cur.execute("SELECT count(*) FROM usage_events WHERE email = %s AND event_month = %s", (email, month))
                used = int(cur.fetchone()[0])
                if used >= quota:
                    conn.commit()
                    return QuotaStatus(self.backend, False, normalized_plan, quota, used, 0)
                cur.execute("INSERT INTO usage_events(email, event_month) VALUES (%s, %s)", (email, month))
                conn.commit()
        used += 1
        return QuotaStatus(self.backend, True, normalized_plan, quota, used, max(quota - used, 0))


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


_quota_store = QuotaStore()


def get_quota_store() -> QuotaStore:
    return _quota_store
