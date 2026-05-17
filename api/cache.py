from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any

try:
    import redis
except ImportError:
    redis = None


@dataclass(frozen=True)
class CacheConfig:
    ttl_seconds: int = 300
    redis_url: str | None = None


class PricingCache:
    def __init__(self, config: CacheConfig | None = None) -> None:
        self.config = config or CacheConfig(redis_url=os.getenv("REDIS_URL"))
        self._memory: dict[str, tuple[float, dict[str, Any]]] = {}
        self._redis = None
        if redis is not None and self.config.redis_url:
            try:
                self._redis = redis.Redis.from_url(self.config.redis_url, decode_responses=True)
                self._redis.ping()
            except Exception:
                self._redis = None

    @property
    def backend(self) -> str:
        return "redis" if self._redis is not None else "memory"

    def make_key(self, payload: dict[str, Any]) -> str:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return f"neuroprice:price:{digest}"

    def get(self, key: str) -> dict[str, Any] | None:
        if self._redis is not None:
            value = self._redis.get(key)
            return json.loads(value) if value is not None else None
        item = self._memory.get(key)
        if item is None:
            return None
        expires_at, value = item
        if expires_at < time.time():
            self._memory.pop(key, None)
            return None
        return value

    def set(self, key: str, value: dict[str, Any]) -> None:
        if self._redis is not None:
            self._redis.setex(key, self.config.ttl_seconds, json.dumps(value))
            return
        self._memory[key] = (time.time() + self.config.ttl_seconds, value)

    def clear(self) -> None:
        self._memory.clear()


_cache = PricingCache()


def get_cache() -> PricingCache:
    return _cache
