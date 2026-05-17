from __future__ import annotations

from locust import HttpUser, between, task


class NeuroPriceUser(HttpUser):
    wait_time = between(0.005, 0.02)

    def on_start(self) -> None:
        response = self.client.get("/health", name="/health")
        if response.status_code != 200:
            raise RuntimeError("NeuroPrice API healthcheck failed; start uvicorn before running Locust")

    @task(8)
    def price_european_call(self) -> None:
        self.client.post(
            "/api/v1/price",
            json={
                "instrument": "european_call",
                "S0": 100.0,
                "K": 100.0,
                "sigma": 0.2,
                "r": 0.05,
                "T": 1.0,
                "greeks": True,
            },
        )

    @task(2)
    def price_batch(self) -> None:
        self.client.post(
            "/api/v1/price/batch",
            json={
                "requests": [
                    {
                        "instrument": "european_call",
                        "S0": 100.0,
                        "K": 100.0,
                        "sigma": 0.2,
                        "r": 0.05,
                        "T": 1.0,
                    },
                    {
                        "instrument": "down_out_barrier_call",
                        "S0": 120.0,
                        "K": 100.0,
                        "barrier": 70.0,
                        "sigma": 0.2,
                        "r": 0.05,
                        "T": 1.0,
                    },
                ]
            },
        )
