from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_price_european_call() -> None:
    response = client.post(
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
    assert response.status_code == 200
    body = response.json()
    assert body["price"] > 0.0
    assert body["method"] == "reference"
    assert body["greeks"] is not None
    assert "delta" in body["greeks"]


def test_repeated_price_request_uses_cache() -> None:
    payload = {
        "instrument": "european_call",
        "S0": 101.0,
        "K": 100.0,
        "sigma": 0.2,
        "r": 0.05,
        "T": 1.0,
    }
    first = client.post("/api/v1/price", json=payload)
    second = client.post("/api/v1/price", json=payload)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["price"] == second.json()["price"]
    assert any("cache hit" in warning for warning in second.json()["warnings"])


def test_price_batch() -> None:
    response = client.post(
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
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert len(body["results"]) == 2
    assert all(item["price"] >= 0.0 for item in body["results"])


def test_validation_error_for_missing_strike() -> None:
    response = client.post(
        "/api/v1/price",
        json={
            "instrument": "european_call",
            "S0": 100.0,
            "sigma": 0.2,
            "r": 0.05,
            "T": 1.0,
        },
    )
    assert response.status_code == 422
