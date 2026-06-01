from __future__ import annotations

import os

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


def test_price_basket_call() -> None:
    response = client.post(
        "/api/v1/price",
        json={
            "instrument": "basket_call",
            "S0": 100.0,
            "K": 100.0,
            "sigma": 0.2,
            "r": 0.05,
            "T": 1.0,
            "spots": [100.0, 105.0, 95.0],
            "sigmas": [0.2, 0.25, 0.18],
            "weights": [0.4, 0.3, 0.3],
            "correlation": 0.25,
            "n_paths": 4000,
            "seed": 42,
            "method": "reference",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["instrument"] == "basket_call"
    assert body["price"] > 0.0
    assert body["method"] == "reference"
    assert body["model_version"] == "basket_monte_carlo_v1"


def test_price_basket_call_model_method_if_checkpoint_available() -> None:
    response = client.post(
        "/api/v1/price",
        json={
            "instrument": "basket_call",
            "S0": 100.0,
            "K": 100.0,
            "sigma": 0.2,
            "r": 0.05,
            "T": 1.0,
            "spots": [100.0, 105.0, 95.0, 110.0, 90.0],
            "sigmas": [0.2, 0.25, 0.18, 0.22, 0.20],
            "weights": [0.2, 0.2, 0.2, 0.2, 0.2],
            "correlation": 0.25,
            "method": "model",
        },
    )
    if response.status_code == 500:
        assert "surrogate" in response.text.lower()
        return
    assert response.status_code == 200
    body = response.json()
    assert body["instrument"] == "basket_call"
    assert body["method"] == "model"
    assert body["model_version"] == "basket_surrogate_v2"


def test_basket_call_validation_error_for_mismatched_dimensions() -> None:
    response = client.post(
        "/api/v1/price",
        json={
            "instrument": "basket_call",
            "S0": 100.0,
            "K": 100.0,
            "sigma": 0.2,
            "r": 0.05,
            "T": 1.0,
            "spots": [100.0, 105.0],
            "sigmas": [0.2],
            "weights": [0.5, 0.5],
        },
    )
    assert response.status_code == 422


def test_price_heston_call() -> None:
    response = client.post(
        "/api/v1/price",
        json={
            "instrument": "heston_call",
            "S0": 100.0,
            "K": 100.0,
            "sigma": 0.2,
            "r": 0.05,
            "T": 1.0,
            "v0": 0.04,
            "kappa": 2.0,
            "theta": 0.04,
            "xi": 0.30,
            "rho": -0.50,
            "n_paths": 2000,
            "n_steps": 32,
            "seed": 123,
            "method": "reference",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["instrument"] == "heston_call"
    assert body["price"] > 0.0
    assert body["method"] == "reference"
    assert body["model_version"] == "heston_monte_carlo_v1"


def test_price_heston_call_model_method_if_checkpoint_available() -> None:
    response = client.post(
        "/api/v1/price",
        json={
            "instrument": "heston_call",
            "S0": 100.0,
            "K": 100.0,
            "sigma": 0.2,
            "r": 0.05,
            "T": 1.0,
            "v0": 0.04,
            "kappa": 2.0,
            "theta": 0.04,
            "xi": 0.30,
            "rho": -0.50,
            "method": "model",
        },
    )
    if response.status_code == 500:
        assert "surrogate" in response.text.lower()
        return
    assert response.status_code == 200
    body = response.json()
    assert body["instrument"] == "heston_call"
    assert body["method"] == "model"
    assert body["model_version"] == "heston_surrogate_v1"


def test_heston_call_validation_error_for_missing_parameters() -> None:
    response = client.post(
        "/api/v1/price",
        json={
            "instrument": "heston_call",
            "S0": 100.0,
            "K": 100.0,
            "sigma": 0.2,
            "r": 0.05,
            "T": 1.0,
        },
    )
    assert response.status_code == 422


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
