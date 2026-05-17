from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from api.schemas import PricingRequest, PricingResponse
from neuroprice.pinn.asian_surrogate import AsianArithmeticSurrogate, AsianSurrogateDomain
from neuroprice.pinn.lookback_surrogate import LookbackFloatingCallSurrogate, LookbackSurrogateDomain
from neuroprice.validation.asian_ref import asian_arithmetic_call_mc_np
from neuroprice.validation.barrier_ref import down_and_out_call_price_np
from neuroprice.validation.black_scholes_ref import black_scholes_call_delta_np, black_scholes_call_gamma_np, black_scholes_call_price_np
from neuroprice.validation.lookback_ref import lookback_floating_call_mc_np

PROJECT_ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents] if (path / "neuroprice").exists())


class PricingError(RuntimeError):
    pass


def preload_models() -> dict[str, bool]:
    get_device()
    return {
        "asian_arithmetic_call": load_asian_surrogate() is not None,
        "lookback_floating_call": load_lookback_surrogate() is not None,
    }


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def load_asian_surrogate() -> tuple[AsianArithmeticSurrogate, AsianSurrogateDomain] | None:
    path = PROJECT_ROOT / "artifacts" / "phase3_asian_surrogate_offline" / "asian_surrogate.pt"
    if not path.exists():
        return None
    device = get_device()
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("option_type") != "asian_arithmetic_call_surrogate":
        return None
    domain = AsianSurrogateDomain(**checkpoint["domain"])
    model = AsianArithmeticSurrogate(hidden_dim=int(checkpoint["hidden_dim"]), hidden_layers=int(checkpoint["hidden_layers"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, domain


@lru_cache(maxsize=1)
def load_lookback_surrogate() -> tuple[LookbackFloatingCallSurrogate, LookbackSurrogateDomain] | None:
    path = PROJECT_ROOT / "artifacts" / "phase3_lookback_surrogate_offline" / "lookback_surrogate.pt"
    if not path.exists():
        return None
    device = get_device()
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("option_type") != "lookback_floating_call_surrogate":
        return None
    domain = LookbackSurrogateDomain(**checkpoint["domain"])
    model = LookbackFloatingCallSurrogate(hidden_dim=int(checkpoint["hidden_dim"]), hidden_layers=int(checkpoint["hidden_layers"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, domain


def price(request: PricingRequest) -> PricingResponse:
    start = time.perf_counter()
    warnings: list[str] = []
    greeks: dict[str, float] | None = None

    if request.instrument == "european_call":
        price_value, method, version = _price_european_call(request)
        if request.greeks:
            greeks = _european_greeks(request)
    elif request.instrument == "down_out_barrier_call":
        price_value, method, version = _price_down_out_barrier_call(request)
    elif request.instrument == "asian_arithmetic_call":
        price_value, method, version, warnings = _price_asian_arithmetic_call(request)
    elif request.instrument == "lookback_floating_call":
        price_value, method, version, warnings = _price_lookback_floating_call(request)
    else:
        raise PricingError(f"Unsupported instrument: {request.instrument}")

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return PricingResponse(
        instrument=request.instrument,
        price=float(price_value),
        method=method,
        model_version=version,
        inference_time_ms=round(elapsed_ms, 4),
        greeks=greeks,
        warnings=warnings,
    )


def _price_european_call(request: PricingRequest) -> tuple[float, str, str]:
    value = black_scholes_call_price_np(
        np.array([[request.S0]]),
        np.array([[request.T]]),
        float(request.K),
        request.r,
        request.sigma,
    )
    return float(np.asarray(value).reshape(-1)[0]), "reference", "black_scholes_analytic"


def _european_greeks(request: PricingRequest) -> dict[str, float]:
    S = np.array([[request.S0]])
    tau = np.array([[request.T]])
    delta = black_scholes_call_delta_np(S, tau, float(request.K), request.r, request.sigma)
    gamma = black_scholes_call_gamma_np(S, tau, float(request.K), request.r, request.sigma)
    return {"delta": float(np.asarray(delta).reshape(-1)[0]), "gamma": float(np.asarray(gamma).reshape(-1)[0])}


def _price_down_out_barrier_call(request: PricingRequest) -> tuple[float, str, str]:
    value = down_and_out_call_price_np(
        np.array([request.S0]),
        np.array([request.T]),
        K=float(request.K),
        B=float(request.barrier),
        r=request.r,
        sigma=request.sigma,
    )
    return float(np.asarray(value).reshape(-1)[0]), "reference", "down_out_barrier_semi_analytic"


def _price_asian_arithmetic_call(request: PricingRequest) -> tuple[float, str, str, list[str]]:
    loaded = load_asian_surrogate() if request.method in {"auto", "model"} else None
    if loaded is not None:
        model, domain = loaded
        device = get_device()
        S_norm = torch.tensor([[request.S0 / domain.S_max]], dtype=torch.float32, device=device)
        tau_norm = torch.tensor([[request.T / domain.T]], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(S_norm, tau_norm).detach().cpu().numpy()
        return float(domain.S_max * pred.reshape(-1)[0]), "model", "asian_surrogate_offline_v1", []
    if request.method == "model":
        raise PricingError("Asian surrogate checkpoint is not available")
    value = asian_arithmetic_call_mc_np(
        np.array([[request.S0]]),
        np.array([[request.S0]]),
        np.array([[request.T]]),
        float(request.K),
        request.r,
        request.sigma,
        n_paths=20000,
        n_steps=64,
        seed=123,
        chunk_size=2000,
    )
    return float(value.reshape(-1)[0]), "reference", "asian_arithmetic_monte_carlo", ["model checkpoint unavailable; used Monte Carlo reference"]


def _price_lookback_floating_call(request: PricingRequest) -> tuple[float, str, str, list[str]]:
    loaded = load_lookback_surrogate() if request.method in {"auto", "model"} else None
    if loaded is not None:
        model, domain = loaded
        device = get_device()
        S_norm = torch.tensor([[request.S0 / domain.S_max]], dtype=torch.float32, device=device)
        tau_norm = torch.tensor([[request.T / domain.T]], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(S_norm, tau_norm).detach().cpu().numpy()
        return float(domain.S_max * pred.reshape(-1)[0]), "model", "lookback_surrogate_offline_v1", []
    if request.method == "model":
        raise PricingError("Lookback surrogate checkpoint is not available")
    value = lookback_floating_call_mc_np(
        np.array([[request.S0]]),
        np.array([[request.T]]),
        request.r,
        request.sigma,
        n_paths=20000,
        n_steps=64,
        seed=123,
        chunk_size=2000,
    )
    return float(value.reshape(-1)[0]), "reference", "lookback_floating_monte_carlo", ["model checkpoint unavailable; used Monte Carlo reference"]
