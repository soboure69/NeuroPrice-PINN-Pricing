from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from api.schemas import PricingRequest, PricingResponse
from neuroprice.validation.asian_ref import asian_arithmetic_call_mc_np
from neuroprice.validation.barrier_ref import down_and_out_call_price_np
from neuroprice.validation.basket_ref import basket_call_mc_np
from neuroprice.validation.black_scholes_ref import black_scholes_call_delta_np, black_scholes_call_gamma_np, black_scholes_call_price_np
from neuroprice.validation.heston_ref import heston_call_mc_np
from neuroprice.validation.lookback_ref import lookback_floating_call_mc_np

try:
    import torch
    from neuroprice.pinn.asian_surrogate import AsianArithmeticSurrogate, AsianSurrogateDomain
    from neuroprice.pinn.basket_surrogate import BasketCallSurrogate, BasketSurrogateDomain
    from neuroprice.pinn.heston_surrogate import HestonCallSurrogate, HestonSurrogateDomain
    from neuroprice.pinn.lookback_surrogate import LookbackFloatingCallSurrogate, LookbackSurrogateDomain
except ImportError:
    torch = None
    AsianArithmeticSurrogate = None
    AsianSurrogateDomain = None
    BasketCallSurrogate = None
    BasketSurrogateDomain = None
    HestonCallSurrogate = None
    HestonSurrogateDomain = None
    LookbackFloatingCallSurrogate = None
    LookbackSurrogateDomain = None

PROJECT_ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents] if (path / "neuroprice").exists())


class PricingError(RuntimeError):
    pass


def preload_models() -> dict[str, bool]:
    get_device()
    return {
        "asian_arithmetic_call": load_asian_surrogate() is not None,
        "lookback_floating_call": load_lookback_surrogate() is not None,
        "basket_call": load_basket_surrogate() is not None,
        "heston_call": load_heston_surrogate() is not None,
    }


@lru_cache(maxsize=1)
def get_device():
    if torch is None:
        return None
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def load_asian_surrogate() -> tuple[AsianArithmeticSurrogate, AsianSurrogateDomain] | None:
    if torch is None or AsianArithmeticSurrogate is None or AsianSurrogateDomain is None:
        return None
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
    if torch is None or LookbackFloatingCallSurrogate is None or LookbackSurrogateDomain is None:
        return None
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


@lru_cache(maxsize=1)
def load_basket_surrogate() -> tuple[BasketCallSurrogate, BasketSurrogateDomain] | None:
    if torch is None or BasketCallSurrogate is None or BasketSurrogateDomain is None:
        return None
    path = PROJECT_ROOT / "artifacts" / "phase6_basket_surrogate" / "basket_surrogate.pt"
    if not path.exists():
        return None
    device = get_device()
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("option_type") != "basket_call_surrogate":
        return None
    domain = BasketSurrogateDomain(**checkpoint["domain"])
    model = BasketCallSurrogate(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        hidden_layers=int(checkpoint["hidden_layers"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, domain


@lru_cache(maxsize=1)
def load_heston_surrogate() -> tuple[HestonCallSurrogate, HestonSurrogateDomain] | None:
    if torch is None or HestonCallSurrogate is None or HestonSurrogateDomain is None:
        return None
    path = PROJECT_ROOT / "artifacts" / "phase6_heston_surrogate" / "heston_surrogate.pt"
    if not path.exists():
        return None
    device = get_device()
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("option_type") != "heston_call_surrogate":
        return None
    domain = HestonSurrogateDomain(**checkpoint["domain"])
    model = HestonCallSurrogate(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        hidden_layers=int(checkpoint["hidden_layers"]),
    ).to(device)
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
    elif request.instrument == "basket_call":
        price_value, method, version, warnings = _price_basket_call(request)
    elif request.instrument == "heston_call":
        price_value, method, version, warnings = _price_heston_call(request)
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


def _basket_features(request: PricingRequest, domain: BasketSurrogateDomain) -> np.ndarray:
    spots = np.asarray(request.spots, dtype=np.float32).reshape(1, -1)
    sigmas = np.asarray(request.sigmas, dtype=np.float32).reshape(1, -1)
    weights = np.asarray(request.weights, dtype=np.float32).reshape(1, -1)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    strike = np.array([[float(request.K)]], dtype=np.float32)
    rate = np.array([[request.r]], dtype=np.float32)
    maturity = np.array([[request.T]], dtype=np.float32)
    correlation = np.array([[float(request.correlation or 0.0)]], dtype=np.float32)
    basket_spot = np.sum(weights * spots, axis=1, keepdims=True)
    effective_sigma = np.sqrt(
        np.sum((weights * sigmas) ** 2, axis=1, keepdims=True)
        + 2.0
        * correlation
        * np.sum(
            [
                weights[:, i : i + 1] * weights[:, j : j + 1] * sigmas[:, i : i + 1] * sigmas[:, j : j + 1]
                for i in range(spots.shape[1])
                for j in range(i + 1, spots.shape[1])
            ],
            axis=0,
        )
    )
    return np.concatenate(
        [
            spots / domain.spot_max,
            sigmas / domain.sigma_max,
            weights,
            strike / domain.strike_max,
            (rate - domain.rate_min) / (domain.rate_max - domain.rate_min),
            maturity / domain.maturity_max,
            (correlation - domain.correlation_min) / (domain.correlation_max - domain.correlation_min),
            basket_spot / domain.spot_max,
            np.clip((basket_spot / strike) / (domain.spot_max / 60.0), 0.0, 1.0),
            np.clip(effective_sigma / domain.sigma_max, 0.0, 1.0),
            np.maximum(basket_spot - strike, 0.0) / domain.spot_max,
        ],
        axis=1,
    ).astype(np.float32)


def _price_basket_call(request: PricingRequest) -> tuple[float, str, str, list[str]]:
    loaded = load_basket_surrogate() if request.method in {"auto", "model"} else None
    if loaded is not None:
        model, domain = loaded
        if len(request.spots or []) == domain.n_assets:
            device = get_device()
            x = torch.tensor(_basket_features(request, domain), dtype=torch.float32, device=device)
            with torch.no_grad():
                pred = model(x).detach().cpu().numpy()
            return float(domain.spot_max * pred.reshape(-1)[0]), "model", "basket_surrogate_v2", []
        if request.method == "model":
            raise PricingError("Basket surrogate asset dimension does not match request")
    if request.method == "model":
        raise PricingError("Basket surrogate checkpoint is not available")
    value = basket_call_mc_np(
        spots=np.asarray(request.spots, dtype=np.float64),
        weights=np.asarray(request.weights, dtype=np.float64),
        sigmas=np.asarray(request.sigmas, dtype=np.float64),
        K=float(request.K),
        r=request.r,
        tau=request.T,
        correlation=float(request.correlation or 0.0),
        n_paths=int(request.n_paths or 20000),
        seed=int(request.seed or 123),
    )
    warning = [] if request.method == "reference" else ["model checkpoint unavailable or incompatible; used Monte Carlo reference"]
    return value, "reference", "basket_monte_carlo_v1", warning


def _heston_features(request: PricingRequest, domain: HestonSurrogateDomain) -> np.ndarray:
    spot = np.array([[request.S0]], dtype=np.float32)
    strike = np.array([[float(request.K)]], dtype=np.float32)
    rate = np.array([[request.r]], dtype=np.float32)
    maturity = np.array([[request.T]], dtype=np.float32)
    v0 = np.array([[float(request.v0)]], dtype=np.float32)
    kappa = np.array([[float(request.kappa)]], dtype=np.float32)
    theta = np.array([[float(request.theta)]], dtype=np.float32)
    xi = np.array([[float(request.xi)]], dtype=np.float32)
    rho = np.array([[float(request.rho)]], dtype=np.float32)
    return np.concatenate(
        [
            spot / domain.spot_max,
            strike / domain.strike_max,
            (rate - domain.rate_min) / (domain.rate_max - domain.rate_min),
            maturity / domain.maturity_max,
            v0 / domain.v0_max,
            kappa / domain.kappa_max,
            theta / domain.theta_max,
            xi / domain.xi_max,
            (rho - domain.rho_min) / (domain.rho_max - domain.rho_min),
            np.clip((spot / strike) / (domain.spot_max / 60.0), 0.0, 1.0),
            np.clip((np.sqrt(v0) / np.sqrt(np.maximum(theta, 1e-8))) / np.sqrt(domain.v0_max / 1e-4), 0.0, 1.0),
            np.clip((2.0 * kappa * theta / np.maximum(xi**2, 1e-8)) / 5.0, 0.0, 1.0),
            np.maximum(spot - strike, 0.0) / domain.spot_max,
        ],
        axis=1,
    ).astype(np.float32)


def _price_heston_call(request: PricingRequest) -> tuple[float, str, str, list[str]]:
    loaded = load_heston_surrogate() if request.method in {"auto", "model"} else None
    if loaded is not None:
        model, domain = loaded
        device = get_device()
        x = torch.tensor(_heston_features(request, domain), dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(x).detach().cpu().numpy()
        return float(domain.spot_max * pred.reshape(-1)[0]), "model", "heston_surrogate_v1", []
    if request.method == "model":
        raise PricingError("Heston surrogate checkpoint is not available")
    value = heston_call_mc_np(
        S0=request.S0,
        K=float(request.K),
        r=request.r,
        T=request.T,
        v0=float(request.v0),
        kappa=float(request.kappa),
        theta=float(request.theta),
        xi=float(request.xi),
        rho=float(request.rho),
        n_paths=int(request.n_paths or 20000),
        n_steps=int(request.n_steps or 128),
        seed=int(request.seed or 123),
    )
    warning = [] if request.method == "reference" else ["model checkpoint unavailable; used Monte Carlo reference"]
    return value, "reference", "heston_monte_carlo_v1", warning


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
