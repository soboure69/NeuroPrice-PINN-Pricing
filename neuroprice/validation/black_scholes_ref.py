from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1_np(S: np.ndarray, tau: np.ndarray, K: float, r: float, sigma: float) -> np.ndarray:
    S_safe = np.maximum(S, 1e-12)
    tau_safe = np.maximum(tau, 1e-12)
    return (np.log(S_safe / K) + (r + 0.5 * sigma * sigma) * tau_safe) / (sigma * np.sqrt(tau_safe))


def black_scholes_call_price_np(S: np.ndarray, tau: np.ndarray, K: float, r: float, sigma: float) -> np.ndarray:
    S_safe = np.maximum(S, 1e-12)
    tau_safe = np.maximum(tau, 1e-12)
    d1 = _d1_np(S_safe, tau_safe, K, r, sigma)
    d2 = d1 - sigma * np.sqrt(tau_safe)
    price = S_safe * norm.cdf(d1) - K * np.exp(-r * tau_safe) * norm.cdf(d2)
    payoff = np.maximum(S - K, 0.0)
    return np.where(tau <= 1e-12, payoff, price)


def black_scholes_call_delta_np(S: np.ndarray, tau: np.ndarray, K: float, r: float, sigma: float) -> np.ndarray:
    d1 = _d1_np(S, tau, K, r, sigma)
    delta = norm.cdf(d1)
    payoff_delta = np.where(S > K, 1.0, 0.0)
    return np.where(tau <= 1e-12, payoff_delta, delta)


def black_scholes_call_gamma_np(S: np.ndarray, tau: np.ndarray, K: float, r: float, sigma: float) -> np.ndarray:
    S_safe = np.maximum(S, 1e-12)
    tau_safe = np.maximum(tau, 1e-12)
    d1 = _d1_np(S_safe, tau_safe, K, r, sigma)
    gamma = norm.pdf(d1) / (S_safe * sigma * np.sqrt(tau_safe))
    return np.where(tau <= 1e-12, 0.0, gamma)


def relative_l2_error(pred: np.ndarray, ref: np.ndarray) -> float:
    numerator = np.linalg.norm(pred - ref)
    denominator = np.linalg.norm(ref) + 1e-12
    return float(numerator / denominator)


def masked_relative_l2_error(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return relative_l2_error(pred[mask], ref[mask])
