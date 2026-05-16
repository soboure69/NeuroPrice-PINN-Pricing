from __future__ import annotations

import numpy as np
from scipy.stats import norm

from neuroprice.validation.black_scholes_ref import black_scholes_call_price_np


def down_and_out_call_price_np(
    S: np.ndarray,
    tau: np.ndarray,
    K: float,
    B: float,
    r: float,
    sigma: float,
) -> np.ndarray:
    if B <= 0.0:
        raise ValueError("B must be positive")
    if B >= K:
        raise ValueError("This reference implementation assumes B < K")
    S_array = np.asarray(S, dtype=np.float64)
    tau_array = np.asarray(tau, dtype=np.float64)
    vanilla = black_scholes_call_price_np(S_array, tau_array, K, r, sigma)
    mirrored_S = (B * B) / np.maximum(S_array, 1e-12)
    power = 2.0 * r / (sigma * sigma) - 1.0
    image = (S_array / B) ** power * black_scholes_call_price_np(mirrored_S, tau_array, K, r, sigma)
    price = vanilla - image
    price = np.where(S_array <= B, 0.0, price)
    price = np.maximum(price, 0.0)
    payoff = np.where(S_array > B, np.maximum(S_array - K, 0.0), 0.0)
    return np.where(tau_array <= 1e-12, payoff, price)
