from __future__ import annotations

import numpy as np


def heston_call_mc_np(
    S0: float,
    K: float,
    r: float,
    T: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    n_paths: int = 20000,
    n_steps: int = 128,
    seed: int = 123,
    chunk_size: int = 5000,
) -> float:
    if T <= 0.0:
        return float(max(S0 - K, 0.0))
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    discount = np.exp(-r * T)
    total_payoff = 0.0
    total_paths = 0
    chol_scale = np.sqrt(max(1.0 - rho * rho, 0.0))
    for start in range(0, n_paths, chunk_size):
        size = min(chunk_size, n_paths - start)
        log_s = np.full(size, np.log(S0), dtype=np.float64)
        variance = np.full(size, v0, dtype=np.float64)
        for _ in range(n_steps):
            z1 = rng.standard_normal(size)
            z2 = rng.standard_normal(size)
            dw_s = np.sqrt(dt) * z1
            dw_v = np.sqrt(dt) * (rho * z1 + chol_scale * z2)
            variance_pos = np.maximum(variance, 0.0)
            log_s += (r - 0.5 * variance_pos) * dt + np.sqrt(variance_pos) * dw_s
            variance += kappa * (theta - variance_pos) * dt + xi * np.sqrt(variance_pos) * dw_v
            variance = np.maximum(variance, 0.0)
        terminal = np.exp(log_s)
        total_payoff += float(np.sum(np.maximum(terminal - K, 0.0)))
        total_paths += size
    return float(discount * total_payoff / total_paths)
