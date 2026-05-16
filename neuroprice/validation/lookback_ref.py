from __future__ import annotations

import numpy as np


def lookback_floating_call_mc_np(
    S0: np.ndarray,
    tau: np.ndarray,
    r: float,
    sigma: float,
    n_paths: int = 20000,
    n_steps: int = 64,
    seed: int = 123,
    chunk_size: int = 2000,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    S0 = np.asarray(S0, dtype=np.float64).reshape(-1, 1)
    tau = np.asarray(tau, dtype=np.float64).reshape(-1, 1)
    estimates = np.zeros_like(S0, dtype=np.float64)
    paths_done = 0
    while paths_done < n_paths:
        current_paths = min(chunk_size, n_paths - paths_done)
        S = np.repeat(S0, current_paths, axis=1)
        running_min = S.copy()
        active_steps = max(n_steps, 1)
        dt = tau / active_steps
        sqrt_dt = np.sqrt(np.maximum(dt, 0.0))
        for _ in range(active_steps):
            z = rng.standard_normal(size=(S0.shape[0], current_paths))
            S = S * np.exp((r - 0.5 * sigma * sigma) * dt + sigma * sqrt_dt * z)
            running_min = np.minimum(running_min, S)
        payoff = np.maximum(S - running_min, 0.0)
        estimates += np.exp(-r * tau) * np.mean(payoff, axis=1, keepdims=True) * (current_paths / n_paths)
        paths_done += current_paths
    return np.where(tau <= 1e-12, np.zeros_like(S0), estimates)
