from __future__ import annotations

import numpy as np


def asian_arithmetic_call_mc_np(
    S0: np.ndarray,
    A0: np.ndarray,
    tau: np.ndarray,
    K: float,
    r: float,
    sigma: float,
    n_paths: int = 20000,
    n_steps: int = 64,
    seed: int = 123,
    chunk_size: int = 2000,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    S0 = np.asarray(S0, dtype=np.float64).reshape(-1, 1)
    A0 = np.asarray(A0, dtype=np.float64).reshape(-1, 1)
    tau = np.asarray(tau, dtype=np.float64).reshape(-1, 1)
    estimates = np.zeros_like(S0, dtype=np.float64)
    paths_done = 0
    while paths_done < n_paths:
        current_paths = min(chunk_size, n_paths - paths_done)
        S = np.repeat(S0, current_paths, axis=1)
        running_sum = np.repeat(A0, current_paths, axis=1)
        active_steps = np.maximum(n_steps, 1)
        dt = tau / active_steps
        sqrt_dt = np.sqrt(np.maximum(dt, 0.0))
        for _ in range(active_steps):
            z = rng.standard_normal(size=(S0.shape[0], current_paths))
            S = S * np.exp((r - 0.5 * sigma * sigma) * dt + sigma * sqrt_dt * z)
            running_sum += S
        average = running_sum / (active_steps + 1.0)
        payoff = np.maximum(average - K, 0.0)
        estimates += np.exp(-r * tau) * np.mean(payoff, axis=1, keepdims=True) * (current_paths / n_paths)
        paths_done += current_paths
    immediate = np.maximum(A0 - K, 0.0)
    return np.where(tau <= 1e-12, immediate, estimates)
