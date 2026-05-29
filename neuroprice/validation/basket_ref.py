from __future__ import annotations

import numpy as np


def basket_call_mc_np(
    spots: np.ndarray,
    weights: np.ndarray,
    sigmas: np.ndarray,
    K: float,
    r: float,
    tau: float,
    correlation: float = 0.0,
    n_paths: int = 20000,
    seed: int = 123,
    chunk_size: int = 2000,
) -> float:
    spots = np.asarray(spots, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    sigmas = np.asarray(sigmas, dtype=np.float64).reshape(-1)
    n_assets = spots.shape[0]
    corr = np.full((n_assets, n_assets), float(correlation), dtype=np.float64)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)
    rng = np.random.default_rng(seed)
    estimate = 0.0
    paths_done = 0
    tau_safe = max(float(tau), 0.0)
    drift = (float(r) - 0.5 * sigmas * sigmas) * tau_safe
    diffusion_scale = sigmas * np.sqrt(tau_safe)
    while paths_done < n_paths:
        current_paths = min(chunk_size, n_paths - paths_done)
        z = rng.standard_normal(size=(current_paths, n_assets)) @ chol.T
        terminal_spots = spots * np.exp(drift + diffusion_scale * z)
        basket_values = terminal_spots @ weights
        payoff = np.maximum(basket_values - float(K), 0.0)
        estimate += float(np.mean(payoff)) * (current_paths / n_paths)
        paths_done += current_paths
    immediate = max(float(spots @ weights) - float(K), 0.0)
    discounted = np.exp(-float(r) * tau_safe) * estimate
    return immediate if tau_safe <= 1e-12 else float(discounted)
