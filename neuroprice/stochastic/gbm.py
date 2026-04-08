from __future__ import annotations

import numpy as np


def simulate_gbm_paths(
    *,
    s0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if s0 <= 0:
        raise ValueError("s0 must be > 0")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if T <= 0:
        raise ValueError("T must be > 0")
    if n_steps <= 0:
        raise ValueError("n_steps must be > 0")
    if n_paths <= 0:
        raise ValueError("n_paths must be > 0")

    rng = np.random.default_rng(seed)
    dt = T / n_steps

    t = np.linspace(0.0, T, n_steps + 1)
    z = rng.standard_normal(size=(n_paths, n_steps))

    increments = (mu - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(increments, axis=1)

    s = np.empty((n_paths, n_steps + 1), dtype=float)
    s[:, 0] = s0
    s[:, 1:] = s0 * np.exp(log_paths)

    return t, s
