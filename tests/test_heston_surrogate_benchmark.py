from __future__ import annotations

import numpy as np
import torch

from neuroprice.pinn.heston_surrogate import HestonCallSurrogate, HestonSurrogateDomain
from scripts.benchmark_heston_surrogate import run_benchmark


def test_heston_surrogate_vs_mc_benchmark_smoke(tmp_path) -> None:
    n_samples = 6
    domain = HestonSurrogateDomain()
    rng = np.random.default_rng(123)
    x = rng.uniform(0.0, 1.0, size=(n_samples, domain.input_dim)).astype(np.float32)
    y = rng.uniform(0.0, 0.5, size=(n_samples, 1)).astype(np.float32)
    dataset_path = tmp_path / "dataset.npz"
    checkpoint_path = tmp_path / "heston_surrogate.pt"
    np.savez_compressed(
        dataset_path,
        x=x,
        y=y,
        spots=rng.uniform(80.0, 120.0, size=(n_samples, 1)).astype(np.float32),
        strikes=rng.uniform(90.0, 110.0, size=(n_samples, 1)).astype(np.float32),
        rates=np.full((n_samples, 1), 0.05, dtype=np.float32),
        maturities=np.full((n_samples, 1), 1.0, dtype=np.float32),
        v0=np.full((n_samples, 1), 0.04, dtype=np.float32),
        kappa=np.full((n_samples, 1), 2.0, dtype=np.float32),
        theta=np.full((n_samples, 1), 0.04, dtype=np.float32),
        xi=np.full((n_samples, 1), 0.30, dtype=np.float32),
        rho=np.full((n_samples, 1), -0.50, dtype=np.float32),
        target_prices=(domain.spot_max * y).astype(np.float32),
    )
    model = HestonCallSurrogate(input_dim=domain.input_dim, hidden_dim=8, hidden_layers=2)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "domain": domain.__dict__,
            "hidden_dim": 8,
            "hidden_layers": 2,
            "input_dim": domain.input_dim,
            "option_type": "heston_call_surrogate",
            "training": {"mode": "test"},
        },
        checkpoint_path,
    )
    result = run_benchmark(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        n_points=3,
        mc_paths=1000,
        mc_steps=16,
        mc_chunk_size=500,
        relative_floor=1.0,
        seed=123,
    )
    assert result["instrument"] == "heston_call"
    assert result["n_points"] == 3
    assert result["surrogate_seconds"] >= 0.0
    assert result["monte_carlo_seconds"] >= 0.0
    assert result["speedup_vs_monte_carlo"] >= 0.0
    assert "rmse" in result["metrics"]
