from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neuroprice.validation.basket_ref import basket_call_mc_np


def sample_basket_parameters(
    n_samples: int,
    n_assets: int,
    *,
    spot_min: float,
    spot_max: float,
    sigma_min: float,
    sigma_max: float,
    strike_min: float,
    strike_max: float,
    rate_min: float,
    rate_max: float,
    maturity_min: float,
    maturity_max: float,
    correlation_min: float,
    correlation_max: float,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    spots = rng.uniform(spot_min, spot_max, size=(n_samples, n_assets)).astype(np.float32)
    sigmas = rng.uniform(sigma_min, sigma_max, size=(n_samples, n_assets)).astype(np.float32)
    raw_weights = rng.uniform(0.0, 1.0, size=(n_samples, n_assets)).astype(np.float32)
    weights = raw_weights / np.sum(raw_weights, axis=1, keepdims=True)
    strikes = rng.uniform(strike_min, strike_max, size=(n_samples, 1)).astype(np.float32)
    rates = rng.uniform(rate_min, rate_max, size=(n_samples, 1)).astype(np.float32)
    maturities = rng.uniform(maturity_min, maturity_max, size=(n_samples, 1)).astype(np.float32)
    correlations = rng.uniform(correlation_min, correlation_max, size=(n_samples, 1)).astype(np.float32)
    return {
        "spots": spots,
        "sigmas": sigmas,
        "weights": weights,
        "strikes": strikes,
        "rates": rates,
        "maturities": maturities,
        "correlations": correlations,
    }


def build_features(
    params: dict[str, np.ndarray],
    *,
    spot_max: float,
    sigma_max: float,
    strike_min: float,
    strike_max: float,
    rate_min: float,
    rate_max: float,
    maturity_max: float,
    correlation_min: float,
    correlation_max: float,
) -> np.ndarray:
    spots_norm = params["spots"] / spot_max
    sigmas_norm = params["sigmas"] / sigma_max
    strikes_norm = params["strikes"] / strike_max
    rates_norm = (params["rates"] - rate_min) / (rate_max - rate_min)
    maturities_norm = params["maturities"] / maturity_max
    correlations_norm = (params["correlations"] - correlation_min) / (correlation_max - correlation_min)
    basket_spot = np.sum(params["weights"] * params["spots"], axis=1, keepdims=True)
    basket_spot_norm = basket_spot / spot_max
    moneyness = basket_spot / params["strikes"]
    moneyness_norm = np.clip(moneyness / (spot_max / strike_min), 0.0, 1.0)
    effective_sigma = np.sqrt(
        np.sum((params["weights"] * params["sigmas"]) ** 2, axis=1, keepdims=True)
        + 2.0
        * params["correlations"]
        * np.sum(
            [
                params["weights"][:, i : i + 1]
                * params["weights"][:, j : j + 1]
                * params["sigmas"][:, i : i + 1]
                * params["sigmas"][:, j : j + 1]
                for i in range(params["spots"].shape[1])
                for j in range(i + 1, params["spots"].shape[1])
            ],
            axis=0,
        )
    )
    effective_sigma_norm = np.clip(effective_sigma / sigma_max, 0.0, 1.0)
    intrinsic_norm = np.maximum(basket_spot - params["strikes"], 0.0) / spot_max
    return np.concatenate(
        [
            spots_norm,
            sigmas_norm,
            params["weights"],
            strikes_norm,
            rates_norm,
            maturities_norm,
            correlations_norm,
            basket_spot_norm,
            moneyness_norm,
            effective_sigma_norm,
            intrinsic_norm,
        ],
        axis=1,
    ).astype(np.float32)


def generate_dataset(
    *,
    n_samples: int,
    n_assets: int,
    n_paths: int,
    chunk_size: int,
    seed: int,
    spot_min: float = 50.0,
    spot_max: float = 150.0,
    sigma_min: float = 0.10,
    sigma_max: float = 0.60,
    strike_min: float = 60.0,
    strike_max: float = 140.0,
    rate_min: float = 0.0,
    rate_max: float = 0.10,
    maturity_min: float = 0.10,
    maturity_max: float = 3.0,
    correlation_min: float = -0.25,
    correlation_max: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, float | int | str]]:
    params = sample_basket_parameters(
        n_samples,
        n_assets,
        spot_min=spot_min,
        spot_max=spot_max,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        strike_min=strike_min,
        strike_max=strike_max,
        rate_min=rate_min,
        rate_max=rate_max,
        maturity_min=maturity_min,
        maturity_max=maturity_max,
        correlation_min=correlation_min,
        correlation_max=correlation_max,
        seed=seed,
    )
    prices = np.zeros((n_samples, 1), dtype=np.float32)
    start = time.perf_counter()
    for index in range(n_samples):
        prices[index, 0] = basket_call_mc_np(
            spots=params["spots"][index],
            weights=params["weights"][index],
            sigmas=params["sigmas"][index],
            K=float(params["strikes"][index, 0]),
            r=float(params["rates"][index, 0]),
            tau=float(params["maturities"][index, 0]),
            correlation=float(params["correlations"][index, 0]),
            n_paths=n_paths,
            seed=seed + index,
            chunk_size=chunk_size,
        )
    elapsed_seconds = time.perf_counter() - start
    x = build_features(
        params,
        spot_max=spot_max,
        sigma_max=sigma_max,
        strike_min=strike_min,
        strike_max=strike_max,
        rate_min=rate_min,
        rate_max=rate_max,
        maturity_max=maturity_max,
        correlation_min=correlation_min,
        correlation_max=correlation_max,
    )
    y = (prices / spot_max).astype(np.float32)
    metadata: dict[str, float | int | str] = {
        "instrument": "basket_call",
        "dataset_version": "basket_mc_dataset_v2",
        "n_samples": n_samples,
        "n_assets": n_assets,
        "n_paths": n_paths,
        "chunk_size": chunk_size,
        "seed": seed,
        "spot_min": spot_min,
        "spot_max": spot_max,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "strike_min": strike_min,
        "strike_max": strike_max,
        "rate_min": rate_min,
        "rate_max": rate_max,
        "maturity_min": maturity_min,
        "maturity_max": maturity_max,
        "correlation_min": correlation_min,
        "correlation_max": correlation_max,
        "feature_order": "spots_norm,sigmas_norm,weights,strike_norm,rate_norm,maturity_norm,correlation_norm,basket_spot_norm,moneyness_norm,effective_sigma_norm,intrinsic_norm",
        "target_scale": "price_divided_by_spot_max",
        "elapsed_seconds": elapsed_seconds,
    }
    params["target_prices"] = prices
    return x, y, params, metadata


def save_dataset(out_dir: Path, x: np.ndarray, y: np.ndarray, params: dict[str, np.ndarray], metadata: dict[str, float | int | str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "dataset.npz",
        x=x,
        y=y,
        spots=params["spots"],
        sigmas=params["sigmas"],
        weights=params["weights"],
        strikes=params["strikes"],
        rates=params["rates"],
        maturities=params["maturities"],
        correlations=params["correlations"],
        target_prices=params["target_prices"],
    )
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="artifacts/phase6_basket_surrogate_dataset")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--n-assets", type=int, default=5)
    parser.add_argument("--n-paths", type=int, default=20000)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    x, y, params, metadata = generate_dataset(
        n_samples=args.n_samples,
        n_assets=args.n_assets,
        n_paths=args.n_paths,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )
    out_dir = Path(args.out_dir)
    save_dataset(out_dir, x, y, params, metadata)
    print(f"Generated basket dataset: samples={args.n_samples} assets={args.n_assets} features={x.shape[1]} paths={args.n_paths}")
    print(f"Price mean={float(np.mean(params['target_prices'])):.6f} price std={float(np.std(params['target_prices'])):.6f}")
    print(f"Saved dataset to {out_dir / 'dataset.npz'}")
    print(f"Saved metadata to {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
