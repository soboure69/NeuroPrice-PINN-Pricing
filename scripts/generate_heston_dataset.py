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

from neuroprice.validation.heston_ref import heston_call_mc_np


def sample_heston_parameters(
    n_samples: int,
    *,
    spot_min: float,
    spot_max: float,
    strike_min: float,
    strike_max: float,
    rate_min: float,
    rate_max: float,
    maturity_min: float,
    maturity_max: float,
    v0_min: float,
    v0_max: float,
    kappa_min: float,
    kappa_max: float,
    theta_min: float,
    theta_max: float,
    xi_min: float,
    xi_max: float,
    rho_min: float,
    rho_max: float,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    spots = rng.uniform(spot_min, spot_max, size=(n_samples, 1)).astype(np.float32)
    strikes = rng.uniform(strike_min, strike_max, size=(n_samples, 1)).astype(np.float32)
    rates = rng.uniform(rate_min, rate_max, size=(n_samples, 1)).astype(np.float32)
    maturities = rng.uniform(maturity_min, maturity_max, size=(n_samples, 1)).astype(np.float32)
    v0 = rng.uniform(v0_min, v0_max, size=(n_samples, 1)).astype(np.float32)
    kappa = rng.uniform(kappa_min, kappa_max, size=(n_samples, 1)).astype(np.float32)
    theta = rng.uniform(theta_min, theta_max, size=(n_samples, 1)).astype(np.float32)
    xi = rng.uniform(xi_min, xi_max, size=(n_samples, 1)).astype(np.float32)
    rho = rng.uniform(rho_min, rho_max, size=(n_samples, 1)).astype(np.float32)
    return {
        "spots": spots,
        "strikes": strikes,
        "rates": rates,
        "maturities": maturities,
        "v0": v0,
        "kappa": kappa,
        "theta": theta,
        "xi": xi,
        "rho": rho,
    }


def build_features(
    params: dict[str, np.ndarray],
    *,
    spot_max: float,
    strike_min: float,
    strike_max: float,
    rate_min: float,
    rate_max: float,
    maturity_max: float,
    v0_max: float,
    kappa_max: float,
    theta_max: float,
    xi_max: float,
    rho_min: float,
    rho_max: float,
) -> np.ndarray:
    spots_norm = params["spots"] / spot_max
    strikes_norm = params["strikes"] / strike_max
    rates_norm = (params["rates"] - rate_min) / (rate_max - rate_min)
    maturities_norm = params["maturities"] / maturity_max
    v0_norm = params["v0"] / v0_max
    kappa_norm = params["kappa"] / kappa_max
    theta_norm = params["theta"] / theta_max
    xi_norm = params["xi"] / xi_max
    rho_norm = (params["rho"] - rho_min) / (rho_max - rho_min)
    moneyness = params["spots"] / params["strikes"]
    moneyness_norm = np.clip(moneyness / (spot_max / strike_min), 0.0, 1.0)
    vol_ratio = np.sqrt(params["v0"]) / np.sqrt(np.maximum(params["theta"], 1e-8))
    vol_ratio_norm = np.clip(vol_ratio / np.sqrt(v0_max / 1e-4), 0.0, 1.0)
    feller_ratio = 2.0 * params["kappa"] * params["theta"] / np.maximum(params["xi"] ** 2, 1e-8)
    feller_ratio_norm = np.clip(feller_ratio / 5.0, 0.0, 1.0)
    intrinsic_norm = np.maximum(params["spots"] - params["strikes"], 0.0) / spot_max
    return np.concatenate(
        [
            spots_norm,
            strikes_norm,
            rates_norm,
            maturities_norm,
            v0_norm,
            kappa_norm,
            theta_norm,
            xi_norm,
            rho_norm,
            moneyness_norm,
            vol_ratio_norm,
            feller_ratio_norm,
            intrinsic_norm,
        ],
        axis=1,
    ).astype(np.float32)


def generate_dataset(
    *,
    n_samples: int,
    n_paths: int,
    n_steps: int,
    chunk_size: int,
    seed: int,
    spot_min: float = 50.0,
    spot_max: float = 150.0,
    strike_min: float = 60.0,
    strike_max: float = 140.0,
    rate_min: float = 0.0,
    rate_max: float = 0.10,
    maturity_min: float = 0.10,
    maturity_max: float = 3.0,
    v0_min: float = 0.01,
    v0_max: float = 0.25,
    kappa_min: float = 0.50,
    kappa_max: float = 5.0,
    theta_min: float = 0.01,
    theta_max: float = 0.25,
    xi_min: float = 0.10,
    xi_max: float = 1.0,
    rho_min: float = -0.90,
    rho_max: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, float | int | str]]:
    params = sample_heston_parameters(
        n_samples,
        spot_min=spot_min,
        spot_max=spot_max,
        strike_min=strike_min,
        strike_max=strike_max,
        rate_min=rate_min,
        rate_max=rate_max,
        maturity_min=maturity_min,
        maturity_max=maturity_max,
        v0_min=v0_min,
        v0_max=v0_max,
        kappa_min=kappa_min,
        kappa_max=kappa_max,
        theta_min=theta_min,
        theta_max=theta_max,
        xi_min=xi_min,
        xi_max=xi_max,
        rho_min=rho_min,
        rho_max=rho_max,
        seed=seed,
    )
    prices = np.zeros((n_samples, 1), dtype=np.float32)
    start = time.perf_counter()
    for index in range(n_samples):
        prices[index, 0] = heston_call_mc_np(
            S0=float(params["spots"][index, 0]),
            K=float(params["strikes"][index, 0]),
            r=float(params["rates"][index, 0]),
            T=float(params["maturities"][index, 0]),
            v0=float(params["v0"][index, 0]),
            kappa=float(params["kappa"][index, 0]),
            theta=float(params["theta"][index, 0]),
            xi=float(params["xi"][index, 0]),
            rho=float(params["rho"][index, 0]),
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed + index,
            chunk_size=chunk_size,
        )
    elapsed_seconds = time.perf_counter() - start
    x = build_features(
        params,
        spot_max=spot_max,
        strike_min=strike_min,
        strike_max=strike_max,
        rate_min=rate_min,
        rate_max=rate_max,
        maturity_max=maturity_max,
        v0_max=v0_max,
        kappa_max=kappa_max,
        theta_max=theta_max,
        xi_max=xi_max,
        rho_min=rho_min,
        rho_max=rho_max,
    )
    y = (prices / spot_max).astype(np.float32)
    metadata: dict[str, float | int | str] = {
        "instrument": "heston_call",
        "dataset_version": "heston_mc_dataset_v1",
        "n_samples": n_samples,
        "n_paths": n_paths,
        "n_steps": n_steps,
        "chunk_size": chunk_size,
        "seed": seed,
        "spot_min": spot_min,
        "spot_max": spot_max,
        "strike_min": strike_min,
        "strike_max": strike_max,
        "rate_min": rate_min,
        "rate_max": rate_max,
        "maturity_min": maturity_min,
        "maturity_max": maturity_max,
        "v0_min": v0_min,
        "v0_max": v0_max,
        "kappa_min": kappa_min,
        "kappa_max": kappa_max,
        "theta_min": theta_min,
        "theta_max": theta_max,
        "xi_min": xi_min,
        "xi_max": xi_max,
        "rho_min": rho_min,
        "rho_max": rho_max,
        "feature_order": "spot_norm,strike_norm,rate_norm,maturity_norm,v0_norm,kappa_norm,theta_norm,xi_norm,rho_norm,moneyness_norm,vol_ratio_norm,feller_ratio_norm,intrinsic_norm",
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
        strikes=params["strikes"],
        rates=params["rates"],
        maturities=params["maturities"],
        v0=params["v0"],
        kappa=params["kappa"],
        theta=params["theta"],
        xi=params["xi"],
        rho=params["rho"],
        target_prices=params["target_prices"],
    )
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="artifacts/phase6_heston_surrogate_dataset")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--n-paths", type=int, default=20000)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    x, y, params, metadata = generate_dataset(
        n_samples=args.n_samples,
        n_paths=args.n_paths,
        n_steps=args.n_steps,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )
    out_dir = Path(args.out_dir)
    save_dataset(out_dir, x, y, params, metadata)
    print(f"Generated Heston dataset: samples={args.n_samples} features={x.shape[1]} paths={args.n_paths} steps={args.n_steps}")
    print(f"Price mean={float(np.mean(params['target_prices'])):.6f} price std={float(np.std(params['target_prices'])):.6f}")
    print(f"Saved dataset to {out_dir / 'dataset.npz'}")
    print(f"Saved metadata to {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
