from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from neuroprice.pinn.parametric_bs import ParametricBlackScholesDomain, ParametricBlackScholesPINN
from neuroprice.validation.black_scholes_ref import black_scholes_call_price_np, relative_l2_error


def scale_unit(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return lower + (upper - lower) * values


def summarize_segment(errors: np.ndarray) -> dict[str, float | int]:
    if errors.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "pct_under_0_5pct": float("nan"),
            "pct_under_1_0pct": float("nan"),
        }
    return {
        "count": int(errors.size),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p95": float(np.quantile(errors, 0.95)),
        "pct_under_0_5pct": float(np.mean(errors < 0.005) * 100.0),
        "pct_under_1_0pct": float(np.mean(errors < 0.010) * 100.0),
    }


def summarize_bins(values: np.ndarray, errors: np.ndarray, edges: list[float]) -> dict[str, dict[str, float | int]]:
    flat_values = values.reshape(-1)
    flat_errors = errors.reshape(-1)
    summary: dict[str, dict[str, float | int]] = {}
    for lower, upper in zip(edges[:-1], edges[1:]):
        if upper == edges[-1]:
            mask = (flat_values >= lower) & (flat_values <= upper)
        else:
            mask = (flat_values >= lower) & (flat_values < upper)
        summary[f"[{lower:.4g}, {upper:.4g}]"] = summarize_segment(flat_errors[mask])
    return summary


def monte_carlo_call_price_np(
    S: np.ndarray,
    tau: np.ndarray,
    K: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    n_paths: int,
    rng: np.random.Generator,
    chunk_size: int,
) -> np.ndarray:
    estimates = np.zeros_like(S, dtype=np.float64)
    paths_done = 0
    while paths_done < n_paths:
        current_paths = min(chunk_size, n_paths - paths_done)
        z = rng.standard_normal(size=(S.shape[0], current_paths))
        drift = (r - 0.5 * sigma**2) * tau
        diffusion = sigma * np.sqrt(np.maximum(tau, 0.0)) * z
        terminal = S * np.exp(drift + diffusion)
        payoff = np.maximum(terminal - K, 0.0)
        estimates += np.exp(-r * tau) * np.mean(payoff, axis=1, keepdims=True) * (current_paths / n_paths)
        paths_done += current_paths
    return estimates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="artifacts/phase2_bs_pinn_parametric/bs_pinn_parametric.pt")
    parser.add_argument("--out", type=str, default="artifacts/phase2_bs_pinn_parametric/benchmark.json")
    parser.add_argument("--n-points", type=int, default=10000)
    parser.add_argument("--relative-floor", type=float, default=1.0)
    parser.add_argument("--min-reference-price", type=float, default=1.0)
    parser.add_argument("--mc-paths", type=int, default=10000)
    parser.add_argument("--mc-chunk-size", type=int, default=1000)
    parser.add_argument("--skip-monte-carlo", action="store_true")
    parser.add_argument("--seed", type=int, default=321)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    if checkpoint.get("coordinate_transform") != "parametric_log_moneyness" or checkpoint.get("output_scale") != "normalized_by_K":
        raise ValueError("Checkpoint is not compatible with parametric validation. Retrain with scripts.train_pinn_bs_parametric.")

    domain = ParametricBlackScholesDomain(**checkpoint["domain"])
    model = ParametricBlackScholesPINN(
        hidden_dim=int(checkpoint["hidden_dim"]),
        hidden_layers=int(checkpoint["hidden_layers"]),
        use_strike_input=bool(checkpoint.get("use_strike_input", True)),
        output_transform=str(checkpoint.get("output_transform", "direct")),
        fourier_features=int(checkpoint.get("fourier_features", 0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    S = rng.uniform(domain.S_min, domain.S_max, size=(args.n_points, 1))
    K = rng.uniform(domain.K_min, domain.K_max, size=(args.n_points, 1))
    sigma = rng.uniform(domain.sigma_min, domain.sigma_max, size=(args.n_points, 1))
    r = rng.uniform(domain.r_min, domain.r_max, size=(args.n_points, 1))
    T = rng.uniform(domain.T_min, domain.T_max, size=(args.n_points, 1))
    tau = rng.uniform(0.0, 1.0, size=(args.n_points, 1)) * T
    x = np.log(S / K)

    x_norm = np.clip((x - domain.x_min) / (domain.x_max - domain.x_min), 0.0, 1.0)
    tau_norm = tau / T
    sigma_norm = (sigma - domain.sigma_min) / (domain.sigma_max - domain.sigma_min)
    r_norm = (r - domain.r_min) / (domain.r_max - domain.r_min)
    T_norm = (T - domain.T_min) / (domain.T_max - domain.T_min)
    K_norm = (K - domain.K_min) / (domain.K_max - domain.K_min)

    tensors = [
        torch.tensor(array, dtype=torch.float32, device=device)
        for array in [x_norm, tau_norm, sigma_norm, r_norm, T_norm, K_norm]
    ]

    start = time.perf_counter()
    with torch.no_grad():
        pred_norm = model(*tensors).detach().cpu().numpy()
    pinn_seconds = time.perf_counter() - start
    pred = K * pred_norm

    start = time.perf_counter()
    ref = black_scholes_call_price_np(S, tau, K, r, sigma)
    analytic_seconds = time.perf_counter() - start

    if args.skip_monte_carlo:
        mc = None
        mc_seconds = float("nan")
        mc_rel_l2 = float("nan")
        mc_mean_point_relative_error = float("nan")
        mc_p95_point_relative_error = float("nan")
        mc_points_per_second = float("nan")
        pinn_vs_mc_speedup = float("nan")
    else:
        start = time.perf_counter()
        mc = monte_carlo_call_price_np(S, tau, K, r, sigma, args.mc_paths, rng, args.mc_chunk_size)
        mc_seconds = time.perf_counter() - start
        mc_abs_err = np.abs(mc - ref)
        mc_rel_point_err = mc_abs_err / np.maximum(np.abs(ref), args.relative_floor)
        mc_rel_l2 = relative_l2_error(mc, ref)
        mc_mean_point_relative_error = float(np.mean(mc_rel_point_err))
        mc_p95_point_relative_error = float(np.quantile(mc_rel_point_err, 0.95))
        mc_points_per_second = float(args.n_points / max(mc_seconds, 1e-12))
        pinn_vs_mc_speedup = float((args.n_points / max(pinn_seconds, 1e-12)) / max(mc_points_per_second, 1e-12))

    abs_err = np.abs(pred - ref)
    rel_point_err = abs_err / np.maximum(np.abs(ref), args.relative_floor)
    tradable_mask = ref.reshape(-1) >= args.min_reference_price
    tradable_rel_point_err = rel_point_err.reshape(-1)[tradable_mask]
    rel_l2 = relative_l2_error(pred, ref)
    pct_under_05 = float(np.mean(rel_point_err < 0.005) * 100.0)
    pct_under_10 = float(np.mean(rel_point_err < 0.010) * 100.0)
    tradable_pct_under_05 = float(np.mean(tradable_rel_point_err < 0.005) * 100.0) if tradable_rel_point_err.size else float("nan")
    tradable_pct_under_10 = float(np.mean(tradable_rel_point_err < 0.010) * 100.0) if tradable_rel_point_err.size else float("nan")
    segmented_metrics = {
        "moneyness_x": summarize_bins(x, rel_point_err, [-2.5, -1.0, -0.25, 0.25, 1.0, 2.5]),
        "tau_over_T": summarize_bins(tau_norm, rel_point_err, [0.0, 0.05, 0.25, 0.50, 0.75, 1.0]),
        "sigma": summarize_bins(sigma, rel_point_err, [domain.sigma_min, 0.15, 0.30, 0.50, domain.sigma_max]),
        "r": summarize_bins(r, rel_point_err, [domain.r_min, 0.025, 0.05, 0.10, domain.r_max]),
        "K": summarize_bins(K, rel_point_err, [domain.K_min, 50.0, 100.0, 150.0, domain.K_max]),
    }

    result = {
        "n_points": args.n_points,
        "relative_floor": args.relative_floor,
        "min_reference_price": args.min_reference_price,
        "mc_paths": args.mc_paths,
        "mc_chunk_size": args.mc_chunk_size,
        "n_tradable_points": int(np.sum(tradable_mask)),
        "relative_l2": rel_l2,
        "mean_point_relative_error": float(np.mean(rel_point_err)),
        "median_point_relative_error": float(np.median(rel_point_err)),
        "p95_point_relative_error": float(np.quantile(rel_point_err, 0.95)),
        "p99_point_relative_error": float(np.quantile(rel_point_err, 0.99)),
        "tradable_mean_point_relative_error": float(np.mean(tradable_rel_point_err)) if tradable_rel_point_err.size else float("nan"),
        "tradable_median_point_relative_error": float(np.median(tradable_rel_point_err)) if tradable_rel_point_err.size else float("nan"),
        "tradable_p95_point_relative_error": float(np.quantile(tradable_rel_point_err, 0.95)) if tradable_rel_point_err.size else float("nan"),
        "tradable_p99_point_relative_error": float(np.quantile(tradable_rel_point_err, 0.99)) if tradable_rel_point_err.size else float("nan"),
        "max_absolute_error": float(np.max(abs_err)),
        "pct_points_under_0_5pct_error": pct_under_05,
        "pct_points_under_1_0pct_error": pct_under_10,
        "tradable_pct_points_under_0_5pct_error": tradable_pct_under_05,
        "tradable_pct_points_under_1_0pct_error": tradable_pct_under_10,
        "pinn_seconds": pinn_seconds,
        "analytic_seconds": analytic_seconds,
        "monte_carlo_seconds": mc_seconds,
        "pinn_points_per_second": float(args.n_points / max(pinn_seconds, 1e-12)),
        "analytic_points_per_second": float(args.n_points / max(analytic_seconds, 1e-12)),
        "monte_carlo_points_per_second": mc_points_per_second,
        "pinn_vs_monte_carlo_speedup": pinn_vs_mc_speedup,
        "monte_carlo_relative_l2": mc_rel_l2,
        "monte_carlo_mean_point_relative_error": mc_mean_point_relative_error,
        "monte_carlo_p95_point_relative_error": mc_p95_point_relative_error,
        "segmented_metrics": segmented_metrics,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    print("segmented_metrics saved to benchmark JSON")
    print(f"Saved benchmark to {out_path}")


if __name__ == "__main__":
    main()
