from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from neuroprice.pinn.barrier import BarrierOptionDomain, DownAndOutBarrierPINN
from neuroprice.validation.barrier_ref import down_and_out_call_price_np
from neuroprice.validation.black_scholes_ref import relative_l2_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="artifacts/phase3_down_out_barrier/down_out_barrier_pinn.pt")
    parser.add_argument("--out", type=str, default="artifacts/phase3_down_out_barrier/benchmark.json")
    parser.add_argument("--n-points", type=int, default=10000)
    parser.add_argument("--relative-floor", type=float, default=1.0)
    parser.add_argument("--min-reference-price", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=321)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    if checkpoint.get("option_type") != "down_and_out_call" or checkpoint.get("output_scale") != "normalized_by_S_max":
        raise ValueError("Checkpoint is not compatible with down-and-out barrier validation.")

    domain = BarrierOptionDomain(**checkpoint["domain"])
    model = DownAndOutBarrierPINN(
        hidden_dim=int(checkpoint["hidden_dim"]),
        hidden_layers=int(checkpoint["hidden_layers"]),
        output_transform=str(checkpoint.get("output_transform", "direct")),
        K=domain.K,
        B=domain.B,
        r=domain.r,
        T=domain.T,
        S_max=domain.S_max,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    S = rng.uniform(domain.B, domain.S_max, size=(args.n_points, 1))
    tau = rng.uniform(0.0, domain.T, size=(args.n_points, 1))
    S_norm = torch.tensor(S / domain.S_max, dtype=torch.float32, device=device)
    tau_norm = torch.tensor(tau / domain.T, dtype=torch.float32, device=device)

    start = time.perf_counter()
    with torch.no_grad():
        pred_norm = model(S_norm, tau_norm).detach().cpu().numpy()
    pinn_seconds = time.perf_counter() - start
    pred = domain.S_max * pred_norm

    start = time.perf_counter()
    ref = down_and_out_call_price_np(S, tau, domain.K, domain.B, domain.r, domain.sigma)
    analytic_seconds = time.perf_counter() - start

    abs_err = np.abs(pred - ref)
    rel_point_err = abs_err / np.maximum(np.abs(ref), args.relative_floor)
    tradable_mask = ref.reshape(-1) >= args.min_reference_price
    tradable_rel_point_err = rel_point_err.reshape(-1)[tradable_mask]
    near_barrier_mask = S.reshape(-1) <= domain.B + 0.1 * (domain.S_max - domain.B)
    near_barrier_tradable_mask = near_barrier_mask & tradable_mask
    near_barrier_abs_err = abs_err.reshape(-1)[near_barrier_mask]
    near_barrier_rel_err = rel_point_err.reshape(-1)[near_barrier_mask]
    near_barrier_tradable_rel_err = rel_point_err.reshape(-1)[near_barrier_tradable_mask]
    near_strike_mask = np.abs(S.reshape(-1) - domain.K) / domain.K <= 0.10
    result = {
        "n_points": args.n_points,
        "relative_floor": args.relative_floor,
        "min_reference_price": args.min_reference_price,
        "n_tradable_points": int(np.sum(tradable_mask)),
        "relative_l2": relative_l2_error(pred, ref),
        "mean_point_relative_error": float(np.mean(rel_point_err)),
        "median_point_relative_error": float(np.median(rel_point_err)),
        "p95_point_relative_error": float(np.quantile(rel_point_err, 0.95)),
        "p99_point_relative_error": float(np.quantile(rel_point_err, 0.99)),
        "tradable_mean_point_relative_error": float(np.mean(tradable_rel_point_err)) if tradable_rel_point_err.size else float("nan"),
        "tradable_median_point_relative_error": float(np.median(tradable_rel_point_err)) if tradable_rel_point_err.size else float("nan"),
        "tradable_p95_point_relative_error": float(np.quantile(tradable_rel_point_err, 0.95)) if tradable_rel_point_err.size else float("nan"),
        "tradable_pct_points_under_1pct_error": float(np.mean(tradable_rel_point_err < 0.01) * 100.0) if tradable_rel_point_err.size else float("nan"),
        "tradable_pct_points_under_5pct_error": float(np.mean(tradable_rel_point_err < 0.05) * 100.0) if tradable_rel_point_err.size else float("nan"),
        "max_absolute_error": float(np.max(abs_err)),
        "near_barrier_count": int(np.sum(near_barrier_mask)),
        "near_barrier_tradable_count": int(np.sum(near_barrier_tradable_mask)),
        "near_barrier_mean_absolute_error": float(np.mean(near_barrier_abs_err)) if near_barrier_abs_err.size else float("nan"),
        "near_barrier_median_absolute_error": float(np.median(near_barrier_abs_err)) if near_barrier_abs_err.size else float("nan"),
        "near_barrier_p95_absolute_error": float(np.quantile(near_barrier_abs_err, 0.95)) if near_barrier_abs_err.size else float("nan"),
        "near_barrier_mean_relative_error": float(np.mean(near_barrier_rel_err)) if near_barrier_rel_err.size else float("nan"),
        "near_barrier_median_relative_error": float(np.median(near_barrier_rel_err)) if near_barrier_rel_err.size else float("nan"),
        "near_barrier_tradable_p95_relative_error": float(np.quantile(near_barrier_tradable_rel_err, 0.95)) if near_barrier_tradable_rel_err.size else float("nan"),
        "near_strike_mean_relative_error": float(np.mean(rel_point_err.reshape(-1)[near_strike_mask])) if np.any(near_strike_mask) else float("nan"),
        "pinn_seconds": pinn_seconds,
        "analytic_seconds": analytic_seconds,
        "pinn_points_per_second": float(args.n_points / max(pinn_seconds, 1e-12)),
        "analytic_points_per_second": float(args.n_points / max(analytic_seconds, 1e-12)),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    print(f"Saved benchmark to {out_path}")


if __name__ == "__main__":
    main()
