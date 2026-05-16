from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from neuroprice.pinn.asian_surrogate import AsianArithmeticSurrogate, AsianSurrogateDomain
from neuroprice.validation.asian_ref import asian_arithmetic_call_mc_np
from neuroprice.validation.black_scholes_ref import relative_l2_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="artifacts/phase3_asian_surrogate/asian_surrogate.pt")
    parser.add_argument("--out", type=str, default="artifacts/phase3_asian_surrogate/benchmark.json")
    parser.add_argument("--n-points", type=int, default=1000)
    parser.add_argument("--mc-paths", type=int, default=20000)
    parser.add_argument("--mc-steps", type=int, default=64)
    parser.add_argument("--mc-chunk-size", type=int, default=2000)
    parser.add_argument("--relative-floor", type=float, default=1.0)
    parser.add_argument("--min-reference-price", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=321)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    if checkpoint.get("option_type") != "asian_arithmetic_call_surrogate" or checkpoint.get("output_scale") != "normalized_by_S_max":
        raise ValueError("Checkpoint is not compatible with Asian surrogate validation.")

    domain = AsianSurrogateDomain(**checkpoint["domain"])
    model = AsianArithmeticSurrogate(hidden_dim=int(checkpoint["hidden_dim"]), hidden_layers=int(checkpoint["hidden_layers"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    S = rng.uniform(1e-6, domain.S_max, size=(args.n_points, 1))
    tau = rng.uniform(0.0, domain.T, size=(args.n_points, 1))
    S_norm = torch.tensor(S / domain.S_max, dtype=torch.float32, device=device)
    tau_norm = torch.tensor(tau / domain.T, dtype=torch.float32, device=device)

    start = time.perf_counter()
    with torch.no_grad():
        pred_norm = model(S_norm, tau_norm).detach().cpu().numpy()
    pinn_seconds = time.perf_counter() - start
    pred = domain.S_max * pred_norm

    start = time.perf_counter()
    ref = asian_arithmetic_call_mc_np(S, S, tau, domain.K, domain.r, domain.sigma, args.mc_paths, args.mc_steps, args.seed, args.mc_chunk_size)
    mc_seconds = time.perf_counter() - start

    abs_err = np.abs(pred - ref)
    rel_point_err = abs_err / np.maximum(np.abs(ref), args.relative_floor)
    tradable_mask = ref.reshape(-1) >= args.min_reference_price
    tradable_rel_point_err = rel_point_err.reshape(-1)[tradable_mask]
    result = {
        "n_points": args.n_points,
        "mc_paths": args.mc_paths,
        "mc_steps": args.mc_steps,
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
        "tradable_pct_points_under_5pct_error": float(np.mean(tradable_rel_point_err < 0.05) * 100.0) if tradable_rel_point_err.size else float("nan"),
        "tradable_pct_points_under_10pct_error": float(np.mean(tradable_rel_point_err < 0.10) * 100.0) if tradable_rel_point_err.size else float("nan"),
        "max_absolute_error": float(np.max(abs_err)),
        "pinn_seconds": pinn_seconds,
        "monte_carlo_seconds": mc_seconds,
        "pinn_points_per_second": float(args.n_points / max(pinn_seconds, 1e-12)),
        "monte_carlo_points_per_second": float(args.n_points / max(mc_seconds, 1e-12)),
        "pinn_vs_monte_carlo_speedup": float((args.n_points / max(pinn_seconds, 1e-12)) / max(args.n_points / max(mc_seconds, 1e-12), 1e-12)),
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
