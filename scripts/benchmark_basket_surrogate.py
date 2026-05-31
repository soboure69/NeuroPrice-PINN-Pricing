from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neuroprice.pinn.basket_surrogate import BasketCallSurrogate, BasketSurrogateDomain
from neuroprice.validation.basket_ref import basket_call_mc_np


def load_surrogate(checkpoint_path: Path, device: torch.device) -> tuple[BasketCallSurrogate, BasketSurrogateDomain, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("option_type") != "basket_call_surrogate":
        raise ValueError("Checkpoint is not compatible with basket_call_surrogate")
    domain = BasketSurrogateDomain(**checkpoint["domain"])
    model = BasketCallSurrogate(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        hidden_layers=int(checkpoint["hidden_layers"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, domain, checkpoint


def select_dataset_points(dataset_path: Path, n_points: int, seed: int) -> dict[str, np.ndarray]:
    data = np.load(dataset_path)
    total = data["x"].shape[0]
    if n_points > total:
        raise ValueError(f"n_points={n_points} exceeds dataset size={total}")
    rng = np.random.default_rng(seed)
    indices = rng.choice(total, size=n_points, replace=False)
    return {key: data[key][indices] for key in data.files}


def monte_carlo_reference(rows: dict[str, np.ndarray], *, n_paths: int, chunk_size: int, seed: int) -> tuple[np.ndarray, float]:
    n_points = rows["x"].shape[0]
    prices = np.zeros((n_points, 1), dtype=np.float32)
    start = time.perf_counter()
    for index in range(n_points):
        prices[index, 0] = basket_call_mc_np(
            spots=rows["spots"][index],
            weights=rows["weights"][index],
            sigmas=rows["sigmas"][index],
            K=float(rows["strikes"][index, 0]),
            r=float(rows["rates"][index, 0]),
            tau=float(rows["maturities"][index, 0]),
            correlation=float(rows["correlations"][index, 0]),
            n_paths=n_paths,
            seed=seed + index,
            chunk_size=chunk_size,
        )
    return prices, time.perf_counter() - start


def surrogate_predict(model: BasketCallSurrogate, x: np.ndarray, *, spot_max: float, device: torch.device) -> tuple[np.ndarray, float]:
    tensor = torch.tensor(x, dtype=torch.float32, device=device)
    start = time.perf_counter()
    with torch.no_grad():
        pred_norm = model(tensor).detach().cpu().numpy()
    elapsed = time.perf_counter() - start
    return (spot_max * pred_norm).astype(np.float32), elapsed


def summarize_errors(pred: np.ndarray, ref: np.ndarray, relative_floor: float) -> dict[str, float]:
    abs_err = np.abs(pred - ref)
    rel_err = abs_err / np.maximum(np.abs(ref), relative_floor)
    return {
        "mae": float(np.mean(abs_err)),
        "rmse": float(np.sqrt(np.mean(abs_err**2))),
        "median_absolute_error": float(np.median(abs_err)),
        "p95_absolute_error": float(np.quantile(abs_err, 0.95)),
        "mean_relative_error": float(np.mean(rel_err)),
        "median_relative_error": float(np.median(rel_err)),
        "p95_relative_error": float(np.quantile(rel_err, 0.95)),
        "pct_under_5pct": float(np.mean(rel_err < 0.05) * 100.0),
        "pct_under_10pct": float(np.mean(rel_err < 0.10) * 100.0),
    }


def run_benchmark(
    *,
    checkpoint_path: Path,
    dataset_path: Path,
    n_points: int,
    mc_paths: int,
    mc_chunk_size: int,
    relative_floor: float,
    seed: int,
) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, domain, checkpoint = load_surrogate(checkpoint_path, device)
    rows = select_dataset_points(dataset_path, n_points, seed)
    pred, surrogate_seconds = surrogate_predict(model, rows["x"], spot_max=domain.spot_max, device=device)
    ref, mc_seconds = monte_carlo_reference(rows, n_paths=mc_paths, chunk_size=mc_chunk_size, seed=seed)
    metrics = summarize_errors(pred, ref, relative_floor)
    return {
        "instrument": "basket_call",
        "benchmark_version": "basket_surrogate_vs_mc_v1",
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "device": str(device),
        "n_points": n_points,
        "mc_paths": mc_paths,
        "mc_chunk_size": mc_chunk_size,
        "relative_floor": relative_floor,
        "surrogate_seconds": surrogate_seconds,
        "monte_carlo_seconds": mc_seconds,
        "surrogate_points_per_second": float(n_points / max(surrogate_seconds, 1e-12)),
        "monte_carlo_points_per_second": float(n_points / max(mc_seconds, 1e-12)),
        "speedup_vs_monte_carlo": float(mc_seconds / max(surrogate_seconds, 1e-12)),
        "price_reference_mean": float(np.mean(ref)),
        "price_surrogate_mean": float(np.mean(pred)),
        "metrics": metrics,
        "training": checkpoint.get("training", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="artifacts/phase6_basket_surrogate/basket_surrogate.pt")
    parser.add_argument("--dataset", type=str, default="artifacts/phase6_basket_surrogate_dataset/dataset.npz")
    parser.add_argument("--out", type=str, default="artifacts/phase6_basket_surrogate/benchmark.json")
    parser.add_argument("--n-points", type=int, default=500)
    parser.add_argument("--mc-paths", type=int, default=20000)
    parser.add_argument("--mc-chunk-size", type=int, default=2000)
    parser.add_argument("--relative-floor", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()

    result = run_benchmark(
        checkpoint_path=Path(args.checkpoint),
        dataset_path=Path(args.dataset),
        n_points=args.n_points,
        mc_paths=args.mc_paths,
        mc_chunk_size=args.mc_chunk_size,
        relative_floor=args.relative_floor,
        seed=args.seed,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"surrogate_seconds: {result['surrogate_seconds']:.6f}")
    print(f"monte_carlo_seconds: {result['monte_carlo_seconds']:.6f}")
    print(f"speedup_vs_monte_carlo: {result['speedup_vs_monte_carlo']:.2f}")
    for key, value in result["metrics"].items():
        print(f"{key}: {value:.6f}")
    print(f"Saved benchmark to {out_path}")


if __name__ == "__main__":
    main()
