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


def build_basket_inputs(n_assets: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spots = np.linspace(90.0, 110.0, n_assets, dtype=np.float64)
    sigmas = np.linspace(0.18, 0.30, n_assets, dtype=np.float64)
    weights = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)
    return spots, sigmas, weights


def benchmark_dimension(
    n_assets: int,
    *,
    n_paths: int,
    repeats: int,
    strike: float,
    rate: float,
    maturity: float,
    correlation: float,
    seed: int,
    chunk_size: int,
) -> dict[str, float | int]:
    spots, sigmas, weights = build_basket_inputs(n_assets)
    prices: list[float] = []
    seconds: list[float] = []
    for repeat in range(repeats):
        start = time.perf_counter()
        price = basket_call_mc_np(
            spots=spots,
            weights=weights,
            sigmas=sigmas,
            K=strike,
            r=rate,
            tau=maturity,
            correlation=correlation,
            n_paths=n_paths,
            seed=seed + repeat,
            chunk_size=chunk_size,
        )
        elapsed = time.perf_counter() - start
        prices.append(price)
        seconds.append(elapsed)
    mean_seconds = float(np.mean(seconds))
    return {
        "n_assets": n_assets,
        "n_paths": n_paths,
        "repeats": repeats,
        "price_mean": float(np.mean(prices)),
        "price_std": float(np.std(prices, ddof=1)) if repeats > 1 else 0.0,
        "seconds_mean": mean_seconds,
        "seconds_std": float(np.std(seconds, ddof=1)) if repeats > 1 else 0.0,
        "paths_per_second": float(n_paths / max(mean_seconds, 1e-12)),
        "spots_min": float(np.min(spots)),
        "spots_max": float(np.max(spots)),
        "sigmas_min": float(np.min(sigmas)),
        "sigmas_max": float(np.max(sigmas)),
    }


def run_benchmark(
    dimensions: list[int],
    *,
    n_paths: int,
    repeats: int,
    strike: float,
    rate: float,
    maturity: float,
    correlation: float,
    seed: int,
    chunk_size: int,
) -> dict[str, object]:
    results = [
        benchmark_dimension(
            n_assets,
            n_paths=n_paths,
            repeats=repeats,
            strike=strike,
            rate=rate,
            maturity=maturity,
            correlation=correlation,
            seed=seed + 1000 * index,
            chunk_size=chunk_size,
        )
        for index, n_assets in enumerate(dimensions)
    ]
    baseline_seconds = float(results[0]["seconds_mean"])
    baseline_throughput = float(results[0]["paths_per_second"])
    for result in results:
        result["time_ratio_vs_first_dimension"] = float(float(result["seconds_mean"]) / max(baseline_seconds, 1e-12))
        result["throughput_ratio_vs_first_dimension"] = float(float(result["paths_per_second"]) / max(baseline_throughput, 1e-12))
    return {
        "instrument": "basket_call",
        "model_version": "basket_monte_carlo_v1",
        "dimensions": dimensions,
        "n_paths": n_paths,
        "repeats": repeats,
        "strike": strike,
        "rate": rate,
        "maturity": maturity,
        "correlation": correlation,
        "seed": seed,
        "chunk_size": chunk_size,
        "results": results,
    }


def parse_dimensions(value: str) -> list[int]:
    dimensions = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not dimensions:
        raise argparse.ArgumentTypeError("at least one dimension is required")
    if any(dimension < 2 or dimension > 10 for dimension in dimensions):
        raise argparse.ArgumentTypeError("dimensions must be between 2 and 10")
    return dimensions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=parse_dimensions, default="2,3,5,10")
    parser.add_argument("--out", type=str, default="artifacts/phase6_basket_dimension/benchmark.json")
    parser.add_argument("--n-paths", type=int, default=50000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--strike", type=float, default=100.0)
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--maturity", type=float, default=1.0)
    parser.add_argument("--correlation", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--chunk-size", type=int, default=5000)
    args = parser.parse_args()

    result = run_benchmark(
        args.dimensions,
        n_paths=args.n_paths,
        repeats=args.repeats,
        strike=args.strike,
        rate=args.rate,
        maturity=args.maturity,
        correlation=args.correlation,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for row in result["results"]:
        print(
            "N={n_assets} price={price_mean:.6f} seconds={seconds_mean:.6f} paths_per_second={paths_per_second:.2f} time_ratio={time_ratio_vs_first_dimension:.3f}".format(
                **row
            )
        )
    print(f"Saved benchmark to {out_path}")


if __name__ == "__main__":
    main()
