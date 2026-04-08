from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from neuroprice.stochastic.gbm import simulate_gbm_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s0", type=float, default=100.0)
    parser.add_argument("--mu", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--n-steps", type=int, default=252)
    parser.add_argument("--n-paths", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=str, default="gbm_paths.png")
    args = parser.parse_args()

    t, s = simulate_gbm_paths(
        s0=args.s0,
        mu=args.mu,
        sigma=args.sigma,
        T=args.T,
        n_steps=args.n_steps,
        n_paths=args.n_paths,
        seed=args.seed,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(s.shape[0]):
        ax.plot(t, s[i], alpha=0.35, linewidth=0.8)

    ax.set_xlabel("t")
    ax.set_ylabel("S_t")
    ax.set_title("Geometric Brownian Motion paths")
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)


if __name__ == "__main__":
    main()
