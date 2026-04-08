from __future__ import annotations

import argparse

from neuroprice.black_scholes.pricing import BlackScholesParams, call_price, put_call_parity_gap, put_price


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--S", type=float, default=100.0)
    parser.add_argument("--K", type=float, default=105.0)
    parser.add_argument("--r", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--T", type=float, default=1.0)
    args = parser.parse_args()

    p = BlackScholesParams(S=args.S, K=args.K, r=args.r, sigma=args.sigma, T=args.T)

    c = call_price(p)
    put = put_price(p)
    gap = put_call_parity_gap(p)

    print(f"Call  : {c:.6f}")
    print(f"Put   : {put:.6f}")
    print(f"Parity gap (should be ~0): {gap:.3e}")


if __name__ == "__main__":
    main()
