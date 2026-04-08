from __future__ import annotations

import argparse

from neuroprice.black_scholes.greeks import (
    call_delta,
    call_rho,
    call_theta,
    gamma,
    put_delta,
    put_rho,
    put_theta,
    vega,
)
from neuroprice.black_scholes.pricing import BlackScholesParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--S", type=float, default=100.0)
    parser.add_argument("--K", type=float, default=105.0)
    parser.add_argument("--r", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--T", type=float, default=1.0)
    args = parser.parse_args()

    p = BlackScholesParams(S=args.S, K=args.K, r=args.r, sigma=args.sigma, T=args.T)

    print(f"Call delta: {call_delta(p):.6f}")
    print(f"Put delta : {put_delta(p):.6f}")
    print(f"Gamma    : {gamma(p):.6f}")
    print(f"Vega     : {vega(p):.6f}")
    print(f"Call theta: {call_theta(p):.6f}")
    print(f"Put theta : {put_theta(p):.6f}")
    print(f"Call rho  : {call_rho(p):.6f}")
    print(f"Put rho   : {put_rho(p):.6f}")


if __name__ == "__main__":
    main()
