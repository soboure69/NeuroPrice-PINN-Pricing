from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm


@dataclass(frozen=True)
class BlackScholesParams:
    S: float
    K: float
    r: float
    sigma: float
    T: float


def _validate_params(p: BlackScholesParams) -> None:
    if p.S <= 0:
        raise ValueError("S must be > 0")
    if p.K <= 0:
        raise ValueError("K must be > 0")
    if p.sigma <= 0:
        raise ValueError("sigma must be > 0")
    if p.T <= 0:
        raise ValueError("T must be > 0")


def d1(p: BlackScholesParams) -> float:
    _validate_params(p)
    return (math.log(p.S / p.K) + (p.r + 0.5 * p.sigma * p.sigma) * p.T) / (p.sigma * math.sqrt(p.T))


def d2(p: BlackScholesParams) -> float:
    return d1(p) - p.sigma * math.sqrt(p.T)


def call_price(p: BlackScholesParams) -> float:
    dd1 = d1(p)
    dd2 = d2(p)
    return p.S * norm.cdf(dd1) - p.K * math.exp(-p.r * p.T) * norm.cdf(dd2)


def put_price(p: BlackScholesParams) -> float:
    dd1 = d1(p)
    dd2 = d2(p)
    return p.K * math.exp(-p.r * p.T) * norm.cdf(-dd2) - p.S * norm.cdf(-dd1)


def put_call_parity_gap(p: BlackScholesParams) -> float:
    """Returns: C - P - (S - K e^{-rT}). Should be ~0."""
    c = call_price(p)
    put = put_price(p)
    return c - put - (p.S - p.K * math.exp(-p.r * p.T))
