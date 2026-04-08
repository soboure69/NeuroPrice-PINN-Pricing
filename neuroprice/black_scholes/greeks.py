from __future__ import annotations

import math

from scipy.stats import norm

from neuroprice.black_scholes.pricing import BlackScholesParams, d1, d2


def call_delta(p: BlackScholesParams) -> float:
    return norm.cdf(d1(p))


def put_delta(p: BlackScholesParams) -> float:
    return call_delta(p) - 1.0


def gamma(p: BlackScholesParams) -> float:
    dd1 = d1(p)
    return norm.pdf(dd1) / (p.S * p.sigma * math.sqrt(p.T))


def vega(p: BlackScholesParams) -> float:
    dd1 = d1(p)
    return p.S * norm.pdf(dd1) * math.sqrt(p.T)


def call_theta(p: BlackScholesParams) -> float:
    dd1 = d1(p)
    dd2 = d2(p)
    term1 = -(p.S * norm.pdf(dd1) * p.sigma) / (2.0 * math.sqrt(p.T))
    term2 = -p.r * p.K * math.exp(-p.r * p.T) * norm.cdf(dd2)
    return term1 + term2


def put_theta(p: BlackScholesParams) -> float:
    dd1 = d1(p)
    dd2 = d2(p)
    term1 = -(p.S * norm.pdf(dd1) * p.sigma) / (2.0 * math.sqrt(p.T))
    term2 = p.r * p.K * math.exp(-p.r * p.T) * norm.cdf(-dd2)
    return term1 + term2


def call_rho(p: BlackScholesParams) -> float:
    dd2 = d2(p)
    return p.K * p.T * math.exp(-p.r * p.T) * norm.cdf(dd2)


def put_rho(p: BlackScholesParams) -> float:
    dd2 = d2(p)
    return -p.K * p.T * math.exp(-p.r * p.T) * norm.cdf(-dd2)
