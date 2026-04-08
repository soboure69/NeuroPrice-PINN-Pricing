from __future__ import annotations

from neuroprice.black_scholes.pricing import BlackScholesParams, call_price, put_call_parity_gap, put_price


def test_put_call_parity_gap_is_small() -> None:
    p = BlackScholesParams(S=100.0, K=105.0, r=0.05, sigma=0.20, T=1.0)
    gap = put_call_parity_gap(p)
    assert abs(gap) < 1e-10


def test_call_put_prices_are_positive() -> None:
    p = BlackScholesParams(S=100.0, K=105.0, r=0.05, sigma=0.20, T=1.0)
    assert call_price(p) > 0.0
    assert put_price(p) > 0.0
